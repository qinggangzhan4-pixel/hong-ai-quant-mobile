from __future__ import annotations
import numpy as np
import pandas as pd
from .features import add_features
from .scoring import score_row
from .relay import relay_watch
from .risk import plan_position, portfolio_risk
from .ml_model import load_model, predict_probability


def _is_excluded(symbol: str, config: dict) -> bool:
    u = config["universe"]
    s = str(symbol).zfill(6)
    if u.get("exclude_star_market", True) and s.startswith("68"):
        return True
    if u.get("exclude_chinext", True) and s.startswith("30"):
        return True
    if u.get("exclude_bse", True) and (s.startswith("4") or s.startswith("8")):
        return True
    return False


def score_universe(market_df: pd.DataFrame, config: dict, holdings: pd.DataFrame | None = None, model_path: str | None = None) -> tuple[pd.DataFrame, dict]:
    feat = add_features(market_df)
    latest_date = feat["date"].max()
    latest = feat.loc[feat["date"] == latest_date].copy()

    # pandas 2.x / object dtype 下，直接对 object 布尔列使用 ~ 可能得到 -1/-2，
    # 随后被 DataFrame 当成“列名”索引，触发 None of [Index([-1,...])] are in the [columns]。
    excluded_mask = latest["symbol"].astype(str).map(lambda x: bool(_is_excluded(x, config))).fillna(False).astype(bool)
    latest = latest.loc[excluded_mask.eq(False)].copy()

    if config["universe"].get("exclude_st", True):
        st_mask = latest["is_st"].fillna(False).map(bool).astype(bool)
        latest = latest.loc[st_mask.eq(False)].copy()
    price_mask = (
        (pd.to_numeric(latest["close"], errors="coerce") >= config["universe"]["min_price"])
        & (pd.to_numeric(latest["close"], errors="coerce") <= config["universe"]["max_price"])
    ).fillna(False).astype(bool)
    latest = latest.loc[price_mask].copy()

    if latest.empty:
        raise ValueError("最新交易日过滤后没有可评分股票，请检查数据日期、市场范围或排除条件。")

    parts = latest.apply(score_row, axis=1, result_type="expand")
    latest = pd.concat([latest.reset_index(drop=True), parts.reset_index(drop=True)], axis=1)

    model = load_model(model_path) if model_path else None
    prob = predict_probability(model, latest) if model is not None else np.full(len(latest), np.nan)
    latest["ml_probability"] = prob
    rw = config["strategy"]["rule_weight"]
    mw = config["strategy"]["model_weight"]
    latest["combined_score"] = latest["rule_score"]
    use_ml = latest["ml_probability"].notna()
    latest.loc[use_ml, "combined_score"] = rw * latest.loc[use_ml, "rule_score"] + mw * (latest.loc[use_ml, "ml_probability"] * 100)

    relay_results = latest.apply(lambda r: relay_watch(r, config), axis=1)
    latest["relay_score"] = [r["relay_score"] for r in relay_results]
    latest["relay_status"] = [r["relay_status"] for r in relay_results]
    latest["relay_reasons"] = ["；".join(r["relay_reasons"]) for r in relay_results]

    plans = []
    for _, row in latest.iterrows():
        plan = plan_position(row, config, holdings)
        plans.append(plan.to_dict())
    plan_df = pd.DataFrame(plans)
    latest = pd.concat([latest.reset_index(drop=True), plan_df.reset_index(drop=True)], axis=1)

    # 生成动作标签
    buy_th = config["strategy"]["score_buy_threshold"]
    watch_th = config["strategy"]["score_watch_threshold"]
    latest["signal"] = np.where(latest["combined_score"] >= buy_th, "BUY_CANDIDATE",
                        np.where(latest["combined_score"] >= watch_th, "WATCH", "AVOID"))

    # 优先普通趋势分，首板则额外要求 relay 分数
    relay_flag = latest["is_limit_up"].fillna(False).map(bool).astype(bool)
    is_relay = relay_flag | (pd.to_numeric(latest["ret_1"], errors="coerce").fillna(-999) >= 0.095)
    latest.loc[is_relay & (latest["relay_score"] < 50), "signal"] = "AVOID_RELAY"
    latest.loc[is_relay & (latest["relay_score"] >= 70), "signal"] = "RELAY_CANDIDATE"

    risk = portfolio_risk(holdings, float(config["account"]["equity"])) if holdings is not None else {}
    cols = [
        "date", "symbol", "name", "industry", "close", "ret_1",
        "market_score", "industry_score", "quality_score", "trend_score", "execution_score",
        "rule_score", "ml_probability", "combined_score", "relay_score", "relay_status", "signal",
        "entry_price", "stop_price", "stop_pct", "suggested_shares", "suggested_value", "action",
        "ma5", "ma10", "ma20", "prior_high20", "breakout20", "gap", "atr14_pct",
        "market_breadth_ma20", "industry_ret_1", "industry_ret_5", "industry_ret_10", "industry_breadth_ma20",
        "vol_ratio5", "turnover_rate", "rsi14", "atr14", "atr14_pct", "prior_high20",
        "roe", "net_profit_yoy", "main_net_inflow", "financial_report_date", "flow_date",
        "pe_ttm", "pb", "market_cap", "amount", "industry_member_count"
    ]
    existing = [c for c in cols if c in latest.columns]
    latest = latest.sort_values(["combined_score", "relay_score"], ascending=False)
    return latest[existing].reset_index(drop=True), risk
