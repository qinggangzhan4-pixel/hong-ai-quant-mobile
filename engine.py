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
    latest = feat[feat["date"] == latest_date].copy()
    latest = latest[~latest["symbol"].map(lambda s: _is_excluded(s, config))].copy()
    if config["universe"].get("exclude_st", True):
        latest = latest[~latest["is_st"].fillna(False)].copy()
    latest = latest[(latest["close"] >= config["universe"]["min_price"]) & (latest["close"] <= config["universe"]["max_price"])]

    parts = latest.apply(score_row, axis=1, result_type="expand")
    latest = pd.concat([latest.reset_index(drop=True), parts.reset_index(drop=True)], axis=1)

    model = load_model(model_path) if model_path else None
    prob = predict_probability(model, latest) if model is not None else np.full(len(latest), np.nan)
    latest["ml_probability"] = prob
    rw = config["strategy"]["rule_weight"]
    mw = config["strategy"]["model_weight"]
    latest["combined_score"] = latest["rule_score"]
    use_ml = ~pd.isna(latest["ml_probability"])
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
    is_relay = latest["is_limit_up"].fillna(False) | (latest["ret_1"] >= 0.095)
    latest.loc[is_relay & (latest["relay_score"] < 50), "signal"] = "AVOID_RELAY"
    latest.loc[is_relay & (latest["relay_score"] >= 70), "signal"] = "RELAY_CANDIDATE"

    risk = portfolio_risk(holdings, float(config["account"]["equity"])) if holdings is not None else {}
    cols = [
        "date", "symbol", "name", "industry", "close", "ret_1",
        "market_score", "industry_score", "quality_score", "trend_score", "execution_score",
        "rule_score", "ml_probability", "combined_score", "relay_score", "relay_status", "signal",
        "stop_price", "stop_pct", "suggested_shares", "suggested_value", "action",
        "market_breadth_ma20", "industry_ret_5", "industry_ret_10", "vol_ratio5", "turnover_rate", "rsi14"
    ]
    existing = [c for c in cols if c in latest.columns]
    latest = latest.sort_values(["combined_score", "relay_score"], ascending=False)
    return latest[existing].reset_index(drop=True), risk
