from pathlib import Path
import sys
import types
import json
import base64
import math
import pandas as pd
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parent
if "ai_quant" not in sys.modules:
    _pkg = types.ModuleType("ai_quant")
    _pkg.__path__ = [str(ROOT)]
    _pkg.__package__ = "ai_quant"
    sys.modules["ai_quant"] = _pkg

from ai_quant.config import load_config
from ai_quant.engine import score_universe
from ai_quant.risk import portfolio_risk
from ai_quant.data import fetch_live_a_scan, fetch_realtime_etf_snapshot, fetch_realtime_a_snapshot, fetch_watchlist_scan, fetch_watchlist_fundamentals_and_flow, add_watchlist_industry_context, score_live_industry_context
from ai_quant.global_context import (
    CONSUMER_CATEGORIES, classify_market_sector, fetch_global_market_snapshot,
    fetch_global_news, score_global_market,
)

ROOT = Path(__file__).resolve().parent
CFG_PATH = ROOT / "config.yaml"
MODEL_PATH = ROOT / "models" / "model.joblib"
DEMO_PATH = ROOT / "examples" / "demo_market.csv"
HOLDINGS_TEMPLATE = ROOT / "examples" / "holdings_template.csv"

st.set_page_config(
    page_title="洪老师AI量化 v2.5.0",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {max-width: 780px; padding-top: 1rem; padding-bottom: 5rem;}
[data-testid="stMetric"] {background: #171B22; border: 1px solid #2B313B; padding: .65rem; border-radius: 14px;}
[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
.card {background:#171B22;border:1px solid #2B313B;border-radius:16px;padding:14px 14px 10px 14px;margin:8px 0 12px 0;}
.card h3 {margin:0 0 8px 0;font-size:1.05rem;}
.badge {display:inline-block;padding:3px 8px;border-radius:999px;background:#232A35;margin-right:5px;font-size:.78rem;}
.muted {opacity:.72;font-size:.85rem;}
.bigscore {font-size:1.7rem;font-weight:700;line-height:1.1;}
.stButton>button, .stDownloadButton>button {width:100%;border-radius:12px;height:44px;}
div[data-baseweb="tab-list"] {gap: 3px; overflow-x:auto;}
button[data-baseweb="tab"] {white-space:nowrap;padding-left:10px;padding-right:10px;}
@media (max-width: 640px) {
  .block-container {padding-left:.75rem;padding-right:.75rem;padding-top:.6rem;}
  h1 {font-size:1.45rem !important;}
  h2 {font-size:1.2rem !important;}
  .card {padding:12px;}
}
</style>
""",
    unsafe_allow_html=True,
)

cfg = load_config(str(CFG_PATH))

# ---------------- helpers ----------------
def pct(v):
    try:
        if pd.isna(v): return "—"
        return f"{float(v)*100:.2f}%"
    except Exception:
        return "—"


def num(v, nd=2):
    try:
        if pd.isna(v): return "—"
        return f"{float(v):.{nd}f}"
    except Exception:
        return "—"


def signal_cn(s):
    return {
        "BUY_CANDIDATE":"允许首仓",
        "WATCH":"观察",
        "AVOID":"回避",
        "AVOID_RELAY":"接力回避",
        "RELAY_CANDIDATE":"接力候选",
    }.get(str(s), str(s))


def load_demo():
    return pd.read_csv(DEMO_PATH, dtype={"symbol": str})


HOLDING_INPUT_COLUMNS = ["symbol","name","industry","shares","available_shares","cost_price","current_price"]
HOLDING_COLUMNS = HOLDING_INPUT_COLUMNS + ["market_value","pnl","pnl_pct"]

CURRENT_HOLDINGS_REV = "20261124_1352"

# 516310 保留在账户总市值/现金核对中，但按用户要求不参与持仓评分、行业集中度和止盈止损。
EXCLUDED_HOLDING_SYMBOLS = {"516310"}

DEFAULT_HOLDINGS = [
    {"symbol":"601398","name":"工商银行","industry":"银行","shares":200,"available_shares":0,"cost_price":9.154,"current_price":9.190},
    {"symbol":"600901","name":"江苏金租","industry":"非银金融","shares":2200,"available_shares":1300,"cost_price":7.741,"current_price":7.780},
    {"symbol":"516310","name":"银行ETF易方达","industry":"ETF忽略","shares":100,"available_shares":100,"cost_price":-3.448,"current_price":1.523},
]


def clean_holdings(df):
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=HOLDING_COLUMNS)
    h = df.copy()
    for c in HOLDING_INPUT_COLUMNS:
        if c not in h.columns:
            h[c] = "" if c in ["symbol","name","industry"] else 0
    h["symbol"] = h["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    for c in ["shares","available_shares","cost_price","current_price"]:
        h[c] = pd.to_numeric(h[c], errors="coerce").fillna(0)
    h["shares"] = h["shares"].clip(lower=0).astype(int)
    h["available_shares"] = h["available_shares"].clip(lower=0).astype(int)
    h["available_shares"] = h[["available_shares","shares"]].min(axis=1).astype(int)
    h["market_value"] = h["shares"] * h["current_price"]
    h["pnl"] = (h["current_price"] - h["cost_price"]) * h["shares"]
    h["pnl_pct"] = np.where(h["cost_price"] > 0, h["current_price"] / h["cost_price"] - 1, np.nan)
    return h[HOLDING_COLUMNS]


def holdings_for_analysis(holdings):
    """排除用户明确要求忽略的ETF，但保留它在账户总市值核对中。"""
    h = clean_holdings(holdings)
    if h.empty:
        return h
    return h.loc[~h["symbol"].isin(EXCLUDED_HOLDING_SYMBOLS)].reset_index(drop=True)


def _encode_holdings_rows(rows):
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def _holdings_from_url():
    try:
        rev = st.query_params.get("holdings_rev", "")
        token = st.query_params.get("holdings", "")
        # 用户刚明确要求更新到最新截图时，用版本号覆盖旧网址里的历史持仓一次。
        if rev != CURRENT_HOLDINGS_REV or not token:
            return clean_holdings(pd.DataFrame(DEFAULT_HOLDINGS))
        pad = "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode((token + pad).encode("ascii")).decode("utf-8")
        rows = json.loads(raw)
        if not rows:
            return pd.DataFrame(columns=HOLDING_COLUMNS)
        return clean_holdings(pd.DataFrame(rows))
    except Exception:
        return clean_holdings(pd.DataFrame(DEFAULT_HOLDINGS))


def _holdings_to_url(df):
    try:
        h = clean_holdings(df)
        rows = [] if h.empty else h[HOLDING_INPUT_COLUMNS].to_dict("records")
        token = _encode_holdings_rows(rows)
        if st.query_params.get("holdings", "") != token:
            st.query_params["holdings"] = token
        if st.query_params.get("holdings_rev", "") != CURRENT_HOLDINGS_REV:
            st.query_params["holdings_rev"] = CURRENT_HOLDINGS_REV
    except Exception:
        pass


def _ceil_lot(shares):
    if shares <= 0:
        return 0
    return int(math.ceil(float(shares) / 100.0) * 100)


def _floor_lot(shares):
    if shares <= 0:
        return 0
    return int(float(shares) // 100 * 100)


def holding_action_table(holdings):
    h = holdings_for_analysis(holdings)
    if h.empty:
        return pd.DataFrame()
    equity = float(cfg["account"]["equity"])
    max_single = float(cfg["account"]["max_single_stock"])
    max_ind = float(cfg["account"]["max_industry_exposure"])
    max_stop = float(cfg["strategy"]["max_stop_pct"])
    tp1 = float(cfg["strategy"]["first_take_profit_pct"])
    tp2 = float(cfg["strategy"]["second_take_profit_pct"])
    ind_values = h.groupby("industry", dropna=False)["market_value"].sum().to_dict()

    score_lookup = {}
    if st.session_state.get("scores") is not None and len(st.session_state.scores):
        tmp = st.session_state.scores.copy()
        tmp["symbol"] = tmp["symbol"].astype(str).str.zfill(6)
        score_lookup = {str(r["symbol"]): r for _, r in tmp.iterrows()}

    rows = []
    for _, r in h.iterrows():
        symbol = str(r["symbol"]).zfill(6)
        mv = float(r["market_value"])
        price = float(r["current_price"])
        shares = int(r["shares"])
        available = int(r["available_shares"])
        weight = mv / equity if equity else 0
        ind_value = float(ind_values.get(r["industry"], 0))
        ind_weight = ind_value / equity if equity else 0
        pnl_pct = float(r["pnl_pct"]) if not pd.isna(r["pnl_pct"]) else np.nan

        single_excess = max(0.0, mv - equity * max_single)
        reduce_for_single = _ceil_lot(single_excess / price) if price > 0 else 0
        reduce_now = min(reduce_for_single, _floor_lot(available))
        industry_excess = max(0.0, ind_value - equity * max_ind)

        cost_price = float(r["cost_price"])
        risk_line = cost_price * (1 - max_stop) if cost_price > 0 else np.nan
        target1_price = cost_price * (1 + tp1) if cost_price > 0 else np.nan
        target2_price = cost_price * (1 + tp2) if cost_price > 0 else np.nan
        planned_trim_shares = min(shares, max(100, _floor_lot(shares / 3))) if shares >= 100 else shares
        sellable_trim_shares = min(planned_trim_shares, _floor_lot(available))

        reasons = []
        priority = "持有观察"
        if weight > max_single:
            priority = "减仓"
            if reduce_for_single > 0:
                if reduce_now >= reduce_for_single:
                    reasons.append(f"单票{weight*100:.1f}%>15%，按上限至少减{reduce_for_single}股")
                elif available > 0:
                    reasons.append(f"单票{weight*100:.1f}%>15%，理论需减{reduce_for_single}股；今日可卖{_floor_lot(available)}股")
                else:
                    reasons.append(f"单票{weight*100:.1f}%>15%，但今日可卖0股（T+1）")
        if ind_weight > max_ind:
            if priority == "持有观察":
                priority = "优先降行业仓"
            reasons.append(f"{r['industry']}仓位{ind_weight*100:.1f}%>30%，行业合计需降约{industry_excess:.0f}元")
        if not pd.isna(pnl_pct):
            if pnl_pct <= -max_stop:
                priority = "止损/减风险"
                reasons.append(f"回撤{pnl_pct*100:.1f}%已触及{max_stop*100:.0f}%成本风险线")
            elif pnl_pct <= -0.05:
                reasons.append(f"回撤{pnl_pct*100:.1f}%接近风险线，禁止补仓")
            elif pnl_pct >= tp2:
                reasons.append(f"浮盈{pnl_pct*100:.1f}%进入第二止盈区；计划再锁利{planned_trim_shares}股，当前可卖{sellable_trim_shares}股")
            elif pnl_pct >= tp1:
                reasons.append(f"浮盈{pnl_pct*100:.1f}%进入第一止盈区；计划锁利{planned_trim_shares}股，当前可卖{sellable_trim_shares}股")

        score = score_lookup.get(symbol)
        combined = np.nan
        signal = ""
        score_stop = np.nan
        if score is not None:
            try:
                combined = float(score.get("combined_score", np.nan))
            except Exception:
                pass
            signal = signal_cn(score.get("signal", ""))
            try:
                score_stop = float(score.get("stop_price", np.nan))
            except Exception:
                pass
            raw_signal = str(score.get("signal", ""))
            if raw_signal in ["AVOID","AVOID_RELAY"] and weight > 0:
                priority = "优先减仓"
                reasons.append("量化评分当前偏弱")
            elif raw_signal in ["BUY_CANDIDATE","RELAY_CANDIDATE"] and priority == "持有观察":
                priority = "持有/等触发"
                reasons.append("量化评分允许进入候选")

        rows.append({
            "代码": symbol,
            "名称": r["name"],
            "持股": shares,
            "可卖": available,
            "成本": round(float(r["cost_price"]), 3),
            "现价": round(price, 3),
            "市值": round(mv, 2),
            "盈亏": round(float(r["pnl"]), 2),
            "盈亏%": None if pd.isna(pnl_pct) else round(pnl_pct * 100, 2),
            "仓位%": round(weight * 100, 1),
            "行业仓位%": round(ind_weight * 100, 1),
            "成本止损价": None if pd.isna(risk_line) else round(risk_line, 3),
            "第一止盈价": None if pd.isna(target1_price) else round(target1_price, 3),
            "第二止盈价": None if pd.isna(target2_price) else round(target2_price, 3),
            "计划分批卖出股数": int(planned_trim_shares),
            "当前可卖止盈股数": int(sellable_trim_shares),
            "量化分": None if pd.isna(combined) else round(combined, 1),
            "信号": signal,
            "评分止损": None if pd.isna(score_stop) else round(score_stop, 3),
            "建议": priority,
            "说明": "；".join(reasons) if reasons else "暂无强制动作，继续观察",
        })
    return pd.DataFrame(rows)



def _live_market_score(meta):
    if not meta:
        return 10.0
    b = float(meta.get("up_ratio", 0.5))
    if b >= 0.65:
        score = 12
    elif b >= 0.55:
        score = 10
    elif b >= 0.45:
        score = 8
    elif b >= 0.35:
        score = 5
    else:
        score = 2
    score += 4 if float(meta.get("avg_pct_change", 0)) > 0 else 1
    lu = int(meta.get("limit_up", 0))
    ld = int(meta.get("limit_down", 0))
    if lu > ld * 1.2:
        score += 4
    elif ld > lu * 1.5:
        score += 1
    else:
        score += 2
    return float(min(score, 20))


def _apply_live_market_context(scores, meta):
    if scores is None or len(scores) == 0:
        return scores
    x = scores.copy()
    live_score = _live_market_score(meta)
    old_market = pd.to_numeric(x["market_score"], errors="coerce").fillna(0)
    x["market_score"] = live_score
    x["rule_score"] = pd.to_numeric(x["rule_score"], errors="coerce").fillna(0) - old_market + live_score

    ml = pd.to_numeric(x.get("ml_probability", np.nan), errors="coerce")
    rw = float(cfg["strategy"]["rule_weight"])
    mw = float(cfg["strategy"]["model_weight"])
    x["combined_score"] = x["rule_score"]
    if isinstance(ml, pd.Series):
        use_ml = ml.notna()
        x.loc[use_ml, "combined_score"] = rw * x.loc[use_ml, "rule_score"] + mw * (ml[use_ml] * 100)

    buy_th = float(cfg["strategy"]["score_buy_threshold"])
    watch_th = float(cfg["strategy"]["score_watch_threshold"])
    x["signal"] = np.where(
        x["combined_score"] >= buy_th,
        "BUY_CANDIDATE",
        np.where(x["combined_score"] >= watch_th, "WATCH", "AVOID"),
    )
    if "relay_status" in x.columns and "relay_score" in x.columns:
        applicable = x["relay_status"].astype(str) != "not_applicable"
        x.loc[applicable & (x["relay_score"] < 50), "signal"] = "AVOID_RELAY"
        x.loc[applicable & (x["relay_score"] >= 70), "signal"] = "RELAY_CANDIDATE"
    return x.sort_values(["combined_score", "relay_score"], ascending=False).reset_index(drop=True)


def _apply_global_market_context(scores, snapshot=None):
    """把全球隔夜情绪纳入市场环境20分；本地A股环境与全球环境各占一半。"""
    if scores is None or len(scores) == 0:
        return scores
    x = scores.copy()
    x["sector_category"] = x.apply(
        lambda r: classify_market_sector(r.get("name"), r.get("industry")), axis=1
    )
    x["global_market_score"] = np.nan
    if snapshot is not None and len(snapshot):
        global_score, breakdown = score_global_market(
            snapshot, st.session_state.get("global_news_adjustment", 0.0)
        )
        old_market = pd.to_numeric(x["market_score"], errors="coerce").fillna(10.0)
        blended_market = old_market * 0.5 + float(global_score) * 0.5
        x["market_score"] = blended_market.round(2)
        x["rule_score"] = (
            pd.to_numeric(x["rule_score"], errors="coerce").fillna(0)
            - old_market + blended_market
        ).round(2)
        x["global_market_score"] = float(global_score)
        st.session_state.global_market_breakdown = breakdown

    ml = pd.to_numeric(x.get("ml_probability", np.nan), errors="coerce")
    rw = float(cfg["strategy"]["rule_weight"])
    mw = float(cfg["strategy"]["model_weight"])
    x["combined_score"] = pd.to_numeric(x["rule_score"], errors="coerce").fillna(0)
    if isinstance(ml, pd.Series):
        use_ml = ml.notna()
        x.loc[use_ml, "combined_score"] = (
            rw * x.loc[use_ml, "rule_score"] + mw * (ml[use_ml] * 100)
        )

    buy_th = float(cfg["strategy"]["score_buy_threshold"])
    watch_th = float(cfg["strategy"]["score_watch_threshold"])
    x["signal"] = np.where(
        x["combined_score"] >= buy_th,
        "BUY_CANDIDATE",
        np.where(x["combined_score"] >= watch_th, "WATCH", "AVOID"),
    )
    if "relay_status" in x.columns and "relay_score" in x.columns:
        applicable = x["relay_status"].astype(str) != "not_applicable"
        x.loc[applicable & (x["relay_score"] < 50), "signal"] = "AVOID_RELAY"
        x.loc[applicable & (x["relay_score"] >= 70), "signal"] = "RELAY_CANDIDATE"
    return x.sort_values(
        ["combined_score", "relay_score"], ascending=False
    ).reset_index(drop=True)


def _refresh_holdings_prices(snapshot):
    h = clean_holdings(st.session_state.holdings)
    if h.empty or snapshot is None or len(snapshot) == 0:
        return h
    live = snapshot[["symbol", "close"]].copy()
    live["symbol"] = live["symbol"].astype(str).str.zfill(6)
    price_map = dict(zip(live["symbol"], pd.to_numeric(live["close"], errors="coerce")))

    # ETF 单独抓取；失败时不影响股票扫描。
    try:
        etf = fetch_realtime_etf_snapshot()
        if etf is not None and len(etf):
            etf["symbol"] = etf["symbol"].astype(str).str.zfill(6)
            for _, r in etf.iterrows():
                try:
                    if pd.notna(r["close"]):
                        price_map[str(r["symbol"])] = float(r["close"])
                except Exception:
                    pass
    except Exception:
        pass

    for i, r in h.iterrows():
        symbol = str(r["symbol"]).zfill(6)
        if symbol in price_map and pd.notna(price_map[symbol]):
            h.at[i, "current_price"] = float(price_map[symbol])
    h = clean_holdings(h)
    st.session_state.holdings = h
    _holdings_to_url(h)
    return h


def risk_panel(holdings):
    h = holdings_for_analysis(holdings)
    if h.empty:
        st.info("暂无纳入风控计算的持仓（银行ETF 516310 已按要求排除）。")
        return
    r = portfolio_risk(h, float(cfg["account"]["equity"]))
    c1, c2, c3 = st.columns(3)
    c1.metric("总仓", f"{r['total_exposure']*100:.1f}%")
    c2.metric("最大单票", f"{r['max_single_exposure']*100:.1f}%")
    c3.metric("最大行业", f"{r['max_industry_exposure']*100:.1f}%")
    if r["warnings"]:
        for w in r["warnings"]:
            st.warning(w)
    else:
        st.success("组合集中度暂未触发红线。")


def _float_or_nan(v):
    try:
        if pd.isna(v):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def trade_levels(row):
    """把量化评分翻译成可执行的买点/卖点。

    评分决定“有没有资格买”；价格触发决定“什么时候买”。
    """
    close = _float_or_nan(row.get("close"))
    score = _float_or_nan(row.get("combined_score", row.get("rule_score")))
    ma5 = _float_or_nan(row.get("ma5"))
    ma10 = _float_or_nan(row.get("ma10"))
    prior_high20 = _float_or_nan(row.get("prior_high20"))
    stop = _float_or_nan(row.get("stop_price"))
    vr = _float_or_nan(row.get("vol_ratio5"))
    rsi = _float_or_nan(row.get("rsi14"))
    turnover = _float_or_nan(row.get("turnover_rate"))
    gap = _float_or_nan(row.get("gap"))
    ret1 = _float_or_nan(row.get("ret_1"))

    # 回踩区：优先围绕 MA5；若 MA5 暂不可用则退回 MA10，再退回当前价附近。
    support = ma5
    if pd.isna(support) or support <= 0:
        support = ma10
    if pd.isna(support) or support <= 0:
        support = close * 0.99 if not pd.isna(close) else np.nan

    buy_low = support * 0.992 if not pd.isna(support) else np.nan
    buy_high = support * 1.008 if not pd.isna(support) else np.nan

    # 突破价：站上此前20日高点约0.2%，并要求量比>=1.2。
    breakout = prior_high20 * 1.002 if not pd.isna(prior_high20) and prior_high20 > 0 else np.nan

    # 当前还没触发时，用回踩区上沿作为计划入场参考；已触发则用现价。
    in_pullback = (
        not pd.isna(close) and not pd.isna(buy_low) and not pd.isna(buy_high)
        and buy_low <= close <= buy_high
    )
    breakout_ok = (
        not pd.isna(close) and not pd.isna(breakout)
        and close >= breakout and (pd.isna(vr) or vr >= 1.2)
    )
    entry_ref = close if (in_pullback or breakout_ok) else buy_high
    if pd.isna(entry_ref) or entry_ref <= 0:
        entry_ref = close

    tp1 = entry_ref * (1 + float(cfg["strategy"]["first_take_profit_pct"])) if not pd.isna(entry_ref) else np.nan
    tp2 = entry_ref * (1 + float(cfg["strategy"]["second_take_profit_pct"])) if not pd.isna(entry_ref) else np.nan

    # 统一换手率到百分数口径用于判断。
    turnover_pct = turnover
    if not pd.isna(turnover_pct) and turnover_pct <= 1:
        turnover_pct *= 100

    overheat = (
        (not pd.isna(rsi) and rsi >= 82)
        or (not pd.isna(turnover_pct) and turnover_pct >= 20)
        or (not pd.isna(gap) and gap >= 0.06)
        or (not pd.isna(ret1) and ret1 >= 0.085)
    )

    if pd.isna(score):
        status = "⚪ 数据不足"
        reason = "先补全评分数据"
    elif score < float(cfg["strategy"]["score_watch_threshold"]):
        status = "❌ 不买"
        reason = "综合分低于观察线"
    elif score < float(cfg["strategy"]["score_buy_threshold"]):
        status = "👀 观察"
        reason = "分数未到75，等评分和价格同时转强"
    elif not pd.isna(stop) and not pd.isna(close) and close <= stop:
        status = "🛑 破位"
        reason = "已到系统止损附近，禁止新开仓"
    elif overheat:
        status = "🔥 过热勿追"
        reason = "RSI/换手/高开/单日涨幅至少一项过热"
    elif in_pullback:
        status = "✅ 回踩买点"
        reason = "进入MA5附近回踩区，可按首仓股数试仓"
    elif breakout_ok:
        status = "✅ 突破买点"
        reason = "突破20日高点且量能满足确认条件"
    elif not pd.isna(close) and not pd.isna(buy_low) and close < buy_low:
        status = "⏳ 等重新站回"
        reason = "回踩偏深，先等重新站回买点区"
    else:
        status = "⏳ 等待触发"
        reason = "不追价，等回踩区或放量突破"

    return {
        "buy_low": buy_low,
        "buy_high": buy_high,
        "breakout": breakout,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "trend_guard": ma5,
        "status": status,
        "reason": reason,
    }


def candidate_card(row):
    sig = signal_cn(row.get("signal", ""))
    name = row.get("name", "") if not pd.isna(row.get("name", np.nan)) else ""
    symbol = str(row.get("symbol", "")).zfill(6)
    industry = row.get("industry", "") if not pd.isna(row.get("industry", np.nan)) else "未分类"
    score = row.get("combined_score", row.get("rule_score", np.nan))
    ml = row.get("ml_probability", np.nan)
    action = row.get("action", "")
    levels = trade_levels(row)

    html = f"""
<div class='card'>
  <h3>{name} <span class='muted'>{symbol}</span></h3>
  <span class='badge'>{industry}</span><span class='badge'>{sig}</span>
  <div style='display:flex;justify-content:space-between;align-items:flex-end;margin-top:10px;'>
    <div><div class='muted'>综合评分</div><div class='bigscore'>{num(score,1)}</div></div>
    <div style='text-align:right'><div class='muted'>现价</div><b>{num(row.get('close'))}</b></div>
  </div>

  <div style='margin-top:12px;padding:10px;border-radius:12px;background:#202630;'>
    <div style='font-size:1.02rem;font-weight:700'>{levels['status']}</div>
    <div class='muted' style='margin-top:3px'>{levels['reason']}</div>
  </div>

  <div style='margin-top:12px;line-height:1.8'>
    <b>回踩买点：</b>{num(levels['buy_low'])} ～ {num(levels['buy_high'])}<br>
    <b>突破确认：</b>{num(levels['breakout'])} <span class='muted'>（量比≥1.2）</span><br>
    <b>系统止损：</b>{num(levels['stop'])} ｜ <b>MA5警戒：</b>{num(levels['trend_guard'])}<br>
    <b>第一止盈：</b>{num(levels['tp1'])} <span class='muted'>（约+8%，可锁1/3）</span><br>
    <b>第二止盈：</b>{num(levels['tp2'])} <span class='muted'>（约+15%，再锁1/3）</span>
  </div>

  <div style='margin-top:10px'>
    建议：<b>{action}</b> ｜ 首仓 <b>{int(row.get('suggested_shares',0) or 0)}股</b>
  </div>

  <div class='muted' style='margin-top:8px'>
    市场 {num(row.get('market_score'),0)}/20 · 行业 {num(row.get('industry_score'),0)}/20 ·
    质量 {num(row.get('quality_score'),0)}/20 · 趋势 {num(row.get('trend_score'),0)}/25 ·
    执行 {num(row.get('execution_score'),0)}/15
  </div>
  <div class='muted' style='margin-top:5px'>
    AI概率：{('未启用' if pd.isna(ml) else f'{float(ml)*100:.1f}%')} ·
    量比 {num(row.get('vol_ratio5'))} · RSI {num(row.get('rsi14'),0)}
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ---------------- header ----------------
st.title("📈 洪老师 AI 量化 v2.5.0")
st.caption("手机网页版 · A股主板全行业轮动 + 美股/韩国/日本隔夜联动 + 消费专属质量因子 + 买卖点/持仓风控。")
st.caption("Build: 2026-11-25-global-consumer-v2.5.0 ｜ 新增全球市场情绪、隔夜新闻线索、消费行业分类与财务因子")
if st.session_state.get("auto_scan_error"):
    st.warning(f"自动评分未完成：{st.session_state.auto_scan_error}")

with st.expander("⚙️ 账户参数", expanded=False):
    equity = st.number_input("账户总资产（元）", min_value=10000.0, value=float(cfg["account"]["equity"]), step=1000.0)
    cfg["account"]["equity"] = float(equity)
    st.caption("默认：首仓6%、单票15%、单行业30%、单笔风险0.7%。可在 config.yaml 改。")

# Session state
if "market" not in st.session_state:
    st.session_state.market = None
if "scores" not in st.session_state:
    st.session_state.scores = None
if "live_meta" not in st.session_state:
    st.session_state.live_meta = None
if "live_snapshot" not in st.session_state:
    st.session_state.live_snapshot = None
if "live_selected" not in st.session_state:
    st.session_state.live_selected = None
if "global_market_snapshot" not in st.session_state:
    st.session_state.global_market_snapshot = pd.DataFrame()
if "global_news" not in st.session_state:
    st.session_state.global_news = pd.DataFrame()
if "global_news_adjustment" not in st.session_state:
    st.session_state.global_news_adjustment = 0.0
if "global_market_breakdown" not in st.session_state:
    st.session_state.global_market_breakdown = {}
if "global_market_error" not in st.session_state:
    st.session_state.global_market_error = ""
if "global_news_error" not in st.session_state:
    st.session_state.global_news_error = ""
if "scores_pre_global" not in st.session_state:
    st.session_state.scores_pre_global = None
if st.session_state.get("holdings_revision") != CURRENT_HOLDINGS_REV:
    st.session_state.holdings = _holdings_from_url()
    st.session_state.holdings_revision = CURRENT_HOLDINGS_REV
elif "holdings" not in st.session_state:
    st.session_state.holdings = _holdings_from_url()
if "auto_scan_once_done" not in st.session_state:
    st.session_state.auto_scan_once_done = False

def _live_snapshot_table(snapshot, limit=100):
    if snapshot is None or len(snapshot) == 0:
        return pd.DataFrame()
    x = snapshot.copy()
    cols = [c for c in ["symbol","name","close","pct_change","open","high","low","turnover_rate","live_volume_ratio","amount"] if c in x.columns]
    x = x[cols].copy()
    x = x.sort_values("pct_change", ascending=False).head(int(limit))
    x = x.rename(columns={
        "symbol":"代码","name":"名称","close":"现价","pct_change":"涨幅%",
        "open":"今开","high":"最高","low":"最低","turnover_rate":"换手%",
        "live_volume_ratio":"量比","amount":"成交额"
    })
    if "成交额" in x.columns:
        x["成交额"] = pd.to_numeric(x["成交额"], errors="coerce") / 1e8
        x = x.rename(columns={"成交额":"成交额(亿)"})
    return x.reset_index(drop=True)


# ---------------- tabs ----------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏆 今日候选", "🔎 查一只", "💼 我的持仓", "📤 数据", "🎯 科技观察池", "🌐 全球/消费"])

with tab4:
    st.subheader("实时量化数据")
    st.caption("这里不再只是“导入行情”。点击一次后会自动抓行情、拉历史、评分，并直接给出买卖点。")

    scan_count = st.selectbox("深度评分数量", [20, 30, 40, 50], index=1, key="data_scan_count")
    if st.button("⚡ 刷新实时行情并生成买卖点", type="primary", use_container_width=True):
        status = st.status("正在运行实时量化……", expanded=True)
        try:
            status.write("① 抓取全A实时行情")
            status.write("② 主板预筛")
            status.write(f"③ 对前 {int(scan_count)} 只拉取约150天历史")
            market, snapshot, selected, meta = fetch_live_a_scan(
                preselect_n=int(scan_count),
                history_calendar_days=150,
                max_workers=4,
            )
            status.write(
                f"✓ 实时行情：{len(snapshot):,}只 ｜ "
                f"接口总数约{snapshot.attrs.get('source_total', len(snapshot)):,}只 ｜ "
                f"{snapshot.attrs.get('total_pages', 1)}页"
            )
            st.session_state.market = market
            st.session_state.live_snapshot = snapshot
            st.session_state.live_selected = selected
            st.session_state.live_meta = meta

            status.write("④ 更新持仓现价")
            h = _refresh_holdings_prices(snapshot)

            status.write("⑤ 计算综合评分、买点、止损和止盈")
            scores, _ = score_universe(
                market,
                cfg,
                holdings_for_analysis(h),
                str(MODEL_PATH) if MODEL_PATH.exists() else None,
            )
            scores = _apply_live_market_context(scores, meta)
            st.session_state.scores = scores

            status.update(
                label=f"完成：全市场 {len(snapshot):,} 只，深度评分 {len(scores)} 只",
                state="complete",
                expanded=False,
            )
            st.success("实时量化完成，结果已经显示在下面。")
        except Exception as e:
            status.update(label="实时量化失败", state="error", expanded=True)
            st.error(f"实时量化失败：{e}")

    meta = st.session_state.live_meta
    snap = st.session_state.live_snapshot
    scores = st.session_state.scores

    if meta:
        c1, c2, c3 = st.columns(3)
        c1.metric("上涨/下跌", f"{meta.get('up',0)}/{meta.get('down',0)}")
        c2.metric("涨停/跌停", f"{meta.get('limit_up',0)}/{meta.get('limit_down',0)}")
        c3.metric("上涨占比", f"{meta.get('up_ratio',0)*100:.1f}%")
        st.caption(
            f"扫描时间：{meta.get('scan_time','')} ｜ 平均涨跌：{meta.get('avg_pct_change',0):+.2f}%"
        )

    if snap is not None and len(snap):
        st.markdown("### 🔥 实时涨幅榜")
        st.dataframe(_live_snapshot_table(snap, 30), use_container_width=True, hide_index=True)

    if scores is not None and len(scores):
        st.markdown("### 🎯 量化候选与买卖点")
        rows = []
        for _, rr in scores.head(20).iterrows():
            lv = trade_levels(rr)
            rows.append({
                "代码": str(rr.get("symbol","")).zfill(6),
                "名称": rr.get("name",""),
                "综合分": round(float(rr.get("combined_score", rr.get("rule_score", 0))), 1),
                "状态": lv["status"],
                "现价": round(float(rr.get("close", 0)), 3),
                "回踩下沿": None if pd.isna(lv["buy_low"]) else round(float(lv["buy_low"]), 3),
                "回踩上沿": None if pd.isna(lv["buy_high"]) else round(float(lv["buy_high"]), 3),
                "突破确认": None if pd.isna(lv["breakout"]) else round(float(lv["breakout"]), 3),
                "止损": None if pd.isna(lv["stop"]) else round(float(lv["stop"]), 3),
                "止盈1": None if pd.isna(lv["tp1"]) else round(float(lv["tp1"]), 3),
                "止盈2": None if pd.isna(lv["tp2"]) else round(float(lv["tp2"]), 3),
                "首仓股数": int(rr.get("suggested_shares",0) or 0),
            })
        result_df = pd.DataFrame(rows)
        st.dataframe(result_df, use_container_width=True, hide_index=True)
        st.caption("优先看“✅回踩买点 / ✅突破买点”；“⏳等待触发”不要追；“🔥过热勿追 / ❌不买”直接跳过。")

        top_ready = result_df[result_df["状态"].astype(str).str.startswith("✅")]
        if len(top_ready):
            st.markdown("### ✅ 当前已触发买点")
            st.dataframe(top_ready, use_container_width=True, hide_index=True)
        else:
            st.info("当前没有股票触发买点。评分高但没到价格触发条件时，继续等。")

    st.divider()
    with st.expander("📄 手动CSV / 演示数据", expanded=False):
        uploaded = st.file_uploader("行情CSV", type=["csv"], key="market_file")
        c1, c2 = st.columns(2)
        use_demo = c1.button("载入演示数据")
        clear_data = c2.button("清空数据")
        if uploaded is not None:
            try:
                st.session_state.market = pd.read_csv(uploaded, dtype={"symbol": str})
                st.success(f"已读取 {len(st.session_state.market):,} 行行情。")
            except Exception as e:
                st.error(f"读取失败：{e}")
        if use_demo:
            st.session_state.market = load_demo()
            st.success("已载入演示数据。")
        if clear_data:
            st.session_state.market = None
            st.session_state.scores = None
            st.session_state.live_snapshot = None
            st.session_state.live_selected = None
            st.session_state.live_meta = None
            st.rerun()

        if st.session_state.market is not None:
            st.dataframe(st.session_state.market.tail(20), use_container_width=True, hide_index=True)
            if st.button("🚀 对当前数据运行评分", type="primary"):
                try:
                    h = holdings_for_analysis(st.session_state.holdings)
                    scores, _ = score_universe(
                        st.session_state.market,
                        cfg,
                        h,
                        str(MODEL_PATH) if MODEL_PATH.exists() else None,
                    )
                    st.session_state.scores_pre_global = scores.copy()
                    scores = _apply_global_market_context(
                        scores, st.session_state.global_market_snapshot
                    )
                    st.session_state.scores = scores
                    st.success(f"完成：{len(scores)} 只股票已评分。")
                except Exception as e:
                    st.error(f"评分失败：{e}")


with tab5:
    st.subheader("主板科技观察池")
    st.caption("固定8只沪深主板股票；逐只抓取实时快照和至少80个交易日的日K，再用现有量化引擎评分。")
    st.caption("评分口径说明：观察池只有8只，市场广度和行业广度采用中性分，不冒充全市场广度；缺失的ROE、净利润增速、主力净流入不会编造。")

    watch_state = st.session_state.get("watchlist_scores")
    watch_meta = st.session_state.get("watchlist_meta")

    if st.button("📥 接入8只股票行情并运行量化", type="primary", use_container_width=True):
        status = st.status("正在接入观察池数据……", expanded=True)
        try:
            status.write("① 获取全A实时快照，并筛出8只主板观察股")
            watch_market, watch_snapshot, watch_meta = fetch_watchlist_scan(
                history_calendar_days=260,
                max_workers=4,
            )
            status.write(f"② 已取得 {watch_market['symbol'].nunique()} 只股票的历史K线，校验至少80个交易日")
            watch_cfg = json.loads(json.dumps(cfg))
            watch_cfg["universe"]["max_price"] = 1000.0
            status.write("③ 补齐最新财报ROE、归母净利润同比和个股主力资金流")
            quality_flow = fetch_watchlist_fundamentals_and_flow(
                watch_market["symbol"].astype(str).str.zfill(6).unique().tolist(),
                max_workers=4,
            )
            watch_market = watch_market.merge(quality_flow, on="symbol", how="left")
            watch_market = add_watchlist_industry_context(
                watch_market, watch_snapshot,
                watch_market["symbol"].astype(str).str.zfill(6).unique().tolist(),
            )

            status.write("④ 使用全A市场环境、真实行业横截面及个股基本面运行评分")
            watch_cfg = json.loads(json.dumps(cfg))
            watch_cfg["universe"]["max_price"] = 1000.0
            watch_scores, _ = score_universe(
                watch_market,
                watch_cfg,
                holdings_for_analysis(st.session_state.holdings),
                None,
            )
            previous_industry = pd.to_numeric(watch_scores["industry_score"], errors="coerce").fillna(10)
            watch_scores = score_live_industry_context(watch_scores, watch_snapshot)
            new_industry = pd.to_numeric(watch_scores["industry_score"], errors="coerce").fillna(10)
            watch_scores["rule_score"] = (
                pd.to_numeric(watch_scores["rule_score"], errors="coerce").fillna(0)
                - previous_industry + new_industry
            ).round(2)

            # 观察池只有8只，不能用8只样本计算全市场广度；
            # 用全A快照的涨跌家数与平均涨跌替换市场环境分。
            live_market_score = _live_market_score(watch_meta)
            old_market = pd.to_numeric(watch_scores["market_score"], errors="coerce").fillna(0)
            watch_scores["market_score"] = live_market_score
            watch_scores["rule_score"] = (
                pd.to_numeric(watch_scores["rule_score"], errors="coerce").fillna(0)
                - old_market + live_market_score
            ).round(2)
            watch_scores["combined_score"] = watch_scores["rule_score"]
            watch_scores = _apply_global_market_context(
                watch_scores, st.session_state.global_market_snapshot
            )

            quality_failures = quality_flow.attrs.get("failures", [])
            watch_meta["quality_flow_failures"] = quality_failures
            watch_meta["quality_flow_success"] = int(
                quality_flow[["roe", "net_profit_yoy", "main_net_inflow"]].notna().any(axis=1).sum()
            )
            watch_meta["quality_flow_complete"] = int(
                quality_flow[["roe", "net_profit_yoy", "main_net_inflow"]].notna().all(axis=1).sum()
            )
            st.session_state.watchlist_market = watch_market
            st.session_state.watchlist_snapshot = watch_snapshot
            st.session_state.watchlist_meta = watch_meta
            st.session_state.watchlist_scores = watch_scores

            status.update(
                label=f"完成：成功接入 {watch_meta['loaded_symbols']}/8 只观察股",
                state="complete",
                expanded=False,
            )
            st.success("观察池行情和历史K线已接入，量化结果已生成。")
        except Exception as exc:
            status.update(label="观察池接入失败", state="error", expanded=True)
            st.error(f"接入失败：{exc}")

    watch_state = st.session_state.get("watchlist_scores")
    watch_meta = st.session_state.get("watchlist_meta")
    if watch_meta:
        c1, c2, c3 = st.columns(3)
        c1.metric("成功接入", f"{watch_meta.get('loaded_symbols',0)}/8")
        c2.metric("全市场上涨占比", f"{watch_meta.get('up_ratio',0)*100:.1f}%")
        c3.metric("耗时", f"{watch_meta.get('elapsed_seconds',0):.1f}s")
        st.caption(f"数据时间：{watch_meta.get('scan_time','')} ｜ 历史窗口：{watch_meta.get('history_calendar_days',0)}个自然日")
        st.caption(f"财务/资金流：{watch_meta.get('quality_flow_success',0)}/8只至少有一项返回；完整三项：{watch_meta.get('quality_flow_complete',0)}/8只")
        if watch_meta.get("failed_symbols"):
            st.warning("行情或历史数据失败：" + "、".join(watch_meta["failed_symbols"]))
        if watch_meta.get("short_history"):
            st.warning("历史不足80个交易日，未参与评分：" + "、".join(watch_meta["short_history"]))
        if watch_meta.get("missing_live"):
            st.warning("实时快照未找到：" + "、".join(watch_meta["missing_live"]))
        if watch_meta.get("quality_flow_failures"):
            st.warning("部分财务/资金流接口未返回：" + "；".join(watch_meta["quality_flow_failures"]))
        st.caption("财务指标取最新公开报告期；主力净流入取最新可用日频资金流，单位元；行业评分依据全A同行业当日涨跌和上涨家数。不同数据源若未返回，字段留空，不用估算值代替。")

    if watch_state is not None and len(watch_state):
        rows = []
        for _, rr in watch_state.iterrows():
            lv = trade_levels(rr)
            rows.append({
                "代码": str(rr.get("symbol","")).zfill(6),
                "名称": rr.get("name",""),
                "评分": round(float(rr.get("combined_score",0)),1),
                "现价": round(float(rr.get("close",0)),2),
                "MA5": round(float(rr.get("ma5",np.nan)),2) if pd.notna(rr.get("ma5")) else None,
                "MA10": round(float(rr.get("ma10",np.nan)),2) if pd.notna(rr.get("ma10")) else None,
                "回踩区间": f"{lv['buy_low']:.2f}–{lv['buy_high']:.2f}" if pd.notna(lv["buy_low"]) and pd.notna(lv["buy_high"]) else "数据不足",
                "突破确认": round(float(lv["breakout"]),2) if pd.notna(lv["breakout"]) else None,
                "止损": round(float(lv["stop"]),2) if pd.notna(lv["stop"]) else None,
                "止盈1": round(float(lv["tp1"]),2) if pd.notna(lv["tp1"]) else None,
                "止盈2": round(float(lv["tp2"]),2) if pd.notna(lv["tp2"]) else None,
                "RSI14": round(float(rr.get("rsi14",np.nan)),1) if pd.notna(rr.get("rsi14")) else None,
                "ATR14%": round(float(rr.get("atr14_pct",np.nan))*100,2) if pd.notna(rr.get("atr14_pct")) else None,
                "量比": round(float(rr.get("vol_ratio5",np.nan)),2) if pd.notna(rr.get("vol_ratio5")) else None,
                "ROE%": round(float(rr.get("roe",np.nan)),2) if pd.notna(rr.get("roe")) else None,
                "归母净利同比%": round(float(rr.get("net_profit_yoy",np.nan)),2) if pd.notna(rr.get("net_profit_yoy")) else None,
                "主力净流入(亿)": round(float(rr.get("main_net_inflow",np.nan))/1e8,2) if pd.notna(rr.get("main_net_inflow")) else None,
                "行业评分": round(float(rr.get("industry_score",np.nan)),1) if pd.notna(rr.get("industry_score")) else None,
                "财报日期": str(rr.get("financial_report_date","") or ""),
                "资金日期": str(rr.get("flow_date","") or ""),
                "状态": lv["status"],
            })
        st.markdown("### 八只股票正式量化结果（已叠加全球市场因子）")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("突破信号要求站上前20日高点上方约0.2%，且量比至少1.2；回踩参考MA5/MA10。止盈按计划入场参考价计算，未触发前不是实际订单。")
        csv = watch_state.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载观察池评分CSV", csv, "mainboard_tech_watchlist_scores.csv", "text/csv")
    else:
        st.info("点击上方按钮，系统才会抓取这8只股票的行情和历史K线。")

with tab6:
    st.subheader("全球市场联动与消费行业")
    st.caption("海外行情通过公开行情接口获取；各市场交易时间不同。全球因子只占市场环境分的一半，数据缺失时不伪造涨跌。")

    if st.button("🌐 刷新美股、韩国、日本行情及隔夜新闻", type="primary", use_container_width=True):
        with st.status("正在更新全球市场数据……", expanded=True) as global_status:
            try:
                global_status.write("① 获取美股指数、费半、英伟达、美光、AMD、戴尔")
                st.session_state.global_market_snapshot = fetch_global_market_snapshot()
                st.session_state.global_market_error = ""
                global_status.write("② 获取韩国KOSPI/KOSDAQ、三星、SK海力士及日经")
                global_status.write("③ 搜集海外市场、宏观、关税、消费政策新闻")
                st.session_state.global_news = fetch_global_news()
                st.session_state.global_news_error = ""
                global_status.update(label="全球市场数据更新完成", state="complete", expanded=False)
            except Exception as global_exc:
                st.session_state.global_market_error = str(global_exc)
                global_status.update(label="部分全球数据更新失败", state="error", expanded=True)

    tone = st.selectbox(
        "隔夜新闻综合影响（请核对新闻后确认，避免仅凭标题自动误判）",
        ["明显利空", "略偏空", "中性", "略偏多", "明显利好"],
        index=2,
        key="global_news_tone",
    )
    tone_adjustment = {
        "明显利空": -1.5, "略偏空": -0.75, "中性": 0.0,
        "略偏多": 0.75, "明显利好": 1.5,
    }[tone]
    if st.session_state.global_news_adjustment != tone_adjustment:
        st.session_state.global_news_adjustment = tone_adjustment
        if st.session_state.scores_pre_global is not None:
            st.session_state.scores = _apply_global_market_context(
                st.session_state.scores_pre_global,
                st.session_state.global_market_snapshot,
            )

    global_snapshot = st.session_state.global_market_snapshot
    if global_snapshot is not None and len(global_snapshot):
        global_score, global_breakdown = score_global_market(
            global_snapshot, st.session_state.global_news_adjustment
        )
        st.markdown("### 隔夜全球风险评分")
        c1, c2, c3 = st.columns(3)
        c1.metric("全球环境分", f"{global_score:.1f}/10", help="全球环境原始分20分，展示时折算为10分；与A股本地10分合并为市场环境20分。")
        c2.metric("成功接入", f"{len(global_snapshot)}/{len(GLOBAL_ASSETS) if 'GLOBAL_ASSETS' in globals() else 13}")
        c3.metric("A股市场分占比", "50%")
        show_global = global_snapshot.rename(columns={
            "ticker": "代码", "name": "市场/股票", "market": "市场",
            "price": "最新价", "pct_change": "涨跌幅%", "market_time_cn": "行情时间",
            "source": "来源", "group": "因子组",
        })
        st.dataframe(show_global, use_container_width=True, hide_index=True)
        st.caption("行情时间按各市场本地交易时间换算为北京时间；美股/日韩休市时，最新一笔数据可能是上一交易日收盘，不等于当前实时价格。")
        if st.session_state.global_market_breakdown:
            st.json(st.session_state.global_market_breakdown)
    else:
        st.info("尚未取得全球行情。点击上方按钮更新；接口不可用时，不会把缺失数据当作利空。")
    if st.session_state.global_market_error:
        st.warning("海外行情提示：" + st.session_state.global_market_error)

    st.markdown("### 隔夜新闻线索")
    news = st.session_state.global_news
    if news is not None and len(news):
        st.dataframe(news, use_container_width=True, hide_index=True)
        st.caption("新闻来自公开RSS聚合，仅用于发现线索；标题不等于事实结论，重要事件请点开原文核实。")
    else:
        st.info("暂无已加载新闻。点击上方刷新按钮获取新闻线索。")
    if st.session_state.global_news_error:
        st.warning("新闻抓取提示：" + st.session_state.global_news_error)

    st.divider()
    st.markdown("### 消费行业主板候选")
    st.caption("食品饮料、家电家居、零售电商、旅游酒店餐饮、美妆服饰珠宝、汽车、消费电子均纳入；专属财务因子仅在真实字段足够时启用。")
    if st.session_state.scores is not None and len(st.session_state.scores):
        consumer = st.session_state.scores.copy()
        if "sector_category" not in consumer.columns:
            consumer["sector_category"] = consumer.apply(
                lambda r: classify_market_sector(r.get("name"), r.get("industry")), axis=1
            )
        consumer = consumer[consumer["sector_category"].astype(str).str.startswith("消费/")]
        if consumer.empty:
            st.info("本次深度评分结果中暂无消费股。全市场预筛会兼顾行业代表性；可提高深度评分数量后重新扫描。")
        else:
            counts = consumer.groupby("sector_category").size().rename("入选数量").reset_index()
            st.dataframe(counts, use_container_width=True, hide_index=True)
            columns = [c for c in [
                "symbol", "name", "sector_category", "industry", "combined_score",
                "close", "signal", "consumer_quality_score", "roe", "net_profit_yoy",
                "revenue_yoy", "gross_margin", "inventory_turnover"
            ] if c in consumer.columns]
            consumer = consumer.sort_values("combined_score", ascending=False)
            consumer_show = consumer[columns].rename(columns={
                "symbol": "代码", "name": "名称", "sector_category": "消费分类",
                "industry": "行业", "combined_score": "综合评分", "close": "现价",
                "signal": "信号", "consumer_quality_score": "消费质量分/10",
                "roe": "ROE", "net_profit_yoy": "净利润同比",
                "revenue_yoy": "营收同比", "gross_margin": "毛利率",
                "inventory_turnover": "存货周转率",
            })
            st.dataframe(consumer_show.head(30), use_container_width=True, hide_index=True)
            st.caption("消费质量分至少需要两项可用的真实财务字段；缺失时显示空值，不用推测值填充。")
            st.download_button(
                "下载消费股候选CSV",
                consumer.to_csv(index=False).encode("utf-8-sig"),
                "consumer_candidates.csv",
                "text/csv",
            )
    else:
        st.info("先到“数据”页运行实时A股扫描，再回来查看消费行业候选。")

with tab3:
    st.subheader("我的持仓")
    st.caption("录入股数、可卖股数、成本价和现价；市值、盈亏和收益率自动计算。")

    c1, c2 = st.columns(2)
    if c1.button("恢复当前截图持仓"):
        st.session_state.holdings = clean_holdings(pd.DataFrame(DEFAULT_HOLDINGS))
        _holdings_to_url(st.session_state.holdings)
        st.rerun()
    if c2.button("清空持仓"):
        st.session_state.holdings = pd.DataFrame(columns=HOLDING_COLUMNS)
        _holdings_to_url(st.session_state.holdings)
        st.rerun()

    edited = st.data_editor(
        st.session_state.holdings[HOLDING_INPUT_COLUMNS],
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "symbol": st.column_config.TextColumn("代码"),
            "name": st.column_config.TextColumn("名称"),
            "industry": st.column_config.TextColumn("行业"),
            "shares": st.column_config.NumberColumn("持股", min_value=0, step=100, format="%d"),
            "available_shares": st.column_config.NumberColumn("可卖", min_value=0, step=100, format="%d"),
            "cost_price": st.column_config.NumberColumn("成本", min_value=-100.0, step=0.01, format="%.3f"),
            "current_price": st.column_config.NumberColumn("现价", min_value=0.0, step=0.01, format="%.3f"),
        },
        key="holdings_editor_v21",
    )
    st.session_state.holdings = clean_holdings(edited)
    _holdings_to_url(st.session_state.holdings)
    st.caption("✅ 持仓已自动保存；刷新页面会恢复。")

    h = st.session_state.holdings
    if not h.empty:
        equity_now = float(cfg["account"]["equity"])
        total_mv = float(h["market_value"].sum())
        total_pnl = float(h["pnl"].sum())
        cash_est = equity_now - total_mv
        c1, c2, c3 = st.columns(3)
        c1.metric("持仓市值", f"¥{total_mv:,.0f}")
        c2.metric("持仓盈亏", f"¥{total_pnl:+,.2f}")
        c3.metric("现金余量", f"¥{cash_est:,.0f}")

        risk_panel(h)

        st.markdown("### 📌 自动操作建议")
        advice = holding_action_table(h)
        st.dataframe(advice, use_container_width=True, hide_index=True)

        st.info("操作规则：成本回撤6%为风险退出线；盈利达到+8%计划先卖约1/3，达到+15%再卖约1/3，剩余仓位用MA5/趋势破位跟踪。A股按100股整数手执行；可卖股数受T+1限制。银行ETF 516310 保留在账户总市值核对中，但不参与本表分析。")
        st.caption("以上止盈价按录入成本计算；实时评分止损和MA5趋势线更新后优先参考，不能把计划价当作保证成交价。")
        csv = h.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载完整持仓CSV", csv, "holdings_full.csv", "text/csv")
        advice_csv = advice.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载操作建议CSV", advice_csv, "holding_actions.csv", "text/csv")
    else:
        risk_panel(h)


with tab1:
    st.subheader("今日候选")

    with st.expander("⚡ 实时 A 股扫描", expanded=True):
        st.caption("一键抓取全A实时快照 → 主板预筛 → 拉取近期历史 → 自动评分。科创板/创业板/北交所按当前配置排除。")
        scan_n = st.selectbox("深度评分股票数", [10, 20, 30, 40], index=1)
        if st.button("⚡ 获取实时行情并自动量化", type="primary", use_container_width=True):
            status = st.status("已收到请求：正在连接实时 A 股行情……", expanded=True)
            try:
                status.write("① 抓取全 A 股实时快照")
                status.write("② 主板快速预筛")
                status.write(f"③ 对前 {int(scan_n)} 只拉历史并计算 MA / ATR / RSI / 量价")
                market, snapshot, selected, meta = fetch_live_a_scan(
                    preselect_n=int(scan_n),
                    history_calendar_days=150,
                    max_workers=4,
                )
                status.write(
                    f"✓ 实时行情：{len(snapshot):,}只 ｜ "
                    f"接口总数约{snapshot.attrs.get('source_total', len(snapshot)):,}只 ｜ "
                    f"{snapshot.attrs.get('total_pages', 1)}页"
                )
                st.session_state.market = market
                st.session_state.live_snapshot = snapshot
                st.session_state.live_meta = meta
                st.session_state.live_selected = selected

                status.write("④ 更新当前持仓现价")
                h = _refresh_holdings_prices(snapshot)

                status.write("⑥ 运行量化评分与仓位风控")
                scores, _ = score_universe(
                    market,
                    cfg,
                    holdings_for_analysis(h),
                    str(MODEL_PATH) if MODEL_PATH.exists() else None,
                )
                scores = _apply_live_market_context(scores, meta)
                st.session_state.scores_pre_global = scores.copy()
                scores = _apply_global_market_context(
                    scores, st.session_state.global_market_snapshot
                )
                st.session_state.scores = scores
                status.update(
                    label=f"完成：{len(scores)} 只深度评分，用时 {meta.get('elapsed_seconds','—')} 秒",
                    state="complete",
                    expanded=False,
                )
                st.success(
                    f"实时扫描完成：全市场 {meta.get('count', 0)} 只，预筛 {meta.get('preselected', 0)} 只，"
                    f"深度评分 {len(scores)} 只。"
                )
                if st.session_state.global_market_error:
                    st.warning("全球市场因子未接入：" + st.session_state.global_market_error)
            except Exception as e:
                status.update(label="实时扫描未完成", state="error", expanded=True)
                st.error(f"实时扫描失败：{e}")
                st.caption("现在所有实时请求都设置了超时，不会再无限卡住。接口忙时等十几秒后重试即可。")

        meta = st.session_state.live_meta
        if meta:
            c1, c2, c3 = st.columns(3)
            c1.metric("上涨/下跌", f"{meta.get('up',0)}/{meta.get('down',0)}")
            c2.metric("涨停/跌停", f"{meta.get('limit_up',0)}/{meta.get('limit_down',0)}")
            c3.metric("上涨占比", f"{meta.get('up_ratio',0)*100:.1f}%")
            st.caption(
                f"扫描时间：{meta.get('scan_time','')} ｜ 平均涨跌：{meta.get('avg_pct_change',0):+.2f}% ｜ "
                f"深度数据成功 {meta.get('deep_loaded',0)} 只 / 失败 {meta.get('failed',0)} 只"
            )

            snapshot = st.session_state.live_snapshot
            if snapshot is not None and len(snapshot):
                st.markdown("#### 🔥 实时涨幅榜 Top 15")
                movers = snapshot.copy()
                main_mask = movers["is_mainboard"].fillna(False).map(bool).astype(bool)
                st_mask = movers["is_st"].fillna(False).map(bool).astype(bool)
                movers = movers.loc[main_mask & st_mask.eq(False)].copy()
                movers = movers.sort_values("pct_change", ascending=False).head(15)
                show_cols = [c for c in ["symbol","name","close","pct_change","turnover_rate","live_volume_ratio","amount"] if c in movers.columns]
                movers_show = movers[show_cols].copy().rename(columns={
                    "symbol":"代码","name":"名称","close":"现价","pct_change":"涨幅%",
                    "turnover_rate":"换手%","live_volume_ratio":"量比","amount":"成交额"
                })
                if "成交额" in movers_show.columns:
                    movers_show["成交额"] = pd.to_numeric(movers_show["成交额"], errors="coerce") / 1e8
                    movers_show = movers_show.rename(columns={"成交额":"成交额(亿)"})
                st.dataframe(movers_show, use_container_width=True, hide_index=True)

            selected = st.session_state.live_selected
            if selected is not None and len(selected):
                st.markdown("#### 🎯 量化预筛 Top 15")
                cand = selected.head(15).copy()
                show_cols = [c for c in ["symbol","name","close","pct_change","turnover_rate","live_volume_ratio","live_pre_score"] if c in cand.columns]
                cand_show = cand[show_cols].copy().rename(columns={
                    "symbol":"代码","name":"名称","close":"现价","pct_change":"涨幅%",
                    "turnover_rate":"换手%","live_volume_ratio":"量比","live_pre_score":"预筛分"
                })
                st.dataframe(cand_show, use_container_width=True, hide_index=True)

            st.caption("实时数据来自公开网页行情快照，不是券商逐笔；盘中点击按钮即可重新刷新。")

    if st.session_state.scores is None:
        st.info("点击上面的“获取实时行情并自动量化”，或者到“数据”页上传CSV/载入演示数据。")
    else:
        scores = st.session_state.scores.copy()
        latest = scores["date"].max() if "date" in scores.columns and len(scores) else ""
        st.caption(f"评分日期：{latest}")
        filt = st.segmented_control("范围", ["全部", "首仓候选", "观察", "接力候选"], default="全部")
        view = scores
        if filt == "首仓候选":
            view = scores[scores["signal"] == "BUY_CANDIDATE"]
        elif filt == "观察":
            view = scores[scores["signal"] == "WATCH"]
        elif filt == "接力候选":
            view = scores[scores["signal"] == "RELAY_CANDIDATE"]
        if view.empty:
            st.warning("当前筛选没有股票。弱市里空结果是正常结果，不强行凑票。")
        else:
            topn = st.slider("显示前N只", 3, 20, 8)
            for _, r in view.head(topn).iterrows():
                candidate_card(r)
        out = scores.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载全部评分结果", out, "scores.csv", "text/csv")

with tab2:
    st.subheader("查一只股票")
    if st.session_state.scores is None:
        st.info("先运行一次全市场评分。")
    else:
        q = st.text_input("输入股票代码或名称", placeholder="例如 600584 / 长电科技")
        if q:
            s = st.session_state.scores.copy()
            mask = s["symbol"].astype(str).str.contains(q, na=False)
            if "name" in s.columns:
                mask = mask | s["name"].astype(str).str.contains(q, na=False)
            hit = s[mask]
            if hit.empty:
                st.warning("当前评分数据里没找到。")
            else:
                r = hit.iloc[0]
                candidate_card(r)
                st.markdown("**执行提示**")
                score = float(r.get("combined_score", 0))
                if score >= cfg["strategy"]["score_buy_threshold"]:
                    st.success("评分已取得买入资格；按卡片里的“回踩买点”或“突破确认”触发，未触发就不追。")
                elif score >= cfg["strategy"]["score_watch_threshold"]:
                    st.warning("观察区：不抢，等待价格和量能触发。")
                else:
                    st.error("当前不主动开仓。")

st.divider()
st.caption("研究与决策辅助工具，不保证收益。AI概率来自历史样本，仓位和止损优先于预测。")
