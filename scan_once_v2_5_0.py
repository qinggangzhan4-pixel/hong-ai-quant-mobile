from __future__ import annotations

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import sys
import types
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
pkg = types.ModuleType("ai_quant")
pkg.__path__ = [str(ROOT)]
pkg.__package__ = "ai_quant"
sys.modules["ai_quant"] = pkg

from ai_quant.config import load_config
from ai_quant.data import (
    fetch_live_a_scan,
    fetch_watchlist_fundamentals_and_flow,
    add_watchlist_industry_context,
    score_live_industry_context,
)
from ai_quant.engine import score_universe
from ai_quant.global_context import (
    fetch_global_market_snapshot,
    fetch_global_news,
    score_global_market,
)

def live_market_score(meta: dict) -> float:
    if not meta:
        return 10.0
    breadth = float(meta.get("up_ratio", 0.5))
    if breadth >= 0.65:
        score = 12
    elif breadth >= 0.55:
        score = 10
    elif breadth >= 0.45:
        score = 8
    elif breadth >= 0.35:
        score = 5
    else:
        score = 2
    score += 4 if float(meta.get("avg_pct_change", 0)) > 0 else 1
    limit_up = int(meta.get("limit_up", 0))
    limit_down = int(meta.get("limit_down", 0))
    if limit_up > limit_down * 1.2:
        score += 4
    elif limit_down > limit_up * 1.5:
        score += 1
    else:
        score += 2
    return float(min(score, 20))


def num(value):
    try:
        if pd.isna(value):
            return np.nan
        return float(value)
    except Exception:
        return np.nan


def trade_levels(row, cfg):
    close = num(row.get("close"))
    score = num(row.get("combined_score", row.get("rule_score")))
    ma5 = num(row.get("ma5"))
    ma10 = num(row.get("ma10"))
    prior_high20 = num(row.get("prior_high20"))
    stop = num(row.get("stop_price"))
    volume_ratio = num(row.get("vol_ratio5"))
    rsi = num(row.get("rsi14"))
    turnover = num(row.get("turnover_rate"))
    gap = num(row.get("gap"))
    ret1 = num(row.get("ret_1"))

    support = ma5
    if pd.isna(support) or support <= 0:
        support = ma10
    if pd.isna(support) or support <= 0:
        support = close * 0.99 if not pd.isna(close) else np.nan

    buy_low = support * 0.992 if not pd.isna(support) else np.nan
    buy_high = support * 1.008 if not pd.isna(support) else np.nan
    breakout = prior_high20 * 1.002 if not pd.isna(prior_high20) and prior_high20 > 0 else np.nan

    in_pullback = (
        not pd.isna(close) and not pd.isna(buy_low) and not pd.isna(buy_high)
        and buy_low <= close <= buy_high
    )
    breakout_ok = (
        not pd.isna(close) and not pd.isna(breakout)
        and close >= breakout and (pd.isna(volume_ratio) or volume_ratio >= 1.2)
    )
    entry_ref = close if (in_pullback or breakout_ok) else buy_high
    if pd.isna(entry_ref) or entry_ref <= 0:
        entry_ref = close

    tp1 = entry_ref * (1 + float(cfg["strategy"]["first_take_profit_pct"])) if not pd.isna(entry_ref) else np.nan
    tp2 = entry_ref * (1 + float(cfg["strategy"]["second_take_profit_pct"])) if not pd.isna(entry_ref) else np.nan

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
        status, reason = "数据不足", "先补全评分数据"
    elif score < float(cfg["strategy"]["score_watch_threshold"]):
        status, reason = "不买", "综合分低于观察线"
    elif score < float(cfg["strategy"]["score_buy_threshold"]):
        status, reason = "观察", "分数未到75，等评分和价格同时转强"
    elif not pd.isna(stop) and not pd.isna(close) and close <= stop:
        status, reason = "破位", "已到系统止损附近，禁止新开仓"
    elif overheat:
        status, reason = "过热勿追", "RSI/换手/高开/单日涨幅至少一项过热"
    elif in_pullback:
        status, reason = "回踩买点", "进入MA5附近回踩区，可按首仓纪律评估"
    elif breakout_ok:
        status, reason = "突破买点", "突破20日高点且量能满足确认条件"
    elif not pd.isna(close) and not pd.isna(buy_low) and close < buy_low:
        status, reason = "等重新站回", "回踩偏深，先等重新站回买点区"
    else:
        status, reason = "等待触发", "不追价，等回踩区或放量突破"

    return {
        "status": status,
        "reason": reason,
        "pullback_low": buy_low,
        "pullback_high": buy_high,
        "breakout_confirm": breakout,
        "stop": stop,
        "take_profit_1": tp1,
        "take_profit_2": tp2,
    }


def main():
    cfg = load_config(str(ROOT / "config.yaml"))
    scan_started = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")

    # Follow the v2.5.0 app's default scan setting: full-A snapshot,
    # 30 sector-stratified main-board candidates, approximately 150 calendar days.
    market, snapshot, selected, market_meta = fetch_live_a_scan(
        preselect_n=30,
        history_calendar_days=150,
        max_workers=4,
    )
    symbols = selected["symbol"].astype(str).str.zfill(6).unique().tolist()

    # Enrich the deep-scan candidates with the v2.5.0 consumer financial and
    # daily main-flow fields. Missing values remain missing; no imputation.
    quality_flow = fetch_watchlist_fundamentals_and_flow(symbols, max_workers=4)
    if quality_flow is not None and not quality_flow.empty:
        quality_flow["symbol"] = quality_flow["symbol"].astype(str).str.zfill(6)
        market["symbol"] = market["symbol"].astype(str).str.zfill(6)
        market = market.merge(quality_flow, on="symbol", how="left")

    market = add_watchlist_industry_context(market, snapshot, symbols)

    # No personal holdings are loaded. No broker connection or order placement.
    # No unvalidated ML model is loaded; the rules-based 100-point score remains active.
    scores, _ = score_universe(market, cfg, holdings=None, model_path=None)

    # Use full-A live breadth and limit-up/down counts, matching the mobile app.
    local_score = live_market_score(market_meta)
    old_market = pd.to_numeric(scores["market_score"], errors="coerce").fillna(0)
    scores["market_score"] = local_score
    scores["rule_score"] = (
        pd.to_numeric(scores["rule_score"], errors="coerce").fillna(0)
        - old_market + local_score
    )

    # Replace the candidate-only industry estimate with the actual full-A
    # industry cross-section from the same live snapshot.
    old_industry = pd.to_numeric(scores["industry_score"], errors="coerce").fillna(10)
    scores = score_live_industry_context(scores, snapshot)
    new_industry = pd.to_numeric(scores["industry_score"], errors="coerce").fillna(10)
    scores["rule_score"] = (
        pd.to_numeric(scores["rule_score"], errors="coerce").fillna(0)
        - old_industry + new_industry
    ).round(2)

    # Global context is scored with the v2.5.0 2/3/2/3 weighting.
    # News headlines remain evidence for manual confirmation; no automatic
    # bullish/bearish news adjustment is applied.
    global_errors = []
    try:
        global_snapshot = fetch_global_market_snapshot()
    except Exception as exc:
        global_snapshot = pd.DataFrame()
        global_errors.append("global_market: " + str(exc))
    try:
        news = fetch_global_news()
        news_errors = news.attrs.get("failures", [])
    except Exception as exc:
        news = pd.DataFrame()
        news_errors = [str(exc)]

    global_breakdown = {}
    if global_snapshot is not None and len(global_snapshot):
        global_score, global_breakdown = score_global_market(global_snapshot, news_adjustment=0.0)
        old_market = pd.to_numeric(scores["market_score"], errors="coerce").fillna(10.0)
        blended_market = old_market * 0.5 + float(global_score)
        scores["market_score"] = blended_market.round(2)
        scores["rule_score"] = (
            pd.to_numeric(scores["rule_score"], errors="coerce").fillna(0)
            - old_market + blended_market
        ).round(2)
    else:
        global_score = None

    # Rebuild the final score and action labels using the unchanged v2.5.0 weights/thresholds.
    ml = pd.to_numeric(scores.get("ml_probability", np.nan), errors="coerce")
    scores["combined_score"] = pd.to_numeric(scores["rule_score"], errors="coerce").fillna(0)
    if isinstance(ml, pd.Series):
        use_ml = ml.notna()
        if use_ml.any():
            rw = float(cfg["strategy"]["rule_weight"])
            mw = float(cfg["strategy"]["model_weight"])
            scores.loc[use_ml, "combined_score"] = (
                rw * scores.loc[use_ml, "rule_score"] + mw * (ml[use_ml] * 100)
            )
    buy_threshold = float(cfg["strategy"]["score_buy_threshold"])
    watch_threshold = float(cfg["strategy"]["score_watch_threshold"])
    scores["signal"] = np.where(
        scores["combined_score"] >= buy_threshold,
        "BUY_CANDIDATE",
        np.where(scores["combined_score"] >= watch_threshold, "WATCH", "AVOID"),
    )
    if "relay_status" in scores.columns and "relay_score" in scores.columns:
        applicable = scores["relay_status"].astype(str) != "not_applicable"
        scores.loc[applicable & (scores["relay_score"] < 50), "signal"] = "AVOID_RELAY"
        scores.loc[applicable & (scores["relay_score"] >= 70), "signal"] = "RELAY_CANDIDATE"

    scores = scores.sort_values(["combined_score", "relay_score"], ascending=False).reset_index(drop=True)

    top = scores.head(20).copy()
    level_rows = []
    for _, row in top.iterrows():
        levels = trade_levels(row, cfg)
        record = {
            "date": str(row.get("date", "")),
            "symbol": str(row.get("symbol", "")).zfill(6),
            "name": row.get("name", ""),
            "industry": row.get("industry", ""),
            "sector_category": row.get("sector_category", ""),
            "close": num(row.get("close")),
            "combined_score": num(row.get("combined_score")),
            "signal": row.get("signal", ""),
            "market_score": num(row.get("market_score")),
            "industry_score": num(row.get("industry_score")),
            "quality_score": num(row.get("quality_score")),
            "trend_score": num(row.get("trend_score")),
            "execution_score": num(row.get("execution_score")),
            "rule_score": num(row.get("rule_score")),
            "ml_probability": num(row.get("ml_probability")),
            "relay_score": num(row.get("relay_score")),
            "vol_ratio5": num(row.get("vol_ratio5")),
            "rsi14": num(row.get("rsi14")),
            "roe": num(row.get("roe")),
            "net_profit_yoy": num(row.get("net_profit_yoy")),
            "main_net_inflow": num(row.get("main_net_inflow")),
            "financial_report_date": row.get("financial_report_date", ""),
            "flow_date": row.get("flow_date", ""),
        }
        record.update({k: (None if pd.isna(v) else float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v) for k, v in levels.items()})
        level_rows.append(record)

    top_df = pd.DataFrame(level_rows)
    top_df.to_csv(ROOT / "v25_scan_top20.csv", index=False, encoding="utf-8-sig")
    scores.to_csv(ROOT / "v25_scan_deep_scores.csv", index=False, encoding="utf-8-sig")

    consumer_fields = [
        "roe", "net_profit_yoy", "revenue_yoy", "gross_margin",
        "inventory_turnover", "operating_cashflow_to_profit",
    ]
    consumer_coverage = 0
    if quality_flow is not None and not quality_flow.empty:
        available = [c for c in consumer_fields if c in quality_flow.columns]
        if available:
            consumer_coverage = int((quality_flow[available].notna().sum(axis=1) >= 2).sum())
    flow_coverage = (
        int(pd.to_numeric(quality_flow.get("main_net_inflow"), errors="coerce").notna().sum())
        if quality_flow is not None and not quality_flow.empty and "main_net_inflow" in quality_flow
        else 0
    )

    sector_counts = (
        scores["sector_category"].fillna("其他").value_counts().to_dict()
        if "sector_category" in scores.columns else {}
    )
    news_records = []
    if news is not None and not news.empty:
        news_records = news.head(20).where(pd.notna(news), None).to_dict("records")
    global_records = []
    if global_snapshot is not None and not global_snapshot.empty:
        global_records = global_snapshot.where(pd.notna(global_snapshot), None).to_dict("records")

    result = {
        "strategy_version": "v2.5.0",
        "scan_started_cn": scan_started,
        "scan_finished_cn": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "universe": "沪深主板；排除创业板、科创板、北交所、ST；价格2-300元；成交额至少2000万元",
        "full_a_snapshot_count": int(len(snapshot)),
        "source_total_approx": int(snapshot.attrs.get("source_total", len(snapshot))),
        "snapshot_pages": int(snapshot.attrs.get("total_pages", 1)),
        "deep_preselected": int(len(selected)),
        "deep_history_loaded": int(market_meta.get("deep_loaded", len(symbols))),
        "deep_history_failed": int(market_meta.get("failed", 0)),
        "deep_history_failed_symbols": market_meta.get("failed_symbols", []),
        "history_window_calendar_days": 150,
        "sector_distribution_in_deep_scan": sector_counts,
        "local_market_context": market_meta,
        "local_market_score": local_score,
        "global_market_score_out_of_10": global_score,
        "global_market_breakdown": global_breakdown,
        "global_market_snapshot": global_records,
        "global_market_errors": global_errors,
        "news_errors": news_errors,
        "news_headlines": news_records,
        "consumer_financial_coverage_at_least_2_fields": consumer_coverage,
        "main_flow_coverage": flow_coverage,
        "ai_layer": "disabled for this run: no validated model path supplied",
        "holdings_loaded": False,
        "broker_orders_placed": False,
        "top20": level_rows,
    }
    with open(ROOT / "v25_scan_meta.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print("V2.5.0_SCAN_RESULT_BEGIN")
    print(json.dumps({
        "strategy_version": result["strategy_version"],
        "scan_started_cn": result["scan_started_cn"],
        "scan_finished_cn": result["scan_finished_cn"],
        "full_a_snapshot_count": result["full_a_snapshot_count"],
        "source_total_approx": result["source_total_approx"],
        "deep_preselected": result["deep_preselected"],
        "deep_history_loaded": result["deep_history_loaded"],
        "deep_history_failed": result["deep_history_failed"],
        "sector_distribution_in_deep_scan": sector_counts,
        "local_market_context": market_meta,
        "local_market_score": local_score,
        "global_market_score_out_of_10": global_score,
        "global_market_breakdown": global_breakdown,
        "consumer_financial_coverage": consumer_coverage,
        "main_flow_coverage": flow_coverage,
        "ai_layer": result["ai_layer"],
        "news_headlines": news_records[:12],
        "top20": level_rows,
    }, ensure_ascii=False, indent=2, default=str))
    print("V2.5.0_SCAN_RESULT_END")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
