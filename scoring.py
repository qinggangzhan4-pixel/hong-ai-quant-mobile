from __future__ import annotations
import math
import numpy as np
import pandas as pd
from .global_context import classify_market_sector, consumer_fundamental_score


def _num(row, key, default=np.nan):
    v = row.get(key, default)
    try:
        if pd.isna(v):
            return default
        return float(v)
    except Exception:
        return default


def market_score(row) -> float:
    score = 0.0
    b = _num(row, "market_breadth_ma20", 0.5)
    if b >= 0.65: score += 12
    elif b >= 0.55: score += 10
    elif b >= 0.45: score += 8
    elif b >= 0.35: score += 5
    else: score += 2
    r5 = _num(row, "market_ret_5", 0)
    r10 = _num(row, "market_ret_10", 0)
    score += 4 if r5 > 0 else 1
    score += 4 if r10 > 0 else 1
    return min(score, 20)


def industry_score(row) -> float:
    # 若无行业数据，给中性10分，避免因数据缺失误杀
    vals = [_num(row, "industry_ret_1"), _num(row, "industry_ret_5"), _num(row, "industry_ret_10")]
    if all(np.isnan(v) for v in vals):
        return 10.0
    score = 0.0
    r1, r5, r10 = [0 if np.isnan(v) else v for v in vals]
    score += 4 if r1 > 0 else 1
    score += 6 if r5 > 0 else 1
    score += 6 if r10 > 0 else 1
    b = _num(row, "industry_breadth_ma20", 0.5)
    if b >= 0.6: score += 4
    elif b >= 0.45: score += 2
    return min(score, 20)


def quality_score(row) -> float:
    if bool(row.get("is_st", False)):
        return 0.0
    score = 10.0
    roe = _num(row, "roe")
    yoy = _num(row, "net_profit_yoy")
    pe = _num(row, "pe_ttm")
    pb = _num(row, "pb")
    if not np.isnan(roe):
        if roe >= 15: score += 4
        elif roe >= 8: score += 2
        elif roe < 0: score -= 4
    if not np.isnan(yoy):
        if yoy >= 20: score += 4
        elif yoy > 0: score += 2
        elif yoy <= -30: score -= 4
    if not np.isnan(pe):
        if 0 < pe <= 60: score += 2
        elif pe > 200 or pe < 0: score -= 2
    if not np.isnan(pb) and pb > 10:
        score -= 1
    score = float(np.clip(score, 0, 20))
    # 消费股使用专属财务质量因子；真实字段不足两项时不调整、不补造数据。
    sector = classify_market_sector(row.get("name"), row.get("industry"))
    if sector.startswith("消费/"):
        consumer_score = consumer_fundamental_score(row)
        if consumer_score is not None:
            score = 0.75 * score + 0.25 * consumer_score * 2
    return float(np.clip(score, 0, 20))


def trend_score(row) -> float:
    score = 0.0
    close = _num(row, "close")
    ma5, ma10, ma20 = _num(row, "ma5"), _num(row, "ma10"), _num(row, "ma20")
    if not np.isnan(close) and not np.isnan(ma5) and close > ma5: score += 5
    if not np.isnan(ma5) and not np.isnan(ma10) and ma5 > ma10: score += 5
    if not np.isnan(ma10) and not np.isnan(ma20) and ma10 > ma20: score += 3
    if _num(row, "ma5_slope3", 0) > 0: score += 3
    if _num(row, "ma10_slope3", 0) > 0: score += 2
    vr = _num(row, "vol_ratio5", 1)
    r1 = _num(row, "ret_1", 0)
    if r1 > 0 and vr >= 1.2: score += 3
    elif r1 < 0 and vr >= 1.5: score -= 3
    bo = _num(row, "breakout20", 0)
    if bo > 0: score += 3
    rsi = _num(row, "rsi14", 50)
    if 55 <= rsi <= 75: score += 1
    elif rsi >= 85: score -= 2
    turnover = _num(row, "turnover_rate")
    if not np.isnan(turnover):
        # 既兼容百分数(20)也兼容小数(0.2)
        t = turnover / 100 if turnover > 1 else turnover
        if t > 0.20: score -= 4
        elif t > 0.16: score -= 3
        elif t > 0.12: score -= 1
    return float(np.clip(score, 0, 25))


def execution_score(row) -> float:
    score = 8.0
    pos = _num(row, "price_position60", 0.5)
    if 0.25 <= pos <= 0.75: score += 3
    elif pos >= 0.9: score -= 3
    gap = _num(row, "gap", 0)
    if -0.02 <= gap <= 0.03: score += 2
    elif gap >= 0.06: score -= 3
    dd = _num(row, "drawdown20", 0)
    if dd > -0.05: score += 1
    flow = _num(row, "main_net_inflow")
    if not np.isnan(flow):
        score += 1 if flow > 0 else -1
    return float(np.clip(score, 0, 15))


def score_row(row) -> dict:
    parts = {
        "market_score": market_score(row),
        "industry_score": industry_score(row),
        "quality_score": quality_score(row),
        "trend_score": trend_score(row),
        "execution_score": execution_score(row),
    }
    parts["rule_score"] = round(sum(parts.values()), 2)
    parts["sector_category"] = classify_market_sector(row.get("name"), row.get("industry"))
    parts["consumer_quality_score"] = consumer_fundamental_score(row)
    return parts
