from __future__ import annotations
import numpy as np
import pandas as pd


def relay_watch(row, config: dict) -> dict:
    """首板/一进二专用模块。

    基于资料库 2026 实盘版“首板竞价三模式”的可迁移阈值，
    但只输出观察/通过，不自动下单。
    """
    rcfg = config["relay"]
    ret1 = float(row.get("ret_1", 0) or 0)
    is_limit = bool(row.get("is_limit_up", False)) or ret1 >= 0.095
    recent4 = float(row.get("ret_4", np.nan)) if not pd.isna(row.get("ret_4", np.nan)) else np.nan
    turnover = row.get("turnover_rate", np.nan)
    if not pd.isna(turnover):
        turnover = float(turnover)
        turnover = turnover / 100 if turnover > 1 else turnover
    score = 0
    reasons = []
    if is_limit:
        score += 25
        reasons.append("当前/昨日具备首板属性")
    else:
        reasons.append("非首板接力场景")
        return {"relay_score": 0, "relay_status": "not_applicable", "relay_reasons": reasons}

    pos60 = row.get("price_position60", np.nan)
    if not pd.isna(pos60) and float(pos60) <= 0.5:
        score += 15; reasons.append("60日位置偏中低")
    amount = row.get("amount", np.nan)
    if not pd.isna(amount) and float(amount) >= 1e8:
        score += 10; reasons.append("成交额达到基础流动性")
    if not pd.isna(turnover):
        if turnover <= 0.12:
            score += 15; reasons.append("换手率健康")
        elif turnover <= rcfg["turnover_redline"]:
            score += 5; reasons.append("换手率进入谨慎区")
        elif turnover >= rcfg["turnover_hard_redline"]:
            score -= 20; reasons.append("换手率超过20%硬红线")
        else:
            score -= 10; reasons.append("换手率超过16%红线")
    if not np.isnan(recent4):
        if recent4 <= rcfg["max_recent_4d_return"]:
            score += 10; reasons.append("近4日涨幅未过热")
        else:
            score -= 10; reasons.append("近4日涨幅过热")
    vr = row.get("auction_volume_ratio", np.nan)
    if not pd.isna(vr):
        if float(vr) >= rcfg["min_auction_volume_ratio"]:
            score += 15; reasons.append("竞价量能>=昨日3%")
        else:
            score -= 10; reasons.append("竞价量能不足")
    gap = row.get("auction_open_ratio", row.get("gap", np.nan))
    if not pd.isna(gap):
        gap = float(gap)
        if rcfg["high_open_min"] <= gap <= rcfg["high_open_max"]:
            score += 10; reasons.append("高开0~6%模式")
        elif rcfg["low_open_min"] <= gap <= rcfg["low_open_max"]:
            score += 10; reasons.append("低开3~4.5%模式")
        elif gap > 0.08:
            score -= 10; reasons.append("竞价高开过猛")
    status = "pass" if score >= 70 else "watch" if score >= 50 else "avoid"
    return {"relay_score": int(np.clip(score, 0, 100)), "relay_status": status, "relay_reasons": reasons}
