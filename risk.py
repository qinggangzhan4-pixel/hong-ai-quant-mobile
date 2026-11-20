from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd

@dataclass
class PositionPlan:
    entry_price: float
    stop_price: float
    stop_pct: float
    suggested_shares: int
    suggested_value: float
    risk_money: float
    max_single_value: float
    industry_room_value: float
    action: str

    def to_dict(self):
        return asdict(self)


def _lot_floor(shares: float) -> int:
    return max(0, int(shares // 100) * 100)


def plan_position(row, config: dict, holdings: pd.DataFrame | None = None) -> PositionPlan:
    acfg = config["account"]
    scfg = config["strategy"]
    equity = float(acfg["equity"])
    entry = float(row["close"])
    atr_pct = row.get("atr14_pct", np.nan)
    if pd.isna(atr_pct) or atr_pct <= 0:
        stop_pct = 0.05
    else:
        stop_pct = float(np.clip(float(atr_pct) * scfg["atr_stop_multiple"], scfg["min_stop_pct"], scfg["max_stop_pct"]))
    stop = entry * (1 - stop_pct)
    risk_money = equity * acfg["risk_per_trade"]
    risk_shares = _lot_floor(risk_money / max(entry - stop, 1e-9))
    max_single_value = equity * acfg["max_single_stock"]
    single_shares = _lot_floor(max_single_value / entry)
    first_value = equity * acfg["first_entry_pct"]
    first_shares = _lot_floor(first_value / entry)
    # A股100股为一手：高价股可能一手就超过“理想首仓6%”。
    # 若一手仍满足单票上限和单笔风险预算，则允许最小一手，而不是机械返回0股。
    if first_shares == 0 and entry * 100 <= max_single_value and risk_shares >= 100:
        first_shares = 100

    industry_room_value = equity * acfg["max_industry_exposure"]
    if holdings is not None and len(holdings) and "industry" in holdings.columns and "market_value" in holdings.columns:
        ind = row.get("industry", None)
        if ind is not None and not pd.isna(ind):
            current = holdings.loc[holdings["industry"] == ind, "market_value"].sum()
            industry_room_value = max(0.0, industry_room_value - float(current))
    industry_shares = _lot_floor(industry_room_value / entry)

    shares = min(v for v in [risk_shares, single_shares, first_shares, industry_shares] if v >= 0)
    score = float(row.get("combined_score", row.get("rule_score", 0)))
    if score >= scfg["score_buy_threshold"] and shares >= 100:
        action = "允许首仓"
    elif score >= scfg["score_watch_threshold"]:
        action = "观察，等待触发"
        shares = 0
    else:
        action = "不买/减风险"
        shares = 0
    return PositionPlan(entry, stop, stop_pct, shares, shares * entry, risk_money, max_single_value, industry_room_value, action)


def portfolio_risk(holdings: pd.DataFrame, equity: float) -> dict:
    if holdings is None or holdings.empty:
        return {"total_exposure": 0, "max_industry_exposure": 0, "max_single_exposure": 0, "warnings": []}
    h = holdings.copy()
    if "market_value" not in h.columns:
        raise ValueError("holdings 需要 market_value 字段")
    h["exposure"] = h["market_value"] / equity
    total = h["market_value"].sum() / equity
    single = h["exposure"].max()
    max_ind = 0.0
    if "industry" in h.columns:
        max_ind = (h.groupby("industry")["market_value"].sum() / equity).max()
    warnings = []
    if total > 0.80: warnings.append("总仓位>80%，容错空间低")
    if single > 0.20: warnings.append("存在单票>20%的集中风险")
    if max_ind > 0.30: warnings.append("存在单行业>30%的同向暴露")
    return {"total_exposure": float(total), "max_industry_exposure": float(max_ind), "max_single_exposure": float(single), "warnings": warnings}
