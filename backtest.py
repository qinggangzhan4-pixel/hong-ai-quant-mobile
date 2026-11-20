from __future__ import annotations
import numpy as np
import pandas as pd
from .features import add_features
from .scoring import score_row


def backtest_signals(market_df: pd.DataFrame, config: dict, min_score: float | None = None) -> tuple[pd.DataFrame, dict]:
    """简化的样本外友好回测：
    - 用 t 日收盘数据打分
    - t+1 开盘买入
    - A股 T+1：买入当天不允许卖
    - 第2个持有日起检查止损/止盈，最长持有 time_stop_days
    - 同日止盈止损都触发时按更保守的止损先发生处理
    这是“信号质量回测”，不是完整组合资金曲线模拟。
    """
    feat = add_features(market_df)
    scores = feat.apply(score_row, axis=1, result_type="expand")
    feat = pd.concat([feat.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    threshold = min_score or config["strategy"]["score_buy_threshold"]
    hold_days = int(config["strategy"]["time_stop_days"])
    take_pct = float(config["strategy"]["first_take_profit_pct"])
    cost = config["costs"]
    trades = []

    for symbol, g in feat.groupby("symbol", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        for i in range(len(g) - hold_days - 1):
            sig = g.iloc[i]
            if sig["rule_score"] < threshold:
                continue
            entry_i = i + 1
            entry = float(g.iloc[entry_i]["open"]) * (1 + cost["slippage"])
            atr_pct = sig.get("atr14_pct", np.nan)
            stop_pct = 0.05 if pd.isna(atr_pct) else float(np.clip(float(atr_pct) * config["strategy"]["atr_stop_multiple"], config["strategy"]["min_stop_pct"], config["strategy"]["max_stop_pct"]))
            stop = entry * (1 - stop_pct)
            take = entry * (1 + take_pct)
            exit_price = float(g.iloc[entry_i + hold_days]["close"]) * (1 - cost["slippage"])
            exit_date = g.iloc[entry_i + hold_days]["date"]
            reason = "time_stop"
            # 买入当天不能卖，故从 entry_i+1 开始检查
            for j in range(entry_i + 1, min(entry_i + hold_days + 1, len(g))):
                day = g.iloc[j]
                hit_stop = float(day["low"]) <= stop
                hit_take = float(day["high"]) >= take
                if hit_stop:
                    exit_price = stop * (1 - cost["slippage"])
                    exit_date = day["date"]
                    reason = "stop"
                    break
                if hit_take:
                    exit_price = take * (1 - cost["slippage"])
                    exit_date = day["date"]
                    reason = "take_profit"
                    break
            gross = exit_price / entry - 1
            # 近似成本，最低佣金无法按百分比精确模拟，这里按比例保守扣减
            net = gross - 2 * cost["commission"] - cost["stamp_duty_sell"]
            trades.append({
                "symbol": symbol, "signal_date": sig["date"], "entry_date": g.iloc[entry_i]["date"], "exit_date": exit_date,
                "score": sig["rule_score"], "entry": entry, "exit": exit_price, "return": net, "reason": reason
            })
    t = pd.DataFrame(trades)
    if t.empty:
        return t, {"trades": 0}
    wins = t[t["return"] > 0]
    losses = t[t["return"] <= 0]
    stats = {
        "trades": int(len(t)),
        "win_rate": float((t["return"] > 0).mean()),
        "avg_return": float(t["return"].mean()),
        "median_return": float(t["return"].median()),
        "avg_win": float(wins["return"].mean()) if len(wins) else 0.0,
        "avg_loss": float(losses["return"].mean()) if len(losses) else 0.0,
        "payoff_ratio": float(wins["return"].mean() / abs(losses["return"].mean())) if len(wins) and len(losses) and losses["return"].mean() != 0 else np.nan,
    }
    return t, stats
