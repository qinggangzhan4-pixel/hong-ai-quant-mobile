from __future__ import annotations
import pandas as pd


def top_report(scores: pd.DataFrame, n: int = 10) -> str:
    if scores.empty:
        return "没有符合条件的数据。"
    lines = []
    for _, r in scores.head(n).iterrows():
        p = "" if pd.isna(r.get("ml_probability", None)) else f" ML={r['ml_probability']:.1%}"
        lines.append(
            f"{r['symbol']} {r.get('name','')} | 综合={r['combined_score']:.1f} 规则={r['rule_score']:.1f}{p} | "
            f"信号={r['signal']} | 动作={r['action']} | 首仓={int(r['suggested_shares'])}股 | 止损≈{r['stop_price']:.2f}"
        )
    return "\n".join(lines)
