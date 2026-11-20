import numpy as np
import pandas as pd
from ai_quant.config import load_config
from ai_quant.engine import score_universe


def test_score_smoke():
    dates = pd.bdate_range("2026-01-01", periods=100)
    rows = []
    for s, drift in [("600001", 0.001), ("000002", -0.0002)]:
        price = 10.0
        for i, d in enumerate(dates):
            price *= (1 + drift + 0.002*np.sin(i/5))
            rows.append({"date":d, "symbol":s, "name":s, "industry":"测试", "open":price*0.995, "high":price*1.01, "low":price*0.99, "close":price, "volume":1_000_000+i*1000})
    df = pd.DataFrame(rows)
    cfg = load_config()
    scores, risk = score_universe(df, cfg)
    assert len(scores) == 2
    assert "combined_score" in scores.columns
