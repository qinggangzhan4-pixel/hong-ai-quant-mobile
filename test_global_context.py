import pandas as pd

from global_context import (
    classify_market_sector,
    consumer_fundamental_score,
    score_global_market,
)


def test_consumer_sector_classification():
    assert classify_market_sector("某白酒公司", "食品饮料") == "消费/食品饮料"
    assert classify_market_sector("某家电公司", "家用电器") == "消费/家电家居"
    assert classify_market_sector("某银行", "银行") == "金融"


def test_consumer_quality_requires_real_fields():
    row = {
        "name": "某食品公司",
        "industry": "食品饮料",
        "roe": 12.0,
        "net_profit_yoy": 15.0,
    }
    assert consumer_fundamental_score(row) is not None
    assert consumer_fundamental_score({"name": "某食品公司", "industry": "食品饮料"}) is None


def test_global_market_score_stays_within_ten_points():
    snapshot = pd.DataFrame([
        {"group": "us_broad", "pct_change": 1.0},
        {"group": "us_tech", "pct_change": -2.0},
        {"group": "asia", "pct_change": 0.5},
    ])
    score, breakdown = score_global_market(snapshot, news_adjustment=0.0)
    assert 0.0 <= score <= 10.0
    assert breakdown["news_macro"]["score"] == 1.5
    assert breakdown["total"] == score
