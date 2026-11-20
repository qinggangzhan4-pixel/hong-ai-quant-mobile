from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).rolling(period).mean()
    loss = (-diff.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(group: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = group["close"].shift(1)
    tr = pd.concat([
        group["high"] - group["low"],
        (group["high"] - prev_close).abs(),
        (group["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def validate_market_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["symbol"] = out["symbol"].astype(str).str.zfill(6)
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=REQUIRED_COLUMNS).sort_values(["symbol", "date"]).reset_index(drop=True)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    x = validate_market_data(df)
    chunks = []
    for symbol, g in x.groupby("symbol", sort=False):
        g = g.copy().sort_values("date")
        g["prev_close"] = g["close"].shift(1)
        for n in [1, 3, 4, 5, 10, 20]:
            g[f"ret_{n}"] = g["close"].pct_change(n)
        for n in [5, 10, 20, 60]:
            g[f"ma{n}"] = g["close"].rolling(n).mean()
            g[f"ma{n}_dist"] = g["close"] / g[f"ma{n}"] - 1
        g["ma5_slope3"] = g["ma5"] / g["ma5"].shift(3) - 1
        g["ma10_slope3"] = g["ma10"] / g["ma10"].shift(3) - 1
        g["vol_ma5"] = g["volume"].rolling(5).mean()
        g["vol_ratio5"] = g["volume"] / g["vol_ma5"].replace(0, np.nan)
        # 盘中实时扫描时，用东方财富快照提供的量比覆盖“当日部分成交量/5日完整日均量”的失真值。
        if "live_volume_ratio" in g.columns:
            live_vr = pd.to_numeric(g["live_volume_ratio"], errors="coerce")
            g.loc[live_vr.notna(), "vol_ratio5"] = live_vr[live_vr.notna()]
        g["volatility10"] = g["close"].pct_change().rolling(10).std()
        g["atr14"] = _atr(g, 14)
        g["atr14_pct"] = g["atr14"] / g["close"]
        g["rsi14"] = _rsi(g["close"], 14)
        prior_high20 = g["high"].shift(1).rolling(20).max()
        rolling_high20 = g["high"].rolling(20).max()
        rolling_low60 = g["low"].rolling(60).min()
        rolling_high60 = g["high"].rolling(60).max()
        g["breakout20"] = g["close"] / prior_high20 - 1
        g["drawdown20"] = g["close"] / rolling_high20 - 1
        denom = (rolling_high60 - rolling_low60).replace(0, np.nan)
        g["price_position60"] = (g["close"] - rolling_low60) / denom
        g["gap"] = g["open"] / g["prev_close"] - 1
        g["intraday_ret"] = g["close"] / g["open"] - 1
        if "turnover_rate" not in g.columns:
            g["turnover_rate"] = np.nan
        if "amount" not in g.columns:
            g["amount"] = np.nan
        if "main_net_inflow" not in g.columns:
            g["main_net_inflow"] = np.nan
        if "market_cap" not in g.columns:
            g["market_cap"] = np.nan
        if "pe_ttm" not in g.columns:
            g["pe_ttm"] = np.nan
        if "pb" not in g.columns:
            g["pb"] = np.nan
        if "roe" not in g.columns:
            g["roe"] = np.nan
        if "net_profit_yoy" not in g.columns:
            g["net_profit_yoy"] = np.nan
        if "is_st" not in g.columns:
            g["is_st"] = False
        if "is_limit_up" not in g.columns:
            # 默认按主板约10%涨停近似，仅用于分类，不替代交易所精确涨停价
            g["is_limit_up"] = g["ret_1"] >= 0.095
        chunks.append(g)

    feat = pd.concat(chunks, ignore_index=True)

    # 市场宽度：资料库“市场宽度”模块的核心思想——Close > MA20 的股票占比
    feat["above_ma20"] = feat["close"] > feat["ma20"]
    breadth = feat.groupby("date")["above_ma20"].mean().rename("market_breadth_ma20")
    market_ret = feat.groupby("date")["ret_1"].mean().rename("market_ret_1")
    feat = feat.merge(breadth, on="date", how="left").merge(market_ret, on="date", how="left")

    market_daily = feat[["date", "market_ret_1"]].drop_duplicates().sort_values("date")
    market_daily["market_ret_5"] = (1 + market_daily["market_ret_1"].fillna(0)).rolling(5).apply(np.prod, raw=True) - 1
    market_daily["market_ret_10"] = (1 + market_daily["market_ret_1"].fillna(0)).rolling(10).apply(np.prod, raw=True) - 1
    feat = feat.merge(market_daily[["date", "market_ret_5", "market_ret_10"]], on="date", how="left")

    # 行业相对强弱：存在 industry 列时计算
    if "industry" in feat.columns:
        ind = feat.groupby(["date", "industry"], dropna=False)["ret_1"].mean().rename("industry_ret_1").reset_index()
        feat = feat.merge(ind, on=["date", "industry"], how="left")
        ind_key = ind.copy().sort_values(["industry", "date"])
        ind_key["industry_ret_5"] = ind_key.groupby("industry")["industry_ret_1"].transform(
            lambda s: (1 + s.fillna(0)).rolling(5).apply(np.prod, raw=True) - 1
        )
        ind_key["industry_ret_10"] = ind_key.groupby("industry")["industry_ret_1"].transform(
            lambda s: (1 + s.fillna(0)).rolling(10).apply(np.prod, raw=True) - 1
        )
        feat = feat.merge(ind_key[["date", "industry", "industry_ret_5", "industry_ret_10"]],
                          on=["date", "industry"], how="left")
        ind_breadth = feat.groupby(["date", "industry"], dropna=False)["above_ma20"].mean().rename("industry_breadth_ma20").reset_index()
        feat = feat.merge(ind_breadth, on=["date", "industry"], how="left")
    else:
        feat["industry_ret_1"] = np.nan
        feat["industry_ret_5"] = np.nan
        feat["industry_ret_10"] = np.nan
        feat["industry_breadth_ma20"] = np.nan

    return feat.sort_values(["date", "symbol"]).reset_index(drop=True)


MODEL_FEATURES = [
    "ret_1", "ret_3", "ret_5", "ret_10",
    "ma5_dist", "ma10_dist", "ma20_dist", "ma60_dist",
    "ma5_slope3", "ma10_slope3", "vol_ratio5", "volatility10",
    "atr14_pct", "rsi14", "breakout20", "drawdown20", "price_position60",
    "gap", "intraday_ret", "market_breadth_ma20", "market_ret_5", "market_ret_10",
    "industry_ret_1", "industry_ret_5", "industry_ret_10", "industry_breadth_ma20",
    "turnover_rate", "main_net_inflow", "market_cap", "pe_ttm", "pb", "roe", "net_profit_yoy"
]
