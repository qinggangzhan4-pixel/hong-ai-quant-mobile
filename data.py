from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_market_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"symbol": str})


def load_holdings_csv(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"symbol": str})


def fetch_akshare_history(symbols: list[str], start: str, end: str, output_path: str) -> pd.DataFrame:
    """通过 AkShare 下载前复权日线。网络/API异常时抛出清晰错误。
    仅保证 OHLCV + amount/turnover_rate；行业/资金/基本面建议后续单独补充。
    """
    try:
        import akshare as ak
    except ImportError as e:
        raise RuntimeError("未安装 akshare，请先 pip install -r requirements.txt") from e
    rows = []
    for s in symbols:
        raw = ak.stock_zh_a_hist(symbol=str(s).zfill(6), period="daily", start_date=start, end_date=end, adjust="qfq")
        if raw is None or raw.empty:
            continue
        mapping = {
            "日期":"date", "股票代码":"symbol", "开盘":"open", "收盘":"close", "最高":"high", "最低":"low",
            "成交量":"volume", "成交额":"amount", "换手率":"turnover_rate", "涨跌幅":"pct_change"
        }
        raw = raw.rename(columns={k:v for k,v in mapping.items() if k in raw.columns})
        raw["symbol"] = str(s).zfill(6)
        keep = [c for c in ["date","symbol","open","high","low","close","volume","amount","turnover_rate","pct_change"] if c in raw.columns]
        rows.append(raw[keep])
    if not rows:
        raise RuntimeError("未下载到数据，请检查代码、日期和网络。")
    df = pd.concat(rows, ignore_index=True)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df
