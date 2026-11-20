from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
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


def _require_akshare():
    try:
        import akshare as ak
        return ak
    except ImportError as e:
        raise RuntimeError("未安装 akshare，请先 pip install -r requirements.txt") from e


def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _is_mainboard_symbol(symbol: str) -> bool:
    s = str(symbol).zfill(6)
    return s.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def fetch_realtime_a_snapshot() -> pd.DataFrame:
    """抓取 A 股实时快照。

    数据通过 AkShare 的东方财富 A 股现货接口获取。它是网页行情快照，
    不是券商逐笔成交；盘中刷新时会更新，休市时通常是最近收盘快照。
    """
    ak = _require_akshare()
    raw = ak.stock_zh_a_spot_em()
    if raw is None or raw.empty:
        raise RuntimeError("实时 A 股快照为空，请稍后重试。")

    mapping = {
        "代码": "symbol",
        "名称": "name",
        "最新价": "close",
        "涨跌幅": "pct_change",
        "涨跌额": "price_change",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "最高": "high",
        "最低": "low",
        "今开": "open",
        "昨收": "prev_close",
        "量比": "live_volume_ratio",
        "换手率": "turnover_rate",
        "市盈率-动态": "pe_ttm",
        "市净率": "pb",
        "总市值": "market_cap",
        "流通市值": "float_market_cap",
        "涨速": "speed",
        "5分钟涨跌": "change_5m",
        "60日涨跌幅": "change_60d",
        "年初至今涨跌幅": "change_ytd",
    }
    out = raw.rename(columns={k: v for k, v in mapping.items() if k in raw.columns}).copy()
    if "symbol" not in out.columns or "close" not in out.columns:
        raise RuntimeError("实时行情字段发生变化，未找到股票代码或最新价。")

    out["symbol"] = out["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    if "name" not in out.columns:
        out["name"] = ""

    numeric = [
        "close", "pct_change", "price_change", "volume", "amount", "amplitude",
        "high", "low", "open", "prev_close", "live_volume_ratio", "turnover_rate",
        "pe_ttm", "pb", "market_cap", "float_market_cap", "speed", "change_5m",
        "change_60d", "change_ytd",
    ]
    out = _to_num(out, numeric)

    now_cn = datetime.now(ZoneInfo("Asia/Shanghai"))
    out["date"] = pd.Timestamp(now_cn.date())
    out["is_st"] = out["name"].astype(str).str.upper().str.contains("ST|退", regex=True)
    out["is_mainboard"] = out["symbol"].map(_is_mainboard_symbol)
    if "pct_change" in out.columns:
        out["is_limit_up"] = out["pct_change"] >= 9.5
    else:
        out["is_limit_up"] = False

    # 只留下能正常交易/定价的记录。
    out = out[pd.to_numeric(out["close"], errors="coerce").fillna(0) > 0].reset_index(drop=True)
    return out


def fetch_realtime_etf_snapshot() -> pd.DataFrame:
    """抓取 ETF 现货快照；失败时由调用方自行忽略。"""
    ak = _require_akshare()
    raw = ak.fund_etf_spot_em()
    if raw is None or raw.empty:
        return pd.DataFrame()
    mapping = {
        "代码": "symbol",
        "名称": "name",
        "最新价": "close",
        "涨跌幅": "pct_change",
        "成交量": "volume",
        "成交额": "amount",
        "最高": "high",
        "最低": "low",
        "今开": "open",
        "昨收": "prev_close",
        "换手率": "turnover_rate",
    }
    out = raw.rename(columns={k: v for k, v in mapping.items() if k in raw.columns}).copy()
    if "symbol" not in out.columns or "close" not in out.columns:
        return pd.DataFrame()
    out["symbol"] = out["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    out = _to_num(out, ["close","pct_change","volume","amount","high","low","open","prev_close","turnover_rate"])
    return out


def live_market_context(snapshot: pd.DataFrame) -> dict:
    x = snapshot.copy()
    if "pct_change" not in x.columns:
        return {}
    x["pct_change"] = pd.to_numeric(x["pct_change"], errors="coerce")
    x = x[x["pct_change"].notna()]
    if x.empty:
        return {}
    up = int((x["pct_change"] > 0).sum())
    down = int((x["pct_change"] < 0).sum())
    flat = int((x["pct_change"] == 0).sum())
    limit_up = int((x["pct_change"] >= 9.5).sum())
    limit_down = int((x["pct_change"] <= -9.5).sum())
    amount = float(pd.to_numeric(x.get("amount", 0), errors="coerce").fillna(0).sum()) if "amount" in x.columns else 0.0
    return {
        "count": int(len(x)),
        "up": up,
        "down": down,
        "flat": flat,
        "up_ratio": float(up / len(x)),
        "avg_pct_change": float(x["pct_change"].mean()),
        "median_pct_change": float(x["pct_change"].median()),
        "limit_up": limit_up,
        "limit_down": limit_down,
        "amount": amount,
    }


def preselect_live_candidates(snapshot: pd.DataFrame, limit: int = 30) -> pd.DataFrame:
    """第一阶段：用全市场实时快照做快速预筛，再对少量候选拉历史做深度评分。"""
    x = snapshot.copy()
    x = x[x["is_mainboard"].fillna(False)]
    x = x[~x["is_st"].fillna(False)]
    x = x[(x["close"] >= 2.0) & (x["close"] <= 300.0)]

    if "amount" in x.columns:
        # 太冷门的股票实时信号噪声大；阈值较低，避免误杀。
        x = x[pd.to_numeric(x["amount"], errors="coerce").fillna(0) >= 20_000_000]

    if x.empty:
        return x

    amt = pd.to_numeric(x.get("amount", 0), errors="coerce").fillna(0)
    chg = pd.to_numeric(x.get("pct_change", 0), errors="coerce").fillna(0)
    turn = pd.to_numeric(x.get("turnover_rate", 0), errors="coerce").fillna(0)
    vr = pd.to_numeric(x.get("live_volume_ratio", 1), errors="coerce").fillna(1)

    amount_score = amt.rank(pct=True).fillna(0)

    # 短线更偏好“上涨但不过热”；涨停票仍保留给接力模块，不直接剔除。
    chg_score = np.where(
        (chg >= 0) & (chg <= 6), 1.0,
        np.where((chg > 6) & (chg < 9.5), 0.78,
        np.where((chg >= -2.5) & (chg < 0), 0.58,
        np.where(chg >= 9.5, 0.68, 0.20)))
    )

    turn_score = np.where(
        (turn >= 1) & (turn <= 12), 1.0,
        np.where((turn > 12) & (turn <= 16), 0.65,
        np.where((turn > 16) & (turn <= 20), 0.35,
        np.where(turn > 20, 0.10, 0.55)))
    )
    vr_score = np.clip((vr - 0.7) / 1.8, 0, 1)

    x["live_pre_score"] = (
        amount_score * 40
        + chg_score * 25
        + turn_score * 20
        + vr_score * 15
    )
    return x.sort_values(["live_pre_score", "amount"], ascending=False).head(int(limit)).reset_index(drop=True)


def _fetch_one_history(symbol: str, name: str, snapshot_row: dict, start: str, end: str) -> pd.DataFrame | None:
    ak = _require_akshare()
    raw = ak.stock_zh_a_hist(
        symbol=str(symbol).zfill(6),
        period="daily",
        start_date=start,
        end_date=end,
        adjust="",
    )
    if raw is None or raw.empty:
        return None

    mapping = {
        "日期": "date",
        "股票代码": "symbol",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover_rate",
        "涨跌幅": "pct_change",
    }
    hist = raw.rename(columns={k: v for k, v in mapping.items() if k in raw.columns}).copy()
    for c in ["open","high","low","close","volume","amount","turnover_rate","pct_change"]:
        if c in hist.columns:
            hist[c] = pd.to_numeric(hist[c], errors="coerce")
    hist["date"] = pd.to_datetime(hist["date"])
    hist["symbol"] = str(symbol).zfill(6)
    hist["name"] = name

    today = pd.Timestamp(datetime.now(ZoneInfo("Asia/Shanghai")).date())
    hist = hist[hist["date"] < today].copy()

    live = {
        "date": today,
        "symbol": str(symbol).zfill(6),
        "name": name,
        "open": snapshot_row.get("open", np.nan),
        "high": snapshot_row.get("high", np.nan),
        "low": snapshot_row.get("low", np.nan),
        "close": snapshot_row.get("close", np.nan),
        "volume": snapshot_row.get("volume", np.nan),
        "amount": snapshot_row.get("amount", np.nan),
        "turnover_rate": snapshot_row.get("turnover_rate", np.nan),
        "pct_change": snapshot_row.get("pct_change", np.nan),
        "live_volume_ratio": snapshot_row.get("live_volume_ratio", np.nan),
        "pe_ttm": snapshot_row.get("pe_ttm", np.nan),
        "pb": snapshot_row.get("pb", np.nan),
        "market_cap": snapshot_row.get("market_cap", np.nan),
        "is_st": bool(snapshot_row.get("is_st", False)),
        "is_limit_up": bool(snapshot_row.get("is_limit_up", False)),
    }

    # 给历史行补齐实时才有的字段。
    for c in ["live_volume_ratio","pe_ttm","pb","market_cap","is_st","is_limit_up"]:
        if c not in hist.columns:
            hist[c] = np.nan
    hist["is_st"] = hist["is_st"].fillna(False)
    hist["is_limit_up"] = hist["is_limit_up"].fillna(False)

    live_df = pd.DataFrame([live])
    cols = sorted(set(hist.columns).union(live_df.columns))
    hist = hist.reindex(columns=cols)
    live_df = live_df.reindex(columns=cols)
    return pd.concat([hist, live_df], ignore_index=True)


def fetch_live_a_scan(preselect_n: int = 30, history_calendar_days: int = 150, max_workers: int = 4):
    """两阶段实时扫描：
    1) 全 A 股实时快照预筛；
    2) 对排名靠前的主板股票拉约 3~5 个月历史，供 MA/ATR/RSI 等深度评分。

    返回 (market_dataset, full_snapshot, selected_snapshot, meta)。
    """
    snapshot = fetch_realtime_a_snapshot()
    selected = preselect_live_candidates(snapshot, limit=preselect_n)
    if selected.empty:
        raise RuntimeError("实时预筛没有候选股票。")

    now_cn = datetime.now(ZoneInfo("Asia/Shanghai"))
    end = now_cn.strftime("%Y%m%d")
    start = (now_cn - timedelta(days=int(history_calendar_days))).strftime("%Y%m%d")

    rows = []
    failures = []
    workers = max(1, min(int(max_workers), 6))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for _, r in selected.iterrows():
            symbol = str(r["symbol"]).zfill(6)
            name = str(r.get("name", ""))
            futures[pool.submit(_fetch_one_history, symbol, name, r.to_dict(), start, end)] = symbol
        for f in as_completed(futures):
            symbol = futures[f]
            try:
                item = f.result()
                if item is not None and not item.empty:
                    rows.append(item)
                else:
                    failures.append(symbol)
            except Exception:
                failures.append(symbol)

    if not rows:
        raise RuntimeError("实时候选历史数据下载失败，请稍后重试。")

    market = pd.concat(rows, ignore_index=True)
    meta = live_market_context(snapshot)
    meta.update({
        "scan_time": now_cn.strftime("%Y-%m-%d %H:%M:%S"),
        "preselected": int(len(selected)),
        "deep_loaded": int(len(rows)),
        "failed": int(len(failures)),
        "failed_symbols": failures,
    })
    return market, snapshot, selected, meta


def fetch_akshare_history(symbols: list[str], start: str, end: str, output_path: str) -> pd.DataFrame:
    """通过 AkShare 下载日线。网络/API异常时抛出清晰错误。"""
    ak = _require_akshare()
    rows = []
    for s in symbols:
        raw = ak.stock_zh_a_hist(symbol=str(s).zfill(6), period="daily", start_date=start, end_date=end, adjust="")
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
