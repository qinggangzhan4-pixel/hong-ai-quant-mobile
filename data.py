from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import time

import numpy as np
import pandas as pd
import requests


def load_market_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"symbol": str})


def load_holdings_csv(path: str | None) -> pd.DataFrame | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"symbol": str})


def _to_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _is_mainboard_symbol(symbol: str) -> bool:
    s = str(symbol).zfill(6)
    return s.startswith(("000", "001", "002", "003", "600", "601", "603", "605"))


def _eastmoney_secid(symbol: str) -> str:
    s = str(symbol).zfill(6)
    return ("1." if s.startswith(("5", "6", "9")) else "0.") + s


def _get_json(url: str, params: dict, timeout: float = 12.0) -> dict:
    last_err = None
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }
    # 多个域名轮换，避免单一节点在云服务器上偶发超时。
    hosts = [
        url,
        url.replace("push2.eastmoney.com", "82.push2.eastmoney.com"),
        url.replace("push2.eastmoney.com", "7.push2.eastmoney.com"),
    ]
    seen = set()
    for u in hosts:
        if u in seen:
            continue
        seen.add(u)
        try:
            r = requests.get(u, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                return data
        except Exception as e:
            last_err = e
    raise RuntimeError(f"公开行情接口暂时无响应：{last_err}")


def fetch_realtime_a_snapshot() -> pd.DataFrame:
    """抓取全 A 股公开网页实时快照（东方财富），自动分页拿全市场。

    这不是券商逐笔数据。盘中可刷新；午休/收盘时返回最近快照。
    """
    fields = [
        "f12","f14","f2","f3","f4","f5","f6","f7","f15","f16","f17","f18",
        "f10","f8","f9","f23","f20","f21","f22","f11"
    ]
    base_params = {
        "pz": 100,
        "po": 1,
        "np": 1,
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": "m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23",
        "fields": ",".join(fields),
    }

    # 第一页先拿总数。东方财富这个接口单页常被限制在约100只，
    # 所以原先只请求一页会误把“涨幅前100”当成“全市场”。
    p1 = dict(base_params)
    p1["pn"] = 1
    j1 = _get_json("https://push2.eastmoney.com/api/qt/clist/get", p1, timeout=10)
    data1 = j1.get("data") or {}
    first = data1.get("diff") or []
    total = int(data1.get("total") or len(first))
    if not first:
        raise RuntimeError("实时 A 股快照为空，请稍后重试。")

    page_size = max(1, len(first))
    pages = max(1, int(np.ceil(total / page_size)))

    all_diff = list(first)

    def fetch_page(pn: int):
        p = dict(base_params)
        p["pn"] = pn
        j = _get_json("https://push2.eastmoney.com/api/qt/clist/get", p, timeout=10)
        return ((j.get("data") or {}).get("diff") or [])

    # 并发补齐后续页面，避免手机端一页页等。
    if pages > 1:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(fetch_page, pn): pn for pn in range(2, pages + 1)}
            for f in as_completed(futs):
                try:
                    all_diff.extend(f.result())
                except Exception:
                    # 少量页面失败时保留已拿到的数据；最终会在 meta 里显示实际数量。
                    pass

    rows = []
    for d in all_diff:
        rows.append({
            "symbol": d.get("f12"),
            "name": d.get("f14"),
            "close": d.get("f2"),
            "pct_change": d.get("f3"),
            "price_change": d.get("f4"),
            "volume": d.get("f5"),
            "amount": d.get("f6"),
            "amplitude": d.get("f7"),
            "high": d.get("f15"),
            "low": d.get("f16"),
            "open": d.get("f17"),
            "prev_close": d.get("f18"),
            "live_volume_ratio": d.get("f10"),
            "turnover_rate": d.get("f8"),
            "pe_ttm": d.get("f9"),
            "pb": d.get("f23"),
            "market_cap": d.get("f20"),
            "float_market_cap": d.get("f21"),
            "speed": d.get("f22"),
            "change_5m": d.get("f11"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("实时 A 股快照解析为空，请稍后重试。")

    out["symbol"] = out["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    out = out.drop_duplicates(subset=["symbol"], keep="first").reset_index(drop=True)
    out["name"] = out["name"].fillna("").astype(str)
    numeric = [
        "close","pct_change","price_change","volume","amount","amplitude","high","low","open",
        "prev_close","live_volume_ratio","turnover_rate","pe_ttm","pb","market_cap",
        "float_market_cap","speed","change_5m",
    ]
    out = _to_num(out, numeric)
    now_cn = datetime.now(ZoneInfo("Asia/Shanghai"))
    out["date"] = pd.Timestamp(now_cn.date())
    out["is_st"] = out["name"].str.upper().str.contains(r"ST|退", regex=True)
    out["is_mainboard"] = out["symbol"].map(_is_mainboard_symbol)
    out["is_limit_up"] = out["pct_change"].fillna(-999) >= 9.5
    out = out[out["close"].fillna(0) > 0].reset_index(drop=True)
    return out

def fetch_realtime_etf_snapshot() -> pd.DataFrame:
    """ETF 快照。当前主扫描不依赖它；仅用于刷新持仓 ETF 现价。失败会由调用方忽略。"""
    try:
        import akshare as ak
        raw = ak.fund_etf_spot_em()
        if raw is None or raw.empty:
            return pd.DataFrame()
        mapping = {
            "代码": "symbol", "名称": "name", "最新价": "close", "涨跌幅": "pct_change",
            "成交量": "volume", "成交额": "amount", "最高": "high", "最低": "low",
            "今开": "open", "昨收": "prev_close", "换手率": "turnover_rate",
        }
        out = raw.rename(columns={k:v for k,v in mapping.items() if k in raw.columns}).copy()
        if "symbol" not in out.columns or "close" not in out.columns:
            return pd.DataFrame()
        out["symbol"] = out["symbol"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        return _to_num(out, ["close","pct_change","volume","amount","high","low","open","prev_close","turnover_rate"])
    except Exception:
        return pd.DataFrame()


def live_market_context(snapshot: pd.DataFrame) -> dict:
    x = snapshot.copy()
    x["pct_change"] = pd.to_numeric(x.get("pct_change"), errors="coerce")
    x = x[x["pct_change"].notna()]
    if x.empty:
        return {}
    up = int((x["pct_change"] > 0).sum())
    down = int((x["pct_change"] < 0).sum())
    flat = int((x["pct_change"] == 0).sum())
    limit_up = int((x["pct_change"] >= 9.5).sum())
    limit_down = int((x["pct_change"] <= -9.5).sum())
    amount = float(pd.to_numeric(x.get("amount"), errors="coerce").fillna(0).sum())
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


def preselect_live_candidates(snapshot: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    """全市场实时预筛：先快速缩小到少量股票，再拉历史做深度评分。"""
    x = snapshot.copy()
    main_mask = x["is_mainboard"].fillna(False).map(bool).astype(bool)
    st_mask = x["is_st"].fillna(False).map(bool).astype(bool)
    x = x.loc[main_mask & st_mask.eq(False)].copy()
    x = x[(x["close"] >= 2.0) & (x["close"] <= 300.0)]
    x = x[pd.to_numeric(x["amount"], errors="coerce").fillna(0) >= 20_000_000]
    if x.empty:
        return x

    amt = pd.to_numeric(x["amount"], errors="coerce").fillna(0)
    chg = pd.to_numeric(x["pct_change"], errors="coerce").fillna(0)
    turn = pd.to_numeric(x["turnover_rate"], errors="coerce").fillna(0)
    vr = pd.to_numeric(x["live_volume_ratio"], errors="coerce").fillna(1)

    amount_score = amt.rank(pct=True).fillna(0)
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
    x["live_pre_score"] = amount_score * 40 + chg_score * 25 + turn_score * 20 + vr_score * 15
    return x.sort_values(["live_pre_score", "amount"], ascending=False).head(int(limit)).reset_index(drop=True)


def _fetch_one_history(symbol: str, name: str, snapshot_row: dict, start: str, end: str) -> pd.DataFrame | None:
    """直接调用东方财富 K 线接口，带超时，避免 AkShare 在云端长时间卡死。"""
    secid = _eastmoney_secid(symbol)
    params = {
        "secid": secid,
        "klt": 101,
        "fqt": 0,
        "beg": start,
        "end": end,
        "lmt": 220,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    j = _get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", params, timeout=10)
    klines = ((j.get("data") or {}).get("klines") or [])
    if not klines:
        return None

    rows = []
    for line in klines:
        p = str(line).split(",")
        if len(p) < 11:
            continue
        rows.append({
            "date": p[0], "open": p[1], "close": p[2], "high": p[3], "low": p[4],
            "volume": p[5], "amount": p[6], "amplitude": p[7], "pct_change": p[8],
            "price_change": p[9], "turnover_rate": p[10],
        })
    hist = pd.DataFrame(rows)
    if hist.empty:
        return None
    hist["date"] = pd.to_datetime(hist["date"])
    for c in ["open","close","high","low","volume","amount","amplitude","pct_change","price_change","turnover_rate"]:
        hist[c] = pd.to_numeric(hist[c], errors="coerce")
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
    for c in ["live_volume_ratio","pe_ttm","pb","market_cap","is_st","is_limit_up"]:
        if c not in hist.columns:
            hist[c] = np.nan
    hist["is_st"] = hist["is_st"].fillna(False)
    hist["is_limit_up"] = hist["is_limit_up"].fillna(False)

    live_df = pd.DataFrame([live])
    cols = sorted(set(hist.columns).union(live_df.columns))
    return pd.concat([hist.reindex(columns=cols), live_df.reindex(columns=cols)], ignore_index=True)


def fetch_live_a_scan(preselect_n: int = 20, history_calendar_days: int = 150, max_workers: int = 4):
    """两阶段实时量化扫描。所有 HTTP 请求都有明确超时，避免手机页面一直没反应。"""
    t0 = time.time()
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
        raise RuntimeError("实时候选历史数据全部下载失败，请稍后重试。")

    market = pd.concat(rows, ignore_index=True)
    meta = live_market_context(snapshot)
    meta.update({
        "scan_time": now_cn.strftime("%Y-%m-%d %H:%M:%S"),
        "preselected": int(len(selected)),
        "deep_loaded": int(len(rows)),
        "failed": int(len(failures)),
        "failed_symbols": failures,
        "elapsed_seconds": round(time.time() - t0, 1),
    })
    return market, snapshot, selected, meta


def fetch_akshare_history(symbols: list[str], start: str, end: str, output_path: str) -> pd.DataFrame:
    """保留原 CLI 功能。"""
    try:
        import akshare as ak
    except ImportError as e:
        raise RuntimeError("未安装 akshare，请先 pip install -r requirements.txt") from e
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
