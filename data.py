from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import time

import numpy as np
import pandas as pd
import requests
from urllib.parse import urlsplit, urlunsplit


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
    """请求东方财富公开 JSON 接口。

    只使用标准域名，不再拼接 82.7.push2.eastmoney.com 这类错误节点域名。
    旧逻辑对已带 82. 前缀的 URL 再做字符串替换，会生成 82.7.push2...
    从而触发 SSL Hostname mismatch。
    """
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()

    # 清除历史节点前缀，统一回官方主机名；不关闭 SSL 证书校验。
    if host.endswith(".push2.eastmoney.com"):
        host = "push2.eastmoney.com"
    elif host.endswith(".push2his.eastmoney.com"):
        host = "push2his.eastmoney.com"

    netloc = host
    if parsed.port:
        netloc = f"{host}:{parsed.port}"
    request_url = urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://quote.eastmoney.com/",
        "Accept": "application/json,text/plain,*/*",
    }

    last_err = None
    for attempt in range(3):
        try:
            r = requests.get(request_url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            if isinstance(data, dict):
                return data
            raise RuntimeError("行情接口返回的不是 JSON 对象")
        except requests.exceptions.SSLError as e:
            # 证书错误不是普通超时，不重试错误域名，更不允许 verify=False 绕过校验。
            raise RuntimeError(
                f"东方财富行情接口 SSL 证书校验失败（{host}）：{e}"
            ) from e
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(0.4 * (attempt + 1))

    raise RuntimeError(f"公开行情接口暂时无响应（{host}）：{last_err}")


def fetch_realtime_a_snapshot() -> pd.DataFrame:
    """分页抓取沪深京 A 股实时快照。

    东财 clist 接口现在会限制单页返回数量，因此不能再用 pz=20000 假设一次拿全。
    这里按 200/页并发抓取全部页面，并对失败页重试一次。
    """
    fields = [
        "f12","f14","f2","f3","f4","f5","f6","f7","f15","f16","f17","f18",
        "f10","f8","f9","f23","f20","f21","f22","f11"
    ]
    base_params = {
        "pz": 200,
        "po": 1,
        "np": 1,
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": 2,
        "invt": 2,
        "fid": "f3",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": ",".join(fields),
    }

    def fetch_page(pn: int, timeout: float = 7.0):
        params = dict(base_params)
        params["pn"] = int(pn)
        j = _get_json("https://push2.eastmoney.com/api/qt/clist/get", params, timeout=timeout)
        data = j.get("data") or {}
        diff = data.get("diff") or []
        if isinstance(diff, dict):
            records = list(diff.values())
        elif isinstance(diff, list):
            records = diff
        else:
            records = []
        return records, int(data.get("total") or 0)

    first, total = fetch_page(1, timeout=8.0)
    if not first:
        raise RuntimeError("实时 A 股第一页为空，请稍后重试。")

    # 接口可能实际只给100/页，所以以“第一页实际数量”作为真实页容量。
    actual_page_size = max(1, len(first))
    if total <= 0:
        total = actual_page_size
    total_pages = max(1, int(np.ceil(total / actual_page_size)))

    page_records = {1: first}
    failed_pages = []

    if total_pages > 1:
        workers = min(12, total_pages - 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(fetch_page, pn, 7.0): pn for pn in range(2, total_pages + 1)}
            for f in as_completed(futs):
                pn = futs[f]
                try:
                    records, _ = f.result()
                    if records:
                        page_records[pn] = records
                    else:
                        failed_pages.append(pn)
                except Exception:
                    failed_pages.append(pn)

    # 失败页再重试一次，避免偶发丢页。
    retry_failed = []
    for pn in failed_pages:
        try:
            records, _ = fetch_page(pn, timeout=6.0)
            if records:
                page_records[pn] = records
            else:
                retry_failed.append(pn)
        except Exception:
            retry_failed.append(pn)

    all_records = []
    for pn in sorted(page_records):
        all_records.extend(page_records[pn])

    rows = []
    for item in all_records:
        if not isinstance(item, dict):
            continue
        rows.append({
            "symbol": item.get("f12"),
            "name": item.get("f14"),
            "close": item.get("f2"),
            "pct_change": item.get("f3"),
            "price_change": item.get("f4"),
            "volume": item.get("f5"),
            "amount": item.get("f6"),
            "amplitude": item.get("f7"),
            "high": item.get("f15"),
            "low": item.get("f16"),
            "open": item.get("f17"),
            "prev_close": item.get("f18"),
            "live_volume_ratio": item.get("f10"),
            "turnover_rate": item.get("f8"),
            "pe_ttm": item.get("f9"),
            "pb": item.get("f23"),
            "market_cap": item.get("f20"),
            "float_market_cap": item.get("f21"),
            "speed": item.get("f22"),
            "change_5m": item.get("f11"),
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

    # 只要拿到的数据明显不完整，就明确报错，不让后续误把残缺行情当“全市场”。
    completeness = len(out) / max(total, 1)
    if len(out) < 1000 or completeness < 0.85:
        raise RuntimeError(
            f"实时行情不完整：接口总数约 {total} 只，实际拿到 {len(out)} 只，"
            f"失败页 {len(retry_failed)} 页。请重试。"
        )

    out.attrs["source_total"] = total
    out.attrs["actual_count"] = len(out)
    out.attrs["page_size"] = actual_page_size
    out.attrs["total_pages"] = total_pages
    out.attrs["failed_pages"] = retry_failed
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



WATCHLIST_MAINBOARD = [
    {"symbol":"601138","name":"工业富联","industry":"AI服务器"},
    {"symbol":"002384","name":"东山精密","industry":"PCB/电子制造"},
    {"symbol":"002281","name":"光迅科技","industry":"光通信"},
    {"symbol":"002463","name":"沪电股份","industry":"AI服务器PCB"},
    {"symbol":"002916","name":"深南电路","industry":"PCB/封装基板"},
    {"symbol":"600183","name":"生益科技","industry":"覆铜板材料"},
    {"symbol":"600667","name":"太极实业","industry":"半导体封测"},
    {"symbol":"000977","name":"浪潮信息","industry":"AI服务器"},
]


def fetch_watchlist_scan(history_calendar_days: int = 260, max_workers: int = 4):
    """为固定主板观察池逐只拉取实时行情和至少80个交易日的日K。"""
    t0 = time.time()
    snapshot = fetch_realtime_a_snapshot()
    wanted = {x["symbol"] for x in WATCHLIST_MAINBOARD}
    live = snapshot[snapshot["symbol"].astype(str).str.zfill(6).isin(wanted)].copy()
    live = live[live["is_mainboard"].fillna(False).map(bool)].copy()
    live = live[live["is_st"].fillna(False).map(bool).eq(False)].copy()

    live_map = {str(r["symbol"]).zfill(6): r.to_dict() for _, r in live.iterrows()}
    missing_live = sorted(wanted - set(live_map))
    now_cn = datetime.now(ZoneInfo("Asia/Shanghai"))
    end = now_cn.strftime("%Y%m%d")
    start = (now_cn - timedelta(days=int(history_calendar_days))).strftime("%Y%m%d")

    rows, failures, short_history = [], [], []
    workers = max(1, min(int(max_workers), 6))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for item in WATCHLIST_MAINBOARD:
            symbol = item["symbol"]
            live_row = live_map.get(symbol)
            if live_row is None:
                failures.append(symbol)
                continue
            futures[pool.submit(
                _fetch_one_history, symbol, item["name"], live_row, start, end
            )] = symbol

        for future in as_completed(futures):
            symbol = futures[future]
            try:
                frame = future.result()
                if frame is None or frame.empty:
                    failures.append(symbol)
                    continue
                history_count = int(frame.loc[frame["date"] < pd.Timestamp(now_cn.date())].shape[0])
                if history_count < 80:
                    short_history.append(f"{symbol}({history_count}日)")
                    continue
                frame["industry"] = next(
                    x["industry"] for x in WATCHLIST_MAINBOARD if x["symbol"] == symbol
                )
                rows.append(frame)
            except Exception as exc:
                failures.append(f"{symbol}:{str(exc)[:120]}")

    if not rows:
        raise RuntimeError(
            "观察池历史K线全部获取失败；请检查东方财富接口、部署依赖或网络。"
        )

    market = pd.concat(rows, ignore_index=True)
    meta = live_market_context(snapshot)
    meta.update({
        "scan_time": now_cn.strftime("%Y-%m-%d %H:%M:%S"),
        "watchlist_size": len(WATCHLIST_MAINBOARD),
        "loaded_symbols": int(market["symbol"].nunique()),
        "missing_live": missing_live,
        "failed_symbols": failures,
        "short_history": short_history,
        "history_calendar_days": int(history_calendar_days),
        "elapsed_seconds": round(time.time() - t0, 1),
    })
    return market, snapshot, meta

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
