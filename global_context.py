from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote
from zoneinfo import ZoneInfo
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests


GLOBAL_ASSETS = [
    {"ticker": "^DJI", "name": "道琼斯工业指数", "group": "us_broad", "market": "美国"},
    {"ticker": "^GSPC", "name": "标普500", "group": "us_broad", "market": "美国"},
    {"ticker": "^IXIC", "name": "纳斯达克综合指数", "group": "us_tech", "market": "美国"},
    {"ticker": "^SOX", "name": "费城半导体指数", "group": "us_tech", "market": "美国"},
    {"ticker": "NVDA", "name": "英伟达", "group": "us_tech", "market": "美国"},
    {"ticker": "AMD", "name": "AMD", "group": "us_tech", "market": "美国"},
    {"ticker": "MU", "name": "美光科技", "group": "us_tech", "market": "美国"},
    {"ticker": "DELL", "name": "戴尔科技", "group": "us_tech", "market": "美国"},
    {"ticker": "^KS11", "name": "韩国综合指数", "group": "asia", "market": "韩国"},
    {"ticker": "^KQ11", "name": "韩国KOSDAQ", "group": "asia", "market": "韩国"},
    {"ticker": "005930.KS", "name": "三星电子", "group": "asia", "market": "韩国"},
    {"ticker": "000660.KS", "name": "SK海力士", "group": "asia", "market": "韩国"},
    {"ticker": "^N225", "name": "日经225", "group": "asia", "market": "日本"},
]

CONSUMER_CATEGORIES = {
    "消费/食品饮料": ["食品饮料", "食品", "饮料", "白酒", "乳制品", "调味品", "啤酒", "黄酒", "葡萄酒", "肉制品", "休闲食品"],
    "消费/家电家居": ["家用电器", "家电", "家具", "家居", "厨卫", "家纺", "装修建材"],
    "消费/零售电商": ["商贸零售", "零售", "电商", "百货", "超市", "商业连锁"],
    "消费/旅游酒店餐饮": ["旅游", "景区", "酒店", "餐饮", "旅游服务", "航空机场"],
    "消费/美妆服饰珠宝": ["美容护理", "化妆品", "纺织服饰", "服装", "鞋帽", "珠宝首饰", "黄金珠宝"],
    "消费/汽车": ["汽车整车", "汽车零部件", "汽车服务", "汽车", "乘用车", "商用车"],
    "消费/消费电子": ["消费电子", "智能穿戴", "家用视听"],
}

SECTOR_CATEGORIES = {
    "科技": ["半导体", "电子", "通信", "计算机", "软件", "互联网", "光学光电子", "元器件", "IT服务"],
    "金融": ["银行", "证券", "保险", "多元金融", "非银金融", "金融租赁"],
    "医药": ["医药", "医疗器械", "医疗服务", "生物制品", "中药", "化学制药", "创新药"],
    "周期": ["化工", "有色金属", "煤炭", "钢铁", "石油", "石化", "建材", "基础化工"],
    "高端制造": ["电力设备", "机械设备", "工业自动化", "机器人", "航空装备", "国防军工", "电网设备"],
    "公用事业": ["电力", "水务", "燃气", "环保", "公用事业"],
}


def classify_market_sector(name=None, industry=None) -> str:
    """依据行情源行业字段优先、名称关键词辅助，给A股打研究分类标签。"""
    text = f"{industry or ''} {name or ''}".strip()
    for category, keywords in CONSUMER_CATEGORIES.items():
        if any(k in text for k in keywords):
            return category
    for category, keywords in SECTOR_CATEGORIES.items():
        if any(k in text for k in keywords):
            return category
    return "其他"


def _safe_float(value):
    try:
        if value is None:
            return np.nan
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def consumer_fundamental_score(row) -> float | None:
    """消费股专属质量分(0-10)。只有至少两项真实财务字段可用时才启用。"""
    sector = classify_market_sector(row.get("name"), row.get("industry"))
    if not sector.startswith("消费/"):
        return None

    scores = []
    roe = _safe_float(row.get("roe"))
    profit_yoy = _safe_float(row.get("net_profit_yoy"))
    revenue_yoy = _safe_float(row.get("revenue_yoy"))
    gross_margin = _safe_float(row.get("gross_margin"))
    inventory_turnover = _safe_float(row.get("inventory_turnover"))
    cashflow_ratio = _safe_float(row.get("operating_cashflow_to_profit"))

    if not np.isnan(roe):
        scores.append(10 if roe >= 15 else 8 if roe >= 10 else 6 if roe >= 5 else 4 if roe >= 0 else 1)
    if not np.isnan(profit_yoy):
        scores.append(10 if profit_yoy >= 30 else 8 if profit_yoy >= 10 else 6 if profit_yoy >= 0 else 4 if profit_yoy > -20 else 1)
    if not np.isnan(revenue_yoy):
        scores.append(10 if revenue_yoy >= 20 else 8 if revenue_yoy >= 8 else 6 if revenue_yoy >= 0 else 3)
    if not np.isnan(gross_margin):
        gm = gross_margin * 100 if abs(gross_margin) <= 1 else gross_margin
        scores.append(10 if gm >= 40 else 8 if gm >= 25 else 6 if gm >= 15 else 4 if gm >= 5 else 2)
    if not np.isnan(inventory_turnover):
        scores.append(10 if inventory_turnover >= 8 else 8 if inventory_turnover >= 4 else 6 if inventory_turnover >= 2 else 4 if inventory_turnover >= 1 else 2)
    if not np.isnan(cashflow_ratio):
        scores.append(10 if cashflow_ratio >= 1 else 8 if cashflow_ratio >= 0.5 else 6 if cashflow_ratio >= 0 else 2)

    if len(scores) < 2:
        return None
    return round(float(np.mean(scores)), 2)


def _fetch_yahoo_asset(asset: dict) -> dict:
    ticker = str(asset["ticker"])
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(ticker, safe='^.')}"
    params = {"range": "5d", "interval": "1d", "includePrePost": "false"}
    headers = {"User-Agent": "Mozilla/5.0 (compatible; HongQuantResearch/1.0)"}
    response = requests.get(url, params=params, headers=headers, timeout=9)
    response.raise_for_status()
    payload = response.json()
    chart = payload.get("chart", {})
    if chart.get("error"):
        raise RuntimeError(str(chart["error"]))
    result = (chart.get("result") or [None])[0]
    if not result:
        raise RuntimeError("行情接口未返回数据")

    meta = result.get("meta") or {}
    timestamps = result.get("timestamp") or []
    quote_data = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    closes = [v for v in (quote_data.get("close") or []) if v is not None]
    price = _safe_float(meta.get("regularMarketPrice"))
    if np.isnan(price) and closes:
        price = _safe_float(closes[-1])
    previous = _safe_float(meta.get("chartPreviousClose"))
    if np.isnan(previous):
        previous = _safe_float(meta.get("previousClose"))
    if np.isnan(previous) and len(closes) >= 2:
        previous = _safe_float(closes[-2])
    if np.isnan(price) or np.isnan(previous) or previous == 0:
        raise RuntimeError("缺少最新价或前收盘价")

    market_time = meta.get("regularMarketTime")
    if market_time is None and timestamps:
        market_time = timestamps[-1]
    market_date = ""
    if market_time is not None:
        market_date = datetime.fromtimestamp(int(market_time), tz=ZoneInfo("UTC")).astimezone(
            ZoneInfo("Asia/Shanghai")
        ).strftime("%Y-%m-%d %H:%M")

    return {
        "ticker": ticker,
        "name": asset["name"],
        "market": asset["market"],
        "group": asset["group"],
        "price": round(float(price), 4),
        "pct_change": round((float(price) / float(previous) - 1) * 100, 3),
        "market_time_cn": market_date,
        "source": "Yahoo Finance公开行情接口",
    }


def fetch_global_market_snapshot(max_workers: int = 6) -> pd.DataFrame:
    """抓取美股、韩国、日本指数及代表性股票；单只失败不影响其他标的。"""
    rows, failures = [], []
    with ThreadPoolExecutor(max_workers=max(1, min(int(max_workers), 8))) as pool:
        futures = {pool.submit(_fetch_yahoo_asset, asset): asset for asset in GLOBAL_ASSETS}
        for future in as_completed(futures):
            asset = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append(f"{asset['ticker']} {asset['name']}: {exc}")
    columns = ["ticker", "name", "market", "group", "price", "pct_change", "market_time_cn", "source"]
    out = pd.DataFrame(rows, columns=columns)
    if not out.empty:
        out = out.sort_values(["group", "market", "ticker"]).reset_index(drop=True)
    out.attrs["failures"] = failures
    out.attrs["fetched_at_cn"] = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    out.attrs["requested_count"] = len(GLOBAL_ASSETS)
    out.attrs["success_count"] = len(out)
    return out


def fetch_global_news(max_per_query: int = 6) -> pd.DataFrame:
    """从Google News RSS收集隔夜宏观、海外科技、韩国半导体及中国消费政策新闻线索。"""
    queries = [
        ("海外市场与美联储", "US stock market Federal Reserve inflation tariffs AI semiconductor"),
        ("韩国与半导体", "韩国股市 三星电子 SK海力士 半导体"),
        ("中国政策与消费", "中国 A股 消费政策 零售 家电 汽车 经济"),
        ("大宗商品与汇率", "oil gold copper US dollar yuan market"),
    ]
    rows, failures = [], []
    headers = {"User-Agent": "Mozilla/5.0"}
    for label, query in queries:
        try:
            response = requests.get(
                "https://news.google.com/rss/search",
                params={"q": query, "hl": "zh-CN", "gl": "CN", "ceid": "CN:zh-Hans"},
                headers=headers,
                timeout=8,
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
            items = root.findall(".//item")
            for item in items[:max_per_query]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub = (item.findtext("pubDate") or "").strip()
                source = (item.findtext("source") or "").strip()
                published = pub
                try:
                    published = parsedate_to_datetime(pub).astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    pass
                if title and link:
                    rows.append({
                        "主题": label,
                        "时间": published,
                        "标题": title,
                        "来源": source,
                        "链接": link,
                    })
        except Exception as exc:
            failures.append(f"{label}: {exc}")
    out = pd.DataFrame(rows, columns=["主题", "时间", "标题", "来源", "链接"])
    if not out.empty:
        out = out.drop_duplicates(subset=["标题"]).sort_values("时间", ascending=False).reset_index(drop=True)
    out.attrs["failures"] = failures
    out.attrs["fetched_at_cn"] = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    return out


def _group_score(values, max_points: float) -> float:
    valid = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if valid.empty:
        return max_points / 2
    avg = float(valid.mean())
    if avg >= 1.5:
        factor = 1.0
    elif avg >= 0.5:
        factor = 0.82
    elif avg >= 0:
        factor = 0.65
    elif avg >= -0.5:
        factor = 0.5
    elif avg >= -1.5:
        factor = 0.3
    else:
        factor = 0.1
    return max_points * factor


def score_global_market(snapshot: pd.DataFrame | None, news_adjustment: float = 0.0) -> tuple[float, dict]:
    """全球环境20分：美股大盘2、美股科技/半导体3、日韩2、隔夜新闻/宏观3；缺失组按中性处理。"""
    weights = {"us_broad": 2.0, "us_tech": 3.0, "asia": 2.0}
    breakdown = {}
    if snapshot is None or snapshot.empty:
        for group, weight in weights.items():
            breakdown[group] = {"score": weight / 2, "weight": weight, "average_pct": None, "count": 0}
    else:
        for group, weight in weights.items():
            values = snapshot.loc[snapshot["group"] == group, "pct_change"] if "group" in snapshot.columns else []
            valid = pd.to_numeric(values, errors="coerce").dropna() if len(values) else pd.Series(dtype=float)
            breakdown[group] = {
                "score": round(_group_score(valid, weight), 2),
                "weight": weight,
                "average_pct": round(float(valid.mean()), 3) if len(valid) else None,
                "count": int(len(valid)),
            }
    event_score = float(np.clip(1.5 + float(news_adjustment or 0), 0, 3))
    breakdown["news_macro"] = {"score": round(event_score, 2), "weight": 3.0, "average_pct": None, "count": 0}
    total = round(sum(item["score"] for item in breakdown.values()), 2)
    breakdown["total"] = total
    return total, breakdown
