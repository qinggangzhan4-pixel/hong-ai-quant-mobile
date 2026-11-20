from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st

from ai_quant.config import load_config
from ai_quant.engine import score_universe
from ai_quant.risk import portfolio_risk

ROOT = Path(__file__).resolve().parent
CFG_PATH = ROOT / "config.yaml"
MODEL_PATH = ROOT / "models" / "model.joblib"
DEMO_PATH = ROOT / "examples" / "demo_market.csv"
HOLDINGS_TEMPLATE = ROOT / "examples" / "holdings_template.csv"

st.set_page_config(
    page_title="洪老师AI量化 v2.0",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container {max-width: 780px; padding-top: 1rem; padding-bottom: 5rem;}
[data-testid="stMetric"] {background: #171B22; border: 1px solid #2B313B; padding: .65rem; border-radius: 14px;}
[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
.card {background:#171B22;border:1px solid #2B313B;border-radius:16px;padding:14px 14px 10px 14px;margin:8px 0 12px 0;}
.card h3 {margin:0 0 8px 0;font-size:1.05rem;}
.badge {display:inline-block;padding:3px 8px;border-radius:999px;background:#232A35;margin-right:5px;font-size:.78rem;}
.muted {opacity:.72;font-size:.85rem;}
.bigscore {font-size:1.7rem;font-weight:700;line-height:1.1;}
.stButton>button, .stDownloadButton>button {width:100%;border-radius:12px;height:44px;}
div[data-baseweb="tab-list"] {gap: 3px; overflow-x:auto;}
button[data-baseweb="tab"] {white-space:nowrap;padding-left:10px;padding-right:10px;}
@media (max-width: 640px) {
  .block-container {padding-left:.75rem;padding-right:.75rem;padding-top:.6rem;}
  h1 {font-size:1.45rem !important;}
  h2 {font-size:1.2rem !important;}
  .card {padding:12px;}
}
</style>
""",
    unsafe_allow_html=True,
)

cfg = load_config(str(CFG_PATH))

# ---------------- helpers ----------------
def pct(v):
    try:
        if pd.isna(v): return "—"
        return f"{float(v)*100:.2f}%"
    except Exception:
        return "—"


def num(v, nd=2):
    try:
        if pd.isna(v): return "—"
        return f"{float(v):.{nd}f}"
    except Exception:
        return "—"


def signal_cn(s):
    return {
        "BUY_CANDIDATE":"允许首仓",
        "WATCH":"观察",
        "AVOID":"回避",
        "AVOID_RELAY":"接力回避",
        "RELAY_CANDIDATE":"接力候选",
    }.get(str(s), str(s))


def load_demo():
    return pd.read_csv(DEMO_PATH, dtype={"symbol": str})


def clean_holdings(df):
    if df is None or len(df) == 0:
        return None
    h = df.copy()
    if "symbol" in h.columns:
        h["symbol"] = h["symbol"].astype(str).str.zfill(6)
    if "market_value" in h.columns:
        h["market_value"] = pd.to_numeric(h["market_value"], errors="coerce").fillna(0)
    return h


def risk_panel(holdings):
    if holdings is None or holdings.empty:
        st.info("还没有录入持仓。录入后会检查总仓、单票和单行业集中度。")
        return
    r = portfolio_risk(holdings, float(cfg["account"]["equity"]))
    c1, c2, c3 = st.columns(3)
    c1.metric("总仓", f"{r['total_exposure']*100:.1f}%")
    c2.metric("最大单票", f"{r['max_single_exposure']*100:.1f}%")
    c3.metric("最大行业", f"{r['max_industry_exposure']*100:.1f}%")
    if r["warnings"]:
        for w in r["warnings"]:
            st.warning(w)
    else:
        st.success("组合集中度暂未触发红线。")


def candidate_card(row):
    sig = signal_cn(row.get("signal", ""))
    name = row.get("name", "") if not pd.isna(row.get("name", np.nan)) else ""
    symbol = str(row.get("symbol", "")).zfill(6)
    industry = row.get("industry", "") if not pd.isna(row.get("industry", np.nan)) else "未分类"
    score = row.get("combined_score", row.get("rule_score", np.nan))
    ml = row.get("ml_probability", np.nan)
    action = row.get("action", "")
    html = f"""
<div class='card'>
  <h3>{name} <span class='muted'>{symbol}</span></h3>
  <span class='badge'>{industry}</span><span class='badge'>{sig}</span>
  <div style='display:flex;justify-content:space-between;align-items:flex-end;margin-top:10px;'>
    <div><div class='muted'>综合评分</div><div class='bigscore'>{num(score,1)}</div></div>
    <div style='text-align:right'><div class='muted'>现价</div><b>{num(row.get('close'))}</b></div>
  </div>
  <div class='muted' style='margin-top:10px'>
    市场 {num(row.get('market_score'),0)}/20 · 行业 {num(row.get('industry_score'),0)}/20 · 质量 {num(row.get('quality_score'),0)}/20 · 趋势 {num(row.get('trend_score'),0)}/25 · 执行 {num(row.get('execution_score'),0)}/15
  </div>
  <div style='margin-top:9px'>
    建议：<b>{action}</b> ｜ 首仓 <b>{int(row.get('suggested_shares',0) or 0)}股</b> ｜ 止损 <b>{num(row.get('stop_price'))}</b>
  </div>
  <div class='muted' style='margin-top:5px'>AI概率：{('未启用' if pd.isna(ml) else f'{float(ml)*100:.1f}%')} · 量比 {num(row.get('vol_ratio5'))} · RSI {num(row.get('rsi14'),0)}</div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)


# ---------------- header ----------------
st.title("📈 洪老师 AI 量化 v2.0")
st.caption("手机网页版 · 市场→行业→个股→买点→仓位→止损。AI不达标自动关闭。")

with st.expander("⚙️ 账户参数", expanded=False):
    equity = st.number_input("账户总资产（元）", min_value=10000.0, value=float(cfg["account"]["equity"]), step=1000.0)
    cfg["account"]["equity"] = float(equity)
    st.caption("默认：首仓6%、单票15%、单行业30%、单笔风险0.7%。可在 config.yaml 改。")

# Session state
if "market" not in st.session_state:
    st.session_state.market = None
if "scores" not in st.session_state:
    st.session_state.scores = None
if "holdings" not in st.session_state:
    st.session_state.holdings = pd.DataFrame(columns=["symbol","name","industry","market_value"])

# ---------------- tabs ----------------
tab1, tab2, tab3, tab4 = st.tabs(["🏆 今日候选", "🔎 查一只", "💼 我的持仓", "📤 数据"])

with tab4:
    st.subheader("导入行情数据")
    st.write("手机最省事：上传标准化 CSV。没有数据时可先点演示数据看效果。")
    uploaded = st.file_uploader("行情CSV", type=["csv"], key="market_file")
    c1, c2 = st.columns(2)
    use_demo = c1.button("载入演示数据")
    clear_data = c2.button("清空数据")
    if uploaded is not None:
        try:
            st.session_state.market = pd.read_csv(uploaded, dtype={"symbol": str})
            st.success(f"已读取 {len(st.session_state.market):,} 行行情。")
        except Exception as e:
            st.error(f"读取失败：{e}")
    if use_demo:
        st.session_state.market = load_demo()
        st.success("已载入演示数据。")
    if clear_data:
        st.session_state.market = None
        st.session_state.scores = None
        st.rerun()

    if st.session_state.market is not None:
        st.dataframe(st.session_state.market.tail(20), use_container_width=True, hide_index=True)
        if st.button("🚀 运行全市场评分", type="primary"):
            try:
                h = clean_holdings(st.session_state.holdings)
                scores, _ = score_universe(
                    st.session_state.market,
                    cfg,
                    h,
                    str(MODEL_PATH) if MODEL_PATH.exists() else None,
                )
                st.session_state.scores = scores
                st.success(f"完成：{len(scores)} 只股票已评分。")
            except Exception as e:
                st.error(f"评分失败：{e}")

    st.divider()
    st.markdown("**最少字段：** `date,symbol,open,high,low,close,volume`")
    st.caption("行业、PE/PB/ROE、资金流、竞价、换手等字段越完整，评分越有意义；缺失字段不会伪造。")

with tab3:
    st.subheader("我的持仓")
    st.caption("直接在手机上改。market_value 填当前市值，不是成本。")
    edited = st.data_editor(
        st.session_state.holdings,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "symbol": st.column_config.TextColumn("代码"),
            "name": st.column_config.TextColumn("名称"),
            "industry": st.column_config.TextColumn("行业"),
            "market_value": st.column_config.NumberColumn("市值", min_value=0.0, step=100.0, format="%.0f"),
        },
        key="holdings_editor",
    )
    st.session_state.holdings = clean_holdings(edited) if edited is not None else st.session_state.holdings
    risk_panel(st.session_state.holdings)
    if not st.session_state.holdings.empty:
        csv = st.session_state.holdings.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载持仓CSV", csv, "holdings.csv", "text/csv")

with tab1:
    st.subheader("今日候选")
    if st.session_state.scores is None:
        st.info("先到“数据”页上传行情并运行评分；也可以载入演示数据。")
    else:
        scores = st.session_state.scores.copy()
        latest = scores["date"].max() if "date" in scores.columns and len(scores) else ""
        st.caption(f"评分日期：{latest}")
        filt = st.segmented_control("范围", ["全部", "首仓候选", "观察", "接力候选"], default="首仓候选")
        view = scores
        if filt == "首仓候选":
            view = scores[scores["signal"] == "BUY_CANDIDATE"]
        elif filt == "观察":
            view = scores[scores["signal"] == "WATCH"]
        elif filt == "接力候选":
            view = scores[scores["signal"] == "RELAY_CANDIDATE"]
        if view.empty:
            st.warning("当前筛选没有股票。弱市里空结果是正常结果，不强行凑票。")
        else:
            topn = st.slider("显示前N只", 3, 20, 8)
            for _, r in view.head(topn).iterrows():
                candidate_card(r)
        out = scores.to_csv(index=False).encode("utf-8-sig")
        st.download_button("下载全部评分结果", out, "scores.csv", "text/csv")

with tab2:
    st.subheader("查一只股票")
    if st.session_state.scores is None:
        st.info("先运行一次全市场评分。")
    else:
        q = st.text_input("输入股票代码或名称", placeholder="例如 600584 / 长电科技")
        if q:
            s = st.session_state.scores.copy()
            mask = s["symbol"].astype(str).str.contains(q, na=False)
            if "name" in s.columns:
                mask = mask | s["name"].astype(str).str.contains(q, na=False)
            hit = s[mask]
            if hit.empty:
                st.warning("当前评分数据里没找到。")
            else:
                r = hit.iloc[0]
                candidate_card(r)
                st.markdown("**执行提示**")
                score = float(r.get("combined_score", 0))
                if score >= cfg["strategy"]["score_buy_threshold"]:
                    st.success("评分允许进入候选，但仍需等回踩确认或放量突破，不等于无条件追价。")
                elif score >= cfg["strategy"]["score_watch_threshold"]:
                    st.warning("观察区：不抢，等待价格和量能触发。")
                else:
                    st.error("当前不主动开仓。")

st.divider()
st.caption("研究与决策辅助工具，不保证收益。AI概率来自历史样本，仓位和止损优先于预测。")
