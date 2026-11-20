# Streamlit Cloud 手机上传修复入口 v4
# v4 新增：持仓在刷新页面后自动恢复（通过浏览器 URL 参数保存，不写服务器文件）。
# 注意：URL 参数只是编码，不是加密；不要把带 ?holdings=... 的完整链接分享给别人。

from pathlib import Path
import sys
import types
import shutil

ROOT = Path(__file__).resolve().parent

# ---------- 兼容被拍平的 ai_quant 包 ----------
if "ai_quant" not in sys.modules:
    pkg = types.ModuleType("ai_quant")
    pkg.__path__ = [str(ROOT)]
    pkg.__package__ = "ai_quant"
    sys.modules["ai_quant"] = pkg

# ---------- 恢复 examples 目录 ----------
def ensure_dir_file(folder: str, filename: str):
    target_dir = ROOT / folder
    target = target_dir / filename
    source = ROOT / filename
    if not target.exists() and source.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

for name in (
    "demo_market.csv",
    "scores_demo.csv",
    "backtest_demo.csv",
    "holdings_template.csv",
    "generate_demo.py",
):
    ensure_dir_file("examples", name)

# ---------- 执行原 app.py，并在运行前打补丁 ----------
app_file = ROOT / "app.py"
if not app_file.exists():
    raise FileNotFoundError("仓库根目录没有 app.py，请先上传 app.py。")

source = app_file.read_text(encoding="utf-8")

# 1) 增加 base64，用 URL 参数持久化持仓
source = source.replace(
    "import json\nimport pandas as pd",
    "import json\nimport base64\nimport pandas as pd",
)

# 2) 修复空持仓返回 None
old_clean = (
    'def clean_holdings(df):\n'
    '    if df is None or len(df) == 0:\n'
    '        return None\n'
)
new_clean = (
    'def clean_holdings(df):\n'
    '    if df is None or len(df) == 0:\n'
    '        return pd.DataFrame(columns=["symbol","name","industry","market_value"])\n'
)
source = source.replace(old_clean, new_clean)

source = source.replace(
    '    if not st.session_state.holdings.empty:\n',
    '    if st.session_state.holdings is not None and not st.session_state.holdings.empty:\n'
)

# 3) 禁用旧的、不兼容的 joblib AI 模型；规则量化仍正常运行
source = source.replace(
    'MODEL_PATH = ROOT / "models" / "model.joblib"',
    'MODEL_PATH = ROOT / "models" / "__cloud_ai_disabled_until_retrain__.joblib"'
)

# 4) 插入持仓读写函数：保存到当前浏览器 URL 参数
marker = "# ---------------- header ----------------\n"
persist_helpers = r"""
# ---------------- browser persistence ----------------
def _holdings_from_url():
    try:
        token = st.query_params.get("holdings", "")
        if not token:
            return pd.DataFrame(columns=["symbol","name","industry","market_value"])
        pad = "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode((token + pad).encode("ascii")).decode("utf-8")
        rows = json.loads(raw)
        df = pd.DataFrame(rows)
        return clean_holdings(df)
    except Exception:
        return pd.DataFrame(columns=["symbol","name","industry","market_value"])


def _holdings_to_url(df):
    try:
        h = clean_holdings(df)
        if h is None or h.empty:
            if "holdings" in st.query_params:
                del st.query_params["holdings"]
            return
        cols = [c for c in ["symbol","name","industry","market_value"] if c in h.columns]
        payload = json.dumps(h[cols].to_dict("records"), ensure_ascii=False, separators=(",", ":"))
        token = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
        if st.query_params.get("holdings", "") != token:
            st.query_params["holdings"] = token
    except Exception:
        pass

"""
source = source.replace(marker, persist_helpers + marker)

# 5) 首次打开 / 刷新时，从 URL 恢复持仓
source = source.replace(
    'if "holdings" not in st.session_state:\n'
    '    st.session_state.holdings = pd.DataFrame(columns=["symbol","name","industry","market_value"])',
    'if "holdings" not in st.session_state:\n'
    '    st.session_state.holdings = _holdings_from_url()'
)

# 6) 每次编辑后自动保存到 URL；刷新后自动回来
old_assign = '    st.session_state.holdings = clean_holdings(edited) if edited is not None else st.session_state.holdings\n'
new_assign = (
    '    st.session_state.holdings = clean_holdings(edited) if edited is not None else st.session_state.holdings\n'
    '    _holdings_to_url(st.session_state.holdings)\n'
    '    st.caption("✅ 持仓已自动保存；刷新页面会恢复。请勿分享带 holdings 参数的完整网址。")\n'
)
source = source.replace(old_assign, new_assign)

exec(compile(source, str(app_file), "exec"), globals(), globals())
