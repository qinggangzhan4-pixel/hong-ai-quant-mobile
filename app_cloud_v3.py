# Streamlit Cloud mobile compatibility entry
from pathlib import Path
import sys
import types
import shutil

ROOT = Path(__file__).resolve().parent

# Treat flattened repo root as the ai_quant package.
if "ai_quant" not in sys.modules:
    pkg = types.ModuleType("ai_quant")
    pkg.__path__ = [str(ROOT)]
    pkg.__package__ = "ai_quant"
    sys.modules["ai_quant"] = pkg

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

app_file = ROOT / "app.py"
if not app_file.exists():
    raise FileNotFoundError("仓库根目录没有 app.py")

source = app_file.read_text(encoding="utf-8")

# Add base64 import for browser persistence.
source = source.replace(
    "import json\nimport pandas as pd",
    "import json\nimport base64\nimport pandas as pd",
)

# Fix empty holdings returning None.
source = source.replace(
    'def clean_holdings(df):\n'
    '    if df is None or len(df) == 0:\n'
    '        return None\n',
    'def clean_holdings(df):\n'
    '    if df is None or len(df) == 0:\n'
    '        return pd.DataFrame(columns=["symbol","name","industry","market_value"])\n'
)

source = source.replace(
    '    if not st.session_state.holdings.empty:\n',
    '    if st.session_state.holdings is not None and not st.session_state.holdings.empty:\n'
)

# Disable incompatible legacy sklearn joblib on Streamlit Cloud.
source = source.replace(
    'MODEL_PATH = ROOT / "models" / "model.joblib"',
    'MODEL_PATH = ROOT / "models" / "__cloud_ai_disabled_until_retrain__.joblib"'
)

marker = "# ---------------- header ----------------\n"
persist_helpers = r"""
# ---------------- browser persistence ----------------
DEFAULT_HOLDINGS = [
    {"symbol":"600584","name":"长电科技","industry":"半导体","market_value":5191.00},
    {"symbol":"002945","name":"华林证券","industry":"金融","market_value":2890.00},
    {"symbol":"600901","name":"江苏金租","industry":"金融","market_value":28564.00},
    {"symbol":"516310","name":"银行ETF易方达","industry":"金融","market_value":3166.80},
]

def _encode_holdings_rows(rows):
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")

def _holdings_from_url():
    try:
        token = st.query_params.get("holdings", "")
        if not token:
            return clean_holdings(pd.DataFrame(DEFAULT_HOLDINGS))
        pad = "=" * (-len(token) % 4)
        raw = base64.urlsafe_b64decode((token + pad).encode("ascii")).decode("utf-8")
        rows = json.loads(raw)
        if not rows:
            return pd.DataFrame(columns=["symbol","name","industry","market_value"])
        return clean_holdings(pd.DataFrame(rows))
    except Exception:
        return clean_holdings(pd.DataFrame(DEFAULT_HOLDINGS))

def _holdings_to_url(df):
    try:
        h = clean_holdings(df)
        if h is None or h.empty:
            rows = []
        else:
            cols = [c for c in ["symbol","name","industry","market_value"] if c in h.columns]
            rows = h[cols].to_dict("records")
        token = _encode_holdings_rows(rows)
        if st.query_params.get("holdings", "") != token:
            st.query_params["holdings"] = token
    except Exception:
        pass

"""
source = source.replace(marker, persist_helpers + marker)

source = source.replace(
    'if "holdings" not in st.session_state:\n'
    '    st.session_state.holdings = pd.DataFrame(columns=["symbol","name","industry","market_value"])',
    'if "holdings" not in st.session_state:\n'
    '    st.session_state.holdings = _holdings_from_url()'
)

source = source.replace(
    '    st.session_state.holdings = clean_holdings(edited) if edited is not None else st.session_state.holdings\n',
    '    st.session_state.holdings = clean_holdings(edited) if edited is not None else st.session_state.holdings\n'
    '    _holdings_to_url(st.session_state.holdings)\n'
    '    st.caption("✅ 持仓已自动保存；刷新页面会恢复。请勿分享带 holdings 参数的完整网址。")\n'
)

exec(compile(source, str(app_file), "exec"), globals(), globals())
