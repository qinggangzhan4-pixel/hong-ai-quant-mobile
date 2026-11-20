# Streamlit Cloud 手机上传修复入口 v3
# 修复：
# 1) iPhone/GitHub 上传后 ai_quant/、examples/、models/ 文件夹被“拍平”
# 2) 空持仓时 clean_holdings() 返回 None 导致 .empty 报错
# 3) 云端 scikit-learn 与旧 model.joblib 不兼容（No module named '_loss'）
#    -> 云端先自动关闭旧AI模型，只运行规则量化；后续可在云端重新训练AI模型

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

# 注意：v3 不再把旧 model.joblib 复制到 models/，避免云端版本不兼容。
# 如果之前运行时已创建 models/model.joblib，也通过 app.py 源码补丁禁用它。

# ---------- 修复当前会话里已经变成 None 的 holdings ----------
try:
    import pandas as pd
    import streamlit as st
    if "holdings" in st.session_state and st.session_state.holdings is None:
        st.session_state.holdings = pd.DataFrame(
            columns=["symbol", "name", "industry", "market_value"]
        )
except Exception:
    pass

# ---------- 执行原 app.py，并在运行前打补丁 ----------
app_file = ROOT / "app.py"
if not app_file.exists():
    raise FileNotFoundError("仓库根目录没有 app.py，请先上传 app.py。")

source = app_file.read_text(encoding="utf-8")

# 修复 clean_holdings 空持仓返回 None
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

# 再加一道保险
source = source.replace(
    '    if not st.session_state.holdings.empty:\n',
    '    if st.session_state.holdings is not None and not st.session_state.holdings.empty:\n'
)

# 禁用旧的、不兼容的 joblib 模型文件。
# score_universe 检测不到模型文件时会自动退回规则评分，AI概率显示“未启用”。
source = source.replace(
    'MODEL_PATH = ROOT / "models" / "model.joblib"',
    'MODEL_PATH = ROOT / "models" / "__cloud_ai_disabled_until_retrain__.joblib"'
)

exec(compile(source, str(app_file), "exec"), globals(), globals())
