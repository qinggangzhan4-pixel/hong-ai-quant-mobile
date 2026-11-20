"""
Streamlit Cloud 手机上传修复入口
用途：当 iPhone/GitHub 上传把 ai_quant/、examples/、models/ 文件夹“拍平”到仓库根目录时，
此文件在运行时恢复包结构，让原 app.py 无需改动也能启动。
"""
from pathlib import Path
import sys
import types
import shutil

ROOT = Path(__file__).resolve().parent

# 1) 把仓库根目录临时当成 ai_quant 包目录。
#    这样 root/config.py 会被解释为 ai_quant.config，
#    root/engine.py 中的 .features 等相对导入也能正常工作。
if "ai_quant" not in sys.modules:
    pkg = types.ModuleType("ai_quant")
    pkg.__path__ = [str(ROOT)]
    pkg.__package__ = "ai_quant"
    sys.modules["ai_quant"] = pkg

# 2) iPhone 上传常把 examples/ 和 models/ 也拍平。
#    原 app.py 会按这些目录读取文件，因此启动时自动重建目录并复制对应文件。
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

for name in ("model.joblib", "model.joblib.json"):
    ensure_dir_file("models", name)

# 3) 执行原始 Streamlit app.py。
app_file = ROOT / "app.py"
if not app_file.exists():
    raise FileNotFoundError("仓库根目录没有 app.py，请先上传 app.py。")

source = app_file.read_text(encoding="utf-8")
exec(compile(source, str(app_file), "exec"), globals(), globals())
