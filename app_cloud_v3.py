# Streamlit Cloud compatibility entry
from pathlib import Path
import sys
import types
import shutil
import importlib

ROOT = Path(__file__).resolve().parent

# Streamlit 热更新时 Python 进程可能继续缓存旧模块。
# 每次脚本重跑都清掉业务子模块，确保 GitHub 最新修复真正生效。
for _mod in [
    "ai_quant.engine",
    "ai_quant.features",
    "ai_quant.scoring",
    "ai_quant.relay",
    "ai_quant.risk",
    "ai_quant.ml_model",
    "ai_quant.data",
    "ai_quant.config",
]:
    sys.modules.pop(_mod, None)

# 手机上传把 ai_quant/ 目录拍平后，让仓库根目录继续充当 ai_quant 包。
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

# 恢复演示数据目录；app.py 已负责持仓持久化和完整持仓字段。
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

# 云端暂时跳过旧版 sklearn 训练出的 joblib，规则量化照常运行。
source = source.replace(
    'MODEL_PATH = ROOT / "models" / "model.joblib"',
    'MODEL_PATH = ROOT / "models" / "__cloud_ai_disabled_until_retrain__.joblib"'
)

exec(compile(source, str(app_file), "exec"), globals(), globals())
