from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score
from .features import MODEL_FEATURES

@dataclass
class ModelInfo:
    enabled: bool
    auc: float | None
    accuracy: float | None
    samples: int
    reason: str


def add_target(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.03) -> pd.DataFrame:
    x = df.copy().sort_values(["symbol", "date"])
    targets = []
    for _, g in x.groupby("symbol", sort=False):
        g = g.copy()
        shifted = pd.concat([g["high"].shift(-i) for i in range(1, horizon + 1)], axis=1)
        future_highs = shifted.max(axis=1)
        full_horizon = shifted.notna().all(axis=1)
        g["future_max_return"] = future_highs / g["close"] - 1
        g["target"] = (g["future_max_return"] >= threshold).astype(float)
        g.loc[~full_horizon, "target"] = np.nan
        targets.append(g)
    return pd.concat(targets, ignore_index=True)


def _date_split(data: pd.DataFrame, train_frac: float = 0.75):
    dates = np.array(sorted(data["date"].dropna().unique()))
    cut = max(1, int(len(dates) * train_frac))
    train_dates = set(dates[:cut])
    test_dates = set(dates[cut:])
    return data[data["date"].isin(train_dates)], data[data["date"].isin(test_dates)]


def train_model(feature_df: pd.DataFrame, config: dict, model_path: str) -> ModelInfo:
    scfg = config["strategy"]
    data = add_target(feature_df, scfg["target_horizon_days"], scfg["target_return_threshold"])
    use_cols = [c for c in MODEL_FEATURES if c in data.columns and not data[c].isna().all()]
    data = data.dropna(subset=["target"]).copy()
    if len(data) < scfg["ml_min_samples"]:
        return ModelInfo(False, None, None, len(data), "样本不足，ML层自动关闭")
    train, test = _date_split(data)
    if len(test) < 50 or train["target"].nunique() < 2 or test["target"].nunique() < 2:
        return ModelInfo(False, None, None, len(data), "时间切分后样本/类别不足")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=42))
    ])
    pipe.fit(train[use_cols], train["target"].astype(int))
    prob = pipe.predict_proba(test[use_cols])[:, 1]
    pred = (prob >= 0.5).astype(int)
    auc = float(roc_auc_score(test["target"].astype(int), prob))
    acc = float(accuracy_score(test["target"].astype(int), pred))
    enabled = auc >= scfg["ml_min_auc"]
    reason = "通过样本外门槛" if enabled else f"样本外AUC {auc:.3f} 低于门槛 {scfg['ml_min_auc']:.2f}，ML层自动关闭"
    payload = {"pipeline": pipe, "features": use_cols, "enabled": enabled, "auc": auc, "accuracy": acc}
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path)
    Path(str(model_path) + ".json").write_text(json.dumps({k:v for k,v in payload.items() if k != "pipeline"}, ensure_ascii=False, indent=2), encoding="utf-8")
    return ModelInfo(enabled, auc, acc, len(data), reason)


def load_model(model_path: str):
    p = Path(model_path)
    if not p.exists():
        return None
    return joblib.load(p)


def predict_probability(model_payload, rows: pd.DataFrame) -> np.ndarray:
    if model_payload is None or not model_payload.get("enabled", False):
        return np.full(len(rows), np.nan)
    cols = model_payload["features"]
    return model_payload["pipeline"].predict_proba(rows[cols])[:, 1]
