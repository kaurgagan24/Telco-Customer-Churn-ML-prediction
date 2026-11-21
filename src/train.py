"""Model construction, training and persistence."""
from __future__ import annotations

import json
import logging
import time

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.config import ensure_parent, resolve_path

logger = logging.getLogger(__name__)


def compute_scale_pos_weight(y: pd.Series) -> float:
    """negatives / positives - compensates for class imbalance."""
    return float((y == 0).sum() / (y == 1).sum())


def get_model_params(cfg: dict, use_tuned: bool = True) -> dict:
    """Base params from config, overridden by tuned params if artifacts/best_params.json exists."""
    params = dict(cfg["model"]["xgb_params"])
    best_path = resolve_path(cfg["paths"]["best_params"])
    if use_tuned and best_path.exists():
        params.update(json.loads(best_path.read_text(encoding="utf-8")))
        logger.info("Using tuned parameters from %s", best_path)
    else:
        logger.info("Using default parameters from config")
    return params


def build_model(params: dict, y_train: pd.Series) -> XGBClassifier:
    params = dict(params)
    params.setdefault("scale_pos_weight", compute_scale_pos_weight(y_train))
    return XGBClassifier(**params)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, params: dict):
    """Fit an XGBClassifier. Returns (model, train_time_seconds)."""
    model = build_model(params, y_train)
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    logger.info("Training time: %.2fs", train_time)
    return model, train_time


def save_model(model: XGBClassifier, cfg: dict) -> None:
    out = ensure_parent(cfg["paths"]["model"])
    joblib.dump(model, out)
    logger.info("Saved model -> %s", out)


def load_model(cfg: dict) -> XGBClassifier:
    path = resolve_path(cfg["paths"]["model"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the 'train' stage first.")
    return joblib.load(path)


def save_train_info(params: dict, train_time: float, cfg: dict) -> None:
    """Persist the exact params + train time so the evaluate stage can log them."""
    info = {
        "train_time": train_time,
        "params": {k: (float(v) if hasattr(v, "item") else v) for k, v in params.items()},
    }
    out = ensure_parent(cfg["paths"]["train_info"])
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_train_info(cfg: dict) -> dict:
    path = resolve_path(cfg["paths"]["train_info"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
