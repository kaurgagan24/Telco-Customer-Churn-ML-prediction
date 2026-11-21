"""Evaluation: metrics, classification report and threshold sweep."""
from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import ensure_parent

logger = logging.getLogger(__name__)


def predict_proba_timed(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Positive-class probabilities and prediction time in seconds."""
    start = time.perf_counter()
    proba = model.predict_proba(X)[:, 1]
    return proba, time.perf_counter() - start


def compute_metrics(y_true, proba, threshold: float) -> dict:
    preds = (np.asarray(proba) >= threshold).astype(int)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, preds, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, preds, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, preds, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def threshold_sweep(y_true, proba, thresholds) -> pd.DataFrame:
    """Precision/recall/F1 of the churn class for each candidate threshold."""
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, proba, t)
        rows.append({k: m[k] for k in ("threshold", "precision", "recall", "f1")})
    return pd.DataFrame(rows)


def evaluate_model(model, X_test, y_test, cfg: dict) -> dict:
    """Evaluate at the configured threshold. Returns metrics + report + sweep."""
    threshold = cfg["model"]["threshold"]
    proba, pred_time = predict_proba_timed(model, X_test)

    metrics = compute_metrics(y_test, proba, threshold)
    metrics["pred_time"] = pred_time

    preds = (proba >= threshold).astype(int)
    report = classification_report(y_test, preds, digits=3, zero_division=0)
    sweep = threshold_sweep(y_test, proba, cfg["model"]["threshold_grid"])

    logger.info("Classification report:\n%s", report)
    logger.info("Threshold sweep:\n%s", sweep.round(3).to_string(index=False))
    return {"metrics": metrics, "report": report, "sweep": sweep}


def save_evaluation(result: dict, cfg: dict) -> None:
    metrics_path = ensure_parent(cfg["paths"]["metrics"])
    metrics_path.write_text(json.dumps(result["metrics"], indent=2), encoding="utf-8")
    result["sweep"].to_csv(ensure_parent(cfg["paths"]["threshold_sweep"]), index=False)
