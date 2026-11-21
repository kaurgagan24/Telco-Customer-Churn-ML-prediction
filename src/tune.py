"""Hyper-parameter tuning with Optuna.

Unlike the notebook (which scored trials on the test set), trials are scored with
stratified cross-validation on the TRAINING data only, so the test set stays
untouched for the final evaluation.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from src.config import ensure_parent
from src.train import build_model

logger = logging.getLogger(__name__)

_SCORERS = {
    "recall": recall_score,
    "precision": precision_score,
    "f1": f1_score,
}


def suggest_params(trial: optuna.Trial, search_space: dict) -> dict:
    """Translate the YAML search space into Optuna suggestions."""
    params = {}
    for name, spec in search_space.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"])
        else:
            raise ValueError(f"Unsupported search-space type for {name}: {spec['type']}")
    return params


def make_objective(X_train: pd.DataFrame, y_train: pd.Series, cfg: dict):
    tcfg = cfg["tuning"]
    threshold = cfg["model"]["threshold"]
    base_params = dict(cfg["model"]["xgb_params"])
    scorer = _SCORERS[tcfg["metric"]]
    skf = StratifiedKFold(n_splits=tcfg["cv_folds"], shuffle=True, random_state=tcfg["seed"])

    def objective(trial: optuna.Trial) -> float:
        params = {**base_params, **suggest_params(trial, tcfg["search_space"])}
        scores = []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            model = build_model(params, y_tr)  # scale_pos_weight from the fold
            model.fit(X_tr, y_tr)
            preds = (model.predict_proba(X_va)[:, 1] >= threshold).astype(int)
            scores.append(scorer(y_va, preds, pos_label=1, zero_division=0))
        return float(np.mean(scores))

    return objective


def run_tuning(
    X_train: pd.DataFrame, y_train: pd.Series, cfg: dict, n_trials: int | None = None
) -> tuple[dict, float]:
    """Run Optuna, save best params to artifacts/best_params.json, return (params, score)."""
    n_trials = n_trials or cfg["tuning"]["n_trials"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg["tuning"]["seed"]),
    )
    study.optimize(make_objective(X_train, y_train, cfg), n_trials=n_trials)

    logger.info("Best CV %s: %.4f", cfg["tuning"]["metric"], study.best_value)
    logger.info("Best params: %s", study.best_params)

    out = ensure_parent(cfg["paths"]["best_params"])
    out.write_text(json.dumps(study.best_params, indent=2), encoding="utf-8")
    return study.best_params, study.best_value
