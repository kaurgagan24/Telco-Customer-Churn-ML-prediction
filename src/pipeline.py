"""Stage functions: preprocess -> tune -> train -> evaluate (or all)."""
from __future__ import annotations

import logging

from src.evaluate import evaluate_model, save_evaluation
from src.load_data import load_processed, load_raw_data, save_processed, validate_raw_data
from src.preprocess import (
    load_feature_columns,
    preprocess,
    save_feature_columns,
    split_data,
    split_features_target,
)
from src.train import (
    get_model_params,
    load_model,
    load_train_info,
    save_model,
    save_train_info,
    train_model,
)
from src.tune import run_tuning

logger = logging.getLogger(__name__)


def _get_splits(cfg: dict):
    """Load processed data and return the deterministic train/test split."""
    df = load_processed(cfg)
    X, y = split_features_target(df, cfg)
    return split_data(X, y, cfg)


def run_preprocess(cfg: dict) -> None:
    df = load_raw_data(cfg)
    validate_raw_data(df, cfg)
    df = preprocess(df, cfg)
    save_processed(df, cfg)
    X, _ = split_features_target(df, cfg)
    save_feature_columns(X.columns.tolist(), cfg)


def run_tune(cfg: dict, n_trials: int | None = None) -> None:
    X_train, _, y_train, _ = _get_splits(cfg)
    run_tuning(X_train, y_train, cfg, n_trials=n_trials)


def run_train(cfg: dict, use_tuned: bool = True) -> None:
    X_train, _, y_train, _ = _get_splits(cfg)
    params = get_model_params(cfg, use_tuned=use_tuned)
    model, train_time = train_model(X_train, y_train, params)
    # store the fully-resolved params (incl. scale_pos_weight) for reproducible logging
    params = {**params, "scale_pos_weight": model.get_params()["scale_pos_weight"]}
    save_model(model, cfg)
    save_train_info(params, train_time, cfg)


def run_evaluate(cfg: dict) -> dict:
    _, X_test, _, y_test = _get_splits(cfg)
    model = load_model(cfg)
    result = evaluate_model(model, X_test, y_test, cfg)
    save_evaluation(result, cfg)

    if cfg["mlflow"]["enabled"]:
        from src.tracking import log_run  # imported lazily so mlflow stays optional

        info = load_train_info(cfg)
        log_run(
            model,
            info.get("params", {}),
            result["metrics"],
            result["report"],
            info.get("train_time", float("nan")),
            cfg,
        )
    return result


def run_all(cfg: dict, tune: bool = True, n_trials: int | None = None) -> dict:
    run_preprocess(cfg)
    if tune:
        run_tune(cfg, n_trials=n_trials)
    run_train(cfg, use_tuned=tune)
    return run_evaluate(cfg)


# Re-exported for convenience (e.g. in the API under app/)
__all__ = [
    "run_preprocess", "run_tune", "run_train", "run_evaluate", "run_all",
    "load_feature_columns",
]
