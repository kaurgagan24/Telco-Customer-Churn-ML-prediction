"""MLflow experiment tracking (SQLite backend, same as the notebook)."""
from __future__ import annotations

import logging

import mlflow
import mlflow.xgboost

from src.config import resolve_path

logger = logging.getLogger(__name__)


def setup_mlflow(cfg: dict) -> None:
    db_path = resolve_path(cfg["mlflow"]["db_path"])
    mlflow.set_tracking_uri("sqlite:///" + db_path.as_posix())
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])


def log_run(model, params: dict, metrics: dict, report: str, train_time: float, cfg: dict) -> None:
    """Log params, metrics, classification report and the model to MLflow."""
    setup_mlflow(cfg)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("threshold", cfg["model"]["threshold"])
        mlflow.log_metric("train_time", train_time)
        mlflow.log_metrics(
            {k: float(v) for k, v in metrics.items() if k != "threshold"}
        )
        mlflow.log_text(report, "classification_report.txt")
        try:  # MLflow >= 3
            mlflow.xgboost.log_model(model, name="model")
        except TypeError:  # MLflow 2.x
            mlflow.xgboost.log_model(model, artifact_path="model")
    logger.info("MLflow run logged (experiment: %s)", cfg["mlflow"]["experiment_name"])
