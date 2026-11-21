import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """Small synthetic frame with the same schema as the Telco dataset."""
    rng = np.random.default_rng(0)
    n = 400
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 110, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n)
    logit = -0.05 * tenure + 0.01 * monthly + (contract == "Month-to-month") * 1.0 - 0.3
    churn = rng.random(n) < 1 / (1 + np.exp(-logit))

    total = (tenure * monthly).round(2).astype(str)
    total[tenure == 0] = " "  # blanks, like the real dataset

    return pd.DataFrame({
        "customerID": [f"ID-{i}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": tenure,
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "Contract": contract,
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)",
             "Credit card (automatic)"], n),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": np.where(churn, "Yes", "No"),
    })


@pytest.fixture()
def cfg(tmp_path) -> dict:
    """Real config, redirected to a temp dir with fast settings and MLflow off."""
    c = copy.deepcopy(load_config())
    for key, name in {
        "raw_data": "raw.csv", "processed_data": "processed.csv",
        "model": "model.joblib", "feature_columns": "features.json",
        "best_params": "best.json", "train_info": "train_info.json",
        "metrics": "metrics.json", "threshold_sweep": "sweep.csv",
    }.items():
        c["paths"][key] = str(tmp_path / name)
    c["model"]["xgb_params"].update(n_estimators=20, max_depth=3, n_jobs=1)
    c["tuning"].update(n_trials=2, cv_folds=2)
    c["tuning"]["search_space"]["n_estimators"] = {"type": "int", "low": 10, "high": 30}
    c["mlflow"]["enabled"] = False
    return c
