"""FastAPI service for Telco customer churn prediction.

Run locally (from the project root):
    uvicorn app.main:app --reload --port 8000

Docs:   http://localhost:8000/docs
Health: http://localhost:8000/health
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from src.config import load_config, resolve_path
from src.preprocess import load_feature_columns, preprocess, split_data
from src.train import load_model

logger = logging.getLogger("churn_api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

YesNo = Literal["Yes", "No"]
YesNoNoInternet = Literal["Yes", "No", "No internet service"]


# ---------------------------------------------------------------------------
# 1) Application initialisation: the model is loaded once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cfg = None
    app.state.model = None
    app.state.feature_columns = None
    try:
        cfg = load_config(os.getenv("CONFIG_PATH"))  # optional override via env var
        app.state.cfg = cfg
        app.state.model = load_model(cfg)
        app.state.feature_columns = load_feature_columns(cfg)
        logger.info("Model and feature columns loaded (%d features)",
                    len(app.state.feature_columns))
    except Exception:
        # Keep the process alive so /health can report 503 instead of crash-looping.
        logger.exception("Failed to load model artifacts - run the pipeline first")
    yield


app = FastAPI(
    title="Telco Customer Churn API",
    description="Predicts the probability that a telecom customer will churn.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 3) Request / response schemas (validation + automatic OpenAPI docs)
# ---------------------------------------------------------------------------
class CustomerData(BaseModel):
    """One customer, using the same fields and values as the raw Telco dataset."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 5,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 85.5,
                "TotalCharges": 420.75,
            }
        },
    )

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1] = Field(description="1 if the customer is a senior citizen")
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=100, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: YesNoNoInternet
    OnlineBackup: YesNoNoInternet
    DeviceProtection: YesNoNoInternet
    TechSupport: YesNoNoInternet
    StreamingTV: YesNoNoInternet
    StreamingMovies: YesNoNoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Optional[float] = Field(
        default=None, ge=0, description="Leave empty for brand-new customers (tenure 0)"
    )


class PredictionResponse(BaseModel):
    churn_probability: float = Field(description="Model probability of churn, 0-1")
    churn_prediction: Literal["Yes", "No"]
    threshold: float = Field(description="Probability cut-off used for the decision")


# ---------------------------------------------------------------------------
# 2) Health check endpoint (point the AWS ALB target-group health check here)
# ---------------------------------------------------------------------------
@app.get("/health", tags=["Health"])
def health(request: Request):
    """Returns 200 when the service is up and the model is loaded, otherwise 503."""
    if request.app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


# ---------------------------------------------------------------------------
# 4) Prediction endpoint
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerData, request: Request):
    """Predict churn for a single customer."""
    state = request.app.state
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        df = pd.DataFrame([customer.model_dump()])
        X = preprocess(df, state.cfg, feature_columns=state.feature_columns)
        proba = float(state.model.predict_proba(X)[0, 1])
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")

    threshold = state.cfg["model"]["threshold"]
    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction="Yes" if proba >= threshold else "No",
        threshold=threshold,
    )


@app.get("/analytics", tags=["Analytics"])
def analytics(request: Request):
    """Read aggregate reporting data from the configured pipeline outputs."""
    cfg = request.app.state.cfg
    if cfg is None:
        raise HTTPException(503, "Pipeline configuration unavailable")
    paths = cfg["paths"]

    def read_json(key):
        return json.loads(resolve_path(paths[key]).read_text(encoding="utf-8"))

    try:
        frame = pd.read_csv(resolve_path(paths["processed_data"]))
        target = frame[cfg["data"]["target"]]
        metrics = read_json("metrics")
        training = read_json("train_info")
        sweep = pd.read_csv(resolve_path(paths["threshold_sweep"]))
        best = read_json("best_params") if resolve_path(paths["best_params"]).is_file() else None
        accuracy_source = "Reported in metrics.json"
        if metrics.get("accuracy") is None:
            # Recover counts only when the saved rates match integer counts on
            # the configured stratified holdout to floating-point precision.
            _, _, _, holdout = split_data(frame.index, target, cfg)
            positives = int(holdout.sum())
            tp = round(metrics["recall"] * positives)
            flagged = round(tp / metrics["precision"]) if metrics["precision"] else 0
            tn = len(holdout) - positives - (flagged - tp)
            if (flagged and positives and 0 <= tp <= min(positives, flagged)
                    and tn >= 0 and abs(tp / flagged - metrics["precision"]) < 1e-10
                    and abs(tp / positives - metrics["recall"]) < 1e-10):
                metrics["accuracy"] = (tp + tn) / len(holdout)
                accuracy_source = f"Derived from saved precision and recall on the configured {len(holdout):,}-customer holdout."
            else:
                accuracy_source = "Unavailable in the evaluation artifacts."
        one_year = int(frame["Contract_One year"].sum())
        two_year = int(frame["Contract_Two year"].sum())
        return {
            "metrics": metrics, "accuracy_source": accuracy_source,
            "customers": len(frame), "churned": int(target.sum()),
            "contracts": {"Month-to-month": len(frame) - one_year - two_year,
                          "One year": one_year, "Two year": two_year},
            "mean_tenure": frame["tenure"].mean(),
            "mean_monthly": frame["MonthlyCharges"].mean(),
            "threshold": cfg["model"]["threshold"], "training": training,
            "best_params": best, "sweep": sweep.sort_values("threshold").to_dict("records"),
        }
    except (OSError, KeyError, ValueError):
        logger.exception("Unable to read analytics artifacts")
        raise HTTPException(503, "Analytics unavailable. Check the configured pipeline artifacts.")


@app.get("/", include_in_schema=False)
def dashboard():
    return RedirectResponse("/dashboard/")


app.mount("/dashboard", StaticFiles(directory=Path(__file__).parent / "frontend", html=True),
          name="dashboard")
