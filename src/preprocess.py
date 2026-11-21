"""Cleaning, encoding and train/test splitting."""
from __future__ import annotations

import json
import logging

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import ensure_parent, resolve_path

logger = logging.getLogger(__name__)


def clean_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop the ID column, coerce numeric-as-string columns, cast bools to int."""
    df = df.copy()
    data_cfg = cfg["data"]

    df = df.drop(columns=[data_cfg["id_column"]], errors="ignore")

    for col in data_cfg["numeric_coerce"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def encode_binary(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Map Yes/No and Male/Female to 1/0 for the configured binary columns."""
    df = df.copy()
    mapping = cfg["data"]["binary_mapping"]
    for col in cfg["data"]["binary_columns"]:
        if col not in df.columns:  # e.g. target is absent at inference time
            continue
        encoded = df[col].map(lambda v: mapping.get(v, v))
        df[col] = pd.to_numeric(encoded, errors="raise").astype(int)
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all remaining non-numeric columns (drop_first=True)."""
    cat_cols = df.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    logger.info("One-hot encoding: %s", cat_cols)
    return pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)


def preprocess(
    df: pd.DataFrame, cfg: dict, feature_columns: list[str] | None = None
) -> pd.DataFrame:
    """Full preprocessing.

    Training: call with feature_columns=None.
    Inference: pass the saved training feature list so the dummy columns line up
    (missing dummies are filled with 0, unseen ones are dropped).
    """
    df = clean_data(df, cfg)
    df = encode_binary(df, cfg)
    df = encode_categorical(df)

    if feature_columns is not None:
        target = cfg["data"]["target"]
        extra = [target] if target in df.columns else []
        df = df.reindex(columns=feature_columns + extra, fill_value=0)
    return df


def split_features_target(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    target = cfg["data"]["target"]
    return df.drop(columns=[target]), df[target]


def split_data(X: pd.DataFrame, y: pd.Series, cfg: dict):
    """Stratified train/test split (deterministic via config random_state)."""
    s = cfg["split"]
    return train_test_split(
        X, y, test_size=s["test_size"], random_state=s["random_state"], stratify=y
    )


def save_feature_columns(columns: list[str], cfg: dict) -> None:
    out = ensure_parent(cfg["paths"]["feature_columns"])
    out.write_text(json.dumps(list(columns), indent=2), encoding="utf-8")


def load_feature_columns(cfg: dict) -> list[str]:
    return json.loads(resolve_path(cfg["paths"]["feature_columns"]).read_text(encoding="utf-8"))
