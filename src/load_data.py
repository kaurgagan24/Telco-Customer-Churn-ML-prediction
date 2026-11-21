"""Load and validate the raw Telco churn data."""
from __future__ import annotations

import logging

import pandas as pd

from src.config import ensure_parent, resolve_path

logger = logging.getLogger(__name__)


def load_raw_data(cfg: dict, path: str | None = None) -> pd.DataFrame:
    """Read the raw CSV. `path` overrides cfg['paths']['raw_data']."""
    csv_path = resolve_path(path or cfg["paths"]["raw_data"])
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw data not found: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Loaded %s rows x %s cols from %s", *df.shape, csv_path)
    return df


def validate_raw_data(df: pd.DataFrame, cfg: dict) -> None:
    """Fail fast on missing columns; warn on duplicate customer IDs."""
    data_cfg = cfg["data"]
    required = {data_cfg["target"], *data_cfg["binary_columns"], *data_cfg["numeric_coerce"]}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing)}")

    id_col = data_cfg["id_column"]
    if id_col in df.columns:
        n_dup = int(df[id_col].duplicated().sum())
        if n_dup:
            logger.warning("Found %d duplicated %s values", n_dup, id_col)


def save_processed(df: pd.DataFrame, cfg: dict) -> None:
    out = ensure_parent(cfg["paths"]["processed_data"])
    df.to_csv(out, index=False)
    logger.info("Saved processed data -> %s", out)


def load_processed(cfg: dict) -> pd.DataFrame:
    path = resolve_path(cfg["paths"]["processed_data"])
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the 'preprocess' stage first.")
    return pd.read_csv(path)
