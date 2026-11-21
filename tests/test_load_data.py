import pytest

from src.load_data import load_raw_data, validate_raw_data


def test_load_raw_data(raw_df, cfg):
    raw_df.to_csv(cfg["paths"]["raw_data"], index=False)
    df = load_raw_data(cfg)
    assert df.shape == raw_df.shape


def test_load_missing_file_raises(cfg):
    with pytest.raises(FileNotFoundError):
        load_raw_data(cfg)


def test_validate_missing_column_raises(raw_df, cfg):
    with pytest.raises(ValueError, match="Churn"):
        validate_raw_data(raw_df.drop(columns=["Churn"]), cfg)


def test_validate_ok(raw_df, cfg):
    validate_raw_data(raw_df, cfg)  # should not raise
