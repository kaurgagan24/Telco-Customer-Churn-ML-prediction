import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.preprocess import preprocess, split_data, split_features_target


def test_preprocess_output_is_numeric_and_clean(raw_df, cfg):
    df = preprocess(raw_df, cfg)
    assert "customerID" not in df.columns
    assert all(is_numeric_dtype(df[c]) for c in df.columns)
    assert set(df["Churn"].unique()) <= {0, 1}
    assert "Contract_Two year" in df.columns
    assert "Contract_Month-to-month" not in df.columns  # drop_first


def test_blank_total_charges_become_nan(raw_df, cfg):
    df = preprocess(raw_df, cfg)
    assert df["TotalCharges"].isna().sum() == (raw_df["tenure"] == 0).sum()


def test_does_not_mutate_input(raw_df, cfg):
    before = raw_df.copy()
    preprocess(raw_df, cfg)
    pd.testing.assert_frame_equal(raw_df, before)


def test_inference_aligns_columns(raw_df, cfg):
    train = preprocess(raw_df, cfg)
    features = train.drop(columns=["Churn"]).columns.tolist()

    one_row = raw_df.drop(columns=["Churn"]).iloc[[0]]  # no target, few categories
    out = preprocess(one_row, cfg, feature_columns=features)
    assert out.columns.tolist() == features
    assert out.shape[0] == 1


def test_split_is_stratified_and_deterministic(raw_df, cfg):
    X, y = split_features_target(preprocess(raw_df, cfg), cfg)
    a = split_data(X, y, cfg)
    b = split_data(X, y, cfg)
    pd.testing.assert_frame_equal(a[1], b[1])
    assert abs(a[2].mean() - a[3].mean()) < 0.03
