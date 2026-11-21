from src.preprocess import preprocess, split_data, split_features_target
from src.train import (compute_scale_pos_weight, get_model_params, load_model,
                       save_model, train_model)


def _splits(raw_df, cfg):
    X, y = split_features_target(preprocess(raw_df, cfg), cfg)
    return split_data(X, y, cfg)


def test_scale_pos_weight(raw_df, cfg):
    _, _, y_train, _ = _splits(raw_df, cfg)
    expected = (y_train == 0).sum() / (y_train == 1).sum()
    assert compute_scale_pos_weight(y_train) == expected


def test_train_predict_and_roundtrip(raw_df, cfg):
    X_train, X_test, y_train, _ = _splits(raw_df, cfg)
    model, t = train_model(X_train, y_train, get_model_params(cfg))
    assert t >= 0
    proba = model.predict_proba(X_test)[:, 1]
    assert proba.shape[0] == len(X_test) and ((proba >= 0) & (proba <= 1)).all()

    save_model(model, cfg)
    reloaded = load_model(cfg)
    assert (reloaded.predict_proba(X_test)[:, 1] == proba).all()
