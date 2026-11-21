import json

from src.preprocess import preprocess, split_data, split_features_target
from src.tune import run_tuning


def test_run_tuning_saves_params_in_bounds(raw_df, cfg):
    X, y = split_features_target(preprocess(raw_df, cfg), cfg)
    X_train, _, y_train, _ = split_data(X, y, cfg)

    best, score = run_tuning(X_train, y_train, cfg)
    space = cfg["tuning"]["search_space"]
    assert set(best) == set(space)
    for k, v in best.items():
        assert space[k]["low"] <= v <= space[k]["high"]
    assert 0.0 <= score <= 1.0

    saved = json.loads(open(cfg["paths"]["best_params"]).read())
    assert saved == best
