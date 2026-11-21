import numpy as np

from src.evaluate import compute_metrics, threshold_sweep


def test_perfect_predictions():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.9, 0.8])
    m = compute_metrics(y, proba, threshold=0.5)
    assert m["precision"] == m["recall"] == m["f1"] == m["roc_auc"] == 1.0


def test_lower_threshold_never_lowers_recall():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    proba = np.clip(y * 0.3 + rng.random(200) * 0.7, 0, 1)
    sweep = threshold_sweep(y, proba, [0.25, 0.5, 0.75])
    assert sweep["recall"].is_monotonic_decreasing
    assert list(sweep.columns) == ["threshold", "precision", "recall", "f1"]
