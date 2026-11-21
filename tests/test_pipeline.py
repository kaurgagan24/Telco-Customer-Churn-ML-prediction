import json
from pathlib import Path

from src.pipeline import run_all


def test_end_to_end(raw_df, cfg):
    raw_df.to_csv(cfg["paths"]["raw_data"], index=False)
    result = run_all(cfg, tune=True, n_trials=2)

    for key in ("processed_data", "model", "feature_columns", "best_params",
                "train_info", "metrics", "threshold_sweep"):
        assert Path(cfg["paths"][key]).exists(), key

    metrics = json.loads(Path(cfg["paths"]["metrics"]).read_text())
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    assert result["metrics"]["threshold"] == cfg["model"]["threshold"]
