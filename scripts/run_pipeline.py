"""CLI entry point.

Examples
--------
    python scripts/run_pipeline.py                       # everything (tune + train + evaluate)
    python scripts/run_pipeline.py --no-tune             # skip Optuna, use config defaults
    python scripts/run_pipeline.py --stage preprocess
    python scripts/run_pipeline.py --stage tune --n-trials 50
    python scripts/run_pipeline.py --stage train
    python scripts/run_pipeline.py --stage evaluate
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.pipeline import (  # noqa: E402
    run_all,
    run_evaluate,
    run_preprocess,
    run_train,
    run_tune,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Telco churn pipeline")
    p.add_argument("--stage", default="all",
                   choices=["all", "preprocess", "tune", "train", "evaluate"])
    p.add_argument("--config", default=None, help="Path to a YAML config")
    p.add_argument("--n-trials", type=int, default=None, help="Override Optuna trials")
    p.add_argument("--no-tune", action="store_true", help="Skip tuning (stage=all / train)")
    p.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    cfg = load_config(args.config)
    if args.no_mlflow:
        cfg["mlflow"]["enabled"] = False

    if args.stage == "all":
        run_all(cfg, tune=not args.no_tune, n_trials=args.n_trials)
    elif args.stage == "preprocess":
        run_preprocess(cfg)
    elif args.stage == "tune":
        run_tune(cfg, n_trials=args.n_trials)
    elif args.stage == "train":
        run_train(cfg, use_tuned=not args.no_tune)
    elif args.stage == "evaluate":
        run_evaluate(cfg)


if __name__ == "__main__":
    main()
