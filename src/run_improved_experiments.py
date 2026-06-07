"""Run add-on experiments without overwriting the baseline result files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.make_model_comparison import make_model_comparison
from src.train_mlp import train_mlp


DATASETS = [
    ("10minute", PROJECT_ROOT / "data" / "Challenger_Ranked_Games_10minute.csv"),
    ("15minute", PROJECT_ROOT / "data" / "Challenger_Ranked_Games_15minute.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--skip-ensemble", action="store_true")
    parser.add_argument("--experiment-label", type=str, default="threshold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.skip_xgboost:
        try:
            from src.train_xgboost import train_xgboost
        except ImportError as exc:
            raise SystemExit(
                "XGBoost is not installed. Install dependencies with "
                "`pip install -r requirements.txt` or rerun with `--skip-xgboost`."
            ) from exc

        for time_label, data_file in DATASETS:
            train_xgboost(
                data_file=data_file,
                time_label=time_label,
                output_dir=args.output_dir,
                model_dir=args.model_dir,
                random_state=args.random_state,
                tune=True,
                quick=args.quick,
                tune_threshold=True,
                threshold_metric="accuracy",
                experiment_label=args.experiment_label,
            )

    if not args.skip_mlp:
        for time_label, data_file in DATASETS:
            train_mlp(
                data_file=data_file,
                time_label=time_label,
                output_dir=args.output_dir,
                model_dir=args.model_dir,
                random_state=args.random_state,
                quick=args.quick,
                tune_threshold=True,
                threshold_metric="accuracy",
                experiment_label=args.experiment_label,
            )

    if not args.skip_ensemble:
        try:
            from src.make_ensemble import make_ensemble

            for time_label, _ in DATASETS:
                make_ensemble(
                    results_dir=args.output_dir,
                    time_label=time_label,
                    experiment_label=args.experiment_label,
                    xgb_weight=0.65,
                    threshold_metric="accuracy",
                )
        except FileNotFoundError:
            print("Skipping ensemble because prediction files are not all available.")

    make_model_comparison(args.output_dir)


if __name__ == "__main__":
    main()
