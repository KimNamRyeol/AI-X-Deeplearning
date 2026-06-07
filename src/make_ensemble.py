"""Create a soft-voting ensemble from saved XGBoost and MLP predictions.

Example:
    python src/make_ensemble.py --time-label 15minute --experiment-label threshold
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

from src.metrics_utils import threshold_predictions
from src.train_mlp import evaluate_binary_classifier


def plot_confusion_matrix(y_true, y_pred, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Lose", "Win"])
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_roc_curve(y_true, y_proba, path: Path, title: str) -> None:
    RocCurveDisplay.from_predictions(y_true, y_proba)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_ensemble(
    results_dir: Path,
    time_label: str,
    experiment_label: str,
    xgb_weight: float = 0.65,
    threshold_metric: str = "accuracy",
) -> dict[str, float]:
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{experiment_label}" if experiment_label else ""
    xgb_path = tables_dir / f"xgboost_{time_label}{suffix}_predictions.csv"
    mlp_path = tables_dir / f"mlp_{time_label}{suffix}_predictions.csv"
    if not xgb_path.exists() or not mlp_path.exists():
        raise FileNotFoundError(
            "Prediction files are required before ensembling. Missing: "
            f"{xgb_path if not xgb_path.exists() else ''} "
            f"{mlp_path if not mlp_path.exists() else ''}"
        )

    xgb = pd.read_csv(xgb_path)
    mlp = pd.read_csv(mlp_path)
    if not xgb["y_true"].equals(mlp["y_true"]):
        raise ValueError("XGBoost and MLP prediction files do not share the same test labels/order.")

    y_true = xgb["y_true"]
    y_proba = xgb_weight * xgb["y_proba"] + (1.0 - xgb_weight) * mlp["y_proba"]
    threshold = 0.5
    y_pred = threshold_predictions(y_proba, threshold)

    metrics = evaluate_binary_classifier(y_true, y_pred, y_proba)
    metrics.update({
        "model": "SoftVotingEnsemble",
        "experiment_label": experiment_label or "baseline",
        "time_label": time_label,
        "xgboost_weight": xgb_weight,
        "mlp_weight": 1.0 - xgb_weight,
        "decision_threshold": threshold,
        "threshold_metric": "fixed_0.5",
        "threshold_val_score": None,
    })

    prefix = f"ensemble_{time_label}{suffix}"
    pd.DataFrame([metrics]).to_csv(tables_dir / f"{prefix}_metrics.csv", index=False)
    pd.DataFrame({
        "y_true": y_true,
        "y_proba": y_proba,
        "y_pred": y_pred,
    }).to_csv(tables_dir / f"{prefix}_predictions.csv", index=False)
    with open(tables_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics}, f, indent=2, ensure_ascii=False)

    plot_confusion_matrix(
        y_true,
        y_pred,
        figures_dir / f"{prefix}_confusion_matrix.png",
        f"Soft Voting Ensemble Confusion Matrix ({time_label})",
    )
    plot_roc_curve(
        y_true,
        y_proba,
        figures_dir / f"{prefix}_roc_curve.png",
        f"Soft Voting Ensemble ROC Curve ({time_label})",
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--time-label", type=str, required=True)
    parser.add_argument("--experiment-label", type=str, default="")
    parser.add_argument("--xgb-weight", type=float, default=0.65)
    parser.add_argument("--threshold-metric", type=str, default="accuracy", choices=["accuracy", "f1"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = make_ensemble(
        results_dir=args.results_dir,
        time_label=args.time_label,
        experiment_label=args.experiment_label,
        xgb_weight=args.xgb_weight,
        threshold_metric=args.threshold_metric,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
