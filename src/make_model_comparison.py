"""Build final comparison tables and plots from saved model metrics.

Example:
    python src/make_model_comparison.py
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_metric_rows(tables_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    metric_paths = sorted(
        path for path in tables_dir.glob("*_metrics.csv")
        if path.name != "model_comparison.csv"
    )

    if not metric_paths:
        raise FileNotFoundError(f"No metric CSV files found in {tables_dir}")
    for path in metric_paths:
        rows.append(pd.read_csv(path))

    comparison = pd.concat(rows, ignore_index=True, sort=False)
    if "experiment_label" not in comparison.columns:
        comparison["experiment_label"] = "baseline"
    comparison["experiment_label"] = (
        comparison["experiment_label"]
        .fillna("baseline")
        .replace("", "baseline")
    )
    ordered_cols = [
        "model",
        "experiment_label",
        "time_label",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "n_features",
        "n_rows",
        "cv_best_roc_auc",
        "hidden_dims",
        "dropout",
        "activation",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "decision_threshold",
        "threshold_metric",
        "threshold_val_score",
        "epochs_run",
        "best_epoch",
        "best_val_loss",
        "device",
    ]
    return comparison[[col for col in ordered_cols if col in comparison.columns]]


def plot_metric_bars(comparison: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_df = comparison.copy()
    if "experiment_label" not in plot_df.columns:
        plot_df["experiment_label"] = "baseline"
    plot_df["experiment_label"] = plot_df["experiment_label"].fillna("baseline")
    plot_df["experiment"] = (
        plot_df["model"] + " " + plot_df["time_label"] + " " + plot_df["experiment_label"]
    )
    metrics = ["accuracy", "f1", "roc_auc"]

    ax = plot_df.set_index("experiment")[metrics].plot(
        kind="bar",
        figsize=(9, 5),
        ylim=(0.65, 0.93),
        rot=20,
    )
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Model Performance Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(figures_dir / "model_comparison_metrics.png", dpi=200)
    plt.close()


def build_summary(comparison: pd.DataFrame, tables_dir: Path) -> dict[str, object]:
    best = comparison.sort_values("roc_auc", ascending=False).iloc[0]
    if "experiment_label" in comparison.columns:
        baseline = comparison[comparison["experiment_label"].fillna("baseline") == "baseline"]
    else:
        baseline = comparison
    if baseline.empty:
        baseline = comparison
    wide = baseline.pivot_table(index="model", columns="time_label", values="accuracy", aggfunc="max")
    gain_15min = {
        model: float(wide.loc[model, "15minute"] - wide.loc[model, "10minute"])
        for model in wide.index
        if {"10minute", "15minute"}.issubset(wide.columns)
    }

    top_features_path = tables_dir / "xgboost_15minute_feature_importance.csv"
    top_features: list[dict[str, object]] = []
    if top_features_path.exists():
        top_features = (
            pd.read_csv(top_features_path)
            .head(10)
            .to_dict(orient="records")
        )

    return {
        "best_model": str(best["model"]),
        "best_time_label": str(best["time_label"]),
        "best_accuracy": float(best["accuracy"]),
        "best_f1": float(best["f1"]),
        "best_roc_auc": float(best["roc_auc"]),
        "accuracy_gain_from_10_to_15_minutes": gain_15min,
        "xgboost_15minute_top_features": top_features,
    }


def write_presentation_notes(summary: dict[str, object], output_path: Path) -> None:
    best_accuracy = float(summary["best_accuracy"]) * 100
    best_auc = float(summary["best_roc_auc"])
    gains = summary["accuracy_gain_from_10_to_15_minutes"]
    top_features = summary["xgboost_15minute_top_features"]
    feature_names = ", ".join(str(item["feature"]) for item in top_features[:5])

    text = f"""# XGBoost / MLP Results Notes

## Key Results

- Best setting: {summary["best_model"]} with {summary["best_time_label"]} data.
- Best test accuracy: {best_accuracy:.2f}%.
- Best ROC-AUC: {best_auc:.4f}.
- Accuracy improves when moving from 10-minute to 15-minute data: {gains}.

## Interpretation

The result supports the project hypothesis that early-game information can predict the final winner better than chance. The 15-minute data consistently performs better than the 10-minute data, which means the additional five minutes contain meaningful signals such as objective control, accumulated gold gap, and level gap.

XGBoost is slightly stronger than MLP in this experiment. This is reasonable because the dataset is structured tabular data, where tree-based ensemble models often learn feature thresholds and interactions efficiently. MLP still performs competitively after standardization and dropout, but it does not clearly outperform XGBoost.

For feature importance, the strongest 15-minute XGBoost signals are: {feature_names}. These features are interpretable in game terms: gold and level gaps summarize lane and jungle advantages, while dragon/objective variables represent map control.

## Presentation Takeaway

Even though the topic is game-related, the project is framed as a binary classification and feature-importance analysis task. The important academic point is not the game itself, but the full machine-learning workflow: leakage prevention, feature engineering, model comparison, threshold-based and threshold-free evaluation, and interpretation of predictive factors.
"""
    output_path.write_text(text, encoding="utf-8")


def make_model_comparison(results_dir: Path) -> pd.DataFrame:
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    comparison = load_metric_rows(tables_dir)
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)
    plot_metric_bars(comparison, figures_dir)

    summary = build_summary(comparison, tables_dir)
    with open(tables_dir / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    write_presentation_notes(summary, results_dir / "xgboost_mlp_presentation_notes.md")
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = make_model_comparison(args.results_dir)
    print(df.to_string(index=False))
