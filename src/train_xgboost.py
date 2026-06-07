"""Train and evaluate an XGBoost classifier for LoL win prediction.

Example:
    python src/train_xgboost.py \
        --data-file data/Challenger_Ranked_Games_10minute.csv \
        --time-label 10minute
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBClassifier

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import load_prepare_split, summarize_dataset, load_dataset
from src.metrics_utils import find_best_threshold, threshold_predictions


def evaluate_binary_classifier(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def make_xgb_model(random_state: int = 42, **kwargs) -> XGBClassifier:
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,
    )
    params.update(kwargs)
    return XGBClassifier(**params)


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


def plot_feature_importance(model: XGBClassifier, feature_names: list[str], path: Path, top_n: int = 20) -> pd.DataFrame:
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    top = importance.head(top_n).sort_values("importance")
    plt.figure(figsize=(8, max(4, top_n * 0.3)))
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return importance


def train_xgboost(
    data_file: Path,
    time_label: str,
    output_dir: Path,
    model_dir: Path,
    random_state: int = 42,
    tune: bool = True,
    quick: bool = False,
    tune_threshold: bool = False,
    threshold_metric: str = "accuracy",
    experiment_label: str = "",
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_file)
    summary = summarize_dataset(df)
    X_train, X_test, y_train, y_test, feature_names = load_prepare_split(data_file, random_state=random_state)

    base_model = make_xgb_model(
        random_state=random_state,
        n_estimators=200 if not quick else 80,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
    )

    # quick=True is intended as a smoke test, so skip expensive hyperparameter search.
    if quick:
        tune = False

    if tune_threshold:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.15,
            random_state=random_state,
            stratify=y_train,
        )
    else:
        X_fit, y_fit = X_train, y_train
        X_val = y_val = None

    if tune:
        param_dist = {
            "n_estimators": [200, 300, 500, 800] if not quick else [50, 80, 100],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.75, 0.85, 1.0],
            "colsample_bytree": [0.75, 0.85, 1.0],
            "min_child_weight": [1, 3, 5],
            "gamma": [0, 0.05, 0.1, 0.2],
            "reg_alpha": [0, 0.01, 0.1],
            "reg_lambda": [0.5, 1.0, 2.0, 5.0],
        }
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=40 if not quick else 8,
            scoring="roc_auc",
            cv=5,
            random_state=random_state,
            n_jobs=1,
            verbose=0,
        )
        search.fit(X_fit, y_fit)
        model = search.best_estimator_
        best_params = search.best_params_
        cv_best_score = float(search.best_score_)
    else:
        model = base_model.fit(X_fit, y_fit)
        best_params = model.get_params()
        cv_best_score = None

    threshold = 0.5
    threshold_val_score = None
    if tune_threshold and X_val is not None and y_val is not None:
        y_val_proba = model.predict_proba(X_val)[:, 1]
        threshold, threshold_val_score = find_best_threshold(
            y_val,
            y_val_proba,
            metric=threshold_metric,
        )

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = threshold_predictions(y_proba, threshold)
    metrics = evaluate_binary_classifier(y_test, y_pred, y_proba)
    metrics.update({
        "model": "XGBoost",
        "experiment_label": experiment_label or "baseline",
        "time_label": time_label,
        "n_features": len(feature_names),
        "n_rows": summary["n_rows"],
        "cv_best_roc_auc": cv_best_score,
        "decision_threshold": threshold,
        "threshold_metric": threshold_metric if tune_threshold else "fixed_0.5",
        "threshold_val_score": threshold_val_score,
    })

    suffix = f"_{experiment_label}" if experiment_label else ""
    prefix = f"xgboost_{time_label}{suffix}"
    with open(output_dir / "tables" / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "best_params": best_params, "data_summary": summary}, f, indent=2, ensure_ascii=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "tables" / f"{prefix}_metrics.csv", index=False)

    importance = plot_feature_importance(model, feature_names, output_dir / "figures" / f"{prefix}_feature_importance.png")
    importance.to_csv(output_dir / "tables" / f"{prefix}_feature_importance.csv", index=False)
    plot_confusion_matrix(y_test, y_pred, output_dir / "figures" / f"{prefix}_confusion_matrix.png", f"XGBoost Confusion Matrix ({time_label})")
    plot_roc_curve(y_test, y_proba, output_dir / "figures" / f"{prefix}_roc_curve.png", f"XGBoost ROC Curve ({time_label})")

    pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "y_proba": y_proba,
        "y_pred": y_pred,
    }).to_csv(output_dir / "tables" / f"{prefix}_predictions.csv", index=False)

    with open(model_dir / f"{prefix}.pkl", "wb") as f:
        pickle.dump(model, f)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--time-label", type=str, required=True, help="Example: 10minute or 15minute")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-tune", action="store_true", help="Skip RandomizedSearchCV and train one configured model.")
    parser.add_argument("--quick", action="store_true", help="Use a smaller search space for a fast smoke test.")
    parser.add_argument("--tune-threshold", action="store_true", help="Tune the decision threshold on a validation split.")
    parser.add_argument("--threshold-metric", type=str, default="accuracy", choices=["accuracy", "f1"])
    parser.add_argument("--experiment-label", type=str, default="", help="Suffix for saving additional experiment outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train_xgboost(
        data_file=args.data_file,
        time_label=args.time_label,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        random_state=args.random_state,
        tune=not args.no_tune,
        quick=args.quick,
        tune_threshold=args.tune_threshold,
        threshold_metric=args.threshold_metric,
        experiment_label=args.experiment_label,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
