"""Shared metric helpers for binary win/loss experiments."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def find_best_threshold(
    y_true,
    y_proba,
    metric: str = "accuracy",
    low: float = 0.30,
    high: float = 0.70,
    step: float = 0.005,
) -> tuple[float, float]:
    """Find a probability threshold on validation predictions.

    The default searches around 0.5 rather than the full 0-1 range to avoid
    selecting extreme thresholds that overfit a validation split.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)

    scorer = accuracy_score if metric == "accuracy" else f1_score
    thresholds = np.arange(low, high + step, step)
    best_threshold = 0.5
    best_score = float(scorer(y_true, y_proba >= best_threshold))

    for threshold in thresholds:
        score = float(scorer(y_true, y_proba >= threshold))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, best_score


def threshold_predictions(y_proba, threshold: float):
    """Convert probabilities into class predictions with a custom threshold."""
    return (np.asarray(y_proba) >= threshold).astype(int)
