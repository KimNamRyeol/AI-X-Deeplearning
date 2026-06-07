"""Preprocessing utilities for LoL blue-team win prediction.

Important leakage note:
- ``blueWins`` is the target.
- ``redWins`` is the exact opposite of the target and must be removed.
- ``gameId`` is an identifier and must be removed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from .feature_engineering import engineer_features, LIST_LIKE_COLUMNS
except ImportError:  # Allows direct execution from src/ if needed.
    from feature_engineering import engineer_features, LIST_LIKE_COLUMNS

TARGET_COLUMN = "blueWins"
LEAKAGE_OR_ID_COLUMNS = ["gameId", "redWins"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Return basic dataset statistics useful for EDA and README reporting."""
    summary: dict[str, Any] = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "missing_values": int(df.isna().sum().sum()),
    }
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts().sort_index()
        summary["blueWins_0_count"] = int(counts.get(0, 0))
        summary["blueWins_1_count"] = int(counts.get(1, 0))
        summary["blueWins_1_ratio"] = float(df[TARGET_COLUMN].mean())
    return summary


def prepare_xy(df: pd.DataFrame, use_engineered_features: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare feature matrix X and target y.

    The function removes leakage/id columns and converts the remaining features
    to numeric values. Object columns created from list-like text are encoded
    first and then dropped in their raw form.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    data = df.copy()
    if use_engineered_features:
        data = engineer_features(data)

    y = data[TARGET_COLUMN].astype(int)

    drop_cols = [TARGET_COLUMN, *LEAKAGE_OR_ID_COLUMNS]
    # Drop raw string/list-like columns after expanding them into numeric flags.
    drop_cols.extend([col for col in LIST_LIKE_COLUMNS if col in data.columns])

    X = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore")

    # Keep only numeric columns for both XGBoost and MLP.
    X = X.select_dtypes(include=[np.number]).copy()

    # Robust missing/infinite handling.
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Create a stratified train/test split shared by XGBoost and MLP."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def scale_train_test(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Standardize features for neural-network models."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def load_prepare_split(
    csv_path: str | Path,
    test_size: float = 0.2,
    random_state: int = 42,
    use_engineered_features: bool = True,
):
    """Convenience function used by training scripts and notebooks."""
    df = load_dataset(csv_path)
    X, y = prepare_xy(df, use_engineered_features=use_engineered_features)
    return (*split_dataset(X, y, test_size=test_size, random_state=random_state), X.columns.tolist())
