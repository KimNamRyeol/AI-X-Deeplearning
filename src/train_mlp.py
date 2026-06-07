"""Train and evaluate a PyTorch MLP classifier for LoL win prediction.

Example:
    python src/train_mlp.py \
        --data-file data/Challenger_Ranked_Games_10minute.csv \
        --time-label 10minute \
        --epochs 100
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
import torch
import torch.nn as nn
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
from sklearn.model_selection import train_test_split

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import load_prepare_split, scale_train_test, summarize_dataset, load_dataset
from src.metrics_utils import find_best_threshold, threshold_predictions


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Small tabular data trains faster and more predictably with limited CPU threads.
    torch.set_num_threads(1)


def make_activation(name: str) -> nn.Module:
    """Create an activation layer from a CLI-friendly name."""
    normalized = name.lower().replace("-", "_")
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32, 16),
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(prev, hidden),
                make_activation(activation),
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
            ])
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate_binary_classifier(y_true, y_pred, y_proba) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def plot_training_curve(history: pd.DataFrame, path: Path, title: str) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_loss"], label="train_loss")
    plt.plot(history["epoch"], history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary cross entropy loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


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


def batch_indices(n_samples: int, batch_size: int, rng: np.random.Generator):
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        yield indices[start:start + batch_size]


def eval_tensor_loss(model, X, y, criterion, device) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        loss = criterion(logits, y.to(device))
    return float(loss.item())


def eval_tensor_metrics(model, X, y, criterion, device) -> dict[str, float]:
    """Evaluate loss and threshold-free validation quality during training."""
    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        loss = criterion(logits, y.to(device))
        y_proba = torch.sigmoid(logits).detach().cpu().numpy().ravel()

    y_true = y.detach().cpu().numpy().ravel().astype(int)
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "loss": float(loss.item()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }


def train_mlp(
    data_file: Path,
    time_label: str,
    output_dir: Path,
    model_dir: Path,
    hidden_dims: tuple[int, ...] = (64, 32, 16),
    dropout: float = 0.2,
    activation: str = "relu",
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    random_state: int = 42,
    quick: bool = False,
    tune_threshold: bool = False,
    threshold_metric: str = "accuracy",
    experiment_label: str = "",
) -> dict[str, float]:
    set_seed(random_state)
    if quick:
        epochs = min(epochs, 15)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_file)
    summary = summarize_dataset(df)
    X_train, X_test, y_train, y_test, feature_names = load_prepare_split(data_file, random_state=random_state)
    X_train_scaled, X_test_scaled, scaler = scale_train_test(X_train, X_test)

    # Validation split from the training portion only.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_scaled,
        y_train.to_numpy(),
        test_size=0.15,
        random_state=random_state,
        stratify=y_train,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr_tensor = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    model = MLPClassifier(
        input_dim=X_train_scaled.shape[1],
        hidden_dims=hidden_dims,
        dropout=dropout,
        activation=activation,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )
    rng = np.random.default_rng(random_state)

    best_val = float("inf")
    best_state = None
    best_epoch = 0
    patience = 15
    patience_count = 0
    rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for idx in batch_indices(len(X_tr_tensor), batch_size, rng):
            xb = X_tr_tensor[idx].to(device)
            yb = y_tr_tensor[idx].to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            total_count += len(idx)

        train_loss = total_loss / total_count
        val_metrics = eval_tensor_metrics(model, X_val_tensor, y_val_tensor, criterion, device)
        val_loss = val_metrics["loss"]
        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_tensor.to(device))
        y_val_proba = torch.sigmoid(val_logits).detach().cpu().numpy().ravel()

    threshold = 0.5
    threshold_val_score = None
    if tune_threshold:
        threshold, threshold_val_score = find_best_threshold(
            y_val,
            y_val_proba,
            metric=threshold_metric,
        )

    model.eval()
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_test_tensor)
        y_proba = torch.sigmoid(logits).detach().cpu().numpy().ravel()
    y_pred = threshold_predictions(y_proba, threshold)

    metrics = evaluate_binary_classifier(y_test, y_pred, y_proba)
    metrics.update({
        "model": "MLP",
        "experiment_label": experiment_label or "baseline",
        "time_label": time_label,
        "hidden_dims": "-".join(map(str, hidden_dims)),
        "dropout": dropout,
        "activation": activation,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "decision_threshold": threshold,
        "threshold_metric": threshold_metric if tune_threshold else "fixed_0.5",
        "threshold_val_score": threshold_val_score,
        "epochs_run": len(rows),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "n_features": len(feature_names),
        "n_rows": summary["n_rows"],
        "device": str(device),
    })

    suffix = f"_{experiment_label}" if experiment_label else ""
    prefix = f"mlp_{time_label}{suffix}"
    history = pd.DataFrame(rows)
    history.to_csv(output_dir / "tables" / f"{prefix}_history.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "tables" / f"{prefix}_metrics.csv", index=False)
    with open(output_dir / "tables" / f"{prefix}_metrics.json", "w", encoding="utf-8") as f:
        json.dump({"metrics": metrics, "data_summary": summary}, f, indent=2, ensure_ascii=False)

    plot_training_curve(history, output_dir / "figures" / f"{prefix}_loss_curve.png", f"MLP Loss Curve ({time_label})")
    plot_confusion_matrix(y_test, y_pred, output_dir / "figures" / f"{prefix}_confusion_matrix.png", f"MLP Confusion Matrix ({time_label})")
    plot_roc_curve(y_test, y_proba, output_dir / "figures" / f"{prefix}_roc_curve.png", f"MLP ROC Curve ({time_label})")

    pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "y_proba": y_proba,
        "y_pred": y_pred,
    }).to_csv(output_dir / "tables" / f"{prefix}_predictions.csv", index=False)

    torch.save(model.state_dict(), model_dir / f"{prefix}.pt")
    with open(model_dir / f"{prefix}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return metrics


def parse_hidden_dims(text: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--time-label", type=str, required=True, help="Example: 10minute or 15minute")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--hidden-dims", type=str, default="64,32,16")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "leaky_relu", "gelu", "silu"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Run only a few epochs for a fast smoke test.")
    parser.add_argument("--tune-threshold", action="store_true", help="Tune the decision threshold on the validation split.")
    parser.add_argument("--threshold-metric", type=str, default="accuracy", choices=["accuracy", "f1"])
    parser.add_argument("--experiment-label", type=str, default="", help="Suffix for saving additional experiment outputs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metrics = train_mlp(
        data_file=args.data_file,
        time_label=args.time_label,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        hidden_dims=parse_hidden_dims(args.hidden_dims),
        dropout=args.dropout,
        activation=args.activation,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        random_state=args.random_state,
        quick=args.quick,
        tune_threshold=args.tune_threshold,
        threshold_metric=args.threshold_metric,
        experiment_label=args.experiment_label,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
