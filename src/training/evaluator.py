from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, f1_score, log_loss, recall_score


def multiclass_brier_score(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_proba - one_hot) ** 2, axis=1)))


def evaluate_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    labels: tuple[int, ...] = (0, 1, 2),
    class_names: tuple[str, ...] = ("Away Win", "Draw", "Home Win"),
    n_bins: int = 10,
    y_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba)
    if y_pred is None:
        y_pred = np.argmax(y_proba, axis=1)
    else:
        y_pred = np.asarray(y_pred).astype(int)

    ll = float(log_loss(y_true, y_proba, labels=list(labels)))
    brier = multiclass_brier_score(y_true, y_proba, n_classes=len(labels))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    recalls = recall_score(y_true, y_pred, labels=list(labels), average=None, zero_division=0)

    calibration: dict[str, dict[str, list[float]]] = {}
    for idx, cname in enumerate(class_names):
        y_bin = (y_true == labels[idx]).astype(int)
        frac_pos, mean_pred = calibration_curve(
            y_bin,
            y_proba[:, idx],
            n_bins=n_bins,
            strategy="quantile",
        )
        calibration[cname] = {
            "mean_predicted_probability": mean_pred.tolist(),
            "fraction_of_positives": frac_pos.tolist(),
        }

    return {
        "accuracy": acc,
        "log_loss": ll,
        "brier_score": brier,
        "macro_f1": macro_f1,
        "class_wise_recall": {class_names[i]: float(recalls[i]) for i in range(len(class_names))},
        "calibration_curve": calibration,
    }
