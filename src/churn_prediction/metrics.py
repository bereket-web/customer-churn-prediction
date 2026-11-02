"""Threshold selection, classification metrics, and explanations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_prediction.schema import FEATURE_LABELS, INPUT_FEATURES, NOMINAL_FEATURES


def choose_f1_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """Select the maximum-F1 threshold on development-only out-of-fold predictions."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5, 0.0
    denominator = precision[:-1] + recall[:-1]
    f1_values = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.argmax(f1_values))
    return float(thresholds[best_index]), float(f1_values[best_index])


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, labels, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, labels, zero_division=0)),
        "recall": float(recall_score(y_true, labels, zero_division=0)),
        "f1": float(f1_score(y_true, labels, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, labels)),
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "alert_rate": float(labels.mean()),
    }


def reference_values(training_inputs: pd.DataFrame) -> dict[str, float | int]:
    """Calculate median baselines and valid category modes from training data only."""
    references: dict[str, float | int] = {}
    for column in INPUT_FEATURES:
        values = training_inputs[column]
        if column in NOMINAL_FEATURES:
            references[column] = int(values.mode(dropna=True).iloc[0])
        else:
            references[column] = float(values.median())
    return references


def local_reference_explanation(
    pipeline: object,
    customer: dict[str, float | int],
    reference: dict[str, float | int],
) -> list[dict[str, float | int | str]]:
    """Show one-feature-at-a-time probability changes vs a training-set reference.

    This is a perturbation explanation, not a causal attribution. Correlated inputs
    and derived features can make effects overlap; each raw field is reset in turn.
    """
    original = pd.DataFrame([{name: customer[name] for name in INPUT_FEATURES}])
    base_probability = float(pipeline.predict_proba(original)[0, 1])
    impacts = []
    for field in INPUT_FEATURES:
        perturbed = original.copy()
        perturbed.loc[0, field] = reference[field]
        changed_probability = float(pipeline.predict_proba(perturbed)[0, 1])
        delta = base_probability - changed_probability
        impacts.append(
            {
                "feature": FEATURE_LABELS[field],
                "field": field,
                "observed_value": float(original.loc[0, field]),
                "reference_value": float(reference[field]),
                "probability_change": float(delta),
                "direction": "raises" if delta > 0 else "lowers" if delta < 0 else "no change",
            }
        )
    return sorted(impacts, key=lambda item: abs(float(item["probability_change"])), reverse=True)
