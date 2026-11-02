"""Shared prediction and explanation functions used by the API and tests."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from churn_prediction.metrics import local_reference_explanation
from churn_prediction.schema import FEATURE_INFO, FEATURE_LABELS, INPUT_FEATURES


def default_model_path() -> Path:
    configured = os.getenv("CHURN_MODEL_PATH")
    return Path(configured) if configured else Path(__file__).resolve().parents[2] / "models" / "churn_model.joblib"


@lru_cache(maxsize=2)
def load_bundle(path: str | None = None) -> dict[str, Any]:
    model_path = Path(path) if path else default_model_path()
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found at {model_path}; run `churn-lab train` first")
    return joblib.load(model_path)


def predict_customer(bundle: dict[str, Any], customer: dict[str, float | int]) -> dict[str, Any]:
    frame = pd.DataFrame([{name: customer[name] for name in INPUT_FEATURES}])
    probability = float(bundle["pipeline"].predict_proba(frame)[0, 1])
    threshold = float(bundle["threshold"])
    return {
        "churn_probability": probability,
        "threshold": threshold,
        "model_flag": bool(probability >= threshold),
        "model_label": "likely_churn" if probability >= threshold else "likely_retained",
        "model_name": bundle["model_name"],
    }


def explain_customer(bundle: dict[str, Any], customer: dict[str, float | int]) -> list[dict[str, Any]]:
    return local_reference_explanation(bundle["pipeline"], customer, bundle["reference_values"])


def input_schema(bundle: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_name": bundle["model_name"],
        "threshold": float(bundle["threshold"]),
        "fields": [
            {
                "name": name,
                "label": FEATURE_LABELS[name],
                "default": bundle["reference_values"][name],
                **FEATURE_INFO[name],
            }
            for name in INPUT_FEATURES
        ],
    }
