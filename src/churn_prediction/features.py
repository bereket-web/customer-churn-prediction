"""Leakage-safe deterministic feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from churn_prediction.schema import INPUT_FEATURES

DERIVED_FEATURES = [
    "calls_per_subscription_month",
    "sms_per_subscription_month",
    "seconds_per_call",
    "distinct_number_share",
    "sms_to_call_ratio",
    "customer_value_per_subscription_month",
    "has_any_call_failures",
]


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive interpretable usage rates without fitting on validation/test data."""
    missing = sorted(set(INPUT_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required raw input fields: {missing}")
    result = frame[INPUT_FEATURES].copy()
    subscription = result["subscription_length"].clip(lower=1)
    calls = result["frequency_of_use"].clip(lower=1)
    result["calls_per_subscription_month"] = result["frequency_of_use"] / subscription
    result["sms_per_subscription_month"] = result["frequency_of_sms"] / subscription
    result["seconds_per_call"] = result["seconds_of_use"] / calls
    result["distinct_number_share"] = result["distinct_called_numbers"] / calls
    result["sms_to_call_ratio"] = result["frequency_of_sms"] / calls
    result["customer_value_per_subscription_month"] = result["customer_value"] / subscription
    result["has_any_call_failures"] = (result["call_failure"] > 0).astype(float)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless transformer suitable for serialization inside a model pipeline."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> FeatureEngineer:
        engineer_features(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return engineer_features(X)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        names = list(input_features or INPUT_FEATURES) + DERIVED_FEATURES
        return np.asarray(names, dtype=object)
