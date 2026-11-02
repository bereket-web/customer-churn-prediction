from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn_prediction.data import duplicate_groups, load_dataset, normalize_columns
from churn_prediction.features import engineer_features
from churn_prediction.metrics import choose_f1_threshold, classification_metrics
from churn_prediction.schema import INPUT_FEATURES


def example_inputs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "call_failure": [2, 0, 4],
            "complains": [1, 0, 0],
            "subscription_length": [10, 0, 5],
            "charge_amount": [3, 2, 5],
            "seconds_of_use": [1200, 0, 500],
            "frequency_of_use": [30, 0, 10],
            "frequency_of_sms": [12, 2, 0],
            "distinct_called_numbers": [15, 0, 8],
            "age_group": [3, 2, 4],
            "tariff_plan": [1, 2, 1],
            "age": [30, 25, 45],
            "customer_value": [100.0, 0.0, 25.0],
        }
    )


def test_normalize_columns_collapses_spaces() -> None:
    normalized = normalize_columns(pd.DataFrame(columns=["Call  Failure", "Subscription Length", "Churn"]))
    assert list(normalized.columns) == ["call_failure", "subscription_length", "churn"]


def test_engineer_features_handles_zero_denominators() -> None:
    output = engineer_features(example_inputs())
    assert output.loc[0, "seconds_per_call"] == pytest.approx(40)
    assert output.loc[0, "calls_per_subscription_month"] == pytest.approx(3)
    assert output.loc[1, "seconds_per_call"] == 0
    assert output.loc[1, "calls_per_subscription_month"] == 0
    assert output.loc[1, "customer_value_per_subscription_month"] == 0
    assert np.isfinite(output.to_numpy()).all()
    assert len(output.columns) == len(INPUT_FEATURES) + 7


def test_duplicate_profiles_get_same_group() -> None:
    inputs = example_inputs()
    inputs.loc[2] = inputs.loc[0]
    groups = duplicate_groups(inputs)
    assert groups.iloc[0] == groups.iloc[2]
    assert groups.iloc[0] != groups.iloc[1]


def test_load_dataset_rejects_nonbinary_target(tmp_path) -> None:
    raw = example_inputs()
    raw["status"] = [1, 1, 2]
    raw["churn"] = [0, 1, 2]
    path = tmp_path / "bad.csv"
    raw.to_csv(path, index=False)
    with pytest.raises(ValueError, match="only 0"):
        load_dataset(path)


def test_threshold_and_metrics_use_positive_class() -> None:
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.3, 0.6, 0.9])
    threshold, oof_f1 = choose_f1_threshold(y, probabilities)
    metrics = classification_metrics(y, probabilities, threshold)
    assert 0 <= threshold <= 1
    assert oof_f1 == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["true_positive"] == 2
    assert metrics["false_positive"] == 0
