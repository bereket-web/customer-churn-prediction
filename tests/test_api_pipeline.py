from __future__ import annotations

import asyncio

import httpx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from churn_prediction import api
from churn_prediction.estimators import make_pipeline
from churn_prediction.inference import explain_customer, predict_customer
from churn_prediction.metrics import reference_values
from churn_prediction.schema import INPUT_FEATURES


def training_frame() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(7)
    n = 80
    frame = pd.DataFrame(
        {
            "call_failure": rng.integers(0, 12, n),
            "complains": rng.integers(0, 2, n),
            "subscription_length": rng.integers(1, 60, n),
            "charge_amount": rng.integers(0, 10, n),
            "seconds_of_use": rng.integers(0, 8000, n),
            "frequency_of_use": rng.integers(0, 250, n),
            "frequency_of_sms": rng.integers(0, 200, n),
            "distinct_called_numbers": rng.integers(0, 80, n),
            "age_group": rng.integers(1, 6, n),
            "tariff_plan": rng.integers(1, 3, n),
            "age": rng.integers(15, 60, n),
            "customer_value": rng.random(n) * 2000,
        }
    )
    labels = ((frame["complains"] + (frame["frequency_of_use"] < 30)) > 0).astype(int).to_numpy()
    return frame, labels


def test_pipeline_predicts_and_explains_input() -> None:
    frame, labels = training_frame()
    pipeline = make_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    pipeline.fit(frame, labels)
    reference = reference_values(frame)
    bundle = {
        "pipeline": pipeline,
        "threshold": 0.4,
        "reference_values": reference,
        "model_name": "test_logistic",
    }
    customer = frame.iloc[0][INPUT_FEATURES].to_dict()
    prediction = predict_customer(bundle, customer)
    effects = explain_customer(bundle, customer)
    assert 0 <= prediction["churn_probability"] <= 1
    assert prediction["model_name"] == "test_logistic"
    assert len(effects) == len(INPUT_FEATURES)
    assert all("probability_change" in effect for effect in effects)


def test_api_schema_predict_explain_and_validation(monkeypatch) -> None:
    frame, labels = training_frame()
    pipeline = make_pipeline(LogisticRegression(max_iter=500, class_weight="balanced"))
    pipeline.fit(frame, labels)
    bundle = {
        "pipeline": pipeline,
        "threshold": 0.4,
        "reference_values": reference_values(frame),
        "model_name": "test_logistic",
    }
    monkeypatch.setattr(api, "load_bundle", lambda: bundle)
    async def exercise_api() -> None:
        transport = httpx.ASGITransport(app=api.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/schema")
            assert response.status_code == 200
            payload = {field["name"]: field["default"] for field in response.json()["fields"]}
            assert set(payload) == set(INPUT_FEATURES)

            prediction = await client.post("/predict", json=payload)
            assert prediction.status_code == 200
            assert "churn_probability" in prediction.json()

            explanation = await client.post("/explain", json=payload)
            assert explanation.status_code == 200
            assert len(explanation.json()["top_effects"]) == 6
            assert "not causal" in explanation.json()["explanation_warning"]

            invalid = {**payload, "status": 1}
            assert (await client.post("/predict", json=invalid)).status_code == 422

    asyncio.run(exercise_api())
