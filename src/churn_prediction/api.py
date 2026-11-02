"""FastAPI application for the trained churn benchmark."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from churn_prediction.inference import explain_customer, input_schema, load_bundle, predict_customer

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Research demo for the UCI Iranian Churn benchmark; not a production retention decision system.",
    version="0.1.0",
)


class CustomerFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    call_failure: float = Field(ge=0)
    complains: int = Field(ge=0, le=1)
    subscription_length: float = Field(ge=0)
    charge_amount: float = Field(ge=0, le=10)
    seconds_of_use: float = Field(ge=0)
    frequency_of_use: float = Field(ge=0)
    frequency_of_sms: float = Field(ge=0)
    distinct_called_numbers: float = Field(ge=0)
    age_group: int = Field(ge=1, le=5)
    tariff_plan: int = Field(ge=1, le=2)
    age: float = Field(ge=0, le=100)
    customer_value: float = Field(ge=0)


def _bundle_or_http_error() -> dict[str, Any]:
    try:
        return load_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        bundle = load_bundle()
        return {"status": "ok", "model_loaded": True, "model_name": bundle["model_name"]}
    except FileNotFoundError:
        return {"status": "not_ready", "model_loaded": False, "next_step": "Run `churn-lab train`"}


@app.get("/schema")
def schema() -> dict[str, Any]:
    return input_schema(_bundle_or_http_error())


@app.post("/predict")
def predict(customer: CustomerFeatures) -> dict[str, Any]:
    return predict_customer(_bundle_or_http_error(), customer.model_dump())


@app.post("/explain")
def explain(customer: CustomerFeatures) -> dict[str, Any]:
    bundle = _bundle_or_http_error()
    prediction = predict_customer(bundle, customer.model_dump())
    explanation = explain_customer(bundle, customer.model_dump())
    return {
        **prediction,
        "explanation_method": "one-field-at-a-time reset to the development-set median/mode",
        "explanation_warning": "Descriptive model perturbations, not causal effects; correlated inputs may overlap.",
        "top_effects": explanation[:6],
    }
