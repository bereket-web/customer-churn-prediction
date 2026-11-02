"""Streamlit demo that sends one record at a time to the local API."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("CHURN_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Customer Churn Lab", page_icon=None, layout="wide")
st.title("Customer Churn Prediction")
st.caption("Research demo using the UCI Iranian Churn dataset. Do not use as a production retention decision.")

try:
    health = requests.get(f"{API_URL}/health", timeout=4).json()
except requests.RequestException:
    st.error(f"API unavailable at {API_URL}. Start it with `uv run uvicorn churn_prediction.api:app --port 8000`.")
    st.stop()

if not health.get("model_loaded"):
    st.warning("No trained model is ready. Run `uv run churn-lab train` in the project directory, then restart the API.")
    st.stop()

schema_response = requests.get(f"{API_URL}/schema", timeout=4)
if not schema_response.ok:
    st.error(f"Could not load the input schema: {schema_response.text}")
    st.stop()
metadata = schema_response.json()
st.write(f"Model: **{metadata['model_name']}** · decision threshold: **{metadata['threshold']:.3f}**")
st.info("Enter an example record. Inputs are sent to the local API for this request only; the demo does not save them.")

with st.form("customer_form"):
    columns = st.columns(3)
    customer: dict[str, int | float] = {}
    for index, field in enumerate(metadata["fields"]):
        container = columns[index % len(columns)]
        with container:
            name, label = field["name"], field["label"]
            if field["type"] == "choice":
                value = st.selectbox(label, field["options"], index=field["options"].index(field["default"]), key=name)
            elif field["type"] == "integer":
                value = st.number_input(
                    label,
                    min_value=int(field.get("min", 0)),
                    max_value=int(field["max"]) if "max" in field else None,
                    value=int(round(field["default"])),
                    step=int(field.get("step", 1)),
                    key=name,
                )
            else:
                value = st.number_input(
                    label,
                    min_value=float(field.get("min", 0)),
                    value=float(field["default"]),
                    step=float(field.get("step", 0.01)),
                    format="%.2f",
                    key=name,
                )
            customer[name] = value
    submitted = st.form_submit_button("Estimate churn score")

if submitted:
    try:
        response = requests.post(f"{API_URL}/explain", json=customer, timeout=15)
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as exc:
        st.error(f"Prediction request failed: {exc}")
        st.stop()

    score = result["churn_probability"]
    label = "Model flag: likely churn" if result["model_flag"] else "Model flag: likely retained"
    left, right = st.columns([1, 2])
    left.metric("Estimated churn probability", f"{score:.1%}")
    left.write(f"{label} (threshold {result['threshold']:.3f})")
    left.progress(min(max(float(score), 0.0), 1.0))
    right.subheader("What changed the score most?")
    effects = result["top_effects"]
    st.dataframe(
        [
            {
                "Input": item["feature"],
                "Observed": item["observed_value"],
                "Reference": item["reference_value"],
                "Change vs. reference (probability points)": 100 * item["probability_change"],
                "Direction": item["direction"],
            }
            for item in effects
        ],
        hide_index=True,
        use_container_width=True,
    )
    st.caption(result["explanation_warning"])
