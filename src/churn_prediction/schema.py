"""Canonical input schema for the UCI Iranian Churn benchmark."""

TARGET = "churn"
EXCLUDED_FIELDS = {"status"}
INPUT_FEATURES = [
    "call_failure",
    "complains",
    "subscription_length",
    "charge_amount",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "age_group",
    "tariff_plan",
    "age",
    "customer_value",
]
NOMINAL_FEATURES = ["complains", "tariff_plan"]

FEATURE_LABELS = {
    "call_failure": "Call failures",
    "complains": "Filed a complaint",
    "subscription_length": "Subscription length (months)",
    "charge_amount": "Charge amount category",
    "seconds_of_use": "Call seconds (observation period)",
    "frequency_of_use": "Call frequency (observation period)",
    "frequency_of_sms": "SMS frequency (observation period)",
    "distinct_called_numbers": "Distinct called numbers",
    "age_group": "Age group",
    "tariff_plan": "Tariff plan",
    "age": "Age category",
    "customer_value": "Customer value (dataset units)",
}

FEATURE_INFO = {
    "call_failure": {"type": "integer", "min": 0, "step": 1},
    "complains": {"type": "choice", "options": [0, 1]},
    "subscription_length": {"type": "integer", "min": 0, "step": 1},
    "charge_amount": {"type": "integer", "min": 0, "max": 10, "step": 1},
    "seconds_of_use": {"type": "integer", "min": 0, "step": 1},
    "frequency_of_use": {"type": "integer", "min": 0, "step": 1},
    "frequency_of_sms": {"type": "integer", "min": 0, "step": 1},
    "distinct_called_numbers": {"type": "integer", "min": 0, "step": 1},
    "age_group": {"type": "choice", "options": [1, 2, 3, 4, 5]},
    "tariff_plan": {"type": "choice", "options": [1, 2]},
    "age": {"type": "integer", "min": 0, "max": 100, "step": 1},
    "customer_value": {"type": "number", "min": 0, "step": 0.01},
}
