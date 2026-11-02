"""Preprocessing and candidate estimators."""

from __future__ import annotations

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from churn_prediction.features import DERIVED_FEATURES, FeatureEngineer
from churn_prediction.schema import INPUT_FEATURES, NOMINAL_FEATURES

NUMERIC_FEATURES = [name for name in INPUT_FEATURES + DERIVED_FEATURES if name not in NOMINAL_FEATURES]


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, NOMINAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def candidate_models() -> dict[str, object]:
    """Return candidates with fixed, documented settings (no hidden tuning)."""
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", solver="liblinear", C=1.0, random_state=2026
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=2026,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=180,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=2026,
        ),
        "rbf_svm": CalibratedClassifierCV(
            estimator=SVC(C=1.0, kernel="rbf", class_weight="balanced", random_state=2026),
            method="sigmoid",
            cv=3,
            ensemble=False,
        ),
    }


def make_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineer()),
            ("preprocessor", make_preprocessor()),
            ("classifier", estimator),
        ]
    )
