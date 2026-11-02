"""Reproducible benchmark runner for the UCI Iranian Churn dataset."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score
from sklearn.model_selection import StratifiedGroupKFold

from churn_prediction.data import duplicate_groups, load_dataset, model_inputs, project_root, sha256_bytes
from churn_prediction.estimators import candidate_models, make_pipeline
from churn_prediction.metrics import choose_f1_threshold, classification_metrics, reference_values
from churn_prediction.schema import TARGET

SEED = 2026


def run_benchmark(data_path: str | Path | None = None, output_root: str | Path | None = None) -> dict:
    root = Path(output_root) if output_root else project_root()
    raw_path = Path(data_path) if data_path else root / "data" / "raw" / "Customer Churn.csv"
    report_dir = root / "reports"
    model_dir = root / "models"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(raw_path)
    X = model_inputs(frame)
    y = frame[TARGET].astype(int).to_numpy()
    groups = duplicate_groups(X).to_numpy()

    # The first stratified grouped fold is reserved as a final test set. Identical
    # model-input profiles remain in one partition, including profiles with label
    # disagreement; the target is never used to form group IDs.
    outer_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    development_idx, test_idx = next(outer_splitter.split(X, y, groups))
    X_dev, X_test = X.iloc[development_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
    y_dev, y_test = y[development_idx], y[test_idx]
    groups_dev = groups[development_idx]

    inner_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    folds = list(inner_splitter.split(X_dev, y_dev, groups_dev))
    reference = reference_values(X_dev)
    results: list[dict] = []
    fitted: dict[str, object] = {}

    for model_name, estimator in candidate_models().items():
        oof = np.full(len(X_dev), np.nan, dtype=float)
        fold_scores = []
        for train_idx, validation_idx in folds:
            pipeline = make_pipeline(estimator)
            pipeline.fit(X_dev.iloc[train_idx], y_dev[train_idx])
            fold_probabilities = pipeline.predict_proba(X_dev.iloc[validation_idx])[:, 1]
            oof[validation_idx] = fold_probabilities
            fold_scores.append(float(average_precision_score(y_dev[validation_idx], fold_probabilities)))

        if np.isnan(oof).any():
            raise RuntimeError(f"Out-of-fold predictions incomplete for {model_name}")
        threshold, oof_f1 = choose_f1_threshold(y_dev, oof)
        final_pipeline = make_pipeline(estimator)
        final_pipeline.fit(X_dev, y_dev)
        test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, test_probabilities, threshold)
        results.append(
            {
                "model": model_name,
                "cv_average_precision_mean": float(np.mean(fold_scores)),
                "cv_average_precision_std": float(np.std(fold_scores, ddof=1)),
                "oof_threshold": threshold,
                "oof_f1_at_selected_threshold": oof_f1,
                **{f"test_{key}": value for key, value in metrics.items()},
            }
        )
        fitted[model_name] = {"pipeline": final_pipeline, "probabilities": test_probabilities}

    comparison = pd.DataFrame(results).sort_values(
        ["cv_average_precision_mean", "model"], ascending=[False, True]
    ).reset_index(drop=True)
    comparison.insert(0, "selection_rank_by_cv_ap", np.arange(1, len(comparison) + 1))
    comparison.to_csv(report_dir / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"])
    best_row = comparison.iloc[0].to_dict()
    best_pipeline = fitted[best_name]["pipeline"]
    best_probabilities = fitted[best_name]["probabilities"]
    best_threshold = float(best_row["oof_threshold"])

    bundle = {
        "pipeline": best_pipeline,
        "threshold": best_threshold,
        "reference_values": reference,
        "model_name": best_name,
        "seed": SEED,
        "input_features": list(X.columns),
        "source_dataset": "UCI Iranian Churn, dataset id 563",
        "excluded_fields": ["status"],
    }
    bundle_path = model_dir / "churn_model.joblib"
    joblib.dump(bundle, bundle_path)

    # Global permutation importance is descriptive on held-out rows and is not used
    # to choose, tune, or retrain the model.
    permutation = permutation_importance(
        best_pipeline,
        X_test,
        y_test,
        scoring="average_precision",
        n_repeats=10,
        random_state=SEED,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean_ap_drop": permutation.importances_mean,
            "importance_std": permutation.importances_std,
        }
    ).sort_values("importance_mean_ap_drop", ascending=False)
    importance.to_csv(report_dir / "permutation_importance.csv", index=False)

    _plot_precision_recall(report_dir / "precision_recall.png", y_test, fitted, comparison, best_name)
    _plot_confusion_matrix(report_dir / "confusion_matrix.png", y_test, best_probabilities, best_threshold)

    raw_bytes = Path(raw_path).read_bytes()
    duplicate_count = int(X.duplicated(keep="first").sum())
    duplicate_profile_count = int(pd.Series(groups).duplicated(keep="first").sum())
    summary = {
        "dataset": {
            "name": "Iranian Churn",
            "uci_dataset_id": 563,
            "source_url": "https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset",
            "license": "CC BY 4.0",
            "rows": int(len(frame)),
            "raw_columns_including_target_and_excluded_status": int(frame.shape[1]),
            "model_input_features_before_engineering": int(X.shape[1]),
            "positive_label": "churn=1",
            "positive_count": int(y.sum()),
            "positive_prevalence": float(y.mean()),
            "raw_csv_sha256": sha256_bytes(raw_bytes),
            "exact_duplicate_model_input_rows": duplicate_count,
            "unique_model_input_profiles": int(len(np.unique(groups))),
            "duplicate_profile_rows_grouped_together": duplicate_profile_count,
        },
        "protocol": {
            "random_seed": SEED,
            "outer_split": "first fold of 5-fold StratifiedGroupKFold (80% development / 20% held-out test approximately)",
            "inner_cv": "5-fold StratifiedGroupKFold on development partition",
            "group_definition": "stable hash of all 12 model input fields; identical profiles never cross partitions",
            "model_selection": "highest mean inner-fold average precision (average precision / stepwise PR-AUC)",
            "threshold_selection": "maximum F1 on development-only out-of-fold probabilities for each candidate",
            "test_usage": "single held-out evaluation; no test-driven model or threshold tuning",
            "excluded_field": "status; UCI timing is ambiguous for this field and it is strongly associated with the outcome",
            "data_timing": "UCI describes features as aggregated over months 1-9 and churn labels at month 12, with a 3-month planning gap",
            "class_distribution_split": {
                "development_churn_rate": float(y_dev.mean()),
                "test_churn_rate": float(y_test.mean()),
            },
        },
        "selected_model": best_name,
        "selected_model_threshold": best_threshold,
        "selected_model_test_metrics": {key.removeprefix("test_"): value for key, value in best_row.items() if key.startswith("test_")},
        "artifacts": {
            "model_comparison": "reports/model_comparison.csv",
            "permutation_importance": "reports/permutation_importance.csv",
            "leakage_audit": "reports/leakage_audit.json",
            "precision_recall_plot": "reports/precision_recall.png",
            "confusion_matrix_plot": "reports/confusion_matrix.png",
            "model_bundle_local_only": "models/churn_model.joblib",
        },
    }
    (report_dir / "evaluation_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_error_analysis(report_dir / "error_analysis.md", summary, comparison, importance)
    _write_leakage_audit(report_dir / "leakage_audit.json", frame, X)
    return summary


def _plot_precision_recall(path: Path, y_test: np.ndarray, fitted: dict, comparison: pd.DataFrame, best_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    for model_name, model_data in fitted.items():
        PrecisionRecallDisplay.from_predictions(
            y_test,
            model_data["probabilities"],
            name=model_name,
            ax=ax,
        )
    ax.axhline(float(np.mean(y_test)), color="gray", linestyle="--", linewidth=1, label=f"Test prevalence ({np.mean(y_test):.1%})")
    ax.set_title("Held-out precision–recall curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_confusion_matrix(path: Path, y_test: np.ndarray, probabilities: np.ndarray, threshold: float) -> None:
    from sklearn.metrics import ConfusionMatrixDisplay

    labels = (probabilities >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, labels, labels=[0, 1], display_labels=["Retained", "Churned"], cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(f"Selected model at OOF threshold {threshold:.3f}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_leakage_audit(path: Path, frame: pd.DataFrame, inputs: pd.DataFrame) -> None:
    """Record the ambiguous status field and label-conflicting duplicate profiles."""
    status_summary = {}
    for status_value, group in frame.groupby("status", sort=True):
        status_summary[str(int(status_value))] = {
            "rows": int(len(group)),
            "churned": int(group[TARGET].sum()),
            "churn_rate": float(group[TARGET].mean()),
        }
    grouped_labels = pd.DataFrame({"profile": duplicate_groups(inputs), TARGET: frame[TARGET].to_numpy()}).groupby("profile")[TARGET]
    conflicting_groups = grouped_labels.nunique().loc[lambda values: values > 1].index
    payload = {
        "excluded_field": "status",
        "uci_definition": "1=active, 2=non-active; timing relative to churn label is not stated clearly",
        "pearson_correlation_with_churn": float(frame["status"].corr(frame[TARGET])),
        "churn_rate_by_status": status_summary,
        "exclusion_reason": "potential outcome proxy and uncertain feature availability at prediction time",
        "duplicate_input_profiles": int(grouped_labels.size().gt(1).sum()),
        "profiles_with_conflicting_churn_labels": int(len(conflicting_groups)),
        "conflicting_profile_row_count": int(grouped_labels.size().reindex(conflicting_groups).sum()),
        "handling": "Status is not used by the model; identical eligible input profiles are grouped across splits and conflicting labels are retained.",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_error_analysis(path: Path, summary: dict, comparison: pd.DataFrame, importance: pd.DataFrame) -> None:
    best = comparison.iloc[0]
    metrics = summary["selected_model_test_metrics"]
    top = importance.head(5)
    lines = [
        "# Held-out error analysis",
        "",
        f"The cross-validation winner was **{summary['selected_model']}** (mean CV average precision "
        f"{best['cv_average_precision_mean']:.3f} ± {best['cv_average_precision_std']:.3f}). Its held-out average "
        f"precision was {metrics['average_precision']:.3f}, precision {metrics['precision']:.3f}, recall "
        f"{metrics['recall']:.3f}, and F1 {metrics['f1']:.3f} at the threshold selected from development-only "
        "out-of-fold predictions.",
        "",
        f"On the held-out fold it generated {metrics['false_positive']} false-positive flags and "
        f"{metrics['false_negative']} false negatives. These counts reflect this dataset's selected threshold, "
        "not a recommended operational policy.",
        "",
        "## Permutation importance",
        "",
        "The following fields caused the largest average precision drop when individually permuted on the held-out "
        "set. This is descriptive, can be unstable with correlated features, and is not causal:",
        "",
        "| Input field | Mean AP drop | Standard deviation |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {row.feature} | {row.importance_mean_ap_drop:.4f} | {row.importance_std:.4f} |"
        for row in top.itertuples(index=False)
    )
    lines += [
        "",
        "## Known evaluation limits",
        "",
        "- The source has 3,150 records and repeated customer-like feature profiles; exact model-input duplicates were grouped so they cannot appear in both development and test partitions.",
        "- Some feature profiles have conflicting churn labels. This is retained as source uncertainty rather than silently deduplicated or relabeled.",
        "- The dataset describes a three-month planning gap, but does not provide a deployable feature timestamp or customer-level cohort metadata. The grouped random holdout is not a true forward-in-time evaluation.",
        "- The `Status` field is deliberately excluded: its observation time is not clear and its value is strongly associated with churn, making it a plausible outcome proxy.",
        "- Probabilities are uncalibrated; the threshold is an experimental max-F1 operating point and not a business action policy.",
        "- No fairness, intervention, retention uplift, cost-effectiveness, or external-dataset validation is claimed.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
