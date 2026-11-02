# Held-out error analysis

The cross-validation winner was **hist_gradient_boosting** (mean CV average precision 0.936 ± 0.032). Its held-out average precision was 0.957, precision 0.850, recall 0.919, and F1 0.883 at the threshold selected from development-only out-of-fold predictions.

On the held-out fold it generated 16 false-positive flags and 8 false negatives. These counts reflect this dataset's selected threshold, not a recommended operational policy.

## Permutation importance

The following fields caused the largest average precision drop when individually permuted on the held-out set. This is descriptive, can be unstable with correlated features, and is not causal:

| Input field | Mean AP drop | Standard deviation |
|---|---:|---:|
| frequency_of_use | 0.2257 | 0.0291 |
| complains | 0.1888 | 0.0299 |
| seconds_of_use | 0.1608 | 0.0231 |
| subscription_length | 0.1055 | 0.0139 |
| age_group | 0.0439 | 0.0083 |

## Known evaluation limits

- The source has 3,150 records and repeated customer-like feature profiles; exact model-input duplicates were grouped so they cannot appear in both development and test partitions.
- Some feature profiles have conflicting churn labels. This is retained as source uncertainty rather than silently deduplicated or relabeled.
- The dataset describes a three-month planning gap, but does not provide a deployable feature timestamp or customer-level cohort metadata. The grouped random holdout is not a true forward-in-time evaluation.
- The `Status` field is deliberately excluded: its observation time is not clear and its value is strongly associated with churn, making it a plausible outcome proxy.
- Probabilities are uncalibrated; the threshold is an experimental max-F1 operating point and not a business action policy.
- No fairness, intervention, retention uplift, cost-effectiveness, or external-dataset validation is claimed.
