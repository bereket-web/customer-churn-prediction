# Contributing

## Local setup

```bash
uv sync --extra test
uv run churn-lab download
uv run churn-lab train
uv run pytest
uv run ruff check .
```

The benchmark downloads a dataset licensed separately from this code. Do not commit raw data, row-level predictions, or fitted model artifacts. Preserve the grouped split and keep model/threshold decisions confined to development folds; never adjust the model using held-out test metrics.

When changing the feature or API schema, update tests, the README, the generated input schema, and the data provenance notes. Explanations must remain clearly labeled as descriptive rather than causal.
