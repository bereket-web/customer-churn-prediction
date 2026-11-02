"""Dataset acquisition, validation, and duplicate-safe grouping."""

from __future__ import annotations

import hashlib
import re
import zipfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

from churn_prediction.schema import EXCLUDED_FIELDS, INPUT_FEATURES, TARGET

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/563/iranian+churn+dataset.zip"
UCI_PAGE_URL = "https://archive.ics.uci.edu/dataset/563/iranian+churn+dataset"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize UCI headers without changing row values."""
    result = frame.copy()
    result.columns = [_canonical_column(column) for column in result.columns]
    if len(set(result.columns)) != len(result.columns):
        raise ValueError("Column normalization produced duplicate column names")
    return result


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load and validate a local copy of the UCI CSV."""
    frame = normalize_columns(pd.read_csv(path))
    required = set(INPUT_FEATURES) | EXCLUDED_FIELDS | {TARGET}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    frame = frame[list(INPUT_FEATURES) + sorted(EXCLUDED_FIELDS) + [TARGET]].copy()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.isna().any().any():
        raise ValueError("The benchmark expects the documented complete UCI dataset; missing values were found")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("The churn target must contain only 0 (retained) and 1 (churned)")
    if frame.empty:
        raise ValueError("Dataset is empty")
    return frame


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only fields that are admissible at prediction time."""
    missing = sorted(set(INPUT_FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing model input fields: {missing}")
    return frame[INPUT_FEATURES].copy()


def duplicate_groups(inputs: pd.DataFrame) -> pd.Series:
    """Return stable hashes so identical model-input profiles never cross a split."""
    return pd.util.hash_pandas_object(inputs[INPUT_FEATURES], index=False).astype("uint64").astype(str)


def download_dataset(raw_dir: str | Path | None = None) -> dict[str, str | int]:
    """Download the public UCI archive and extract its single CSV locally."""
    destination = Path(raw_dir) if raw_dir else project_root() / "data" / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    response = requests.get(UCI_DATASET_URL, timeout=60)
    response.raise_for_status()
    archive = response.content
    with zipfile.ZipFile(BytesIO(archive)) as zipped:
        csv_members = [name for name in zipped.namelist() if name.lower().endswith(".csv")]
        if len(csv_members) != 1:
            raise ValueError(f"Expected exactly one CSV in UCI archive; found {csv_members}")
        csv_name = Path(csv_members[0]).name
        csv_bytes = zipped.read(csv_members[0])
    archive_path = destination / "iranian-churn-uci.zip"
    csv_path = destination / csv_name
    archive_path.write_bytes(archive)
    csv_path.write_bytes(csv_bytes)
    return {
        "archive_path": str(archive_path),
        "archive_sha256": sha256_bytes(archive),
        "csv_path": str(csv_path),
        "csv_sha256": sha256_bytes(csv_bytes),
        "csv_bytes": len(csv_bytes),
    }
