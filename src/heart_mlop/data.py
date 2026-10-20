"""Download and validate the processed Cleveland Heart Disease dataset."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlopen

import pandas as pd

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "target"


def download_dataset(destination: str | Path) -> Path:
    """Download the original comma-separated Cleveland data file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(DATA_URL, timeout=30) as response:
        destination.write_bytes(response.read())
    return destination


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Read the raw UCI file and normalize its missing values and target."""
    frame = pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        na_values="?",
        skipinitialspace=True,
    )
    if frame.shape[1] != len(COLUMN_NAMES):
        raise ValueError(f"Expected 14 columns, found {frame.shape[1]}.")
    for column in COLUMN_NAMES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[TARGET_COLUMN]).copy()
    frame[TARGET_COLUMN] = (frame[TARGET_COLUMN] > 0).astype("int64")
    return frame


def prepare_dataset(raw_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Clean source data and save a stable CSV for analysis and training."""
    frame = load_dataset(raw_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame