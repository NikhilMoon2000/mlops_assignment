"""Fetch the public UCI source file and write the normalized dataset."""

from pathlib import Path

from heart_mlop.data import download_dataset, prepare_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "processed.cleveland.data"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_disease.csv"


def main() -> None:
    download_dataset(RAW_PATH)
    frame = prepare_dataset(RAW_PATH, CLEAN_PATH)
    print(f"Saved {len(frame)} rows and {len(frame.columns) - 1} features to {CLEAN_PATH}")
    print(f"Missing feature values before pipeline imputation: {int(frame.isna().sum().sum())}")


if __name__ == "__main__":
    main()