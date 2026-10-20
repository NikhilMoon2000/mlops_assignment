"""Generate assignment EDA plots from the cleaned dataset."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from heart_mlop.data import NUMERIC_FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "heart_disease.csv"
OUTPUT = ROOT / "artifacts" / "eda"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA_PATH)
    sns.set_theme(style="whitegrid", palette="crest")
    numeric = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    frame[numeric].hist(figsize=(10, 7), bins=18, edgecolor="white")
    plt.tight_layout()
    plt.savefig(OUTPUT / "numeric_distributions.png", dpi=160)
    plt.close()

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=frame, x="target", ax=axes[0])
    axes[0].set_xticks([0, 1], ["No disease", "Disease"])
    sns.heatmap(
        frame[NUMERIC_FEATURES + ["target"]].corr(),
        cmap="vlag",
        center=0,
        ax=axes[1],
    )
    axes[1].set_title("Numeric feature and target correlations")
    figure.tight_layout()
    figure.savefig(OUTPUT / "class_balance_and_correlation.png", dpi=160)
    plt.close(figure)
    print(f"EDA plots written to {OUTPUT}")


if __name__ == "__main__":
    main()