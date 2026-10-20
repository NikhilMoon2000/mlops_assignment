"""Train and compare candidate models, recording every run in MLflow."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

from heart_mlop.data import FEATURE_COLUMNS, TARGET_COLUMN
from heart_mlop.modeling import build_candidates

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data" / "processed" / "heart_disease.csv"
DEFAULT_ARTIFACTS = ROOT / "artifacts"
SCORING = ["accuracy", "precision", "recall", "roc_auc"]


def _save_evaluation_plots(y_true: pd.Series, probabilities, predictions, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    RocCurveDisplay.from_predictions(y_true, probabilities, ax=axes[0], name="Held-out test")
    ConfusionMatrixDisplay.from_predictions(
        y_true, predictions, display_labels=["No disease", "Disease"], cmap="Blues", ax=axes[1]
    )
    figure.tight_layout()
    figure.savefig(directory / "evaluation.png", dpi=160)
    plt.close(figure)


def train(data_path: str | Path = DEFAULT_DATA, artifact_dir: str | Path = DEFAULT_ARTIFACTS) -> dict:
    frame = pd.read_csv(data_path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURE_COLUMNS],
        frame[TARGET_COLUMN].astype(int),
        test_size=0.2,
        stratify=frame[TARGET_COLUMN],
        random_state=42,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", f"file:{ROOT / 'mlruns'}"))
    mlflow.set_experiment("uci-heart-disease")

    results = {}
    for name, pipeline in build_candidates().items():
        with mlflow.start_run(run_name=name) as run:
            cv_result = cross_validate(
                pipeline,
                x_train,
                y_train,
                scoring=SCORING,
                cv=cv,
                n_jobs=1,
                return_train_score=False,
            )
            for metric in SCORING:
                mlflow.log_metric(f"cv_{metric}_mean", float(cv_result[f"test_{metric}"].mean()))
                mlflow.log_metric(f"cv_{metric}_std", float(cv_result[f"test_{metric}"].std()))
            mlflow.log_params(
                {
                    "model": name,
                    "cv_folds": 5,
                    "test_size": 0.2,
                    "random_state": 42,
                    "features": len(FEATURE_COLUMNS),
                }
            )

            if name == "logistic_regression":
                search = GridSearchCV(
                    pipeline,
                    {"classifier__C": [0.01, 0.1, 1.0]},
                    scoring="roc_auc",
                    cv=cv,
                    n_jobs=1,
                )
                search.fit(x_train, y_train)
                fitted_model = search.best_estimator_
                mlflow.log_param("tuned_C", float(search.best_params_["classifier__C"]))
                mlflow.log_metric("tuning_best_cv_roc_auc", float(search.best_score_))
            else:
                fitted_model = pipeline.fit(x_train, y_train)

            probabilities = fitted_model.predict_proba(x_test)[:, 1]
            predictions = fitted_model.predict(x_test)
            holdout_metrics = {
                "test_accuracy": accuracy_score(y_test, predictions),
                "test_precision": precision_score(y_test, predictions, zero_division=0),
                "test_recall": recall_score(y_test, predictions, zero_division=0),
                "test_roc_auc": roc_auc_score(y_test, probabilities),
            }
            mlflow.log_metrics({key: float(value) for key, value in holdout_metrics.items()})
            plot_dir = artifact_dir / "plots" / name
            _save_evaluation_plots(y_test, probabilities, predictions, plot_dir)
            mlflow.log_artifact(str(plot_dir / "evaluation.png"), artifact_path="plots")
            mlflow.sklearn.log_model(
                fitted_model,
                artifact_path="model",
                input_example=x_train.iloc[:2],
            )
            results[name] = {
                "run_id": run.info.run_id,
                "cv": {
                    metric: float(cv_result[f"test_{metric}"].mean()) for metric in SCORING
                },
                "test": {key.removeprefix("test_"): float(value) for key, value in holdout_metrics.items()},
                "model": fitted_model,
            }

    selected_name = max(results, key=lambda candidate: results[candidate]["cv"]["roc_auc"])
    selected = results[selected_name]
    import joblib

    model_path = artifact_dir / "model.joblib"
    joblib.dump(selected["model"], model_path)
    metadata = {
        "model_name": selected_name,
        "run_id": selected["run_id"],
        "features": FEATURE_COLUMNS,
        "cv_metrics": selected["cv"],
        "test_metrics": selected["test"],
        "target_definition": "1 = target > 0; 0 = target == 0",
        "dataset": "UCI Heart Disease, processed Cleveland subset",
    }
    (artifact_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"selected_model": selected_name, "models": results, "model_path": model_path}


if __name__ == "__main__":
    summary = train()
    print(json.dumps({
        "selected_model": summary["selected_model"],
        "models": {
            name: {"cv": result["cv"], "test": result["test"]}
            for name, result in summary["models"].items()
        },
        "model_path": str(summary["model_path"]),
    }, indent=2))