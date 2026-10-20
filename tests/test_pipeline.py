import numpy as np
import pandas as pd
from sklearn.base import is_classifier

from heart_mlop.data import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    load_dataset,
)
from heart_mlop.modeling import build_candidates


def _sample_frame(rows=40):
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "age": rng.integers(30, 80, rows),
            "trestbps": rng.integers(90, 180, rows),
            "chol": rng.integers(150, 350, rows),
            "thalach": rng.integers(90, 200, rows),
            "oldpeak": rng.uniform(0, 5, rows),
            "sex": rng.integers(0, 2, rows),
            "cp": rng.integers(1, 5, rows),
            "fbs": rng.integers(0, 2, rows),
            "restecg": rng.integers(0, 3, rows),
            "exang": rng.integers(0, 2, rows),
            "slope": rng.integers(1, 4, rows),
            "ca": rng.integers(0, 4, rows),
            "thal": rng.integers(3, 8, rows),
        }
    )
    frame.loc[0, "chol"] = np.nan
    return frame[FEATURE_COLUMNS], pd.Series([0, 1] * (rows // 2))


def test_dataset_converts_multiclass_target_to_binary(tmp_path):
    raw = tmp_path / "tiny.data"
    raw.write_text("".join(
        f"50,1,1,120,200,0,1,150,0,1.0,2,0,3,{target}\n"
        for target in (0, 1, 2, 3, 4)
    ))
    cleaned = load_dataset(raw)
    assert cleaned["target"].tolist() == [0, 1, 1, 1, 1]


def test_preprocessing_imputes_and_models_predict():
    features, target = _sample_frame()
    for model in build_candidates().values():
        assert is_classifier(model)
        model.fit(features, target)
        assert model.predict(features.iloc[:3]).shape == (3,)
        assert np.isfinite(model.predict_proba(features.iloc[:3])).all()


def test_feature_groups_cover_all_inputs():
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert set(NUMERIC_FEATURES + CATEGORICAL_FEATURES) == set(FEATURE_COLUMNS)