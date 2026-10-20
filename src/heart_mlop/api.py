"""FastAPI prediction service with request logging and Prometheus metrics."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from heart_mlop.data import FEATURE_COLUMNS

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("heart_mlop.api")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))
_metrics_lock = Lock()
_request_count = 0
_prediction_count = 0


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float = Field(ge=0, le=120)
    trestbps: float = Field(gt=0, le=300)
    chol: float = Field(gt=0, le=1000)
    thalach: float = Field(gt=0, le=300)
    oldpeak: float = Field(ge=0, le=20)
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    exang: Literal[0, 1]
    slope: Literal[1, 2, 3]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[3, 6, 7]


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float
    disease_probability: float


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    if MODEL_PATH.exists():
        application.state.model = joblib.load(MODEL_PATH)
        logger.info("Model loaded from %s", MODEL_PATH)
    else:
        application.state.model = None
        logger.warning("Model artifact not found at %s; predictions are unavailable", MODEL_PATH)
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)
app.state.model = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": app.state.model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures) -> PredictionResponse:
    global _request_count, _prediction_count
    with _metrics_lock:
        _request_count += 1
    model = app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train the model first.")
    features = pd.DataFrame([patient.model_dump()], columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(features)[0]
    prediction = int(probabilities.argmax())
    confidence = float(probabilities[prediction])
    with _metrics_lock:
        _prediction_count += 1
    logger.info("Prediction completed: class=%d confidence=%.4f", prediction, confidence)
    return PredictionResponse(
        prediction=prediction,
        label="disease_present" if prediction else "no_disease_detected",
        confidence=confidence,
        disease_probability=float(probabilities[1]),
    )


@app.get("/metrics", response_class=Response)
def metrics() -> Response:
    with _metrics_lock:
        requests = _request_count
        predictions = _prediction_count
    body = (
        "# HELP heart_api_requests_total Total /predict requests received.\n"
        "# TYPE heart_api_requests_total counter\n"
        f"heart_api_requests_total {requests}\n"
        "# HELP heart_api_predictions_total Successful predictions returned.\n"
        "# TYPE heart_api_predictions_total counter\n"
        f"heart_api_predictions_total {predictions}\n"
    )
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")