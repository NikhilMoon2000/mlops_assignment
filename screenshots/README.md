# Evidence folder

## Captured local evidence

- `eda-numeric-distributions.png`: generated continuous-feature histograms.
- `eda-class-balance-correlation.png`: target balance and numeric correlation heatmap.
- `logistic-regression-evaluation.png` and `random-forest-evaluation.png`: held-out ROC and confusion-matrix plots.
- `mlflow-experiment-runs.png`: local MLflow `uci-heart-disease` experiment with six runs from the training executions.
- `api-docs.png`: live FastAPI endpoint documentation.
- `prediction-response.png`: successful live `/predict` response for `examples/patient.json`.

These are genuine local-run artifacts/screenshots. A GitHub Actions run, Docker build/container, Kubernetes or cloud deployment, and Prometheus server screenshot are not included because those environments were unavailable. Capture those separately after running the deployment instructions.