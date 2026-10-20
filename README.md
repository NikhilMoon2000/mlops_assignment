# Heart Disease MLOps Assignment

An end-to-end, reproducible binary-classification project using the processed Cleveland subset of the UCI Heart Disease dataset. It includes data acquisition, EDA, leakage-safe preprocessing, two model candidates, MLflow tracking, tests, CI, a FastAPI service, container and Kubernetes manifests, and a 10-page report.

> Educational demonstration only. This model is not a medical device and must not be used for diagnosis or treatment decisions.

## Quick start

Use Python 3.11 for a clean installation (the local development machine used Python 3.9.6). From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=src
```

Download and clean the public UCI source file, generate EDA plots, train both models, and write the selected model and metadata:

```bash
python scripts/download_data.py
python scripts/eda.py
python -m heart_mlop.train
pytest -q
ruff check src scripts tests
```

The data acquisition script writes `data/raw/processed.cleveland.data` and `data/processed/heart_disease.csv`. The source includes 303 Cleveland records, 13 predictors, six missing predictor values, and an original target from 0 to 4. The cleaning step maps target 0 to class 0 and values 1-4 to class 1. Imputation is fitted inside each training fold and serialized with the classifier.

## Experiment tracking and outputs

MLflow uses a local file store at `mlruns/` unless `MLFLOW_TRACKING_URI` is set. Each candidate run records five-fold stratified CV means and standard deviations, held-out metrics, hyperparameters, an evaluation plot, and a serialized MLflow model. Logistic regression tunes `C` over `[0.01, 0.1, 1.0]`; the random forest is a fixed seeded baseline. The final `artifacts/model.joblib` is selected by mean CV ROC-AUC. `artifacts/model_metadata.json` records selection and evaluation details.

Open the experiment UI with `mlflow ui --backend-store-uri ./mlruns` and visit `http://127.0.0.1:5000`.

## Run the API

Training must finish first so `artifacts/model.joblib` exists:

```bash
PYTHONPATH=src uvicorn heart_mlop.api:app --host 127.0.0.1 --port 8000
```

Interactive API documentation is at `http://127.0.0.1:8000/docs`. Check `GET /health`, submit the example with `POST /predict`, and inspect Prometheus text at `GET /metrics`:

```bash
curl -s http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data @examples/patient.json
```

The response contains the binary prediction, class label, predicted-class confidence, and estimated disease probability. Request fields are validated and unexpected fields are rejected. API logs go to standard output. The metrics endpoint exposes request and successful-prediction counters.

## Container and local Kubernetes

After training:

```bash
docker build -t heart-disease-api:local .
docker run --rm -p 8000:8000 heart-disease-api:local
```

For Minikube, start the cluster, build/load the local image into its runtime, then apply `deployment/kubernetes.yaml`:

```bash
minikube start
minikube image build -t heart-disease-api:local .
kubectl apply -f deployment/kubernetes.yaml
kubectl get pods,service
minikube service heart-disease-api --url
```

The manifest defines two replicas, health probes, resource requests/limits, Prometheus scrape annotations, and a `LoadBalancer` service. The provided `deployment/prometheus.yml` targets a locally port-forwarded API; adjust its target for a cluster deployment. A Prometheus server or Grafana dashboard is not provisioned by this repository; `/metrics` is directly scrapeable.

## CI and report

GitHub Actions runs Ruff, downloads/prepares data, generates EDA figures, trains both candidates, runs pytest, and uploads model/MLflow artifacts. It runs on pushes and pull requests.

The ten-page report is `report/assignment_report.pdf`; its editable source is `report/assignment_report.md`. Rebuild it after editing with:

```bash
python scripts/build_report.py
```

## Submission evidence to complete

This workspace does not have Docker, kubectl, a public cloud account, or a configured GitHub remote. Therefore no cloud endpoint, container-build proof, deployment screenshot, CI run screenshot, or pipeline video is claimed. After running the local container or Minikube flow, save genuine screenshots under `screenshots/` and record a short demonstration covering data preparation, MLflow runs, tests/CI, and a live API prediction. Add your own name, course details, repository URL, and chosen deployment evidence to the report before submitting. The data is historical and small; apparent held-out performance is not evidence of clinical generalization.