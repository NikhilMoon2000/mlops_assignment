FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/artifacts/model.joblib
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY artifacts ./artifacts
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "heart_mlop.api:app", "--host", "0.0.0.0", "--port", "8000"]