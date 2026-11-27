# Group G190 — MP129: Tiffin Relay Delivery Time (Patna)

SCSE3040 MLOps Mini Project for **Patna Dabba Relay Union**.

- **Group ID:** G190
- **Problem ID:** MP129
- **Faculty:** Saurabh Shrivastava (Batch 28)

## Team Members

| Enrolment No. | Name 
|---|---|---|
| `S24CSEU0373` | Rohit Kumar
| `S24CSEU0377` | Ronak.
| `S24CSEU0457` | Samrik Dutta 
| `S24CSEU0385` | Saanann Roy 

## Repository Structure

- `params.yaml` — pipeline paths, feature lists, model candidates, and monitoring thresholds
- `src/tiffin/` — modular data checks, cleaning, MLflow training/registry, drift monitoring, retraining, and holdout prediction
- `app/main.py` — FastAPI service (`GET /health`, `POST /predict`) with Pydantic input rules (`temp_c <= 75`, `tiffins >= 2.5`) and stdout JSON logging
- `tests/` — pytest suite with FastAPI `TestClient` and synthetic-data pipeline tests
- `data/raw/` — DVC pointers (`train.csv.dvc`, `later.csv.dvc`) and `holdout_features.csv`
- `k8s/` — Kubernetes Deployment (`mp129-api`, 4 replicas, `/health` probes) and Service (`mp129-svc`, port `80 -> 8000`)
- `Dockerfile` & `.dockerignore` — container build for `mp129-api:1.0`
- `.github/workflows/ci.yml` — GitHub Actions CI running `pytest` on push
- `reports/findings.yaml` — verified data issues, MLflow tracking metadata, and drift/retraining metrics
- `predictions/holdout_predictions.csv` — final champion predictions (`record_id,prediction`)

## Quickstart (Run Locally)

1. Install pinned dependencies in your virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the full MLOps pipeline (data checks, cleaning, MLflow v1 training, PSI & group MAE monitoring, v2 retraining, `@champion` promotion, holdout predictions, and `reports/findings.yaml` update):

```bash
PYTHONPATH=src python3 -m tiffin.pipeline
```

3. Run automated tests:

```bash
PYTHONPATH=src:. pytest -v
```

4. Serve the FastAPI prediction API locally:

```bash
PYTHONPATH=src:. uvicorn app.main:app --host 0.0.0.0 --port 8000
```

5. Build the Docker image (optional local check):

```bash
docker build -t mp129-api:1.0 .
```
