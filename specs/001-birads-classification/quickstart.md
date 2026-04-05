# Quickstart: BI-RADS Classification Pipeline

**Feature**: 001-birads-classification

## Prerequisites

- Python >=3.11, <3.13
- [uv](https://github.com/astral-sh/uv) package manager
- Docker & Docker Compose
- Kaggle API credentials at `~/.kaggle/kaggle.json`

## 1. Install dependencies

```bash
uv sync
```

## 2. Start infrastructure

```bash
make up
```

This starts PostgreSQL (MLflow backend), MinIO (artifact + DVC storage), and MLflow server.

| Service | URL | Credentials |
|---|---|---|
| MLflow UI | http://localhost:5000 | — |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |

## 3. Initialize DVC

```bash
dvc init
dvc remote add -d minio s3://dvc
dvc remote modify minio endpointurl http://localhost:9002
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin
```

## 4. Download and version data

```bash
make init
dvc add data/raw data/processed
git add data/raw.dvc data/processed.dvc data/.gitignore
dvc push
```

## 5. Configure the competition metric

Ensure `.env` has:

```bash
METRIC="f1_macro"
TASK_TYPE="classification"
TARGET_COLUMN="target"
```

## 6. Run the PoC notebook

```bash
# Open Jupyter and run notebooks/poc_tfidf.ipynb
jupyter lab notebooks/
```

The notebook trains TF-IDF + LightGBM with macro-F1 optimization and class-weight balancing.

## 7. Train models via CLI (optional)

```bash
make train-lightgbm_          # Single model
make train-all                # All models
```

## 8. View results

Open MLflow UI at http://localhost:5000 to compare experiment runs, metrics, and artifacts.

## 9. Generate Kaggle submission

```bash
# Run notebooks/kaggle_submission.ipynb end-to-end
# Output: data/submission.csv (or /kaggle/working/submission.csv on Kaggle)
```

## 10. Pull data on a fresh clone

```bash
git clone <repo-url>
uv sync
make up
dvc pull
```

## Key Commands

| Command | Description |
|---|---|
| `make up` | Start Docker services (MLflow + MinIO + PostgreSQL) |
| `make down` | Stop Docker services |
| `make init` | Download Kaggle data + preprocess |
| `make train-<model>` | Train a single model with Optuna HPO |
| `dvc push` | Push data to MinIO remote |
| `dvc pull` | Pull data from MinIO remote |
| `dvc status` | Check DVC tracking status |
