# Base Kaggle Competition Template

A ready-to-use ML experiment pipeline for Kaggle competitions. Download data, preprocess, train multiple models with automated hyperparameter tuning (Optuna), and track everything in MLflow — all orchestrated through a simple Makefile.

## Stack

- **Python 3.13+** managed with [uv](https://github.com/astral-sh/uv)
- **MLflow** for experiment tracking and artifact storage
- **Optuna** for Bayesian hyperparameter optimization (50 trials, 5-fold CV)
- **Docker Compose** stack: PostgreSQL (backend store) + MinIO (S3-compatible artifact store) + MLflow server
- **scikit-learn**, **XGBoost**, **pandas**, **NumPy**, **matplotlib**

## Project Structure

```
├── data/
│   ├── raw/                # Downloaded CSV files (DVC tracked)
│   └── processed/          # Preprocessed NumPy arrays (DVC tracked)
├── notebooks/
│   ├── baseline.ipynb      # US1: Majority-class and regex baselines
│   ├── tfidf_classical.ipynb # US2: TF-IDF + Gradient Boosting
│   ├── transformer.ipynb   # US3: BERTimbau fine-tuning
│   ├── ensemble.ipynb      # US4: Model ensembling
│   ├── kaggle_submission.ipynb # US4: Submission generator
│   └── eda.ipynb           # Exploratory data analysis
├── src/
│   ├── config/
│   │   └── mlflow_init.py  # MLflow experiment setup
│   ├── models/
│   │   ├── model.py        # BaseModel + training loop
│   │   ├── classification/ # 12 classifier implementations
│   │   └── regression/     # Regression models (unused)
│   ├── preprocessing/
│   │   └── preprocess.py   # Feature engineering pipeline
│   └── utils/
│       └── download_dataset.py
├── docker-compose.yml
├── Dockerfile.mlflow
├── Makefile
├── pyproject.toml
└── .env.example
```

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Edit .env:
# METRIC="f1_macro"
# TASK_TYPE="classification"
# TARGET_COLUMN="target"
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Start infrastructure

```bash
make up
```

### 4. Initialize DVC (MinIO remote)

```bash
dvc init
dvc remote add -d minio s3://dvc
dvc remote modify minio endpointurl http://localhost:9002
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin
```

### 5. Download and version data

```bash
make init
dvc add data/raw data/processed
dvc push
```

### 6. Run Experiment Notebooks

```bash
# Established baselines
papermill notebooks/baseline.ipynb notebooks/outputs/baseline.ipynb

# TF-IDF + Classical ML
papermill notebooks/tfidf_classical.ipynb notebooks/outputs/tfidf.ipynb

# Transformer Fine-tuning (BERTimbau)
papermill notebooks/transformer.ipynb notebooks/outputs/transformer.ipynb

# Ensemble & Submission
papermill notebooks/ensemble.ipynb notebooks/outputs/ensemble.ipynb
```

## MLflow Tracking

Every training run logs:
- Model type, task type, and optimization metric (macro-F1)
- Best hyperparameters from Optuna
- Cross-validation score (Stratified 5-Fold)
- Per-class precision, recall, and F1 scores
- Confusion matrix plot artifact

## Preprocessing

The pipeline in `src/preprocessing/preprocess.py` applies:

- **Numerical features:** KNN imputation + standard scaling
- **Categorical features:** most-frequent imputation + one-hot encoding

Outputs are saved as `.npy` arrays under `data/processed/`.

## MLflow Tracking

Every training run logs:

- Model type, task type, optimization metric, and number of trials
- Best hyperparameters from Optuna
- Cross-validation score and full test-set metrics
- Trained model artifact (sklearn format)
- Plots: confusion matrix, ROC curve, precision-recall curve (classification) or predicted-vs-actual, residuals (regression), plus Optuna optimization history and parameter importances

## Adding a New Model

1. Create `src/models/my_model.py`
2. Subclass `BaseModel` and implement `build_model(params)` and `suggest_params(trial)`
3. Add a `__main__` block that instantiates and calls `.run()`
4. Add the model name to the `MODELS` list in the `Makefile`

```python
from src.models.model import BaseModel

class MyModel(BaseModel):
    name = "my_model"

    def build_model(self, params):
        ...

    def suggest_params(self, trial):
        ...

if __name__ == "__main__":
    MyModel().run()
```

## Makefile Reference

```
make help           Show all available commands
make up             Start Postgres + MinIO + MLflow
make down           Stop all services
make restart        Restart all services
make logs           Tail service logs
make clean          Stop services and delete volumes
make init           Download dataset + run preprocessing
make train-<model>  Train a single model
make train-all      Train all models sequentially
```
