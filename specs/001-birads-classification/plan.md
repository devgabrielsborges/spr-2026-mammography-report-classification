# Implementation Plan: BI-RADS Classification from Mammography Reports

**Spec**: [specs/001-birads-classification/spec.md](spec.md)
**Created**: 2026-04-04
**Status**: Draft

## Approach Summary

We take a **PoC-first, incremental** approach: fix the evaluation metric to macro-F1, set up DVC data versioning with MinIO, establish correct baselines, then layer in TF-IDF features and transformer models. The current codebase (regex features + 12 classical ML models + Optuna HPO + MLflow tracking) serves as the baseline stack — new dependencies are added exclusively via `uv add`. All experiments produce Kaggle-compatible notebooks that generate `submission.csv` with no internet access.

**Key trade-off accepted**: We prioritize getting a correct macro-F1 baseline and TF-IDF PoC before investing in transformer fine-tuning, because the current pipeline evaluates on accuracy (meaningless with 87% class-2 dominance). Getting the metric right unlocks valid comparisons for all downstream work.

## Technical Context

**Language/Version**: Python >=3.11, <3.13 (via uv)
**Primary Dependencies**: scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, pandas, NumPy, SciPy, matplotlib, sentence-transformers, mlflow 3.9.0, DVC (new), transformers + torch (existing via sentence-transformers)
**New Dependencies**: `dvc[s3]` (data versioning with MinIO remote)
**Storage**: MinIO (S3-compatible, via Docker Compose) for MLflow artifacts and DVC remote; PostgreSQL for MLflow backend
**Experiment Tracking**: MLflow (existing)
**Hyperparameter Optimization**: Optuna (existing)
**Target Platform**: Kaggle notebook (CPU/GPU, 9h runtime, no internet) for submissions; local dev with Docker Compose for MLflow/MinIO
**Project Type**: ML competition pipeline (notebooks + reusable `src/` modules)
**Constraints**: Kaggle code competition — submissions via notebooks, 9h runtime, no internet during inference

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| I. Reproducibility First | **PASS** | Seeds set (RANDOM_STATE=42), deps pinned via `pyproject.toml`/`uv.lock`, DVC to be added for data versioning |
| II. Statistical Rigor | **PASS (with fix)** | Stratified CV in place; metric MUST be changed from accuracy → macro-F1 in `.env` and `src/models/model.py` |
| III. Experiment Tracking | **PASS** | MLflow already integrated; need to add feature_set_version and CV strategy tags |
| IV. Data Integrity | **PASS** | Raw data immutable in `data/raw/`; DVC will version both raw and processed |
| V. Clean Code & Modularity | **PASS** | `src/` modules are importable; notebooks follow Config→Load→Train→Evaluate flow |
| VI. Iterative & Baseline-First | **PASS** | Baseline will be established first (majority-class → regex+LR → TF-IDF → transformer) |
| VII. Validation-Driven Submission | **PASS** | CV as primary signal; retrain on full data before submission |

**Gate 1 violation requiring fix**: The `.env` file sets `METRIC="accuracy"` and `src/models/model.py` does not include `f1_macro` in its scoring dictionary. This MUST be fixed in Phase 1 before any valid experiment can run.

## Project Structure

### Documentation (this feature)

```text
specs/001-birads-classification/
├── spec.md              # Feature specification
├── plan.md              # This file
├── research.md          # Phase 0 research output
├── data-model.md        # Phase 1 data model
└── quickstart.md        # Phase 1 quickstart guide
```

### Source Code (repository root)

```text
src/
├── config/
│   └── mlflow_init.py          # MLflow experiment setup (existing)
├── models/
│   ├── model.py                # BaseModel + training loop (existing, needs macro-F1 fix)
│   ├── classification/         # 12 classifier implementations (existing)
│   └── regression/             # Regression models (existing, unused)
├── preprocessing/
│   └── preprocess.py           # Feature engineering pipeline (existing, to extend with TF-IDF)
└── utils/
    ├── download_dataset.py     # Kaggle data download (existing)
    └── generate_submission_file.py  # Submission CSV generator (existing)

notebooks/
├── train.ipynb                 # Original training pipeline (existing, reference only)
├── eda.ipynb                   # Exploratory data analysis (existing)
├── baseline.ipynb              # Papermill-parameterized: regex baseline (new)
├── tfidf_classical.ipynb       # Papermill-parameterized: TF-IDF + classical ML (new)
├── transformer.ipynb           # Papermill-parameterized: transformer fine-tuning (new, Phase 3)
├── ensemble.ipynb              # Papermill-parameterized: ensemble methods (new)
├── kaggle_submission.ipynb     # Kaggle-compatible final submission notebook (new)
└── outputs/                    # Papermill output notebooks (DVC-tracked, .gitignore'd)

data/
├── raw/                        # Immutable competition data (DVC tracked)
│   ├── train.csv
│   ├── test.csv
│   └── submission.csv
└── processed/                  # Engineered features (DVC tracked)
```

## Phase 1: Data Acquisition and Infrastructure

### Data Collection

- [x] Download competition data via `make init` (already done — `data/raw/train.csv`, `data/raw/test.csv`)
- [x] Validate raw data schema: 18,272 train rows, 3 columns (ID, report, target); 4 test rows in sample
- [ ] Set up DVC with MinIO remote for `data/raw/` and `data/processed/`

### DVC Setup

- [ ] Install DVC: `uv add "dvc[s3]"`
- [ ] Initialize DVC: `dvc init`
- [ ] Configure MinIO remote:
  ```
  dvc remote add -d minio s3://dvc-data
  dvc remote modify minio endpointurl http://localhost:9002
  dvc remote modify minio access_key_id minioadmin
  dvc remote modify minio secret_access_key minioadmin
  ```
- [ ] Add MinIO bucket for DVC in `docker-compose.yml` (add `mc mb myminio/dvc-data --ignore-existing`)
- [ ] Track data directories: `dvc add data/raw data/processed`
- [ ] Add `data/raw/` and `data/processed/` to `.gitignore`
- [ ] Commit `.dvc` files to git

### Metric Fix (CRITICAL)

- [ ] Add `f1_macro` to `SKLEARN_SCORING` dict in `src/models/model.py`:
  ```python
  "f1_macro": "f1_macro",
  ```
- [ ] Add `f1_macro` to `METRIC_DIRECTION` and `CLASSIFICATION_METRICS` dicts
- [ ] Update `.env`: change `METRIC="accuracy"` → `METRIC="f1_macro"`
- [ ] Verify `cross_val_score` uses the correct `scoring="f1_macro"` parameter

### Exploratory Data Analysis

- [x] Class distribution profiled: 87.4% class 2, extreme imbalance (class 5: 29 samples, class 6: 45)
- [x] Report length distribution: mean 401 chars, range [183, 1622]
- [ ] Analyze per-class text patterns: which keywords/phrases are discriminative per BI-RADS category
- [ ] Document findings in `notebooks/eda.ipynb`

### Data Preparation

- [x] Stratified train/test split (80/20) implemented in `preprocess.py`
- [ ] Ensure preprocessor is fit only on training fold (currently fits on `X_train` — correct)
- [ ] Add stratified K-fold CV infrastructure (K=5) to replace single holdout

### Papermill + DVC Pipeline Setup

- [ ] Install papermill: `uv add papermill`
- [ ] Add `notebooks/outputs/` to `.gitignore`
- [ ] Add papermill parameter cells (tagged `parameters`) to each notebook: `notebooks/baseline.ipynb`, `notebooks/tfidf_classical.ipynb`, `notebooks/transformer.ipynb`, `notebooks/ensemble.ipynb`
- [ ] Create `dvc.yaml` with pipeline stages wrapping papermill calls:
  ```yaml
  stages:
    train-baseline:
      cmd: papermill notebooks/baseline.ipynb notebooks/outputs/baseline-output.ipynb
      deps: [notebooks/baseline.ipynb, data/raw/train.csv]
      outs: [notebooks/outputs/baseline-output.ipynb]
    train-tfidf:
      cmd: papermill notebooks/tfidf_classical.ipynb notebooks/outputs/tfidf-output.ipynb
      deps: [notebooks/tfidf_classical.ipynb, data/raw/train.csv, src/preprocessing/preprocess.py]
      outs: [notebooks/outputs/tfidf-output.ipynb]
    train-transformer:
      cmd: papermill notebooks/transformer.ipynb notebooks/outputs/transformer-output.ipynb
      deps: [notebooks/transformer.ipynb, data/raw/train.csv]
      outs: [notebooks/outputs/transformer-output.ipynb]
    ensemble:
      cmd: papermill notebooks/ensemble.ipynb notebooks/outputs/ensemble-output.ipynb
      deps: [notebooks/ensemble.ipynb, data/processed/oof_predictions/]
      outs: [notebooks/outputs/ensemble-output.ipynb]
  ```
- [ ] Verify `dvc repro` executes notebooks end-to-end via papermill
- [ ] Output notebooks in `notebooks/outputs/` are tracked as DVC outputs (pushed to MinIO, not git)

## Phase 2: PoC Modeling (TF-IDF + Classical ML)

### Baseline Establishment

- [ ] Implement majority-class baseline: always predict class 2 → compute macro-F1 (expected ≈ 0.07)
- [ ] Run current regex pipeline with macro-F1 metric → establish regex baseline score
- [ ] Log both baselines to MLflow with `baseline=true` tag

### Experiment Design

| Experiment ID | Description | Variable Changed | Hypothesis |
|---|---|---|---|
| EXP-001 | Majority-class baseline | — | Establishes macro-F1 floor (≈ 0.07) |
| EXP-002 | Regex features + LR (macro-F1) | Metric: accuracy → macro-F1 | Regex features capture some BI-RADS signal (expected 0.40-0.60) |
| EXP-003 | TF-IDF word (1,2)-grams | Feature set: regex → TF-IDF | Full text captures more signal than 12 binary features |
| EXP-004 | TF-IDF word + char (3,5)-grams | Feature set: +char n-grams | Character n-grams capture morphological patterns in Portuguese medical text |
| EXP-005 | TF-IDF + regex combined | Feature set: TF-IDF + regex | Domain knowledge features complement statistical text features |
| EXP-006 | Class-weight balancing | Training: unweighted → class_weight='balanced' | Balanced weights improve minority class recall (4, 5, 6) |
| EXP-007 | Top 3 models (LightGBM, XGBoost, CatBoost) with best feature set | Model selection | Gradient boosting models handle sparse TF-IDF features well |

### PoC Implementation

- [ ] Create `notebooks/tfidf_classical.ipynb` with:
  - TF-IDF vectorizer (word 1-2 grams, max_features=10000)
  - TF-IDF vectorizer (char 3-5 grams, max_features=20000)
  - Combined with existing regex features via `ColumnTransformer`
  - LightGBM with `class_weight='balanced'` and Optuna HPO (20 trials for PoC)
  - Stratified 5-fold CV, macro-F1 as primary metric
  - Per-class precision/recall/F1 breakdown
  - Confusion matrix visualization
- [ ] Ensure notebook is Kaggle-compatible: no internet, reads from `/kaggle/input/` or local `data/raw/`
- [ ] Log all runs to MLflow

### Iteration Criteria

- **Proceed to Phase 3 when**: Mean 5-fold CV macro-F1 > 0.50 with TF-IDF features
- **Pivot if**: Best TF-IDF model macro-F1 < regex baseline + 2 pp after EXP-003 through EXP-007 (i.e., no statistically meaningful improvement)

## Phase 3: Transformer Fine-Tuning

**Prerequisites**: Phase 2 complete, mean 5-fold CV macro-F1 > 0.50 achieved

### Pre-trained Model Selection

- **Primary**: `neuralmind/bert-base-portuguese-cased` (BERTimbau base, 110M params)
  - Available on Kaggle as dataset: `fernandosr85/bertimbau-portuguese-bert`
  - 512 max token length (sufficient for max report length of 1,622 chars ≈ ~400 tokens)
- **Fallback**: `bert-base-multilingual-cased` (mBERT, 110M params) — already available in most Kaggle environments

### Experiment Design (Phase 3)

| Experiment ID | Description | Variable Changed | Hypothesis |
|---|---|---|---|
| EXP-008 | BERTimbau base + classification head | Model: TF-IDF → transformer | Contextual embeddings capture negation and semantic patterns |
| EXP-009 | BERTimbau + focal loss (γ=2) | Loss: CE → focal loss | Focal loss improves minority class performance over class weights |
| EXP-010 | BERTimbau + class-weighted CE | Loss: CE → weighted CE | Class weights more stable than focal loss for 7-class problem |
| EXP-011 | Freeze lower 6 layers | Training: full fine-tune → partial | Reduces overfitting risk on 18K samples |

### Implementation

- [ ] Add dependencies: `uv add transformers accelerate` (torch already available via sentence-transformers)
- [ ] Upload BERTimbau to Kaggle as a dataset (for offline use)
- [ ] Create `notebooks/transformer.ipynb`:
  - Load BERTimbau from local path (Kaggle dataset or `models/` dir)
  - Tokenize reports (max_length=256, padding, truncation)
  - Classification head on [CLS] token (7-class output)
  - Focal loss or class-weighted CrossEntropyLoss
  - AdamW, lr=2e-5, linear warmup (10% steps), weight decay=0.01
  - Stratified 5-fold CV, early stopping on validation macro-F1
  - 3-5 epochs per fold
  - Per-class metrics + confusion matrix
- [ ] Profile Kaggle runtime: estimate total training + inference time
- [ ] Log all runs to MLflow

### Iteration Criteria

- **Proceed to Phase 4 when**: Transformer mean 5-fold CV macro-F1 exceeds TF-IDF baseline by ≥ 3 pp
- **Pivot if**: Transformer overfits (CV-holdout gap > 3 pp; investigate at > 2 pp per constitution) or exceeds Kaggle runtime

## Phase 4: Ensemble & Submission

**Prerequisites**: Phase 3 complete or Phase 2 results satisfactory for competition

### Ensemble Strategy

- [ ] Collect out-of-fold predictions from top 3 models (TF-IDF + gradient boosting variants + transformer)
- [ ] Implement weighted soft voting: optimize weights on validation fold predictions
- [ ] Implement logistic regression stacking as alternative
- [ ] Compare ensemble vs best single model on macro-F1

### Kaggle Submission Notebook

- [ ] Create `notebooks/kaggle_submission.ipynb`:
  - Self-contained: all preprocessing, feature engineering, model loading in one notebook
  - Reads data from `/kaggle/input/spr-2026-mammography-report-classification/`
  - Loads pre-trained model weights from Kaggle dataset (uploaded separately)
  - Produces `submission.csv` at `/kaggle/working/submission.csv`
  - No internet access required
  - Total runtime < 9 hours (target < 2 hours)
- [ ] Submission sanity checks:
  - Correct columns: `ID`, `target`
  - Correct row count (matches test set)
  - Valid class values (integers 0–6)
  - No missing values
  - Plausible class distribution

## Phase 5: Evaluation and Validation

- [ ] Evaluate final model on held-out test set
- [ ] Compute all metrics: macro-F1, per-class P/R/F1, confusion matrix, log loss
- [ ] Confidence intervals via bootstrap (1000 iterations) on holdout set
- [ ] Error analysis: characterize failure modes per class
- [ ] Compare CV score vs public leaderboard score for calibration
- [ ] Document results in model comparison report

## Phase 6: Documentation and Reproducibility

- [ ] Archive final experiment configuration (best params, feature set, preprocessing)
- [ ] Ensure all data artifacts are tracked in DVC and pushed to MinIO
- [ ] Update `README.md` with new Makefile targets and DVC workflow
- [ ] Write final methodology notes in `specs/001-birads-classification/`

## Dependencies

| Dependency | Owner | Status | Blocker? |
|---|---|---|---|
| DVC with S3 support | `uv add "dvc[s3]"` | To install | N |
| transformers + accelerate | `uv add transformers accelerate` | To install (Phase 3) | N |
| BERTimbau on Kaggle | `fernandosr85/bertimbau-portuguese-bert` | Available | N (Phase 3) |
| Docker Compose (MinIO + MLflow) | Local infra | Available | N |
| Kaggle API credentials | `~/.kaggle/kaggle.json` | Available | N |

## Risks and Mitigations

| Risk | Mitigation | Contingency |
|---|---|---|
| Extreme class imbalance (classes 5,6 < 50 samples) | Focal loss, class weights, stratified CV | Merge classes 5+6 into single "high suspicion" category if F1 remains low |
| Transformer overfitting on 18K samples | Early stopping, low LR, weight decay, partial layer freezing | Fall back to TF-IDF + gradient boosting ensemble |
| Kaggle 9h runtime exceeded by transformer | Profile runtime early; use base (not large) model | Submit TF-IDF-only notebook as safe submission |
| DVC + MinIO integration issues | Test locally before committing; MinIO already proven via MLflow | Fall back to git-lfs for data versioning |
| macro-F1 scoring not in sklearn BaseModel | Add `f1_macro` to scoring dict; test with `cross_val_score` | Use custom scorer: `make_scorer(f1_score, average='macro')` |

## Timeline

| Phase | Estimated Duration | Start | End |
|---|---|---|---|
| Phase 1: Infrastructure + DVC + Metric Fix | 1-2 days | 2026-04-04 | 2026-04-06 |
| Phase 2: PoC — TF-IDF + Classical ML | 3-4 days | 2026-04-06 | 2026-04-10 |
| Phase 3: Transformer Fine-Tuning | 4-5 days | 2026-04-10 | 2026-04-15 |
| Phase 4: Ensemble + Kaggle Submission | 2-3 days | 2026-04-15 | 2026-04-18 |
| Phase 5: Evaluation + Error Analysis | 2 days | 2026-04-18 | 2026-04-20 |
| Phase 6: Documentation + Final Submit | 1-2 days | 2026-04-20 | 2026-04-22 |
| Buffer (leaderboard iteration) | 6 days | 2026-04-22 | 2026-04-28 |
