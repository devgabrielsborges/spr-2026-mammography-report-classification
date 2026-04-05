# Data Model: BI-RADS Classification

**Feature**: 001-birads-classification
**Date**: 2026-04-04

## Entities

### MammographyReport

The primary input entity. Each record represents one mammography exam with its associated text report.

| Field | Type | Source | Description |
|---|---|---|---|
| `ID` | string | `train.csv`, `test.csv` | Unique exam identifier (e.g., "Acc1") |
| `report` | string | `train.csv`, `test.csv` | Full mammography report text in Portuguese (impression section removed) |
| `target` | int (0–6) | `train.csv` only | BI-RADS category assigned by the radiologist |

**Validation rules**:
- `ID` MUST be unique across the dataset
- `report` MUST NOT be null or empty
- `target` MUST be an integer in {0, 1, 2, 3, 4, 5, 6}

**Derived fields** (from report text):
- `indicacao_class`: categorical — classification of the clinical indication section
- `achado_*`: 12 binary features — presence of specific findings patterns via regex
- `analise_comparativa`: string — extracted comparative analysis section text
- `report_length`: int — character count of report text

### ReportSection

Logical segments within a MammographyReport. Not stored as separate records but extracted during feature engineering.

| Section | Extraction Pattern | Present in |
|---|---|---|
| Indicação clínica | `r"Indicação clínica:\s*[\n\r]+\s*(.*?)(?=...)"` | ~100% of reports |
| Achados | `r"Achados:\s*[\n\r]+(.*?)(?=...)"` | ~98% of reports |
| Análise comparativa | `r"Análise comparativa:\s*[\n\r]+(.*?)$"` | ~70% of reports |

### FeatureSet

A versioned collection of extracted features for a given preprocessing configuration.

| Field | Type | Description |
|---|---|---|
| `version` | string | Semantic version (e.g., "v1.0" for regex-only, "v2.0" for TF-IDF + regex) |
| `feature_names` | list[string] | Ordered list of feature column names |
| `n_features` | int | Total feature dimensionality |
| `preprocessor` | sklearn ColumnTransformer | Fitted preprocessing pipeline |
| `storage_path` | string | DVC-tracked path under `data/processed/` |

**Feature set versions**:
- `v1.0`: 12 regex binary + 1 categorical (indicacao_class) + 1 ordinal (analise_comparativa) → ~20 features after encoding
- `v2.0`: v1.0 + TF-IDF word (1,2)-grams (10K) + TF-IDF char (3,4,5)-grams (20K) → ~30,020 features
- `v3.0`: Transformer [CLS] token embedding (768-dim for BERT base) — replaces all manual features

### ExperimentRun

Tracked via MLflow. Each run represents one model training with a specific configuration.

| Field | Type | Description |
|---|---|---|
| `run_id` | string | MLflow auto-generated UUID |
| `model_type` | string | Model name (e.g., "lightgbm", "bertimbau_base") |
| `feature_set_version` | string | Feature set version used (e.g., "v2.0") |
| `cv_strategy` | string | CV configuration (e.g., "stratified_5fold") |
| `cv_f1_macro` | float | Mean macro-F1 across CV folds |
| `holdout_f1_macro` | float | Macro-F1 on the 20% holdout set |
| `per_class_f1` | dict[int, float] | F1 per BI-RADS class |
| `best_params` | dict | Optuna-selected hyperparameters |
| `model_artifact` | binary | Serialized model stored in MLflow |
| `submission_artifact` | csv | Generated submission file (if applicable) |

### BIRADSCategory

Reference entity defining the 7 BI-RADS categories. Static, not stored in database.

| Value | Name | Clinical Action | Training Count | Percentage |
|---|---|---|---|---|
| 0 | Incomplete | Recall for additional imaging | 610 | 3.34% |
| 1 | Negative | Routine screening | 693 | 3.79% |
| 2 | Benign | Routine screening | 15,968 | 87.39% |
| 3 | Probably benign | Short-interval follow-up | 713 | 3.90% |
| 4 | Suspicious | Consider biopsy | 214 | 1.17% |
| 5 | Highly suggestive | Appropriate action | 29 | 0.16% |
| 6 | Known malignancy | Treatment | 45 | 0.25% |

## Relationships

```
MammographyReport ──1:N──> ReportSection (extracted, not persisted)
MammographyReport ──1:1──> BIRADSCategory (target label)
MammographyReport ──1:N──> FeatureSet (different versions)
FeatureSet ──1:N──> ExperimentRun (used by multiple experiments)
ExperimentRun ──N:1──> BIRADSCategory (predicts)
```

## Data Flow

```
data/raw/train.csv
    │
    ├─── extract_features() ──→ regex features (v1.0)
    │                           │
    │                           ├─── preprocess (ColumnTransformer) ──→ data/processed/v1/
    │                           │
    │                           └─── + TF-IDF vectorizer ──→ data/processed/v2/
    │
    └─── BERTimbau tokenizer ──→ token IDs + attention masks ──→ (in-memory, no disk)
                                                                      │
                                                                      └─── [CLS] embedding (v3.0)
```

## Storage Layout (DVC-tracked)

```
data/
├── raw/                              # DVC tracked, immutable
│   ├── raw.dvc
│   ├── train.csv                     # 18,272 rows × 3 cols (~7.9 MB)
│   ├── test.csv                      # 4 rows × 2 cols (~1.8 KB, sample)
│   └── submission.csv                # Sample submission format
└── processed/                        # DVC tracked, regenerable
    ├── processed.dvc
    ├── X_train_preprocessed.npy      # Feature matrix (train split)
    ├── X_test_preprocessed.npy       # Feature matrix (holdout split)
    ├── y_train.npy                   # Target vector (train split)
    ├── y_test.npy                    # Target vector (holdout split)
    └── X_submission_preprocessed.npy # Feature matrix (competition test set)
```
