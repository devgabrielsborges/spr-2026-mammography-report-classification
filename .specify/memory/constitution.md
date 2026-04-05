<!--
Sync Impact Report
==================
Version change: 1.0.0 → 1.1.0 (MINOR — metric and version corrections)

Modified principles:
  - VI. Iterative & Baseline-First: "accuracy" → "macro-F1" (competition metric correction)

Modified sections:
  - Technology Stack & Constraints: Python "3.13+" → ">=3.11, <3.13"; competition metric "accuracy" → "macro-averaged F1-score"

Removed sections: None

Templates requiring updates: None (no structural changes)

Follow-up TODOs: None
-->

# Mammography Report Classification — Constitution

## Core Principles

### I. Reproducibility First (NON-NEGOTIABLE)

Every experiment, result, and artifact MUST be fully reproducible by any team member at any time.

- All random seeds MUST be explicitly set and tracked (NumPy, scikit-learn, Optuna, XGBoost, PyTorch, Python `hash`)
- Environment dependencies MUST be pinned to exact versions via `pyproject.toml` / `uv.lock`
- Raw data MUST never be modified in place; all transformations produce new artifacts
- Every training run MUST log: dataset version, feature set, preprocessing steps, hyperparameters, evaluation metrics, and model artifacts
- Notebook outputs MUST be clearable and re-runnable end-to-end without manual intervention

### II. Statistical Rigor

All modeling decisions MUST be grounded in sound statistical methodology. Intuition without evidence is insufficient.

- Stratified cross-validation MUST be used for all model evaluation (minimum 5 folds for this competition's class distribution)
- Train/validation/test splits MUST preserve class distributions via stratification
- Data leakage MUST be prevented: all preprocessing (encoding, scaling, imputation) MUST be fit only on training folds, never on validation or test data
- Feature importance and model comparison MUST use appropriate statistical tests or confidence intervals, not single-point estimates alone
- Class imbalance MUST be quantified and addressed explicitly (via stratification, class weights, or resampling — with documented justification)
- Overfitting MUST be monitored by comparing CV scores against holdout performance; gaps exceeding 2 percentage points require investigation

### III. Experiment Tracking & Versioning

All experiments MUST be tracked systematically. No experiment runs "off the books."

- MLflow MUST be used for logging all training runs: parameters, metrics, artifacts, and tags
- Each experiment MUST be tagged with: model type, feature set version, preprocessing pipeline version, and CV strategy
- Model artifacts (serialized models, plots, confusion matrices) MUST be stored via MLflow's artifact store
- Optuna study results (best params, optimization history, parameter importances) MUST be logged alongside the final model
- Submissions to Kaggle MUST be versioned and linked to their originating experiment run

### IV. Data Integrity & Feature Engineering Discipline

The feature engineering pipeline is the backbone of model quality. It MUST be treated with the same rigor as model code.

- Raw data MUST be stored immutably under `data/raw/` and never overwritten
- All feature extraction logic MUST be deterministic and idempotent — running the same code on the same input MUST always produce identical output
- Feature definitions (regex patterns, clinical categories, derived variables) MUST be documented with their clinical rationale
- New features MUST be validated against the target distribution before inclusion in the model pipeline
- The preprocessing pipeline MUST be a single, composable `sklearn.pipeline.Pipeline` or `ColumnTransformer` to prevent fit/transform mismatches
- Text-based feature extraction (regex on mammography reports) MUST include edge-case tests for malformed or missing report sections

### V. Clean Code & Modularity

Code MUST be readable, maintainable, and structured for collaboration. Clever code that obscures intent is a liability.

- Functions MUST do one thing, be named descriptively, and stay under 50 lines (excluding docstrings)
- No magic numbers: all constants (e.g., `N_TRIALS`, `RANDOM_STATE`, thresholds) MUST be defined in a configuration section or file
- DRY principle: shared logic (evaluation metrics, plotting, data loading) MUST be extracted into reusable functions or modules
- Notebooks MUST follow a linear, top-to-bottom narrative: Config → Load → EDA → Features → Preprocessing → Train → Evaluate → Submit
- Module code under `src/` MUST be importable and testable independently of notebooks
- Type hints MUST be used for all function signatures in `src/` modules
- Dead code, commented-out experiments, and unused imports MUST be removed before any commit

### VI. Iterative & Baseline-First Development

Progress is measured against baselines, not in isolation. Always know where you stand.

- A simple baseline model (e.g., most-frequent class, logistic regression with minimal features) MUST be established before any complex modeling
- Every modeling iteration MUST be compared against the current best baseline using the competition metric (macro-F1)
- Feature engineering and model tuning MUST be done incrementally: one change at a time, measured and logged
- Premature optimization is prohibited: ensemble methods and stacking MUST only be introduced after individual model performance plateaus
- All experiments MUST track the delta from the previous best score to quantify marginal improvement

### VII. Validation-Driven Submission

Submissions MUST be informed by rigorous local validation. Leaderboard probing is not a strategy.

- Local CV score MUST be the primary signal for model selection; public leaderboard score is secondary
- The final model MUST be retrained on the full training set only after hyperparameters are locked via CV
- Submission predictions MUST pass sanity checks: correct shape, no missing values, valid class labels, and plausible class distribution
- A maximum of 2-3 submissions per day SHOULD be targeted to avoid overfitting to the public leaderboard
- Every submission MUST be accompanied by a logged experiment run linking model, features, and CV score

## Technology Stack & Constraints

- **Language**: Python >=3.11, <3.13 managed with [uv](https://github.com/astral-sh/uv)
- **Core Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost, pandas, NumPy, SciPy, matplotlib
- **Experiment Tracking**: MLflow (PostgreSQL backend + MinIO artifact store via Docker Compose)
- **Hyperparameter Optimization**: Optuna (Bayesian optimization, 50+ trials per model, 5-fold stratified CV)
- **Notebook Runtime**: Jupyter (must be compatible with Kaggle kernel environment)
- **Infrastructure**: Docker Compose for local MLflow stack (PostgreSQL, MinIO, MLflow server)
- **Competition**: Kaggle `spr-2026-mammography-report-classification` — multiclass classification, macro-averaged F1-score metric
- **Data Constraints**: Mammography reports in Portuguese; text-based feature extraction via regex; no external data unless competition rules permit

## Development Workflow & Quality Gates

### Workflow

1. **Explore** — EDA in notebooks under `notebooks/`; document findings as markdown cells
2. **Extract** — Implement feature engineering in reusable functions; validate against target
3. **Model** — Train with Optuna HPO, log to MLflow, compare against baseline
4. **Validate** — Confirm CV and holdout metrics are consistent; investigate any overfitting
5. **Submit** — Retrain on full data, generate submission, log experiment, submit to Kaggle

### Quality Gates

- **Gate 1 — Data Integrity**: Raw data checksum matches; no NaN in critical columns; class distribution logged
- **Gate 2 — No Leakage**: Preprocessing fit exclusively on training folds; no future information in features
- **Gate 3 — Reproducibility**: Re-running the notebook from scratch with the same seed produces identical results
- **Gate 4 — Baseline Comparison**: Every model MUST report its delta versus the established baseline
- **Gate 5 — Submission Sanity**: Output CSV has correct columns, correct row count, valid class values, and plausible distribution

## Governance

- This constitution supersedes all ad-hoc practices and informal conventions within the project
- Amendments require: (1) documented rationale, (2) impact analysis on existing experiments, and (3) version bump following semantic versioning
- All code changes (notebooks and `src/` modules) MUST be reviewed against the principles above before merge
- Complexity MUST be justified: adding a new model, feature, or pipeline step requires evidence of marginal improvement over the current best
- Use `README.md` and MLflow tags for runtime development guidance and experiment context

**Version**: 1.1.0 | **Ratified**: 2026-04-04 | **Last Amended**: 2026-04-04
