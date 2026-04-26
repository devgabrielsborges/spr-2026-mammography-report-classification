# Tasks: BI-RADS Classification from Mammography Reports

**Input**: Design documents from `/specs/001-birads-classification/`
**Prerequisites**: plan.md (required), spec.md (required), research.md, data-model.md, quickstart.md

**Tests**: Not explicitly requested — test tasks are omitted.

**Organization**: Tasks are grouped by user story (US1–US4) from spec.md, enabling independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3, US4)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Install DVC for data versioning and papermill for notebook parameterization. Configure MinIO remote for DVC storage.

- [X] T001 Install DVC with S3 support: run `uv add "dvc[s3]"`
- [X] T002 [P] Add DVC MinIO bucket creation (`mc mb myminio/dvc --ignore-existing`) to the `minio-setup` service entrypoint in `docker-compose.yml`
- [X] T003 Initialize DVC repo and configure MinIO remote: run `dvc init`, then `dvc remote add -d minio s3://dvc`, set `endpointurl`, `access_key_id`, and `secret_access_key` per research.md R2
- [X] T004 Track data directories with DVC: run `dvc add data/raw data/processed`, ensure `data/raw/` and `data/processed/` are in `.gitignore`, commit `.dvc` files to git
- [X] T005 [P] Install papermill: run `uv add papermill`
- [X] T006 [P] Add `notebooks/outputs/` to `.gitignore` for papermill output notebooks

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Fix the evaluation metric to macro-F1 (the competition metric) and add per-class evaluation. Without this, all model comparisons are meaningless due to 87.4% class-2 dominance.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete — all downstream experiments depend on the correct metric.

- [X] T007 Add `f1_macro` entry to `SKLEARN_SCORING`, `METRIC_DIRECTION`, and `CLASSIFICATION_METRICS` dictionaries in `src/models/model.py`. In `SKLEARN_SCORING`: `"f1_macro": "f1_macro"`. In `METRIC_DIRECTION`: `"f1_macro": "maximize"`. In `CLASSIFICATION_METRICS`: `"f1_macro": lambda y, p, **_: f1_score(y, p, average="macro", zero_division=0)`
- [X] T008 [P] Update `.env`: change `METRIC="accuracy"` to `METRIC="f1_macro"`
- [X] T009 Add per-class precision, recall, and F1 logging to `BaseModel.evaluate()` in `src/models/model.py` — when `LOG_ALL_METRICS=True` and `task_type="classification"`, compute and return `sklearn.metrics.classification_report` metrics per BI-RADS class (0–6)
- [X] T010 Refactor `BaseModel._objective()` in `src/models/model.py` to use stratified 5-fold CV explicitly via `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` instead of relying on the default `cv=5` integer, ensuring class proportions are preserved in each fold

**Checkpoint**: Foundation ready — macro-F1 is the primary metric, per-class metrics are logged, and stratified 5-fold CV is the default validation strategy. User story implementation can now begin.

---

## Phase 3: User Story 1 — Establish Correct Evaluation Baseline (Priority: P1) 🎯 MVP

**Goal**: Establish a majority-class baseline and a regex+LR baseline, both evaluated with macro-F1, to set the performance floor for all subsequent experiments.

**Independent Test**: Run `notebooks/baseline.ipynb` end-to-end; confirm majority-class macro-F1 ≈ 0.07; confirm regex+LR baseline macro-F1 is logged to MLflow; all models rank by macro-F1.

### Implementation for User Story 1

- [X] T011 [US1] Create `notebooks/baseline.ipynb` with a tagged `parameters` cell containing: `RANDOM_STATE = 42`, `N_FOLDS = 5`, `METRIC = "f1_macro"`, `DATA_PATH = "data/raw/train.csv"`, `MLFLOW_EXPERIMENT = "birads-baseline"`
- [X] T012 [US1] Implement majority-class baseline in `notebooks/baseline.ipynb`: load `train.csv`, compute majority class (class 2), generate constant predictions, calculate macro-F1 (expected ≈ 0.07), and print per-class classification report
- [X] T013 [US1] Implement regex + logistic regression baseline in `notebooks/baseline.ipynb`: use existing `extract_features()` from `src/preprocessing/preprocess.py`, fit `LogisticRegression(class_weight='balanced')` with stratified 5-fold CV, compute mean macro-F1 across folds, and print per-class metrics
- [X] T014 [US1] Log both baselines to MLflow in `notebooks/baseline.ipynb`: log `cv_f1_macro`, per-class F1 scores, confusion matrix plot, and set tags `baseline=true`, `feature_set_version=v0.0` (majority) / `v1.0` (regex), `cv_strategy=stratified_5fold`

**Checkpoint**: User Story 1 complete — majority-class and regex baselines established with macro-F1. The performance floor is documented in MLflow.

---

## Phase 4: User Story 2 — TF-IDF + Classical ML Pipeline (Priority: P2)

**Goal**: Build a TF-IDF feature extraction pipeline (word + char n-grams) combined with existing regex features. Train gradient boosting models with class-weight balancing and compare against the regex baseline using macro-F1.

**Independent Test**: Train TF-IDF + LightGBM/XGBoost with class weights; compare macro-F1 against the regex baseline from US1; minority classes (4, 5, 6) show recall > 0.30; all runs logged to MLflow.

### Implementation for User Story 2

- [X] T015 [P] [US2] Extend `src/preprocessing/preprocess.py` with a `build_tfidf_pipeline()` function that creates a `ColumnTransformer` combining: (a) TF-IDF word (1,2)-grams (`max_features=10000`), (b) TF-IDF char (3,5)-grams (`max_features=20000`), (c) existing regex binary features as passthrough, (d) existing categorical features via `OneHotEncoder`. Preprocessing before TF-IDF: lowercase, remove `<DATA>` tokens, keep Portuguese accented characters, no stemming (per research.md R6)
- [X] T016 [US2] Create `notebooks/tfidf_classical.ipynb` with a tagged `parameters` cell containing: `RANDOM_STATE = 42`, `N_FOLDS = 5`, `METRIC = "f1_macro"`, `DATA_PATH = "data/raw/train.csv"`, `MODEL_TYPE = "lightgbm"`, `N_TRIALS = 20`, `MAX_WORD_FEATURES = 10000`, `MAX_CHAR_FEATURES = 20000`, `MLFLOW_EXPERIMENT = "birads-tfidf"`
- [X] T017 [US2] Implement TF-IDF + regex combined feature extraction in `notebooks/tfidf_classical.ipynb`: load `train.csv`, call `extract_features()` for regex features, build TF-IDF pipeline using `build_tfidf_pipeline()` from `src/preprocessing/preprocess.py`, fit only on training folds to prevent data leakage
- [X] T018 [US2] Implement model training loop in `notebooks/tfidf_classical.ipynb`: train LightGBM, XGBoost, and CatBoost with `class_weight='balanced'` (or equivalent), Optuna HPO (20 trials per model), stratified 5-fold CV, optimize on macro-F1
- [X] T019 [US2] Add evaluation and visualization in `notebooks/tfidf_classical.ipynb`: per-class precision/recall/F1 table, confusion matrix heatmap, comparison bar chart of all models vs regex baseline, and identify best feature set (word-only, char-only, combined, combined+regex)
- [X] T020 [US2] Log all runs to MLflow in `notebooks/tfidf_classical.ipynb`: log `cv_f1_macro`, `model_type`, `feature_set_version=v2.0`, `cv_strategy=stratified_5fold`, per-class F1, best Optuna params, confusion matrix plot artifact
- [X] T021 [US2] Ensure Kaggle compatibility in `notebooks/tfidf_classical.ipynb`: add dual-path data loading (check `/kaggle/input/spr-2026-mammography-report-classification/` first, fall back to local `data/raw/`), verify no internet-dependent operations

**Checkpoint**: User Story 2 complete — TF-IDF + gradient boosting models trained and compared against regex baseline. Best feature set and model type identified. Proceed to Phase 5 if mean 5-fold CV macro-F1 > 0.50.

---

## Phase 5: User Story 3 — Transformer-Based Classification (Priority: P3)

**Goal**: Fine-tune BERTimbau (Portuguese BERT) for BI-RADS classification using contextual embeddings. Compare against the TF-IDF baseline to quantify the benefit of semantic text understanding (e.g., negation handling).

**Independent Test**: Fine-tune BERTimbau with stratified 5-fold CV; macro-F1 exceeds TF-IDF baseline by ≥ 3 pp; CV-holdout gap < 3 pp (validates generalization); total Kaggle runtime < 9h.

### Implementation for User Story 3

- [X] T022 [US3] Install transformer dependencies: run `uv add transformers accelerate`
- [X] T023 [US3] Create `notebooks/transformer.ipynb` with a tagged `parameters` cell containing: `RANDOM_STATE = 42`, `N_FOLDS = 5`, `METRIC = "f1_macro"`, `DATA_PATH = "data/raw/train.csv"`, `MODEL_NAME = "neuralmind/bert-base-portuguese-cased"`, `MAX_LENGTH = 256`, `BATCH_SIZE = 16`, `LEARNING_RATE = 2e-5`, `NUM_EPOCHS = 5`, `WARMUP_RATIO = 0.1`, `WEIGHT_DECAY = 0.01`, `LOSS_TYPE = "focal"`, `FOCAL_GAMMA = 2.0`, `FREEZE_LAYERS = 0`, `MLFLOW_EXPERIMENT = "birads-transformer"`
- [X] T024 [US3] Implement data loading and tokenization in `notebooks/transformer.ipynb`: load `train.csv`, tokenize reports using BERTimbau tokenizer (`max_length=256`, `padding="max_length"`, `truncation=True`), create PyTorch `Dataset` and `DataLoader`, profile truncation rate (expected < 5% of reports)
- [X] T025 [US3] Implement model architecture in `notebooks/transformer.ipynb`: load BERTimbau from local path or HuggingFace, add classification head on `[CLS]` token (7-class linear layer + dropout), implement focal loss (`γ=2`) and class-weighted `CrossEntropyLoss` as selectable options, support optional freezing of lower N layers
- [X] T026 [US3] Implement training loop in `notebooks/transformer.ipynb`: AdamW optimizer with linear warmup (10% steps) and cosine decay, stratified 5-fold CV, 3–5 epochs per fold, early stopping on validation macro-F1 (patience=2), gradient clipping (max_norm=1.0)
- [X] T027 [US3] Add evaluation in `notebooks/transformer.ipynb`: per-class precision/recall/F1, confusion matrix, CV-holdout gap analysis (flag if gap > 2 pp), comparison vs TF-IDF baseline
- [X] T028 [US3] Log all runs to MLflow in `notebooks/transformer.ipynb`: log `cv_f1_macro`, `model_name`, `feature_set_version=v3.0`, `loss_type`, `freeze_layers`, per-fold F1, learning curves, confusion matrix plot, model checkpoint artifact
- [X] T029 [US3] Profile Kaggle runtime in `notebooks/transformer.ipynb`: measure wall time for tokenization, training (per fold), and inference; estimate total time for full 5-fold training + test inference; flag if projected runtime exceeds 7h (leaving 2h buffer)

**Checkpoint**: User Story 3 complete — transformer model trained and compared against TF-IDF baseline. Proceed to Phase 6 if transformer macro-F1 exceeds TF-IDF by ≥ 3 pp. If transformer overfits (gap > 3 pp) or exceeds Kaggle runtime, fall back to TF-IDF ensemble.

---

## Phase 6: User Story 4 — Ensemble & Submission Optimization (Priority: P4)

**Goal**: Combine the best-performing models from US2 and US3 via ensembling and produce a Kaggle-compatible submission notebook that generates `submission.csv` within the 9h runtime limit.

**Independent Test**: Ensemble macro-F1 meets or exceeds best single model; Kaggle submission is accepted; LB score within 5 pp of local CV.

### Implementation for User Story 4

- [X] T030 [US4] Create `notebooks/ensemble.ipynb` with a tagged `parameters` cell containing: `RANDOM_STATE = 42`, `N_FOLDS = 5`, `METRIC = \"f1_macro\"`, `DATA_PATH = \"data/raw/train.csv\"`, `ENSEMBLE_METHOD = \"soft_voting\"`, `MLFLOW_EXPERIMENT = \"birads-ensemble\"`
- [X] T031 [US4] Implement out-of-fold (OOF) prediction collection in `notebooks/ensemble.ipynb`: load saved OOF probability matrices from top 3+ diverse models (at least 2 model families and 2 feature sets), stack into a single array
- [X] T032 [US4] Implement weighted soft voting in `notebooks/ensemble.ipynb`: optimize class-probability weights across models using validation fold predictions (scipy.optimize or grid search), evaluate ensemble macro-F1
- [X] T033 [US4] Implement logistic regression stacking in `notebooks/ensemble.ipynb`: train a `LogisticRegression(class_weight='balanced')` meta-learner on OOF predictions, evaluate via nested CV, compare vs weighted soft voting
- [X] T034 [US4] Log ensemble results to MLflow in `notebooks/ensemble.ipynb`: log `cv_f1_macro`, `ensemble_method`, component model names, ensemble weights, per-class F1, comparison vs best single model

- [X] T035 [US4] Create `notebooks/kaggle_submission.ipynb` as a self-contained Kaggle-compatible submission notebook: all preprocessing, feature engineering, model loading, and inference in one notebook; reads data from `/kaggle/input/spr-2026-mammography-report-classification/`; loads pre-trained model weights from Kaggle dataset; produces `submission.csv` at `/kaggle/working/submission.csv`; no internet access required; total runtime < 2h target
- [X] T036 [US4] Implement submission sanity checks in `notebooks/kaggle_submission.ipynb`: verify correct columns (`ID`, `target`), correct row count (matches `test.csv`), valid class values (integers 0–6), no missing values, plausible class distribution, and retrain final model on full training set before generating test predictions

**Checkpoint**: User Story 4 complete — ensemble model optimized and Kaggle submission notebook ready. Submit to Kaggle and track CV-vs-LB gap.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final evaluation, error analysis, documentation, and reproducibility validation.

- [X] T037 [P] Evaluate final model on held-out test set: compute macro-F1, per-class P/R/F1, confusion matrix, log loss, and bootstrap confidence intervals (1000 iterations) on holdout set
- [X] T038 [P] Perform error analysis: characterize failure modes per BI-RADS class, identify systematic misclassification patterns (e.g., class 4→2 confusion), analyze misclassified report text for patterns
- [X] T039 Compare CV score vs public leaderboard score for calibration: track CV-vs-LB gap per submission, flag if gap > 5 pp
- [X] T040 [P] Archive final experiment configuration: save best hyperparameters, feature set version, preprocessing config, and model selection rationale to `specs/001-birads-classification/`
- [X] T041 [P] Update `README.md` with new Makefile targets, DVC workflow (`dvc pull`/`dvc push`), experiment replication instructions, and notebook execution guide
- [X] T042 Run `quickstart.md` validation: execute all steps from `specs/001-birads-classification/quickstart.md` on a clean environment to verify end-to-end reproducibility

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion — **BLOCKS all user stories**
- **User Story 1 (Phase 3)**: Depends on Foundational (Phase 2) completion
- **User Story 2 (Phase 4)**: Depends on Foundational (Phase 2) completion; benefits from US1 baseline for comparison but can start independently
- **User Story 3 (Phase 5)**: Depends on Foundational (Phase 2) completion; benefits from US2 TF-IDF baseline for comparison; requires T022 (transformer deps)
- **User Story 4 (Phase 6)**: Depends on US2 and/or US3 producing trained models and OOF predictions
- **Polish (Phase 7)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Phase 2 — No dependencies on other stories
- **User Story 2 (P2)**: Can start after Phase 2 — Compares against US1 baseline but is independently testable
- **User Story 3 (P3)**: Can start after Phase 2 + T022 — Compares against US2 baseline but is independently testable
- **User Story 4 (P4)**: Requires OOF predictions from US2 and/or US3 — Cannot start until at least US2 is complete

### Within Each User Story

- Notebook creation before implementation
- Feature engineering before model training
- Model training before evaluation
- Evaluation before MLflow logging
- Story complete before moving to next priority

### Parallel Opportunities

- T002, T005, T006 can all run in parallel (different files)
- T007 and T008 are in different files and can run in parallel
- T015 (preprocess.py extension) can run in parallel with T011–T014 (different files)
- US1 and US2 implementation can start in parallel once Phase 2 is complete (different notebooks)
- T037, T038, T040, T041 in the Polish phase can run in parallel

---

## Parallel Example: Setup Phase

```bash
# These can run in parallel (different files/commands):
Task T001: "Install DVC with S3 support via uv"
Task T002: "Add DVC bucket to minio-setup in docker-compose.yml"
Task T005: "Install papermill via uv"
Task T006: "Add notebooks/outputs/ to .gitignore"
```

## Parallel Example: User Story 2

```bash
# T015 can run in parallel with US1 tasks (different files):
Task T015: "Extend src/preprocessing/preprocess.py with TF-IDF pipeline"
Task T011: "Create notebooks/baseline.ipynb" (US1, different file)
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (DVC + papermill installation)
2. Complete Phase 2: Foundational (macro-F1 metric fix — **CRITICAL**)
3. Complete Phase 3: User Story 1 (majority-class + regex baselines)
4. **STOP and VALIDATE**: Verify macro-F1 ≈ 0.07 for majority-class; regex baseline logged to MLflow
5. All subsequent experiments now have a valid comparison floor

### Incremental Delivery

1. Setup + Foundational → Correct evaluation metric established
2. User Story 1 → Baselines established (MVP!)
3. User Story 2 → TF-IDF + gradient boosting models compared → Test if macro-F1 > 0.50
4. User Story 3 → Transformer model fine-tuned → Test if macro-F1 improves by ≥ 3 pp over TF-IDF
5. User Story 4 → Ensemble + Kaggle submission → Final competition entry
6. Each story adds value and is independently verifiable

### Decision Gates

- **After US1**: If regex macro-F1 is competitive (> 0.50), consider submitting early as a safe baseline
- **After US2**: If TF-IDF macro-F1 < regex + 2 pp, pivot strategy (investigate feature engineering before moving to transformers)
- **After US3**: If transformer overfits (CV-holdout gap > 3 pp) or exceeds Kaggle runtime, fall back to TF-IDF ensemble
- **After US4**: If CV-vs-LB gap > 5 pp, investigate distribution shift

---

## Notes

- [P] tasks = different files, no dependencies on concurrent tasks
- [Story] label maps task to specific user story for traceability
- Each user story is independently completable and testable
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- The `.env` metric fix (T008) and `model.py` scoring fix (T007) are the highest-priority tasks — nothing else produces valid results without them
- Research decision R2: DVC is used for data versioning only (no `dvc.yaml` pipeline stages)
- All notebooks must be Kaggle-compatible with dual-path data loading
