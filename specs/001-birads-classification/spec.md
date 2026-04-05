# Feature Specification: BI-RADS Classification from Mammography Reports

**Feature Branch**: `001-birads-classification`
**Created**: 2026-04-04
**Status**: Draft
**Input**: User description: "Predict BI-RADS category from mammography report text for SPR 2026 Kaggle competition"

## Clarifications

### Session 2026-04-04

- Q: What data artifacts should DVC version-track? → A: Raw data + processed features (`data/raw/` + `data/processed/`). Model artifacts stay in MLflow.
- Q: How should papermill integrate with the project workflow? → A: Papermill for parameterized experiments — sweep model types, feature sets, hyperparams via notebook parameters, with each run producing an output notebook as a logged artifact.
- Q: Single notebook or separate notebooks per approach? → A: Separate notebooks per approach (`baseline.ipynb`, `tfidf_classical.ipynb`, `transformer.ipynb`, `ensemble.ipynb`), each with its own parameter cell.
- Q: Should DVC pipeline stages orchestrate papermill notebook runs? → A: Yes — `dvc.yaml` defines stages wrapping papermill calls (e.g., `train-baseline: papermill notebooks/baseline.ipynb ...`), enabling full reproducibility via `dvc repro`.
- Q: What MinIO bucket and endpoint for DVC remote? → A: `bucket=dvc-data`, endpoint `http://localhost:9002` (host-mapped from container port 9000), reusing existing Docker Compose MinIO credentials.
- Q: Where should papermill output notebooks be stored? → A: `notebooks/outputs/` as DVC outputs — tracked by DVC, pushed to MinIO, not committed to git.

## Problem Statement

Given the textual findings of a mammography report (in Brazilian Portuguese, with the impression/conclusion section removed), predict the BI-RADS category (0–6) assigned by the radiologist. The measurable outcome is maximizing the macro-averaged F1-score on the hidden Kaggle test set.

## Hypothesis

- **H₀**: Text-based features extracted from mammography report findings cannot predict the BI-RADS category better than the majority-class baseline (always predicting class 2).
- **H₁**: A model trained on textual features (lexical, structural, and/or semantic) from the report findings can predict the BI-RADS category with macro-F1 > 0.50 (substantially above majority-class baseline macro-F1 ≈ 0.07).

The expected input-output relationship: specific clinical descriptors in the findings section (e.g., "espiculado", "distorção arquitetural", "calcificações pleomórficas agrupadas", "resultado benigno") are strongly correlated with the BI-RADS assessment. Higher BI-RADS categories systematically co-occur with suspicious imaging descriptors, while lower categories appear with benign or absent findings.

## Background and Motivation

### Clinical Context

BI-RADS (Breast Imaging Reporting and Data System) is the ACR standard for mammography reporting. Categories range from 0 (incomplete) to 6 (known malignancy). The assignment depends on the radiologist's interpretation of findings: calcifications, masses, architectural distortions, asymmetries, and associated features.

### Prior Work

- **Rule-based NLP** for radiology report classification has been explored extensively (e.g., Percha et al., 2012; Banerjee et al., 2019). Portuguese-language mammography NLP is less studied.
- **Transformer-based models** (BERT, BERTimbau for Portuguese) have shown strong performance on medical text classification tasks, achieving F1 > 0.90 on structured radiology reports.
- **TF-IDF + classical ML** baselines typically achieve moderate performance on radiology NLP tasks but struggle with rare classes.

### Gap This Project Fills

The current codebase uses handcrafted regex features with classical ML models (12 binary pattern features + 1 categorical). This approach:
- Captures only a small subset of clinically relevant patterns
- Uses `accuracy` as the metric (the competition uses **macro-F1**)
- Does not leverage the full text semantics available in the report
- Has no mechanism to handle the extreme class imbalance (class 2 = 87.4%)

### Current Baseline Performance

The existing `notebooks/train.ipynb` pipeline extracts 12 binary regex features + 1 categorical indication classifier, trains 12 model types with Optuna HPO (50 trials, 5-fold CV), and selects the best by accuracy. This serves as the regex-baseline.

## Data Requirements

### Sources

| Dataset | Source | Records | Columns |
|---|---|---|---|
| `train.csv` | Kaggle competition data | 18,272 | `ID`, `report`, `target` |
| `test.csv` | Kaggle competition data | Unknown (4 in sample, full hidden) | `ID`, `report` |

### Format

- **report**: Free-text mammography report in Brazilian Portuguese. Contains sections: "Indicação clínica" (clinical indication), "Achados" (findings), and optionally "Análise comparativa" (comparative analysis). The "impression" section (containing BI-RADS) has been removed.
- **target**: Integer 0–6, representing BI-RADS category.
- **ID**: Unique exam identifier (e.g., "Acc1", "Acc201").
- Report length: mean 401 chars, std 103, range [183, 1,622].

### Class Distribution (Extreme Imbalance)

| Class | BI-RADS | Count | Percentage | Description |
|---|---|---|---|---|
| 0 | Incomplete | 610 | 3.34% | Need additional imaging |
| 1 | Negative | 693 | 3.79% | Routine screening |
| 2 | Benign | 15,968 | 87.39% | Routine screening |
| 3 | Probably benign | 713 | 3.90% | Short-interval follow-up |
| 4 | Suspicious | 214 | 1.17% | Consider biopsy |
| 5 | Highly suggestive | 29 | 0.16% | Appropriate action |
| 6 | Known malignancy | 45 | 0.25% | Treatment |

### Data Versioning & Pipeline Orchestration

- **DVC** MUST track `data/raw/` (original competition CSVs) and `data/processed/` (engineered feature arrays)
- **DVC remote**: MinIO (S3-compatible), bucket `dvc-data`, endpoint `http://localhost:9002` (host-mapped from container port 9000), reusing existing Docker Compose MinIO instance credentials
- **Model artifacts** remain under MLflow's artifact store (no duplication via DVC)
- `.dvc` files and `dvc.lock` MUST be committed to git; large data files MUST be in `.gitignore`
- **DVC pipeline** (`dvc.yaml`) MUST define stages that execute notebooks via papermill (e.g., `train-baseline`, `train-tfidf`, `train-transformer`, `ensemble`), enabling full reproducibility via `dvc repro`
- Each DVC stage specifies its notebook, parameters, dependencies (data files), and outputs (output notebook, metrics)
- **Papermill output notebooks** stored in `notebooks/outputs/`, tracked as DVC outputs (pushed to MinIO, not committed to git)
- `notebooks/outputs/` MUST be in `.gitignore`

### Data Assumptions

- Reports are de-identified (dates replaced with `<DATA>` token)
- The training set is i.i.d. sampled from the same distribution as the test set
- No external data is permitted unless competition rules explicitly allow it
- Reports are well-formed with consistent section structure (minor formatting variations exist)

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Establish Correct Evaluation Baseline (Priority: P1)

Fix the evaluation pipeline to use **macro-F1** (the competition metric) instead of accuracy. Establish a majority-class baseline and a simple regex+logistic regression baseline to set the performance floor.

**Why this priority**: Without the correct metric, all model comparisons are misleading. The extreme class imbalance (87.4% class 2) means accuracy is uninformative — a trivial majority-class predictor scores ~87.4% accuracy but ~0.07 macro-F1.

**Independent Test**: Run the notebook end-to-end; confirm all model comparisons rank by macro-F1; baseline scores are logged.

**Acceptance Scenarios**:

1. **Given** the training data, **When** the majority-class baseline is evaluated, **Then** macro-F1 ≈ 0.07 is reported
2. **Given** the regex feature set from the current codebase, **When** a logistic regression model is trained and evaluated with 5-fold stratified CV, **Then** the macro-F1 baseline score is logged to MLflow

---

### User Story 2 - TF-IDF + Classical ML Pipeline (Priority: P2)

Build a text-based feature extraction pipeline using TF-IDF (word and character n-grams) on the full report text, combined with the existing regex features. Train and compare classical ML models using macro-F1 with class-weight balancing.

**Why this priority**: TF-IDF captures lexical patterns beyond the 12 handcrafted regex features and is fast to iterate on. It tests whether richer text representation improves minority-class recall.

**Independent Test**: Train TF-IDF + LightGBM/XGBoost with class weights; compare macro-F1 against the regex baseline; all runs logged to MLflow.

**Acceptance Scenarios**:

1. **Given** the training data, **When** TF-IDF features (word 1-2 grams + char 3-5 grams) are combined with regex features, **Then** macro-F1 improves over the regex-only baseline
2. **Given** class-weight balancing, **When** the model is evaluated per class, **Then** minority classes (4, 5, 6) show recall > 0.30

---

### User Story 3 - Transformer-Based Classification (Priority: P3)

Fine-tune a pre-trained Portuguese language model (e.g., BERTimbau, multilingual BERT, or a medical-domain model) for BI-RADS classification, leveraging contextual text embeddings.

**Why this priority**: Transformer models can capture semantic relationships and contextual patterns that TF-IDF cannot (e.g., negation handling: "não se observam calcificações suspeitas" vs "calcificações suspeitas agrupadas"). This is expected to be the strongest single-model approach.

**Independent Test**: Fine-tune a transformer model with stratified 5-fold CV; compare macro-F1 against the TF-IDF baseline; validate that the model generalizes (CV-holdout gap < 3 points).

**Acceptance Scenarios**:

1. **Given** a pre-trained Portuguese language model, **When** fine-tuned on the training data with appropriate learning rate and class weights, **Then** macro-F1 exceeds the TF-IDF baseline by at least 5 points
2. **Given** the Kaggle kernel constraints (9h runtime, no internet), **When** the model is deployed as a Kaggle notebook, **Then** inference completes within the time limit

---

### User Story 4 - Ensemble & Submission Optimization (Priority: P4)

Combine the best-performing models (classical + transformer) via ensembling (soft voting, stacking, or blending) and optimize the submission for the public leaderboard.

**Why this priority**: Ensembles typically provide 1-3% improvement over single models by reducing variance. This is the final optimization step before competition deadline.

**Independent Test**: Generate ensemble predictions; compare macro-F1 against the best single model; submit to Kaggle and verify score.

**Acceptance Scenarios**:

1. **Given** predictions from 3+ diverse models (diversity = at least 2 model families and 2 feature sets), **When** ensembled via soft voting or stacking, **Then** macro-F1 meets or exceeds the best single model
2. **Given** the final ensemble model, **When** submitted to Kaggle, **Then** the submission is accepted and scores within expected range of local CV

---

### Edge Cases

- Reports with missing "Achados" section — the model MUST handle reports where standard sections are absent or malformed
- Extremely short reports (< 200 chars) — may lack discriminative content for rare classes
- Class 5 and 6 have very few samples (29 and 45) — models may fail to learn meaningful patterns; augmentation or specialized handling may be needed
- Reports with `<DATA>` date tokens — ensure tokenization handles these correctly
- Class confusion between adjacent BI-RADS categories (e.g., 2 vs 3, 4 vs 5) — the clinical distinction is often subtle

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Pipeline MUST evaluate all models using macro-averaged F1-score as the primary metric
- **FR-002**: Pipeline MUST use stratified K-fold cross-validation (K=5) preserving class proportions
- **FR-003**: Pipeline MUST handle extreme class imbalance via class weights, stratified sampling, or oversampling (with documented approach)
- **FR-004**: Pipeline MUST log all experiment runs (parameters, metrics, artifacts) to MLflow
- **FR-005**: Pipeline MUST produce a valid `submission.csv` with columns `ID` and `target` (integer 0–6)
- **FR-006**: Pipeline MUST be executable as a Kaggle notebook with no internet access and within 9-hour runtime
- **FR-007**: Pipeline MUST retrain the final model on the full training set before generating test predictions
- **FR-008**: Pipeline MUST include per-class precision, recall, and F1 in the evaluation output
- **FR-009**: All text preprocessing MUST be fit only on training folds to prevent data leakage
- **FR-010**: Pipeline MUST support both classical ML (TF-IDF based) and transformer-based approaches
- **FR-011**: Notebooks MUST be executable via papermill with parameterized inputs (model type, feature set, hyperparameters) for PoC experimentation
- **FR-012**: Each papermill run MUST produce an output notebook artifact logged alongside experiment metrics in MLflow

### Key Entities

- **MammographyReport**: The raw text input containing "Indicação clínica", "Achados", and optionally "Análise comparativa" sections. Key attributes: report text, clinical indication type, extracted findings, report length.
- **BIRADSPrediction**: The output classification (integer 0–6) with associated confidence/probability per class. Must map to the standard BI-RADS categories.
- **ExperimentRun**: An MLflow-tracked experiment with model type, feature set version, hyperparameters, CV macro-F1, holdout macro-F1, per-class metrics, and model artifacts.

## Methodology

### Approach 1: Enhanced Regex + TF-IDF + Classical ML

1. **Existing regex features** (12 binary pattern features from current codebase) as domain-knowledge features
2. **TF-IDF vectorization** of full report text: word-level (1,2)-grams + character-level (3,4,5)-grams
3. **Clinical indication classification** (existing categorical feature)
4. **Additional handcrafted features**: report length, section presence flags, count of BI-RADS lexicon terms (e.g., mass descriptors, calcification types, architectural distortion — specific list to be derived from EDA per-class keyword analysis)
5. **Models**: LightGBM, XGBoost, CatBoost with class-weight balancing and Optuna HPO
6. **Metric**: macro-F1 with stratified 5-fold CV

### Approach 2: Transformer Fine-Tuning

1. **Pre-trained model**: BERTimbau (bert-base-portuguese-cased) or neuralmind/bert-large-portuguese-cased
2. **Fine-tuning**: Classification head on [CLS] token, with focal loss or class-weighted cross-entropy for imbalance
3. **Training**: AdamW optimizer, linear warmup + cosine decay, batch size 16-32, 3-5 epochs
4. **Validation**: Stratified 5-fold CV, early stopping on macro-F1
5. **Truncation note**: max_length=256 tokens may truncate ~5% of reports (max 1,622 chars ≈ 400 tokens); truncation rate MUST be profiled during EDA and impact on per-class performance evaluated

### Approach 3: Ensemble

1. Blend predictions from top classical ML model + transformer via weighted soft voting or logistic regression stacking
2. Optimize ensemble weights on the validation fold predictions

### Baseline to Beat

- **Majority-class predictor**: Always predict class 2 → macro-F1 ≈ 0.07
- **Regex + Logistic Regression** (current codebase adapted to macro-F1): expected macro-F1 ≈ 0.40-0.60
- **Target**: macro-F1 > 0.80 (competitive range based on similar Kaggle NLP competitions)

## Evaluation

### Primary Metric

- **Macro-averaged F1-score** (competition metric): `sklearn.metrics.f1_score(y_true, y_pred, average='macro')`
- Justification: Macro-F1 equally weights all 7 classes regardless of support, penalizing models that ignore rare classes (5, 6). This aligns with the clinical importance of correctly classifying suspicious and malignant findings.

### Secondary Metrics

- **Per-class F1, Precision, Recall**: To diagnose which classes the model struggles with
- **Confusion matrix**: To identify systematic misclassification patterns (e.g., class 4 predicted as class 2)
- **Log loss**: To evaluate probability calibration quality
- **CV-holdout gap**: To detect overfitting (gap > 2 percentage points triggers investigation per constitution; gap > 3 pp is minimum acceptable)

### Target Thresholds

| Metric | Minimum Acceptable | Target | Stretch |
|---|---|---|---|
| Macro-F1 (CV) | 0.50 | 0.75 | 0.85+ |
| Minority class recall (4,5,6) | 0.30 | 0.60 | 0.75+ |
| CV-holdout gap | < 3 pp | < 2 pp | < 1 pp |

### Validation Strategy

- **Stratified 5-fold CV** as primary validation (preserves class distribution in each fold)
- **80/20 stratified holdout** as secondary check for overfitting
- **Statistical significance**: Compare models using paired t-test on per-fold F1 scores (p < 0.05) or bootstrapped confidence intervals on the holdout set
- **Submission validation**: Each submission MUST track CV-vs-LB gap; flag if any single submission gap > 5 pp (Pearson r across submissions is aspirational but impractical with limited submission count)

## Scope and Constraints

### In Scope

- Text-based feature engineering from mammography reports (regex, TF-IDF, embeddings)
- Classical ML models (gradient boosting, linear models, SVMs)
- Transformer-based models (BERT variants for Portuguese)
- Ensemble methods (voting, stacking)
- Hyperparameter optimization with Optuna
- Experiment tracking with MLflow
- Kaggle-compatible submission notebook

### Out of Scope

- Image-based features (no mammography images are provided)
- External data augmentation (unless competition rules permit)
- Model distillation or deployment beyond Kaggle notebook
- Real-time inference optimization
- Multi-task learning with auxiliary objectives

### Constraints

- **Compute**: Kaggle kernel limits — CPU: 9h, GPU: 9h. No internet access during submission.
- **Data**: 18,272 training samples with extreme imbalance. No additional labeled data.
- **Timeline**: ~24 days until competition close (2026-04-28)
- **Model size**: Must fit within Kaggle notebook memory (13GB RAM CPU, 16GB GPU). Pre-trained models must be uploaded as Kaggle datasets.
- **Language**: Reports are in Brazilian Portuguese — English-only models require adaptation.

## Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Data: Extreme class imbalance** — Classes 5 (n=29) and 6 (n=45) have too few samples for reliable learning | High | High | Focal loss, class weights, SMOTE on TF-IDF features, stratified CV, data augmentation via paraphrasing |
| **Data: Train/test distribution shift** — Test set may have different class distribution or report styles from different institutions | Medium | High | Robust cross-validation, avoid overfitting to training distribution, ensemble diverse models |
| **Model: Transformer overfitting on small dataset** — Fine-tuning a large model on 18K samples risks memorization | Medium | Medium | Early stopping, low learning rate, weight decay, dropout, freeze lower layers, use smaller model variants |
| **Model: Adjacent-class confusion** — BI-RADS 2 vs 3 and 4 vs 5 are clinically similar and textually ambiguous | High | Medium | Ordinal-aware loss functions, hierarchical classification, error analysis on confused pairs |
| **Infra: Kaggle runtime constraints** — Transformer inference on full test set may exceed 9h limit | Low | High | Profile inference time, use distilled models, batch inference, or fall back to classical ML if needed |
| **Data: Report formatting inconsistency** — Some reports have different section structures or missing sections | Medium | Low | Robust parsing with fallbacks, feature engineering that degrades gracefully on malformed inputs |

## Deliverables

1. **Parameterized notebooks** — separate papermill-executable notebooks per approach: `notebooks/baseline.ipynb`, `notebooks/tfidf_classical.ipynb`, `notebooks/transformer.ipynb`, `notebooks/ensemble.ipynb`; each with a tagged parameter cell and producing output notebook artifacts
2. **Kaggle submission notebook** — final consolidated notebook for Kaggle kernel submission (`submission.csv` generation), compatible with Kaggle kernel constraints
3. **Experiment tracking** — All model runs logged to MLflow with parameters, metrics, artifacts, and tags
4. **Model comparison report** — Summary table of all approaches with macro-F1 scores, per-class metrics, and statistical significance tests
5. **Best model artifacts** — Serialized model, preprocessing pipeline, and configuration for reproducibility
6. **`submission.csv`** — Final predictions for the test set in the required format (`ID`, `target`)

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Macro-F1 on local stratified 5-fold CV exceeds 0.50 (minimum viable) or 0.75 (target)
- **SC-002**: Per-class recall for classes 4, 5, and 6 exceeds 0.30 on the validation set
- **SC-003**: CV-holdout F1 gap is less than 2 percentage points (target) or less than 3 percentage points (minimum acceptable); gaps exceeding 2 pp require investigation per constitution
- **SC-004**: Kaggle submission is accepted and public leaderboard score is within 5 pp of local CV estimate; each submission tracks CV-vs-LB gap
- **SC-005**: All experiment runs are reproducible — re-running with the same seed produces identical results

## Assumptions

- The hidden test set follows the same distribution as the training set (standard Kaggle assumption)
- Pre-trained Portuguese language models (BERTimbau) are available as Kaggle datasets for offline use
- The `<DATA>` date token replacement is consistent across all reports and does not carry predictive signal
- The competition permits the use of pre-trained language models (code competition FAQ does not prohibit it)
- GPU kernels with 9-hour runtime are sufficient for transformer fine-tuning and inference

## References

- ACR BI-RADS Atlas: https://www.acr.org/Clinical-Resources/Clinical-Tools-and-Reference/Reporting-and-Data-Systems/BI-RADS
- Competition page: https://www.kaggle.com/competitions/spr-2026-mammography-report-classification
- BERTimbau (Souza et al., 2020): https://github.com/neuralmind-ai/portuguese-bert
- scikit-learn macro-F1: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
- Optuna: https://optuna.readthedocs.io/
- Focal Loss (Lin et al., 2017): "Focal Loss for Dense Object Detection", ICCV 2017
