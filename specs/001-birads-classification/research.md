# Research: BI-RADS Classification

**Feature**: 001-birads-classification
**Date**: 2026-04-04

## R1: Competition Metric — macro-F1 vs accuracy

**Decision**: Use `f1_macro` as the primary optimization metric everywhere (Optuna objective, model selection, reporting).

**Rationale**: The competition evaluates on `sklearn.metrics.f1_score(y_true, y_pred, average='macro')`. With 87.4% of samples in class 2, accuracy is uninformative — a majority-class predictor achieves ~87.4% accuracy but only ~0.07 macro-F1. The current codebase uses accuracy throughout (`METRIC="accuracy"` in `.env`, `SKLEARN_SCORING` dict in `model.py`), which means all prior experiments were optimized for the wrong objective.

**Alternatives considered**:
- `f1_weighted`: Weights by support, still dominated by class 2. Rejected because it doesn't align with competition metric.
- `f1_micro`: Equivalent to accuracy for multiclass. Rejected for same reason.
- Custom scorer: `make_scorer(f1_score, average='macro')` is available if the string shorthand `f1_macro` doesn't work with `cross_val_score`.

**Implementation**: `f1_macro` is a valid string scorer in scikit-learn (`sklearn.metrics.get_scorer('f1_macro')`). Add it to `SKLEARN_SCORING`, `METRIC_DIRECTION`, and `CLASSIFICATION_METRICS` dicts in `src/models/model.py`.

---

## R2: DVC + MinIO Configuration

**Decision**: Use DVC with S3-compatible MinIO remote for versioning `data/raw/` and `data/processed/`. Data-only mode (no `dvc.yaml` pipeline orchestration).

**Rationale**: The existing Docker Compose stack already runs MinIO on port 9002 (API) / 9001 (console). Adding a `dvc` bucket alongside the existing `mlflow` bucket avoids additional infrastructure. DVC tracks large binary files (`.npy` arrays, CSVs) that shouldn't be in git, while MLflow handles model artifacts — no overlap.

**Alternatives considered**:
- Git LFS: Simpler but no deduplication, no remote push/pull semantics for team sharing. Rejected.
- DVC with local remote: No remote sharing capability. Rejected.
- DVC with full pipeline (`dvc.yaml`): Would duplicate Makefile orchestration and add complexity. Rejected per clarification session.

**Configuration**:
```bash
uv add "dvc[s3]"
dvc init
dvc remote add -d minio s3://dvc
dvc remote modify minio endpointurl http://localhost:9002
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin
```

The `minio-setup` service in `docker-compose.yml` needs an additional `mc mb myminio/dvc --ignore-existing` command.

---

## R3: Class Imbalance Handling for macro-F1

**Decision**: Use class-weighted loss functions as the primary strategy, with focal loss as an alternative for transformer models. Avoid SMOTE on raw text.

**Rationale**: Research (SemEval 2025, arxiv 2603.23534) shows that for NLP text classification with extreme imbalance:
1. **Focal loss** (γ=2) with transformers achieves strong macro-F1 (0.75+ on similar tasks)
2. **Class-weighted CE** is more stable and easier to tune than focal loss
3. **SMOTE** is effective on TF-IDF embeddings but not on raw text; applicable to our classical ML pipeline
4. **Threshold tuning** per class can further optimize macro-F1 at inference time

For classical ML (TF-IDF + gradient boosting): use `class_weight='balanced'` and optionally SMOTE on the TF-IDF feature matrix.
For transformers: use focal loss or class-weighted CrossEntropyLoss.

**Alternatives considered**:
- Random oversampling: Risk of overfitting on duplicated minority samples. Rejected as primary method.
- Random undersampling: Loses 87% of data. Rejected.
- Class merging (5+6 → "high suspicion"): Last resort if individual class F1 remains zero.

---

## R4: Pre-trained Portuguese Language Model

**Decision**: Use `neuralmind/bert-base-portuguese-cased` (BERTimbau base) as the primary transformer model.

**Rationale**: BERTimbau is the state-of-the-art Portuguese BERT model, trained on BrWaC (Brazilian Web as Corpus) with whole-word masking. It outperforms mBERT on Portuguese NLP tasks (NER, STS, RTE). The base variant (110M params) is preferred over large (335M params) due to Kaggle memory constraints and overfitting risk on 18K samples.

**Kaggle availability**: A Kaggle dataset `fernandosr85/bertimbau-portuguese-bert` exists, enabling offline use in code competitions (no internet required).

**Alternatives considered**:
- `bert-base-multilingual-cased` (mBERT): Lower Portuguese performance than BERTimbau. Fallback option.
- `neuralmind/bert-large-portuguese-cased`: Better performance but 335M params may exceed Kaggle memory and increase overfitting risk. Deferred.
- Medical-domain BERT (e.g., BioBERT, PubMedBERT): English-only; no Portuguese medical BERT exists. Rejected.

**Tokenization**: Max sequence length = 256 tokens (sufficient for 99%+ of reports at mean 401 chars ≈ ~100-150 tokens). Reports exceeding 256 tokens will be truncated (only affects the longest ~1% of reports).

---

## R5: Kaggle Notebook Compatibility

**Decision**: Produce a self-contained notebook that runs end-to-end on Kaggle without internet, reading data from `/kaggle/input/` and outputting `submission.csv` to `/kaggle/working/`.

**Rationale**: This is a code competition — the only way to submit is via a Kaggle notebook. The notebook must be completely self-contained: all preprocessing, model loading, inference, and output in a single file. Pre-trained model weights are uploaded as Kaggle datasets and referenced via local paths.

**Constraints**:
- CPU runtime: 9 hours max
- GPU runtime: 9 hours max
- Internet: disabled during commit/submission
- Memory: 13GB RAM (CPU) or 16GB (GPU)
- Disk: limited to `/kaggle/working/` for outputs

**Strategy**:
- Develop and iterate locally (with MLflow tracking, DVC data versioning)
- Export final pipeline to a clean Kaggle-compatible notebook
- Upload model artifacts (trained models, BERTimbau weights) as Kaggle datasets
- Dual-path data loading: check for `/kaggle/input/` first, fall back to local `data/raw/`

---

## R6: TF-IDF Configuration for Portuguese Medical Text

**Decision**: Use word-level (1,2)-grams with max_features=10,000 and character-level (3,4,5)-grams with max_features=20,000, combined via horizontal stacking.

**Rationale**: Portuguese medical text has rich morphological patterns (e.g., "calcificações", "lipossubstituídas", "espiculado") that character n-grams capture well. Word-level bigrams capture clinical phrases like "calcificações benignas" vs "calcificações pleomórficas". The combined feature space (~30K features) is manageable for gradient boosting models.

**Alternatives considered**:
- Word-level only: Misses morphological patterns. Rejected.
- Character-level only: Loses word-level semantics. Rejected.
- Higher max_features (50K+): Diminishing returns, slower training. Deferred for tuning.
- Subword tokenization (BPE): Better handled by transformer model. Deferred to Phase 3.

**Preprocessing before TF-IDF**:
- Lowercase normalization
- Remove `<DATA>` tokens (replaced dates, no predictive signal)
- Keep Portuguese accented characters (critical for clinical terms)
- No stemming or lemmatization (preserve morphological features for char n-grams)
