# Final Experiment Summary: BI-RADS Classification

## Overview
This document archives the final configuration and rationale for the BI-RADS classification competition.

## Best Model Configuration
- **Model Type**: Ensemble (Transformer + LightGBM + XGBoost)
- **Primary Metric**: macro-F1
- **Cross-Validation**: Stratified 5-Fold

## Feature Set Versions
- **v1.0**: Regex features only (baseline)
- **v2.0**: TF-IDF (word 1-2, char 3-5) + Regex
- **v3.0**: BERTimbau (Portuguese BERT) Contextual Embeddings

## Hyperparameters (Best Single Model: BERTimbau)
- `model_name`: neuralmind/bert-base-portuguese-cased
- `max_length`: 256
- `learning_rate`: 2e-5
- `batch_size`: 16
- `num_epochs`: 5
- `warmup_ratio`: 0.1
- `weight_decay`: 0.01
- `loss_type`: focal (gamma=2.0)

## Performance Summary
- **Regex Baseline macro-F1**: ~0.55
- **TF-IDF + GBM macro-F1**: ~0.65
- **BERTimbau macro-F1**: ~0.72
- **Ensemble macro-F1**: ~0.74

## Error Analysis Summary
- Primary confusion: Class 4 vs Class 2.
- Minority class performance (4, 5, 6) significantly improved by Focal Loss and BERTimbau embeddings.
- Class 5 (29 samples) remains the hardest to predict.

## Reproducibility
- Seed: 42
- Data versioned with DVC (MinIO remote).
- Experiments tracked in MLflow.
