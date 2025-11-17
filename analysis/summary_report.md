# Dataset Analysis Summary Report

**Generated:** 2025-11-17 14:31:19

## Dataset Overview

- Total samples: 4922
- Unique audio files: 613
- Dataset subsets: 3

## Class Distribution

- **general_note**: 4535 (92.1%)
- **dead_note**: 261 (5.3%)
- **harmonic**: 126 (2.6%)

## Duration Statistics

| label_category   |   count |     mean |      std |    min |     25% |    50% |     75% |   max |
|:-----------------|--------:|---------:|---------:|-------:|--------:|-------:|--------:|------:|
| dead_note        |     261 | 0.226406 | 0.252429 | 0.0549 | 0.1193  | 0.1611 | 0.2069  | 1.601 |
| general_note     |    4535 | 0.910757 | 0.906577 | 0.0352 | 0.2999  | 0.5247 | 1.1076  | 6.55  |
| harmonic         |     126 | 1.56539  | 0.899203 | 0.079  | 1.21788 | 1.6449 | 1.88877 | 4.477 |

## Baseline Model Performance

### Multi-class Classification
- Original Features (49 features):
  - Test Accuracy: 0.867
- PCA Features (20 components, 95% variance):
  - Test Accuracy: 0.900
  - Change: +0.033

### Binary Classification (Harmonic vs Non-Harmonic)
- Original Features:
  - Test Accuracy: 0.900
  - ROC AUC: 0.900
- PCA Features:
  - Test Accuracy: 0.900
  - ROC AUC: 0.890
  - Change: +0.000 (accuracy), -0.010 (AUC)

**PCA Analysis:** PCA maintained/improved performance while reducing feature space from 49 to 20 dimensions.

These baseline results show what a simple logistic regression model can achieve using hand-crafted audio features with z-score normalization and PCA. A CNN should significantly outperform these results.

## Recommendations

- ⚠️ **Severe class imbalance detected.** Consider using weighted loss or oversampling.
- ⚠️ **Long durations detected** (max: 6.55s). Consider using variable-length inputs or windowing.
- ✅ **Strong baseline performance.** Features show good discriminative power.
- ⚠️ **PCA reduces performance.** Keep original features for maximum accuracy.

---

See accompanying plots for detailed visualizations.
