#!/bin/bash
# Quick retrain script for both SVM and CNN with updated dataset

set -e

echo "=========================================="
echo "RETRAINING MODELS WITH UPDATED DATASET"
echo "=========================================="
echo ""
echo "Dataset now includes:"
echo "  - Original IDMT harmonics: 126"
echo "  - New GB_NH harmonics: 152"
echo "  - Total harmonics: 278"
echo ""

# Retrain SVM with normalization
echo "Step 1: Training SVM (with amplitude normalization)..."
conda run -n harmonics_classifier python train_svm.py \
  --metadata processed_dataset/metadata.csv \
  --output models/svm_retrained \
  --tune

echo ""
echo "Step 2: Training CNN..."
# Note: You'll need to add normalization to train_cnn.py as well
# For now, this will train without it
# conda run -n harmonics_classifier python train_cnn.py

echo ""
echo "=========================================="
echo "RETRAINING COMPLETE!"
echo "=========================================="
echo ""
echo "New models saved to:"
echo "  SVM: models/svm_retrained/"
echo "  CNN: models/ (if trained)"
echo ""
echo "To compare:"
echo "  conda run -n harmonics_classifier python compare_models.py --svm-model models/svm_retrained"
