# SVM-Based Harmonics Classification

This directory contains an alternative approach to the CNN-based harmonics classifier, using traditional machine learning with Support Vector Machines (SVM).

## Overview

While the CNN approach learns features directly from mel spectrograms, the SVM approach uses hand-crafted acoustic features that are more interpretable and computationally efficient.

### Approach Comparison

| Aspect | CNN | SVM |
|--------|-----|-----|
| **Features** | Mel spectrograms (learned) | Hand-crafted audio features |
| **Training time** | Longer (GPU recommended) | Faster (CPU sufficient) |
| **Inference speed** | Moderate | Very fast |
| **Model size** | Large (~10MB) | Small (~1MB) |
| **Interpretability** | Low (black box) | High (feature importance) |
| **Data requirements** | More data preferred | Works well with less data |

## Feature Extraction

The SVM model uses ~147 engineered features across multiple categories:

### 1. MFCC Features (80 features)
- 20 MFCC coefficients
- Statistical summaries: mean, std, min, max for each coefficient
- Captures timbral characteristics

### 2. Spectral Features (~30 features)
- **Spectral Centroid**: "Brightness" of sound
- **Spectral Rolloff**: Frequency below which 85% of energy is contained
- **Spectral Bandwidth**: Width of frequency distribution
- **Spectral Contrast**: Difference between peaks and valleys (7 bands)
- **Zero-Crossing Rate**: Rate of sign changes (percussive vs harmonic)

### 3. Chroma Features (24 features)
- 12 chroma bins (pitch classes)
- Mean and std for each bin
- Captures pitch content

### 4. Energy Features (7 features)
- **RMS Energy**: Overall loudness
- **Onset Strength**: Attack characteristics

### 5. Temporal Features (6 features)
- **Attack Time**: Time to reach peak
- **Energy Decay**: How quickly sound fades
- **Quartile Energy**: Energy distribution over time

## File Structure

```
HarmonicsClassifier/
├── feature_extraction.py       # Audio feature extraction module
├── train_svm.py                # SVM training script
├── evaluate_svm.py             # Model evaluation and comparison
├── inference_svm.py            # Real-time classification
├── SVM_README.md               # This file
└── models/
    └── svm/                    # SVM model directory
        ├── svm_model.pkl       # Trained SVM classifier
        ├── svm_scaler.pkl      # Feature standardization scaler
        ├── feature_config.json # Feature extractor configuration
        ├── svm_results.json    # Training metrics
        └── confusion_matrix_svm.png
```

## Installation

All dependencies are included in the existing `environment.yml`:

```bash
# Activate the existing environment
conda activate harmonics_env

# If you need additional dependencies
pip install scikit-learn tqdm
```

## Usage

### 1. Train SVM Model

Basic training with default parameters:
```bash
python train_svm.py --metadata processed_dataset/metadata.csv --output models/svm
```

With hyperparameter tuning:
```bash
python train_svm.py --metadata processed_dataset/metadata.csv --output models/svm --tune
```

Custom parameters:
```bash
python train_svm.py \
  --metadata processed_dataset/metadata.csv \
  --output models/svm \
  --kernel rbf \
  --C 10.0 \
  --gamma scale \
  --test-size 0.2
```

#### Training Parameters

- `--metadata`: Path to metadata CSV with columns:
  - `source_audio`: Path to audio file
  - `onset_sec`: Start time
  - `duration_sec`: Duration (optional)
  - `label_category`: One of `harmonic`, `dead_note`, `general_note`
  
- `--output`: Output directory for model and results
- `--tune`: Perform grid search for hyperparameters (C, gamma, kernel)
- `--kernel`: SVM kernel type (`rbf`, `linear`, `poly`, `sigmoid`)
- `--C`: Regularization parameter (default: 10.0)
- `--gamma`: Kernel coefficient (`scale`, `auto`, or float)
- `--test-size`: Fraction for test set (default: 0.2)

### 2. Evaluate Model

Evaluate on test data:
```bash
python evaluate_svm.py \
  --metadata test_data/metadata.csv \
  --model models/svm \
  --output results/svm_eval
```

Compare SVM with CNN:
```bash
python evaluate_svm.py \
  --metadata test_data/metadata.csv \
  --model models/svm \
  --compare-cnn models/best_model.pt \
  --output results/comparison
```

### 3. Real-time Inference

Classify a single audio file:
```bash
python inference_svm.py --model models/svm --audio recording.wav
```

Classify multiple files with probabilities:
```bash
python inference_svm.py \
  --model models/svm \
  --audio recording1.wav recording2.wav recording3.wav \
  --probabilities
```

Verbose output with probability bars:
```bash
python inference_svm.py --model models/svm --audio recording.wav -v
```

Save results to JSON:
```bash
python inference_svm.py \
  --model models/svm \
  --audio *.wav \
  --probabilities \
  --output results.json
```

## Python API

### Feature Extraction

```python
from feature_extraction import AudioFeatureExtractor

# Initialize extractor
extractor = AudioFeatureExtractor(
    sr=22050,
    duration=3.0,
    n_mfcc=20,
    n_fft=2048,
    hop_length=512,
    n_chroma=12,
)

# Extract features from file
features = extractor.extract_from_file(
    audio_path='recording.wav',
    onset_sec=0.0,
    duration_sec=3.0,
)

# Extract from audio array
import numpy as np
audio = np.random.randn(66150)  # 3 seconds at 22050 Hz
features = extractor.extract_all_features(audio)

# Get feature names
feature_names = extractor.get_feature_names()
```

### Classification

```python
from inference_svm import HarmonicsSVMClassifier

# Load classifier
classifier = HarmonicsSVMClassifier('models/svm')

# Classify a file
result = classifier.classify_file(
    'recording.wav',
    return_proba=True
)

print(f"Label: {result['label']}")
print(f"Confidence: {result['confidence']:.3f}")
print(f"Probabilities: {result['probabilities']}")

# Classify raw audio
import librosa
audio, sr = librosa.load('recording.wav', sr=22050)
result = classifier.classify_audio(audio, return_proba=True)

# Batch classification
paths = ['file1.wav', 'file2.wav', 'file3.wav']
results = classifier.classify_batch(paths, return_proba=True)
```

## Results Format

### Training Output

`svm_results.json`:
```json
{
  "model_type": "SVM",
  "kernel": "rbf",
  "C": 10.0,
  "gamma": "scale",
  "n_features": 147,
  "n_train": 800,
  "n_test": 200,
  "test_accuracy": 0.8750,
  "test_f1_macro": 0.8723,
  "cv_f1_mean": 0.8654,
  "cv_f1_std": 0.0234,
  "harmonic_precision": 0.92,
  "harmonic_recall": 0.89,
  "harmonic_f1": 0.90,
  "dead_note_precision": 0.85,
  "dead_note_recall": 0.88,
  "dead_note_f1": 0.86,
  "general_note_precision": 0.86,
  "general_note_recall": 0.84,
  "general_note_f1": 0.85
}
```

### Inference Output

```json
{
  "label": "harmonic",
  "class_id": 0,
  "audio_path": "recording.wav",
  "confidence": 0.943,
  "probabilities": {
    "harmonic": 0.943,
    "dead_note": 0.032,
    "general_note": 0.025
  }
}
```

## Hyperparameter Tuning

When using `--tune`, the script performs grid search over:

```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'linear'],
}
```

This tests 40 different parameter combinations using 5-fold cross-validation.

**Note**: Grid search can take 10-30 minutes depending on dataset size.

## Model Interpretability

For **linear kernel** models, you can analyze feature importance:

```python
import pickle
import numpy as np

# Load model
with open('models/svm/svm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Get feature importance (average absolute coefficients)
if model.kernel == 'linear':
    importances = np.mean(np.abs(model.coef_), axis=0)
    # Higher values = more important features
```

The training script automatically generates a feature importance plot for linear models.

## Tips for Best Performance

1. **Hyperparameter Tuning**: Always run with `--tune` initially to find best parameters for your dataset

2. **Class Imbalance**: The script uses `class_weight='balanced'` by default. Disable with `--no-class-weight` if classes are roughly equal

3. **Feature Selection**: For very large datasets, consider reducing features:
   - Use fewer MFCC coefficients (`n_mfcc=13` instead of 20)
   - Skip chroma features if pitch content isn't critical

4. **Kernel Selection**:
   - **RBF kernel**: Best general-purpose choice (default)
   - **Linear kernel**: Faster, more interpretable, try if RBF overfits
   - **Poly kernel**: Rarely better than RBF for audio

5. **Cross-Dataset Validation**: Train on one recording session, test on another to verify generalization

## Troubleshooting

### Out of Memory
- Reduce training set size with `--test-size`
- Use linear kernel instead of RBF
- Process data in batches

### Poor Performance on Test Set
- Run with `--tune` to optimize hyperparameters
- Check for data leakage (ensure train/test split is by recording session)
- Verify audio quality (clipping, noise can affect features)

### Slow Training
- Use linear kernel for faster training
- Skip hyperparameter tuning for quick experiments
- Use fewer CV folds (modify `cv=3` in code)

## Integration with GuitarBot

The SVM classifier can be integrated into the real-time recording workflow:

```python
from inference_svm import HarmonicsSVMClassifier
import sounddevice as sd
import numpy as np

# Initialize classifier
classifier = HarmonicsSVMClassifier('models/svm')

# Record from GuitarBot
duration = 3.0
sample_rate = 22050
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
sd.wait()

# Classify
audio = audio.flatten()
result = classifier.classify_audio(audio, return_proba=True)

print(f"Detected: {result['label']} (confidence: {result['confidence']:.2f})")
```

## Comparison with CNN

Expected performance characteristics:

| Metric | SVM | CNN |
|--------|-----|-----|
| **Test Accuracy** | 85-90% | 88-93% |
| **Training Time** | 2-5 min | 30-60 min |
| **Inference Time** | <10ms | 20-50ms |
| **Model Size** | ~1 MB | ~10 MB |
| **Requires GPU** | No | Recommended |

The SVM approach is recommended when:
- Fast inference is critical
- Limited computational resources
- Model interpretability is important
- Dataset is small (<1000 samples)

The CNN approach is recommended when:
- Maximum accuracy is critical
- Large dataset available (>5000 samples)
- GPU resources available
- Fine-grained spectral patterns are important

## Further Development

Potential improvements:

1. **Feature Engineering**:
   - Add harmonic-to-noise ratio
   - Include pitch tracking features
   - Add temporal modulation features

2. **Ensemble Methods**:
   - Combine SVM with CNN predictions
   - Use voting classifier with multiple kernels

3. **Data Augmentation**:
   - Time stretching
   - Pitch shifting
   - Adding background noise

4. **Active Learning**:
   - Identify uncertain predictions for manual labeling
   - Iteratively improve model with targeted data collection

## References

- Scikit-learn SVM documentation: https://scikit-learn.org/stable/modules/svm.html
- Librosa feature extraction: https://librosa.org/doc/main/feature.html
- MFCC explanation: https://en.wikipedia.org/wiki/Mel-frequency_cepstrum

## Authors

Created as part of the HarmonicsClassifier project for automated guitar harmonics detection.
