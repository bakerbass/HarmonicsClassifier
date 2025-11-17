# Agent Task: Train a CNN for Guitar Note Event Classification

## 1. Goal

Build a **PyTorch** training pipeline for a **CNN** that classifies **single-note guitar audio events** into:

- `harmonic`        – long sustained natural harmonic
- `dead_note`       – muted/percussive notes
- `general_note`    – all other notes

The model will operate on **pre-extracted note snippets** with variable durations. The agent should create a **reproducible training scaffold**, including data loading, model definition, training loop, and evaluation scripts.

---

## 2. Assumptions About Input Data

Assume an upstream preprocessing step has produced:

1. A **metadata table** (CSV or Parquet), e.g. `processed_dataset/metadata.csv`, with:

   - `subset_id` (dataset identifier)
   - `source_audio` (path to original WAV file)
   - `source_annotation` (path to XML annotation)
   - `event_index` (event number within file)
   - `onset_sec` (float, start time)
   - `offset_sec` (float, end time)
   - `duration_sec` (float, note duration)
   - `pitch_midi` (int)
   - `string_number` (int, optional)
   - `fret_number` (int, optional)
   - `excitation_style` (string)
   - `expression_style` (string, e.g. HA, DN, NO)
   - `label_category` (string ∈ {`harmonic`, `dead_note`, `general_note`})
   - `is_harmonic` (boolean)
   - `is_dead_note` (boolean)
   - `is_general_note` (boolean)

2. Audio characteristics:

   - Mono WAV files
   - Sample rate: typically 44.1 kHz (pipeline should resample as needed)
   - Variable duration notes (dead notes ~0.1-0.5s, general notes ~0.5-2s, harmonics ~1-5s)

---

## 3. Data Strategy

### 3.1 Audio Normalization

**DO NOT force all samples to fixed length.** Instead:

1. **Adaptive windowing**:
   - Short notes (< 1s): Use full duration, pad to next power-of-2 or fixed minimum (e.g., 0.5s)
   - Medium notes (1-3s): Use full duration or truncate to max_duration (e.g., 3s)
   - Long harmonics (> 3s): Use first 3s (captures attack + sustain) OR use multiple windows

2. **Variable-length handling**:
   - Option A: Pad all to `max_duration` (e.g., 5s) and use masking
   - Option B: Use adaptive pooling to normalize spectrogram time dimension
   - **Recommended**: Option B with adaptive pooling

3. **Resampling**: Target sample rate of **22,050 Hz** (reduces computation while preserving harmonics up to 11kHz)

4. **Amplitude normalization**: Peak normalize to [-1.0, 1.0] or use RMS normalization

### 3.2 Feature Representation

**Primary**: Log-mel spectrogram with optimized parameters for guitar:

```python
sample_rate = 22050
n_fft = 2048          # Higher resolution for harmonic structure
hop_length = 128      # ~174 frames/sec for good temporal resolution
n_mels = 128          # Higher for better frequency resolution
fmin = 80             # E2 (lowest guitar note ~82 Hz)
fmax = 8000           # Covers fundamentals + harmonics
```

**Output shape**: `[1, n_mels, T]` where T varies by duration

**Alternative features to consider**:
- Constant-Q Transform (CQT) - better for musical pitch
- Harmonic-Percussive Source Separation (HPSS) - separate harmonic/percussive components
- Chromagram - pitch class representation

---

## 4. CNN Architecture

### 4.1 Model Design

Use a **ResNet-inspired** architecture with:

1. **Initial Conv Block**: Extract low-level features
2. **Residual Blocks**: 3-4 blocks with skip connections
3. **Adaptive Pooling**: Handle variable time dimension
4. **Global Pooling**: Convert to fixed-size representation
5. **Classifier Head**: Dense layers with dropout

### 4.2 Example Architecture

```python
class HarmonicsClassifier(nn.Module):
    def __init__(self, n_mels=128, n_classes=3, dropout=0.3):
        super().__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 128, blocks=2)
        self.layer2 = self._make_layer(128, 256, blocks=2)
        self.layer3 = self._make_layer(256, 512, blocks=2)
        
        # Adaptive pooling to handle variable time dimension
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, blocks):
        layers = []
        # First block with stride for downsampling
        layers.append(ResidualBlock(in_channels, out_channels, stride=2))
        # Remaining blocks
        for _ in range(blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # x: [B, 1, n_mels, T]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x
```

**Key Design Choices**:
- **Adaptive pooling** handles variable-length inputs
- **Residual connections** help with gradient flow
- **Progressive downsampling** reduces computation
- **Global average pooling** instead of flattening

---

## 5. Project Structure

```
harmonics_classifier/
├── config.py                    # Configuration management
├── data/
│   ├── __init__.py
│   ├── dataset.py              # PyTorch Dataset
│   ├── transforms.py           # Audio -> Spectrogram transforms
│   ├── augmentation.py         # Data augmentation
│   └── splits.py               # Train/val/test splitting
├── models/
│   ├── __init__.py
│   ├── cnn.py                  # CNN architecture
│   └── blocks.py               # Reusable building blocks (ResidualBlock)
├── training/
│   ├── __init__.py
│   ├── trainer.py              # Training loop orchestration
│   ├── metrics.py              # Evaluation metrics
│   └── losses.py               # Loss functions (weighted CE, focal loss)
├── utils/
│   ├── __init__.py
│   ├── logging.py              # Logging utilities
│   ├── visualization.py        # Plot spectrograms, confusion matrices
│   ├── checkpoint.py           # Model checkpointing
│   └── seed.py                 # Reproducibility
├── scripts/
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── predict.py              # Single-file inference
│   └── export_onnx.py          # Model export
├── notebooks/
│   └── exploratory_analysis.ipynb
├── requirements.txt
├── environment.yml
└── README.md
```

---

## 6. Configuration Management

Use **Hydra** or simple YAML config:

```yaml
# config/default.yaml
data:
  metadata_path: "processed_dataset/metadata.csv"
  audio_root: "."
  sample_rate: 22050
  max_duration: 5.0
  min_duration: 0.1

features:
  n_fft: 2048
  hop_length: 128
  n_mels: 128
  fmin: 80
  fmax: 8000

model:
  architecture: "resnet"
  n_classes: 3
  dropout: 0.3

training:
  batch_size: 32
  num_epochs: 100
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: "adam"
  scheduler: "plateau"
  early_stopping_patience: 15
  
  # Class balancing
  use_class_weights: true
  focal_loss_gamma: 2.0  # 0 = standard CE

augmentation:
  enabled: true
  time_stretch_range: [0.9, 1.1]
  gain_db_range: [-6, 6]
  noise_snr_db: [20, 40]
  mixup_alpha: 0.2
  
splits:
  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15
  stratify: true
  group_by_file: true
  random_seed: 42

logging:
  log_interval: 10
  save_dir: "runs"
  tensorboard: true
  save_predictions: true
```

---

## 7. Dataset Implementation

### 7.1 Core Dataset Class

```python
class GuitarNoteDataset(Dataset):
    def __init__(self, metadata_df, config, split='train', transform=None, augment=None):
        self.metadata = metadata_df[metadata_df['split'] == split].reset_index(drop=True)
        self.config = config
        self.transform = transform
        self.augment = augment if split == 'train' else None
        
        # Label encoding
        self.label_map = {'harmonic': 0, 'dead_note': 1, 'general_note': 2}
        
        # Compute class weights for balanced training
        self.class_counts = self.metadata['label_category'].value_counts()
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load audio segment
        audio = self._load_audio_segment(
            row['source_audio'],
            row['onset_sec'],
            row['offset_sec']
        )
        
        # Apply augmentation (training only)
        if self.augment:
            audio = self.augment(audio)
        
        # Convert to spectrogram
        spec = self.transform(audio)
        
        # Get label
        label = self.label_map[row['label_category']]
        
        return {
            'spectrogram': spec,
            'label': label,
            'duration': row['duration_sec'],
            'pitch': row['pitch_midi'],
            'note_id': f"{row['subset_id']}_{row['event_index']}"
        }
```

### 7.2 Data Splitting Strategy

**Critical**: Group by source audio file to prevent data leakage

```python
def create_splits(metadata_df, config):
    """
    Split data by source audio file, not by individual events.
    Ensures no notes from the same recording appear in multiple splits.
    """
    # Group by source audio
    audio_files = metadata_df['source_audio'].unique()
    
    # Stratify by ensuring each split has all classes
    # (group files by dominant class)
    file_labels = {}
    for audio_file in audio_files:
        subset = metadata_df[metadata_df['source_audio'] == audio_file]
        dominant_class = subset['label_category'].mode()[0]
        file_labels[audio_file] = dominant_class
    
    # Split files
    train_files, test_files = train_test_split(
        audio_files, 
        test_size=config.splits.test_ratio,
        stratify=[file_labels[f] for f in audio_files],
        random_state=config.splits.random_seed
    )
    
    train_files, val_files = train_test_split(
        train_files,
        test_size=config.splits.val_ratio / (1 - config.splits.test_ratio),
        stratify=[file_labels[f] for f in train_files],
        random_state=config.splits.random_seed
    )
    
    # Assign splits
    metadata_df['split'] = 'train'
    metadata_df.loc[metadata_df['source_audio'].isin(val_files), 'split'] = 'val'
    metadata_df.loc[metadata_df['source_audio'].isin(test_files), 'split'] = 'test'
    
    return metadata_df
```

---

## 8. Data Augmentation (ESSENTIAL)

```python
class AudioAugmentation:
    def __init__(self, config):
        self.config = config
        
    def __call__(self, audio):
        if random.random() < 0.5:
            audio = self.time_stretch(audio)
        if random.random() < 0.5:
            audio = self.gain_augment(audio)
        if random.random() < 0.3:
            audio = self.add_noise(audio)
        return audio
    
    def time_stretch(self, audio):
        """Stretch/compress time without changing pitch"""
        rate = random.uniform(*self.config.augmentation.time_stretch_range)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def gain_augment(self, audio):
        """Random gain adjustment"""
        gain_db = random.uniform(*self.config.augmentation.gain_db_range)
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def add_noise(self, audio):
        """Add Gaussian noise at random SNR"""
        snr_db = random.uniform(*self.config.augmentation.noise_snr_db)
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise
```

---

## 9. Training Pipeline

### 9.1 Loss Function with Class Balancing

```python
def get_loss_function(class_counts, config):
    """
    Get loss function with class balancing.
    """
    if config.training.use_class_weights:
        # Inverse frequency weighting
        total = sum(class_counts.values())
        weights = [total / class_counts[cls] for cls in ['harmonic', 'dead_note', 'general_note']]
        weights = torch.FloatTensor(weights)
        
        if config.training.focal_loss_gamma > 0:
            return FocalLoss(alpha=weights, gamma=config.training.focal_loss_gamma)
        else:
            return nn.CrossEntropyLoss(weight=weights)
    else:
        return nn.CrossEntropyLoss()
```

### 9.2 Metrics

**Essential metrics**:
- Overall accuracy
- **Balanced accuracy** (accounts for class imbalance)
- Macro-F1, Micro-F1
- Per-class precision, recall, F1
- Confusion matrix
- Matthews Correlation Coefficient (MCC)

```python
def compute_metrics(y_true, y_pred, y_prob):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'micro_f1': f1_score(y_true, y_pred, average='micro'),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'per_class_metrics': classification_report(y_true, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
```

---

## 10. Scripts

### 10.1 Training Script

```bash
python scripts/train.py \
    --config config/default.yaml \
    --experiment_name "resnet_baseline" \
    --gpu 0
```

Features:
- Automatic checkpoint saving (best + latest)
- TensorBoard logging
- Early stopping
- Learning rate scheduling
- Validation every N epochs
- Save misclassified samples for analysis

### 10.2 Evaluation Script

```bash
python scripts/evaluate.py \
    --checkpoint runs/resnet_baseline/best_model.pt \
    --split test \
    --save_predictions
```

Outputs:
- Detailed metrics report
- Confusion matrix visualization
- Per-class performance breakdown
- Misclassified samples analysis
- Calibration curves

### 10.3 Prediction Script

```bash
python scripts/predict.py \
    --checkpoint runs/resnet_baseline/best_model.pt \
    --audio_file path/to/note.wav \
    --visualize
```

Features:
- Single file or batch prediction
- Confidence scores
- Spectrogram visualization
- Grad-CAM attention maps

---

## 11. Evaluation and Analysis

### 11.1 Error Analysis

Automatically identify and save:
- **High-confidence mistakes** (wrong but confident)
- **Low-confidence correct** (right but uncertain)
- **Class confusion pairs** (which classes are confused?)

### 11.2 Model Interpretation

- **Grad-CAM**: Show which time-frequency regions influenced prediction
- **Feature importance**: Which mel bands are most discriminative?
- **t-SNE/UMAP**: Visualize learned embedding space

---

## 12. Baseline Experiments

**Experiment 1: Baseline CNN**
- Standard ResNet18-style architecture
- Log-mel spectrograms
- Basic augmentation
- Target: 75-80% balanced accuracy

**Experiment 2: Feature Ablation**
- Compare log-mel vs CQT vs chromagram
- Test different n_mels values
- Evaluate impact of fmin/fmax range

**Experiment 3: Augmentation Ablation**
- Measure impact of each augmentation technique
- Find optimal augmentation strength

**Experiment 4: Architecture Search**
- Compare ResNet vs EfficientNet vs MobileNet
- Test different depths

---

## 13. Deliverables

1. **Training Pipeline**: End-to-end trainable system
2. **Evaluation Tools**: Comprehensive metrics and visualizations
3. **Inference API**: Easy-to-use prediction interface
4. **Documentation**: Setup guide, usage examples, results analysis
5. **Trained Models**: At least 3 checkpoints with different configurations
6. **Analysis Report**: Performance breakdown, error analysis, recommendations

---

## 14. Success Criteria

**Minimum viable performance**:
- Balanced accuracy > 75%
- Harmonic recall > 70% (main class of interest)
- Dead note precision > 80% (avoid false positives)

**Stretch goals**:
- Balanced accuracy > 85%
- All per-class F1 scores > 0.75
- Inference time < 50ms per sample (CPU)

---

## 15. Next Steps After Training

1. **Error analysis**: Study misclassified examples
2. **Ensemble methods**: Combine multiple models or features
3. **Active learning**: Identify samples needing manual review
4. **Transfer learning**: Pre-train on larger audio datasets
5. **Real-time deployment**: Optimize for low-latency inference
