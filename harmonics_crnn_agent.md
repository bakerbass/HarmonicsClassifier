# Agent Task: Train a CRNN for Guitar Note Event Classification (≤ 1 s Inputs)

## 1. Goal

Build a **PyTorch** training pipeline for a **CRNN** that classifies **single-note guitar audio events** (≤ 1 second each) into:

- `harmonic`        – long sustained natural harmonic
- `dead_note`       – muted/percussive notes
- `general_note`    – all other notes

The model will operate on **pre-extracted note snippets** or fixed windows from longer recordings. The agent should create a **reproducible training scaffold**, including data loading, model definition, training loop, and evaluation scripts.

---

## 2. Assumptions About Input Data

Assume an upstream preprocessing step (separate agent or script) has produced:

1. A **metadata table** (CSV or Parquet), e.g. `idmt_guitar_events.csv`, with at least:

   - `note_id` (unique identifier)
   - `audio_path` (relative path to WAV file; can be snippet or full track)
   - `onset_sec` (float; may be 0.0 if pre-snipped)
   - `offset_sec` (float; may be ≤ 1.0 if pre-snipped)
   - `pitch_midi` (int)
   - `expression_style` (string, e.g. HA, DN, NO, etc.)
   - `label_category` (string ∈ {`harmonic`, `dead_note`, `general_note`})

2. Audio characteristics:

   - Mono WAV
   - Sample rate: typically 44.1 kHz (the pipeline should resample as needed)
   - Snippets may vary in length but are ≤ 1.0–1.2 seconds.

The training pipeline should:

- Normalize all audio to a **single target sample rate** (e.g. 22,050 Hz).
- Normalize each example to a **fixed window length** (e.g. exactly 1.0 s) using padding / cropping.
- Normalize audio levels to be between -1.0 to 1.0
---

## 3. Overall Architecture

Implement a **Convolutional Recurrent Neural Network (CRNN)**:

1. **CNN Encoder (2D convolution on log-mel spectrogram)**

   - Input: `[batch_size, 1, n_mels, time_frames]`
   - Several Conv2d → BatchNorm → ReLU → MaxPool blocks.
   - Pool more aggressively in the frequency axis than the time axis to preserve temporal resolution.

2. **RNN Layer**

   - Flatten the frequency dimension after CNN, treat the result as a sequence over time.
   - Feed this into a GRU

3. **Classifier Head**

   - Use the final hidden state (or temporal average) of the RNN.
   - MLP → 3-way classification head for `harmonic`, `dead_note`, `general_note`.

Loss: cross-entropy.  
Metrics: accuracy, macro-F1, per-class precision/recall.

---

## 4. Data Representation

### 4.1 Audio Handling

- Target sample rate: `sr = 22050` (configurable).
- Target duration per event: `target_duration = 1.0` seconds (configurable).

For each sample:

- Load the raw audio from `audio_path`.
- If using full-track audio:
  - Extract window `[onset_sec, onset_sec + target_duration]`, with:
    - Start index `start = int(onset_sec * sr)`
    - End index `end = start + int(target_duration * sr)`
    - Clip to valid range; pad with zeros if next frames are missing.
- If using snippet audio:
  - Center-crop or pad to `target_duration` seconds.

Return a float32 numpy array or torch tensor of shape `[samples]`.

### 4.2 Spectrogram / Features

Default: **log-mel spectrogram**.

Suggested defaults (configurable):

- `n_fft = 1024`
- `hop_length = 256`
- `n_mels = 64` or `80`
- `window = "hann"`
- Convert power spectrogram to log scale using `librosa.power_to_db` or `log(1 + α·x)`.

Final tensor:

- Shape: `[1, n_mels, T]`, where `T ≈ target_duration * sr / hop_length`.

Optional channel additions (nice-to-have, not required):

- Framewise RMS envelope
- Delta features

---

## 5. Project Structure

Create a modular repository structure like:

```
harmonics_crnn/
    config.py
    data/
        __init__.py
        dataset.py
        splits.py
        transforms.py
    models/
        __init__.py
        crnn.py
    training/
        __init__.py
        engine.py
        metrics.py
    utils/
        __init__.py
        logging.py
        seed.py
        io.py
    scripts/
        train.py
        eval.py
        inspect_sample.py
    README.md
```

### 5.1 config.py

Use a simple config object (dataclass or dict), for example:

```python
from dataclasses import dataclass

@dataclass
class Config:
    metadata_path: str = "data/idmt_guitar_events.csv"
    audio_root: str = "data/audio"
    output_dir: str = "runs/crnn_v1"

    sample_rate: int = 22050
    target_duration: float = 1.0

    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 64

    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    rnn_hidden_size: int = 128
    rnn_num_layers: int = 1
    rnn_bidirectional: bool = True

    num_workers: int = 4
    seed: int = 42
```

The agent should implement a way to override these via command-line arguments or a YAML/JSON config file.

---

## 6. Dataset and Dataloaders

### 6.1 NoteEventDataset

Implement `NoteEventDataset` in `data/dataset.py`:

- Constructor takes:
  - `metadata_df`
  - `split` (train/val/test)
  - `config`
  - `transform` (callable to convert waveform to spectrogram and apply augmentation)

- `__getitem__(index)`:
  - Retrieve row from metadata.
  - Resolve `audio_path` as `Path(config.audio_root) / row["audio_path"]`.
  - Load audio with `librosa.load(path, sr=config.sample_rate)`.
  - Apply onset-based cropping if needed.
  - Pad/crop to `target_duration`.
  - Apply `transform` to get spectrogram tensor.
  - Convert label_category to integer index:
    - `harmonic` → 0
    - `dead_note` → 1
    - `general_note` → 2
  - Return `(spectrogram_tensor, label_index)` and optionally metadata.

### 6.2 Train/Val/Test Splits

In `data/splits.py`, implement logic to:

- Group events by track or by `audio_path` stem to avoid leakage.
- Perform deterministic split, e.g. 80/10/10.
- Store split assignment in the metadata DataFrame (a column like `split`).
- Optionally write split info to disk for reproducibility.

---

## 7. CRNN Model Definition

In `models/crnn.py`, implement a PyTorch model like:

```python
import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, n_mels, n_classes=3, rnn_hidden_size=128, rnn_num_layers=1, bidirectional=True):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # more pooling in freq than time
        )

        # RNN input size depends on CNN output shape; compute dynamically
        self.rnn_hidden_size = rnn_hidden_size
        self.bidirectional = bidirectional
        rnn_input_size = 128 * (n_mels // 8)  # adjust if pooling changes

        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        rnn_out_size = rnn_hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        x = self.cnn(x)           # [B, C, F, T']
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2) # [B, T, C, F]
        x = x.reshape(B, T, C * F)  # [B, T, C*F]

        rnn_out, _ = self.rnn(x)  # [B, T, H*dir]
        # Take last time step
        last = rnn_out[:, -1, :]  # [B, H*dir]
        logits = self.classifier(last)
        return logits
```

The agent should:

- Compute `rnn_input_size` dynamically based on the actual CNN output shape (e.g. via a dummy forward pass).
- Make channel sizes and pool patterns configurable.

---

## 8. Training Loop

In `training/engine.py`, implement:

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    # iterate over batches, compute loss, backprop, update optimizer
    # accumulate loss and accuracy metrics
    # return averaged metrics

def evaluate(model, loader, criterion, device):
    model.eval()
    # forward without gradients
    # accumulate loss, accuracy, macro-F1, confusion matrix
    # return metrics
```

Use:

- Loss: `nn.CrossEntropyLoss`, optionally with class weights if label distribution is imbalanced.
- Metrics:
  - Overall accuracy
  - Macro-F1 (treat classes equally)
  - Per-class precision/recall/F1
  - Confusion matrix

---

## 9. Scripts

### 9.1 Training Script (`scripts/train.py`)

Responsibilities:

- Parse arguments:
  - `--config` (optional path to config file)
  - `--metadata-path`
  - `--audio-root`
  - `--output-dir`
- Load config, set random seed, create output directory.
- Load metadata and create train/val splits.
- Instantiate datasets and dataloaders.
- Create CRNN model, optimizer, criterion.
- For each epoch:
  - Train one epoch.
  - Evaluate on validation set.
  - Log metrics.
  - Save best model based on validation macro-F1.

Example CLI:

```bash
python -m scripts.train   --metadata-path data/idmt_guitar_events.csv   --audio-root data/audio   --output-dir runs/crnn_v1
```

### 9.2 Evaluation Script (`scripts/eval.py`)

- Load trained checkpoint and config.
- Run evaluation on val or test split.
- Print:
  - Accuracy
  - Macro-F1
  - Per-class metrics
  - Confusion matrix

---

## 10. Logging, Checkpoints, and Reproducibility

In `utils/logging.py`:

- Simple console logger.
- Optional CSV logging or TensorBoard support.

In `utils/seed.py`:

- Function to set seeds for `random`, `numpy`, `torch` (and CUDA if present).

Checkpointing behavior:

- Save model weights after each epoch to `OUTPUT_DIR/checkpoints/epoch_{epoch}.pt`.
- Also maintain `OUTPUT_DIR/best_model.pt` based on validation macro-F1.

---

## 11. Optional Enhancements (Nice-to-Haves)

If time permits, the agent may implement:

- **Data augmentation**:
  - Random gain (e.g. ±6 dB)
  - Small random time shifts (e.g. ±50 ms)
  - Low-level noise injection
- **Multi-task training**:
  - Additional head for binary `harmonic vs non-harmonic`, sharing the same encoder.
- **LR scheduler**:
  - e.g. `torch.optim.lr_scheduler.ReduceLROnPlateau`.
- **Early stopping**:
  - Stop when validation macro-F1 does not improve for N epochs.

---

## 12. Deliverables

The final scaffold should include:

1. A runnable training script (`scripts/train.py`) that:
   - Trains a CRNN on note-event clips ≤ 1 s.
   - Logs training/validation metrics.
   - Saves checkpoints and best model.

2. An evaluation script (`scripts/eval.py`) that:
   - Loads a trained model.
   - Reports performance on validation/test sets.

3. Clear documentation in `README.md` explaining:
   - How to prepare the metadata CSV and audio structure.
   - How to run training and evaluation.
   - Configurable parameters and their defaults.

This Markdown file should be sufficient for a coding agent to implement the full CRNN training pipeline for your harmonic/dead/general note classification task.
