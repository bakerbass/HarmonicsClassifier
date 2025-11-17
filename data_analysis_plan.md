# Data Analysis Plan: Guitar Harmonics Dataset

## Objective

Perform comprehensive exploratory data analysis (EDA) on the IDMT-SMT-GUITAR dataset to inform model design, understand class distributions, identify potential issues, and establish baseline expectations.

---

## 1. Metadata Analysis

### 1.1 Load and Inspect Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load metadata
df = pd.read_csv('processed_dataset/metadata.csv')

# Basic info
print(f"Total samples: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
```

### 1.2 Class Distribution Analysis

**Questions to answer**:
- How imbalanced are the classes?
- Is the imbalance severe enough to require special handling?
- Are any classes critically underrepresented?

```python
# Overall class distribution
class_counts = df['label_category'].value_counts()
class_percentages = df['label_category'].value_counts(normalize=True) * 100

print("\nClass Distribution:")
print(class_counts)
print("\nClass Percentages:")
print(class_percentages)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
class_counts.plot(kind='bar', ax=axes[0], color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
axes[0].set_title('Class Counts')
axes[0].set_ylabel('Number of Samples')

class_percentages.plot(kind='pie', ax=axes[1], autopct='%1.1f%%')
axes[1].set_title('Class Distribution (%)')
plt.tight_layout()
plt.savefig('analysis/class_distribution.png')
```

**Expected findings**:
- Harmonics likely < 10% of dataset
- General notes likely > 70%
- May need weighted sampling or loss

### 1.3 Duration Analysis by Class

**Critical question**: Do harmonics actually sustain longer than other notes?

```python
# Duration statistics by class
duration_stats = df.groupby('label_category')['duration_sec'].describe()
print("\nDuration Statistics by Class:")
print(duration_stats)

# Visualize distributions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Box plot
df.boxplot(column='duration_sec', by='label_category', ax=axes[0, 0])
axes[0, 0].set_title('Duration Distribution by Class (Box Plot)')
axes[0, 0].set_ylabel('Duration (seconds)')

# Violin plot
sns.violinplot(data=df, x='label_category', y='duration_sec', ax=axes[0, 1])
axes[0, 1].set_title('Duration Distribution by Class (Violin Plot)')
axes[0, 1].set_ylabel('Duration (seconds)')

# Histograms
for idx, label in enumerate(['harmonic', 'dead_note', 'general_note']):
    subset = df[df['label_category'] == label]['duration_sec']
    axes[1, 0].hist(subset, bins=50, alpha=0.6, label=label)
axes[1, 0].set_xlabel('Duration (seconds)')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('Duration Histograms (Overlayed)')
axes[1, 0].legend()

# CDF comparison
for label in ['harmonic', 'dead_note', 'general_note']:
    subset = df[df['label_category'] == label]['duration_sec'].sort_values()
    cdf = np.arange(1, len(subset) + 1) / len(subset)
    axes[1, 1].plot(subset, cdf, label=label, linewidth=2)
axes[1, 1].set_xlabel('Duration (seconds)')
axes[1, 1].set_ylabel('Cumulative Probability')
axes[1, 1].set_title('Duration CDF by Class')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('analysis/duration_analysis.png', dpi=150)
```

**Action items based on findings**:
- If harmonics are 3-5x longer: Use full duration or longer windows
- If dead notes are very short (< 0.5s): Consider separate preprocessing
- Determine appropriate max_duration for padding

### 1.4 Pitch Distribution

```python
# Pitch distribution by class
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, label in enumerate(['harmonic', 'dead_note', 'general_note']):
    subset = df[df['label_category'] == label]
    pitch_counts = subset['pitch_midi'].value_counts().sort_index()
    
    axes[idx].bar(pitch_counts.index, pitch_counts.values)
    axes[idx].set_title(f'{label.replace("_", " ").title()} - Pitch Distribution')
    axes[idx].set_xlabel('MIDI Pitch')
    axes[idx].set_ylabel('Count')
    axes[idx].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('analysis/pitch_distribution.png', dpi=150)

# Check if harmonics cluster at specific pitches
harmonic_pitches = df[df['label_category'] == 'harmonic']['pitch_midi'].value_counts()
print("\nTop 10 Harmonic Pitches:")
print(harmonic_pitches.head(10))
```

**Questions**:
- Do harmonics occur on specific frets (5th, 7th, 12th)?
- Are certain pitch ranges more represented?

### 1.5 Dataset Subset Analysis

```python
# Distribution across dataset subsets
subset_dist = df.groupby(['subset_id', 'label_category']).size().unstack(fill_value=0)
print("\nSamples per Subset and Class:")
print(subset_dist)

# Check for subset bias
subset_percentages = subset_dist.div(subset_dist.sum(axis=1), axis=0) * 100
print("\nClass percentages per subset:")
print(subset_percentages)
```

**Potential issue**: If dataset3 has 90% harmonics and dataset1 has 5%, we need careful splitting

---

## 2. Audio Content Analysis

### 2.1 Sample Spectrograms

Visualize representative examples from each class:

```python
import librosa
import librosa.display

def plot_sample_spectrograms(df, n_samples=3):
    """Plot spectrograms for random samples from each class"""
    fig, axes = plt.subplots(3, n_samples, figsize=(15, 10))
    
    for row_idx, label in enumerate(['harmonic', 'dead_note', 'general_note']):
        subset = df[df['label_category'] == label].sample(n_samples, random_state=42)
        
        for col_idx, (_, sample) in enumerate(subset.iterrows()):
            # Load audio segment
            audio, sr = librosa.load(
                sample['source_audio'],
                sr=22050,
                offset=sample['onset_sec'],
                duration=min(sample['duration_sec'], 3.0)
            )
            
            # Compute spectrogram
            S = librosa.feature.melspectrogram(
                y=audio, sr=sr,
                n_fft=2048, hop_length=128, n_mels=128,
                fmin=80, fmax=8000
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Plot
            ax = axes[row_idx, col_idx]
            img = librosa.display.specshow(
                S_db, sr=sr, hop_length=128,
                x_axis='time', y_axis='mel',
                ax=ax, cmap='viridis'
            )
            ax.set_title(f'{label}\nPitch: {sample["pitch_midi"]}, Dur: {sample["duration_sec"]:.2f}s')
            
            if col_idx == 0:
                ax.set_ylabel('Frequency (Hz)')
            else:
                ax.set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('analysis/sample_spectrograms.png', dpi=150)

plot_sample_spectrograms(df)
```

**Look for**:
- Harmonics: Clear harmonic structure, long sustain, overtones visible
- Dead notes: Sharp attack, quick decay, noisy/percussive character
- General notes: Normal harmonic structure with moderate sustain

### 2.2 Spectral Analysis

Compare spectral characteristics:

```python
def analyze_spectral_features(df, n_samples=50):
    """Compute average spectral features for each class"""
    results = {label: {'spectral_centroid': [], 'spectral_rolloff': [], 
                       'zero_crossing_rate': [], 'rms_energy': []}
               for label in ['harmonic', 'dead_note', 'general_note']}
    
    for label in results.keys():
        subset = df[df['label_category'] == label].sample(min(n_samples, len(df[df['label_category'] == label])))
        
        for _, sample in subset.iterrows():
            audio, sr = librosa.load(
                sample['source_audio'],
                sr=22050,
                offset=sample['onset_sec'],
                duration=min(sample['duration_sec'], 3.0)
            )
            
            # Compute features
            results[label]['spectral_centroid'].append(
                np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
            )
            results[label]['spectral_rolloff'].append(
                np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
            )
            results[label]['zero_crossing_rate'].append(
                np.mean(librosa.feature.zero_crossing_rate(audio))
            )
            results[label]['rms_energy'].append(
                np.mean(librosa.feature.rms(y=audio))
            )
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    features = ['spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate', 'rms_energy']
    
    for idx, feature in enumerate(features):
        ax = axes[idx // 2, idx % 2]
        data = [results[label][feature] for label in ['harmonic', 'dead_note', 'general_note']]
        ax.boxplot(data, labels=['Harmonic', 'Dead Note', 'General'])
        ax.set_title(f'{feature.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis/spectral_features.png', dpi=150)

analyze_spectral_features(df)
```

**Hypotheses to test**:
- Dead notes have higher zero-crossing rate (more noise)
- Harmonics have lower spectral centroid (more fundamental energy)
- Dead notes have sharper RMS envelope

---

## 3. Data Quality Checks

### 3.1 Check for Anomalies

```python
# Very short or very long notes
print("\nPotential Anomalies:")
print(f"Very short notes (< 0.05s): {(df['duration_sec'] < 0.05).sum()}")
print(f"Very long notes (> 10s): {(df['duration_sec'] > 10).sum()}")

# Invalid timestamps
invalid = df[df['offset_sec'] <= df['onset_sec']]
print(f"Invalid timestamps (offset <= onset): {len(invalid)}")

# Check for duplicates
duplicates = df.duplicated(subset=['source_audio', 'onset_sec', 'offset_sec'])
print(f"Duplicate events: {duplicates.sum()}")
```

### 3.2 Missing Audio Files

```python
from pathlib import Path

missing_files = []
for audio_path in df['source_audio'].unique():
    if not Path(audio_path).exists():
        missing_files.append(audio_path)

print(f"\nMissing audio files: {len(missing_files)}")
if missing_files:
    print("Examples:")
    for f in missing_files[:5]:
        print(f"  - {f}")
```

---

## 4. Train/Val/Test Split Preview

### 4.1 File-Based Splitting

```python
# Simulate file-based splitting
from sklearn.model_selection import train_test_split

audio_files = df['source_audio'].unique()
train_files, test_files = train_test_split(audio_files, test_size=0.15, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.15/0.85, random_state=42)

# Check class distribution in splits
for split_name, files in [('Train', train_files), ('Val', val_files), ('Test', test_files)]:
    split_df = df[df['source_audio'].isin(files)]
    print(f"\n{split_name} Split:")
    print(f"  Files: {len(files)}")
    print(f"  Samples: {len(split_df)}")
    print(f"  Class distribution:")
    print(split_df['label_category'].value_counts())
    print(f"  Class percentages:")
    print(split_df['label_category'].value_counts(normalize=True) * 100)
```

**Verify**:
- All splits have all 3 classes
- Class proportions are roughly similar across splits
- No file appears in multiple splits

---

## 5. Analysis Script Structure

Create `scripts/analyze_dataset.py`:

```python
"""
Comprehensive dataset analysis script.

Usage:
    python scripts/analyze_dataset.py --metadata processed_dataset/metadata.csv --output analysis/
"""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Analyze guitar harmonics dataset")
    parser.add_argument('--metadata', required=True, help='Path to metadata CSV')
    parser.add_argument('--output', default='analysis/', help='Output directory for plots')
    parser.add_argument('--n-samples', type=int, default=50, help='Samples for audio analysis')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Run all analyses
    print("Loading metadata...")
    df = load_metadata(args.metadata)
    
    print("\n1. Class distribution analysis...")
    analyze_class_distribution(df, args.output)
    
    print("\n2. Duration analysis...")
    analyze_durations(df, args.output)
    
    print("\n3. Pitch distribution...")
    analyze_pitch_distribution(df, args.output)
    
    print("\n4. Spectrograms...")
    plot_sample_spectrograms(df, args.output)
    
    print("\n5. Spectral features...")
    analyze_spectral_features(df, args.output, args.n_samples)
    
    print("\n6. Data quality checks...")
    check_data_quality(df)
    
    print("\n7. Split preview...")
    preview_splits(df, args.output)
    
    print(f"\nAnalysis complete! Results saved to {args.output}")

if __name__ == "__main__":
    main()
```

---

## 6. Key Questions to Answer

1. **Class Balance**:
   - What's the harmonic:dead:general ratio?
   - Do we need weighted sampling?

2. **Duration Characteristics**:
   - What's a reasonable max_duration?
   - Should we use variable-length inputs?

3. **Spectral Separability**:
   - Are classes visually/spectrally distinct?
   - Which features seem most discriminative?

4. **Data Quality**:
   - Are there annotation errors?
   - Missing or corrupted files?

5. **Splitting Strategy**:
   - File-based split maintains class balance?
   - Enough samples in validation set?

---

## 7. Deliverables

After running analysis:

1. **analysis/summary_report.md** - Written findings and recommendations
2. **analysis/class_distribution.png** - Class counts and percentages
3. **analysis/duration_analysis.png** - Duration comparisons
4. **analysis/pitch_distribution.png** - Pitch histograms per class
5. **analysis/sample_spectrograms.png** - Visual examples
6. **analysis/spectral_features.png** - Feature comparisons
7. **analysis/split_preview.txt** - Train/val/test statistics

---

## 8. Expected Outcomes

**Inform decisions on**:
- Input duration strategy (fixed vs variable)
- Augmentation types and strengths
- Loss function (weighted vs focal)
- Architecture complexity needed
- Feature representation (mel vs CQT vs other)

**Identify issues**:
- Severe class imbalance → adjust strategy
- Poor annotation quality → manual review needed
- Insufficient validation data → adjust split ratios
- High class overlap → may need different features
