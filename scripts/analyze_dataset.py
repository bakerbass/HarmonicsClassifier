"""
Comprehensive dataset analysis script.

Usage:
    python scripts/analyze_dataset.py --metadata processed_dataset/metadata.csv --output analysis/
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """Load metadata CSV file."""
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    return df


def analyze_class_distribution(df: pd.DataFrame, output_dir: Path):
    """Analyze and visualize class distribution."""
    class_counts = df['label_category'].value_counts()
    class_percentages = df['label_category'].value_counts(normalize=True) * 100
    
    print("\nClass Distribution:")
    print(class_counts)
    print("\nClass Percentages:")
    print(class_percentages)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    colors = {'harmonic': '#ff6b6b', 'dead_note': '#4ecdc4', 'general_note': '#45b7d1'}
    class_counts.plot(kind='bar', ax=axes[0], color=[colors.get(x, '#999') for x in class_counts.index])
    axes[0].set_title('Class Counts', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Number of Samples')
    axes[0].set_xlabel('Class')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Pie chart
    class_percentages.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                          colors=[colors.get(x, '#999') for x in class_percentages.index])
    axes[1].set_title('Class Distribution (%)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'class_distribution.png'}")
    plt.close()


def analyze_durations(df: pd.DataFrame, output_dir: Path):
    """Analyze duration distributions by class."""
    duration_stats = df.groupby('label_category')['duration_sec'].describe()
    print("\nDuration Statistics by Class:")
    print(duration_stats)
    
    # Visualize distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Box plot
    df.boxplot(column='duration_sec', by='label_category', ax=axes[0, 0])
    axes[0, 0].set_title('Duration Distribution by Class (Box Plot)')
    axes[0, 0].set_ylabel('Duration (seconds)')
    axes[0, 0].set_xlabel('Class')
    axes[0, 0].get_figure().suptitle('')  # Remove default title
    
    # Violin plot
    sns.violinplot(data=df, x='label_category', y='duration_sec', ax=axes[0, 1])
    axes[0, 1].set_title('Duration Distribution by Class (Violin Plot)')
    axes[0, 1].set_ylabel('Duration (seconds)')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Histograms
    for label in ['harmonic', 'dead_note', 'general_note']:
        subset = df[df['label_category'] == label]['duration_sec']
        axes[1, 0].hist(subset, bins=50, alpha=0.6, label=label)
    axes[1, 0].set_xlabel('Duration (seconds)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Duration Histograms (Overlayed)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # CDF comparison
    for label in ['harmonic', 'dead_note', 'general_note']:
        subset = df[df['label_category'] == label]['duration_sec'].sort_values()
        if len(subset) > 0:
            cdf = np.arange(1, len(subset) + 1) / len(subset)
            axes[1, 1].plot(subset, cdf, label=label, linewidth=2)
    axes[1, 1].set_xlabel('Duration (seconds)')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title('Duration CDF by Class')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'duration_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'duration_analysis.png'}")
    plt.close()


def analyze_pitch_distribution(df: pd.DataFrame, output_dir: Path):
    """Analyze pitch distributions by class."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, label in enumerate(['harmonic', 'dead_note', 'general_note']):
        subset = df[df['label_category'] == label]
        if len(subset) > 0:
            pitch_counts = subset['pitch_midi'].value_counts().sort_index()
            
            axes[idx].bar(pitch_counts.index, pitch_counts.values, color='steelblue')
            axes[idx].set_title(f'{label.replace("_", " ").title()} - Pitch Distribution')
            axes[idx].set_xlabel('MIDI Pitch')
            axes[idx].set_ylabel('Count')
            axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pitch_distribution.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'pitch_distribution.png'}")
    plt.close()
    
    # Print top harmonic pitches
    harmonic_subset = df[df['label_category'] == 'harmonic']
    if len(harmonic_subset) > 0:
        harmonic_pitches = harmonic_subset['pitch_midi'].value_counts()
        print("\nTop 10 Harmonic Pitches:")
        print(harmonic_pitches.head(10))


def analyze_subset_distribution(df: pd.DataFrame):
    """Analyze distribution across dataset subsets."""
    subset_dist = df.groupby(['subset_id', 'label_category']).size().unstack(fill_value=0)
    print("\nSamples per Subset and Class:")
    print(subset_dist)
    
    # Check for subset bias
    subset_percentages = subset_dist.div(subset_dist.sum(axis=1), axis=0) * 100
    print("\nClass percentages per subset:")
    print(subset_percentages)


def plot_sample_spectrograms(df: pd.DataFrame, output_dir: Path, n_samples: int = 3):
    """Plot spectrograms for random samples from each class."""
    print("\nGenerating sample spectrograms...")
    
    fig, axes = plt.subplots(3, n_samples, figsize=(15, 10))
    
    for row_idx, label in enumerate(['harmonic', 'dead_note', 'general_note']):
        subset = df[df['label_category'] == label]
        if len(subset) == 0:
            print(f"  Warning: No samples found for {label}")
            continue
            
        samples = subset.sample(min(n_samples, len(subset)), random_state=42)
        
        for col_idx, (_, sample) in enumerate(samples.iterrows()):
            try:
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
                ax.set_title(f'{label}\nPitch: {sample["pitch_midi"]}, Dur: {sample["duration_sec"]:.2f}s', 
                           fontsize=10)
                
                if col_idx == 0:
                    ax.set_ylabel('Frequency (Hz)')
                else:
                    ax.set_ylabel('')
                    
            except Exception as e:
                print(f"  Error processing sample: {e}")
                axes[row_idx, col_idx].text(0.5, 0.5, 'Error loading', 
                                           ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_spectrograms.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'sample_spectrograms.png'}")
    plt.close()


def extract_comprehensive_features(audio, sr):
    """Extract comprehensive audio features for harmonic identification."""
    features = {}
    
    # Spectral features
    features['spectral_centroid_mean'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    features['spectral_centroid_std'] = np.std(librosa.feature.spectral_centroid(y=audio, sr=sr))
    features['spectral_rolloff_mean'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85))
    features['spectral_rolloff_std'] = np.std(librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.85))
    features['spectral_flatness_mean'] = np.mean(librosa.feature.spectral_flatness(y=audio))
    features['spectral_flatness_std'] = np.std(librosa.feature.spectral_flatness(y=audio))
    features['spectral_bandwidth_mean'] = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    features['spectral_bandwidth_std'] = np.std(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # RMS Energy
    rms = librosa.feature.rms(y=audio)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    # Spectral contrast (differentiates harmonics from noise)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features['spectral_contrast_mean'] = np.mean(contrast)
    features['spectral_contrast_std'] = np.std(contrast)
    
    # Harmonic-to-noise ratio approximation
    # Separate harmonic and percussive components
    harmonic, percussive = librosa.effects.hpss(audio)
    harmonic_energy = np.sum(harmonic ** 2)
    percussive_energy = np.sum(percussive ** 2)
    features['harmonic_ratio'] = harmonic_energy / (harmonic_energy + percussive_energy + 1e-10)
    
    # Pitch-related features
    try:
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        if pitch_values:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
    except:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
    
    # MFCCs (first 13 coefficients)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])
    
    # Chroma features (pitch class distribution)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_std'] = np.std(chroma)
    features['chroma_max'] = np.max(chroma)
    
    # Temporal features
    features['duration'] = len(audio) / sr
    
    # Onset strength (attack characteristics)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    features['onset_strength_mean'] = np.mean(onset_env)
    features['onset_strength_max'] = np.max(onset_env)
    
    return features


def analyze_spectral_features(df: pd.DataFrame, output_dir: Path, n_samples: int = 50):
    """Compute and compare spectral features for each class."""
    print("\nAnalyzing spectral features...")
    
    results = {label: {'spectral_centroid': [], 'spectral_rolloff': [], 
                       'zero_crossing_rate': [], 'rms_energy': [],
                       'spectral_flatness': [], 'harmonic_ratio': [],
                       'spectral_contrast': []}
               for label in ['harmonic', 'dead_note', 'general_note']}
    
    for label in results.keys():
        subset = df[df['label_category'] == label]
        if len(subset) == 0:
            continue
            
        samples = subset.sample(min(n_samples, len(subset)), random_state=42)
        
        for _, sample in samples.iterrows():
            try:
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
                results[label]['spectral_flatness'].append(
                    np.mean(librosa.feature.spectral_flatness(y=audio))
                )
                
                # Harmonic ratio
                harmonic, percussive = librosa.effects.hpss(audio)
                h_energy = np.sum(harmonic ** 2)
                p_energy = np.sum(percussive ** 2)
                results[label]['harmonic_ratio'].append(
                    h_energy / (h_energy + p_energy + 1e-10)
                )
                
                # Spectral contrast
                contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
                results[label]['spectral_contrast'].append(np.mean(contrast))
                
            except Exception as e:
                print(f"  Error processing sample: {e}")
                continue
    
    # Visualize
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    features = ['spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate', 
                'rms_energy', 'spectral_flatness', 'harmonic_ratio', 'spectral_contrast']
    
    for idx, feature in enumerate(features):
        ax = axes[idx // 3, idx % 3]
        data = [results[label][feature] for label in ['harmonic', 'dead_note', 'general_note'] 
                if len(results[label][feature]) > 0]
        labels = [label for label in ['harmonic', 'dead_note', 'general_note'] 
                 if len(results[label][feature]) > 0]
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax.set_title(f'{feature.replace("_", " ").title()}', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
    
    # Remove extra subplots
    for idx in range(len(features), 9):
        fig.delaxes(axes[idx // 3, idx % 3])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'spectral_features.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'spectral_features.png'}")
    plt.close()


def check_data_quality(df: pd.DataFrame):
    """Check for data quality issues."""
    print("\n" + "="*60)
    print("DATA QUALITY CHECKS")
    print("="*60)
    
    # Very short or very long notes
    print("\nPotential Anomalies:")
    print(f"  Very short notes (< 0.05s): {(df['duration_sec'] < 0.05).sum()}")
    print(f"  Very long notes (> 10s): {(df['duration_sec'] > 10).sum()}")
    
    # Invalid timestamps
    invalid = df[df['offset_sec'] <= df['onset_sec']]
    print(f"  Invalid timestamps (offset <= onset): {len(invalid)}")
    
    # Check for duplicates
    duplicates = df.duplicated(subset=['source_audio', 'onset_sec', 'offset_sec'])
    print(f"  Duplicate events: {duplicates.sum()}")
    
    # Missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Missing audio files
    missing_files = []
    print("\nChecking audio file existence...")
    for audio_path in df['source_audio'].unique():
        if not Path(audio_path).exists():
            missing_files.append(audio_path)
    
    print(f"  Missing audio files: {len(missing_files)}")
    if missing_files:
        print("  Examples:")
        for f in missing_files[:5]:
            print(f"    - {f}")


def preview_splits(df: pd.DataFrame, output_dir: Path):
    """Preview train/val/test splits."""
    print("\n" + "="*60)
    print("TRAIN/VAL/TEST SPLIT PREVIEW")
    print("="*60)
    
    audio_files = df['source_audio'].unique()
    
    # Create file-to-dominant-class mapping
    file_labels = {}
    for audio_file in audio_files:
        subset = df[df['source_audio'] == audio_file]
        dominant_class = subset['label_category'].mode()[0] if len(subset) > 0 else 'unknown'
        file_labels[audio_file] = dominant_class
    
    try:
        # Split files
        train_files, test_files = train_test_split(
            audio_files, 
            test_size=0.15,
            stratify=[file_labels[f] for f in audio_files],
            random_state=42
        )
        
        train_files, val_files = train_test_split(
            train_files,
            test_size=0.15 / 0.85,
            stratify=[file_labels[f] for f in train_files],
            random_state=42
        )
    except ValueError as e:
        print(f"Warning: Stratified split failed ({e}), using random split")
        train_files, test_files = train_test_split(audio_files, test_size=0.15, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.15/0.85, random_state=42)
    
    # Analyze splits
    split_summary = []
    for split_name, files in [('Train', train_files), ('Val', val_files), ('Test', test_files)]:
        split_df = df[df['source_audio'].isin(files)]
        
        print(f"\n{split_name} Split:")
        print(f"  Files: {len(files)}")
        print(f"  Samples: {len(split_df)}")
        print(f"  Class distribution:")
        class_dist = split_df['label_category'].value_counts()
        print(class_dist)
        print(f"  Class percentages:")
        class_pct = split_df['label_category'].value_counts(normalize=True) * 100
        print(class_pct)
        
        split_summary.append({
            'split': split_name,
            'files': len(files),
            'samples': len(split_df),
            **class_dist.to_dict()
        })
    
    # Save summary
    summary_df = pd.DataFrame(split_summary)
    summary_path = output_dir / 'split_preview.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSplit summary saved to: {summary_path}")


def optimize_pca_components(X_train_scaled, X_test_scaled, y_train, y_test, output_dir: Path):
    """Optimize number of PCA components for best accuracy vs dimensionality tradeoff."""
    print("\n" + "="*60)
    print("PCA OPTIMIZATION (Multi-class Classification)")
    print("="*60)
    
    # Test different variance thresholds
    variance_thresholds = [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
    # Also test fixed number of components
    fixed_components = [5, 10, 15, 20, 25, 30]
    
    results = []
    
    print("\nTesting variance thresholds...")
    for threshold in variance_thresholds:
        pca = PCA(n_components=threshold, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # Multi-class logistic regression
        lr = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
        lr.fit(X_train_pca, y_train)
        
        accuracy = lr.score(X_test_pca, y_test)
        
        # Calculate macro F1 score for multi-class
        from sklearn.metrics import f1_score
        y_pred = lr.predict(X_test_pca)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        results.append({
            'method': f'{int(threshold*100)}% variance',
            'n_components': X_train_pca.shape[1],
            'variance_explained': pca.explained_variance_ratio_.sum(),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'efficiency': accuracy / X_train_pca.shape[1]  # Accuracy per component
        })
        print(f"  {int(threshold*100)}% variance -> {X_train_pca.shape[1]} components: "
              f"Acc={accuracy:.4f}, F1={f1_macro:.4f}")
    
    print("\nTesting fixed number of components...")
    max_components = min(X_train_scaled.shape[0], X_train_scaled.shape[1])
    for n_comp in fixed_components:
        if n_comp >= max_components:
            continue
            
        pca = PCA(n_components=n_comp, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        
        # Multi-class logistic regression
        lr = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
        lr.fit(X_train_pca, y_train)
        
        accuracy = lr.score(X_test_pca, y_test)
        
        # Calculate macro F1 score
        y_pred = lr.predict(X_test_pca)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        results.append({
            'method': f'{n_comp} components',
            'n_components': n_comp,
            'variance_explained': pca.explained_variance_ratio_.sum(),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'efficiency': accuracy / n_comp
        })
        print(f"  {n_comp} components ({pca.explained_variance_ratio_.sum():.1%} variance): "
              f"Acc={accuracy:.4f}, F1={f1_macro:.4f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Find optimal configurations
    best_accuracy = results_df.loc[results_df['accuracy'].idxmax()]
    best_f1 = results_df.loc[results_df['f1_macro'].idxmax()]
    best_efficiency = results_df.loc[results_df['efficiency'].idxmax()]
    
    print("\n" + "="*60)
    print("OPTIMAL CONFIGURATIONS")
    print("="*60)
    print(f"\nBest Accuracy: {best_accuracy['method']}")
    print(f"  Components: {best_accuracy['n_components']}, Accuracy: {best_accuracy['accuracy']:.4f}, F1: {best_accuracy['f1_macro']:.4f}")
    
    print(f"\nBest F1 Score: {best_f1['method']}")
    print(f"  Components: {best_f1['n_components']}, Accuracy: {best_f1['accuracy']:.4f}, F1: {best_f1['f1_macro']:.4f}")
    
    print(f"\nBest Efficiency (Acc/Component): {best_efficiency['method']}")
    print(f"  Components: {best_efficiency['n_components']}, Accuracy: {best_efficiency['accuracy']:.4f}, F1: {best_efficiency['f1_macro']:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy vs Components
    axes[0, 0].plot(results_df['n_components'], results_df['accuracy'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].axvline(x=best_accuracy['n_components'], color='r', linestyle='--', 
                       label=f"Best: {best_accuracy['n_components']} comp")
    axes[0, 0].set_xlabel('Number of Components')
    axes[0, 0].set_ylabel('Test Accuracy')
    axes[0, 0].set_title('Accuracy vs Number of Components')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # F1 Score vs Components
    axes[0, 1].plot(results_df['n_components'], results_df['f1_macro'], 'go-', linewidth=2, markersize=8)
    axes[0, 1].axvline(x=best_f1['n_components'], color='r', linestyle='--',
                       label=f"Best: {best_f1['n_components']} comp")
    axes[0, 1].set_xlabel('Number of Components')
    axes[0, 1].set_ylabel('F1 Score (Macro)')
    axes[0, 1].set_title('F1 Score vs Number of Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Efficiency (Accuracy per component)
    axes[1, 0].plot(results_df['n_components'], results_df['efficiency'], 'mo-', linewidth=2, markersize=8)
    axes[1, 0].axvline(x=best_efficiency['n_components'], color='r', linestyle='--',
                       label=f"Best: {best_efficiency['n_components']} comp")
    axes[1, 0].set_xlabel('Number of Components')
    axes[1, 0].set_ylabel('Efficiency (Accuracy / Component)')
    axes[1, 0].set_title('Model Efficiency vs Dimensionality')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Variance Explained vs Components
    axes[1, 1].plot(results_df['n_components'], results_df['variance_explained'], 'co-', linewidth=2, markersize=8)
    axes[1, 1].axhline(y=0.95, color='gray', linestyle='--', label='95% variance')
    axes[1, 1].set_xlabel('Number of Components')
    axes[1, 1].set_ylabel('Variance Explained')
    axes[1, 1].set_title('Variance Explained vs Number of Components')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_optimization.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_dir / 'pca_optimization.png'}")
    plt.close()
    
    # Save results table
    results_df.to_csv(output_dir / 'pca_optimization_results.csv', index=False)
    print(f"  Saved: {output_dir / 'pca_optimization_results.csv'}")
    
    # Determine recommended configuration (balance accuracy, F1, and efficiency)
    # Use weighted score: 50% accuracy, 30% F1, 15% efficiency, 5% variance
    results_df['score'] = (0.50 * results_df['accuracy'] + 
                          0.30 * results_df['f1_macro'] + 
                          0.15 * results_df['efficiency'] + 
                          0.05 * results_df['variance_explained'])
    recommended = results_df.loc[results_df['score'].idxmax()]
    
    print(f"\n🎯 RECOMMENDED CONFIGURATION: {recommended['method']}")
    print(f"   Components: {recommended['n_components']}")
    print(f"   Variance: {recommended['variance_explained']:.1%}")
    print(f"   Accuracy: {recommended['accuracy']:.4f}")
    print(f"   F1 Score: {recommended['f1_macro']:.4f}")
    print(f"   Efficiency: {recommended['efficiency']:.6f}")
    
    return int(recommended['n_components'])


def train_logistic_regression_baseline(df: pd.DataFrame, output_dir: Path, n_samples: int = 200):
    """Train logistic regression models using extracted features."""
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION BASELINE MODELS")
    print("="*60)
    
    # Extract features for all samples
    print(f"\nExtracting features from up to {n_samples} samples per class...")
    
    features_list = []
    labels_list = []
    label_map = {'harmonic': 0, 'dead_note': 1, 'general_note': 2}
    
    for label in ['harmonic', 'dead_note', 'general_note']:
        subset = df[df['label_category'] == label]
        if len(subset) == 0:
            continue
        
        samples = subset.sample(min(n_samples, len(subset)), random_state=42)
        print(f"  Processing {len(samples)} {label} samples...")
        
        for _, sample in samples.iterrows():
            try:
                audio, sr = librosa.load(
                    sample['source_audio'],
                    sr=22050,
                    offset=sample['onset_sec'],
                    duration=min(sample['duration_sec'], 3.0)
                )
                
                # Extract comprehensive features
                features = extract_comprehensive_features(audio, sr)
                features_list.append(features)
                labels_list.append(label_map[label])
                
            except Exception as e:
                continue
    
    if len(features_list) < 10:
        print("  Not enough samples for training. Skipping baseline models.")
        return
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features_list)
    y = np.array(labels_list)
    
    print(f"\nExtracted {len(features_df)} samples with {len(features_df.columns)} features")
    
    # Handle any NaN or inf values
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(features_df.mean())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features (z-score normalization)
    print("\nApplying z-score normalization...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compute and display z-score statistics
    print(f"  Training set - Mean: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.6f}")
    print(f"  Test set - Mean: {X_test_scaled.mean():.6f}, Std: {X_test_scaled.std():.6f}")
    
    # Optimize PCA components using multi-class classification
    optimal_n_components = optimize_pca_components(
        X_train_scaled, X_test_scaled, y_train, y_test, output_dir
    )
    
    # Binary classification setup (for additional evaluation)
    y_binary = (y == 0).astype(int)
    y_train_binary = (y_train == 0).astype(int)
    y_test_binary = (y_test == 0).astype(int)
    
    # Apply PCA with optimal configuration
    print("\n" + "="*60)
    print("APPLYING OPTIMAL PCA CONFIGURATION")
    print("="*60)
    pca_optimal = PCA(n_components=optimal_n_components, random_state=42)
    X_train_pca_optimal = pca_optimal.fit_transform(X_train_scaled)
    X_test_pca_optimal = pca_optimal.transform(X_test_scaled)
    
    print(f"\nOptimal PCA configuration:")
    print(f"  Original features: {X_train_scaled.shape[1]}")
    print(f"  Optimal components: {X_train_pca_optimal.shape[1]}")
    print(f"  Variance explained: {pca_optimal.explained_variance_ratio_.sum():.4f}")
    print(f"  Dimensionality reduction: {(1 - X_train_pca_optimal.shape[1]/X_train_scaled.shape[1]):.1%}")
    
    # Also keep standard 95% variance PCA for comparison
    pca_95 = PCA(n_components=0.95, random_state=42)
    X_train_pca_95 = pca_95.fit_transform(X_train_scaled)
    X_test_pca_95 = pca_95.transform(X_test_scaled)
    
    print(f"\nStandard 95% variance PCA:")
    print(f"  Components: {X_train_pca_95.shape[1]}")
    print(f"  Variance explained: {pca_95.explained_variance_ratio_.sum():.4f}")
    
    # Plot PCA explained variance (using optimal PCA)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cumulative explained variance
    cumsum_var = np.cumsum(pca_optimal.explained_variance_ratio_)
    axes[0].plot(range(1, len(cumsum_var) + 1), cumsum_var, 'b-', linewidth=2, label='Optimal PCA')
    axes[0].axhline(y=0.95, color='r', linestyle='--', label='95% Variance', alpha=0.5)
    axes[0].axvline(x=optimal_n_components, color='g', linestyle='--', 
                    label=f'Optimal: {optimal_n_components} Components')
    axes[0].set_xlabel('Number of Components')
    axes[0].set_ylabel('Cumulative Explained Variance')
    axes[0].set_title('PCA: Cumulative Explained Variance (Optimal Config)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Individual component variance
    n_bars = min(20, len(pca_optimal.explained_variance_ratio_))
    axes[1].bar(range(1, n_bars + 1), 
                pca_optimal.explained_variance_ratio_[:n_bars], color='steelblue')
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Explained Variance Ratio')
    axes[1].set_title(f'Top {n_bars} Principal Components (Optimal PCA)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_dir / 'pca_analysis.png'}")
    plt.close()
    
    # Visualize first 2 PCA components (optimal configuration)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    labels_names = ['harmonic', 'dead_note', 'general_note']
    
    # Training set
    for i, label in enumerate(labels_names):
        mask = y_train == i
        if mask.sum() > 0:
            axes[0].scatter(X_train_pca_optimal[mask, 0], X_train_pca_optimal[mask, 1], 
                          c=colors[i], label=label, alpha=0.6, s=30)
    axes[0].set_xlabel('First Principal Component')
    axes[0].set_ylabel('Second Principal Component')
    axes[0].set_title(f'PCA Visualization - Training Set ({optimal_n_components} components)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Test set
    for i, label in enumerate(labels_names):
        mask = y_test == i
        if mask.sum() > 0:
            axes[1].scatter(X_test_pca_optimal[mask, 0], X_test_pca_optimal[mask, 1], 
                          c=colors[i], label=label, alpha=0.6, s=30)
    axes[1].set_xlabel('First Principal Component')
    axes[1].set_ylabel('Second Principal Component')
    axes[1].set_title(f'PCA Visualization - Test Set ({optimal_n_components} components)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pca_visualization.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'pca_visualization.png'}")
    plt.close()
    
    # Train models on both original and PCA-reduced features
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # 1. Multi-class with original features
    print("\n1. Multi-class Logistic Regression (Original Features)")
    lr_multi_orig = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
    lr_multi_orig.fit(X_train_scaled, y_train)
    y_pred_multi_orig = lr_multi_orig.predict(X_test_scaled)
    acc_multi_orig = (y_pred_multi_orig == y_test).mean()
    
    print(f"  Test Accuracy: {acc_multi_orig:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_multi_orig, 
                                target_names=['harmonic', 'dead_note', 'general_note']))
    
    # 2. Multi-class with optimal PCA features
    print("\n2. Multi-class Logistic Regression (Optimal PCA Features)")
    lr_multi_pca = LogisticRegression(max_iter=1000, random_state=42, multi_class='ovr')
    lr_multi_pca.fit(X_train_pca_optimal, y_train)
    y_pred_multi_pca = lr_multi_pca.predict(X_test_pca_optimal)
    acc_multi_pca = (y_pred_multi_pca == y_test).mean()
    
    print(f"  Test Accuracy: {acc_multi_pca:.4f}")
    print(f"  Accuracy change: {acc_multi_pca - acc_multi_orig:+.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_multi_pca, 
                                target_names=['harmonic', 'dead_note', 'general_note']))
    
    # 3. Binary with original features
    print("\n3. Binary Logistic Regression (Original Features)")
    lr_binary_orig = LogisticRegression(max_iter=1000, random_state=42)
    lr_binary_orig.fit(X_train_scaled, y_train_binary)
    y_pred_binary_orig = lr_binary_orig.predict(X_test_scaled)
    y_pred_proba_binary_orig = lr_binary_orig.predict_proba(X_test_scaled)[:, 1]
    acc_binary_orig = (y_pred_binary_orig == y_test_binary).mean()
    roc_auc_orig = roc_auc_score(y_test_binary, y_pred_proba_binary_orig)
    
    print(f"  Test Accuracy: {acc_binary_orig:.4f}")
    print(f"  ROC AUC: {roc_auc_orig:.4f}")
    print("\nBinary Classification Report:")
    print(classification_report(y_test_binary, y_pred_binary_orig, 
                                target_names=['Non-Harmonic', 'Harmonic']))
    
    # 4. Binary with optimal PCA features
    print("\n4. Binary Logistic Regression (Optimal PCA Features)")
    lr_binary_pca = LogisticRegression(max_iter=1000, random_state=42)
    lr_binary_pca.fit(X_train_pca_optimal, y_train_binary)
    y_pred_binary_pca = lr_binary_pca.predict(X_test_pca_optimal)
    y_pred_proba_binary_pca = lr_binary_pca.predict_proba(X_test_pca_optimal)[:, 1]
    acc_binary_pca = (y_pred_binary_pca == y_test_binary).mean()
    roc_auc_pca = roc_auc_score(y_test_binary, y_pred_proba_binary_pca)
    
    print(f"  Test Accuracy: {acc_binary_pca:.4f} ({acc_binary_pca - acc_binary_orig:+.4f})")
    print(f"  ROC AUC: {roc_auc_pca:.4f} ({roc_auc_pca - roc_auc_orig:+.4f})")
    print("\nBinary Classification Report:")
    print(classification_report(y_test_binary, y_pred_binary_pca, 
                                target_names=['Non-Harmonic', 'Harmonic']))
    
    # Plot confusion matrices (compare original vs PCA)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Multi-class - Original features
    cm_multi_orig = confusion_matrix(y_test, y_pred_multi_orig)
    sns.heatmap(cm_multi_orig, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                xticklabels=['harmonic', 'dead_note', 'general_note'],
                yticklabels=['harmonic', 'dead_note', 'general_note'])
    axes[0, 0].set_title(f'Multi-class (Original)\nAcc: {acc_multi_orig:.3f}')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')
    
    # Multi-class - PCA features
    cm_multi_pca = confusion_matrix(y_test, y_pred_multi_pca)
    sns.heatmap(cm_multi_pca, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['harmonic', 'dead_note', 'general_note'],
                yticklabels=['harmonic', 'dead_note', 'general_note'])
    axes[0, 1].set_title(f'Multi-class (PCA)\nAcc: {acc_multi_pca:.3f}')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')
    
    # Binary - Original features
    cm_binary_orig = confusion_matrix(y_test_binary, y_pred_binary_orig)
    sns.heatmap(cm_binary_orig, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0],
                xticklabels=['Non-Harmonic', 'Harmonic'],
                yticklabels=['Non-Harmonic', 'Harmonic'])
    axes[1, 0].set_title(f'Binary (Original)\nAcc: {acc_binary_orig:.3f}, AUC: {roc_auc_orig:.3f}')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')
    
    # Binary - PCA features
    cm_binary_pca = confusion_matrix(y_test_binary, y_pred_binary_pca)
    sns.heatmap(cm_binary_pca, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1],
                xticklabels=['Non-Harmonic', 'Harmonic'],
                yticklabels=['Non-Harmonic', 'Harmonic'])
    axes[1, 1].set_title(f'Binary (PCA)\nAcc: {acc_binary_pca:.3f}, AUC: {roc_auc_pca:.3f}')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'logistic_regression_confusion.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {output_dir / 'logistic_regression_confusion.png'}")
    plt.close()
    
    # ROC curves comparison (Original vs PCA)
    fpr_orig, tpr_orig, _ = roc_curve(y_test_binary, y_pred_proba_binary_orig)
    fpr_pca, tpr_pca, _ = roc_curve(y_test_binary, y_pred_proba_binary_pca)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_orig, tpr_orig, linewidth=2, label=f'Original Features (AUC = {roc_auc_orig:.3f})')
    plt.plot(fpr_pca, tpr_pca, linewidth=2, label=f'PCA Features (AUC = {roc_auc_pca:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Harmonic vs Non-Harmonic')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve_harmonic.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'roc_curve_harmonic.png'}")
    plt.close()
    
    # Feature importance (top 15 features from original model)
    feature_importance = pd.DataFrame({
        'feature': features_df.columns,
        'importance': np.abs(lr_binary_orig.coef_[0])
    }).sort_values('importance', ascending=False).head(15)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance['importance'])
    plt.yticks(range(len(feature_importance)), feature_importance['feature'].tolist())
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Top 15 Features for Harmonic Detection (Original Features)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_dir / 'feature_importance.png'}")
    plt.close()
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE SUMMARY")
    print("="*60)
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Performance comparison table
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    comparison_data = {
        'Model': ['Multi-class (Orig)', 'Multi-class (Optimal PCA)', 
                  'Binary (Orig)', 'Binary (Optimal PCA)'],
        'Accuracy': [acc_multi_orig, acc_multi_pca, acc_binary_orig, acc_binary_pca],
        'Features': [X_train_scaled.shape[1], X_train_pca_optimal.shape[1], 
                     X_train_scaled.shape[1], X_train_pca_optimal.shape[1]],
        'ROC AUC': ['-', '-', roc_auc_orig, roc_auc_pca]
    }
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))
    
    # Additional comparison with 95% variance PCA
    print("\n" + "="*60)
    print("COMPARISON: OPTIMAL vs 95% VARIANCE PCA")
    print("="*60)
    
    # Test 95% PCA on binary task
    lr_95 = LogisticRegression(max_iter=1000, random_state=42)
    lr_95.fit(X_train_pca_95, y_train_binary)
    acc_95 = lr_95.score(X_test_pca_95, y_test_binary)
    roc_auc_95 = roc_auc_score(y_test_binary, lr_95.predict_proba(X_test_pca_95)[:, 1])
    
    print(f"\nOptimal PCA ({optimal_n_components} components):")
    print(f"  Accuracy: {acc_binary_pca:.4f}, AUC: {roc_auc_pca:.4f}")
    print(f"\n95% Variance PCA ({X_train_pca_95.shape[1]} components):")
    print(f"  Accuracy: {acc_95:.4f}, AUC: {roc_auc_95:.4f}")
    print(f"\nDifference (Optimal - 95%):")
    print(f"  Accuracy: {acc_binary_pca - acc_95:+.4f}")
    print(f"  AUC: {roc_auc_pca - roc_auc_95:+.4f}")
    print(f"  Components saved: {X_train_pca_95.shape[1] - optimal_n_components}")
    
    return {
        'multi_class_model_orig': lr_multi_orig,
        'multi_class_model_pca': lr_multi_pca,
        'binary_model_orig': lr_binary_orig,
        'binary_model_pca': lr_binary_pca,
        'scaler': scaler,
        'pca': pca_optimal,
        'pca_95': pca_95,
        'features': features_df.columns.tolist(),
        'n_components': optimal_n_components,
        'n_components_95': X_train_pca_95.shape[1],
        'test_accuracy_multi_orig': acc_multi_orig,
        'test_accuracy_multi_pca': acc_multi_pca,
        'test_accuracy_binary_orig': acc_binary_orig,
        'test_accuracy_binary_pca': acc_binary_pca,
        'test_accuracy_binary_95': acc_95,
        'roc_auc_orig': roc_auc_orig,
        'roc_auc_pca': roc_auc_pca,
        'roc_auc_95': roc_auc_95
    }


def generate_summary_report(df: pd.DataFrame, output_dir: Path, baseline_results=None):
    """Generate a markdown summary report."""
    report_path = output_dir / 'summary_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Dataset Analysis Summary Report\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Dataset Overview\n\n")
        f.write(f"- Total samples: {len(df)}\n")
        f.write(f"- Unique audio files: {df['source_audio'].nunique()}\n")
        f.write(f"- Dataset subsets: {df['subset_id'].nunique()}\n\n")
        
        f.write("## Class Distribution\n\n")
        class_counts = df['label_category'].value_counts()
        class_pct = df['label_category'].value_counts(normalize=True) * 100
        for label in class_counts.index:
            f.write(f"- **{label}**: {class_counts[label]} ({class_pct[label]:.1f}%)\n")
        
        f.write("\n## Duration Statistics\n\n")
        duration_stats = df.groupby('label_category')['duration_sec'].describe()
        f.write(duration_stats.to_markdown())
        
        if baseline_results:
            f.write("\n\n## Baseline Model Performance\n\n")
            
            f.write("### Multi-class Classification\n")
            f.write(f"- Original Features ({len(baseline_results['features'])} features):\n")
            f.write(f"  - Test Accuracy: {baseline_results['test_accuracy_multi_orig']:.3f}\n")
            f.write(f"- PCA Features ({baseline_results['n_components']} components, 95% variance):\n")
            f.write(f"  - Test Accuracy: {baseline_results['test_accuracy_multi_pca']:.3f}\n")
            f.write(f"  - Change: {baseline_results['test_accuracy_multi_pca'] - baseline_results['test_accuracy_multi_orig']:+.3f}\n\n")
            
            f.write("### Binary Classification (Harmonic vs Non-Harmonic)\n")
            f.write(f"- Original Features:\n")
            f.write(f"  - Test Accuracy: {baseline_results['test_accuracy_binary_orig']:.3f}\n")
            f.write(f"  - ROC AUC: {baseline_results['roc_auc_orig']:.3f}\n")
            f.write(f"- PCA Features:\n")
            f.write(f"  - Test Accuracy: {baseline_results['test_accuracy_binary_pca']:.3f}\n")
            f.write(f"  - ROC AUC: {baseline_results['roc_auc_pca']:.3f}\n")
            f.write(f"  - Change: {baseline_results['test_accuracy_binary_pca'] - baseline_results['test_accuracy_binary_orig']:+.3f} (accuracy), ")
            f.write(f"{baseline_results['roc_auc_pca'] - baseline_results['roc_auc_orig']:+.3f} (AUC)\n\n")
            
            f.write("**PCA Analysis:** ")
            if baseline_results['test_accuracy_binary_pca'] >= baseline_results['test_accuracy_binary_orig']:
                f.write(f"PCA maintained/improved performance while reducing feature space from ")
                f.write(f"{len(baseline_results['features'])} to {baseline_results['n_components']} dimensions.\n\n")
            else:
                f.write(f"PCA reduced dimensionality significantly but with minor performance tradeoff.\n\n")
            
            f.write("These baseline results show what a simple logistic regression model can achieve ")
            f.write("using hand-crafted audio features with z-score normalization and PCA. ")
            f.write("A CNN should significantly outperform these results.\n")
        
        f.write("\n## Recommendations\n\n")
        
        # Class imbalance
        min_class = class_pct.min()
        if min_class < 10:
            f.write("- ⚠️ **Severe class imbalance detected.** Consider using weighted loss or oversampling.\n")
        
        # Duration analysis
        max_dur = df['duration_sec'].max()
        if max_dur > 5:
            f.write(f"- ⚠️ **Long durations detected** (max: {max_dur:.2f}s). Consider using variable-length inputs or windowing.\n")
        
        if baseline_results:
            best_auc = max(baseline_results['roc_auc_orig'], baseline_results['roc_auc_pca'])
            if best_auc > 0.8:
                f.write("- ✅ **Strong baseline performance.** Features show good discriminative power.\n")
            elif best_auc > 0.7:
                f.write("- ⚠️ **Moderate baseline performance.** Consider feature engineering or deep learning.\n")
            else:
                f.write("- ⚠️ **Weak baseline performance.** Task may require deep learning approaches.\n")
            
            # PCA recommendation
            pca_improvement = baseline_results['roc_auc_pca'] - baseline_results['roc_auc_orig']
            if pca_improvement > 0.01:
                f.write("- ✅ **PCA beneficial.** Use PCA for dimensionality reduction in production models.\n")
            elif abs(pca_improvement) < 0.01:
                f.write("- ℹ️ **PCA neutral.** Consider using PCA for computational efficiency with minimal performance impact.\n")
            else:
                f.write("- ⚠️ **PCA reduces performance.** Keep original features for maximum accuracy.\n")
        
        f.write("\n---\n\n")
        f.write("See accompanying plots for detailed visualizations.\n")
    
    print(f"\nSummary report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze guitar harmonics dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--metadata', default='processed_dataset/metadata.csv',
                       help='Path to metadata CSV')
    parser.add_argument('--output', default='analysis/',
                       help='Output directory for plots and reports')
    parser.add_argument('--n-samples', type=int, default=50,
                       help='Number of samples for audio analysis')
    
    args = parser.parse_args()
    
    # Resolve paths relative to script location
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # If metadata path is relative, make it relative to project root
    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = project_root / metadata_path
    
    # If output path is relative, make it relative to project root
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GUITAR HARMONICS DATASET ANALYSIS")
    print("="*60)
    
    # Run all analyses
    print("\nLoading metadata...")
    df = load_metadata(str(metadata_path))
    
    print("\n1. Class distribution analysis...")
    analyze_class_distribution(df, output_dir)
    
    print("\n2. Duration analysis...")
    analyze_durations(df, output_dir)
    
    print("\n3. Pitch distribution...")
    analyze_pitch_distribution(df, output_dir)
    
    print("\n4. Subset distribution...")
    analyze_subset_distribution(df)
    
    print("\n5. Sample spectrograms...")
    plot_sample_spectrograms(df, output_dir)
    
    print("\n6. Spectral features analysis...")
    analyze_spectral_features(df, output_dir, args.n_samples)
    
    print("\n7. Data quality checks...")
    check_data_quality(df)
    
    print("\n8. Split preview...")
    preview_splits(df, output_dir)
    
    print("\n9. Training baseline logistic regression models...")
    baseline_results = train_logistic_regression_baseline(df, output_dir, args.n_samples)
    
    print("\n10. Generating summary report...")
    generate_summary_report(df, output_dir, baseline_results)
    
    print("\n" + "="*60)
    print(f"✓ Analysis complete! Results saved to {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
