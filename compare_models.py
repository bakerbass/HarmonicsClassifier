"""
Quick comparison script for SVM vs CNN models.

Usage:
    python compare_models.py --metadata processed_dataset/metadata.csv
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
from feature_extraction import AudioFeatureExtractor
from inference_svm import HarmonicsSVMClassifier

# CNN class definition (matching train_cnn.py)
class HarmonicsCNN(torch.nn.Module):
    """CNN for guitar harmonics classification."""
    
    def __init__(self, num_classes=3, dropout=0.5):
        super(HarmonicsCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout2d(0.25)
        )
        
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout2d(0.25)
        )
        
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2),
            torch.nn.Dropout2d(0.25)
        )
        
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        )
        
        # Fully connected layers
        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x


def load_svm_model(model_path: Path):
    """Load SVM model and scaler."""
    with open(model_path / 'svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(model_path / 'svm_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def load_cnn_model(model_path: Path, device='cpu'):
    """Load CNN model."""
    model = HarmonicsCNN(num_classes=3)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model


def extract_mel_spectrogram(audio, sr=22050, normalize=False, target_rms=0.1):
    """Extract mel spectrogram for CNN."""
    # Normalize amplitude to match training data
    if normalize and len(audio) > 0:
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 1e-6:
            audio = audio * (target_rms / current_rms)
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=2048, hop_length=512, n_mels=128,
        fmin=80, fmax=8000
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
    
    # Resize to 128x128
    if mel_spec_db.shape[1] < 128:
        mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, 128 - mel_spec_db.shape[1])))
    else:
        mel_spec_db = mel_spec_db[:, :128]
    
    return mel_spec_db


def evaluate_models(metadata_path: str, svm_path: Path, cnn_path: Path, max_samples=None):
    """Evaluate both models on test data."""
    
    print("="*60)
    print("LOADING MODELS")
    print("="*60)
    
    # Load models
    print("\nLoading SVM model...")
    svm_model, scaler = load_svm_model(svm_path)
    # Create feature extractor WITHOUT normalization (to match training)
    feature_extractor = AudioFeatureExtractor(normalize_audio=False)
    
    print("Loading CNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model = load_cnn_model(cnn_path, device)
    
    print(f"Using device: {device}")
    
    # Load metadata
    print("\nLoading metadata...")
    df = pd.read_csv(metadata_path)
    
    # Resolve relative paths to absolute (relative to metadata file location)
    metadata_dir = Path(metadata_path).parent.parent  # Go up to project root
    df['source_audio'] = df['source_audio'].apply(lambda x: str(metadata_dir / x))
    
    # Filter to only existing files (important if testing on new computer)
    print("\nFiltering to existing files...")
    df['file_exists'] = df['source_audio'].apply(lambda x: Path(x).exists())
    existing_count = df['file_exists'].sum()
    print(f"  Files in metadata: {len(df)}")
    print(f"  Files that exist: {existing_count}")
    df = df[df['file_exists']].drop(columns=['file_exists'])
    
    if len(df) == 0:
        print("❌ ERROR: No audio files found!")
        return
    
    # Filter to test split or sample
    if 'split' in df.columns:
        test_df = df[df['split'] == 'test']
        print(f"Found {len(test_df)} test samples")
    else:
        # Prioritize GB_NH files (newly added harmonics) if available
        gb_nh_files = df[df['source_audio'].str.contains('GB_NH_harmonic')]
        if len(gb_nh_files) > 50:
            print(f"Found {len(gb_nh_files)} GB_NH harmonic files - using these for testing")
            test_df = gb_nh_files
        else:
            # Use last 20% as test
            test_size = int(len(df) * 0.2)
            test_df = df.iloc[-test_size:]
            print(f"Using last {len(test_df)} samples as test set")
    
    if max_samples and len(test_df) > max_samples:
        test_df = test_df.sample(max_samples, random_state=42)
        print(f"Sampled {max_samples} for quick comparison")
    
    # Predictions storage
    y_true = []
    y_pred_svm = []
    y_pred_cnn = []
    
    label_map = {'harmonic': 0, 'dead_note': 1, 'general_note': 2}
    label_names = ['harmonic', 'dead_note', 'general_note']
    
    print("\n" + "="*60)
    print("RUNNING INFERENCE")
    print("="*60)
    
    errors = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Processing samples"):
        try:
            # Load audio
            audio, sr = librosa.load(
                row['source_audio'],
                sr=22050,
                offset=row['onset_sec'],
                duration=min(row['duration_sec'], 3.0)
            )
            
            if len(audio) < 1000:
                continue
            
            # SVM prediction
            features = feature_extractor.extract_all_features(audio)
            features_scaled = scaler.transform([list(features.values())])
            svm_pred = svm_model.predict(features_scaled)[0]
            
            # CNN prediction
            mel_spec = extract_mel_spectrogram(audio, sr)
            mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = cnn_model(mel_tensor)
                cnn_pred = torch.argmax(logits, dim=1).item()
            
            # Only append if both predictions succeeded
            y_true.append(label_map[row['label_category']])
            y_pred_svm.append(svm_pred)
            y_pred_cnn.append(cnn_pred)
            
        except Exception as e:
            # Track errors for debugging
            if len(errors) < 3:
                errors.append((row['source_audio'], str(e)))
            continue
    
    if errors:
        print("\nSample errors encountered:")
        for path, error in errors:
            print(f"  {Path(path).name}: {error}")
    
    y_true = np.array(y_true)
    y_pred_svm = np.array(y_pred_svm)
    y_pred_cnn = np.array(y_pred_cnn)
    
    print(f"\nSuccessfully processed {len(y_true)} samples")
    
    if len(y_true) == 0:
        print("\n❌ ERROR: No samples were successfully processed!")
        print("This likely means the audio files couldn't be loaded.")
        print("Check that the paths in metadata are correct.")
        return
    
    # Calculate metrics
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    svm_acc = accuracy_score(y_true, y_pred_svm)
    cnn_acc = accuracy_score(y_true, y_pred_cnn)
    svm_f1 = f1_score(y_true, y_pred_svm, average='macro')
    cnn_f1 = f1_score(y_true, y_pred_cnn, average='macro')
    
    print(f"\n{'Metric':<20} {'SVM':<12} {'CNN':<12} {'Difference':<12}")
    print("-" * 60)
    print(f"{'Accuracy':<20} {svm_acc:<12.4f} {cnn_acc:<12.4f} {cnn_acc - svm_acc:+.4f}")
    print(f"{'F1 Score (Macro)':<20} {svm_f1:<12.4f} {cnn_f1:<12.4f} {cnn_f1 - svm_f1:+.4f}")
    
    # Per-class F1 scores (only for classes present in test data)
    unique_classes = np.unique(y_true)
    present_labels = [label_names[i] for i in unique_classes]
    
    svm_f1_per_class = f1_score(y_true, y_pred_svm, average=None, labels=unique_classes)
    cnn_f1_per_class = f1_score(y_true, y_pred_cnn, average=None, labels=unique_classes)
    
    print("\nPer-Class F1 Scores:")
    print(f"{'Class':<20} {'SVM':<12} {'CNN':<12} {'Difference':<12}")
    print("-" * 60)
    for i, label in enumerate(present_labels):
        print(f"{label:<20} {svm_f1_per_class[i]:<12.4f} {cnn_f1_per_class[i]:<12.4f} {cnn_f1_per_class[i] - svm_f1_per_class[i]:+.4f}")
    
    # Classification reports
    print("\n" + "="*60)
    print("SVM CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred_svm, target_names=present_labels, labels=unique_classes))
    
    print("\n" + "="*60)
    print("CNN CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_true, y_pred_cnn, target_names=present_labels, labels=unique_classes))
    
    # Confusion matrices
    cm_svm = confusion_matrix(y_true, y_pred_svm, labels=unique_classes)
    cm_cnn = confusion_matrix(y_true, y_pred_cnn, labels=unique_classes)
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SVM confusion matrix
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=present_labels, yticklabels=present_labels)
    axes[0].set_title(f'SVM Confusion Matrix\nAccuracy: {svm_acc:.3f}, F1: {svm_f1:.3f}')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # CNN confusion matrix
    sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=present_labels, yticklabels=present_labels)
    axes[1].set_title(f'CNN Confusion Matrix\nAccuracy: {cnn_acc:.3f}, F1: {cnn_f1:.3f}')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    output_path = Path('comparison_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot: {output_path}")
    plt.show()
    
    # Agreement analysis
    agreement = (y_pred_svm == y_pred_cnn).sum()
    both_correct = ((y_pred_svm == y_true) & (y_pred_cnn == y_true)).sum()
    svm_only_correct = ((y_pred_svm == y_true) & (y_pred_cnn != y_true)).sum()
    cnn_only_correct = ((y_pred_svm != y_true) & (y_pred_cnn == y_true)).sum()
    both_wrong = ((y_pred_svm != y_true) & (y_pred_cnn != y_true)).sum()
    
    print("\n" + "="*60)
    print("AGREEMENT ANALYSIS")
    print("="*60)
    print(f"Total samples:        {len(y_true)}")
    print(f"Models agree:         {agreement} ({100*agreement/len(y_true):.1f}%)")
    print(f"Both correct:         {both_correct} ({100*both_correct/len(y_true):.1f}%)")
    print(f"Only SVM correct:     {svm_only_correct} ({100*svm_only_correct/len(y_true):.1f}%)")
    print(f"Only CNN correct:     {cnn_only_correct} ({100*cnn_only_correct/len(y_true):.1f}%)")
    print(f"Both incorrect:       {both_wrong} ({100*both_wrong/len(y_true):.1f}%)")
    
    # Show some disagreement examples
    disagreements = np.where(y_pred_svm != y_pred_cnn)[0]
    if len(disagreements) > 0:
        print(f"\nExample disagreements (showing up to 5):")
        for idx in disagreements[:5]:
            print(f"  Sample {idx}: True={label_names[y_true[idx]]}, "
                  f"SVM={label_names[y_pred_svm[idx]]}, "
                  f"CNN={label_names[y_pred_cnn[idx]]}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    winner = "CNN" if cnn_acc > svm_acc else "SVM" if svm_acc > cnn_acc else "TIE"
    diff_pct = abs(cnn_acc - svm_acc) * 100
    print(f"Winner: {winner} (+{diff_pct:.2f}% accuracy)")
    
    if cnn_acc > svm_acc + 0.05:
        print("✓ CNN shows significant advantage - use for production")
    elif svm_acc > cnn_acc + 0.05:
        print("✓ SVM shows significant advantage - use for production")
    else:
        print("✓ Models perform similarly - choose based on inference speed/size needs")
        print("  (SVM: ~1MB, <10ms | CNN: ~500KB, ~20ms)")


def main():
    parser = argparse.ArgumentParser(description='Compare SVM and CNN models')
    parser.add_argument('--metadata', default='processed_dataset/metadata.csv',
                       help='Path to metadata CSV')
    parser.add_argument('--svm-model', default='models/svm',
                       help='Path to SVM model directory')
    parser.add_argument('--cnn-model', default='models/best_model.pt',
                       help='Path to CNN model checkpoint')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to test (for quick comparison)')
    
    args = parser.parse_args()
    
    svm_path = Path(args.svm_model)
    cnn_path = Path(args.cnn_model)
    
    # Check if models exist
    if not (svm_path / 'svm_model.pkl').exists():
        print(f"Error: SVM model not found at {svm_path}")
        print("Train SVM first: python train_svm.py")
        return
    
    if not cnn_path.exists():
        print(f"Error: CNN model not found at {cnn_path}")
        print("Train CNN first: python train_cnn.py")
        return
    
    evaluate_models(args.metadata, svm_path, cnn_path, args.max_samples)


if __name__ == '__main__':
    main()
