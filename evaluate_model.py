"""
Evaluate a trained model and generate misclassification reports.

This script loads a trained model, evaluates it on the test set,
and generates detailed misclassification reports without retraining.

Usage:
    python evaluate_model.py --model models/best_model.pt --metadata processed_dataset/metadata.csv
    python evaluate_model.py --model models/best_model.pt --metadata processed_dataset/metadata.csv --output results/
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


class HarmonicsDataset(Dataset):
    """Dataset for guitar harmonics audio clips."""
    
    def __init__(self, metadata_df, sr=22050, duration=3.0, n_mels=128, n_fft=2048, hop_length=512):
        self.metadata = metadata_df.reset_index(drop=True)
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Label mapping
        self.label_map = {'harmonic': 0, 'dead_note': 1, 'general_note': 2}
        self.labels = [self.label_map[label] for label in self.metadata['label_category']]
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        try:
            # Load audio
            audio, _ = librosa.load(
                row['source_audio'],
                sr=self.sr,
                offset=row['onset_sec'],
                duration=min(row['duration_sec'], self.duration)
            )
            
            # Pad or trim to fixed length
            target_length = int(self.sr * self.duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            # Compute mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=80,
                fmax=8000
            )
            
            # Convert to log scale (dB)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to [0, 1]
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            # Convert to tensor (add channel dimension)
            mel_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0)
            label = torch.LongTensor([self.labels[idx]])[0]
            
            return mel_tensor, label
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return zeros on error
            mel_tensor = torch.zeros((1, self.n_mels, 130))
            label = torch.LongTensor([self.labels[idx]])[0]
            return mel_tensor, label


class HarmonicsCNN(nn.Module):
    """CNN for guitar harmonics classification."""
    
    def __init__(self, num_classes=3, dropout=0.5):
        super(HarmonicsCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return running_loss / len(loader), 100. * correct / total, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['harmonic', 'dead_note', 'general_note'],
                yticklabels=['harmonic', 'dead_note', 'general_note'])
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'confusion_matrix.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model and generate misclassification reports')
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--metadata', default='processed_dataset/metadata.csv', help='Path to metadata CSV')
    parser.add_argument('--output', default=None, help='Output directory (default: same as model directory)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--test-size', type=float, default=0.15, help='Test set size (default: 0.15)')
    
    args = parser.parse_args()
    
    # Resolve paths
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    metadata_path = Path(args.metadata)
    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = model_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = HarmonicsCNN(num_classes=3, dropout=0.5).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from epoch {checkpoint['epoch'] + 1}")
    if 'val_acc' in checkpoint:
        print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Load metadata
    print("\nLoading metadata...")
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} samples")
    
    # Split dataset (same logic as training)
    print("\nSplitting dataset by audio files...")
    audio_files = df['source_audio'].unique()
    
    # Create file-to-dominant-class mapping for stratification
    file_labels = {}
    for audio_file in audio_files:
        subset = df[df['source_audio'] == audio_file]
        dominant_class = subset['label_category'].mode()[0] if len(subset) > 0 else 'unknown'
        file_labels[audio_file] = dominant_class
    
    # Split files
    try:
        train_files, test_files = train_test_split(
            audio_files, test_size=args.test_size, 
            stratify=[file_labels[f] for f in audio_files], random_state=42
        )
    except ValueError:
        print("Warning: Stratified split failed, using random split")
        train_files, test_files = train_test_split(audio_files, test_size=args.test_size, random_state=42)
    
    # Create test dataset
    test_df = df[df['source_audio'].isin(test_files)]
    print(f"Test set: {len(test_df)} samples from {len(test_files)} files")
    
    # Print class distribution
    print("\nTest set class distribution:")
    for label in ['harmonic', 'dead_note', 'general_note']:
        count = (test_df['label_category'] == label).sum()
        pct = 100 * count / len(test_df)
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Create dataset and dataloader
    print("\nCreating dataset...")
    test_dataset = HarmonicsDataset(test_df)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
    
    # Compute metrics
    test_f1_macro = f1_score(test_labels, test_preds, average='macro')
    test_f1_per_class = f1_score(test_labels, test_preds, average=None)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test F1 Score (macro): {test_f1_macro:.4f}")
    print(f"\nPer-class Test F1 Scores:")
    print(f"  Harmonic: {test_f1_per_class[0]:.4f}")
    print(f"  Dead Note: {test_f1_per_class[1]:.4f}")
    print(f"  General Note: {test_f1_per_class[2]:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['harmonic', 'dead_note', 'general_note']))
    
    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    plot_confusion_matrix(test_labels, test_preds, output_dir)
    
    # Identify and save harmonic misclassifications
    print("\nAnalyzing harmonic misclassifications...")
    
    # Convert to numpy arrays
    test_preds_np = np.array(test_preds)
    test_labels_np = np.array(test_labels)
    
    # Find false positives: predicted harmonic (0) but actually not harmonic
    false_positives_idx = np.where((test_preds_np == 0) & (test_labels_np != 0))[0]
    
    # Find false negatives: predicted not harmonic but actually harmonic (0)
    false_negatives_idx = np.where((test_preds_np != 0) & (test_labels_np == 0))[0]
    
    # Create detailed reports
    class_names = ['harmonic', 'dead_note', 'general_note']
    misclassifications = {
        'false_positives': [],
        'false_negatives': []
    }
    
    print(f"  Found {len(false_positives_idx)} false positives (predicted harmonic, actually not)")
    for idx in false_positives_idx:
        test_sample = test_df.iloc[idx]
        misclassifications['false_positives'].append({
            'index': int(idx),
            'audio_file': str(test_sample['source_audio']),
            'onset_sec': float(test_sample['onset_sec']),
            'duration_sec': float(test_sample['duration_sec']),
            'true_label': class_names[test_labels_np[idx]],
            'predicted_label': class_names[test_preds_np[idx]],
            'pitch_midi': int(test_sample['pitch_midi']) if pd.notna(test_sample['pitch_midi']) else None
        })
    
    print(f"  Found {len(false_negatives_idx)} false negatives (predicted not harmonic, actually harmonic)")
    for idx in false_negatives_idx:
        test_sample = test_df.iloc[idx]
        misclassifications['false_negatives'].append({
            'index': int(idx),
            'audio_file': str(test_sample['source_audio']),
            'onset_sec': float(test_sample['onset_sec']),
            'duration_sec': float(test_sample['duration_sec']),
            'true_label': class_names[test_labels_np[idx]],
            'predicted_label': class_names[test_preds_np[idx]],
            'pitch_midi': int(test_sample['pitch_midi']) if pd.notna(test_sample['pitch_midi']) else None
        })
    
    # Save misclassifications to JSON
    misclass_path = output_dir / 'harmonic_misclassifications.json'
    with open(misclass_path, 'w') as f:
        json.dump(misclassifications, f, indent=2)
    
    print(f"✓ Misclassifications saved to {misclass_path}")
    
    # Save to CSV for easier viewing
    if len(false_positives_idx) > 0:
        fp_df = pd.DataFrame(misclassifications['false_positives'])
        fp_csv_path = output_dir / 'false_positives_harmonics.csv'
        fp_df.to_csv(fp_csv_path, index=False)
        print(f"✓ False positives saved to {fp_csv_path}")
    
    if len(false_negatives_idx) > 0:
        fn_df = pd.DataFrame(misclassifications['false_negatives'])
        fn_csv_path = output_dir / 'false_negatives_harmonics.csv'
        fn_df.to_csv(fn_csv_path, index=False)
        print(f"✓ False negatives saved to {fn_csv_path}")
    
    # Save evaluation results
    results = {
        'test_accuracy': float(test_acc),
        'test_f1_macro': float(test_f1_macro),
        'test_f1_harmonic': float(test_f1_per_class[0]),
        'test_f1_dead_note': float(test_f1_per_class[1]),
        'test_f1_general_note': float(test_f1_per_class[2]),
        'test_loss': float(test_loss),
        'test_samples': int(len(test_df)),
        'false_positives': int(len(false_positives_idx)),
        'false_negatives': int(len(false_negatives_idx)),
        'model_path': str(model_path),
        'model_epoch': int(checkpoint['epoch'] + 1)
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Evaluation results saved to {results_path}")
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nTo review misclassifications, run:")
    print(f"  python review_misclassifications.py --results {output_dir}")


if __name__ == "__main__":
    main()
