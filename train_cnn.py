"""
Train CNN model for guitar harmonics classification.

Usage:
    python train_cnn.py --metadata processed_dataset/metadata.csv --output models/
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


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
            mel_tensor = torch.zeros((1, self.n_mels, 130))  # Approximate time frames
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
            nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
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


def compute_class_weights(labels, harmonic_multiplier=1.0):
    """Compute class weights for imbalanced dataset.
    
    Args:
        labels: Array of class labels
        harmonic_multiplier: Additional weight multiplier for harmonic class (default 1.0)
                           Set > 1.0 to prioritize harmonic detection
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique) * counts)
    
    # Apply additional multiplier to harmonic class (index 0)
    weights[0] *= harmonic_multiplier
    
    return torch.FloatTensor(weights)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'acc': 100. * correct / total})
    
    return running_loss / len(loader), 100. * correct / total, all_preds, all_labels


def validate(model, loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation'):
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


def plot_training_history(history, output_dir):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'training_history.png'}")
    plt.close()


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
    parser = argparse.ArgumentParser(description='Train CNN for harmonics classification')
    parser.add_argument('--metadata', default='processed_dataset/metadata.csv',
                       help='Path to metadata CSV')
    parser.add_argument('--output', default='models/',
                       help='Output directory for model and results')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout rate')
    parser.add_argument('--n-samples', type=int, default=None,
                       help='Limit samples per class (None for all)')
    parser.add_argument('--harmonic-weight', type=float, default=2.0,
                       help='Additional weight multiplier for harmonic class (default 2.0)')
    parser.add_argument('--use-harmonic-f1', action='store_true',
                       help='Use harmonic F1 score (instead of val accuracy) for model selection')
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = script_dir / metadata_path
    
    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load metadata
    print("\nLoading metadata...")
    df = pd.read_csv(metadata_path)
    print(f"Loaded {len(df)} samples")
    
    # Optionally limit samples per class
    if args.n_samples:
        print(f"\nLimiting to {args.n_samples} samples per class...")
        df_balanced = []
        for label in ['harmonic', 'dead_note', 'general_note']:
            subset = df[df['label_category'] == label]
            sampled = subset.sample(min(args.n_samples, len(subset)), random_state=42)
            df_balanced.append(sampled)
        df = pd.concat(df_balanced, ignore_index=True)
        print(f"Dataset size: {len(df)} samples")
    
    # Split by audio files to prevent data leakage
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
            audio_files, test_size=0.15, stratify=[file_labels[f] for f in audio_files], random_state=42
        )
        train_files, val_files = train_test_split(
            train_files, test_size=0.15/0.85, stratify=[file_labels[f] for f in train_files], random_state=42
        )
    except ValueError:
        print("Warning: Stratified split failed, using random split")
        train_files, test_files = train_test_split(audio_files, test_size=0.15, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.15/0.85, random_state=42)
    
    # Create dataset splits
    train_df = df[df['source_audio'].isin(train_files)]
    val_df = df[df['source_audio'].isin(val_files)]
    test_df = df[df['source_audio'].isin(test_files)]
    
    print(f"  Train: {len(train_df)} samples from {len(train_files)} files")
    print(f"  Val: {len(val_df)} samples from {len(val_files)} files")
    print(f"  Test: {len(test_df)} samples from {len(test_files)} files")
    
    # Print class distribution
    print("\nClass distribution:")
    for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"  {split_name}:")
        for label in ['harmonic', 'dead_note', 'general_note']:
            count = (split_df['label_category'] == label).sum()
            pct = 100 * count / len(split_df)
            print(f"    {label}: {count} ({pct:.1f}%)")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = HarmonicsDataset(train_df)
    val_dataset = HarmonicsDataset(val_df)
    test_dataset = HarmonicsDataset(test_df)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Compute class weights with harmonic emphasis
    class_weights = compute_class_weights(train_dataset.labels, harmonic_multiplier=args.harmonic_weight).to(device)
    print(f"\nClass weights (with harmonic_multiplier={args.harmonic_weight}): {class_weights.cpu().numpy()}")
    print(f"  harmonic weight: {class_weights[0]:.4f}")
    print(f"  dead_note weight: {class_weights[1]:.4f}")
    print(f"  general_note weight: {class_weights[2]:.4f}")
    
    # Create model
    print("\nCreating model...")
    model = HarmonicsCNN(num_classes=3, dropout=args.dropout).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_harmonic_f1': [],
        'val_harmonic_f1': []
    }
    
    best_val_acc = 0.0
    best_val_metric = 0.0  # For harmonic F1 or overall accuracy
    best_epoch = 0
    metric_name = 'Harmonic F1' if args.use_harmonic_f1 else 'Val Acc'
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_loss, train_acc, train_preds, train_labels = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Compute F1 scores
        val_f1_macro = f1_score(val_labels, val_preds, average='macro')
        train_f1_per_class = f1_score(train_labels, train_preds, average=None)
        val_f1_per_class = f1_score(val_labels, val_preds, average=None)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_harmonic_f1'].append(float(train_f1_per_class[0]))
        history['val_harmonic_f1'].append(float(val_f1_per_class[0]))
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Per-class F1 - Harmonic: {train_f1_per_class[0]:.4f}, Dead: {train_f1_per_class[1]:.4f}, General: {train_f1_per_class[2]:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val F1 (macro): {val_f1_macro:.4f}")
        print(f"  Per-class F1 - Harmonic: {val_f1_per_class[0]:.4f}, Dead: {val_f1_per_class[1]:.4f}, General: {val_f1_per_class[2]:.4f}")
        
        # Determine current metric for model selection
        current_metric = val_f1_per_class[0] if args.use_harmonic_f1 else val_acc
        
        # Save best model based on selected metric
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1_macro': val_f1_macro,
                'val_harmonic_f1': float(val_f1_per_class[0]),
                'val_f1_per_class': val_f1_per_class.tolist(),
            }, output_dir / 'best_model.pt')
            print(f"✓ Saved best model ({metric_name}: {current_metric:.4f})")
    
    print(f"\nBest {metric_name}: {best_val_metric:.4f} at epoch {best_epoch}")
    print(f"Corresponding validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    print("\nPlotting training history...")
    plot_training_history(history, output_dir)
    
    # Test on best model
    print("\n" + "="*60)
    print("TESTING")
    print("="*60)
    
    # Load best model
    checkpoint = torch.load(output_dir / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    test_loss, test_acc, test_preds, test_labels = validate(model, test_loader, criterion, device)
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
    
    # Save results
    results = {
        'test_accuracy': float(test_acc),
        'test_f1_macro': float(test_f1_macro),
        'test_f1_harmonic': float(test_f1_per_class[0]),
        'test_f1_dead_note': float(test_f1_per_class[1]),
        'test_f1_general_note': float(test_f1_per_class[2]),
        'test_loss': float(test_loss),
        'best_val_accuracy': float(best_val_acc),
        'best_val_metric': float(best_val_metric),
        'selection_metric': metric_name,
        'best_epoch': int(best_epoch),
        'total_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'dropout': args.dropout,
        'harmonic_weight_multiplier': args.harmonic_weight,
        'n_samples_per_class': args.n_samples,
    }
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir / 'results.json'}")
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
