"""
Inference script for guitar harmonics classification.

Usage:
    python inference.py --audio path/to/audio.wav --model models/best_model.pt
    python inference.py --audio path/to/audio.wav --model models/best_model.pt --duration 3.0
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import librosa
import numpy as np


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


def load_model(model_path, device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    model = HarmonicsCNN(num_classes=3, dropout=0.5)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def preprocess_audio(audio_path, sr=22050, duration=3.0, n_mels=128, n_fft=2048, hop_length=512, offset=0.0):
    """Load and preprocess audio file to mel spectrogram."""
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr, offset=offset, duration=duration)
    
    # Pad or trim to fixed length
    target_length = int(sr * duration)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    
    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=80,
        fmax=8000
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    # Convert to tensor (add batch and channel dimensions)
    mel_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0)
    
    return mel_tensor


def predict(model, audio_tensor, device):
    """Run inference on audio tensor."""
    audio_tensor = audio_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(audio_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    return predicted_class, confidence, probabilities[0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Inference for guitar harmonics classification')
    parser.add_argument('--audio', required=True, help='Path to audio file')
    parser.add_argument('--model', default='models/best_model.pt', help='Path to trained model')
    parser.add_argument('--duration', type=float, default=3.0, help='Duration to analyze (seconds)')
    parser.add_argument('--offset', type=float, default=0.0, help='Start time offset (seconds)')
    parser.add_argument('--show-probs', action='store_true', help='Show probabilities for all classes')
    
    args = parser.parse_args()
    
    # Check if files exist
    audio_path = Path(args.audio)
    model_path = Path(args.model)
    
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model, checkpoint = load_model(model_path, device)
    print(f"Model loaded from epoch {checkpoint['epoch'] + 1}")
    if 'val_acc' in checkpoint:
        print(f"Validation accuracy: {checkpoint['val_acc']:.2f}%")
    
    # Class names
    class_names = ['harmonic', 'dead_note', 'general_note']
    
    # Preprocess audio
    print(f"\nProcessing audio: {audio_path}")
    print(f"  Duration: {args.duration}s")
    print(f"  Offset: {args.offset}s")
    audio_tensor = preprocess_audio(audio_path, duration=args.duration, offset=args.offset)
    
    # Predict
    print("\nRunning inference...")
    predicted_class, confidence, probabilities = predict(model, audio_tensor, device)
    
    # Display results
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    print(f"Predicted class: {class_names[predicted_class].upper()}")
    print(f"Confidence: {confidence*100:.2f}%")
    
    if args.show_probs:
        print("\nClass probabilities:")
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            marker = "→" if i == predicted_class else " "
            print(f"  {marker} {class_name:12s}: {prob*100:5.2f}%")
    
    print("="*60)


if __name__ == "__main__":
    main()
