"""
Real-time guitar harmonics classifier with OSC control.

This script sets up an OSC server to control a guitar simulator plugin,
captures audio in real-time, and classifies notes as harmonic/dead/general.

Usage:
    python osc_realtime_classifier.py --model models/best_model.pt
    python osc_realtime_classifier.py --model models/best_model.pt --osc-host localhost --osc-port 9000

Requirements:
    pip install python-osc sounddevice
"""

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import librosa
import sounddevice as sd
from pythonosc import udp_client
import queue
import threading
import time
from collections import deque
import matplotlib.pyplot as plt


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


class RealtimeClassifier:
    """Real-time audio classifier with OSC control."""
    
    def __init__(self, model_path, device, device_id, osc_host='127.0.0.1', osc_port=9000, 
                 duration=1, plot_enabled=False):
        self.device = device
        self.device_id = device_id
        self.model_sr = 22050  # Model's expected sample rate
        self.duration = duration
        self.plot_enabled = plot_enabled
        self.n_mels = 128
        self.n_fft = 2048
        self.hop_length = 512
        
        # Get device's default sample rate
        device_info = sd.query_devices(device_id)
        self.record_sr = int(device_info['default_samplerate'])
        print(f"Audio device sample rate: {self.record_sr} Hz")
        print(f"Model sample rate: {self.model_sr} Hz")
        if self.record_sr != self.model_sr:
            print(f"Will resample audio from {self.record_sr} Hz to {self.model_sr} Hz")
        
        # Load model
        print(f"\nLoading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        self.model = HarmonicsCNN(num_classes=3, dropout=0.5).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print(f"✓ Model loaded from epoch {checkpoint['epoch'] + 1}")
        
        # OSC client
        self.osc_client = udp_client.SimpleUDPClient(osc_host, osc_port)
        print(f"✓ OSC client connected to {osc_host}:{osc_port}")
        
        # Class names
        self.class_names = ['harmonic', 'dead_note', 'general_note']
    
    def record_audio(self, duration=None):
        """Record audio for specified duration."""
        if duration is None:
            duration = self.duration
        
        print(f"Recording for {duration} seconds...")
        recording = sd.rec(
            int(duration * self.record_sr),
            samplerate=self.record_sr,
            channels=1,
            device=self.device_id,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        print("✓ Recording complete")
        
        # Return as 1D array
        audio = recording[:, 0]
        
        # Resample to model's sample rate if needed
        if self.record_sr != self.model_sr:
            audio = librosa.resample(audio, orig_sr=self.record_sr, target_sr=self.model_sr)
        
        return audio
    
    def preprocess_audio(self, audio):
        """Preprocess audio to mel spectrogram."""
        # Pad or trim to exact duration
        target_length = int(self.model_sr * self.duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.model_sr,
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
        
        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel_spec_norm).unsqueeze(0).unsqueeze(0)
        return mel_tensor
    
    def predict(self, audio_tensor):
        """Run inference on audio tensor."""
        audio_tensor = audio_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(audio_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        return predicted_class, confidence, probabilities[0].cpu().numpy()
    
    def plot_spectrogram(self, audio):
        """Plot waveform and mel spectrogram."""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Waveform
        time_axis = np.arange(len(audio)) / self.model_sr
        axes[0].plot(time_axis, audio)
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title('Waveform')
        axes[0].grid(True, alpha=0.3)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.model_sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=80,
            fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        img = librosa.display.specshow(
            mel_spec_db,
            sr=self.model_sr,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel',
            fmin=80,
            fmax=8000,
            ax=axes[1]
        )
        axes[1].set_title('Mel Spectrogram')
        fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        plt.tight_layout()
        plt.show()
    
    def classify_audio(self, audio):
        """Classify recorded audio."""
        # Trim silence from beginning and end
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=30)
        
        # If trimmed audio is too short, use original
        if len(audio_trimmed) < self.model_sr * 0.1:  # At least 0.1 seconds
            print("Warning: Audio too short after trimming, using original")
            audio_trimmed = audio
        else:
            print(f"Trimmed silence: {len(audio)/self.model_sr:.2f}s → {len(audio_trimmed)/self.model_sr:.2f}s")
        
        # Plot if enabled
        if self.plot_enabled:
            self.plot_spectrogram(audio_trimmed)
        
        print("Classifying...")
        audio_tensor = self.preprocess_audio(audio_trimmed)
        predicted_class, confidence, probabilities = self.predict(audio_tensor)
        
        return {
            'class': self.class_names[predicted_class],
            'class_id': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'harmonic': float(probabilities[0]),
                'dead_note': float(probabilities[1]),
                'general_note': float(probabilities[2])
            }
        }
    
    def play_and_classify(self, fret, velocity=1.0):
        """Send OSC, record audio, and classify."""
        # Start recording first (non-blocking)
        print(f"Recording for {self.duration} seconds...")
        recording = sd.rec(
            int(self.duration * self.record_sr),
            samplerate=self.record_sr,
            channels=1,
            device=self.device_id,
            dtype='float32'
        )
        
        # Send OSC message immediately after starting recording
        self.send_fret(fret, velocity)
        
        # Wait for recording to complete
        sd.wait()
        print("✓ Recording complete")
        
        # Get audio as 1D array and resample if needed
        audio = recording[:, 0]
        if self.record_sr != self.model_sr:
            audio = librosa.resample(audio, orig_sr=self.record_sr, target_sr=self.model_sr)
        
        # Classify
        result = self.classify_audio(audio)
        
        # Display results
        print("\n" + "="*60)
        print(f"Prediction: {result['class'].upper()}")
        print(f"Confidence: {result['confidence']*100:.1f}%")
        print("\nProbabilities:")
        for cls, prob in result['probabilities'].items():
            marker = "→" if cls == result['class'] else " "
            print(f"  {marker} {cls:12s}: {prob*100:5.1f}%")
        print("="*60)
        
        return result
    
    def send_fret(self, fret, velocity=1.0):
        """Send fret OSC message."""
        self.osc_client.send_message("/fret", [float(fret), float(velocity)])
        print(f"→ OSC: /fret {fret} {velocity}")
    
    def send_string(self, string_name):
        """Send string OSC message."""
        valid_strings = ['lowe', 'a', 'd', 'g', 'b', 'highe']
        if string_name.lower() not in valid_strings:
            print(f"Invalid string name: {string_name}")
            return
        self.osc_client.send_message("/string", string_name.lower())
        print(f"→ OSC: /string {string_name}")


def select_audio_device():
    """Let user select audio input device."""
    print("\n" + "="*70)
    print("AVAILABLE AUDIO INPUT DEVICES")
    print("="*70)
    
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device))
            print(f"{len(input_devices)-1}: {device['name']}")
            print(f"   Channels: {device['max_input_channels']} | Sample Rate: {device['default_samplerate']} Hz")
    
    print("="*70)
    
    while True:
        try:
            choice = int(input("\nSelect device number: "))
            if 0 <= choice < len(input_devices):
                device_id, device_info = input_devices[choice]
                print(f"\n✓ Selected: {device_info['name']}")
                return device_id
            else:
                print(f"Please enter a number between 0 and {len(input_devices)-1}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled")
            exit(0)


def interactive_mode(classifier):
    """Interactive mode for testing."""
    print("\n" + "="*70)
    print("INTERACTIVE MODE")
    print("="*70)
    print("Commands:")
    print("  fret <0-10> [velocity]  - Play fret, record 3s, and classify")
    print("  string <name>           - Change string (lowe/a/d/g/b/highe)")
    print("  quit                    - Exit")
    print("="*70)
    
    try:
        while True:
            try:
                cmd = input("\n> ").strip()
                
                if not cmd:
                    continue
                
                parts = cmd.split()
                command = parts[0].lower()
                
                if command == 'quit' or command == 'exit':
                    break
                
                elif command == 'fret':
                    if len(parts) < 2:
                        print("Usage: fret <0-10> [velocity]")
                        continue
                    fret = float(parts[1])
                    velocity = float(parts[2]) if len(parts) > 2 else 1.0
                    classifier.play_and_classify(fret, velocity)
                
                elif command == 'string':
                    if len(parts) < 2:
                        print("Usage: string <lowe/a/d/g/b/highe>")
                        continue
                    classifier.send_string(parts[1])
                
                else:
                    print(f"Unknown command: {command}")
            
            except ValueError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")


def main():
    parser = argparse.ArgumentParser(description='Real-time guitar harmonics classifier with OSC control')
    parser.add_argument('--model', default='models/best_model.pt', help='Path to trained model')
    parser.add_argument('--osc-host', default='127.0.0.1', help='OSC server host')
    parser.add_argument('--osc-port', type=int, default=9000, help='OSC server port')
    parser.add_argument('--duration', type=float, default=3.0, help='Audio duration for classification (seconds)')
    parser.add_argument('--plot', action='store_true', help='Plot waveform and spectrogram before classification')
    
    args = parser.parse_args()
    
    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Select audio device
    device_id = select_audio_device()
    
    # Create classifier
    print("\nInitializing classifier...")
    classifier = RealtimeClassifier(
        model_path, device, device_id,
        osc_host=args.osc_host,
        osc_port=args.osc_port,
        duration=args.duration,
        plot_enabled=args.plot
    )
    
    # Send test message to verify OSC connection
    print("\nSending test OSC message...")
    classifier.osc_client.send_message("/test", [1, 2, 3])
    print("→ OSC: /test 1 2 3")
    
    try:
        # Interactive mode
        interactive_mode(classifier)
    finally:
        print("\nShutdown complete")


if __name__ == "__main__":
    main()
