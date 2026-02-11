"""
Real-time inference using trained SVM model for harmonics classification.

This script provides a simple interface for classifying audio files or
audio snippets using the trained SVM model.

Usage:
    # Classify a single audio file
    python inference_svm.py --model models/svm --audio path/to/audio.wav
    
    # Classify multiple files
    python inference_svm.py --model models/svm --audio file1.wav file2.wav file3.wav
    
    # Classify with probability scores
    python inference_svm.py --model models/svm --audio audio.wav --probabilities
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from feature_extraction import AudioFeatureExtractor


class HarmonicsSVMClassifier:
    """Real-time classifier using trained SVM model."""
    
    def __init__(self, model_dir: str):
        """
        Initialize classifier by loading model files.
        
        Args:
            model_dir: Directory containing SVM model, scaler, and config
        """
        model_dir = Path(model_dir)
        
        # Load model
        with open(model_dir / "svm_model.pkl", 'rb') as f:
            self.model = pickle.load(f)
        
        # Load scaler
        with open(model_dir / "svm_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)
        
        # Load feature config
        with open(model_dir / "feature_config.json", 'r') as f:
            config = json.load(f)
        
        # Create feature extractor
        self.extractor = AudioFeatureExtractor(
            sr=config['sr'],
            duration=config['duration'],
            n_mfcc=config['n_mfcc'],
            n_fft=config['n_fft'],
            hop_length=config['hop_length'],
            n_chroma=config['n_chroma'],
        )
        
        self.feature_names = config['feature_names']
        self.label_names = ['harmonic', 'dead_note', 'general_note']
        
        print(f"✓ Loaded SVM model from {model_dir}")
        print(f"  Kernel: {self.model.kernel}")
        print(f"  Features: {len(self.feature_names)}")
    
    def classify_file(
        self,
        audio_path: str,
        onset_sec: float = 0.0,
        duration_sec: float = None,
        return_proba: bool = False,
    ) -> Dict:
        """
        Classify a single audio file.
        
        Args:
            audio_path: Path to audio file
            onset_sec: Start time in seconds
            duration_sec: Duration to analyze (None = use model default)
            return_proba: Whether to return probability scores
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.extractor.extract_from_file(
            audio_path, onset_sec, duration_sec
        )
        
        # Convert to array in correct order
        X = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        label = self.label_names[prediction]
        
        result = {
            'label': label,
            'class_id': int(prediction),
            'audio_path': str(audio_path),
        }
        
        if return_proba:
            probabilities = self.model.predict_proba(X_scaled)[0]
            result['probabilities'] = {
                name: float(prob) 
                for name, prob in zip(self.label_names, probabilities)
            }
            result['confidence'] = float(np.max(probabilities))
        
        return result
    
    def classify_audio(
        self,
        audio: np.ndarray,
        return_proba: bool = False,
    ) -> Dict:
        """
        Classify raw audio signal.
        
        Args:
            audio: Audio signal as numpy array
            return_proba: Whether to return probability scores
            
        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features = self.extractor.extract_all_features(audio)
        
        # Convert to array in correct order
        X = np.array([features[name] for name in self.feature_names]).reshape(1, -1)
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        label = self.label_names[prediction]
        
        result = {
            'label': label,
            'class_id': int(prediction),
        }
        
        if return_proba:
            probabilities = self.model.predict_proba(X_scaled)[0]
            result['probabilities'] = {
                name: float(prob) 
                for name, prob in zip(self.label_names, probabilities)
            }
            result['confidence'] = float(np.max(probabilities))
        
        return result
    
    def classify_batch(
        self,
        audio_paths: List[str],
        return_proba: bool = False,
    ) -> List[Dict]:
        """
        Classify multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            return_proba: Whether to return probability scores
            
        Returns:
            List of prediction results
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.classify_file(audio_path, return_proba=return_proba)
                results.append(result)
            except Exception as e:
                results.append({
                    'audio_path': str(audio_path),
                    'error': str(e),
                })
        
        return results


def print_result(result: Dict, verbose: bool = False):
    """Pretty print classification result."""
    if 'error' in result:
        print(f"  ✗ {result['audio_path']}")
        print(f"    Error: {result['error']}")
        return
    
    # Get label with color
    label = result['label']
    if label == 'harmonic':
        label_colored = f"\033[92m{label}\033[0m"  # Green
    elif label == 'dead_note':
        label_colored = f"\033[91m{label}\033[0m"  # Red
    else:
        label_colored = f"\033[94m{label}\033[0m"  # Blue
    
    if 'audio_path' in result:
        print(f"  {Path(result['audio_path']).name}")
    print(f"    → {label_colored}", end="")
    
    if 'confidence' in result:
        conf = result['confidence']
        print(f"  (confidence: {conf:.3f})")
    else:
        print()
    
    if verbose and 'probabilities' in result:
        print("    Probabilities:")
        for name, prob in result['probabilities'].items():
            bar_len = int(prob * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"      {name:15s} {bar} {prob:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Classify audio using trained SVM model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a single file
  python inference_svm.py --model models/svm --audio recording.wav
  
  # Classify multiple files with probabilities
  python inference_svm.py --model models/svm --audio *.wav --probabilities
  
  # Classify with verbose output
  python inference_svm.py --model models/svm --audio recording.wav -v
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to SVM model directory'
    )
    parser.add_argument(
        '--audio',
        type=str,
        nargs='+',
        required=True,
        help='Audio file(s) to classify'
    )
    parser.add_argument(
        '--probabilities',
        action='store_true',
        help='Show probability scores for each class'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output with probability bars'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to JSON file (optional)'
    )
    
    args = parser.parse_args()
    
    # Initialize classifier
    print("\n" + "=" * 70)
    print("  HARMONICS SVM CLASSIFIER")
    print("=" * 70 + "\n")
    
    classifier = HarmonicsSVMClassifier(args.model)
    
    # Classify files
    print(f"\nClassifying {len(args.audio)} file(s)...\n")
    
    results = classifier.classify_batch(
        args.audio,
        return_proba=args.probabilities or args.verbose
    )
    
    # Print results
    for result in results:
        print_result(result, verbose=args.verbose)
        print()
    
    # Summary
    successful = [r for r in results if 'error' not in r]
    if successful:
        label_counts = {}
        for r in successful:
            label = r['label']
            label_counts[label] = label_counts.get(label, 0) + 1
        
        print("─" * 70)
        print("Summary:")
        for label, count in sorted(label_counts.items()):
            print(f"  {label:15s}: {count}")
        print(f"  Total:          {len(successful)}/{len(results)}")
        print("=" * 70 + "\n")
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {args.output}\n")


if __name__ == "__main__":
    main()
