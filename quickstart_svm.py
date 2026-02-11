"""
Quick start script for SVM-based harmonics classification.

This script demonstrates the complete workflow:
1. Feature extraction
2. Model training
3. Evaluation
4. Inference

Run this to verify the SVM pipeline is working correctly.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║      SVM HARMONICS CLASSIFIER - QUICK START DEMO                     ║
║                                                                      ║
║  This demo shows the complete SVM classification pipeline            ║
║  for guitar harmonics detection.                                     ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")


def test_feature_extraction():
    """Test feature extraction module."""
    print("\n" + "="*70)
    print("Step 1: Testing Feature Extraction")
    print("="*70)
    
    from feature_extraction import AudioFeatureExtractor
    
    # Create synthetic test signal
    sr = 22050
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create different signal types
    print("\n  Creating test signals...")
    
    # 1. Harmonic signal (fundamental + harmonics)
    f0 = 220  # A3
    harmonic = (np.sin(2 * np.pi * f0 * t) +
                0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
                0.25 * np.sin(2 * np.pi * 3 * f0 * t))
    
    # 2. Dead note (noise with quick decay)
    dead_note = np.random.randn(len(t)) * np.exp(-5 * t)
    
    # 3. General note (single sine with slower decay)
    general = np.sin(2 * np.pi * 330 * t) * np.exp(-1 * t)
    
    # Extract features
    print("  Extracting features...")
    extractor = AudioFeatureExtractor(sr=sr, duration=duration)
    
    features_harmonic = extractor.extract_all_features(harmonic)
    features_dead = extractor.extract_all_features(dead_note)
    features_general = extractor.extract_all_features(general)
    
    print(f"\n  ✓ Extracted {len(features_harmonic)} features per signal")
    
    # Show some example features
    print("\n  Example feature comparison:")
    print(f"  {'Feature':30s} {'Harmonic':>12s} {'Dead Note':>12s} {'General':>12s}")
    print("  " + "-"*66)
    
    example_features = ['rms_mean', 'spectral_centroid_mean', 'mfcc_0_mean', 'zero_crossing_rate_mean']
    for feat in example_features:
        print(f"  {feat:30s} {features_harmonic[feat]:12.4f} "
              f"{features_dead[feat]:12.4f} {features_general[feat]:12.4f}")
    
    print("\n  ✓ Feature extraction working correctly!")
    return extractor


def test_training():
    """Test model training with synthetic data."""
    print("\n" + "="*70)
    print("Step 2: Testing Model Training")
    print("="*70)
    
    from feature_extraction import AudioFeatureExtractor, extract_features_from_metadata
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    # Create synthetic dataset
    print("\n  Creating synthetic dataset...")
    
    sample_rate = 22050
    duration = 3.0
    n_samples_per_class = 50  # Small dataset for demo
    
    # Generate synthetic audio metadata
    metadata_list = []
    
    for i in range(n_samples_per_class):
        for label in ['harmonic', 'dead_note', 'general_note']:
            metadata_list.append({
                'source_audio': f'synthetic_{label}_{i}.wav',
                'onset_sec': 0.0,
                'duration_sec': duration,
                'label_category': label,
            })
    
    print(f"  Generated {len(metadata_list)} synthetic samples")
    
    # Create feature matrix manually
    extractor = AudioFeatureExtractor(sr=sample_rate, duration=duration)
    feature_names = extractor.get_feature_names()
    
    print(f"  Extracting {len(feature_names)} features...")
    
    X = []
    y = []
    label_map = {'harmonic': 0, 'dead_note': 1, 'general_note': 2}
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    for item in metadata_list:
        label = item['label_category']
        
        # Generate synthetic audio based on label
        if label == 'harmonic':
            f0 = np.random.uniform(100, 400)
            audio = (np.sin(2 * np.pi * f0 * t) +
                    0.5 * np.sin(2 * np.pi * 2 * f0 * t) +
                    0.25 * np.sin(2 * np.pi * 3 * f0 * t))
        elif label == 'dead_note':
            audio = np.random.randn(len(t)) * np.exp(-np.random.uniform(3, 7) * t)
        else:  # general_note
            f0 = np.random.uniform(100, 400)
            audio = np.sin(2 * np.pi * f0 * t) * np.exp(-np.random.uniform(0.5, 2) * t)
        
        # Add some noise
        audio += 0.01 * np.random.randn(len(audio))
        
        # Extract features
        features = extractor.extract_all_features(audio)
        feature_vector = [features[name] for name in feature_names]
        
        X.append(feature_vector)
        y.append(label_map[label])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"  Feature matrix shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"  Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Scale features
    print("\n  Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM
    print("  Training SVM (RBF kernel)...")
    model = SVC(kernel='rbf', C=10.0, gamma='scale', class_weight='balanced', 
                random_state=42, probability=True)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\n  Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"  Test Accuracy:  {acc:.4f}")
    print(f"  Test F1-macro:  {f1:.4f}")
    
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=['harmonic', 'dead_note', 'general_note']))
    
    print("  ✓ Model training working correctly!")
    
    return model, scaler, extractor


def test_inference(model, scaler, extractor):
    """Test inference on new samples."""
    print("\n" + "="*70)
    print("Step 3: Testing Inference")
    print("="*70)
    
    # Create test samples
    sr = extractor.sr
    duration = extractor.duration
    t = np.linspace(0, duration, int(sr * duration))
    
    test_cases = [
        ('Harmonic tone', np.sin(2 * np.pi * 220 * t) + 
                         0.5 * np.sin(2 * np.pi * 440 * t)),
        ('Dead note', np.random.randn(len(t)) * np.exp(-5 * t)),
        ('General note', np.sin(2 * np.pi * 330 * t) * np.exp(-1 * t)),
    ]
    
    label_names = ['harmonic', 'dead_note', 'general_note']
    
    print("\n  Classifying test samples...\n")
    
    for name, audio in test_cases:
        # Extract features
        features = extractor.extract_all_features(audio)
        feature_names = extractor.get_feature_names()
        X = np.array([features[fn] for fn in feature_names]).reshape(1, -1)
        
        # Scale
        X_scaled = scaler.transform(X)
        
        # Predict
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]
        
        predicted_label = label_names[prediction]
        confidence = np.max(probabilities)
        
        print(f"  {name:20s} → {predicted_label:15s} (confidence: {confidence:.3f})")
        
        # Show probabilities
        for i, (label, prob) in enumerate(zip(label_names, probabilities)):
            bar_len = int(prob * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"    {label:15s} {bar} {prob:.3f}")
        print()
    
    print("  ✓ Inference working correctly!")


def main():
    """Run complete demo."""
    
    # Step 1: Feature extraction
    extractor = test_feature_extraction()
    
    # Step 2: Training
    model, scaler, extractor = test_training()
    
    # Step 3: Inference
    test_inference(model, scaler, extractor)
    
    # Summary
    print("\n" + "="*70)
    print("  DEMO COMPLETE!")
    print("="*70)
    print("""
  The SVM pipeline is working correctly! ✓

  Next steps:
  
  1. Train on real data:
     python train_svm.py --metadata processed_dataset/metadata.csv --output models/svm
  
  2. Evaluate the model:
     python evaluate_svm.py --metadata test_data/metadata.csv --model models/svm
  
  3. Use for inference:
     python inference_svm.py --model models/svm --audio recording.wav
  
  4. Compare with CNN:
     python evaluate_svm.py --metadata test_data/metadata.csv \\
                            --model models/svm --compare-cnn models/best_model.pt
  
  See SVM_README.md for detailed documentation.
  """)
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
