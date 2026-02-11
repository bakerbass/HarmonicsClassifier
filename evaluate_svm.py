"""
Evaluate and compare SVM and CNN models for harmonics classification.

This script loads saved models and evaluates them on test data,
providing detailed comparisons and performance metrics.

Usage:
    # Evaluate SVM model
    python evaluate_svm.py --metadata test_data/metadata.csv --model models/svm/svm_model.pkl
    
    # Compare SVM and CNN
    python evaluate_svm.py --metadata test_data/metadata.csv --model models/svm/svm_model.pkl --compare-cnn models/best_model.pt
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)

from feature_extraction import AudioFeatureExtractor, extract_features_from_metadata


def load_svm_model(model_dir):
    """Load SVM model, scaler, and feature config."""
    model_dir = Path(model_dir)
    
    # Load model
    with open(model_dir / "svm_model.pkl", 'rb') as f:
        model = pickle.load(f)
    
    # Load scaler
    with open(model_dir / "svm_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    
    # Load feature config
    with open(model_dir / "feature_config.json", 'r') as f:
        config = json.load(f)
    
    return model, scaler, config


def load_cnn_model(model_path):
    """Load PyTorch CNN model."""
    # Import CNN class
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from train_cnn import HarmonicsCNN
    
    model = HarmonicsCNN(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model


def evaluate_svm(metadata_path, model_dir, output_dir=None):
    """
    Evaluate SVM model on test data.
    
    Args:
        metadata_path: Path to test metadata CSV
        model_dir: Directory containing SVM model files
        output_dir: Optional output directory for results
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\n" + "=" * 70)
    print("  SVM MODEL EVALUATION")
    print("=" * 70)
    
    # Load model
    print("\nLoading SVM model...")
    model, scaler, config = load_svm_model(model_dir)
    print(f"  Model: {model}")
    print(f"  Kernel: {model.kernel}")
    print(f"  C: {model.C}")
    print(f"  Gamma: {model.gamma}")
    
    # Load metadata
    print("\nLoading test data...")
    df = pd.read_csv(metadata_path)
    print(f"  Test samples: {len(df)}")
    
    # Extract features
    print("\nExtracting features...")
    extractor = AudioFeatureExtractor(
        sr=config['sr'],
        duration=config['duration'],
        n_mfcc=config['n_mfcc'],
        n_fft=config['n_fft'],
        hop_length=config['hop_length'],
        n_chroma=config['n_chroma'],
    )
    X, y, feature_names = extract_features_from_metadata(df, extractor, verbose=True)
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Predict
    print("\nMaking predictions...")
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    
    # Compute metrics
    accuracy = accuracy_score(y, y_pred)
    f1_macro = f1_score(y, y_pred, average='macro')
    f1_weighted = f1_score(y, y_pred, average='weighted')
    
    label_names = ['harmonic', 'dead_note', 'general_note']
    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, labels=[0, 1, 2]
    )
    
    # Print results
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"  Accuracy:       {accuracy:.4f}")
    print(f"  F1-macro:       {f1_macro:.4f}")
    print(f"  F1-weighted:    {f1_weighted:.4f}")
    print("\n  Per-class metrics:")
    for i, label in enumerate(label_names):
        print(f"    {label:15s}: P={precision[i]:.3f}  R={recall[i]:.3f}  F1={f1[i]:.3f}  (n={support[i]})")
    
    print("\n  Classification Report:")
    print(classification_report(y, y_pred, target_names=label_names))
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print("\n  Confusion Matrix:")
    print(cm)
    
    # Save results if output directory specified
    results = {
        'model_type': 'SVM',
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
    }
    
    for i, label in enumerate(label_names):
        results[f'{label}_precision'] = float(precision[i])
        results[f'{label}_recall'] = float(recall[i])
        results[f'{label}_f1'] = float(f1[i])
        results[f'{label}_support'] = int(support[i])
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_path / "svm_eval_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_names, yticklabels=label_names)
        plt.title('SVM Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path / "svm_confusion_matrix.png", dpi=150)
        plt.close()
        
        print(f"\n  Results saved to: {output_path}")
    
    print("=" * 70 + "\n")
    
    return results, y_pred, y_proba


def compare_models(metadata_path, svm_model_dir, cnn_model_path, output_dir):
    """
    Compare SVM and CNN models side by side.
    
    Args:
        metadata_path: Path to test metadata CSV
        svm_model_dir: Directory containing SVM model
        cnn_model_path: Path to CNN model file
        output_dir: Output directory for comparison results
    """
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON: SVM vs CNN")
    print("=" * 70)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Evaluate SVM
    svm_results, svm_pred, svm_proba = evaluate_svm(
        metadata_path, svm_model_dir, output_dir=None
    )
    
    # TODO: Evaluate CNN (would need to implement CNN evaluation)
    # For now, we can load CNN results if they exist
    cnn_results_path = Path(cnn_model_path).parent / "results.json"
    if cnn_results_path.exists():
        with open(cnn_results_path, 'r') as f:
            cnn_results = json.load(f)
        
        print("\nCNN Results (from saved metrics):")
        print(f"  Test Accuracy: {cnn_results.get('test_accuracy', 'N/A')}")
        print(f"  Test F1-macro: {cnn_results.get('test_f1_macro', 'N/A')}")
    else:
        print("\nCNN results not found. Skipping comparison.")
        cnn_results = None
    
    # Create comparison visualization
    if cnn_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        models = ['SVM', 'CNN']
        accuracies = [
            svm_results['accuracy'],
            cnn_results.get('test_accuracy', 0)
        ]
        f1_scores = [
            svm_results['f1_macro'],
            cnn_results.get('test_f1_macro', 0)
        ]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0].bar(x - width/2, accuracies, width, label='Accuracy', color='steelblue')
        axes[0].bar(x + width/2, f1_scores, width, label='F1-macro', color='coral')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models)
        axes[0].legend()
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Per-class F1 comparison
        labels = ['harmonic', 'dead_note', 'general_note']
        svm_f1s = [svm_results[f'{l}_f1'] for l in labels]
        cnn_f1s = [cnn_results.get(f'{l}_f1', 0) for l in labels]
        
        x = np.arange(len(labels))
        axes[1].bar(x - width/2, svm_f1s, width, label='SVM', color='steelblue')
        axes[1].bar(x + width/2, cnn_f1s, width, label='CNN', color='coral')
        axes[1].set_ylabel('F1 Score')
        axes[1].set_title('Per-Class F1 Scores')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=15)
        axes[1].legend()
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "model_comparison.png", dpi=150)
        plt.close()
        
        print(f"\n  Comparison plot saved to: {output_path / 'model_comparison.png'}")
        
        # Save comparison table
        comparison = {
            'SVM': svm_results,
            'CNN': cnn_results,
        }
        with open(output_path / "model_comparison.json", 'w') as f:
            json.dump(comparison, f, indent=2)
    
    print("\n" + "=" * 70)
    print("  COMPARISON COMPLETE")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SVM model for harmonics classification"
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to test metadata CSV'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to SVM model directory (containing svm_model.pkl)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/svm_eval',
        help='Output directory for evaluation results'
    )
    parser.add_argument(
        '--compare-cnn',
        type=str,
        default=None,
        help='Path to CNN model for comparison (optional)'
    )
    
    args = parser.parse_args()
    
    if args.compare_cnn:
        compare_models(
            args.metadata,
            args.model,
            args.compare_cnn,
            args.output
        )
    else:
        evaluate_svm(
            args.metadata,
            args.model,
            args.output
        )


if __name__ == "__main__":
    main()
