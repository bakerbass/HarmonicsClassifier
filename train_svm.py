"""
Train SVM model for guitar harmonics classification.

This script uses traditional machine learning features (MFCCs, spectral features, etc.)
instead of deep learning, providing a simpler, more interpretable baseline.

Usage:
    python train_svm.py --metadata processed_dataset/metadata.csv --output models/
    
    # With hyperparameter tuning
    python train_svm.py --metadata processed_dataset/metadata.csv --tune
    
    # Custom train/test split
    python train_svm.py --metadata processed_dataset/metadata.csv --test-size 0.3
"""

import argparse
import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_class_weight

from feature_extraction import AudioFeatureExtractor, extract_features_from_metadata


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix - SVM Classifier', fontsize=14, pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix to {output_path}")


def plot_feature_importance(model, feature_names, top_n=20, output_path=None):
    """
    Plot feature importance based on SVM weights.
    
    For linear SVM, uses coefficient magnitudes.
    For RBF kernel, this is not applicable.
    """
    if model.kernel != 'linear':
        print("  Feature importance only available for linear kernel")
        return
    
    # Average absolute weights across classes for multi-class
    if len(model.coef_.shape) > 1:
        importances = np.mean(np.abs(model.coef_), axis=0)
    else:
        importances = np.abs(model.coef_)
    
    # Get top N features
    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), top_importances, color='steelblue')
    plt.yticks(range(top_n), top_features)
    plt.xlabel('Importance (|coefficient|)', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14, pad=15)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved feature importance to {output_path}")
    else:
        plt.show()


def tune_hyperparameters(X_train, y_train, class_weights=None, cv=5):
    """
    Perform grid search to find best hyperparameters.
    
    Args:
        X_train: Training features
        y_train: Training labels
        class_weights: Class weight dictionary
        cv: Number of cross-validation folds
        
    Returns:
        Best estimator from grid search
    """
    print("\nHyperparameter Tuning")
    print("=" * 60)
    
    # Parameter grid
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'kernel': ['rbf', 'linear'],
    }
    
    print(f"  Parameter grid: {param_grid}")
    print(f"  Cross-validation folds: {cv}")
    print(f"  This may take several minutes...\n")
    
    # Create SVM with class weights
    svm = SVC(class_weight=class_weights, random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        svm,
        param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=2,
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV F1-macro score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def train_svm(
    metadata_path: str,
    output_dir: str,
    test_size: float = 0.2,
    random_state: int = 42,
    tune: bool = False,
    kernel: str = 'rbf',
    C: float = 10.0,
    gamma: str = 'scale',
    class_weight: str = 'balanced',
):
    """
    Main training function.
    
    Args:
        metadata_path: Path to metadata CSV
        output_dir: Output directory for models and results
        test_size: Fraction of data for testing
        random_state: Random seed
        tune: Whether to perform hyperparameter tuning
        kernel: SVM kernel type
        C: Regularization parameter
        gamma: Kernel coefficient
        class_weight: Class weight strategy ('balanced' or None)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("  SVM HARMONICS CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"  Metadata: {metadata_path}")
    print(f"  Output:   {output_dir}")
    print(f"  Test split: {test_size * 100:.0f}%")
    print(f"=" * 70 + "\n")
    
    # ─── Load metadata ────────────────────────────────────────────────────
    print("Step 1: Loading metadata...")
    df = pd.read_csv(metadata_path)
    print(f"  Total samples: {len(df)}")
    print(f"  Label distribution:")
    for label, count in df['label_category'].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {label:15s}: {count:5d} ({pct:5.1f}%)")
    
    # ─── Extract features ─────────────────────────────────────────────────
    print("\nStep 2: Extracting audio features...")
    extractor = AudioFeatureExtractor(sr=22050, duration=3.0)
    X, y, feature_names = extract_features_from_metadata(df, extractor, verbose=True)
    
    print(f"\n  Feature matrix shape: {X.shape}")
    print(f"  Number of features: {len(feature_names)}")
    print(f"  Sample feature names: {feature_names[:5]} ...")
    
    # ─── Train/test split ─────────────────────────────────────────────────
    print("\nStep 3: Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Training samples:   {len(X_train)}")
    print(f"  Test samples:       {len(X_test)}")
    
    # ─── Feature scaling ──────────────────────────────────────────────────
    print("\nStep 4: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"  Mean: {scaler.mean_[:5]} ...")
    print(f"  Std:  {scaler.scale_[:5]} ...")
    
    # ─── Compute class weights ────────────────────────────────────────────
    if class_weight == 'balanced':
        class_weights_array = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights = {i: w for i, w in enumerate(class_weights_array)}
        print(f"\nClass weights: {class_weights}")
    else:
        class_weights = None
    
    # ─── Train SVM ────────────────────────────────────────────────────────
    print("\nStep 5: Training SVM...")
    
    if tune:
        # Hyperparameter tuning
        model = tune_hyperparameters(X_train_scaled, y_train, class_weights)
    else:
        # Train with specified parameters
        print(f"  Kernel: {kernel}")
        print(f"  C: {C}")
        print(f"  Gamma: {gamma}")
        print(f"  Class weight: {class_weight}")
        
        model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            class_weight=class_weights,
            random_state=random_state,
            probability=True,  # Enable probability estimates
        )
        
        model.fit(X_train_scaled, y_train)
        print("  Training complete!")
    
    # ─── Evaluate on training set ─────────────────────────────────────────
    print("\nStep 6: Evaluating on training set...")
    y_train_pred = model.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average='macro')
    print(f"  Training accuracy: {train_acc:.4f}")
    print(f"  Training F1-macro: {train_f1:.4f}")
    
    # ─── Evaluate on test set ─────────────────────────────────────────────
    print("\nStep 7: Evaluating on test set...")
    y_test_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='macro')
    
    print(f"  Test accuracy: {test_acc:.4f}")
    print(f"  Test F1-macro: {test_f1:.4f}")
    
    # Detailed classification report
    label_names = ['harmonic', 'dead_note', 'general_note']
    print("\n  Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=label_names))
    
    # ─── Cross-validation ─────────────────────────────────────────────────
    print("\nStep 8: Cross-validation (5-fold)...")
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, cv=5, scoring='f1_macro', n_jobs=-1
    )
    print(f"  CV F1-macro: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # ─── Save results ─────────────────────────────────────────────────────
    print("\nStep 9: Saving model and results...")
    
    # Save model
    model_path = output_path / "svm_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved model: {model_path}")
    
    # Save scaler
    scaler_path = output_path / "svm_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  Saved scaler: {scaler_path}")
    
    # Save feature extractor config
    extractor_config = {
        'sr': extractor.sr,
        'duration': extractor.duration,
        'n_mfcc': extractor.n_mfcc,
        'n_fft': extractor.n_fft,
        'hop_length': extractor.hop_length,
        'n_chroma': extractor.n_chroma,
        'feature_names': feature_names,
    }
    config_path = output_path / "feature_config.json"
    with open(config_path, 'w') as f:
        json.dump(extractor_config, f, indent=2)
    print(f"  Saved feature config: {config_path}")
    
    # Save results
    results = {
        'model_type': 'SVM',
        'kernel': model.kernel,
        'C': model.C,
        'gamma': model.gamma,
        'n_features': len(feature_names),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'train_accuracy': float(train_acc),
        'train_f1_macro': float(train_f1),
        'test_accuracy': float(test_acc),
        'test_f1_macro': float(test_f1),
        'cv_f1_mean': float(cv_scores.mean()),
        'cv_f1_std': float(cv_scores.std()),
        'timestamp': datetime.now().isoformat(),
    }
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_test_pred, labels=[0, 1, 2]
    )
    for i, label in enumerate(label_names):
        results[f'{label}_precision'] = float(precision[i])
        results[f'{label}_recall'] = float(recall[i])
        results[f'{label}_f1'] = float(f1[i])
        results[f'{label}_support'] = int(support[i])
    
    results_path = output_path / "svm_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results: {results_path}")
    
    # ─── Visualizations ───────────────────────────────────────────────────
    print("\nStep 10: Creating visualizations...")
    
    # Confusion matrix
    cm_path = output_path / "confusion_matrix_svm.png"
    plot_confusion_matrix(y_test, y_test_pred, label_names, cm_path)
    
    # Feature importance (for linear kernel)
    if model.kernel == 'linear':
        fi_path = output_path / "feature_importance_svm.png"
        plot_feature_importance(model, feature_names, top_n=20, output_path=fi_path)
    
    # ─── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Test F1-macro:  {test_f1:.4f}")
    print(f"  CV F1-macro:    {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"\n  Model saved to: {output_path}")
    print("=" * 70 + "\n")
    
    return model, scaler, results


def main():
    parser = argparse.ArgumentParser(
        description="Train SVM model for harmonics classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--metadata',
        type=str,
        required=True,
        help='Path to metadata CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/svm',
        help='Output directory for model and results (default: models/svm)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Fraction of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--tune',
        action='store_true',
        help='Perform hyperparameter tuning via grid search'
    )
    parser.add_argument(
        '--kernel',
        type=str,
        default='rbf',
        choices=['rbf', 'linear', 'poly', 'sigmoid'],
        help='SVM kernel (default: rbf)'
    )
    parser.add_argument(
        '--C',
        type=float,
        default=10.0,
        help='Regularization parameter (default: 10.0)'
    )
    parser.add_argument(
        '--gamma',
        type=str,
        default='scale',
        help='Kernel coefficient (default: scale)'
    )
    parser.add_argument(
        '--no-class-weight',
        action='store_true',
        help='Disable balanced class weights'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    class_weight = None if args.no_class_weight else 'balanced'
    
    train_svm(
        metadata_path=args.metadata,
        output_dir=args.output,
        test_size=args.test_size,
        random_state=args.seed,
        tune=args.tune,
        kernel=args.kernel,
        C=args.C,
        gamma=args.gamma,
        class_weight=class_weight,
    )


if __name__ == "__main__":
    main()
