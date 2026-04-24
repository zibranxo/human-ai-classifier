"""
GPU-OPTIMIZED AI vs Human Text Classifier Training Pipeline
PyTorch Random Forest (GPU) + XGBoost (GPU) Ensemble
Optimized for RTX 4050 with 10K dataset
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
import torch
import time
import warnings
warnings.filterwarnings('ignore')

from gpu_feature_extractor import GPUFeatureExtractor
from pytorch_random_forest import PyTorchRandomForest
from gpu_xgboost_trainer import GPUXGBoostTrainer
from gpu_hyperparameter_tuner import GPUHyperparameterTuner
from gpu_ensemble import GPUEnsembleClassifier
from metrics import MetricsCalculator


def print_gpu_info():
    """Print GPU information"""
    print("\n" + "="*70)
    print("  GPU INFORMATION")
    print("="*70)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA Available: YES")
        print(f"✓ GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ PyTorch Version: {torch.__version__}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"✗ CUDA Available: NO")
        print(f"  Training will use CPU (slower)")
    
    print("="*70)


def load_data(filepath):
    """Load dataset from CSV"""
    print("\n" + "="*70)
    print("  LOADING DATASET")
    print("="*70)
    
    df = pd.read_csv(filepath)
    
    # Check required columns
    if 'text' not in df.columns:
        raise ValueError("Dataset must have a 'text' column")
    
    # Handle label column
    label_col = None
    for col in ['label', 'label_id', 'is_ai', 'class']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Dataset must have a label column")
    
    # Convert labels to binary (0=human, 1=AI)
    df['label_binary'] = df[label_col].apply(
        lambda x: 1 if x in [1, '1', 'ai', 'AI', 'machine', 'bot'] else 0
    )
    
    print(f"✓ Loaded {len(df)} samples")
    print(f"  Human samples: {(df['label_binary'] == 0).sum()}")
    print(f"  AI samples: {(df['label_binary'] == 1).sum()}")
    print(f"  Class balance: {(df['label_binary'] == 1).sum() / len(df) * 100:.2f}% AI")
    
    return df['text'].to_numpy(), df['label_binary'].to_numpy()


def extract_and_select_features(texts, labels, top_k_features=40, device='cuda'):
    """Extract features and select most important ones"""
    print("\n" + "="*70)
    print("  FEATURE EXTRACTION & SELECTION (GPU)")
    print("="*70)
    
    # Extract features with GPU acceleration
    extractor = GPUFeatureExtractor(device=device, batch_size=64)
    features = extractor.extract_features(texts)
    feature_names = extractor.feature_names
    
    print(f"\n✓ Extracted {features.shape[1]} features")
    
    # Feature selection using mutual information
    print(f"\n📊 Selecting top {top_k_features} features...")
    
    selector = SelectKBest(mutual_info_classif, k=min(top_k_features, features.shape[1]))
    features_selected = selector.fit_transform(features, labels)
    
    # Get selected feature indices and names
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_indices]
    
    print(f"✓ Selected {len(selected_feature_names)} features")
    
    # Get feature scores
    scores = selector.scores_
    feature_scores = [(feature_names[i], scores[i]) for i in selected_indices]
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n📈 Top 10 features:")
    for i, (name, score) in enumerate(feature_scores[:10], 1):
        print(f"  {i}. {name}: {score:.4f}")
    
    # Save feature information
    feature_info = {
        'selected_features': selected_feature_names,
        'feature_scores': {name: float(score) for name, score in feature_scores}
    }
    
    return features_selected, selected_feature_names, feature_info, extractor


def train_ensemble(X_train, y_train, X_val, y_val, 
                   tune_hyperparameters=False, 
                   n_trials=25,
                   device='cuda'):
    """Train GPU ensemble model"""
    print("\n" + "="*70)
    print("  TRAINING GPU ENSEMBLE MODEL")
    print("="*70)
    
    rf_params = {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5}
    xgb_params = {}
    
    # Hyperparameter tuning (optional)
    if tune_hyperparameters:
        print("\n🎯 Starting hyperparameter optimization...")
        print(f"  (Optimized for 10K dataset: {n_trials} trials, 3-fold CV)")
        
        tuner = GPUHyperparameterTuner(
            n_jobs=4,  # Parallel trials
            n_trials=n_trials,
            cv_folds=3  # Faster for 10K dataset
        )
        
        # Combine train and val for tuning
        X_tune = np.vstack([X_train, X_val])
        y_tune = np.hstack([y_train, y_val])
        
        all_params = tuner.tune_both(X_tune, y_tune, use_gpu=True)
        rf_params = all_params['pytorch_random_forest']
        xgb_params = all_params['xgboost']
    
    # Train PyTorch Random Forest (GPU)
    print("\n" + "-"*70)
    rf_model = PyTorchRandomForest(
        device=device,
        verbose=1,
        **rf_params
    )
    rf_model.fit(X_train, y_train)
    
    # Validation performance
    rf_val_acc = rf_model.score(X_val, y_val)
    print(f"✓ Validation Accuracy: {rf_val_acc:.4f}")
    
    # Train XGBoost (GPU)
    print("\n" + "-"*70)
    xgb_trainer = GPUXGBoostTrainer(use_gpu=True, use_fp16=False, **xgb_params)
    xgb_model = xgb_trainer.train(X_train, y_train, X_val, y_val, early_stopping_rounds=30)
    
    # Create ensemble
    print("\n" + "-"*70)
    ensemble = GPUEnsembleClassifier(rf_model, xgb_model)
    ensemble.optimize_weights(X_val, y_val)
    
    return ensemble, rf_model, xgb_model


def main():
    parser = argparse.ArgumentParser(description='GPU-Optimized AI vs Human Text Classifier')
    parser.add_argument('--data', type=str, required=True, help='Path to training data CSV')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--n-trials', type=int, default=25, help='Number of tuning trials (25 optimal for 10K)')
    parser.add_argument('--top-features', type=int, default=40, help='Number of top features (40 optimal for 10K)')
    parser.add_argument('--test-size', type=float, default=0.1, help='Test set size')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--output-dir', type=str, default='./models_gpu', help='Output directory')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode (disable GPU)')
    
    args = parser.parse_args()
    
    # Set device
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    
    # Print GPU info
    print_gpu_info()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    start_time = time.time()
    
    # 1. Load data
    texts, labels = load_data(args.data)
    
    # 2. Split data
    print("\n" + "="*70)
    print("  SPLITTING DATA")
    print("="*70)
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=args.test_size, random_state=42, stratify=labels
    )
    
    val_size_adjusted = args.val_size / (1 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Validation set: {len(X_val)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # 3. Feature extraction and selection (GPU)
    X_train_feats, selected_features, feature_info, extractor = extract_and_select_features(
        X_train, y_train, 
        top_k_features=args.top_features,
        device=device
    )
    
    # Transform validation and test sets
    print("\n📊 Transforming validation and test sets...")
    X_val_full = extractor.extract_features(X_val)
    X_test_full = extractor.extract_features(X_test)
    
    # Select same features
    feature_indices = [extractor.feature_names.index(f) for f in selected_features]
    X_val_feats = X_val_full[:, feature_indices]
    X_test_feats = X_test_full[:, feature_indices]
    
    # 4. Feature scaling
    print("\n📊 Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_feats)
    X_val_scaled = scaler.transform(X_val_feats)
    X_test_scaled = scaler.transform(X_test_feats)
    print(f"✓ Features scaled to mean=0, std=1")
    
    # 5. Train ensemble (GPU)
    ensemble, rf_model, xgb_model = train_ensemble(
        X_train_scaled, y_train, 
        X_val_scaled, y_val,
        tune_hyperparameters=args.tune,
        n_trials=args.n_trials,
        device=device
    )
    
    # 6. Evaluate on test set
    print("\n" + "="*70)
    print("  FINAL EVALUATION ON TEST SET")
    print("="*70)
    
    calc = MetricsCalculator()
    
    # Individual models
    print("\n🌲 PyTorch Random Forest (GPU):")
    rf_pred = rf_model.predict(X_test_scaled)
    rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
    rf_metrics = calc.calculate_all_metrics(y_test, rf_pred, rf_proba)
    calc.print_metrics(rf_metrics, "PyTorch Random Forest - Test Set")
    
    print("\n⚡ XGBoost (GPU):")
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    xgb_metrics = calc.calculate_all_metrics(y_test, xgb_pred, xgb_proba)
    calc.print_metrics(xgb_metrics, "XGBoost - Test Set")
    
    print("\n🚀 GPU Ensemble:")
    ensemble_pred = ensemble.predict(X_test_scaled)
    ensemble_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
    ensemble_metrics = calc.calculate_all_metrics(y_test, ensemble_pred, ensemble_proba)
    calc.print_metrics(ensemble_metrics, "GPU Ensemble - Test Set")
    
    # 7. Save models
    print("\n" + "="*70)
    print("  SAVING MODELS")
    print("="*70)
    
    # Save ensemble
    ensemble_path = os.path.join(args.output_dir, 'gpu_ensemble_model.pkl')
    ensemble.save(ensemble_path)
    
    # Save pipeline
    pipeline_path = os.path.join(args.output_dir, 'gpu_pipeline.pkl')
    pipeline_data = {
        'extractor': extractor,
        'scaler': scaler,
        'selected_features': selected_features,
        'feature_indices': feature_indices,
        'device': device
    }
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline_data, f)
    print(f"✓ Pipeline saved to {pipeline_path}")
    
    # Save feature info
    feature_info_path = os.path.join(args.output_dir, 'feature_info.json')
    with open(feature_info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"✓ Feature info saved to {feature_info_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    metrics_summary = {
        'pytorch_random_forest': {
            k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
            for k, v in rf_metrics.items() if k != 'confusion_matrix'
        },
        'xgboost': {
            k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
            for k, v in xgb_metrics.items() if k != 'confusion_matrix'
        },
        'gpu_ensemble': {
            k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
            for k, v in ensemble_metrics.items() if k != 'confusion_matrix'
        }
    }
    with open(metrics_path, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")
    
    # Plot curves
    try:
        calc.plot_confusion_matrix(
            ensemble_metrics['confusion_matrix'],
            os.path.join(args.output_dir, 'confusion_matrix.png')
        )
        calc.plot_roc_curve(
            y_test, ensemble_proba,
            os.path.join(args.output_dir, 'roc_curve.png')
        )
        calc.plot_pr_curve(
            y_test, ensemble_proba,
            os.path.join(args.output_dir, 'pr_curve.png')
        )
        print("✓ Plots saved to output directory")
    except Exception as e:
        print(f"⚠ Could not save plots: {e}")
    
    # 8. Summary
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("  🎉 TRAINING COMPLETE")
    print("="*70)
    print(f"\n⏱️  Total training time: {total_time/60:.2f} minutes")
    print(f"\n📊 Final GPU Ensemble Test Metrics:")
    print(f"  Accuracy : {ensemble_metrics['accuracy']:.4f}")
    print(f"  F1 Score : {ensemble_metrics['f1']:.4f}")
    print(f"  MCC      : {ensemble_metrics['mcc']:.4f}")
    print(f"  AUC-ROC  : {ensemble_metrics['auc_roc']:.4f}")
    print(f"\n💾 Models saved to: {args.output_dir}")
    print(f"\n🚀 GPU Utilization: {'FULL' if device == 'cuda' else 'NONE (CPU MODE)'}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
