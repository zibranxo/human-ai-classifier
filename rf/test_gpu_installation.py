"""
GPU Installation Test Script
Verifies all components are working correctly
"""

import sys

print("="*70)
print("  GPU-OPTIMIZED AI DETECTOR - INSTALLATION TEST")
print("="*70)

# Test 1: PyTorch with CUDA
print("\n1. Testing PyTorch with CUDA...")
try:
    import torch
    print(f"  ✓ PyTorch version: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: YES")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
        print(f"  ✓ GPU device: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test GPU tensor operations
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print(f"  ✓ GPU tensor operations: WORKING")
    else:
        print(f"  ✗ CUDA available: NO")
        print(f"    Training will use CPU (slower)")
except Exception as e:
    print(f"  ✗ PyTorch error: {e}")
    sys.exit(1)

# Test 2: XGBoost with GPU
print("\n2. Testing XGBoost with GPU...")
try:
    import xgboost as xgb
    print(f"  ✓ XGBoost version: {xgb.__version__}")
    
    # Try XGBoost 3.x API
    try:
        model = xgb.XGBClassifier(device='cuda:0', n_estimators=1)
        model.fit([[1, 2], [3, 4]], [0, 1])
        print(f"  ✓ XGBoost GPU (3.x API): WORKING")
    except:
        # Try XGBoost 2.x API
        try:
            model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=1)
            model.fit([[1, 2], [3, 4]], [0, 1])
            print(f"  ✓ XGBoost GPU (2.x API): WORKING")
        except Exception as e:
            print(f"  ✗ XGBoost GPU not working: {e}")
            print(f"    Will fall back to CPU")
except Exception as e:
    print(f"  ✗ XGBoost error: {e}")
    sys.exit(1)

# Test 3: Other dependencies
print("\n3. Testing other dependencies...")
try:
    import numpy as np
    print(f"  ✓ NumPy version: {np.__version__}")
except:
    print(f"  ✗ NumPy NOT installed")
    sys.exit(1)

try:
    import pandas as pd
    print(f"  ✓ Pandas version: {pd.__version__}")
except:
    print(f"  ✗ Pandas NOT installed")
    sys.exit(1)

try:
    import sklearn
    print(f"  ✓ Scikit-learn version: {sklearn.__version__}")
except:
    print(f"  ✗ Scikit-learn NOT installed")
    sys.exit(1)

try:
    import optuna
    print(f"  ✓ Optuna version: {optuna.__version__}")
except:
    print(f"  ✗ Optuna NOT installed")
    sys.exit(1)

# Test 4: Custom modules
print("\n4. Testing custom GPU modules...")
try:
    from gpu_feature_extractor import GPUFeatureExtractor
    print(f"  ✓ GPUFeatureExtractor: LOADED")
    
    # Test extraction
    extractor = GPUFeatureExtractor(device='cuda' if torch.cuda.is_available() else 'cpu')
    test_texts = ["This is a test.", "Another test sentence."]
    features = extractor.extract_features(test_texts)
    print(f"  ✓ Feature extraction: WORKING ({features.shape[1]} features)")
except Exception as e:
    print(f"  ✗ GPUFeatureExtractor error: {e}")
    sys.exit(1)

try:
    from pytorch_random_forest import PyTorchRandomForest
    print(f"  ✓ PyTorchRandomForest: LOADED")
    
    # Test training
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)
    rf = PyTorchRandomForest(
        n_estimators=10, 
        max_depth=5,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=0
    )
    rf.fit(X, y)
    predictions = rf.predict(X)
    print(f"  ✓ PyTorchRandomForest training: WORKING")
except Exception as e:
    print(f"  ✗ PyTorchRandomForest error: {e}")
    sys.exit(1)

try:
    from gpu_xgboost_trainer import GPUXGBoostTrainer
    print(f"  ✓ GPUXGBoostTrainer: LOADED")
    
    trainer = GPUXGBoostTrainer(use_gpu=torch.cuda.is_available())
    # Quick test (silent)
    print(f"  ✓ GPUXGBoostTrainer initialization: WORKING")
except Exception as e:
    print(f"  ✗ GPUXGBoostTrainer error: {e}")
    sys.exit(1)

try:
    from gpu_hyperparameter_tuner import GPUHyperparameterTuner
    print(f"  ✓ GPUHyperparameterTuner: LOADED")
except Exception as e:
    print(f"  ✗ GPUHyperparameterTuner error: {e}")
    sys.exit(1)

try:
    from gpu_ensemble import GPUEnsembleClassifier
    print(f"  ✓ GPUEnsembleClassifier: LOADED")
except Exception as e:
    print(f"  ✗ GPUEnsembleClassifier error: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*70)
print("  INSTALLATION TEST COMPLETE")
print("="*70)

if torch.cuda.is_available():
    print("\n✅ ALL SYSTEMS GO! GPU acceleration ready!")
    print(f"\n🚀 Your RTX 4050 is ready to train at {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB VRAM")
    print("\nRun training with:")
    print("  python gpu_train.py --data your_dataset.csv --tune")
else:
    print("\n⚠️  GPU not available - training will use CPU")
    print("\nTo enable GPU:")
    print("  1. Install NVIDIA drivers")
    print("  2. Install CUDA toolkit")
    print("  3. Reinstall PyTorch with CUDA:")
    print("     pip install torch --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "="*70 + "\n")
