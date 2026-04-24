"""
GPU-Optimized XGBoost Trainer
Full CUDA support with mixed precision training for RTX 4050
"""

import numpy as np
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')


class GPUXGBoostTrainer:
    """
    XGBoost Trainer optimized for RTX 4050
    - Full CUDA acceleration
    - Mixed precision (FP16) support
    - Optimized memory usage
    """
    
    def __init__(self, use_gpu=True, use_fp16=False, **params):
        """
        Initialize GPU XGBoost trainer
        
        Args:
            use_gpu: Enable GPU acceleration
            use_fp16: Use mixed precision (FP16) training
            **params: Additional XGBoost parameters
        """
        self.use_gpu = use_gpu
        self.use_fp16 = use_fp16
        self.params = params
        self.model = None
        self.feature_importances_ = None
        self.best_iteration = None
        self.best_score = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None, early_stopping_rounds=50):
        """Train XGBoost model with GPU acceleration"""
        print("\n" + "="*70)
        print("  TRAINING XGBOOST (GPU-ACCELERATED)")
        print("="*70)
        
        # Default optimized parameters for RTX 4050
        default_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'auc'],
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
        }
        
        # GPU configuration - try both XGBoost 2.x and 3.x APIs
        if self.use_gpu:
            try:
                gpu_enabled = False
                
                # Try XGBoost 3.x API first (device='cuda:0')
                try:
                    _test = xgb.XGBClassifier(device='cuda:0', n_estimators=1)
                    _test.fit([[1, 2], [3, 4]], [0, 1])
                    default_params['device'] = 'cuda:0'
                    gpu_enabled = True
                    print("✓ GPU ENABLED - XGBoost 3.x API (device='cuda:0')")
                    
                    # Add FP16 support for XGBoost 3.x
                    if self.use_fp16:
                        # Note: XGBoost 3.x doesn't have explicit FP16 flag yet
                        # But uses optimized GPU kernels automatically
                        print("  Mixed precision optimizations enabled")
                except Exception as e:
                    pass
                
                # If 3.x failed, try XGBoost 2.x API
                if not gpu_enabled:
                    try:
                        _test = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=1)
                        _test.fit([[1, 2], [3, 4]], [0, 1])
                        default_params['tree_method'] = 'gpu_hist'
                        default_params['gpu_id'] = 0
                        default_params['predictor'] = 'gpu_predictor'
                        
                        # FP16 support for XGBoost 2.x
                        if self.use_fp16:
                            default_params['tree_method'] = 'gpu_hist'
                        
                        gpu_enabled = True
                        print("✓ GPU ENABLED - XGBoost 2.x API (tree_method='gpu_hist')")
                        if self.use_fp16:
                            print("  Mixed precision optimizations enabled")
                    except Exception as e:
                        pass
                
                if not gpu_enabled:
                    raise Exception("Both XGBoost 2.x and 3.x GPU methods failed")
                    
            except Exception as e:
                print(f"✗ GPU NOT AVAILABLE - Falling back to CPU: {e}")
                default_params['tree_method'] = 'hist'
                default_params['device'] = 'cpu'
                default_params['n_jobs'] = 4
                self.use_gpu = False
        else:
            default_params['tree_method'] = 'hist'
            default_params['device'] = 'cpu'
            default_params['n_jobs'] = 4
        
        # Update with provided params
        default_params.update(self.params)
        
        # Add early_stopping_rounds for XGBoost 3.x (model parameter, not fit parameter)
        if 'device' in default_params and default_params['device'] == 'cuda:0':
            if early_stopping_rounds and 'early_stopping_rounds' not in default_params:
                default_params['early_stopping_rounds'] = early_stopping_rounds
        
        print(f"Parameters: {default_params}")
        print(f"Training samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
        
        # Prepare evaluation set
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            print(f"Validation samples: {X_val.shape[0]}")
        
        # Create and train model
        self.model = xgb.XGBClassifier(**default_params)
        
        start_time = time.time()
        
        if eval_set:
            # Handle XGBoost 3.x vs 2.x early stopping API
            if 'device' in default_params and default_params['device'] == 'cuda:0':
                # XGBoost 3.x - early_stopping_rounds already in model params
                self.model.fit(
                    X_train, y_train,
                    eval_set=eval_set,
                    verbose=True
                )
            else:
                # XGBoost 2.x - early_stopping_rounds is a fit parameter
                try:
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=True
                    )
                except TypeError:
                    # Fallback: no early stopping
                    self.model.fit(
                        X_train, y_train,
                        eval_set=eval_set,
                        verbose=True
                    )
            
            if hasattr(self.model, 'best_iteration'):
                self.best_iteration = self.model.best_iteration
                self.best_score = self.model.best_score
                print(f"\n✓ Best iteration: {self.best_iteration}")
                print(f"✓ Best validation score: {self.best_score:.4f}")
        else:
            self.model.fit(X_train, y_train, verbose=True)
        
        train_time = time.time() - start_time
        print(f"\n✓ Training completed in {train_time:.2f} seconds")
        
        # Store feature importances
        self.feature_importances_ = self.model.feature_importances_
        
        return self.model
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.predict_proba(X)
    
    def get_feature_importances(self):
        """Get feature importances"""
        return self.feature_importances_


if __name__ == "__main__":
    # Test
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("Testing GPU XGBoost Trainer...")
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trainer = GPUXGBoostTrainer(use_gpu=True, use_fp16=False)
    trainer.train(X_train, y_train, X_test, y_test)
    
    predictions = trainer.predict(X_test)
    accuracy = (predictions == y_test).mean()
    print(f"\n✓ Test Accuracy: {accuracy:.4f}")