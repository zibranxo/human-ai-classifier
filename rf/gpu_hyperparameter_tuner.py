"""
GPU-Optimized Hyperparameter Tuning with Optuna
Parallel trials for maximum GPU utilization
Optimized for RTX 4050 + 10K dataset
"""

import numpy as np
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


class GPUHyperparameterTuner:
    """
    Optimized hyperparameter tuner for small datasets (10K)
    - Reduced trials (20-30 instead of 100)
    - Parallel execution
    - 3-fold CV instead of 5-fold
    - Early pruning of bad trials
    """
    
    def __init__(self, n_jobs=4, n_trials=25, cv_folds=3):
        """
        Initialize tuner
        
        Args:
            n_jobs: Parallel workers (4 recommended for RTX 4050)
            n_trials: Number of trials (25 optimal for 10K dataset)
            cv_folds: CV folds (3 for speed with 10K dataset)
        """
        self.n_jobs = n_jobs
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.best_rf_params = None
        self.best_xgb_params = None
        
        print(f"🎯 Hyperparameter Tuner initialized:")
        print(f"   Parallel workers: {n_jobs}")
        print(f"   Trials per model: {n_trials}")
        print(f"   CV folds: {cv_folds}")
    
    def tune_xgboost(self, X, y, use_gpu=True):
        """
        Tune XGBoost hyperparameters
        Optimized search space for 10K dataset
        """
        print("\n" + "="*70)
        print("  TUNING XGBOOST HYPERPARAMETERS (GPU)")
        print("="*70)
        print(f"Running {self.n_trials} trials with {self.cv_folds}-fold CV")
        print(f"Parallel workers: {self.n_jobs}")
        
        # Detect GPU capability once
        xgb_device_params = {}
        if use_gpu:
            try:
                # Try XGBoost 3.x API
                try:
                    _test = xgb.XGBClassifier(device='cuda:0', n_estimators=1)
                    _test.fit([[1, 2], [3, 4]], [0, 1])
                    xgb_device_params = {'device': 'cuda:0'}
                    print("✓ GPU enabled for tuning (XGBoost 3.x)")
                except:
                    # Try XGBoost 2.x API
                    _test = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0, n_estimators=1)
                    _test.fit([[1, 2], [3, 4]], [0, 1])
                    xgb_device_params = {
                        'tree_method': 'gpu_hist',
                        'gpu_id': 0,
                        'predictor': 'gpu_predictor'
                    }
                    print("✓ GPU enabled for tuning (XGBoost 2.x)")
            except Exception as e:
                print(f"✗ GPU not available, using CPU: {e}")
                xgb_device_params = {'tree_method': 'hist', 'device': 'cpu', 'n_jobs': 4}
        else:
            xgb_device_params = {'tree_method': 'hist', 'device': 'cpu', 'n_jobs': 4}
        
        def objective(trial):
            # Optimized search space for 10K dataset
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 4, 10),  # Reduced range
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
                'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 0.3),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 0.5),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0),
                'random_state': 42,
            }
            
            # Add GPU params
            params.update(xgb_device_params)
            
            # Add early_stopping_rounds to model params for XGBoost 3.x
            if 'device' in params and params['device'] == 'cuda:0':
                params['early_stopping_rounds'] = 20  # XGBoost 3.x: model parameter
            
            # Cross-validation with pruning
            skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            scores = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBClassifier(**params)
                
                # Fit model (API handled by params)
                if 'device' in params and params['device'] == 'cuda:0':
                    # XGBoost 3.x - early_stopping_rounds already in params
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                else:
                    # XGBoost 2.x - early_stopping_rounds is a fit parameter
                    try:
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=20,
                            verbose=False
                        )
                    except TypeError:
                        # Fallback: no early stopping
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            verbose=False
                        )
                
                y_pred = model.predict(X_val)
                score = matthews_corrcoef(y_val, y_pred)
                scores.append(score)
                
                # Prune trial if performing poorly
                trial.report(score, fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        # Create study with pruner for early stopping
        study = optuna.create_study(
            direction='maximize',
            study_name='xgb_tuning',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        )
        
        # Run optimization with parallel trials
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=self.n_jobs,  # Parallel trials
            show_progress_bar=True
        )
        
        self.best_xgb_params = study.best_params
        
        print(f"\n✓ Best XGBoost Parameters:")
        for param, value in self.best_xgb_params.items():
            print(f"  {param}: {value}")
        print(f"\n✓ Best CV MCC Score: {study.best_value:.4f}")
        print(f"✓ Completed trials: {len(study.trials)}")
        print(f"✓ Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        
        return self.best_xgb_params
    
    def tune_both(self, X, y, use_gpu=True):
        """
        Tune XGBoost only (PyTorch RF uses fixed params for 10K dataset)
        
        For 10K dataset, PyTorch RF benefits more from fixed optimal params
        than from tuning, so we skip RF tuning to save time.
        """
        # Fixed optimal params for PyTorch RF with 10K dataset
        self.best_rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5
        }
        
        print(f"\n📌 Using fixed optimal parameters for PyTorch Random Forest:")
        print(f"  (Tuning skipped - fixed params optimal for 10K dataset)")
        for param, value in self.best_rf_params.items():
            print(f"  {param}: {value}")
        
        # Tune XGBoost
        xgb_params = self.tune_xgboost(X, y, use_gpu=use_gpu)
        
        return {
            'pytorch_random_forest': self.best_rf_params,
            'xgboost': xgb_params
        }


if __name__ == "__main__":
    # Test
    from sklearn.datasets import make_classification
    
    print("Testing GPU Hyperparameter Tuner...")
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    
    tuner = GPUHyperparameterTuner(n_jobs=2, n_trials=10, cv_folds=3)
    params = tuner.tune_both(X, y, use_gpu=True)
    
    print(f"\n✓ Tuning complete!")
    print(f"  XGBoost params: {params['xgboost']}")