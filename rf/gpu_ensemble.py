"""
GPU-Optimized Ensemble Classifier
Combines PyTorch Random Forest (GPU) + XGBoost (GPU)
"""

import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, f1_score
import pickle
import warnings
warnings.filterwarnings('ignore')


class GPUEnsembleClassifier:
    """
    Ensemble of PyTorch Random Forest (GPU) and XGBoost (GPU)
    Optimized for RTX 4050
    """
    
    def __init__(self, rf_model=None, xgb_model=None, rf_weight=0.5, xgb_weight=0.5):
        """
        Initialize ensemble
        
        Args:
            rf_model: Trained PyTorch Random Forest model
            xgb_model: Trained XGBoost model
            rf_weight: Weight for RF predictions (0-1)
            xgb_weight: Weight for XGB predictions (0-1)
        """
        self.rf_model = rf_model
        self.xgb_model = xgb_model
        self.rf_weight = rf_weight
        self.xgb_weight = xgb_weight
        self.weights_optimized = False
    
    def optimize_weights(self, X_val, y_val):
        """
        Optimize ensemble weights on validation set
        Tests different weight combinations and selects best
        """
        print("\n" + "="*70)
        print("  OPTIMIZING ENSEMBLE WEIGHTS")
        print("="*70)
        
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("Both models must be trained before optimizing weights!")
        
        # Get predictions from both models
        rf_proba = self.rf_model.predict_proba(X_val)[:, 1]
        xgb_proba = self.xgb_model.predict_proba(X_val)[:, 1]
        
        # Test different weight combinations
        best_mcc = -1
        best_weights = (0.5, 0.5)
        
        weight_combinations = [
            (0.2, 0.8), (0.3, 0.7), (0.4, 0.6), (0.5, 0.5), 
            (0.6, 0.4), (0.7, 0.3), (0.8, 0.2)
        ]
        
        print("\nTesting weight combinations:")
        for rf_w, xgb_w in weight_combinations:
            # Weighted average of probabilities
            ensemble_proba = rf_w * rf_proba + xgb_w * xgb_proba
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)
            
            # Calculate MCC
            mcc = matthews_corrcoef(y_val, ensemble_pred)
            f1 = f1_score(y_val, ensemble_pred)
            
            print(f"  RF={rf_w:.1f}, XGB={xgb_w:.1f} -> MCC={mcc:.4f}, F1={f1:.4f}")
            
            if mcc > best_mcc:
                best_mcc = mcc
                best_weights = (rf_w, xgb_w)
        
        self.rf_weight, self.xgb_weight = best_weights
        self.weights_optimized = True
        
        print(f"\n✓ Optimal weights: RF={self.rf_weight:.2f}, XGB={self.xgb_weight:.2f}")
        print(f"✓ Best Validation MCC: {best_mcc:.4f}")
        
        return best_weights
    
    def predict_proba(self, X):
        """
        Predict class probabilities using weighted ensemble
        
        Returns:
            Array of shape (n_samples, 2) with probabilities for each class
        """
        if self.rf_model is None or self.xgb_model is None:
            raise ValueError("Both models must be trained!")
        
        # Get probabilities from both models
        rf_proba = self.rf_model.predict_proba(X)
        xgb_proba = self.xgb_model.predict_proba(X)
        
        # Weighted average
        ensemble_proba = (
            self.rf_weight * rf_proba +
            self.xgb_weight * xgb_proba
        )
        
        return ensemble_proba
    
    def predict(self, X):
        """
        Predict class labels using weighted ensemble
        
        Returns:
            Array of predicted class labels (0 or 1)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def predict_with_confidence(self, X):
        """
        Predict with confidence scores
        
        Returns:
            predictions: Array of predicted labels
            confidences: Array of confidence scores (0-1)
        """
        proba = self.predict_proba(X)
        predictions = (proba[:, 1] >= 0.5).astype(int)
        
        # Confidence is the absolute distance from 0.5
        confidences = np.abs(proba[:, 1] - 0.5) * 2
        
        return predictions, confidences
    
    def get_individual_predictions(self, X):
        """
        Get predictions from both models separately
        
        Returns:
            rf_pred: Random Forest predictions
            xgb_pred: XGBoost predictions
            ensemble_pred: Ensemble predictions
        """
        rf_pred = self.rf_model.predict(X)
        xgb_pred = self.xgb_model.predict(X)
        ensemble_pred = self.predict(X)
        
        return rf_pred, xgb_pred, ensemble_pred
    
    def save(self, filepath):
        """Save ensemble model"""
        # Convert PyTorch model to CPU for saving
        if hasattr(self.rf_model, 'device'):
            device_backup = self.rf_model.device
            # Move trees to CPU for saving
            for tree in self.rf_model.trees:
                if hasattr(tree, 'device'):
                    tree.device = torch.device('cpu')
        
        model_data = {
            'rf_model': self.rf_model,
            'xgb_model': self.xgb_model,
            'rf_weight': self.rf_weight,
            'xgb_weight': self.xgb_weight,
            'weights_optimized': self.weights_optimized
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Ensemble model saved to {filepath}")
        
        # Restore device
        if hasattr(self.rf_model, 'device'):
            for tree in self.rf_model.trees:
                if hasattr(tree, 'device'):
                    tree.device = device_backup
    
    @staticmethod
    def load(filepath):
        """Load ensemble model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        ensemble = GPUEnsembleClassifier(
            rf_model=model_data['rf_model'],
            xgb_model=model_data['xgb_model'],
            rf_weight=model_data['rf_weight'],
            xgb_weight=model_data['xgb_weight']
        )
        ensemble.weights_optimized = model_data['weights_optimized']
        
        print(f"\n✓ Ensemble model loaded from {filepath}")
        print(f"  Weights: RF={ensemble.rf_weight:.2f}, XGB={ensemble.xgb_weight:.2f}")
        
        # Move RF model back to GPU if available
        if torch.cuda.is_available() and hasattr(ensemble.rf_model, 'device'):
            ensemble.rf_model.device = torch.device('cuda')
            for tree in ensemble.rf_model.trees:
                if hasattr(tree, 'device'):
                    tree.device = torch.device('cuda')
        
        return ensemble


if __name__ == "__main__":
    # Test ensemble
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from pytorch_random_forest import PyTorchRandomForest
    import xgboost as xgb
    
    print("Testing GPU Ensemble...")
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    rf_model = PyTorchRandomForest(n_estimators=50, max_depth=10, device='cuda', verbose=0)
    rf_model.fit(X_train, y_train)
    
    xgb_model = xgb.XGBClassifier(n_estimators=50, device='cuda:0', random_state=42)
    xgb_model.fit(X_train, y_train)
    
    # Create ensemble
    ensemble = GPUEnsembleClassifier(rf_model, xgb_model)
    ensemble.optimize_weights(X_test, y_test)
    
    # Test
    predictions = ensemble.predict(X_test)
    mcc = matthews_corrcoef(y_test, predictions)
    print(f"\n✓ Ensemble Test MCC: {mcc:.4f}")
