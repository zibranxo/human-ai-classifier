"""
PyTorch GPU-Accelerated Random Forest Classifier
Optimized for NVIDIA RTX 4050
Uses PyTorch tensors for GPU acceleration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
import time


class DecisionTree:
    """Single decision tree with GPU acceleration"""
    
    def __init__(self, max_depth=10, min_samples_split=2, device='cuda'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.device = device
        self.tree = None
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """Build decision tree"""
        self.tree = self._build_tree(X, y, depth=0)
        return self
    
    def _build_tree(self, X: torch.Tensor, y: torch.Tensor, depth: int) -> dict:
        """Recursively build tree"""
        n_samples, n_features = X.shape
        n_classes = len(torch.unique(y))
        
        # Stopping criteria
        if depth >= self.max_depth or n_samples < self.min_samples_split or n_classes == 1:
            leaf_value = torch.mode(y)[0].item()
            return {'value': leaf_value}
        
        # Find best split
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            leaf_value = torch.mode(y)[0].item()
            return {'value': leaf_value}
        
        # Split data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Build subtrees
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def _best_split(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[Optional[int], Optional[float]]:
        """Find best feature and threshold for split using GPU"""
        n_samples, n_features = X.shape
        
        if n_samples <= 1:
            return None, None
        
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        # Calculate parent gini
        parent_gini = self._gini_impurity(y)
        
        # Try subset of features (for speed)
        n_features_to_try = max(1, int(np.sqrt(n_features)))
        features_to_try = torch.randperm(n_features, device=self.device)[:n_features_to_try]
        
        for feature_idx in features_to_try:
            feature_values = X[:, feature_idx]
            
            # Get unique thresholds (sample for speed)
            unique_values = torch.unique(feature_values)
            if len(unique_values) > 10:
                # Sample thresholds
                indices = torch.randperm(len(unique_values), device=self.device)[:10]
                thresholds = unique_values[indices]
            else:
                thresholds = unique_values
            
            for threshold in thresholds:
                # Split
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if left_mask.sum() < self.min_samples_split or right_mask.sum() < self.min_samples_split:
                    continue
                
                # Calculate weighted gini
                n_left = left_mask.sum().item()
                n_right = right_mask.sum().item()
                
                gini_left = self._gini_impurity(y[left_mask])
                gini_right = self._gini_impurity(y[right_mask])
                
                weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples
                
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_idx.item()
                    best_threshold = threshold.item()
        
        return best_feature, best_threshold
    
    def _gini_impurity(self, y: torch.Tensor) -> float:
        """Calculate Gini impurity"""
        if len(y) == 0:
            return 0.0
        
        _, counts = torch.unique(y, return_counts=True)
        probabilities = counts.float() / len(y)
        gini = 1.0 - torch.sum(probabilities ** 2).item()
        
        return gini
    
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict class labels"""
        predictions = []
        
        for i in range(X.shape[0]):
            predictions.append(self._traverse_tree(X[i], self.tree))
        
        return torch.tensor(predictions, device=self.device)
    
    def _traverse_tree(self, x: torch.Tensor, node: dict) -> int:
        """Traverse tree to make prediction"""
        if 'value' in node:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])


class PyTorchRandomForest:
    """
    GPU-Accelerated Random Forest using PyTorch
    Optimized for RTX 4050
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        max_features: str = 'sqrt',
        n_jobs: int = -1,
        device: str = 'cuda',
        verbose: int = 1
    ):
        """
        Initialize GPU Random Forest
        
        Args:
            n_estimators: Number of trees
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split
            max_features: Number of features to consider ('sqrt', 'log2', or int)
            n_jobs: Not used (GPU handles parallelization)
            device: 'cuda' or 'cpu'
            verbose: Verbosity level
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.verbose = verbose
        self.trees = []
        self.feature_importances_ = None
        
        if self.verbose:
            if self.device.type == 'cuda':
                print(f"🌲 PyTorch Random Forest initialized on GPU: {torch.cuda.get_device_name(0)}")
            else:
                print(f"🌲 PyTorch Random Forest initialized on CPU")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'PyTorchRandomForest':
        """
        Train random forest on GPU
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (n_samples,)
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"  TRAINING PYTORCH RANDOM FOREST (GPU)")
            print(f"{'='*70}")
            print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
            print(f"Trees: {self.n_estimators}, Max Depth: {self.max_depth}")
        
        start_time = time.time()
        
        # Convert to PyTorch tensors on GPU
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
        
        n_samples, n_features = X.shape
        
        # Train trees
        self.trees = []
        
        for i in range(self.n_estimators):
            if self.verbose and (i + 1) % 20 == 0:
                print(f"Training tree {i+1}/{self.n_estimators}...", end='\r')
            
            # Bootstrap sample (with replacement)
            indices = torch.randint(0, n_samples, (n_samples,), device=self.device)
            X_bootstrap = X_tensor[indices]
            y_bootstrap = y_tensor[indices]
            
            # Train tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                device=self.device
            )
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)
        
        train_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n✓ Training completed in {train_time:.2f} seconds")
            print(f"  Average time per tree: {train_time/self.n_estimators:.3f} seconds")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Get predictions from all trees
        all_predictions = []
        
        for tree in self.trees:
            predictions = tree.predict(X_tensor)
            all_predictions.append(predictions)
        
        # Stack predictions
        all_predictions = torch.stack(all_predictions, dim=0)  # (n_trees, n_samples)
        
        # Majority vote
        final_predictions = torch.mode(all_predictions, dim=0)[0]
        
        return final_predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Features (n_samples, n_features)
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        # Convert to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        # Get predictions from all trees
        all_predictions = []
        
        for tree in self.trees:
            predictions = tree.predict(X_tensor)
            all_predictions.append(predictions)
        
        # Stack predictions
        all_predictions = torch.stack(all_predictions, dim=0)  # (n_trees, n_samples)
        
        n_samples = X.shape[0]
        n_classes = 2  # Binary classification
        
        # Calculate probabilities
        probabilities = torch.zeros((n_samples, n_classes), device=self.device)
        
        for i in range(n_samples):
            class_counts = torch.bincount(all_predictions[:, i], minlength=n_classes)
            probabilities[i] = class_counts.float() / self.n_estimators
        
        return probabilities.cpu().numpy()
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy"""
        predictions = self.predict(X)
        accuracy = (predictions == y).mean()
        return accuracy
    
    def get_params(self, deep: bool = True) -> dict:
        """Get parameters"""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'max_features': self.max_features,
            'device': str(self.device),
            'verbose': self.verbose
        }


if __name__ == "__main__":
    # Test
    print("Testing PyTorch Random Forest...")
    
    # Generate dummy data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    rf = PyTorchRandomForest(n_estimators=50, max_depth=10, device='cuda')
    rf.fit(X_train, y_train)
    
    # Test
    accuracy = rf.score(X_test, y_test)
    print(f"\n✓ Test Accuracy: {accuracy:.4f}")
