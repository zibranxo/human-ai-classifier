"""
Comprehensive Metrics for Model Evaluation
Includes accuracy, precision, recall, F1, MCC, AUC-ROC, PR-AUC, and cross-validation
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    def __init__(self):
        pass
    
    def calculate_all_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate all classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Probability-based metrics
        if y_pred_proba is not None:
            try:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['auc_pr'] = average_precision_score(y_true, y_pred_proba)
            except:
                metrics['auc_roc'] = 0.0
                metrics['auc_pr'] = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # Detailed confusion matrix values
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
            # Specificity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # False positive rate
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # False negative rate
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def print_metrics(self, metrics, title="Classification Metrics"):
        """Pretty print metrics"""
        print("\n" + "="*70)
        print(f"  {title}")
        print("="*70)
        
        # Main metrics
        print(f"  Accuracy    : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision   : {metrics['precision']:.4f}")
        print(f"  Recall      : {metrics['recall']:.4f}")
        print(f"  F1 Score    : {metrics['f1']:.4f}")
        print(f"  MCC         : {metrics['mcc']:.4f}")
        
        if 'auc_roc' in metrics:
            print(f"  AUC-ROC     : {metrics['auc_roc']:.4f}")
        if 'auc_pr' in metrics:
            print(f"  AUC-PR      : {metrics['auc_pr']:.4f}")
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            cm = metrics['confusion_matrix']
            print(f"\n  Confusion Matrix:")
            print(f"                Pred Human   Pred AI")
            print(f"  True Human :      {cm[0][0]:5d}      {cm[0][1]:5d}")
            print(f"  True AI    :      {cm[1][0]:5d}      {cm[1][1]:5d}")
        
        # Additional metrics
        if 'specificity' in metrics:
            print(f"\n  Specificity : {metrics['specificity']:.4f}")
            print(f"  FPR         : {metrics['fpr']:.4f}")
            print(f"  FNR         : {metrics['fnr']:.4f}")
        
        print("="*70 + "\n")
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Human', 'AI'],
                   yticklabels=['Human', 'AI'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pr_curve(self, y_true, y_pred_proba, save_path=None):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR Curve (AP = {avg_precision:.4f})', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def cross_validate(self, model, X, y, cv=5, metric='f1'):
        """Perform k-fold cross-validation"""
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'mcc': [],
            'auc_roc': []
        }
        
        fold = 1
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone and train model
            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_train, y_train)
            
            # Predict
            y_pred = model_clone.predict(X_val)
            y_pred_proba = model_clone.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            scores['accuracy'].append(accuracy_score(y_val, y_pred))
            scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
            scores['mcc'].append(matthews_corrcoef(y_val, y_pred))
            scores['auc_roc'].append(roc_auc_score(y_val, y_pred_proba))
            
            print(f"  Fold {fold}: F1={scores['f1'][-1]:.4f}, MCC={scores['mcc'][-1]:.4f}, AUC={scores['auc_roc'][-1]:.4f}")
            fold += 1
        
        # Calculate mean and std
        cv_results = {}
        for metric_name, values in scores.items():
            cv_results[f'{metric_name}_mean'] = np.mean(values)
            cv_results[f'{metric_name}_std'] = np.std(values)
        
        print(f"\nCross-Validation Results:")
        print(f"  Accuracy : {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
        print(f"  F1 Score : {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
        print(f"  MCC      : {cv_results['mcc_mean']:.4f} ± {cv_results['mcc_std']:.4f}")
        print(f"  AUC-ROC  : {cv_results['auc_roc_mean']:.4f} ± {cv_results['auc_roc_std']:.4f}")
        
        return cv_results


if __name__ == "__main__":
    # Test metrics calculation
    calc = MetricsCalculator()
    
    # Dummy data
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0])
    y_pred_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.85, 0.4, 0.15, 0.95, 0.25])
    
    metrics = calc.calculate_all_metrics(y_true, y_pred, y_pred_proba)
    calc.print_metrics(metrics)
