import torch
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import os

# Set random seeds for reproducibility (CPU and GPU)
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration - GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class ANN:
    """
    Custom ANN with 3 hidden layers
    Architecture: 21 -> 64 -> 32 -> 16 -> 1
    Activation: ReLU (hidden), Sigmoid (output)
    Backpropagation implemented manually
    """

    def __init__(self, input_size=21, hidden1=64, hidden2=32, hidden3=16, output_size=1):
        # Initialize weights using He initialization (for ReLU)
        self.W1 = self._he_init(input_size, hidden1)
        self.b1 = torch.zeros(hidden1)

        self.W2 = self._he_init(hidden1, hidden2)
        self.b2 = torch.zeros(hidden2)

        self.W3 = self._he_init(hidden2, hidden3)
        self.b3 = torch.zeros(hidden3)

        self.W4 = self._he_init(hidden3, output_size)
        self.b4 = torch.zeros(output_size)

        self.training = True  # Track mode

        self.layers = [
            (self.W1, self.b1),
            (self.W2, self.b2),
            (self.W3, self.b3),
            (self.W4, self.b4)
        ]

    def train(self):
        """Set model to training mode"""
        self.training = True
        return self

    def eval(self):
        """Set model to evaluation mode"""
        self.training = False
        return self

    def to(self, device):
        """Move all weights and biases to specified device"""
        self.W1 = self.W1.to(device)
        self.b1 = self.b1.to(device)
        self.W2 = self.W2.to(device)
        self.b2 = self.b2.to(device)
        self.W3 = self.W3.to(device)
        self.b3 = self.b3.to(device)
        self.W4 = self.W4.to(device)
        self.b4 = self.b4.to(device)
        self.layers = [(self.W1, self.b1), (self.W2, self.b2), (self.W3, self.b3), (self.W4, self.b4)]
        return self

    def _he_init(self, fan_in, fan_out):
        """He initialization for ReLU activation"""
        std = np.sqrt(2.0 / fan_in)
        return torch.randn(fan_in, fan_out) * std

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function: σ(x) = 1/(1 + e^(-x))"""
        return 1.0 / (1.0 + torch.exp(-torch.clamp(x, -500, 500)))

    @staticmethod
    def sigmoid_derivative(sigmoid_output):
        """Derivative of sigmoid: σ'(x) = σ(x) * (1 - σ(x))"""
        return sigmoid_output * (1.0 - sigmoid_output)

    @staticmethod
    def relu(x):
        """ReLU activation function: max(0, x)"""
        return torch.clamp(x, min=0)

    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU: 1 if x > 0, else 0"""
        return (x > 0).float()

    def forward(self, X):
        """Forward pass through the network"""
        # Input X: (batch_size, input_size)

        # Hidden Layer 1 (ReLU)
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        # Hidden Layer 2 (ReLU)
        self.z2 = torch.matmul(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)

        # Hidden Layer 3 (ReLU)
        self.z3 = torch.matmul(self.a2, self.W3) + self.b3
        self.a3 = self.relu(self.z3)

        # Output Layer (Sigmoid)
        self.z4 = torch.matmul(self.a3, self.W4) + self.b4
        self.a4 = self.sigmoid(self.z4)

        return self.a4

    def backward(self, X, y, learning_rate, lambda_reg=0.001):
        """
        Backpropagation algorithm with L2 regularization
        Weight update rule: w = w - η * (∂E/∂w + λ*w)

        Note: L2 regularization gradient is added here for weight decay.
        The loss function should NOT duplicate this L2 term.
        """
        batch_size = X.shape[0]

        # Output layer error: ∂E/∂a4 * ∂a4/∂z4
        # For BCE with sigmoid, simplifies to: a4 - y
        delta4 = (self.a4 - y).view(-1, 1)  # (batch_size, 1)
        grad_W4 = torch.matmul(self.a3.t(), delta4) / batch_size + lambda_reg * self.W4
        grad_b4 = delta4.mean(dim=0)

        # Hidden Layer 3 error (ReLU derivative)
        delta3 = torch.matmul(delta4, self.W4.t()) * self.relu_derivative(self.z3)
        grad_W3 = torch.matmul(self.a2.t(), delta3) / batch_size + lambda_reg * self.W3
        grad_b3 = delta3.mean(dim=0)

        # Hidden Layer 2 error (ReLU derivative)
        delta2 = torch.matmul(delta3, self.W3.t()) * self.relu_derivative(self.z2)
        grad_W2 = torch.matmul(self.a1.t(), delta2) / batch_size + lambda_reg * self.W2
        grad_b2 = delta2.mean(dim=0)

        # Hidden Layer 1 error (ReLU derivative)
        delta1 = torch.matmul(delta2, self.W2.t()) * self.relu_derivative(self.z1)
        grad_W1 = torch.matmul(X.t(), delta1) / batch_size + lambda_reg * self.W1
        grad_b1 = delta1.mean(dim=0)

        # Update weights: w = w - η * ∂E/∂w
        self.W4 = self.W4 - learning_rate * grad_W4
        self.b4 = self.b4 - learning_rate * grad_b4

        self.W3 = self.W3 - learning_rate * grad_W3
        self.b3 = self.b3 - learning_rate * grad_b3

        self.W2 = self.W2 - learning_rate * grad_W2
        self.b2 = self.b2 - learning_rate * grad_b2

        self.W1 = self.W1 - learning_rate * grad_W1
        self.b1 = self.b1 - learning_rate * grad_b1

    def predict(self, X):
        """Forward pass and return predictions (0 or 1)"""
        output = self.forward(X)
        return (output > 0.5).float()

    def save_weights(self, filepath):
        """Save model weights"""
        torch.save({
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3,
            'W4': self.W4, 'b4': self.b4,
        }, filepath)

    def load_weights(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.W1 = checkpoint['W1']
        self.b1 = checkpoint['b1']
        self.W2 = checkpoint['W2']
        self.b2 = checkpoint['b2']
        self.W3 = checkpoint['W3']
        self.b3 = checkpoint['b3']
        self.W4 = checkpoint['W4']
        self.b4 = checkpoint['b4']


def load_and_preprocess_data(filepath):
    """Load CSV and preprocess data"""
    print("Loading dataset...")
    df = pd.read_csv(filepath)

    # Assuming last column is label, rest are features
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {df.columns[:-1].tolist()}")
    print(f"Labels distribution: {np.bincount(y.astype(int))}")

    # Normalize features to [0, 1]
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler


def compute_loss(predictions, targets, model, lambda_reg=0.001):
    """
    Binary Cross-Entropy Loss (without L2 - L2 is handled in backward pass)

    Note: L2 regularization is applied via weight decay in backward(),
    so we don't duplicate it here to avoid double regularization.
    """
    eps = 1e-7  # Prevent log(0)
    predictions = torch.clamp(predictions, eps, 1 - eps)
    bce_loss = -(targets * torch.log(predictions) + (1 - targets) * torch.log(1 - predictions))
    return bce_loss.mean()


def compute_class_weights(y):
    """Compute class weights for imbalanced data (handles non-contiguous labels)"""
    classes = np.unique(y)
    total = len(y)
    weights = {}
    for c in classes:
        count = np.sum(y == c)
        weights[int(c)] = total / (len(classes) * count)
    return weights


def weighted_bce_loss(predictions, targets, class_weights):
    """Weighted BCE loss for class imbalance (device-aware)"""
    eps = 1e-7
    predictions = torch.clamp(predictions, eps, 1 - eps)

    # Apply class weights (ensure tensors are on same device as targets)
    weights = torch.where(
        targets == 1,
        torch.tensor(class_weights[1], device=targets.device, dtype=targets.dtype),
        torch.tensor(class_weights[0], device=targets.device, dtype=targets.dtype)
    )

    loss = -weights * (targets * torch.log(predictions) +
                       (1 - targets) * torch.log(1 - predictions))
    return loss.mean()


def train_model(model, X_train, y_train, X_val, y_val,
                learning_rate=0.005, batch_size=64, num_epochs=500, patience=20,
                lambda_reg=0.001, class_weights=None):
    """Train the ANN model with mini-batch gradient descent and early stopping"""
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_val, y_val = X_val.to(device), y_val.to(device)

    num_samples = X_train.shape[0]
    best_val_loss = float('inf')
    patience_counter = 0
    best_weights = None

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    model.train()

    for epoch in range(num_epochs):
        # Shuffle training data
        indices = torch.randperm(num_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        epoch_loss = 0.0
        num_batches = 0

        # Mini-batch training
        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Forward pass
            predictions = model.forward(X_batch)

            # Compute loss (L2 is handled in backward pass via weight decay)
            if class_weights is not None:
                loss = weighted_bce_loss(predictions, y_batch, class_weights)
            else:
                loss = compute_loss(predictions, y_batch, model)

            # Backward pass
            model.backward(X_batch, y_batch, learning_rate, lambda_reg)

            epoch_loss += loss.item()
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches

        # Compute training accuracy (batched for memory safety)
        with torch.inference_mode():
            correct = 0
            total = 0
            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                preds = model.predict(X_batch)
                correct += (preds.squeeze() == y_batch.squeeze()).sum().item()
                total += y_batch.size(0)
            train_acc = correct / total

        # Validation (batched for memory safety)
        with torch.inference_mode():
            # Batched forward pass for validation loss and accuracy
            correct = 0
            total = 0
            val_loss_sum = 0.0
            num_val_batches = 0

            for i in range(0, X_val.shape[0], batch_size):
                X_batch = X_val[i:i+batch_size]
                y_batch = y_val[i:i+batch_size]

                output = model.forward(X_batch)
                preds = (output > 0.5).float()

                # Accumulate loss
                if class_weights is not None:
                    val_loss_sum += weighted_bce_loss(output, y_batch, class_weights).item()
                else:
                    val_loss_sum += compute_loss(output, y_batch, model).item()
                num_val_batches += 1

                # Accumulate accuracy
                correct += (preds.squeeze() == y_batch.squeeze()).sum().item()
                total += y_batch.size(0)

            val_loss = val_loss_sum / num_val_batches
            val_acc = correct / total

        # Clear GPU cache periodically to prevent memory buildup
        if torch.cuda.is_available() and (epoch + 1) % 50 == 0:
            torch.cuda.empty_cache()

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save a copy of weights
            best_weights = {
                'W1': model.W1.clone(), 'b1': model.b1.clone(),
                'W2': model.W2.clone(), 'b2': model.b2.clone(),
                'W3': model.W3.clone(), 'b3': model.b3.clone(),
                'W4': model.W4.clone(), 'b4': model.b4.clone(),
            }
        else:
            patience_counter += 1

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Restore best weights
    if best_weights is not None:
        model.W1 = best_weights['W1']
        model.b1 = best_weights['b1']
        model.W2 = best_weights['W2']
        model.b2 = best_weights['b2']
        model.W3 = best_weights['W3']
        model.b3 = best_weights['b3']
        model.W4 = best_weights['W4']
        model.b4 = best_weights['b4']

    model.eval()
    return model, train_losses, val_losses, train_accs, val_accs


def _batched_forward(model, X, batch_size=64):
    """Forward pass with batching to prevent GPU memory spikes on large datasets"""
    if X.shape[0] <= batch_size:
        return model.forward(X)

    outputs = []
    for i in range(0, X.shape[0], batch_size):
        batch = X[i:i+batch_size]
        outputs.append(model.forward(batch))
    return torch.cat(outputs, dim=0)


def evaluate_model(model, X_test, y_test, batch_size=64):
    """Evaluate model on test set with batched inference"""
    model.eval()
    X_test, y_test = X_test.to(device), y_test.to(device)

    with torch.no_grad():
        # Get probabilities (for ROC/AUC analysis)
        probabilities = _batched_forward(model, X_test, batch_size)
        predictions = (probabilities > 0.5).float()
        accuracy = (predictions.squeeze() == y_test).float().mean().item()

        # Get numpy arrays for sklearn metrics
        y_pred = predictions.squeeze().cpu().numpy()
        y_prob = probabilities.squeeze().cpu().numpy()  # Probabilities for ROC
        y_true = y_test.cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred,
        target_names=['Human', 'AI'],
        zero_division=0
    )

    return accuracy, cm, report, y_pred, y_prob


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    """Plot comprehensive training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss plot
    ax1 = axes[0, 0]
    ax1.plot(train_losses, label='Train Loss', color='#1f77b4', linewidth=1.5)
    ax1.plot(val_losses, label='Val Loss', color='#ff7f0e', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    min_val_loss_idx = val_losses.index(min(val_losses)) if val_losses else 0
    ax1.axvline(x=min_val_loss_idx, color='red', linestyle='--', alpha=0.5, label=f'Best epoch ({min_val_loss_idx+1})')

    # 2. Accuracy plot
    ax2 = axes[0, 1]
    ax2.plot(train_accs, label='Train Accuracy', color='#2ca02c', linewidth=1.5)
    ax2.plot(val_accs, label='Val Accuracy', color='#d62728', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    max_val_acc_idx = val_accs.index(max(val_accs)) if val_accs else 0
    ax2.axvline(x=max_val_acc_idx, color='red', linestyle='--', alpha=0.5)

    # 3. Loss gap (overfitting indicator)
    ax3 = axes[1, 0]
    loss_gap = [t - v for t, v in zip(train_losses, val_losses)]
    ax3.plot(loss_gap, label='Loss Gap (Train - Val)', color='#9467bd', linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.fill_between(range(len(loss_gap)), loss_gap, alpha=0.3, color='#9467bd')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss Gap')
    ax3.set_title('Overfitting Indicator (Loss Gap)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Learning rate effect (loss improvement per epoch)
    ax4 = axes[1, 1]
    loss_improvement = [train_losses[i] - train_losses[i+1] for i in range(len(train_losses)-1)]
    ax4.bar(range(len(loss_improvement)), loss_improvement, color='#8c564b', alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Loss Reduction')
    ax4.set_title('Per-Epoch Loss Improvement')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150)
    print("Training plots saved as 'training_history.png'")
    plt.show()


def plot_evaluation_results(y_true, y_pred, cm, accuracy):
    """Plot evaluation results including confusion matrix and prediction analysis"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Confusion Matrix heatmap
    ax1 = axes[0]
    im = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.figure.colorbar(im, ax=ax1)
    ax1.set(xticks=[0, 1], yticks=[0, 1],
            xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'],
            ylabel='True label', xlabel='Predicted label',
            title=f'Confusion Matrix (Acc: {accuracy:.4f})')
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)

    # 2. Prediction distribution
    ax2 = axes[1]
    ax2.hist(y_pred[y_true == 0], bins=50, alpha=0.6, label='Human', color='#1f77b4')
    ax2.hist(y_pred[y_true == 1], bins=50, alpha=0.6, label='AI', color='#ff7f0e')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Prediction Distribution by Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Class-wise accuracy bar chart
    ax3 = axes[2]
    human_acc = ((y_pred[y_true == 0] < 0.5).sum() / len(y_true[y_true == 0])) if (y_true == 0).any() else 0
    ai_acc = ((y_pred[y_true == 1] >= 0.5).sum() / len(y_true[y_true == 1])) if (y_true == 1).any() else 0
    classes = ['Human', 'AI']
    accuracies = [human_acc, ai_acc]
    colors = ['#1f77b4', '#ff7f0e']
    bars = ax3.bar(classes, accuracies, color=colors, alpha=0.8)
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Per-Class Accuracy')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accuracies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150)
    print("Evaluation plots saved as 'evaluation_results.png'")
    plt.show()


def main():
    # Hyperparameters
    INPUT_SIZE = 21
    HIDDEN1 = 64      # Increased capacity
    HIDDEN2 = 32
    HIDDEN3 = 16
    LEARNING_RATE = 0.005   # Lower for stability
    BATCH_SIZE = 64
    NUM_EPOCHS = 500
    PATIENCE = 20
    TEST_SIZE = 0.2
    LAMBDA_REG = 0.001     # L2 regularization strength

    # Load data - UPDATE THIS PATH TO YOUR DATASET
    DATA_PATH = 'dataset.csv'

    X, y, scaler = load_and_preprocess_data(DATA_PATH)

    # Compute class weights for imbalanced data
    class_counts = np.bincount(y.astype(int))
    if len(class_counts) == 2 and abs(class_counts[0] - class_counts[1]) > 0.1 * len(y):
        class_weights = compute_class_weights(y)
        print(f"Class imbalance detected. Using class weights: {class_weights}")
    else:
        class_weights = None

    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).view(-1, 1)

    # Split data: 80% train, 20% test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_tensor, y_tensor, test_size=TEST_SIZE, random_state=42
    )

    # Further split train into train and validation (80/20 of train = 64/16 of total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Initialize model
    model = ANN(INPUT_SIZE, HIDDEN1, HIDDEN2, HIDDEN3).to(device)

    # Print architecture
    print(f"\n{'='*50}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*50}")
    print(f"Input Layer:  {INPUT_SIZE} neurons")
    print(f"Hidden Layer 1: {HIDDEN1} neurons (ReLU)")
    print(f"Hidden Layer 2: {HIDDEN2} neurons (ReLU)")
    print(f"Hidden Layer 3: {HIDDEN3} neurons (ReLU)")
    print(f"Output Layer: 1 neuron (Sigmoid)")
    print(f"\nTotal parameters: {INPUT_SIZE*HIDDEN1 + HIDDEN1*HIDDEN2 + HIDDEN2*HIDDEN3 + HIDDEN3*1 + HIDDEN1 + HIDDEN2 + HIDDEN3 + 1}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"L2 regularization: {LAMBDA_REG}")
    print(f"{'='*50}")

    # Train
    print("\nStarting training...")
    model, train_losses, val_losses, train_accs, val_accs = train_model(
        model, X_train, y_train, X_val, y_val,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        patience=PATIENCE,
        lambda_reg=LAMBDA_REG,
        class_weights=class_weights
    )

    # Evaluate on test set
    print("\nEvaluating on test set...")
    accuracy, cm, report, predictions, probabilities = evaluate_model(model, X_test, y_test)

    print(f"\n{'='*50}")
    print("TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(report)

    # Plot training history
    plot_training_history(train_losses, val_losses, train_accs, val_accs)

    # Plot evaluation results
    y_true = y_test.cpu().numpy()
    plot_evaluation_results(y_true, predictions, cm, accuracy)

    # Print probabilities for ROC/AUC analysis
    print(f"\nProbabilities shape: {probabilities.shape}")
    print("Sample probabilities (first 10):", probabilities[:10].flatten())

    # Save model and scaler
    model.save_weights('ann_model.pth')
    print("\nModel saved as 'ann_model.pth'")

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Scaler saved as 'scaler.pkl'")


if __name__ == '__main__':
    main()