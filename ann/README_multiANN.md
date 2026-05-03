# Custom Artificial Neural Network (ANN) for AI vs Human Classification

## 📌 Overview

This project implements a **fully custom Artificial Neural Network (ANN)** from scratch using **PyTorch tensors (without autograd)** to classify data into **AI-generated vs Human-generated** categories.

Unlike high-level frameworks, this implementation manually defines:

* Forward propagation
* Backpropagation
* Weight updates
* Loss computation

This makes it ideal for **deep understanding of neural networks internals**.

---

## 🧠 Model Architecture

The network follows a **4-layer fully connected architecture**:

```
Input Layer (21 features)
        ↓
Hidden Layer 1 (64 neurons, ReLU)
        ↓
Hidden Layer 2 (32 neurons, ReLU)
        ↓
Hidden Layer 3 (16 neurons, ReLU)
        ↓
Output Layer (1 neuron, Sigmoid)
```

### Key Design Choices:

* **ReLU activation** → prevents vanishing gradients
* **Sigmoid output** → binary classification
* **He Initialization** → optimized for ReLU layers
* **L2 Regularization** → prevents overfitting

---

## ⚙️ Methodology

### 1. Data Loading & Preprocessing

* Dataset: `dataset.csv`
* Assumption:

  * Last column → label
  * Remaining columns → features

#### Preprocessing Steps:

* Normalize features using **MinMaxScaler (0–1 range)**
* Convert to PyTorch tensors

---

### 2. Train / Validation / Test Split

* 80% Training + 20% Testing
* Training further split into:

  * 80% Train
  * 20% Validation

---

### 3. Handling Class Imbalance

* Automatically detects imbalance
* Applies **class-weighted Binary Cross-Entropy loss**

---

## 🔁 Training Strategy

### Optimization Technique:

* **Mini-batch Gradient Descent**
* Batch size: `64`

### Loss Function:

* Binary Cross-Entropy (BCE)
* Optional **weighted BCE** for imbalance

### Regularization:

* **L2 weight decay** applied during backpropagation

---

### ⏹️ Early Stopping

* Monitors validation loss
* Stops training if no improvement for `20 epochs`
* Restores best model weights

---

## 🔬 Core Learning Components

### 🔹 Forward Propagation

* Matrix multiplications + activations layer by layer

### 🔹 Backpropagation (Manual)

* Computes gradients layer-by-layer using chain rule
* Updates weights using:

```
w = w - η * (gradient + λ * w)
```

### 🔹 Activation Functions

* ReLU:

  * Fast, avoids vanishing gradient
* Sigmoid:

  * Outputs probability (0–1)

---

## 📊 Evaluation Metrics

* **Accuracy**
* **Confusion Matrix**
* **Classification Report**

  * Precision
  * Recall
  * F1-score

---

## 📈 Visualization

### Training Insights:

* Loss vs Epoch
* Accuracy vs Epoch
* Loss Gap (Overfitting indicator)
* Per-epoch loss improvement

### Evaluation Insights:

* Confusion matrix heatmap
* Prediction distribution
* Per-class accuracy

---

## 🛠️ Implementation Highlights

### Key Features:

* GPU support (CUDA if available)
* Batch-wise inference (memory safe)
* Modular ANN class design
* Model checkpointing (best weights)
* Scaler persistence (`scaler.pkl`)

---

## 🚀 How to Run

### 1. Install dependencies

```bash id="d2l9s1"
pip install torch numpy pandas scikit-learn matplotlib
```

### 2. Prepare dataset

Place `dataset.csv` in the project directory.

---

### 3. Run training

```bash id="f8x2p0"
python train.py
```

---

### 4. Output files

* `ann_model.pth` → trained model weights
* `scaler.pkl` → preprocessing scaler
* `training_history.png` → training curves
* `evaluation_results.png` → performance visuals

---

## 📊 Strengths & Limitations

### ✅ Strengths

* Full control over training pipeline
* Deep understanding of NN internals
* Handles class imbalance effectively
* Regularization + early stopping improves generalization

---

### ❌ Limitations

* No automatic differentiation (manual gradients = error-prone)
* Slower than optimized frameworks
* Fixed architecture (not dynamically configurable)
* Requires numerical features (no raw text handling)

---

## 🔮 Future Improvements

* Replace manual backprop with **PyTorch autograd**
* Add **Dropout layers** for better regularization
* Support **dynamic architectures**
* Integrate **text embeddings (BERT, TF-IDF)**
* Add **learning rate scheduling**
* Implement **ROC-AUC evaluation**

---

## 🎯 Key Takeaways

* This project demonstrates **true understanding of neural networks**, not just usage.
* Combines:

  * Mathematical rigor (manual gradients)
  * Practical ML (class imbalance, early stopping)
  * Engineering (GPU support, batching, persistence)

👉 Strong foundation for advanced deep learning systems.

---
