# Single-Layer Artificial Neural Network (ANN) from Scratch

## 📌 Overview

This project implements a **Single-Layer Artificial Neural Network (ANN)** from scratch using **NumPy** for classification tasks.

Unlike high-level frameworks:

* No PyTorch / TensorFlow
* Manual implementation of:

  * Forward propagation
  * Backpropagation
  * Weight updates

👉 Focuses on **core neural network fundamentals**

---

## 🧠 Model Architecture

### 🔹 Structure

```text id="eq1"
Input Layer → Output Layer (Sigmoid)
```

* No hidden layers (single-layer perceptron)
* Output: probability (0–1)

---

### 🔹 Mathematical Model

```text id="eq2"
z = XW + b  
a = sigmoid(z)
```

---

## ⚙️ Methodology

### 1. Dataset

* Input: `iris_extended.csv`
* Target column:

```text id="eq3"
species
```

---

### 2. Preprocessing

#### 🔹 One-Hot Encoding

* Column: `soil_type`
* Converts categorical → numeric

---

#### 🔹 Label Encoding

* Converts target labels into integers

---

#### 🔹 Feature Scaling

* Uses `MinMaxScaler`
* Range: [0, 1]

---

### 3. Train-Test Split

* 80% Training
* 20% Testing

---

## 🔁 Training Process

### 🔹 Forward Propagation

```text id="eq4"
a = sigmoid(XW + b)
```

---

### 🔹 Loss Function

```text id="eq5"
MSE = mean((y - ŷ)^2)
```

---

### 🔹 Backpropagation

* Compute gradients:

```text id="eq6"
dz = ŷ - y
dw = Xᵀ * dz / m
db = sum(dz) / m
```

---

### 🔹 Weight Update

```text id="eq7"
W = W - lr * dw  
b = b - lr * db
```

---

### 🔹 Training Configuration

* Epochs: `1000`
* Learning rate: `0.05`

---

## 🔍 Prediction

```text id="eq8"
ŷ = sigmoid(XW + b)
```

* Threshold:

```text id="eq9"
ŷ > 0.5 → Class 1  
Else → Class 0
```

---

## 📊 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Classification Report:

  * Precision
  * Recall
  * F1-score

---

## 📈 Training Output

* Prints loss every 100 epochs
* Helps monitor convergence

---

## 🛠️ Implementation Flow

1. Load dataset
2. Encode categorical features
3. Normalize features
4. Initialize weights and bias
5. Train model:

   * Forward pass
   * Compute loss
   * Backpropagation
   * Update weights
6. Predict on test data
7. Evaluate performance

---

## 🚀 How to Run

### 1. Install dependencies

```bash id="cmd1"
pip install numpy pandas scikit-learn
```

---

### 2. Run script

```bash id="cmd2"
python ann.py
```

---

## 📊 Strengths & Limitations

### ✅ Strengths

* Full control over learning process
* Deep understanding of neural networks
* Lightweight and easy to debug
* No external deep learning frameworks

---

### ❌ Limitations

* Only linear decision boundary (no hidden layers)
* Uses MSE instead of better loss (BCE)
* No regularization
* Cannot model complex patterns

---

## 🔮 Future Improvements

* Add **hidden layers (MLP)**
* Use **Binary Cross-Entropy loss**
* Add **ReLU activation**
* Implement **mini-batch training**
* Add **momentum / Adam optimizer**

---

## 🎯 Key Takeaways

* ANN learns via **gradient descent + backpropagation**
* Even a single-layer network is a **logistic regression equivalent**
* Foundation for deep learning models

👉 This implementation demonstrates **true understanding of neural networks**, not just usage.

---
