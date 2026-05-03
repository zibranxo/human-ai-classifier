# Logistic Regression Classifier for AI vs Human / Multi-Class Data

## 📌 Overview

This project implements **Logistic Regression**, a fundamental supervised learning algorithm used for **classification tasks**.

Unlike Linear Regression:

* Output is **probability (0–1)**
* Uses a **sigmoid (logistic) function**
* Suitable for **binary and multi-class classification**

---

## 🧠 Algorithm: Logistic Regression

### 🔹 Core Idea

Instead of predicting raw values, Logistic Regression predicts probability:

```text id="eq1"
p(y=1 | x) = 1 / (1 + e^-(wx + b))
```

---

### 🔹 Decision Rule

```text id="eq2"
If p ≥ 0.5 → Class 1  
Else → Class 0
```

---

### 🔹 Multi-Class Handling

* Uses **One-vs-Rest (OvR)** internally
* Predicts class with highest probability

---

## ⚙️ Methodology

### 1. Dataset

* Input file: `iris_extended.csv`
* Target column:

```text id="eq3"
species
```

---

### 2. Preprocessing

#### 🔹 One-Hot Encoding

* Column: `soil_type`
* Converts categorical → binary features
* Uses:

```text id="eq4"
pd.get_dummies(..., drop_first=True)
```

---

#### 🔹 Label Encoding

* Encodes target variable (`species`) into numeric classes

---

#### 🔹 Feature Scaling

* Uses `StandardScaler`
* Important because Logistic Regression is sensitive to feature scale

---

### 3. Train-Test Split

* 80% Training
* 20% Testing
* `random_state = 42`

---

## 🔁 Model Training

* Algorithm: `LogisticRegression`
* Configuration:

```text id="eq5"
max_iter = 1000
```

👉 Ensures convergence during optimization

---

## 🔍 Prediction

* Predicts class labels:

```text id="eq6"
y_pred = model.predict(X_test)
```

---

## 📊 Evaluation Metrics

### 🔹 1. Accuracy

```text id="eq7"
Accuracy = Correct Predictions / Total Predictions
```

---

### 🔹 2. Confusion Matrix

* Shows:

  * True Positives
  * False Positives
  * False Negatives
  * True Negatives

---

### 🔹 3. Classification Report

Includes:

* Precision
* Recall
* F1-score

---

## 📈 Visualization

### Confusion Matrix Heatmap

* Uses `seaborn`
* Color-coded matrix for easy interpretation

---

## 🛠️ Implementation Flow

1. Load dataset
2. Apply one-hot encoding (features)
3. Encode target labels
4. Scale features
5. Split dataset
6. Train Logistic Regression model
7. Predict on test set
8. Evaluate using:

   * Accuracy
   * Confusion matrix
   * Classification report
9. Visualize confusion matrix

---

## 🚀 How to Run

### 1. Install dependencies

```bash id="cmd1"
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

### 2. Run script

```bash id="cmd2"
python lr(1).py
```

---

## 📊 Strengths & Limitations

### ✅ Strengths

* Simple and interpretable
* Fast training
* Works well for linearly separable data
* Provides probabilistic output

---

### ❌ Limitations

* Assumes linear decision boundary
* Struggles with complex patterns
* Sensitive to outliers
* Requires feature scaling

---

## 🔮 Future Improvements

* Add **Regularization**:

  * L1 (Lasso)
  * L2 (Ridge)
* Tune hyperparameters (C, solver)
* Try **Polynomial features**
* Compare with non-linear models (SVM, ANN)

---

## 🎯 Key Takeaways

* Logistic Regression is a **classification algorithm (not regression despite name)**
* Outputs **probabilities using sigmoid**
* Strong baseline for many ML tasks

👉 Essential model for understanding **decision boundaries and probabilistic ML**.

---
