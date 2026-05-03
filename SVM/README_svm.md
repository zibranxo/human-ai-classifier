# Support Vector Machine (SVM) Classifier for AI vs Human / Multi-Class Data

## 📌 Overview

This project implements a **Support Vector Machine (SVM)** classifier for classification tasks.

SVM is a powerful algorithm that:

* Finds the **optimal decision boundary (hyperplane)**
* Maximizes the **margin between classes**
* Uses **kernel tricks** to handle non-linear data

---

## 🧠 Algorithm: Support Vector Machine

### 🔹 Core Idea

Find a hyperplane that separates classes with **maximum margin**:

```text id="eq1"
w · x + b = 0
```

---

### 🔹 Margin Maximization

* SVM chooses the boundary that maximizes distance between:

  * Closest points (support vectors)
  * Decision boundary

---

### 🔹 Kernel Trick

This implementation uses:

```text id="eq2"
kernel = 'rbf'
```

👉 RBF (Radial Basis Function):

* Maps data to higher dimensions
* Handles non-linear relationships

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

#### 🔹 Column Cleaning

* Removes whitespace from column names

---

#### 🔹 Label Encoding

* Converts categorical values → numeric

---

#### 🔹 Feature Scaling

* Uses `StandardScaler`
* Critical for SVM performance

---

### 3. Train-Test Split

* 80% Training
* 20% Testing
* `random_state = 42`

---

## 🔁 Model Training

* Algorithm: `SVC`
* Kernel:

```text id="eq4"
RBF (Radial Basis Function)
```

---

## 🔍 Prediction

```text id="eq5"
y_pred = model.predict(X_test)
```

---

## 📊 Evaluation Metrics

### 🔹 Accuracy

* Overall correctness of predictions

---

### 🔹 Confusion Matrix

* Shows classification breakdown

---

### 🔹 Classification Report

Includes:

* Precision
* Recall
* F1-score

---

## 🛠️ Implementation Flow

1. Load dataset
2. Clean column names
3. Encode categorical features
4. Split features and target
5. Scale features
6. Train SVM model
7. Predict on test set
8. Evaluate results

---

## 🚀 How to Run

### 1. Install dependencies

```bash id="cmd1"
pip install pandas scikit-learn
```

---

### 2. Run script

```bash id="cmd2"
python svm.py
```

---

## 📊 Strengths & Limitations

### ✅ Strengths

* Effective in high-dimensional spaces
* Works well with small/medium datasets
* Can model non-linear decision boundaries
* Robust to overfitting (with proper tuning)

---

### ❌ Limitations

* Slower on large datasets
* Requires careful hyperparameter tuning
* Harder to interpret than linear models
* Sensitive to feature scaling

---

## 🔮 Future Improvements

* Tune hyperparameters:

  * `C` (regularization)
  * `gamma` (kernel coefficient)
* Try different kernels:

  * Linear
  * Polynomial
* Use GridSearchCV for optimization
* Add probability outputs (`probability=True`)

---

## 🎯 Key Takeaways

* SVM finds **maximum-margin decision boundaries**
* Kernel trick allows **non-linear classification**
* One of the most powerful classical ML algorithms

👉 Complements your project by adding a **geometric + optimization-based model**

---
