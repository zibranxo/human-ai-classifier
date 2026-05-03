# K-Nearest Neighbors (KNN) Classifier for AI vs Human / Multi-Class Data

## 📌 Overview

This project implements the **K-Nearest Neighbors (KNN)** algorithm, a simple yet powerful **instance-based learning method**.

Unlike most models:

* KNN does **not learn a model explicitly**
* It **stores training data** and makes predictions at runtime
* Classification is based on **similarity (distance)**

---

## 🧠 Algorithm: K-Nearest Neighbors

### 🔹 Core Idea

To classify a new data point:

1. Find the **K closest points** in training data
2. Assign the class based on **majority voting**

---

### 🔹 Distance Metric

Typically uses **Euclidean distance**:

```text id="eq1"
d(x, y) = √Σ (x_i - y_i)^2
```

---

## ⚙️ Methodology

### 1. Dataset

* Input file: `iris.csv`
* Target column: `variety`

---

### 2. Preprocessing

#### 🔹 Categorical Encoding

* Uses `LabelEncoder`
* Converts categorical labels → numerical

---

#### 🔹 Feature Scaling

* Uses `StandardScaler`
* Ensures all features contribute equally

👉 Important because KNN is **distance-based**

---

### 3. Train-Test Split

* 80% Training
* 20% Testing
* `random_state = 0` for reproducibility

---

### 4. Model Training

* Algorithm: `KNeighborsClassifier`
* Parameter:

```text id="eq2"
k = 3
```

---

## 🔍 Prediction Process

For each test sample:

1. Compute distance to all training points
2. Select top **K nearest neighbors**
3. Predict most frequent class

---

## 📊 Evaluation

### Outputs:

#### ✅ Correct Predictions

* Displays samples where prediction matches actual label

#### ❌ Wrong Predictions

* Displays misclassified samples

---

### 📈 Accuracy

```text id="eq3"
Accuracy = (Correct Predictions) / (Total Predictions)
```

---

## 🛠️ Implementation Flow

1. Load dataset using `pandas`
2. Encode categorical features
3. Separate features (`X`) and target (`y`)
4. Scale features using `StandardScaler`
5. Split dataset into train/test
6. Train KNN model
7. Predict on test data
8. Print:

   * Correct predictions
   * Incorrect predictions
   * Accuracy

---

## 🚀 How to Run

### 1. Install dependencies

```bash id="cmd1"
pip install pandas numpy scikit-learn
```

---

### 2. Run script

```bash id="cmd2"
python knn.py
```

---

## 📊 Strengths & Limitations

### ✅ Strengths

* Simple and intuitive
* No training phase (lazy learning)
* Works well for small datasets
* Naturally handles multi-class problems

---

### ❌ Limitations

* Slow at prediction time (computes all distances)
* Memory-intensive (stores full dataset)
* Sensitive to:

  * Feature scaling
  * Noise
* Performance drops in high dimensions (curse of dimensionality)

---

## 🔮 Future Improvements

* Optimize K using **cross-validation**
* Try different distance metrics:

  * Manhattan
  * Minkowski
* Use **KD-Trees / Ball Trees** for faster search
* Apply dimensionality reduction (PCA)

---

## 🎯 Key Takeaways

* KNN is a **lazy learning algorithm**
* Relies on **distance-based similarity**
* Strong baseline for classification tasks

👉 Complements other models by showing a **data-driven, non-parametric approach**.

---
