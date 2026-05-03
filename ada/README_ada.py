# Ensemble Learning for AI vs Human Text Classification (Bagging & AdaBoost)

## 📌 Overview

This project implements **ensemble learning techniques** to classify text as **AI-generated or human-written**. Specifically, it compares two powerful ensemble methods:

* **Bagging (Bootstrap Aggregating)**
* **AdaBoost (Adaptive Boosting)**

Both models are built on top of **Decision Trees** and trained on TF-IDF features extracted from text data.

---

## ⚙️ Methodology

### 1. Dataset

* Input file: `output1.csv`
* Features:

  * `text`: raw textual data
  * `label_id`: target label (AI vs Human)

---

### 2. Text Preprocessing & Feature Engineering

* Text is converted into numerical form using **TF-IDF (Term Frequency–Inverse Document Frequency)**.
* Configuration:

  * `max_features = 5000`
* This ensures:

  * Important words are weighted higher
  * Noise and rare words are reduced

---

### 3. Train-Test Split

* Data is split into:

  * **80% Training**
  * **20% Testing**
* `random_state = 42` ensures reproducibility

---

## 🧠 Models Used

### 🔹 Bagging Classifier

* Base model: `DecisionTreeClassifier`
* Number of estimators: `50`

#### Intuition:

Bagging trains multiple decision trees on **different random subsets** of the data and averages their predictions.

👉 Helps reduce:

* Variance
* Overfitting (common in decision trees)

---

### 🔹 AdaBoost Classifier

* Base model: `DecisionTreeClassifier(max_depth=1)` (decision stumps)
* Number of estimators: `50`
* Learning rate: `1`

#### Intuition:

AdaBoost trains models **sequentially**, where:

* Each new model focuses more on previously misclassified samples
* Misclassified points get higher weights

👉 Result:

* Stronger overall classifier from weak learners

---

## 📊 Evaluation Metrics

Both models are evaluated using:

* **Accuracy Score**
* **Confusion Matrix**
* **Classification Report**

  * Precision
  * Recall
  * F1-score

---

## 📈 Visualization

A bar plot is generated to compare model performance:

* X-axis: Models (Bagging vs AdaBoost)
* Y-axis: Accuracy

---

## 🛠️ Implementation Details

### Pipeline Flow:

1. Load dataset using `pandas`
2. Extract features (`text`) and labels (`label_id`)
3. Convert text → TF-IDF vectors
4. Split dataset into train/test
5. Train:

   * Bagging model
   * AdaBoost model
6. Predict on test set
7. Evaluate using metrics
8. Plot accuracy comparison

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install pandas scikit-learn matplotlib
```

### 2. Place dataset

Ensure `output1.csv` is in the same directory.

### 3. Run the script

```bash
python ada.py
```

---

## 📊 Strengths & Limitations

### ✅ Bagging

**Strengths:**

* Reduces overfitting
* Works well with high-variance models

**Limitations:**

* Does not focus on hard examples
* May require many estimators

---

### ✅ AdaBoost

**Strengths:**

* Focuses on difficult samples
* Often achieves higher accuracy

**Limitations:**

* Sensitive to noise and outliers
* Can overfit if too many estimators

---

## 🔮 Future Improvements

* Use **n-grams in TF-IDF** for better context capture
* Try **word embeddings (Word2Vec, BERT)**
* Hyperparameter tuning (GridSearchCV)
* Combine multiple ensemble methods (stacking)
* Handle class imbalance (if present)

---

## 🎯 Key Takeaway

* **Bagging → reduces variance**
* **AdaBoost → reduces bias by focusing on mistakes**

Together, they provide a strong baseline for detecting AI-generated text using classical ML techniques.

---
