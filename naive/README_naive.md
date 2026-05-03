# Naive Bayes Text Classifier for AI vs Human Detection

## 📌 Overview

This project implements a **Naive Bayes classifier** for detecting whether text is **AI-generated or human-written**.

It combines:

* **TF-IDF vectorization** (for feature extraction)
* **Multinomial Naive Bayes** (for probabilistic classification)

This is a strong baseline for **Natural Language Processing (NLP)** tasks.

---

## 🧠 Algorithm: Naive Bayes

### 🔹 Core Idea

Based on **Bayes’ Theorem**:

```text id="eq1"
P(y | x) = (P(x | y) * P(y)) / P(x)
```

---

### 🔹 “Naive” Assumption

* Assumes all features (words) are **independent**
* Not true in reality, but works surprisingly well

---

### 🔹 Multinomial Naive Bayes

* Designed for **text data**
* Uses **word frequencies / counts**

---

## ⚙️ Methodology

### 1. Dataset

* Input file: `output1.csv`
* Columns:

  * `text`: input text
  * `label_id`: target (AI vs Human)

---

### 2. Preprocessing

#### 🔹 Cleaning

* Removes rows with missing values

---

### 3. Train-Test Split

* 80% Training
* 20% Testing
* Stratified split ensures class balance

---

## 🧩 Pipeline Design

The model uses a **Scikit-learn Pipeline**:

```text id="eq2"
TF-IDF Vectorizer → Multinomial Naive Bayes
```

---

### 🔹 TF-IDF Configuration

* `ngram_range = (1, 2)` → unigrams + bigrams
* `stop_words = "english"`
* `max_features = 10000`

👉 Captures both:

* Individual words
* Short phrases

---

### 🔹 Naive Bayes Model

* `MultinomialNB(alpha=1.0)`
* Uses **Laplace smoothing** to handle unseen words

---

## 🔁 Training

```text id="eq3"
model.fit(X_train, y_train)
```

* Learns probability distributions of words per class

---

## 🔍 Prediction

```text id="eq4"
model.predict(X_test)
```

* Chooses class with highest posterior probability

---

## 📊 Evaluation Metrics

### 🔹 Accuracy

* Overall correctness

---

### 🔹 Classification Report

Includes:

* Precision
* Recall
* F1-score

---

### 🔹 Confusion Matrix

* Shows classification breakdown

---

## 🤖 Custom Inference

### Function:

```text id="eq5"
predict_text(text)
```

### Features:

* Takes raw text input
* Outputs:

  * Prediction (AI / Human)
  * Confidence score

---

### Example:

```text id="eq6"
Artificial intelligence is transforming industries rapidly.
```

---

## 🛠️ Implementation Flow

1. Load dataset
2. Clean missing values
3. Split data (train/test)
4. Create pipeline:

   * TF-IDF vectorizer
   * Naive Bayes classifier
5. Train model
6. Predict on test data
7. Evaluate metrics
8. Test custom inputs

---

## 🚀 How to Run

### 1. Install dependencies

```bash id="cmd1"
pip install pandas numpy scikit-learn
```

---

### 2. Run script

```bash id="cmd2"
python naive.py
```

---

## 📊 Strengths & Limitations

### ✅ Strengths

* Fast and efficient
* Works very well for text classification
* Handles high-dimensional data
* Simple to implement

---

### ❌ Limitations

* Assumes feature independence
* Cannot capture word order beyond n-grams
* Less powerful than deep learning models

---

## 🔮 Future Improvements

* Use **TF-IDF + Logistic Regression / SVM**
* Try **character-level features**
* Use **word embeddings (Word2Vec, GloVe)**
* Upgrade to **transformer models (BERT, RoBERTa)**

---

## 🎯 Key Takeaways

* Naive Bayes is a **probabilistic classifier**
* Extremely effective for **text classification tasks**
* Strong baseline before deep learning

👉 A must-know algorithm for NLP pipelines.

---
