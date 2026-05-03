# C4.5 Decision Tree Classifier (From Scratch) for AI vs Human Detection

## 📌 Overview

This project implements the **C4.5 Decision Tree algorithm from scratch**, designed to classify data into **AI-generated vs Human-generated** categories.

Unlike library-based approaches, this implementation:

* Computes **entropy, information gain, and gain ratio manually**
* Handles **missing values (C4.5-style)**
* Builds a full decision tree recursively
* Provides **interpretability, visualization, and inference tools**

---

## 🧠 Algorithm: C4.5 Decision Tree

C4.5 is an extension of ID3 with key improvements:

| Feature                | ID3              | C4.5       |
| ---------------------- | ---------------- | ---------- |
| Split metric           | Information Gain | Gain Ratio |
| Handles missing values | ❌                | ✅          |
| Bias correction        | ❌                | ✅          |
| Continuous features    | Limited          | Better     |

---

### 🔹 Core Idea

* Select feature that maximizes:

```text id="c2g9kx"
Gain Ratio = Information Gain / Split Information
```

* Prevents bias toward features with many unique values

---

## ⚙️ Methodology

### 1. Data Loading

* Input: CSV file
* Target column: `label_id`
* Ignores:

  * `sr_no`
  * empty columns

---

### 2. Preprocessing

* Missing values handled using sentinel:

```text id="x8v2ml"
MISSING = "?"
```

---

### 3. Train-Test Split

* **Stratified sampling**
* Maintains class balance
* Default:

  * 80% train
  * 20% test

---

## 🌳 Tree Construction

### 🔁 Recursive Building

Stops when:

* All labels are same → leaf node
* No features left
* Max depth reached

---

### 🔹 Feature Selection Strategy

1. Compute **Information Gain** for all features
2. Filter features with **above-average IG**
3. Select feature with **maximum Gain Ratio**

---

### 🔹 Handling Missing Values

* Ignores missing values when computing splits
* Distributes missing samples **proportionally across branches**
* During prediction:

  * Uses **weighted voting across branches**

---

## 🔍 Prediction Logic

### Cases handled:

* Known feature value → follow branch
* Missing value → weighted vote across branches
* Unseen value → fallback to most frequent branch

---

## 📊 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Per-sample prediction table (debug-friendly)

---

## 📈 Visualizations

This implementation includes a **rich visualization suite**:

### 📊 Core Plots

* Confusion Matrix (train + test)
* Feature importance (Gain Ratio)
* Accuracy comparison (train vs test)

---

### 📉 Advanced Analysis

* Information Gain vs Gain Ratio scatter
* Label distribution (train vs test)
* Per-feature class distribution

---

## 🛠️ Implementation Highlights

### Key Components:

#### 🔹 Entropy Calculation

```text id="v1k3mz"
H(S) = - Σ p(x) log2 p(x)
```

---

#### 🔹 Information Gain (C4.5 adjusted)

* Accounts for missing values
* Uses only known samples

---

#### 🔹 Gain Ratio

* Penalizes high-cardinality features

---

#### 🔹 Tree Structure

Stored as nested dictionary:

```text id="7f4lpn"
{
  "feature": "feature_name",
  "branches": {
    "value1": subtree,
    "value2": subtree
  }
}
```

---

## 🌲 Tree Visualization

* ASCII tree printing with structure:

```text id="9sj2xp"
SPLIT on 'feature' (gain_ratio=...)
├── value1 → ...
└── value2 → ...
```

---

## 🤖 Inference Modes

### 1. Interactive Mode

* User inputs features one-by-one

### 2. Quick Mode

* Single-line comma-separated input

---

## 💾 Output

### Plots saved to:

```text id="k29zvb"
./plots_c45/
```

Includes:

* Confusion matrices
* Feature importance
* Distribution plots
* Accuracy summary

---

### Optional:

* Save trained tree:

```text id="l1pz8r"
c45_tree.json
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash id="q4z8nx"
pip install matplotlib numpy
```

---

### 2. Run script

```bash id="j3v9sa"
python c4.5.py
```

---

### 3. Provide inputs

* CSV file path
* Test split ratio (optional)
* Max depth (optional)
* Output directory

---

## 📊 Strengths & Limitations

### ✅ Strengths

* Fully interpretable model
* Handles missing values elegantly
* No external ML libraries used
* Strong visualization + debugging tools

---

### ❌ Limitations

* Works best with categorical features
* Not optimized for very large datasets
* No pruning (can overfit)
* Slower than optimized implementations

---

## 🔮 Future Improvements

* Add **post-pruning** (reduced error pruning)
* Support **continuous feature splitting**
* Convert to **Random Forest / Gradient Boosting**
* Optimize recursion for large datasets
* Add **feature selection preprocessing**

---

## 🎯 Key Takeaways

* C4.5 improves over ID3 using **Gain Ratio**
* Handling missing values is crucial in real-world data
* Decision trees provide **interpretability + explainability**

👉 This implementation demonstrates **deep algorithmic understanding**, not just usage.

---
