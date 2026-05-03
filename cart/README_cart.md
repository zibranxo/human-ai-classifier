# CART Decision Tree Classifier (From Scratch) for AI vs Human Detection

## 📌 Overview

This project implements the **CART (Classification and Regression Trees)** algorithm from scratch to classify data into **AI-generated vs Human-generated** categories.

Unlike ID3 and C4.5, CART:

* Uses **Gini Impurity** instead of entropy
* Builds **binary trees only**
* Supports both **categorical and numerical features**
* Allows **feature reuse across splits**

---

## 🧠 Algorithm: CART

### 🔹 Core Idea

At each node, choose the split that minimizes:

```text id="g8x2mn"
Gini = 1 - Σ p(x)^2
```

---

### 🔹 Split Strategy

#### For Numerical Features:

```text id="t9m2ka"
Split: value ≤ threshold  vs  value > threshold
```

#### For Categorical Features:

```text id="l5z8qp"
Split: value == category  vs  value != category
```

---

## ⚙️ Methodology

### 1. Data Loading

* Input: CSV file
* Target column: `label_id`
* Automatically detects:

  * Numerical features
  * Categorical features

---

### 2. Train-Test Split

* Stratified sampling
* Default:

  * 80% Train
  * 20% Test

---

### 3. Tree Construction

### 🔁 Recursive Splitting

Stops when:

* All samples belong to one class
* Max depth reached (optional)
* Minimum samples condition met

---

### 🔹 Feature Selection

* Evaluate all features
* Compute best binary split
* Choose split with **lowest Gini impurity**

---

### 🔹 Key Difference from C4.5

* CART does **not remove features after split**
* Same feature can be reused multiple times

---

## 🌳 Tree Structure

Each node stores:

```text id="u8k4mv"
{
  "feature": feature_name,
  "split": {
    "type": "numeric" | "categorical",
    "threshold" OR "value"
  },
  "left": subtree,
  "right": subtree
}
```

---

## 🔍 Prediction Logic

* Traverse tree based on split condition
* Continue until leaf node
* Output predicted class

---

## 📊 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Per-sample predictions

---

## 📈 Visualizations

### 📊 Core Plots

* Confusion Matrix (train + test)
* Accuracy comparison (train vs test)
* Feature importance (Gini reduction)

---

### 📉 Advanced Analysis

* Gini vs Entropy comparison
* Label distribution (train vs test)
* Per-feature class distribution

---

## 🛠️ Implementation Highlights

### 🔹 Gini Impurity

```text id="m2x7vr"
Gini = 1 - Σ (p_i)^2
```

---

### 🔹 Gini Split

* Weighted impurity of left and right nodes

---

### 🔹 Feature Importance

* Based on **total impurity reduction**
* Aggregated across tree

---

### 🔹 Numeric Feature Handling

* Automatically detects numeric columns
* Computes optimal thresholds dynamically

---

## 🌲 Tree Visualization

* ASCII tree output:

```text id="d7p3kw"
SPLIT 'feature' <= threshold
├── left branch
└── right branch
```

Includes:

* Gini values
* Sample counts
* Class distribution

---

## 🤖 Inference Modes

### 1. Interactive Mode

* User inputs features manually

### 2. Quick Mode

* Comma-separated input

---

## 💾 Output

### Plot directory:

```text id="v9x3pl"
./plots_cart/
```

Includes:

* Confusion matrices
* Feature importance
* Distribution plots
* Accuracy summary

---

### Optional:

* Save tree:

```text id="h4k8sq"
cart_tree.json
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash id="z3n8qp"
pip install matplotlib numpy
```

---

### 2. Run script

```bash id="k9w2xz"
python cart.py
```

---

### 3. Provide inputs

* CSV file path
* Test split ratio
* Max depth (optional)
* Min samples split (optional)

---

## 📊 Strengths & Limitations

### ✅ Strengths

* Handles both numerical & categorical features
* Binary splits simplify tree structure
* Strong interpretability
* Feature importance insight

---

### ❌ Limitations

* Can overfit without pruning
* Greedy splitting (no global optimization)
* Sensitive to noisy data
* Larger trees compared to C4.5

---

## 🔮 Future Improvements

* Add **pruning techniques** (cost-complexity pruning)
* Convert to **Random Forest / Gradient Boosting**
* Add **handling for missing values**
* Optimize split search for large datasets

---

## 🎯 Key Takeaways

* CART builds **binary trees using Gini impurity**
* Supports **continuous + categorical features**
* Forms the foundation of:

  * Random Forest
  * Gradient Boosting

👉 This implementation shows **practical + theoretical mastery of decision trees**.

---
