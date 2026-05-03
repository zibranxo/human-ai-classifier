# ID3 Decision Tree Classifier (From Scratch) + Graph Visualization

## 📌 Overview

This project implements the **ID3 (Iterative Dichotomiser 3)** algorithm from scratch for classifying data into **AI-generated vs Human-written text**.

It also includes a **Graphviz-based visualization tool** to convert the trained decision tree into a visual graph.

---

## 🧠 Algorithm: ID3

ID3 builds a decision tree using:

```text id="eq1"
Information Gain = Entropy(parent) − Weighted Entropy(children)
```

---

### 🔹 Core Idea

* Select the feature that maximizes **Information Gain**
* Split dataset based on feature values
* Repeat recursively

---

## ⚙️ Methodology

### 1. Data Loading

* Input: CSV file
* Target column: `label_id`
* Features: all remaining columns

---

### 2. Train-Test Split

* Stratified sampling (preserves class balance)
* Default:

  * 80% Train
  * 20% Test

---

### 3. Tree Construction

### 🔁 Recursive Splitting

Stops when:

* All labels are identical
* No features remain
* Max depth reached (optional)

---

### 🔹 Feature Selection

* Compute **Information Gain** for each feature
* Select feature with highest gain

---

### 🔹 Important Limitation

* Only supports **categorical features**
* No handling of missing values
* No pruning (can overfit)

---

## 🌳 Tree Structure

Stored as nested dictionary:

```text id="eq2"
{
  "feature": feature_name,
  "branches": {
    "value1": subtree,
    "value2": subtree
  }
}
```

---

## 🔍 Prediction Logic

* Traverse tree using feature values
* If unseen value:

  * Fallback to majority branch
* Return leaf label

---

## 📊 Evaluation Metrics

* Accuracy
* Confusion Matrix
* Per-sample predictions

---

## 📈 Visualizations

### 📊 Core Plots

* Confusion Matrix (train + test)
* Feature importance (Information Gain)
* Accuracy comparison (train vs test)

---

### 📉 Advanced Analysis

* Label distribution (train vs test)
* Per-feature class distribution

---

## 🌲 Tree Visualization (Graphviz)

The project includes a separate script to convert the trained tree into a graph:

📄 Script: `treemaker.py` 

---

### 🔹 How it works

* Loads `id3_tree.json`
* Converts tree into **DOT format**
* Generates `tree.dot`

---

### 🔹 Output Example

```text id="eq3"
digraph Tree {
  node0 [label="feature_name"];
  node0 -> node1 [label="value"];
}
```

---

### 🔹 Visualize the Tree

Use Graphviz:

```bash id="cmd1"
dot -Tpng tree.dot -o tree.png
```

---

## 🛠️ Implementation Highlights

### 🔹 Entropy

```text id="eq4"
H(S) = - Σ p(x) log2 p(x)
```

---

### 🔹 Information Gain

* Measures reduction in uncertainty
* Greedy feature selection

---

### 🔹 ASCII Tree Printing

* Displays hierarchical structure
* Shows:

  * Feature splits
  * Node counts

---

## 🤖 Inference Modes

### 1. Interactive Mode

* User inputs features manually

### 2. Quick Mode

* Comma-separated input

---

## 💾 Output

### Plot directory:

```text id="eq5"
./plots_id3/
```

Includes:

* Confusion matrix
* Feature importance
* Distribution plots
* Accuracy summary

---

### Tree storage:

```text id="eq6"
id3_tree.json
```

---

### Visualization files:

```text id="eq7"
tree.dot → tree.png
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash id="cmd2"
pip install matplotlib numpy
```

---

### 2. Run ID3 model

```bash id="cmd3"
python id3.py
```

---

### 3. Generate visualization

```bash id="cmd4"
python treemaker.py
```

---

### 4. Render tree

```bash id="cmd5"
dot -Tpng tree.dot -o tree.png
```

---

## 📊 Strengths & Limitations

### ✅ Strengths

* Simple and interpretable
* Easy to implement
* Strong baseline model
* Clear feature importance

---

### ❌ Limitations

* Overfits easily (no pruning)
* Cannot handle continuous features
* Biased toward high-cardinality features
* No missing value handling

---

## 🔮 Future Improvements

* Extend to **C4.5 (Gain Ratio)** ✅ (already implemented in repo)
* Add **pruning techniques**
* Support **continuous features**
* Integrate with ensemble methods

---

## 🎯 Key Takeaways

* ID3 is the **foundation of decision tree learning**
* Uses entropy and information gain for splitting
* Forms the basis for:

  * C4.5
  * CART
  * Random Forest

👉 This implementation shows **fundamental understanding of tree-based ML algorithms**.

---
