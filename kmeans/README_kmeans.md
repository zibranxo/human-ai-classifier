# K-Means Clustering for Pattern Discovery in Text/Data

## 📌 Overview

This project implements **K-Means Clustering**, an unsupervised learning algorithm, to discover hidden patterns in data.

Unlike supervised models (Logistic Regression, Trees, ANN), K-Means:

* Does **not use labels during training**
* Groups data into clusters based on similarity
* Helps analyze structure in AI vs Human-like data distributions

---

## 🧠 Algorithm: K-Means

### 🔹 Core Idea

Partition data into **K clusters** such that:

```text id="eq1"
Minimize: Within-Cluster Sum of Squares (WCSS)
```

---

### 🔁 Steps

1. Initialize K centroids
2. Assign each point to nearest centroid
3. Update centroids (mean of cluster points)
4. Repeat until convergence

---

## ⚙️ Methodology

### 1. Dataset

* Input file: `iris_extended.csv`
* Target column: `species` (removed before training)

---

### 2. Preprocessing

#### 🔹 Categorical Encoding

* Uses `LabelEncoder`
* Converts text → numerical values

---

#### 🔹 Feature Scaling

* Uses `StandardScaler`
* Ensures:

  * Mean = 0
  * Std Dev = 1

👉 Important because K-Means is **distance-based**

---

### 3. Model Training

* Algorithm: `KMeans`
* Number of clusters:

```text id="eq2"
k = 3
```

---

## 📊 Evaluation Metrics

### 🔹 1. Inertia (WCSS)

* Measures compactness of clusters
* Lower = better

---

### 🔹 2. Silhouette Score

```text id="eq3"
Range: [-1, 1]
```

* Measures how well-separated clusters are
* Closer to 1 = better clustering

---

## 📈 Model Selection: Elbow Method

### 🔹 Idea

* Run K-Means for multiple values of K
* Plot:

```text id="eq4"
K vs WCSS
```

* Choose K where curve bends ("elbow")

---

### 📊 Output Plot

* X-axis: Number of clusters (K)
* Y-axis: WCSS

---

## 🛠️ Implementation Flow

1. Load dataset using `pandas`
2. Encode categorical features
3. Remove target column
4. Scale features using `StandardScaler`
5. Apply K-Means clustering
6. Compute:

   * Inertia
   * Silhouette Score
7. Perform Elbow Method
8. Plot results

---

## 🚀 How to Run

### 1. Install dependencies

```bash id="cmd1"
pip install pandas numpy scikit-learn matplotlib
```

---

### 2. Run script

```bash id="cmd2"
python kmeans.py
```

---

## 📊 Output

* Prints:

  * Inertia (WCSS)
  * Silhouette Score

* Displays:

  * Elbow curve plot

---

## 📊 Strengths & Limitations

### ✅ Strengths

* Simple and fast
* Works well on structured data
* Scales to large datasets
* Useful for exploratory analysis

---

### ❌ Limitations

* Requires choosing K manually
* Sensitive to initialization
* Assumes spherical clusters
* Not ideal for high-dimensional text data

---

## 🔮 Future Improvements

* Use **K-Means++ initialization**
* Try **Dimensionality Reduction (PCA, t-SNE)**
* Apply to **TF-IDF or embeddings**
* Compare with:

  * DBSCAN
  * Hierarchical clustering

---

## 🎯 Key Takeaways

* K-Means is an **unsupervised learning algorithm**
* Helps discover **hidden patterns without labels**
* Complements classification models in ML pipelines

👉 Adds an important dimension to your project: **data understanding before modeling**.

---
