# Linear Regression for Continuous Prediction

## 📌 Overview

This project implements **Linear Regression**, a fundamental supervised learning algorithm used to predict **continuous values**.

Unlike classification models (Logistic Regression, Trees, KNN), this model:

* Predicts **numerical outputs**
* Learns a **linear relationship** between features and target

---

## 🧠 Algorithm: Linear Regression

### 🔹 Core Idea

Model the relationship between input features and output as:

```text id="eq1"
y = w₁x₁ + w₂x₂ + ... + wn*xn + b
```

---

### 🔹 Objective

Minimize prediction error using:

```text id="eq2"
Mean Squared Error (MSE)
```

---

## ⚙️ Methodology

### 1. Dataset

* Input file: `iris_extended.csv`
* Target variable:

```text id="eq3"
petal_length
```

---

### 2. Preprocessing

#### 🔹 One-Hot Encoding

* Uses `pd.get_dummies()`
* Converts categorical features → binary columns
* Avoids ordinal bias (better than LabelEncoder)

---

#### 🔹 Feature Scaling

* Uses `StandardScaler`
* Ensures:

  * Mean = 0
  * Std Dev = 1

---

### 3. Train-Test Split

* 80% Training
* 20% Testing
* `random_state = 42`

---

## 🔁 Model Training

* Algorithm: `LinearRegression`
* Learns optimal weights using least squares

---

## 🔍 Prediction

* Generates predicted values:

```text id="eq4"
y_pred = model.predict(X_test)
```

---

## 📊 Evaluation Metrics

### 🔹 1. Mean Squared Error (MSE)

```text id="eq5"
MSE = (1/n) Σ (y_true - y_pred)²
```

* Lower = better

---

### 🔹 2. R² Score

```text id="eq6"
R² = 1 - (SS_res / SS_total)
```

* Measures how well model explains variance
* Range:

  * 1 → perfect fit
  * 0 → no explanatory power

---

## 📈 Visualizations

### 1. Actual vs Predicted

* Scatter plot comparing real vs predicted values
* Ideal: points lie on diagonal

---

### 2. Residual Plot

* X-axis: predicted values
* Y-axis: residuals (errors)

👉 Used to detect:

* Non-linearity
* Heteroscedasticity

---

### 3. Residual Distribution

* Histogram of errors
* Ideal:

  * Symmetrical
  * Centered around 0

---

## 🛠️ Implementation Flow

1. Load dataset
2. Apply one-hot encoding
3. Split features and target
4. Scale features
5. Train linear regression model
6. Predict on test data
7. Compute metrics (MSE, R²)
8. Generate plots

---

## 🚀 How to Run

### 1. Install dependencies

```bash id="cmd1"
pip install pandas numpy scikit-learn matplotlib
```

---

### 2. Run script

```bash id="cmd2"
python lr.py
```

---

## 📊 Strengths & Limitations

### ✅ Strengths

* Simple and interpretable
* Fast to train
* Works well for linear relationships
* Good baseline model

---

### ❌ Limitations

* Assumes linear relationship
* Sensitive to outliers
* Cannot capture complex patterns
* Requires proper feature engineering

---

## 🔮 Future Improvements

* Add **Polynomial Regression**
* Use **Regularization**:

  * Ridge (L2)
  * Lasso (L1)
* Perform feature selection
* Check multicollinearity (VIF)

---

## 🎯 Key Takeaways

* Linear Regression is the **foundation of supervised learning**
* Focuses on predicting **continuous values**
* Helps understand:

  * Model assumptions
  * Error analysis
  * Bias-variance tradeoff

👉 Essential stepping stone before advanced ML models.

---
