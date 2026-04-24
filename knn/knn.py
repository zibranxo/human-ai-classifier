import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("iris.csv")

# Encode categorical columns (only 'variety')
le = LabelEncoder()
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop('variety', axis=1)
y = df['variety']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=0
)

# KNN model
k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Correct Predictions
print("\nCorrect Predictions:")
for i in range(len(y_test)):
    if y_pred[i] == y_test.iloc[i]:
        print(f"Actual: {y_test.iloc[i]}  Predicted: {y_pred[i]}")

# Wrong Predictions
print("\nWrong Predictions:")
for i in range(len(y_test)):
    if y_pred[i] != y_test.iloc[i]:
        print(f"Actual: {y_test.iloc[i]}  Predicted: {y_pred[i]}")

# Accuracy
accuracy = np.mean(y_pred == y_test)
print("\nAccuracy:", accuracy)