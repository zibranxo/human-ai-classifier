import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("iris_extended.csv")

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = le.fit_transform(df[col])

# Drop target column (species)
X = df.drop('species', axis=1)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans
k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(X_scaled)

# Metrics
inertia = kmeans.inertia_
sil_score = silhouette_score(X_scaled, labels)

print("Inertia (WCSS):", inertia)
print("Silhouette Score:", sil_score)

# Elbow Method
wcss = []
K_range = range(1, 10)

for i in K_range:
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(X_scaled)
    wcss.append(km.inertia_)

# Plot
plt.plot(K_range, wcss, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()