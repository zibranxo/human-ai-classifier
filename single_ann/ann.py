import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("iris_extended.csv")

# Encode categorical columns
df = pd.get_dummies(df, columns=['soil_type'], drop_first=True)

le = LabelEncoder()
df['species'] = le.fit_transform(df['species'])

X = df.drop('species', axis=1).values
y = df['species'].values.reshape(-1, 1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

class SingleLayerANN:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size, 1)
        self.bias = np.zeros((1,))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        self.z = np.dot(X, self.weights) + self.bias
        self.a = self.sigmoid(self.z)
        return self.a

    def backward(self, X, y, output, lr):
        m = X.shape[0]
        dz = output - y
        dw = np.dot(X.T, dz) / m
        db = np.sum(dz) / m

        self.weights -= lr * dw
        self.bias -= lr * db

    def train(self, X, y, epochs=1000, lr=0.01):
        for i in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)
            self.backward(X, y, output, lr)

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss}")

    def predict(self, X):
        probs = self.forward(X)
        return (probs > 0.5).astype(int)


model = SingleLayerANN(X_train.shape[1])
model.train(X_train, y_train, epochs=1000, lr=0.05)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))