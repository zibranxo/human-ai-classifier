import pandas as pd
import numpy as np

# ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
df = pd.read_csv("output1.csv")

# Basic cleaning
df = df.dropna(subset=["text", "label_id"])

X = df["text"]
y = df["label_id"]

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Pipeline (Vectorizer + Model)
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),      # unigrams + bigrams
        stop_words="english",
        max_features=10000
    )),
    ("nb", MultinomialNB(alpha=1.0))
])

# 4. Train
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Test on custom input
def predict_text(text):
    pred = model.predict([text])[0]
    prob = model.predict_proba([text])[0]
    
    label = "AI Generated" if pred == 1 else "Human Written"
    
    print(f"\nText: {text}")
    print(f"Prediction: {label}")
    print(f"Confidence: {np.max(prob):.4f}")

# Example
predict_text("Artificial intelligence is transforming industries rapidly.")