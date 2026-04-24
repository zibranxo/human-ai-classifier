import os
import re
import string
import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
import textstat
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.sparse import hstack, csr_matrix

# =========================
# 1. DOWNLOAD NLTK DATA
# =========================
nltk.download('punkt')
nltk.download('stopwords')

# =========================
# 2. CONFIG
# =========================
DATA_PATH = "dataset.csv"   # change if your file name is different
TEXT_COLUMN = "text"
LABEL_COLUMN = "label_id"

TFIDF_MAX_FEATURES = 10000
MAX_TEXT_LENGTH_FOR_SPEED = None  # set to e.g. 3000 if dataset is huge and too slow
RANDOM_STATE = 42

# =========================
# 3. LOAD DATA
# =========================
print("Loading dataset...")
df = pd.read_csv("finaldatset.csv")

print("Original shape:", df.shape)

# Keep only needed columns
df = df[[TEXT_COLUMN, LABEL_COLUMN]].copy()

# Drop missing values
df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)

# Convert text to string
df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str)

# Convert labels to int
df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

# Remove blank texts
df = df[df[TEXT_COLUMN].str.strip() != ""]

# Optional: remove duplicates
df.drop_duplicates(subset=[TEXT_COLUMN], inplace=True)

# Optional: trim huge texts for speed (only if needed)
if MAX_TEXT_LENGTH_FOR_SPEED is not None:
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(lambda x: x[:MAX_TEXT_LENGTH_FOR_SPEED])

print("After cleaning shape:", df.shape)
print(df.head())

# =========================
# 4. BASIC TEXT CLEANING
# =========================
# NOTE:
# We are NOT aggressively cleaning because punctuation/style matters for human-vs-AI detection.
# So we only normalize spaces.

def basic_clean_text(text):
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(basic_clean_text)

# =========================
# 5. STYLOMETRIC FEATURE EXTRACTION
# =========================
stop_words = set(stopwords.words("english"))

def safe_div(a, b):
    return a / b if b != 0 else 0

def extract_stylometric_features(text):
    text = str(text)

    # Tokenization
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    # Basic counts
    char_count = len(text)
    char_count_no_spaces = len(text.replace(" ", ""))
    word_count = len(words)
    sentence_count = len(sentences)

    # Clean word list (alphabetic words only for some metrics)
    alpha_words = [w for w in words if w.isalpha()]
    alpha_word_count = len(alpha_words)

    # Word lengths
    avg_word_length = np.mean([len(w) for w in alpha_words]) if alpha_word_count > 0 else 0

    # Sentence lengths
    sentence_lengths = [len(word_tokenize(sent)) for sent in sentences] if sentence_count > 0 else [0]
    avg_sentence_length = np.mean(sentence_lengths) if len(sentence_lengths) > 0 else 0
    std_sentence_length = np.std(sentence_lengths) if len(sentence_lengths) > 0 else 0

    # Unique words / lexical diversity
    lower_alpha_words = [w.lower() for w in alpha_words]
    unique_words = len(set(lower_alpha_words))
    type_token_ratio = safe_div(unique_words, alpha_word_count)

    # Stopwords
    stopword_count = sum(1 for w in lower_alpha_words if w in stop_words)
    stopword_ratio = safe_div(stopword_count, alpha_word_count)

    # Uppercase
    uppercase_chars = sum(1 for c in text if c.isupper())
    uppercase_ratio = safe_div(uppercase_chars, char_count)

    # Digits
    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = safe_div(digit_count, char_count)

    # Punctuation counts
    punctuation_count = sum(1 for c in text if c in string.punctuation)
    punctuation_ratio = safe_div(punctuation_count, char_count)

    comma_count = text.count(",")
    period_count = text.count(".")
    semicolon_count = text.count(";")
    colon_count = text.count(":")
    exclamation_count = text.count("!")
    question_count = text.count("?")
    quote_count = text.count('"') + text.count("'")
    hyphen_count = text.count("-")
    parenthesis_count = text.count("(") + text.count(")")

    # Paragraph-ish features
    newline_count = text.count("\n")

    # Long / short words
    long_word_count = sum(1 for w in alpha_words if len(w) >= 7)
    short_word_count = sum(1 for w in alpha_words if len(w) <= 3)
    long_word_ratio = safe_div(long_word_count, alpha_word_count)
    short_word_ratio = safe_div(short_word_count, alpha_word_count)

    # Average chars per sentence
    avg_chars_per_sentence = safe_div(char_count, sentence_count)

    # Readability features
    try:
        flesch_reading_ease = textstat.flesch_reading_ease(text)
    except:
        flesch_reading_ease = 0

    try:
        flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    except:
        flesch_kincaid_grade = 0

    try:
        gunning_fog = textstat.gunning_fog(text)
    except:
        gunning_fog = 0

    try:
        smog_index = textstat.smog_index(text)
    except:
        smog_index = 0

    # Repetition
    word_freq = {}
    for w in lower_alpha_words:
        word_freq[w] = word_freq.get(w, 0) + 1
    repeated_words = sum(1 for v in word_freq.values() if v > 1)
    repeated_word_ratio = safe_div(repeated_words, unique_words)

    # Average punctuation per sentence
    avg_punct_per_sentence = safe_div(punctuation_count, sentence_count)

    # Features list
    features = [
        char_count,
        char_count_no_spaces,
        word_count,
        sentence_count,
        alpha_word_count,
        avg_word_length,
        avg_sentence_length,
        std_sentence_length,
        unique_words,
        type_token_ratio,
        stopword_count,
        stopword_ratio,
        uppercase_chars,
        uppercase_ratio,
        digit_count,
        digit_ratio,
        punctuation_count,
        punctuation_ratio,
        comma_count,
        period_count,
        semicolon_count,
        colon_count,
        exclamation_count,
        question_count,
        quote_count,
        hyphen_count,
        parenthesis_count,
        newline_count,
        long_word_count,
        short_word_count,
        long_word_ratio,
        short_word_ratio,
        avg_chars_per_sentence,
        flesch_reading_ease,
        flesch_kincaid_grade,
        gunning_fog,
        smog_index,
        repeated_words,
        repeated_word_ratio,
        avg_punct_per_sentence
    ]

    return features

print("Extracting stylometric features...")
tqdm.pandas()
stylometric_features = df[TEXT_COLUMN].progress_apply(extract_stylometric_features)

X_style = np.array(stylometric_features.tolist(), dtype=np.float32)
y = df[LABEL_COLUMN].values

print("Stylometric feature shape:", X_style.shape)

# =========================
# 6. TRAIN / TEST SPLIT
# =========================
X_text_train, X_text_test, X_style_train, X_style_test, y_train, y_test = train_test_split(
    df[TEXT_COLUMN].values,
    X_style,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train size:", len(X_text_train))
print("Test size:", len(X_text_test))

# =========================
# 7. TF-IDF FEATURES
# =========================
print("Building TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_tfidf_train = tfidf.fit_transform(X_text_train)
X_tfidf_test = tfidf.transform(X_text_test)

print("TF-IDF train shape:", X_tfidf_train.shape)
print("TF-IDF test shape:", X_tfidf_test.shape)

# =========================
# 8. SCALE STYLOMETRIC FEATURES
# =========================
scaler = StandardScaler()
X_style_train_scaled = scaler.fit_transform(X_style_train)
X_style_test_scaled = scaler.transform(X_style_test)

# =========================
# 9. BUILD HYBRID ANN MODEL (2-BRANCH)
# =========================
# Branch 1: TF-IDF
tfidf_input = Input(shape=(X_tfidf_train.shape[1],), name="tfidf_input")
x1 = Dense(512, activation="relu")(tfidf_input)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.4)(x1)

x1 = Dense(256, activation="relu")(x1)
x1 = BatchNormalization()(x1)
x1 = Dropout(0.3)(x1)

x1 = Dense(128, activation="relu")(x1)

# Branch 2: Stylometric
style_input = Input(shape=(X_style_train_scaled.shape[1],), name="style_input")
x2 = Dense(64, activation="relu")(style_input)
x2 = BatchNormalization()(x2)
x2 = Dropout(0.2)(x2)

x2 = Dense(32, activation="relu")(x2)

# Merge
combined = Concatenate()([x1, x2])

x = Dense(128, activation="relu")(combined)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=[tfidf_input, style_input], outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

print("\nModel Summary:")
model.summary()

# =========================
# 10. CALLBACKS
# =========================
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# =========================
# 11. TRAIN MODEL
# =========================
print("\nTraining model...")

history = model.fit(
    [X_tfidf_train.toarray(), X_style_train_scaled],
    y_train,
    validation_split=0.1,
    epochs=15,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# =========================
# 12. EVALUATE MODEL
# =========================
print("\nEvaluating on test set...")
test_results = model.evaluate(
    [X_tfidf_test.toarray(), X_style_test_scaled],
    y_test,
    verbose=0
)

print("\nTest Results:")
for name, value in zip(model.metrics_names, test_results):
    print(f"{name}: {value:.4f}")

# =========================
# 13. PREDICTIONS
# =========================
y_prob = model.predict([X_tfidf_test.toarray(), X_style_test_scaled]).ravel()
y_pred = (y_prob >= 0.5).astype(int)

# =========================
# 14. METRICS
# =========================
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {auc:.4f}")

# =========================
# 15. CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# =========================
# 16. TRAINING CURVES
# =========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 17. SAVE MODEL + PREPROCESSORS
# =========================
print("\nSaving model and preprocessors...")

model.save("human_vs_ai_ann_model.h5")

import pickle

with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("style_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved:")
print("- human_vs_ai_ann_model.h5")
print("- tfidf_vectorizer.pkl")
print("- style_scaler.pkl")

print("\nDone! Model training pipeline complete.")
