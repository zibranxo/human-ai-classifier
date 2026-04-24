import pandas as pd
import numpy as np
import re
import string
import math
import multiprocessing as mp
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# =========================
# DOWNLOAD NLTK DATA
# =========================
nltk.download('punkt', quiet=True)

# For some newer NLTK versions:
try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass


# =========================
# GLOBAL RESOURCES
# =========================
vader_analyzer = SentimentIntensityAnalyzer()

stopwords_set = set([
    'the','is','in','and','to','of','a','for','on','with','that','this','it',
    'as','an','are','was','were','be','by','at','or','from','but','not','have',
    'has','had','they','their','them','he','she','his','her','you','your','we',
    'our','us','i','me','my','mine','yours','hers','ours','theirs'
])

pronouns_set = set([
    'i','me','my','mine','you','your','yours','he','him','his','she','her','hers',
    'it','its','we','us','our','ours','they','them','their','theirs'
])

first_person_pronouns = set([
    'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'
])

conjunctions_set = set([
    'and','but','or','so','because','although','though','while','whereas','if','unless',
    'since','until','before','after','once'
])

transition_words_set = set([
    'however','therefore','moreover','furthermore','thus','consequently',
    'nevertheless','meanwhile','additionally','instead','otherwise','similarly',
    'likewise','hence'
])

modal_verbs_set = set([
    'can','could','may','might','must','shall','should','will','would'
])

passive_clues_set = set([
    'is','are','was','were','be','been','being'
])

common_contractions = [
    "n't", "'re", "'ve", "'ll", "'d", "'m", "'s"
]

adverb_suffixes = ('ly',)

complex_verbs_set = set([
    'demonstrate', 'facilitate', 'establish', 'indicate', 'suggest',
    'highlight', 'illustrate', 'implement', 'optimize', 'generate',
    'construct', 'analyze', 'synthesize', 'evaluate'
])

abstract_noun_suffixes = (
    'tion', 'sion', 'ment', 'ness', 'ity', 'ism', 'ship', 'ance', 'ence'
)

sophisticated_adjective_set = set([
    'significant', 'substantial', 'comprehensive', 'innovative', 'efficient',
    'robust', 'dynamic', 'critical', 'essential', 'notable', 'complex',
    'advanced', 'strategic', 'systematic', 'reliable'
])

vowels = set('aeiou')


# =========================
# HELPER FUNCTIONS
# =========================
def safe_sent_tokenize(text):
    try:
        sents = sent_tokenize(text)
        return [s.strip() for s in sents if s.strip()]
    except:
        sents = re.split(r'[.!?]+', text)
        return [s.strip() for s in sents if s.strip()]

def safe_word_tokenize(text):
    try:
        words = word_tokenize(text)
        return words
    except:
        return re.findall(r"\b\w+(?:'\w+)?\b", text)

def syllable_count(word):
    word = word.lower()
    word = re.sub(r'[^a-z]', '', word)
    if not word:
        return 0

    count = 0
    prev_char_was_vowel = False

    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                count += 1
            prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False

    # silent e adjustment
    if word.endswith('e') and count > 1:
        count -= 1

    return max(1, count)

def is_complex_word(word):
    return syllable_count(word) >= 3

def shannon_entropy(items):
    if not items:
        return 0.0
    total = len(items)
    counts = Counter(items)
    entropy = 0.0
    for freq in counts.values():
        p = freq / total
        entropy -= p * math.log2(p)
    return entropy

def compute_yules_k(words):
    """
    Yule's K measure of lexical richness
    K = 10^4 * (M2 - M1) / M1^2
    where:
    M1 = total tokens
    M2 = sum(i^2 * V_i), V_i = number of types occurring i times
    """
    if not words:
        return 0.0

    freq = Counter(words)
    freq_of_freq = Counter(freq.values())

    M1 = len(words)
    M2 = sum((i ** 2) * v_i for i, v_i in freq_of_freq.items())

    if M1 == 0:
        return 0.0

    return 10000 * (M2 - M1) / (M1 ** 2)

def detect_passive_voice(sentences):
    """
    Very heuristic passive detection:
    looks for be-verb + past participle-ish pattern
    """
    passive_count = 0

    for sent in sentences:
        tokens = [w.lower() for w in safe_word_tokenize(sent)]
        for i in range(len(tokens) - 1):
            if tokens[i] in passive_clues_set:
                nxt = tokens[i + 1]
                # crude past participle heuristic
                if nxt.endswith('ed') or nxt in {
                    'given', 'known', 'seen', 'made', 'taken', 'done',
                    'written', 'built', 'found', 'used', 'created'
                }:
                    passive_count += 1
                    break

    return passive_count

def count_contractions(text):
    count = 0
    lower_text = text.lower()
    for c in common_contractions:
        count += lower_text.count(c)
    return count

def count_adverbs(words):
    return sum(1 for w in words if w.endswith(adverb_suffixes))

def count_complex_verbs(words):
    return sum(1 for w in words if w in complex_verbs_set)

def count_abstract_nouns(words):
    return sum(1 for w in words if w.endswith(abstract_noun_suffixes))

def count_sophisticated_adjectives(words):
    return sum(1 for w in words if w in sophisticated_adjective_set)

def trigram_uniqueness(words):
    if len(words) < 3:
        return 0.0
    trigrams = list(zip(words, words[1:], words[2:]))
    unique_trigrams = len(set(trigrams))
    return unique_trigrams / len(trigrams)

def syntax_variety(sentences):
    """
    Heuristic syntax variety:
    Count how many sentence types appear:
    declarative, interrogative, exclamatory, colon/semicolon rich, short fragments
    Return ratio of unique sentence style categories / 5
    """
    if not sentences:
        return 0.0

    categories = set()

    for s in sentences:
        stripped = s.strip()
        if not stripped:
            continue

        if stripped.endswith('?'):
            categories.add('question')
        elif stripped.endswith('!'):
            categories.add('exclamation')
        else:
            categories.add('declarative')

        if ':' in stripped or ';' in stripped:
            categories.add('complex_punct')

        word_count = len(re.findall(r'\b\w+\b', stripped))
        if word_count <= 5:
            categories.add('short_fragment')

    return len(categories) / 5.0

def gunning_fog_index(words, sentences):
    if not words or not sentences:
        return 0.0

    total_words = len(words)
    total_sentences = len(sentences)

    complex_words = sum(1 for w in words if is_complex_word(w))
    avg_sentence_len = total_words / total_sentences
    pct_complex = (complex_words / total_words) * 100

    return 0.4 * (avg_sentence_len + pct_complex)


# =========================
# MAIN FEATURE EXTRACTION
# =========================
def extract_stylometric_features(text):
    if not isinstance(text, str):
        text = str(text)

    text = text.replace('\n', ' ').replace('\r', ' ').strip()

    chars = len(text)
    sentences = safe_sent_tokenize(text)
    words_raw = safe_word_tokenize(text)

    # Keep alphabetic + apostrophe style tokens
    words = [w.lower() for w in words_raw if re.match(r"^[A-Za-z]+(?:'[A-Za-z]+)?$", w)]
    num_words = len(words)
    num_sentences = len(sentences)

    if num_words == 0:
        num_words = 1
    if num_sentences == 0:
        num_sentences = 1

    unique_words = len(set(words))
    word_lengths = [len(w) for w in words] if words else [0]
    sentence_word_counts = [
        len([w for w in safe_word_tokenize(s) if re.match(r"^[A-Za-z]+(?:'[A-Za-z]+)?$", w)])
        for s in sentences
    ] if sentences else [0]

    # Punctuation
    comma_count = text.count(',')
    period_count = text.count('.')
    question_count = text.count('?')
    exclamation_count = text.count('!')
    semicolon_count = text.count(';')
    colon_count = text.count(':')
    quote_count = text.count('"') + text.count("'")
    punctuation_count = sum(1 for c in text if c in string.punctuation)

    # Character categories
    uppercase_count = sum(1 for c in text if c.isupper())
    digit_count = sum(1 for c in text if c.isdigit())
    whitespace_count = sum(1 for c in text if c.isspace())

    # Lexical stats
    word_freq = Counter(words)
    hapax_count = sum(1 for w, c in word_freq.items() if c == 1)
    repeated_words = sum(c for w, c in word_freq.items() if c > 1)

    # Ratios / counts
    stopword_count = sum(1 for w in words if w in stopwords_set)
    pronoun_count = sum(1 for w in words if w in pronouns_set)
    first_person_pronoun_count = sum(1 for w in words if w in first_person_pronouns)
    conjunction_count = sum(1 for w in words if w in conjunctions_set)
    transition_count = sum(1 for w in words if w in transition_words_set)
    modal_count = sum(1 for w in words if w in modal_verbs_set)
    passive_count = sum(1 for w in words if w in passive_clues_set)

    long_word_count = sum(1 for w in words if len(w) > 6)
    short_word_count = sum(1 for w in words if len(w) <= 3)

    # Advanced features
    yules_k = compute_yules_k(words)
    word_length_variance = float(np.var(word_lengths)) if word_lengths else 0.0
    char_entropy = shannon_entropy(list(text)) if text else 0.0
    word_entropy = shannon_entropy(words) if words else 0.0

    passive_voice_count = detect_passive_voice(sentences)
    passive_voice_rate = passive_voice_count / num_sentences if num_sentences > 0 else 0.0

    gunning_fog = gunning_fog_index(words, sentences)

    contraction_count = count_contractions(text)
    adverb_count = count_adverbs(words)
    complex_verb_count = count_complex_verbs(words)
    abstract_noun_count = count_abstract_nouns(words)
    sophisticated_adjective_count = count_sophisticated_adjectives(words)

    trigram_unique_ratio = trigram_uniqueness(words)
    syntax_variety_score = syntax_variety(sentences)

    vader_compound_score = vader_analyzer.polarity_scores(text)['compound'] if text else 0.0

    # Final feature dict
    features = {
        'char_count': chars,
        'word_count': num_words,
        'sentence_count': num_sentences,
        'avg_word_length': float(np.mean(word_lengths)) if word_lengths else 0.0,
        'avg_sentence_length': float(np.mean(sentence_word_counts)) if sentence_word_counts else 0.0,
        'sentence_length_std': float(np.std(sentence_word_counts)) if sentence_word_counts else 0.0,
        'unique_words': unique_words,
        'type_token_ratio': unique_words / num_words if num_words > 0 else 0.0,
        'hapax_ratio': hapax_count / num_words if num_words > 0 else 0.0,

        'comma_count': comma_count,
        'period_count': period_count,
        'question_count': question_count,
        'exclamation_count': exclamation_count,
        'semicolon_count': semicolon_count,
        'colon_count': colon_count,
        'quote_count': quote_count,
        'punctuation_density': punctuation_count / chars if chars > 0 else 0.0,

        'uppercase_ratio': uppercase_count / chars if chars > 0 else 0.0,
        'digit_ratio': digit_count / chars if chars > 0 else 0.0,
        'whitespace_ratio': whitespace_count / chars if chars > 0 else 0.0,

        'stopword_ratio': stopword_count / num_words if num_words > 0 else 0.0,
        'pronoun_ratio': pronoun_count / num_words if num_words > 0 else 0.0,
        'conjunction_ratio': conjunction_count / num_words if num_words > 0 else 0.0,
        'transition_ratio': transition_count / num_words if num_words > 0 else 0.0,
        'modal_ratio': modal_count / num_words if num_words > 0 else 0.0,

        'passive_ratio': passive_count / num_words if num_words > 0 else 0.0,

        'long_word_ratio': long_word_count / num_words if num_words > 0 else 0.0,
        'short_word_ratio': short_word_count / num_words if num_words > 0 else 0.0,
        'repeated_word_ratio': repeated_words / num_words if num_words > 0 else 0.0,

        'yules_k': yules_k,
        'word_length_variance': word_length_variance,
        'char_entropy': char_entropy,
        'word_entropy': word_entropy,

        'passive_voice_rate': passive_voice_rate,
        'punctuation_count': punctuation_count,

        'gunning_fog': gunning_fog,

        'first_person_pronoun_count': first_person_pronoun_count,
        'contraction_count': contraction_count,

        'adverb_count': adverb_count,
        'complex_verb_count': complex_verb_count,
        'abstract_noun_count': abstract_noun_count,
        'sophisticated_adjective_count': sophisticated_adjective_count,

        'trigram_uniqueness': trigram_unique_ratio,
        'syntax_variety': syntax_variety_score,

        'vader_compound_score': vader_compound_score
    }

    return features


# =========================
# MULTIPROCESSING FEATURE EXTRACTION
# =========================
def extract_features_parallel(texts, n_processes=None):
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)

    print(f"\nUsing multiprocessing with {n_processes} processes...")

    with mp.Pool(processes=n_processes) as pool:
        features_list = pool.map(extract_stylometric_features, texts)

    return pd.DataFrame(features_list)


# =========================
# MAIN PIPELINE
# =========================
def main():
    # =========================
    # CHANGE THIS PATH
    # =========================
    file_path = "your_dataset.csv"   # <-- CHANGE THIS

    # =========================
    # LOAD DATA
    # =========================
    print("Loading dataset...")
    df = pd.read_csv("finaldatset.csv")

    print("\nColumns in dataset:", df.columns.tolist())
    print("Shape:", df.shape)

    # =========================
    # REQUIRED COLUMNS
    # =========================
    # Change these if needed
    text_col = "text"
    label_col = "label_id"

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Dataset must contain columns: '{text_col}' and '{label_col}'")

    df = df[[text_col, label_col]].copy()
    df = df.dropna(subset=[text_col, label_col])

    df[text_col] = df[text_col].astype(str)
    df[text_col] = df[text_col].str.replace('Â', '', regex=False)
    df[text_col] = df[text_col].str.replace('\n', ' ', regex=False)
    df[text_col] = df[text_col].str.strip()

    # Optional: remove very short texts
    df["word_len_temp"] = df[text_col].apply(lambda x: len(re.findall(r'\b\w+\b', str(x))))
    df = df[df["word_len_temp"] >= 5].copy()
    df.drop(columns=["word_len_temp"], inplace=True)

    print("\nAfter cleaning shape:", df.shape)
    print("\nClass distribution:")
    print(df[label_col].value_counts())

    # =========================
    # FEATURE EXTRACTION
    # =========================
    print("\nExtracting stylometric features...")
    X = extract_features_parallel(df[text_col].tolist())

    y = df[label_col].values

    print("\nExtracted feature shape:", X.shape)
    print("\nFeature columns:")
    print(X.columns.tolist())

    # Save extracted features
    feature_output = X.copy()
    feature_output[label_col] = y
    feature_output.to_csv("extracted_stylometric_features.csv", index=False)
    print("\nSaved: extracted_stylometric_features.csv")

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("\nTrain shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    # =========================
    # CHI-SQUARE FEATURE SELECTION
    # =========================
    # chi2 requires non-negative features
    # Our features are mostly non-negative EXCEPT vader_compound_score can be negative
    # So shift all features to non-negative
    X_train_chi = X_train.copy()
    X_test_chi = X_test.copy()

    for col in X_train_chi.columns:
        min_val = min(X_train_chi[col].min(), X_test_chi[col].min())
        if min_val < 0:
            shift = abs(min_val) + 1e-9
            X_train_chi[col] = X_train_chi[col] + shift
            X_test_chi[col] = X_test_chi[col] + shift

    k = min(20, X_train_chi.shape[1])  # top 20 or less if fewer features
    selector = SelectKBest(score_func=chi2, k=k)

    X_train_selected = selector.fit_transform(X_train_chi, y_train)
    X_test_selected = selector.transform(X_test_chi)

    selected_features = X_train.columns[selector.get_support()]
    chi_scores = selector.scores_

    chi_result = pd.DataFrame({
        'Feature': X_train.columns,
        'Chi2 Score': chi_scores
    }).sort_values(by='Chi2 Score', ascending=False)

    print("\nTop Chi-square Features:")
    print(chi_result.head(20))

    print("\nSelected Features:")
    print(selected_features.tolist())

    chi_result.to_csv("chi2_feature_scores.csv", index=False)
    print("\nSaved: chi2_feature_scores.csv")

    # =========================
    # DECISION TREE MODEL
    # =========================
    clf = DecisionTreeClassifier(
        criterion='gini',
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    clf.fit(X_train_selected, y_train)
    y_pred = clf.predict(X_test_selected)

    # =========================
    # EVALUATION
    # =========================
    acc = accuracy_score(y_test, y_pred)

    print("\n" + "="*50)
    print("DECISION TREE RESULTS")
    print("="*50)
    print(f"Accuracy: {acc:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    importances = clf.feature_importances_

    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\nDecision Tree Feature Importances:")
    print(importance_df)

    importance_df.to_csv("decision_tree_feature_importance.csv", index=False)
    print("\nSaved: decision_tree_feature_importance.csv")

    # =========================
    # PLOT FEATURE IMPORTANCE
    # =========================
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'])
    plt.gca().invert_yaxis()
    plt.title("Decision Tree Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("decision_tree_feature_importance.png", dpi=300)
    plt.show()

    # =========================
    # PLOT DECISION TREE
    # =========================
    plt.figure(figsize=(24, 12))
    plot_tree(
        clf,
        feature_names=selected_features,
        class_names=['Class 0', 'Class 1'],
        filled=True,
        rounded=True,
        fontsize=8
    )
    plt.title("Decision Tree (Selected Stylometric Features)")
    plt.tight_layout()
    plt.savefig("decision_tree_plot.png", dpi=300)
    plt.show()

    print("\nSaved: decision_tree_feature_importance.png")
    print("Saved: decision_tree_plot.png")

    print("\nPipeline completed successfully!")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    mp.freeze_support()  # important for Windows
    main()
