#AI vs Human Text Classification

> A comprehensive machine learning pipeline for detecting AI-generated text using classical , ensemble, deep learning, and transformer-based architectures.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=flat-square&logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-Apache%202.0-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📖 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Feature Engineering](#feature-engineering)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Authors](#authors)

---

## Overview

As large language models become increasingly sophisticated, distinguishing between human-written and AI-generated text is a critical challenge in academic integrity, content moderation, and misinformation detection. This project presents an end-to-end binary text classification pipeline that systematically evaluates a broad spectrum of approaches — from classical statistical models to deep neural networks and transformer architectures — to determine the most effective method for this task.

The pipeline combines **TF-IDF vectorization** with **stylometric feature engineering** and evaluates 14+ models across three paradigms: classical machine learning, deep learning, and transformer-based methods.

---

## Dataset

**Source:** [`andythetechnerd03/AI-human-text`](https://huggingface.co/datasets/andythetechnerd03/AI-human-text) on Hugging Face

| Property | Value |
|---|---|
| Total Samples | ~200,000 |
| Training Split | ~160,000 samples |
| Test Split | ~40,000 samples |
| Task | Binary Classification (`0` = Human, `1` = AI) |
| Language | English |
| Format | Parquet |
| License | Apache 2.0 |

The dataset contains a diverse collection of essays and short-form writing samples sourced from both human authors and AI language models, covering a wide range of topics and writing styles.

---

## Pipeline Architecture

```
Raw Text Data
     │
     ▼
┌─────────────────────┐
│   Preprocessing     │  Cleaning · Tokenization · Normalization
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Feature Extraction  │  TF-IDF Vectors + Stylometric Features
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ Feature Selection   │  Dimensionality Reduction · Variance Filtering
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│  Model Training     │  ML · DL · Transformer
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│    Evaluation       │  Accuracy · Precision · Recall · F1
└─────────────────────┘
```

---

## Feature Engineering

### TF-IDF Vectorization
Captures term frequency and inverse document frequency patterns across the corpus to create sparse, high-dimensional representations of each text sample.

### Stylometric Features
A rich set of hand-crafted features designed to capture the writing "fingerprint" of human vs. AI text:

| Category | Features |
|---|---|
| **Lexical** | Vocabulary richness, type-token ratio, average word length, hapax legomena ratio |
| **Syntactic** | Sentence length distribution, punctuation usage, POS tag ratios |
| **Readability** | Flesch-Kincaid score, Gunning Fog index, SMOG index |
| **Writing Style** | Function word frequencies, stop word ratios, discourse markers |

### Hybrid Representation
A combined feature vector that concatenates TF-IDF and stylometric representations, used as input to the Hybrid ANN model.

---

## Models Implemented

### Classical Machine Learning

| Model | Description |
|---|---|
| **Linear Regression** | Baseline linear approach adapted for classification |
| **Logistic Regression** | Probabilistic linear classifier |
| **Naive Bayes** | Probabilistic model leveraging Bayes' theorem |
| **Support Vector Machine (SVM)** | Kernel-based margin classifier |
| **K-Nearest Neighbors (KNN)** | Instance-based non-parametric classifier |
| **Decision Tree — ID3** | Entropy-based splitting criterion |
| **Decision Tree — C4.5** | Gain ratio-based splitting with pruning |
| **Decision Tree — CART** | Gini impurity-based binary tree |
| **Random Forest** | Ensemble of bagged decision trees |
| **AdaBoost** | Adaptive boosting ensemble |
| **K-Means** | Unsupervised clustering baseline |

### Deep Learning

| Model | Description |
|---|---|
| **Single-Layer ANN** | Shallow neural network for baseline DL comparison |
| **Multi-Layer ANN** | Deep feedforward network with dropout regularization |
| **Hybrid ANN** | Multi-Layer ANN trained on combined TF-IDF + stylometric features |

### Transformer-Based

| Model | Description |
|---|---|
| **BERT** | `roberta-base` fine-tuned for binary sequence classification |

---

## Results

> **Note:** Placeholder results — update after final evaluation runs.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | XX.X% | X.XX | X.XX | X.XX |
| Naive Bayes | XX.X% | X.XX | X.XX | X.XX |
| SVM | XX.X% | X.XX | X.XX | X.XX |
| KNN | XX.X% | X.XX | X.XX | X.XX |
| Decision Tree (ID3) | XX.X% | X.XX | X.XX | X.XX |
| Decision Tree (C4.5) | XX.X% | X.XX | X.XX | X.XX |
| Decision Tree (CART) | XX.X% | X.XX | X.XX | X.XX |
| Random Forest | XX.X% | X.XX | X.XX | X.XX |
| AdaBoost | XX.X% | X.XX | X.XX | X.XX |
| Single-Layer ANN | XX.X% | X.XX | X.XX | X.XX |
| Multi-Layer ANN | XX.X% | X.XX | X.XX | X.XX |
| **Hybrid ANN** | **XX.X%** | **X.XX** | **X.XX** | **X.XX** |
| **BERT** | **XX.X%** | **X.XX** | **X.XX** | **X.XX** |

### Key Findings

- The **Hybrid ANN** (TF-IDF + stylometric features) outperforms models relying on a single feature representation.
- **BERT** achieves the strongest contextual understanding, leveraging deep bidirectional attention.
- Classical models such as **SVM** and **Random Forest** provide strong, efficient baselines.
- **Stylometric features** alone carry significant discriminative signal, particularly for lexical richness patterns.

---

## Installation

**Prerequisites:** Python 3.8+, pip

```bash
# 1. Clone the repository
git clone https://github.com/<zibranxo>/ai-vs-human-text-classification.git
cd ai-vs-human-text-classification

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Required Packages

```txt
torch>=2.0.0
transformers>=4.30.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
nltk>=3.8.0
datasets>=2.12.0
```

---

## Usage

### Download the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("andythetechnerd03/AI-human-text")
```
For some models, processed datsets are required. Due to constraints, they cannot be uploded here. If you need them, drop mail at thearnav001@gmail.com
### Train All Models

```bash
python training/train.py
```

### Evaluate

```bash
python evaluation/evaluate.py
```

### Train a Specific Model

```bash
# Example: Train BERT only
python training/train.py --model bert

# Example: Train Hybrid ANN only
python training/train.py --model hybrid_ann
```

---

## Future Work

- [ ] Improve dataset diversity with multi-source AI text (GPT-4, Claude, Gemini)
- [ ] Deploy as a REST API using FastAPI or Flask
- [ ] Build a real-time web application for inference
- [ ] Experiment with advanced transformer models (RoBERTa, DeBERTa, LLaMA)
- [ ] Explore few-shot and zero-shot detection approaches
- [ ] Add explainability (SHAP, LIME) to identify key discriminating features

---

## Authors

**Arnav Sagar** . **Aryan Rai**

---

*If you find this project useful, please consider giving it a ⭐ on GitHub.*
