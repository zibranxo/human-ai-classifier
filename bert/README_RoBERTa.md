# RoBERTa-based AI vs Human Text Classifier (Production-Grade Pipeline)

## 📌 Overview

This project implements a **state-of-the-art transformer-based classifier** using **RoBERTa (`roberta-base`)** to distinguish between **AI-generated and human-written text**.

Unlike traditional ML approaches, this system leverages:

* **Pretrained transformer embeddings**
* **Fine-tuning on labeled data**
* **Advanced training strategies**
* **Comprehensive evaluation & visualization suite**

Built using:

* HuggingFace Transformers
* PyTorch
* Custom training callbacks
* Rich diagnostic plotting pipeline

---

## 🧠 Model Architecture

### Base Model:

* `roberta-base` (125M parameters)
* Pretrained on large-scale corpora
* Fine-tuned for binary classification

### Classification Head:

```text
RoBERTa Encoder → Dense Layer → Softmax (2 classes)
```

### Labels:

* `0 → HUMAN`
* `1 → AI`

---

## ⚙️ Methodology

### 1. Dataset Processing

* Input file: `input_final.csv`
* Columns:

  * `text`: input text
  * `label`: binary label

#### Cleaning:

* Remove null values
* Strip whitespace
* Filter short texts (`len > 10`)

---

### 2. Exploratory Data Analysis (EDA)

Generates a **6-panel visualization** including:

* Class distribution
* Character length distribution
* Word count distribution
* Box plots (length vs class)
* Cumulative token coverage
* Dataset summary table

---

### 3. Data Splitting

* **Train: 80%**
* **Validation: 10%**
* **Test: 10%**
* Stratified sampling ensures class balance

---

### 4. Tokenization

* Tokenizer: `AutoTokenizer.from_pretrained("roberta-base")`
* Max sequence length: `128`
* Dynamic padding using `DataCollatorWithPadding`

---

## 🔁 Training Strategy

### Hyperparameters:

* Epochs: `3`
* Batch size: `32`
* Gradient accumulation: `2` → effective batch = 64
* Learning rate: `2e-5`
* Warmup: `5%`
* Weight decay: `0.01`

---

### Optimization:

* **AdamW (fused)**
* **OneCycle-style LR scheduling**
* Mixed precision:

  * `fp16 = True`
  * `tf32 = True`

---

### ⚡ Performance Optimizations

* Gradient checkpointing (reduces memory)
* CUDA optimizations (TF32 + cuDNN tuning)
* Multi-worker dataloading
* Torch compile (disabled on Windows)

---

### ⏹️ Early Stopping

* Stops training if validation performance stagnates
* Metric: **F1 Score**
* Patience: `3 evaluations`

---

## 📊 Evaluation Metrics

* Accuracy
* Weighted F1 Score
* Class-wise F1:

  * F1 (AI)
  * F1 (Human)

---

## 📈 Advanced Diagnostics & Visualizations

This project includes a **full ML observability suite**:

### 📉 Training Monitoring

* Loss curves (smoothed + raw)
* Validation accuracy
* F1 score tracking
* Learning rate schedule
* Loss gap (overfitting indicator)
* Epoch time tracking

---

### 📊 Evaluation Plots

1. Confusion Matrix (raw + normalized)
2. ROC Curve + AUC
3. Precision-Recall Curve + AP
4. Threshold vs Metrics analysis
5. Confidence distribution (correct vs wrong)
6. Calibration (reliability diagram)
7. Per-class metrics (bar + radar chart)

---

### 🧠 Model Behavior Insights

* Threshold optimization using **Youden’s J statistic**
* Confidence calibration analysis
* Class-wise performance breakdown

---

### 💾 System Monitoring

* GPU VRAM tracking during training
* Step-wise memory usage logging

---

### 🔥 Checkpoint Analysis

* Heatmap of all checkpoints
* Tracks:

  * Loss
  * Accuracy
  * F1 (overall + per class)
* Automatically highlights best checkpoint

---

## 🛠️ Implementation Highlights

### Custom Trainer Callbacks:

* **LiveProgressCallback**

  * Real-time progress bar
  * Tracks training + eval metrics
  * Saves logs as JSON

* **VRAMMonitorCallback**

  * Tracks GPU memory usage

---

### Logging & Persistence:

* Model saved to: `./ai-detector-200k/`
* Plots saved to: `./ai-detector-200k/plots/`
* Training logs: `training_log.json`
* Config snapshot: `run_config.json`

---

## 🚀 How to Run

### 1. Install dependencies

```bash id="r8k2mz"
pip install torch transformers datasets scikit-learn pandas matplotlib seaborn
```

---

### 2. Prepare dataset

Place:

```text
input_final.csv
```

---

### 3. Run training

```bash id="k3x9qp"
python train.py
```

---

## 📂 Output Structure

```text
ai-detector-200k/
│
├── config.json
├── pytorch_model.bin
├── tokenizer/
├── run_config.json
│
└── plots/
    ├── 01_eda.png
    ├── 02_training_curves.png
    ├── 03_confusion_matrix.png
    ├── 04_roc_curve.png
    ├── 05_precision_recall.png
    ├── 06_confidence_calibration.png
    ├── 07_per_class_metrics.png
    ├── 08_vram_usage.png
    └── 09_checkpoint_heatmap.png
```

---

## 📊 Strengths & Limitations

### ✅ Strengths

* Uses **state-of-the-art transformer architecture**
* Highly optimized training pipeline
* Extensive evaluation and diagnostics
* Handles real-world large-scale datasets
* Production-ready logging and monitoring

---

### ❌ Limitations

* Requires GPU for efficient training
* High computational cost
* Sequence length limited to 128 tokens
* Interpretability is lower than classical ML

---

## 🔮 Future Improvements

* Use larger models (RoBERTa-large, DeBERTa)
* Increase max sequence length
* Add adversarial training (for robustness)
* Deploy as API (FastAPI / Flask)
* Integrate with real-time detection systems
* Add explainability (LIME / SHAP)

---

## 🎯 Key Takeaways

* Transformer models capture **deep semantic patterns** in text
* Proper training strategy + monitoring = huge performance gains
* This pipeline goes beyond modeling → **full ML system design**

👉 This is not just a classifier — it's a **complete AI detection system**.

---
