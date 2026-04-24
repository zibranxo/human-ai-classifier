import os, sys, json, time, warnings
from pathlib import Path
from datetime import datetime
from copy import deepcopy

import platform
IS_WINDOWS = platform.system() == "Windows"

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                         # headless — safe on any server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay,
)
import torch
from torch.optim.lr_scheduler import OneCycleLR
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainerCallback,
)
from datasets import Dataset

import torch._dynamo
torch._dynamo.config.suppress_errors = True   # fallback to eager if compile fails

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
torch.backends.cudnn.benchmark        = True    # autotune convolution kernels
os.environ["TOKENIZERS_PARALLELISM"]  = "false"

MODEL_NAME   = "roberta-base"
MAX_LEN      = 128
BATCH_SIZE   = 32           # safe for 6 GB VRAM
GRAD_ACCUM   = 2           # effective batch = 32
EPOCHS       = 3
LR           = 2e-5
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
OUTPUT_DIR   = Path("./ai-detector-200k")
PLOTS_DIR    = OUTPUT_DIR / "plots"
SEED         = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor"  : "#0f1117",
    "axes.facecolor"    : "#1a1d27",
    "axes.edgecolor"    : "#2e3147",
    "axes.labelcolor"   : "#c8cad4",
    "axes.titlecolor"   : "#ffffff",
    "axes.grid"         : True,
    "grid.color"        : "#2e3147",
    "grid.linewidth"    : 0.6,
    "text.color"        : "#c8cad4",
    "xtick.color"       : "#8889a0",
    "ytick.color"       : "#8889a0",
    "xtick.labelsize"   : 9,
    "ytick.labelsize"   : 9,
    "axes.titlesize"    : 12,
    "axes.labelsize"    : 10,
    "legend.framealpha" : 0.3,
    "legend.edgecolor"  : "#2e3147",
    "legend.labelcolor" : "#c8cad4",
    "font.family"       : "monospace",
    "figure.dpi"        : 150,
    "savefig.dpi"       : 150,
    "savefig.bbox"      : "tight",
    "savefig.facecolor" : "#0f1117",
})

ACCENT   = "#6c8fff"
ACCENT2  = "#ff6b9d"
ACCENT3  = "#50e3c2"
ACCENT4  = "#ffb347"
DIM      = "#3a3f5c"

def gpu_info():
    if not torch.cuda.is_available():
        print("⚠  No CUDA GPU found — training on CPU (slow).")
        return
    props = torch.cuda.get_device_properties(0)
    total = props.total_memory / 1e9
    print(f"\n{'─'*55}")
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {total:.1f} GB")
    print(f"  CUDA   : {torch.version.cuda}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"{'─'*55}\n")

gpu_info()

def plot_eda(df: pd.DataFrame, save_path: Path):
    """Four-panel EDA: class balance, text-length dist, label vs length box, char dist."""
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.suptitle("Dataset Exploratory Analysis", color="#ffffff", fontsize=14, weight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    human_df = df[df["label"] == 0]["text"]
    ai_df    = df[df["label"] == 1]["text"]
    df       = df.copy()
    df["length"] = df["text"].str.len()
    df["words"]  = df["text"].str.split().str.len()

    # ── 1. Class balance bar ───────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["label"].value_counts().sort_index()
    bars   = ax1.bar(["Human", "AI"], counts.values,
                     color=[ACCENT, ACCENT2], width=0.5, edgecolor="none")
    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + counts.max()*0.01,
                 f"{val:,}", ha="center", va="bottom", fontsize=10, color="#ffffff")
    ax1.set_title("Class Balance")
    ax1.set_ylabel("Samples")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # ── 2. Character length histogram ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(human_df.str.len(), bins=60, alpha=0.7, color=ACCENT,  label="Human", edgecolor="none")
    ax2.hist(ai_df.str.len(),    bins=60, alpha=0.7, color=ACCENT2, label="AI",    edgecolor="none")
    ax2.axvline(MAX_LEN * 4, color=ACCENT4, linewidth=1.5, linestyle="--", label=f"≈MAX_LEN×4")
    ax2.set_title("Character Length Distribution")
    ax2.set_xlabel("Characters")
    ax2.set_ylabel("Frequency")
    ax2.legend(fontsize=8)

    # ── 3. Word count histogram ────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(df[df["label"]==0]["words"], bins=50, alpha=0.7, color=ACCENT,  label="Human", edgecolor="none")
    ax3.hist(df[df["label"]==1]["words"], bins=50, alpha=0.7, color=ACCENT2, label="AI",    edgecolor="none")
    ax3.set_title("Word Count Distribution")
    ax3.set_xlabel("Words")
    ax3.set_ylabel("Frequency")
    ax3.legend(fontsize=8)

    # ── 4. Box plot — length by label ─────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    data_box = [df[df["label"]==0]["length"].values, df[df["label"]==1]["length"].values]
    bp = ax4.boxplot(data_box, patch_artist=True, notch=True, widths=0.4,
                     medianprops=dict(color="#ffffff", linewidth=2))
    bp["boxes"][0].set_facecolor(ACCENT + "80")
    bp["boxes"][1].set_facecolor(ACCENT2 + "80")
    ax4.set_xticks([1, 2]); ax4.set_xticklabels(["Human", "AI"])
    ax4.set_title("Length Distribution by Class")
    ax4.set_ylabel("Characters")

    # ── 5. Cumulative length coverage ─────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    for label_val, color, name in [(0, ACCENT, "Human"), (1, ACCENT2, "AI")]:
        lens = np.sort(df[df["label"]==label_val]["words"].values)
        cdf  = np.arange(1, len(lens)+1) / len(lens)
        ax5.plot(lens, cdf, color=color, label=name, linewidth=1.5)
    ax5.axvline(MAX_LEN, color=ACCENT4, linewidth=1.5, linestyle="--", label=f"MAX_LEN={MAX_LEN}")
    ax5.set_title("Cumulative Word Coverage")
    ax5.set_xlabel("Word Count")
    ax5.set_ylabel("Fraction of Dataset")
    ax5.legend(fontsize=8)

    # ── 6. Split summary table ─────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis("off")
    stats = [
        ["Total samples", f"{len(df):,}"],
        ["Human", f"{(df['label']==0).sum():,}"],
        ["AI", f"{(df['label']==1).sum():,}"],
        ["Avg words (Human)", f"{df[df['label']==0]['words'].mean():.0f}"],
        ["Avg words (AI)", f"{df[df['label']==1]['words'].mean():.0f}"],
        ["Max char length", f"{df['length'].max():,}"],
        ["Truncated at token", str(MAX_LEN)],
    ]
    table = ax6.table(cellText=stats, colLabels=["Metric", "Value"],
                      loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#2e3147")
        cell.set_facecolor("#1a1d27" if row > 0 else "#252840")
        cell.set_text_props(color="#c8cad4")
    ax6.set_title("Dataset Summary", pad=12)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓  EDA plot saved → {save_path}")

class LiveProgressCallback(TrainerCallback):
    """
    Prints a rich per-step progress bar and collects all metrics for
    post-training plots. Saves a checkpoint JSON after every eval.
    """
    def __init__(self, total_steps: int, eval_steps: int, plots_dir: Path):
        self.total_steps  = total_steps
        self.eval_steps   = eval_steps
        self.plots_dir    = plots_dir
        self.train_log    = []   # {"step", "loss", "lr", "epoch"}
        self.eval_log     = []   # {"step", "eval_loss", "accuracy", "f1", "f1_ai", "f1_human"}
        self.best_f1      = 0.0
        self.best_step    = 0
        self.epoch_times  = []
        self._epoch_start = None
        self._step_start  = time.time()
        self._bar_width   = 40

    def _bar(self, step):
        frac  = step / max(self.total_steps, 1)
        done  = int(frac * self._bar_width)
        rest  = self._bar_width - done
        return f"[{'█' * done}{'░' * rest}] {frac*100:5.1f}%"

    def on_epoch_begin(self, args, state, control, **kw):
        self._epoch_start = time.time()
        epoch = int(state.epoch) + 1 if state.epoch else 1
        print(f"\n{'═'*60}")
        print(f"  EPOCH {epoch}/{args.num_train_epochs}")
        print(f"{'═'*60}")

    def on_epoch_end(self, args, state, control, **kw):
        elapsed = time.time() - self._epoch_start if self._epoch_start else 0
        self.epoch_times.append(elapsed)
        print(f"\n  ⏱  Epoch time: {elapsed/60:.1f} min")

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        step = state.global_step
        # ── Training step log ──────────────────────────────────────────────
        if "loss" in logs and "eval_loss" not in logs:
            loss = logs["loss"]
            lr   = logs.get("learning_rate", 0)
            ep   = logs.get("epoch", 0)
            self.train_log.append({"step": step, "loss": loss, "lr": lr, "epoch": ep})
            elapsed   = time.time() - self._step_start
            steps_rem = self.total_steps - step
            eta_min   = (elapsed / max(step, 1)) * steps_rem / 60
            bar = self._bar(step)
            print(f"\r  Step {step:>5}/{self.total_steps}  {bar}"
                  f"  loss={loss:.4f}  lr={lr:.2e}  ETA={eta_min:.0f}m",
                  end="", flush=True)

        # ── Eval log ──────────────────────────────────────────────────────
        if "eval_loss" in logs:
            print()  # newline after progress bar
            rec = {
                "step"      : step,
                "eval_loss" : logs.get("eval_loss", 0),
                "accuracy"  : logs.get("eval_accuracy", 0),
                "f1"        : logs.get("eval_f1", 0),
                "f1_ai"     : logs.get("eval_f1_ai", 0),
                "f1_human"  : logs.get("eval_f1_human", 0),
            }
            self.eval_log.append(rec)

            flag = ""
            if rec["f1"] > self.best_f1:
                self.best_f1  = rec["f1"]
                self.best_step = step
                flag = "  ★ NEW BEST"
            print(f"  EVAL @ step {step:>5}  "
                  f"loss={rec['eval_loss']:.4f}  "
                  f"acc={rec['accuracy']:.4f}  "
                  f"f1={rec['f1']:.4f}{flag}")

            # Save running log as JSON
            log_path = self.plots_dir / "training_log.json"
            with open(log_path, "w") as f:
                json.dump({"train": self.train_log, "eval": self.eval_log,
                           "best_step": self.best_step, "best_f1": self.best_f1}, f, indent=2)

    def on_train_end(self, args, state, control, **kw):
        print(f"\n\n  ✓  Training complete.")
        print(f"  ★  Best F1  : {self.best_f1:.4f}  (step {self.best_step})")
        if self.epoch_times:
            print(f"  ⏱  Total   : {sum(self.epoch_times)/60:.1f} min")

def plot_training_curves(callback: LiveProgressCallback, save_path: Path):
    train = pd.DataFrame(callback.train_log)
    evl   = pd.DataFrame(callback.eval_log)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle("Training Curves", color="#ffffff", fontsize=14, weight="bold")

    # ── 1. Training loss (smoothed) ────────────────────────────────────────
    ax = axes[0, 0]
    if not train.empty:
        ax.plot(train["step"], train["loss"], color=DIM, alpha=0.4, linewidth=0.8, label="Raw")
        smooth = train["loss"].ewm(span=20).mean()
        ax.plot(train["step"], smooth, color=ACCENT, linewidth=2, label="EMA(20)")
    if not evl.empty:
        ax.plot(evl["step"], evl["eval_loss"], "o--", color=ACCENT2, linewidth=1.5,
                markersize=5, label="Val loss")
    ax.set_title("Loss"); ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8)
    if callback.best_step:
        ax.axvline(callback.best_step, color=ACCENT4, linewidth=1, linestyle=":", alpha=0.8)

    # ── 2. Validation accuracy ─────────────────────────────────────────────
    ax = axes[0, 1]
    if not evl.empty:
        ax.plot(evl["step"], evl["accuracy"], "o-", color=ACCENT3, linewidth=2, markersize=5)
        ax.fill_between(evl["step"], evl["accuracy"], alpha=0.15, color=ACCENT3)
        best_idx = evl["accuracy"].idxmax()
        ax.scatter(evl.loc[best_idx, "step"], evl.loc[best_idx, "accuracy"],
                   color="#ffffff", s=80, zorder=5, label=f"Best {evl.loc[best_idx,'accuracy']:.4f}")
    ax.set_title("Validation Accuracy"); ax.set_xlabel("Step"); ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1); ax.legend(fontsize=8)

    # ── 3. F1 scores ──────────────────────────────────────────────────────
    ax = axes[0, 2]
    if not evl.empty:
        ax.plot(evl["step"], evl["f1"],       "o-",  color=ACCENT,  linewidth=2, markersize=4, label="Weighted F1")
        ax.plot(evl["step"], evl["f1_ai"],    "s--", color=ACCENT2, linewidth=1.5, markersize=4, label="F1 (AI)")
        ax.plot(evl["step"], evl["f1_human"], "^--", color=ACCENT3, linewidth=1.5, markersize=4, label="F1 (Human)")
    ax.set_title("F1 Scores"); ax.set_xlabel("Step"); ax.set_ylabel("F1")
    ax.set_ylim(0, 1); ax.legend(fontsize=8)

    # ── 4. Learning rate schedule ──────────────────────────────────────────
    ax = axes[1, 0]
    if not train.empty and train["lr"].sum() > 0:
        ax.plot(train["step"], train["lr"], color=ACCENT4, linewidth=1.5)
        ax.fill_between(train["step"], train["lr"], alpha=0.2, color=ACCENT4)
    ax.set_title("Learning Rate Schedule"); ax.set_xlabel("Step"); ax.set_ylabel("LR")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.1e}"))

    # ── 5. Train vs eval loss gap ──────────────────────────────────────────
    ax = axes[1, 1]
    if not train.empty and not evl.empty:
        train_smooth = train.set_index("step")["loss"].ewm(span=20).mean()
        for row in evl.itertuples():
            nearby = train_smooth.index[train_smooth.index <= row.step]
            if len(nearby):
                t_loss = train_smooth.loc[nearby[-1]]
                ax.scatter(row.step, row.eval_loss - t_loss,
                           color=ACCENT2 if row.eval_loss - t_loss > 0 else ACCENT3,
                           s=40, zorder=4)
        ax.axhline(0, color="#ffffff", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_title("Val − Train Loss Gap (overfitting check)")
        ax.set_xlabel("Step"); ax.set_ylabel("Gap")

    # ── 6. Epoch timeline ─────────────────────────────────────────────────
    ax = axes[1, 2]
    if callback.epoch_times:
        epochs = [f"Epoch {i+1}" for i in range(len(callback.epoch_times))]
        bars   = ax.barh(epochs, callback.epoch_times, color=ACCENT, edgecolor="none")
        for bar, val in zip(bars, callback.epoch_times):
            ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                    f"{val/60:.1f} min", va="center", fontsize=9, color="#c8cad4")
        ax.set_title("Epoch Duration")
        ax.set_xlabel("Seconds")
    else:
        ax.text(0.5, 0.5, "No epoch data", transform=ax.transAxes,
                ha="center", va="center", color=DIM)
        ax.set_title("Epoch Duration")

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓  Training curves saved → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3 — CONFUSION MATRIX
# ═══════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(labels, preds, save_path: Path):
    cm  = confusion_matrix(labels, preds)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle("Confusion Matrix", color="#ffffff", fontsize=14, weight="bold")

    for ax, normalize, title in zip(axes, [None, "true"], ["Raw Counts", "Row-Normalised"]):
        cm_plot = confusion_matrix(labels, preds, normalize=normalize)
        disp = ConfusionMatrixDisplay(cm_plot, display_labels=["Human", "AI"])
        disp.plot(ax=ax, colorbar=False,
                  cmap=plt.cm.Blues if normalize else plt.cm.YlOrBr)
        ax.set_title(title, color="#ffffff")
        ax.set_xlabel("Predicted", color="#c8cad4")
        ax.set_ylabel("True", color="#c8cad4")
        for txt in ax.texts:
            txt.set_color("#ffffff")
            txt.set_fontsize(12)
        ax.tick_params(colors="#8889a0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2e3147")

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓  Confusion matrix saved → {save_path}")

def plot_roc(labels, probs, save_path: Path):
    fpr, tpr, thresholds = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    # Youden J
    j_idx   = np.argmax(tpr - fpr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("ROC Analysis", color="#ffffff", fontsize=14, weight="bold")

    # ROC curve
    ax = axes[0]
    ax.plot(fpr, tpr, color=ACCENT, linewidth=2.5, label=f"AUC = {roc_auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.15, color=ACCENT)
    ax.plot([0, 1], [0, 1], "--", color=DIM, linewidth=1, label="Random")
    ax.scatter(fpr[j_idx], tpr[j_idx], color=ACCENT4, s=80, zorder=5,
               label=f"Best threshold = {thresholds[j_idx]:.3f}")
    ax.set_title(f"ROC Curve  (AUC = {roc_auc:.4f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)

    # Threshold sweep — TPR / FPR / F1 vs threshold
    ax = axes[1]
    thres_range = thresholds[thresholds <= 1.0]
    tpr_t = tpr[:len(thres_range)]
    fpr_t = fpr[:len(thres_range)]
    f1_t  = np.where((tpr_t + (1-fpr_t)) > 0,
                     2 * tpr_t * (1-fpr_t) / (tpr_t + (1-fpr_t) + 1e-9), 0)
    ax.plot(thres_range, tpr_t,   color=ACCENT3, linewidth=1.5, label="TPR (recall AI)")
    ax.plot(thres_range, 1-fpr_t, color=ACCENT2, linewidth=1.5, label="TNR (recall Human)")
    ax.plot(thres_range, f1_t,    color=ACCENT4, linewidth=2,   label="F1")
    ax.axvline(thresholds[j_idx], color="#ffffff", linewidth=1, linestyle=":", alpha=0.7)
    ax.set_title("Metrics vs Decision Threshold")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓  ROC curve saved → {save_path}")


def plot_precision_recall(labels, probs, save_path: Path):
    prec, rec, thresholds = precision_recall_curve(labels, probs[:, 1])
    ap = average_precision_score(labels, probs[:, 1])
    # Best F1 on PR curve
    f1_pr = np.where((prec + rec) > 0, 2*prec*rec/(prec+rec+1e-9), 0)
    best  = np.argmax(f1_pr[:-1])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Precision-Recall Analysis", color="#ffffff", fontsize=14, weight="bold")

    ax = axes[0]
    ax.plot(rec, prec, color=ACCENT2, linewidth=2.5, label=f"AP = {ap:.4f}")
    ax.fill_between(rec, prec, alpha=0.15, color=ACCENT2)
    ax.scatter(rec[best], prec[best], color=ACCENT4, s=80, zorder=5,
               label=f"Best F1 @ thr={thresholds[best]:.3f}")
    ax.axhline(labels.mean(), color=DIM, linewidth=1, linestyle="--", label="Baseline")
    ax.set_title(f"Precision-Recall  (AP = {ap:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(thresholds, prec[:-1], color=ACCENT3, linewidth=1.5, label="Precision")
    ax.plot(thresholds, rec[:-1],  color=ACCENT2, linewidth=1.5, label="Recall")
    ax.plot(thresholds, f1_pr[:-1], color=ACCENT4, linewidth=2,  label="F1")
    ax.axvline(thresholds[best], color="#ffffff", linewidth=1, linestyle=":", alpha=0.7)
    ax.set_title("Precision / Recall / F1 vs Threshold")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓  Precision-Recall saved → {save_path}")


def plot_confidence(labels, probs, save_path: Path):
    ai_probs    = probs[:, 1]
    correct     = (np.argmax(probs, axis=1) == labels)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.suptitle("Model Confidence & Calibration", color="#ffffff", fontsize=14, weight="bold")

    # ── Confidence histogram ───────────────────────────────────────────────
    ax = axes[0]
    ax.hist(ai_probs[labels == 0], bins=50, alpha=0.7, color=ACCENT,  label="Human", edgecolor="none")
    ax.hist(ai_probs[labels == 1], bins=50, alpha=0.7, color=ACCENT2, label="AI",    edgecolor="none")
    ax.set_title("Predicted AI Probability by True Label")
    ax.set_xlabel("P(AI)"); ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)

    # ── Correct vs wrong confidence ────────────────────────────────────────
    ax = axes[1]
    ax.hist(np.max(probs[correct],  axis=1), bins=40, alpha=0.75, color=ACCENT3,
            label="Correct", edgecolor="none")
    ax.hist(np.max(probs[~correct], axis=1), bins=40, alpha=0.75, color=ACCENT2,
            label="Wrong",   edgecolor="none")
    ax.set_title("Confidence: Correct vs Wrong Predictions")
    ax.set_xlabel("Max Probability"); ax.set_ylabel("Frequency")
    ax.legend(fontsize=8)

    # ── Reliability diagram (calibration) ─────────────────────────────────
    ax = axes[2]
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2
    mean_conf, mean_acc, counts = [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (ai_probs >= lo) & (ai_probs < hi)
        if mask.sum():
            mean_conf.append(ai_probs[mask].mean())
            mean_acc.append(labels[mask].mean())
            counts.append(mask.sum())
    mean_conf = np.array(mean_conf)
    mean_acc  = np.array(mean_acc)
    ax.plot([0, 1], [0, 1], "--", color=DIM, linewidth=1.5, label="Perfect calibration")
    ax.bar(mean_conf, mean_acc, width=0.08, alpha=0.6, color=ACCENT4,
           edgecolor="none", label="Observed accuracy")
    ax.plot(mean_conf, mean_acc, "o-", color=ACCENT4, linewidth=2, markersize=5)
    ax.fill_between(mean_conf, mean_conf, mean_acc, alpha=0.15,
                    color=ACCENT2, label="Calibration gap")
    ax.set_title("Reliability Diagram (Calibration)")
    ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction Positive")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=8)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓  Confidence/calibration saved → {save_path}")


def plot_per_class_metrics(labels, preds, probs, save_path: Path):
    from sklearn.metrics import precision_score, recall_score

    metrics = {}
    for cls, name in [(0, "Human"), (1, "AI")]:
        metrics[name] = {
            "Precision" : precision_score(labels, preds, pos_label=cls),
            "Recall"    : recall_score(labels, preds, pos_label=cls),
            "F1"        : f1_score(labels, preds, pos_label=cls, average="binary"),
            "AUC"       : auc(*roc_curve(labels, probs[:, cls])[:2]),
        }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Per-Class Metrics", color="#ffffff", fontsize=14, weight="bold")

    # ── Grouped bar ───────────────────────────────────────────────────────
    ax = axes[0]
    met_names = list(next(iter(metrics.values())).keys())
    x = np.arange(len(met_names))
    w = 0.3
    for i, (cls_name, vals) in enumerate(metrics.items()):
        color = ACCENT if cls_name == "Human" else ACCENT2
        bars  = ax.bar(x + i*w, list(vals.values()), w, label=cls_name,
                       color=color, edgecolor="none", alpha=0.85)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", fontsize=7.5, color="#ffffff")
    ax.set_xticks(x + w/2); ax.set_xticklabels(met_names)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("Metric Comparison by Class")
    ax.legend(fontsize=9)

    # ── Radar chart ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 2, 2, polar=True)
    categories = met_names
    N     = len(categories)
    angles= [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    for (cls_name, vals), color in zip(metrics.items(), [ACCENT, ACCENT2]):
        vals_list = list(vals.values()) + [list(vals.values())[0]]
        ax2.plot(angles, vals_list, "o-", color=color, linewidth=2, markersize=5, label=cls_name)
        ax2.fill(angles, vals_list, alpha=0.15, color=color)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, size=9)
    ax2.set_ylim(0, 1)
    ax2.set_facecolor("#1a1d27")
    ax2.tick_params(colors="#8889a0")
    ax2.spines["polar"].set_color("#2e3147")
    ax2.yaxis.set_ticklabels([])
    for spine in ax2.spines.values():
        spine.set_color("#2e3147")
    ax2.set_title("Radar — Per-Class Performance", color="#ffffff", pad=18)
    ax2.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓  Per-class metrics saved → {save_path}")

class VRAMMonitorCallback(TrainerCallback):
    def __init__(self):
        self.vram_log = []   # {"step", "allocated_gb", "reserved_gb"}

    def on_step_end(self, args, state, control, **kw):
        if torch.cuda.is_available() and state.global_step % 50 == 0:
            alloc   = torch.cuda.memory_allocated()  / 1e9
            reserved= torch.cuda.memory_reserved()   / 1e9
            self.vram_log.append({
                "step"        : state.global_step,
                "allocated_gb": alloc,
                "reserved_gb" : reserved,
            })

def plot_vram(vram_log, save_path: Path):
    if not vram_log:
        print("  ⚠  No VRAM data collected (no CUDA?).")
        return
    df  = pd.DataFrame(vram_log)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9

    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    fig.suptitle("GPU VRAM Usage During Training", color="#ffffff", fontsize=13, weight="bold")
    ax.fill_between(df["step"], df["reserved_gb"],  alpha=0.3, color=ACCENT2, label="Reserved")
    ax.fill_between(df["step"], df["allocated_gb"], alpha=0.6, color=ACCENT,  label="Allocated")
    ax.axhline(total, color=ACCENT4, linewidth=1.5, linestyle="--", label=f"Total VRAM ({total:.1f} GB)")
    ax.set_ylim(0, total * 1.05)
    ax.set_xlabel("Training Step"); ax.set_ylabel("GB")
    ax.legend(fontsize=9)

    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓  VRAM usage saved → {save_path}")

def plot_checkpoint_summary(eval_log: list, best_step: int, save_path: Path):
    if not eval_log:
        return
    df = pd.DataFrame(eval_log)
    metrics_cols = ["eval_loss", "accuracy", "f1", "f1_ai", "f1_human"]
    # Normalise each column 0-1 for heatmap colour
    norm = df[metrics_cols].copy()
    for col in metrics_cols:
        rng = norm[col].max() - norm[col].min()
        norm[col] = (norm[col] - norm[col].min()) / (rng if rng else 1)
    # For loss lower = better, flip
    norm["eval_loss"] = 1 - norm["eval_loss"]

    fig, ax = plt.subplots(figsize=(max(10, len(df)*1.2), 4), constrained_layout=True)
    fig.suptitle("Checkpoint Performance Heatmap", color="#ffffff", fontsize=13, weight="bold")

    im = ax.imshow(norm[metrics_cols].T.values, cmap="RdYlGn", aspect="auto",
                   vmin=0, vmax=1)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([f"step\n{s}" for s in df["step"]], fontsize=8)
    ax.set_yticks(range(len(metrics_cols)))
    ax.set_yticklabels(["Loss↓", "Accuracy", "F1 weighted", "F1 AI", "F1 Human"], fontsize=9)

    # Raw value annotations
    for i, col in enumerate(metrics_cols):
        for j, val in enumerate(df[col]):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7.5, color="#000000")

    # Highlight best checkpoint column
    if best_step and best_step in df["step"].values:
        best_col = df.index[df["step"] == best_step][0]
        for row in range(len(metrics_cols)):
            ax.add_patch(plt.Rectangle((best_col - 0.5, row - 0.5), 1, 1,
                                       fill=False, edgecolor=ACCENT4, linewidth=2))
        ax.text(best_col, -0.8, "★ BEST", ha="center", color=ACCENT4, fontsize=9, weight="bold")

    plt.colorbar(im, ax=ax, label="Normalised score (higher = better)",
                 fraction=0.03, pad=0.02)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  ✓  Checkpoint heatmap saved → {save_path}")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy" : accuracy_score(labels, preds),
        "f1"       : f1_score(labels, preds, average="weighted"),
        "f1_ai"    : f1_score(labels, preds, pos_label=1, average="binary"),
        "f1_human" : f1_score(labels, preds, pos_label=0, average="binary"),
    }

def main():
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n  Run: {run_stamp}")

    # ── Load & clean ───────────────────────────────────────────────────────
    df = pd.read_csv("input_final.csv")
    print(f"\nTotal samples : {len(df)}")
    print(f"Label balance :\n{df['label'].value_counts()}")

    df = df.dropna(subset=["text", "label"])
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)
    df = df[df["text"].str.len() > 10]

    # ── EDA plot ───────────────────────────────────────────────────────────
    print("\n[1/9] Generating EDA plots …")
    plot_eda(df, PLOTS_DIR / "01_eda.png")

    # ── Split ──────────────────────────────────────────────────────────────
    train_df, temp_df = train_test_split(df, test_size=0.2,  random_state=SEED, stratify=df["label"])
    val_df,   test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"])
    print(f"\nTrain:{len(train_df)}  Val:{len(val_df)}  Test:{len(test_df)}")

    # ── HuggingFace datasets ───────────────────────────────────────────────
    train_ds = Dataset.from_pandas(train_df[["text","label"]].reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df[["text","label"]].reset_index(drop=True))
    test_ds  = Dataset.from_pandas(test_df[["text","label"]].reset_index(drop=True))

    # ── Tokenize ───────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding=False)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    val_ds   = val_ds.map(tokenize,   batched=True, remove_columns=["text"])
    test_ds  = test_ds.map(tokenize,  batched=True, remove_columns=["text"])
    data_collator = DataCollatorWithPadding(tokenizer)

    # ── Model ──────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        id2label={0:"HUMAN",1:"AI"}, label2id={"HUMAN":0,"AI":1},
    )
    model.gradient_checkpointing_enable()

    # ── Step / eval counts ─────────────────────────────────────────────────
    steps_per_epoch = len(train_ds) // (BATCH_SIZE * GRAD_ACCUM)
    total_steps     = steps_per_epoch * EPOCHS
    eval_steps      = 500

    # ── Callbacks ─────────────────────────────────────────────────────────
    progress_cb = LiveProgressCallback(total_steps, eval_steps, PLOTS_DIR)
    vram_cb     = VRAMMonitorCallback()

    # ── TrainingArguments ──────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir                  = str(OUTPUT_DIR),
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        warmup_ratio                = WARMUP_RATIO,
        weight_decay                = WEIGHT_DECAY,
        eval_strategy         = "steps",
        eval_steps                  = eval_steps,
        save_strategy               = "steps",
        save_steps                  = eval_steps,
        save_total_limit            = 3,            # keep top-3 checkpoints
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        fp16                        = True,
        tf32                        = True,
        torch_compile               = False if IS_WINDOWS else True,
        optim                       = "adamw_torch_fused",
        dataloader_num_workers      = 8,
        dataloader_pin_memory       = True,
        logging_steps               = 100,
        report_to                   = "none",
        seed                        = SEED,
        ddp_find_unused_parameters  = False,
    )

    trainer = Trainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = tokenizer,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
        callbacks       = [
            EarlyStoppingCallback(early_stopping_patience=3),
            progress_cb,
            vram_cb,
        ],
    )

    # ── Train ──────────────────────────────────────────────────────────────
    trainer.train()

    # ── Save best model + tokenizer ────────────────────────────────────────
    print(f"\n[Saving] Best model → {OUTPUT_DIR}/")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    # Also save model config summary JSON
    config_summary = {
        "run"           : run_stamp,
        "model"         : MODEL_NAME,
        "max_len"       : MAX_LEN,
        "batch_size"    : BATCH_SIZE,
        "grad_accum"    : GRAD_ACCUM,
        "effective_bs"  : BATCH_SIZE * GRAD_ACCUM,
        "epochs"        : EPOCHS,
        "lr"            : LR,
        "best_f1"       : progress_cb.best_f1,
        "best_step"     : progress_cb.best_step,
        "output_dir"    : str(OUTPUT_DIR),
    }
    with open(OUTPUT_DIR / "run_config.json", "w") as f:
        json.dump(config_summary, f, indent=2)
    print(f"  ✓  Config saved → {OUTPUT_DIR / 'run_config.json'}")

    # ── Test-set inference ─────────────────────────────────────────────────
    print("\n── Final Test Set Evaluation ──")
    preds_out = trainer.predict(test_ds)
    logits    = preds_out.predictions
    labels    = preds_out.label_ids
    preds     = np.argmax(logits, axis=-1)
    probs     = torch.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()

    print(classification_report(labels, preds, target_names=["HUMAN", "AI"], digits=4))
    print(f"\nPeak VRAM used : {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ── Post-training plots ────────────────────────────────────────────────
    print("\n── Generating all diagnostic plots ──")
    print("[2/9] Training curves …")
    plot_training_curves(progress_cb, PLOTS_DIR / "02_training_curves.png")

    print("[3/9] Confusion matrix …")
    plot_confusion_matrix(labels, preds, PLOTS_DIR / "03_confusion_matrix.png")

    print("[4/9] ROC curve …")
    plot_roc(labels, probs, PLOTS_DIR / "04_roc_curve.png")

    print("[5/9] Precision-Recall …")
    plot_precision_recall(labels, probs, PLOTS_DIR / "05_precision_recall.png")

    print("[6/9] Confidence & calibration …")
    plot_confidence(labels, probs, PLOTS_DIR / "06_confidence_calibration.png")

    print("[7/9] Per-class metrics …")
    plot_per_class_metrics(labels, preds, probs, PLOTS_DIR / "07_per_class_metrics.png")

    print("[8/9] VRAM usage …")
    plot_vram(vram_cb.vram_log, PLOTS_DIR / "08_vram_usage.png")

    print("[9/9] Checkpoint heatmap …")
    plot_checkpoint_summary(progress_cb.eval_log, progress_cb.best_step,
                            PLOTS_DIR / "09_checkpoint_heatmap.png")

    # ── Final summary ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  ✅  All done!")
    print(f"  Model  : {OUTPUT_DIR}/")
    print(f"  Plots  : {PLOTS_DIR}/  ({len(list(PLOTS_DIR.glob('*.png')))} files)")
    print(f"  Best F1: {progress_cb.best_f1:.4f}  (step {progress_cb.best_step})")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()