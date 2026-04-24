
import csv, math, json, os, random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

#  LOAD DATA
MISSING = "?"   # sentinel for unknown values

def load_csv(filepath):
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV file is empty.")
    skip = {"sr_no", "label_id", ""}
    feature_keys = [k for k in rows[0].keys() if k not in skip]
    features = [{k: (v.strip() if v.strip() != "" else MISSING)
                 for k, v in row.items() if k in feature_keys} for row in rows]
    labels   = [row["label_id"].strip() for row in rows]
    return features, labels

#  STRATIFIED TRAIN / TEST SPLIT
def stratified_split(features, labels, test_ratio=0.2, seed=42):
    random.seed(seed)
    class_buckets = {}
    for i, lbl in enumerate(labels):
        class_buckets.setdefault(lbl, []).append(i)
    train_idx, test_idx = [], []
    for lbl, idxs in class_buckets.items():
        shuffled = idxs[:]
        random.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_ratio))
        test_idx.extend(shuffled[:n_test])
        train_idx.extend(shuffled[n_test:])
    X_train = [features[i] for i in train_idx]
    y_train = [labels[i]   for i in train_idx]
    X_test  = [features[i] for i in test_idx]
    y_test  = [labels[i]   for i in test_idx]
    return X_train, y_train, X_test, y_test

#  C4.5 CORE
def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def info_gain_c45(features, labels, feature):
    """Information gain accounting for missing values (C4.5 style)."""
    total = len(labels)
    # Only consider non-missing examples for this feature
    non_missing_idx = [i for i, f in enumerate(features) if f[feature] != MISSING]
    if not non_missing_idx:
        return 0.0, 0.0

    ratio_known = len(non_missing_idx) / total
    known_labels = [labels[i] for i in non_missing_idx]
    known_features = [features[i] for i in non_missing_idx]

    base_entropy = entropy(known_labels)
    n_known = len(known_labels)

    values = set(f[feature] for f in known_features)
    weighted = 0.0
    split_info = 0.0
    for val in values:
        subset = [labels[non_missing_idx[i]] for i, f in enumerate(known_features) if f[feature] == val]
        prop = len(subset) / n_known
        weighted += prop * entropy(subset)
        if prop > 0:
            split_info -= prop * math.log2(prop)

    # Penalise for missing values in split_info
    miss_prop = 1 - ratio_known
    if miss_prop > 0:
        split_info -= miss_prop * math.log2(miss_prop)

    ig = ratio_known * (base_entropy - weighted)
    return ig, split_info


def gain_ratio(features, labels, feature):
    ig, si = info_gain_c45(features, labels, feature)
    if si == 0:
        return 0.0
    return ig / si


def best_feature_c45(features, labels, available):
    """C4.5 heuristic: consider only features with above-average IG, pick best GR."""
    ig_map = {}
    gr_map = {}
    for f in available:
        ig, si = info_gain_c45(features, labels, f)
        ig_map[f] = ig
        gr_map[f] = ig / si if si > 0 else 0.0
    avg_ig = sum(ig_map.values()) / len(ig_map) if ig_map else 0
    candidates = [f for f in available if ig_map[f] >= avg_ig] or list(available)
    best = max(candidates, key=lambda f: gr_map[f])
    return best, gr_map


def build_tree(features, labels, available, depth=0, max_depth=None):
    if len(set(labels)) == 1:
        return {"leaf": True, "label": labels[0], "count": len(labels)}
    if not available or (max_depth is not None and depth >= max_depth):
        majority = Counter(labels).most_common(1)[0][0]
        return {"leaf": True, "label": majority, "count": len(labels)}

    feat, gr_map = best_feature_c45(features, labels, available)
    remaining = [f for f in available if f != feat]

    # Distribute missing-value examples proportionally
    known_idx   = [i for i, f in enumerate(features) if f[feat] != MISSING]
    missing_idx = [i for i, f in enumerate(features) if f[feat] == MISSING]
    total_known = len(known_idx)

    values = sorted(set(features[i][feat] for i in known_idx))
    if not values:
        majority = Counter(labels).most_common(1)[0][0]
        return {"leaf": True, "label": majority, "count": len(labels)}

    tree = {
        "leaf": False, "feature": feat,
        "gain_ratio": round(gr_map[feat], 6),
        "depth": depth, "count": len(labels),
        "branches": {}
    }

    for val in values:
        val_idx = [i for i in known_idx if features[i][feat] == val]
        frac = len(val_idx) / total_known if total_known > 0 else 0
        # fractional missing: add proportional copies (weight-based; we simplify here
        # by including missing examples with majority val repeated by fraction)
        extra = missing_idx  # passed as-is; tree sees them all; fraction noted
        sub_idx = val_idx + [i for i in missing_idx]  # include missing in all branches
        sub_f = [features[i] for i in val_idx]        # but only train on non-missing for split
        sub_l = [labels[i]   for i in val_idx]
        # For children we add proportional missing examples
        n_extra = round(frac * len(missing_idx))
        extra_sample = missing_idx[:n_extra]
        sub_f += [features[i] for i in extra_sample]
        sub_l += [labels[i]   for i in extra_sample]

        if not sub_l:
            majority = Counter(labels).most_common(1)[0][0]
            tree["branches"][val] = {"leaf": True, "label": majority, "count": 0}
        else:
            tree["branches"][val] = build_tree(sub_f, sub_l, remaining, depth + 1, max_depth)

    return tree


def predict(tree, sample):
    if tree["leaf"]:
        return tree["label"]
    feat = tree["feature"]
    val  = sample.get(feat, MISSING)
    if val == MISSING:
        # Vote across branches weighted by count
        votes = Counter()
        for branch_val, sub in tree["branches"].items():
            pred = predict(sub, sample)
            votes[pred] += sub["count"]
        return votes.most_common(1)[0][0]
    if val not in tree["branches"]:
        best = max(tree["branches"].items(), key=lambda x: x[1]["count"])
        return predict(best[1], sample)
    return predict(tree["branches"][val], sample)

#  EVALUATION
def evaluate(tree, features, labels, split_name="Test"):
    preds = [predict(tree, f) for f in features]
    correct = sum(p == l for p, l in zip(preds, labels))
    acc = correct / len(labels)
    classes = sorted(set(labels))

    print(f"\n  [{split_name}]  Accuracy: {correct}/{len(labels)} = {acc*100:.1f}%")
    print(f"\n  {'Sample':<10} {'Predicted':>12} {'Actual':>12} {'Match':>8}")
    print("  " + "-" * 46)
    for i, (p, l) in enumerate(zip(preds, labels)):
        print(f"  {i:<10} {p:>12} {l:>12} {'✓' if p==l else '✗':>8}")

    print(f"\n  Confusion Matrix ({split_name}):")
    header = f"  {'':14}" + "".join(f"  Pred:{c:<6}" for c in classes)
    print(header)
    for actual in classes:
        row = f"  Actual:{actual:<6}"
        for pred_c in classes:
            cnt = sum(1 for p, l in zip(preds, labels) if l == actual and p == pred_c)
            row += f"  {cnt:>10}"
        print(row)

    return preds, acc, classes

#  MATPLOTLIB VISUALISATIONS
def _label_name(lbl):
    return "AI-gen" if str(lbl) == "1.0" else "Human"


def plot_confusion_matrix(preds, labels, classes, out_dir, tag="test"):
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for p, l in zip(preds, labels):
        cm[idx[l]][idx[p]] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([f"Pred:{_label_name(c)}" for c in classes], fontsize=9)
    ax.set_yticklabels([f"Act:{_label_name(c)}" for c in classes], fontsize=9)
    ax.set_title(f"Confusion Matrix [{tag}] — C4.5", fontsize=12, pad=10)
    plt.colorbar(im, ax=ax)
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)
    plt.tight_layout()
    path = os.path.join(out_dir, f"c45_confusion_matrix_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_gain_ratio_importances(features, labels, feature_names, out_dir):
    gr_vals = {}
    for f in feature_names:
        ig, si = info_gain_c45(features, labels, f)
        gr_vals[f] = ig / si if si > 0 else 0.0
    ranked = sorted(gr_vals.items(), key=lambda x: x[1])
    names, values = [r[0] for r in ranked], [r[1] for r in ranked]
    top3 = sorted(values)[-3:]
    colors = ["#1B5E20" if v in top3 else "#A5D6A7" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.42)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Gain Ratio", fontsize=11)
    ax.set_title("C4.5 — Feature Importances (Gain Ratio)", fontsize=13, pad=12)
    ax.set_xlim(0, max(values) * 1.18 if values else 1)
    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "c45_gain_ratio_importances.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_ig_vs_gr(features, labels, feature_names, out_dir):
    """Scatter: Information Gain vs Gain Ratio to show C4.5 correction effect."""
    ig_vals, gr_vals_list = [], []
    for f in feature_names:
        ig, si = info_gain_c45(features, labels, f)
        gr_vals_list.append(ig / si if si > 0 else 0.0)
        ig_vals.append(ig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ig_vals, gr_vals_list, color="#2E7D32", s=80, zorder=3)
    for i, fname in enumerate(feature_names):
        ax.annotate(fname, (ig_vals[i], gr_vals_list[i]),
                    textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("Information Gain (ID3)", fontsize=11)
    ax.set_ylabel("Gain Ratio (C4.5)", fontsize=11)
    ax.set_title("C4.5 — IG vs Gain Ratio Comparison", fontsize=13, pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "c45_ig_vs_gr.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_label_distribution(train_labels, test_labels, out_dir):
    all_classes = sorted(set(train_labels) | set(test_labels))
    train_c = Counter(train_labels)
    test_c  = Counter(test_labels)
    x = np.arange(len(all_classes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w/2, [train_c.get(c, 0) for c in all_classes], w,
           label="Train", color="#66BB6A", edgecolor="white")
    ax.bar(x + w/2, [test_c.get(c, 0) for c in all_classes], w,
           label="Test",  color="#EF9A9A", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([_label_name(c) for c in all_classes])
    ax.set_ylabel("Count")
    ax.set_title("C4.5 — Label Distribution (Train vs Test)", fontsize=12)
    ax.legend()
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, str(int(h)),
                    ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "c45_label_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_per_feature_class_dist(features, labels, feature_names, out_dir):
    classes = sorted(set(labels))
    n = len(feature_names)
    cols = 4
    rows = math.ceil(n / cols)
    palette = ["#66BB6A", "#EF5350", "#42A5F5", "#FFA726"]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    for fi, feat in enumerate(feature_names):
        ax = axes[fi]
        vals = sorted(set(f[feat] for f in features if f[feat] != MISSING))
        bottoms = np.zeros(len(vals))
        for ci, cls in enumerate(classes):
            counts = [sum(1 for f, l in zip(features, labels)
                         if f[feat] == v and l == cls) for v in vals]
            ax.bar(vals, counts, bottom=bottoms,
                   color=palette[ci % len(palette)], label=cls, edgecolor="white", width=0.6)
            bottoms += np.array(counts, dtype=float)
        ax.set_title(feat, fontsize=8, pad=4)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    for fi in range(n, len(axes)):
        fig.delaxes(axes[fi])
    handles = [mpatches.Patch(color=palette[i], label=_label_name(c)) for i, c in enumerate(classes)]
    fig.legend(handles=handles, title="Class", loc="lower right", fontsize=9)
    plt.suptitle("C4.5 — Per-Feature Class Distribution (Train)", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "c45_per_feature_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_accuracy_summary(train_acc, test_acc, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Train Accuracy", "Test Accuracy"],
           [train_acc * 100, test_acc * 100],
           color=["#66BB6A", "#2E7D32"], edgecolor="white", width=0.4)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("C4.5 — Train vs Test Accuracy", fontsize=12)
    for i, v in enumerate([train_acc * 100, test_acc * 100]):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "c45_accuracy_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

#  ASCII TREE PRINT
def print_tree(tree, prefix="", is_last=True, branch_val=None):
    connector = "└── " if is_last else "├── "
    extension = "    " if is_last else "│   "
    bv = f"[{branch_val}] " if branch_val is not None else ""
    if tree["leaf"]:
        print(f"{prefix}{connector}{bv}LEAF → {_label_name(tree['label'])} ({tree['label']})  [n={tree['count']}]")
    else:
        gr = tree.get("gain_ratio", "?")
        print(f"{prefix}{connector}{bv}SPLIT on '{tree['feature']}'  (gain_ratio={gr}, n={tree['count']})")
        branches = list(tree["branches"].items())
        for i, (val, sub) in enumerate(branches):
            print_tree(sub, prefix + extension, i == len(branches) - 1, val)


def print_feature_importances(features, labels, feature_names):
    print("\n  Feature Gain Ratios (on training set):")
    gr_vals = {}
    for f in feature_names:
        ig, si = info_gain_c45(features, labels, f)
        gr_vals[f] = ig / si if si > 0 else 0.0
    ranked = sorted(gr_vals.items(), key=lambda x: x[1], reverse=True)
    max_g = ranked[0][1] if ranked[0][1] > 0 else 1
    for fname, g in ranked:
        bar = "█" * int((g / max_g) * 30) + "░" * (30 - int((g / max_g) * 30))
        print(f"  {fname:<35} {bar}  {g:.4f}")

#  INFERENCE
def interactive_inference(tree, feature_names):
    print("\n" + "=" * 60)
    print("  INFERENCE MODE  (Ctrl+C to quit) — blank = missing value")
    print("=" * 60)
    while True:
        try:
            sample = {}
            print("\n  --- New Sample ---")
            for feat in feature_names:
                val = input(f"  {feat} [blank=?]: ").strip().upper()
                sample[feat] = val if val else MISSING
            pred = predict(tree, sample)
            print(f"\n  ► Prediction: {_label_name(pred)}  (label_id = {pred})\n")
        except KeyboardInterrupt:
            print("\n  Exiting.")
            break


def quick_inference(tree, feature_names):
    print("\n  Order:", ", ".join(feature_names))
    raw = input("  Comma-separated values (blank for ?): ").strip().split(",")
    if len(raw) != len(feature_names):
        print(f"  [!] Expected {len(feature_names)} values, got {len(raw)}")
        return
    sample = {feature_names[i]: (raw[i].strip().upper() or MISSING) for i in range(len(feature_names))}
    pred = predict(tree, sample)
    print(f"\n  ► Prediction: {_label_name(pred)}  (label_id = {pred})\n")

#  MAIN
def main():
    print("=" * 60)
    print("  C4.5 DECISION TREE — FROM SCRATCH (Gain Ratio)")
    print("=" * 60)

    path = input("\nCSV file path: ").strip()
    try:
        features, labels = load_csv(path)
        print(f"  Loaded {len(features)} rows, {len(features[0])} features.")
    except Exception as e:
        print(f"  [!] Error: {e}")
        return

    feature_names = list(features[0].keys())

    ratio_in = input("Test split ratio [0.2]: ").strip()
    test_ratio = float(ratio_in) if ratio_in else 0.2

    depth_in = input("Max depth (blank = unlimited): ").strip()
    max_depth = int(depth_in) if depth_in.isdigit() else None

    X_train, y_train, X_test, y_test = stratified_split(features, labels, test_ratio)
    print(f"\n  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    print("\n  Training C4.5 tree...")
    tree = build_tree(X_train, y_train, feature_names[:], max_depth=max_depth)
    print("  Tree built successfully.")

    print("\n  Decision Tree Structure")
    print("=" * 60)
    print_tree(tree)

    print_feature_importances(X_train, y_train, feature_names)

    print("\n  Evaluating...")
    tr_preds, tr_acc, classes = evaluate(tree, X_train, y_train, "Train")
    te_preds, te_acc, _       = evaluate(tree, X_test,  y_test,  "Test")

    out_dir = input("\nOutput folder for plots [./plots_c45]: ").strip() or "./plots_c45"
    os.makedirs(out_dir, exist_ok=True)

    print("  Generating plots...")
    plot_confusion_matrix(tr_preds, y_train, classes, out_dir, "train")
    plot_confusion_matrix(te_preds, y_test,  classes, out_dir, "test")
    plot_gain_ratio_importances(X_train, y_train, feature_names, out_dir)
    plot_ig_vs_gr(X_train, y_train, feature_names, out_dir)
    plot_label_distribution(y_train, y_test, out_dir)
    plot_per_feature_class_dist(X_train, y_train, feature_names, out_dir)
    plot_accuracy_summary(tr_acc, te_acc, out_dir)
    print(f"  All plots saved to '{out_dir}/'")

    save = input("\nSave tree to JSON? [y/n]: ").strip().lower()
    if save == "y":
        jpath = input("Output path [c45_tree.json]: ").strip() or "c45_tree.json"
        with open(jpath, "w") as f:
            json.dump(tree, f, indent=2)
        print(f"  Tree saved to {jpath}")

    print("\n  (a) Interactive inference  (b) Quick inference  (c) Skip")
    ch = input("  Choice: ").strip().lower()
    if ch == "a":
        interactive_inference(tree, feature_names)
    elif ch == "b":
        quick_inference(tree, feature_names)

    print("\nDone.")


if __name__ == "__main__":
    main()