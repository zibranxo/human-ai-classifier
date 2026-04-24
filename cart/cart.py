import csv, math, json, os, random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

#  LOAD DATA
def _try_float(v):
    try:
        return float(v)
    except ValueError:
        return None


def load_csv(filepath):
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV file is empty.")
    skip = {"sr_no", "label_id", ""}
    feature_keys = [k for k in rows[0].keys() if k not in skip]
    features = [{k: v.strip() for k, v in row.items() if k in feature_keys} for row in rows]
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

#  CART CORE
def gini(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def gini_split(left_labels, right_labels):
    total = len(left_labels) + len(right_labels)
    if total == 0:
        return 0.0
    return (len(left_labels) / total) * gini(left_labels) + \
           (len(right_labels) / total) * gini(right_labels)


def _is_numeric_feature(features, feature):
    """Check if all non-empty values of a feature can be floats."""
    return all(_try_float(f[feature]) is not None for f in features if f[feature] != "")


def best_binary_split(features, labels, feature):
    """
    CART binary split:
      - Numeric: threshold t  → left if value <= t
      - Categorical: one value vs rest  → left if value == target_val
    Returns (best_gini, split_spec) where split_spec is a dict.
    """
    best_g = float("inf")
    best_spec = None

    vals = [f[feature] for f in features]

    if _is_numeric_feature(features, feature):
        float_vals = sorted(set(float(v) for v in vals))
        thresholds = [(float_vals[i] + float_vals[i+1]) / 2
                      for i in range(len(float_vals) - 1)]
        if not thresholds:
            thresholds = [float_vals[0]]

        for t in thresholds:
            left  = [labels[i] for i, f in enumerate(features) if float(f[feature]) <= t]
            right = [labels[i] for i, f in enumerate(features) if float(f[feature]) >  t]
            if not left or not right:
                continue
            g = gini_split(left, right)
            if g < best_g:
                best_g = g
                best_spec = {"type": "numeric", "threshold": t}
    else:
        unique_vals = set(vals)
        for target in unique_vals:
            left  = [labels[i] for i, f in enumerate(features) if f[feature] == target]
            right = [labels[i] for i, f in enumerate(features) if f[feature] != target]
            if not left or not right:
                continue
            g = gini_split(left, right)
            if g < best_g:
                best_g = g
                best_spec = {"type": "categorical", "value": target}

    return best_g, best_spec


def best_feature_cart(features, labels, available):
    best_g = float("inf")
    best_feat = None
    best_spec = None
    gini_map = {}

    for feat in available:
        g, spec = best_binary_split(features, labels, feat)
        gini_map[feat] = g
        if g < best_g:
            best_g = g
            best_feat = feat
            best_spec = spec

    return best_feat, best_spec, best_g, gini_map


def _split_samples(features, labels, feature, spec):
    if spec["type"] == "numeric":
        t = spec["threshold"]
        left_idx  = [i for i, f in enumerate(features) if float(f[feature]) <= t]
        right_idx = [i for i, f in enumerate(features) if float(f[feature]) >  t]
        left_label  = f"<= {t}"
        right_label = f"> {t}"
    else:
        v = spec["value"]
        left_idx  = [i for i, f in enumerate(features) if f[feature] == v]
        right_idx = [i for i, f in enumerate(features) if f[feature] != v]
        left_label  = f"== {v}"
        right_label = f"!= {v}"
    return left_idx, right_idx, left_label, right_label


def build_tree(features, labels, available, depth=0, max_depth=None, min_samples_split=2):
    majority = Counter(labels).most_common(1)[0][0]

    if len(set(labels)) == 1:
        return {"leaf": True, "label": labels[0], "count": len(labels),
                "gini": 0.0, "class_dist": dict(Counter(labels))}

    if (not available or
            (max_depth is not None and depth >= max_depth) or
            len(labels) < min_samples_split):
        return {"leaf": True, "label": majority, "count": len(labels),
                "gini": round(gini(labels), 6), "class_dist": dict(Counter(labels))}

    feat, spec, g, gini_map = best_feature_cart(features, labels, available)

    if feat is None or spec is None:
        return {"leaf": True, "label": majority, "count": len(labels),
                "gini": round(gini(labels), 6), "class_dist": dict(Counter(labels))}

    left_idx, right_idx, left_label, right_label = _split_samples(features, labels, feat, spec)

    if not left_idx or not right_idx:
        return {"leaf": True, "label": majority, "count": len(labels),
                "gini": round(gini(labels), 6), "class_dist": dict(Counter(labels))}

    # CART keeps the split feature available (binary splits can reuse features)
    remaining = available  # all features remain eligible

    tree = {
        "leaf": False, "feature": feat, "split": spec,
        "gini": round(g, 6), "depth": depth, "count": len(labels),
        "left_label": left_label, "right_label": right_label,
        "gini_map": {k: round(v, 6) for k, v in gini_map.items()},
        "left":  build_tree([features[i] for i in left_idx],  [labels[i] for i in left_idx],
                             remaining, depth + 1, max_depth, min_samples_split),
        "right": build_tree([features[i] for i in right_idx], [labels[i] for i in right_idx],
                             remaining, depth + 1, max_depth, min_samples_split)
    }
    return tree


def predict(tree, sample):
    if tree["leaf"]:
        return tree["label"]
    feat = tree["feature"]
    spec = tree["split"]
    val  = sample.get(feat, "")
    if spec["type"] == "numeric":
        fv = _try_float(val)
        go_left = (fv is not None and fv <= spec["threshold"])
    else:
        go_left = (val == spec["value"])
    return predict(tree["left"] if go_left else tree["right"], sample)

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

#  FEATURE IMPORTANCE  (Gini-based: total impurity reduction)
def collect_gini_importances(tree, importances=None):
    if importances is None:
        importances = {}
    if tree["leaf"]:
        return importances
    feat = tree["feature"]
    # impurity reduction = parent_gini - weighted child ginis
    n   = tree["count"]
    nl  = tree["left"]["count"]
    nr  = tree["right"]["count"]
    gl  = tree["left"].get("gini", 0)
    gr  = tree["right"].get("gini", 0)
    reduction = tree["gini"] - (nl / n) * gl - (nr / n) * gr
    importances[feat] = importances.get(feat, 0) + reduction * n

    collect_gini_importances(tree["left"],  importances)
    collect_gini_importances(tree["right"], importances)
    return importances

#  MATPLOTLIB VISUALISATIONS
def _label_name(lbl):
    return "AI-gen" if str(lbl) == "1.0" else "Human"


def plot_confusion_matrix(preds, labels, classes, out_dir, tag="test"):
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for p, l in zip(preds, labels):
        cm[idx[l]][idx[p]] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels([f"Pred:{_label_name(c)}" for c in classes], fontsize=9)
    ax.set_yticklabels([f"Act:{_label_name(c)}" for c in classes], fontsize=9)
    ax.set_title(f"Confusion Matrix [{tag}] — CART", fontsize=12, pad=10)
    plt.colorbar(im, ax=ax)
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)
    plt.tight_layout()
    path = os.path.join(out_dir, f"cart_confusion_matrix_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_gini_importances(tree, feature_names, out_dir):
    raw = collect_gini_importances(tree)
    # Normalise
    total = sum(raw.values()) or 1
    importances = {f: raw.get(f, 0) / total for f in feature_names}
    ranked = sorted(importances.items(), key=lambda x: x[1])
    names, values = [r[0] for r in ranked], [r[1] for r in ranked]
    top3 = sorted(values)[-3:]
    colors = ["#E65100" if v in top3 else "#FFCC80" for v in values]

    fig, ax = plt.subplots(figsize=(10, max(6, len(names) * 0.42)))
    bars = ax.barh(names, values, color=colors, edgecolor="white", height=0.7)
    ax.set_xlabel("Normalised Gini Importance", fontsize=11)
    ax.set_title("CART — Feature Importances (Gini Impurity Reduction)", fontsize=13, pad=12)
    ax.set_xlim(0, max(values) * 1.18 if values else 1)
    for bar, val in zip(bars, values):
        ax.text(val + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, "cart_gini_importances.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_gini_vs_entropy(features, labels, feature_names, out_dir):
    """Compare Gini vs Entropy for each feature on training data."""
    def _entropy(lbls):
        n = len(lbls)
        if n == 0: return 0.0
        counts = Counter(lbls)
        return -sum((c/n)*math.log2(c/n) for c in counts.values() if c > 0)

    gini_vals, ent_vals = [], []
    for feat in feature_names:
        vals_set = set(f[feat] for f in features)
        w_gini, w_ent = 0.0, 0.0
        for v in vals_set:
            sub = [labels[i] for i, f in enumerate(features) if f[feat] == v]
            prop = len(sub) / len(labels)
            w_gini += prop * gini(sub)
            w_ent  += prop * _entropy(sub)
        gini_vals.append(w_gini)
        ent_vals.append(w_ent)

    x = np.arange(len(feature_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(10, len(feature_names) * 0.5), 5))
    ax.bar(x - w/2, gini_vals, w, label="Gini",    color="#E65100", edgecolor="white")
    ax.bar(x + w/2, ent_vals,  w, label="Entropy", color="#1565C0", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Weighted Impurity")
    ax.set_title("CART — Gini vs Entropy per Feature", fontsize=13, pad=12)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "cart_gini_vs_entropy.png")
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
           label="Train", color="#FFA726", edgecolor="white")
    ax.bar(x + w/2, [test_c.get(c, 0) for c in all_classes], w,
           label="Test",  color="#EF5350", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([_label_name(c) for c in all_classes])
    ax.set_ylabel("Count")
    ax.set_title("CART — Label Distribution (Train vs Test)", fontsize=12)
    ax.legend()
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.05, str(int(h)),
                    ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "cart_label_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_per_feature_class_dist(features, labels, feature_names, out_dir):
    classes = sorted(set(labels))
    n = len(feature_names)
    cols = 4
    rows = math.ceil(n / cols)
    palette = ["#FFA726", "#EF5350", "#42A5F5", "#66BB6A"]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten()
    for fi, feat in enumerate(feature_names):
        ax = axes[fi]
        vals = sorted(set(f[feat] for f in features))
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
    plt.suptitle("CART — Per-Feature Class Distribution (Train)", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "cart_per_feature_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


def plot_accuracy_summary(train_acc, test_acc, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Train Accuracy", "Test Accuracy"],
           [train_acc * 100, test_acc * 100],
           color=["#FFA726", "#E65100"], edgecolor="white", width=0.4)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("CART — Train vs Test Accuracy", fontsize=12)
    for i, v in enumerate([train_acc * 100, test_acc * 100]):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, "cart_accuracy_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

#  ASCII TREE PRINT
def print_tree(tree, prefix="", is_last=True, branch_label=None):
    connector = "└── " if is_last else "├── "
    extension = "    " if is_last else "│   "
    bl = f"[{branch_label}] " if branch_label is not None else ""
    if tree["leaf"]:
        dist = ", ".join(f"{_label_name(k)}:{v}" for k, v in tree["class_dist"].items())
        print(f"{prefix}{connector}{bl}LEAF → {_label_name(tree['label'])}  "
              f"[n={tree['count']}, gini={tree['gini']:.4f}] ({dist})")
    else:
        spec = tree["split"]
        if spec["type"] == "numeric":
            cond = f"<= {spec['threshold']}"
        else:
            cond = f"== '{spec['value']}'"
        print(f"{prefix}{connector}{bl}SPLIT '{tree['feature']}' {cond}  "
              f"(gini={tree['gini']:.4f}, n={tree['count']})")
        print_tree(tree["left"],  prefix + extension, False, tree["left_label"])
        print_tree(tree["right"], prefix + extension, True,  tree["right_label"])


def print_feature_importances(tree, feature_names):
    raw = collect_gini_importances(tree)
    total = sum(raw.values()) or 1
    importances = {f: raw.get(f, 0) / total for f in feature_names}
    ranked = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    print("\n  Feature Gini Importances (normalised, on training tree):")
    max_g = ranked[0][1] if ranked[0][1] > 0 else 1
    for fname, g in ranked:
        bar = "█" * int((g / max_g) * 30) + "░" * (30 - int((g / max_g) * 30))
        print(f"  {fname:<35} {bar}  {g:.4f}")

#  INFERENCE
def interactive_inference(tree, feature_names):
    print("\n" + "=" * 60)
    print("  INFERENCE MODE  (Ctrl+C to quit)")
    print("=" * 60)
    while True:
        try:
            sample = {}
            print("\n  --- New Sample ---")
            for feat in feature_names:
                while True:
                    val = input(f"  {feat}: ").strip()
                    if val:
                        sample[feat] = val.upper()
                        break
                    print("    [!] Cannot be empty.")
            pred = predict(tree, sample)
            print(f"\n  ► Prediction: {_label_name(pred)}  (label_id = {pred})\n")
        except KeyboardInterrupt:
            print("\n  Exiting.")
            break


def quick_inference(tree, feature_names):
    print("\n  Order:", ", ".join(feature_names))
    raw = input("  Comma-separated values: ").strip().split(",")
    if len(raw) != len(feature_names):
        print(f"  [!] Expected {len(feature_names)} values, got {len(raw)}")
        return
    sample = {feature_names[i]: raw[i].strip().upper() for i in range(len(feature_names))}
    pred = predict(tree, sample)
    print(f"\n  ► Prediction: {_label_name(pred)}  (label_id = {pred})\n")

#  MAIN
def main():
    print("=" * 60)
    print("  CART DECISION TREE — FROM SCRATCH (Gini Impurity)")
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

    mss_in = input("Min samples to split [2]: ").strip()
    min_samples_split = int(mss_in) if mss_in.isdigit() else 2

    X_train, y_train, X_test, y_test = stratified_split(features, labels, test_ratio)
    print(f"\n  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")

    print("\n  Training CART tree...")
    tree = build_tree(X_train, y_train, feature_names[:],
                      max_depth=max_depth, min_samples_split=min_samples_split)
    print("  Tree built successfully.")

    print("\n  Decision Tree Structure")
    print("=" * 60)
    print_tree(tree)

    print_feature_importances(tree, feature_names)

    print("\n  Evaluating...")
    tr_preds, tr_acc, classes = evaluate(tree, X_train, y_train, "Train")
    te_preds, te_acc, _       = evaluate(tree, X_test,  y_test,  "Test")

    out_dir = input("\nOutput folder for plots [./plots_cart]: ").strip() or "./plots_cart"
    os.makedirs(out_dir, exist_ok=True)

    print("  Generating plots...")
    plot_confusion_matrix(tr_preds, y_train, classes, out_dir, "train")
    plot_confusion_matrix(te_preds, y_test,  classes, out_dir, "test")
    plot_gini_importances(tree, feature_names, out_dir)
    plot_gini_vs_entropy(X_train, y_train, feature_names, out_dir)
    plot_label_distribution(y_train, y_test, out_dir)
    plot_per_feature_class_dist(X_train, y_train, feature_names, out_dir)
    plot_accuracy_summary(tr_acc, te_acc, out_dir)
    print(f"  All plots saved to '{out_dir}/'")

    save = input("\nSave tree to JSON? [y/n]: ").strip().lower()
    if save == "y":
        jpath = input("Output path [cart_tree.json]: ").strip() or "cart_tree.json"
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