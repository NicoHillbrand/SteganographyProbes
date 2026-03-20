"""
Step 7: Text-only baselines for steganography detection.

Tests whether surface-level text features can distinguish stego from clean,
providing a lower bound that activation probes must beat to be meaningful.

Two baselines:
  1. Character features: length, tab_count, zwc_count, digit_ratio, newline_count, cap_ratio
  2. Character 3-gram TF-IDF + logistic regression

Uses the same grouped splits as the activation probes for fair comparison.

Usage:
    python scripts/07_text_baselines.py --dataset data/combined_dataset.jsonl
"""

import argparse
import json
import os
import numpy as np

parser = argparse.ArgumentParser(description="Text-only baselines for stego detection")
parser.add_argument("--dataset", type=str, default="data/combined_dataset.jsonl")
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--n_seeds", type=int, default=5)
parser.add_argument("--output_dir", type=str, default="data/probe_results")
args = parser.parse_args()


ZW_CHARS = {"\u200c", "\u200d", "\u200b", "\u2060", "\ufeff"}


def extract_char_features(text):
    """Extract surface-level character features from text."""
    length = len(text)
    tab_count = text.count("\t")
    zwc_count = sum(1 for c in text if c in ZW_CHARS)
    digit_ratio = sum(1 for c in text if c.isdigit()) / max(length, 1)
    newline_count = text.count("\n")
    cap_ratio = sum(1 for c in text if c.isupper()) / max(length, 1)
    space_ratio = sum(1 for c in text if c == " ") / max(length, 1)
    punct_ratio = sum(1 for c in text if c in ".,;:!?-()[]{}\"'") / max(length, 1)

    return [length, tab_count, zwc_count, digit_ratio, newline_count,
            cap_ratio, space_ratio, punct_ratio]


CHAR_FEATURE_NAMES = [
    "length", "tab_count", "zwc_count", "digit_ratio",
    "newline_count", "cap_ratio", "space_ratio", "punct_ratio"
]


def train_baseline(X, y, test_size, seed, groups=None):
    """Train logistic regression baseline and return metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    from sklearn.preprocessing import StandardScaler

    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    y_prob = probe.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def train_tfidf_baseline(texts, y, test_size, seed, groups=None):
    """Train TF-IDF + logistic regression, fitting vectorizer on train data only."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    from sklearn.preprocessing import StandardScaler

    texts_arr = np.array(texts)

    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(texts_arr, y, groups))
    else:
        from sklearn.model_selection import train_test_split
        indices = np.arange(len(texts))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=y
        )

    train_texts = [texts[i] for i in train_idx]
    test_texts = [texts[i] for i in test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=10000)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    probe = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_test)
    y_prob = probe.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "f1": float(f1_score(y_test, y_pred)),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": X_train.shape[1],
    }


def main():
    # Load dataset
    examples = []
    with open(args.dataset) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    texts = [ex["response"] for ex in examples]
    labels = np.array([1 if ex["is_stego"] else 0 for ex in examples])
    methods = [ex.get("method", "unknown") for ex in examples]

    # Load group IDs for grouped splitting
    groups = None
    group_list = []
    for ex in examples:
        gid = ex.get("carrier_group_id")
        if gid is None:
            gid = ex.get("metadata", {}).get("base_text_index")
        group_list.append(gid)

    if all(g is not None for g in group_list):
        groups = np.array(group_list)
        print(f"Using grouped splits: {len(set(groups.tolist()))} unique groups")
    else:
        print("No group IDs found, using random splits")

    print(f"Dataset: {len(examples)} examples ({labels.sum()} stego, {len(labels) - labels.sum()} clean)")
    print(f"Seeds: {args.n_seeds}")
    print()

    results = {}

    # === Baseline 1: Character features ===
    print("=" * 60)
    print("Baseline 1: Character features")
    print("=" * 60)

    X_char = np.array([extract_char_features(t) for t in texts])
    print(f"Feature matrix: {X_char.shape}")

    char_seed_results = []
    for seed in range(args.n_seeds):
        r = train_baseline(X_char, labels, args.test_size, seed, groups)
        char_seed_results.append(r)

    avg_auc = np.mean([r["auc_roc"] for r in char_seed_results])
    avg_acc = np.mean([r["accuracy"] for r in char_seed_results])
    avg_f1 = np.mean([r["f1"] for r in char_seed_results])
    print(f"  AUC-ROC: {avg_auc:.3f}")
    print(f"  Accuracy: {avg_acc:.3f}")
    print(f"  F1: {avg_f1:.3f}")

    results["char_features"] = {
        "features": CHAR_FEATURE_NAMES,
        "mean_auc_roc": float(avg_auc),
        "mean_accuracy": float(avg_acc),
        "mean_f1": float(avg_f1),
        "per_seed": char_seed_results,
    }

    # === Baseline 2: Character 3-gram TF-IDF ===
    print()
    print("=" * 60)
    print("Baseline 2: Character 3-gram TF-IDF (fit on train only)")
    print("=" * 60)

    tfidf_seed_results = []
    for seed in range(args.n_seeds):
        r = train_tfidf_baseline(texts, labels, args.test_size, seed, groups)
        tfidf_seed_results.append(r)

    avg_auc = np.mean([r["auc_roc"] for r in tfidf_seed_results])
    avg_acc = np.mean([r["accuracy"] for r in tfidf_seed_results])
    avg_f1 = np.mean([r["f1"] for r in tfidf_seed_results])
    print(f"  AUC-ROC: {avg_auc:.3f}")
    print(f"  Accuracy: {avg_acc:.3f}")
    print(f"  F1: {avg_f1:.3f}")

    results["tfidf_char_ngram"] = {
        "ngram_range": [2, 4],
        "max_features": 10000,
        "mean_auc_roc": float(avg_auc),
        "mean_accuracy": float(avg_acc),
        "mean_f1": float(avg_f1),
        "per_seed": tfidf_seed_results,
    }

    # === Per-method breakdown (char features) ===
    print()
    print("=" * 60)
    print("Per-method breakdown (char features)")
    print("=" * 60)

    unique_methods = sorted(set(methods))
    method_results = {}

    for method in unique_methods:
        if method == "clean":
            continue

        mask = np.array([(m == method or m == "clean") for m in methods])
        X_m = X_char[mask]
        y_m = labels[mask]
        groups_m = groups[mask] if groups is not None else None

        if len(set(y_m.tolist())) < 2:
            continue

        method_seeds = []
        for seed in range(args.n_seeds):
            try:
                r = train_baseline(X_m, y_m, args.test_size, seed, groups_m)
                method_seeds.append(r)
            except Exception:
                pass

        if method_seeds:
            m_auc = np.mean([r["auc_roc"] for r in method_seeds])
            m_acc = np.mean([r["accuracy"] for r in method_seeds])
            print(f"  {method:20s}: auc={m_auc:.3f}  acc={m_acc:.3f}  (n={mask.sum()})")
            method_results[method] = {
                "mean_auc_roc": float(m_auc),
                "mean_accuracy": float(m_acc),
                "n_examples": int(mask.sum()),
            }

    results["per_method_char"] = method_results

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "text_baseline_results.json")

    output = {
        "dataset": args.dataset,
        "n_examples": len(examples),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "grouped_splits": groups is not None,
        "n_groups": int(len(set(groups.tolist()))) if groups is not None else None,
        "n_seeds": args.n_seeds,
        "test_size": args.test_size,
        "baselines": results,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
