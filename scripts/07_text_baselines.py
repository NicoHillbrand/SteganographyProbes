"""
Step 7: Text-only baselines for steganography detection.

Tests whether surface-level text features can distinguish stego from clean,
providing a lower bound that activation probes must beat to be meaningful.

Two baselines:
  1. Character features
  2. Character n-gram TF-IDF + logistic regression

Uses the same grouped splits as the activation probes for fair comparison.

Usage:
    python scripts/07_text_baselines.py --dataset data/combined_dataset.jsonl
"""

import argparse
import json
import os

import numpy as np


SUBSET_PRESETS = {
    "all_methods": None,
    "synonym_only": ["synonym"],
    "hard_plus_acrostic": ["synonym", "acrostic"],
    "easy_methods": ["prefix_binary", "unicode", "whitespace"],
}

ZW_CHARS = {"\u200c", "\u200d", "\u200b", "\u2060", "\ufeff"}
CHAR_FEATURE_NAMES = [
    "length",
    "tab_count",
    "zwc_count",
    "digit_ratio",
    "newline_count",
    "cap_ratio",
    "space_ratio",
    "punct_ratio",
]


parser = argparse.ArgumentParser(description="Text-only baselines for steganography detection")
parser.add_argument("--dataset", type=str, default="data/combined_dataset.jsonl")
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--n_seeds", type=int, default=5)
parser.add_argument("--output_dir", type=str, default="data/probe_results")
parser.add_argument("--run_dir", type=str, default=None,
                    help="If provided, save results to <run_dir>/text_baselines/")
parser.add_argument(
    "--class_weight",
    type=str,
    default="auto",
    choices=["auto", "none", "balanced"],
    help="Class weighting mode for logistic regression",
)
parser.add_argument(
    "--output_suffix",
    type=str,
    default="",
    help="Optional suffix appended to the output filename before .json",
)
args = parser.parse_args()


def resolve_class_weight(y):
    """Resolve the class-weight setting for the current label distribution."""
    if args.class_weight == "none":
        return None
    if args.class_weight == "balanced":
        return "balanced"

    pos_rate = float(np.mean(y))
    return "balanced" if pos_rate < 0.1 or pos_rate > 0.9 else None


def split_indices(X, y, test_size, seed, groups=None):
    """Return train/test indices for the requested split strategy."""
    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit

        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        return next(gss.split(X, y, groups))

    from sklearn.model_selection import train_test_split

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=y
    )
    return train_idx, test_idx


def select_threshold(y_true, y_prob):
    """Select a probability threshold by maximizing train F1."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if thresholds.size == 0:
        return 0.5, 0.0

    precision = precision[:-1]
    recall = recall[:-1]
    denom = precision + recall
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_scores = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def summarize_seed_results(seed_results):
    """Aggregate a list of per-seed metrics into a summary dictionary."""
    metrics = [
        "accuracy",
        "auc_roc",
        "average_precision",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "threshold_accuracy",
        "threshold_f1",
        "threshold_precision",
        "threshold_recall",
    ]

    summary = {"per_seed": seed_results}
    for metric in metrics:
        values = [r[metric] for r in seed_results]
        summary[f"mean_{metric}"] = float(np.mean(values))
        summary[f"std_{metric}"] = float(np.std(values))

    summary["class_weight_mode"] = seed_results[0].get("class_weight", "none")
    return summary


def extract_char_features(text):
    """Extract surface-level character features from text."""
    length = len(text)
    tab_count = text.count("\t")
    zwc_count = sum(1 for char in text if char in ZW_CHARS)
    digit_ratio = sum(1 for char in text if char.isdigit()) / max(length, 1)
    newline_count = text.count("\n")
    cap_ratio = sum(1 for char in text if char.isupper()) / max(length, 1)
    space_ratio = sum(1 for char in text if char == " ") / max(length, 1)
    punct_ratio = sum(1 for char in text if char in ".,;:!?-()[]{}\"'") / max(length, 1)

    return [
        length,
        tab_count,
        zwc_count,
        digit_ratio,
        newline_count,
        cap_ratio,
        space_ratio,
        punct_ratio,
    ]


def fit_and_score_probe(X_train, y_train, X_test, y_test, seed, class_weight):
    """Fit a logistic regression baseline and return evaluation metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        raise ValueError("train/test split must contain both classes")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    probe = LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=seed,
        class_weight=class_weight,
    )
    probe.fit(X_train, y_train)

    y_train_prob = probe.predict_proba(X_train)[:, 1]
    y_prob = probe.predict_proba(X_test)[:, 1]
    y_pred = probe.predict(X_test)

    threshold, train_best_f1 = select_threshold(y_train, y_train_prob)
    y_pred_threshold = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "threshold_selected": float(threshold),
        "threshold_train_best_f1": float(train_best_f1),
        "threshold_accuracy": float(accuracy_score(y_test, y_pred_threshold)),
        "threshold_f1": float(f1_score(y_test, y_pred_threshold, zero_division=0)),
        "threshold_precision": float(
            precision_score(y_test, y_pred_threshold, zero_division=0)
        ),
        "threshold_recall": float(
            recall_score(y_test, y_pred_threshold, zero_division=0)
        ),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "class_weight": class_weight or "none",
    }


def train_baseline(X, y, test_size, groups=None, seed=0):
    """Train a baseline probe from a dense feature matrix."""
    class_weight = resolve_class_weight(y)
    train_idx, test_idx = split_indices(X, y, test_size, seed, groups)
    return fit_and_score_probe(
        X[train_idx],
        y[train_idx],
        X[test_idx],
        y[test_idx],
        seed,
        class_weight,
    )


def train_tfidf_baseline(texts, y, test_size, groups=None, seed=0):
    """Train TF-IDF + logistic regression, fitting vectorizer on train data only."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts_arr = np.array(texts)
    class_weight = resolve_class_weight(y)
    train_idx, test_idx = split_indices(texts_arr, y, test_size, seed, groups)

    train_texts = [texts[idx] for idx in train_idx]
    test_texts = [texts[idx] for idx in test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=10000)
    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_test = vectorizer.transform(test_texts).toarray()

    result = fit_and_score_probe(X_train, y_train, X_test, y_test, seed, class_weight)
    result["n_features"] = int(X_train.shape[1])
    return result


def run_seeded_suite(train_fn, *train_args):
    """Run a seeded baseline suite using the provided trainer function."""
    seed_results = []
    for seed in range(args.n_seeds):
        try:
            seed_results.append(train_fn(*train_args, seed=seed))
        except ValueError as exc:
            print(f"  seed {seed} failed: {exc}")

    if not seed_results:
        return None
    return summarize_seed_results(seed_results)


def evaluate_subset_results(X_char, labels, methods, groups):
    """Evaluate pooled method subsets using the char-feature baseline."""
    subset_results = {}
    stego_methods = sorted(method for method in set(methods) if method != "clean")

    for subset_name, subset_methods in SUBSET_PRESETS.items():
        selected_methods = stego_methods if subset_methods is None else subset_methods
        mask = np.array(
            [(method in selected_methods) or (method == "clean") for method in methods]
        )
        X_subset = X_char[mask]
        y_subset = labels[mask]
        groups_subset = groups[mask] if groups is not None else None

        if len(np.unique(y_subset)) < 2:
            continue

        summary = run_seeded_suite(train_baseline, X_subset, y_subset, args.test_size, groups_subset)
        if summary is None:
            continue

        subset_results[subset_name] = {
            "methods": list(selected_methods),
            "n_examples": int(mask.sum()),
            **summary,
        }
        print(
            f"  {subset_name:20s}: auc={summary['mean_auc_roc']:.3f}  "
            f"ap={summary['mean_average_precision']:.3f}  "
            f"thr_f1={summary['mean_threshold_f1']:.3f}"
        )

    return subset_results


def main():
    examples = []
    with open(args.dataset) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    texts = [ex["response"] for ex in examples]
    labels = np.array([1 if ex["is_stego"] else 0 for ex in examples])
    methods = [ex.get("method", "unknown") for ex in examples]

    groups = None
    group_list = []
    for ex in examples:
        group_id = ex.get("carrier_group_id")
        if group_id is None:
            group_id = ex.get("metadata", {}).get("base_text_index")
        group_list.append(group_id)

    if all(group is not None for group in group_list):
        groups = np.array(group_list)
        print(f"Using grouped splits: {len(set(groups.tolist()))} unique groups")
    else:
        print("No group IDs found, using random splits")

    print(
        f"Dataset: {len(examples)} examples "
        f"({int(labels.sum())} stego, {len(labels) - int(labels.sum())} clean)"
    )
    print(f"Seeds: {args.n_seeds}")
    print(f"Class weight mode: {args.class_weight}")
    print()

    results = {}

    print("=" * 60)
    print("Baseline 1: Character features")
    print("=" * 60)
    X_char = np.array([extract_char_features(text) for text in texts])
    print(f"Feature matrix: {X_char.shape}")

    char_summary = run_seeded_suite(train_baseline, X_char, labels, args.test_size, groups)
    if char_summary is None:
        raise SystemExit("Character baseline failed for all seeds")

    print(f"  AUC-ROC:           {char_summary['mean_auc_roc']:.3f}")
    print(f"  Average Precision: {char_summary['mean_average_precision']:.3f}")
    print(f"  Accuracy:          {char_summary['mean_accuracy']:.3f}")
    print(f"  F1 @ 0.5:          {char_summary['mean_f1']:.3f}")
    print(f"  F1 @ tuned thr:    {char_summary['mean_threshold_f1']:.3f}")

    results["char_features"] = {
        "features": CHAR_FEATURE_NAMES,
        **char_summary,
    }

    print()
    print("=" * 60)
    print("Baseline 2: Character n-gram TF-IDF (fit on train only)")
    print("=" * 60)

    tfidf_summary = run_seeded_suite(
        train_tfidf_baseline,
        texts,
        labels,
        args.test_size,
        groups,
    )
    if tfidf_summary is None:
        raise SystemExit("TF-IDF baseline failed for all seeds")

    print(f"  AUC-ROC:           {tfidf_summary['mean_auc_roc']:.3f}")
    print(f"  Average Precision: {tfidf_summary['mean_average_precision']:.3f}")
    print(f"  Accuracy:          {tfidf_summary['mean_accuracy']:.3f}")
    print(f"  F1 @ 0.5:          {tfidf_summary['mean_f1']:.3f}")
    print(f"  F1 @ tuned thr:    {tfidf_summary['mean_threshold_f1']:.3f}")

    results["tfidf_char_ngram"] = {
        "ngram_range": [2, 4],
        "max_features": 10000,
        **tfidf_summary,
    }

    print()
    print("=" * 60)
    print("Per-method breakdown (char features)")
    print("=" * 60)

    method_results = {}
    for method in sorted(set(methods)):
        if method == "clean":
            continue

        mask = np.array([(m == method or m == "clean") for m in methods])
        X_method = X_char[mask]
        y_method = labels[mask]
        groups_method = groups[mask] if groups is not None else None

        if len(np.unique(y_method)) < 2:
            continue

        summary = run_seeded_suite(train_baseline, X_method, y_method, args.test_size, groups_method)
        if summary is None:
            continue

        print(
            f"  {method:20s}: auc={summary['mean_auc_roc']:.3f}  "
            f"ap={summary['mean_average_precision']:.3f}  "
            f"thr_f1={summary['mean_threshold_f1']:.3f}  "
            f"(n={int(mask.sum())})"
        )
        method_results[method] = {
            "n_examples": int(mask.sum()),
            **summary,
        }

    print()
    print("=" * 60)
    print("Method-subset breakdown (char features)")
    print("=" * 60)
    subset_results = evaluate_subset_results(X_char, labels, methods, groups)

    out_dir = os.path.join(args.run_dir, "text_baselines") if args.run_dir else args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    suffix = args.output_suffix
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    output_path = os.path.join(out_dir, f"text_baseline_results{suffix}.json")

    output = {
        "dataset": args.dataset,
        "target_label": "is_stego",
        "n_examples": int(len(examples)),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "grouped_splits": groups is not None,
        "n_groups": int(len(set(groups.tolist()))) if groups is not None else None,
        "n_seeds": args.n_seeds,
        "test_size": args.test_size,
        "class_weight_mode": args.class_weight,
        "baselines": {
            **results,
            "per_method_char": method_results,
            "subset_char": subset_results,
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
