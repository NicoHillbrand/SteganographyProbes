"""
Step 6: Train linear probes on activations to detect steganography.

Takes the activations saved by step 5 and trains logistic regression probes
to classify the task target at each layer.

Can run on CPU.

Usage:
    python scripts/06_train_probes.py --task stego_detection
    python scripts/06_train_probes.py --task decode_task --decode_rescore_mode anchored_last
    python scripts/06_train_probes.py --task stego_detection --activations_dir data/activations
"""

import argparse
import json
import os

import numpy as np

from decode_scoring import score_decode_response


SUBSET_PRESETS = {
    "all_methods": None,
    "synonym_only": ["synonym"],
    "hard_plus_acrostic": ["synonym", "acrostic"],
    "easy_methods": ["prefix_binary", "unicode", "whitespace"],
}

TARGET_LABELS = {
    "stego_detection": "is_stego",
    "decode_task": "recognized",
    "benign_task": "is_stego",
    "game_scenarios": "keyword_present",
}


parser = argparse.ArgumentParser(description="Train linear probes on activations")
parser.add_argument(
    "--run_dir",
    type=str,
    default=None,
    help="Path to a run directory (e.g. data/Meta-Llama-3-8B-Instruct/runs/2026-03-21_whitespace). "
         "If provided, reads activations from <run_dir>/activations/ and saves results to "
         "<run_dir>/probe_results/. Overrides --activations_dir and --output_dir.",
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Which task's activations to probe (subdirectory name under activations_dir). "
         "Not needed when using --run_dir.",
)
parser.add_argument(
    "--activations_dir",
    type=str,
    default="data/activations",
    help="Directory containing activation .npy files (legacy mode, use --run_dir instead)",
)
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument(
    "--n_seeds",
    type=int,
    default=5,
    help="Number of random seeds for train/test splits",
)
parser.add_argument("--output_dir", type=str, default="data/probe_results")
parser.add_argument(
    "--class_weight",
    type=str,
    default="auto",
    choices=["auto", "none", "balanced"],
    help="Class weighting mode for logistic regression",
)
parser.add_argument(
    "--decode_rescore_mode",
    type=str,
    default="saved_labels",
    choices=["saved_labels", "unique_anywhere", "anchored_last", "last_label"],
    help="For decode_task, optionally rebuild labels from saved responses",
)
parser.add_argument(
    "--metric_for_best_layer",
    type=str,
    default="auc_roc",
    choices=["auc_roc", "average_precision"],
    help="Metric used to choose the best layer",
)
parser.add_argument(
    "--output_suffix",
    type=str,
    default="",
    help="Optional suffix appended to the output filename before .json",
)
parser.add_argument(
    "--decode_dataset",
    type=str,
    default=None,
    help="Optional path to the decode-task JSONL used to recover per-example methods",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    default=None,
    help="Optional path to the stego-detection dataset JSONL used to recover groups/methods",
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


def fit_and_score_probe(X_train, y_train, X_test, y_test, seed, class_weight):
    """Fit a logistic regression probe and return rich evaluation metrics."""
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
        "train_pos_rate": float(np.mean(y_train)),
        "test_pos_rate": float(np.mean(y_test)),
        "coef_norm": float(np.linalg.norm(probe.coef_)),
        "class_weight": class_weight or "none",
        "positive_predictions_default": int(np.sum(y_pred)),
        "positive_predictions_threshold": int(np.sum(y_pred_threshold)),
    }


def train_probe_at_layer(X, y, test_size, seed, groups=None):
    """Train a probe on one layer and return metrics."""
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


def run_probe_suite(X, y, groups=None):
    """Run the full seeded evaluation suite on a feature matrix."""
    seed_results = []
    for seed in range(args.n_seeds):
        try:
            seed_results.append(
                train_probe_at_layer(X, y, args.test_size, seed=seed, groups=groups)
            )
        except ValueError as exc:
            print(f"  seed {seed} failed: {exc}")

    if not seed_results:
        return None

    return summarize_seed_results(seed_results)


def prepare_probe_features(X, token_position, sequence_lengths=None):
    """Convert saved activations into a 2D feature matrix for probe training."""
    if token_position == "all":
        if X.ndim != 3:
            raise ValueError(
                f"Expected 3D activations for token_position='all', got shape {X.shape}"
            )
        if sequence_lengths is None:
            raise ValueError("sequence_lengths are required when token_position='all'")

        X = X.copy()
        for i, seq_len in enumerate(sequence_lengths):
            X[i, int(seq_len):, :] = 0.0
        return X.reshape(X.shape[0], -1)

    if X.ndim != 2:
        raise ValueError(
            f"Expected 2D activations for token_position='{token_position}', got shape {X.shape}"
        )

    return X


def load_decode_methods(meta):
    """Reconstruct per-example decode methods from the decode task JSONL."""
    decode_jsonl = args.decode_dataset or meta.get("decode_task_path") or "data/decode_task.jsonl"
    if not os.path.exists(decode_jsonl):
        return None

    id_to_method = {}
    with open(decode_jsonl) as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            id_to_method[ex["id"]] = ex.get("method", "unknown")

    example_ids = meta.get("ids", [])
    if not example_ids:
        return None
    return np.array([id_to_method.get(eid, "unknown") for eid in example_ids])


def maybe_rescore_decode_labels(task_dir, meta, labels):
    """Optionally rebuild decode-task labels from saved responses."""
    if args.task != "decode_task" or args.decode_rescore_mode == "saved_labels":
        return labels, {}

    responses_path = os.path.join(task_dir, "responses.jsonl")
    if not os.path.exists(responses_path):
        print(
            f"Warning: requested decode rescoring but {responses_path} is missing; "
            "using saved labels"
        )
        return labels, {"decode_rescore_mode": "saved_labels"}

    id_to_record = {}
    with open(responses_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            id_to_record[record["id"]] = record

    example_ids = meta.get("ids", [])
    rescored = []
    for example_id in example_ids:
        record = id_to_record.get(example_id)
        if record is None:
            raise ValueError(f"Missing decode response for example id {example_id}")
        score = score_decode_response(
            record["response"],
            record.get("expected_color"),
            record["is_stego"],
            parse_mode=args.decode_rescore_mode,
        )
        rescored.append(1 if score["recognized"] else 0)

    rescored = np.array(rescored, dtype=np.int64)
    print(
        f"Decode rescoring ({args.decode_rescore_mode}): "
        f"{int(rescored.sum())}/{len(rescored)} positives"
    )
    return rescored, {
        "decode_rescore_mode": args.decode_rescore_mode,
        "rescored_positive_count": int(rescored.sum()),
        "rescored_positive_rate": float(np.mean(rescored)),
    }


def evaluate_subset_results(X_best, labels, methods_per_example, groups):
    """Evaluate pooled method subsets on the best layer."""
    subset_results = {}
    stego_methods = sorted(m for m in set(methods_per_example.tolist()) if m != "clean")

    for subset_name, subset_methods in SUBSET_PRESETS.items():
        selected_methods = stego_methods if subset_methods is None else subset_methods
        mask = np.isin(methods_per_example, selected_methods) | (methods_per_example == "clean")
        X_subset = X_best[mask]
        y_subset = labels[mask]
        groups_subset = groups[mask] if groups is not None else None

        if len(np.unique(y_subset)) < 2:
            continue

        summary = run_probe_suite(X_subset, y_subset, groups_subset)
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
            f"threshold_f1={summary['mean_threshold_f1']:.3f}"
        )

    return subset_results


def train_transfer_probe(X, y, methods_per_example, groups, train_method, test_method, seed):
    """Train on one method-vs-clean split and test on another method-vs-clean split."""
    eligible_mask = np.isin(methods_per_example, ["clean", train_method, test_method])
    eligible_groups = groups[eligible_mask]
    eligible_labels = y[eligible_mask]

    train_idx, test_idx = split_indices(
        np.zeros((len(eligible_groups), 1)),
        eligible_labels,
        args.test_size,
        seed,
        groups=eligible_groups,
    )
    train_groups = set(eligible_groups[train_idx].tolist())
    test_groups = set(eligible_groups[test_idx].tolist())

    train_mask = (
        np.isin(methods_per_example, ["clean", train_method])
        & np.isin(groups, list(train_groups))
    )
    test_mask = (
        np.isin(methods_per_example, ["clean", test_method])
        & np.isin(groups, list(test_groups))
    )

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    class_weight = resolve_class_weight(y_train)
    return fit_and_score_probe(X_train, y_train, X_test, y_test, seed, class_weight)


def evaluate_transfer_matrix(X_best, labels, methods_per_example, groups):
    """Evaluate a cross-method transfer matrix on the best layer."""
    stego_methods = sorted(m for m in set(methods_per_example.tolist()) if m != "clean")
    auc_matrix = []
    ap_matrix = []
    pair_results = {}

    print(f"\n{'=' * 60}")
    print("Cross-method transfer (best layer)")

    for train_method in stego_methods:
        auc_row = []
        ap_row = []
        pair_results[train_method] = {}
        for test_method in stego_methods:
            seed_results = []
            for seed in range(args.n_seeds):
                try:
                    seed_results.append(
                        train_transfer_probe(
                            X_best,
                            labels,
                            methods_per_example,
                            groups,
                            train_method,
                            test_method,
                            seed,
                        )
                    )
                except ValueError as exc:
                    print(f"  {train_method:15s}->{test_method:15s}: seed {seed} failed: {exc}")

            if seed_results:
                summary = summarize_seed_results(seed_results)
                pair_results[train_method][test_method] = summary
                auc_row.append(summary["mean_auc_roc"])
                ap_row.append(summary["mean_average_precision"])
                print(
                    f"  {train_method:15s}->{test_method:15s}: "
                    f"auc={summary['mean_auc_roc']:.3f}  ap={summary['mean_average_precision']:.3f}"
                )
            else:
                pair_results[train_method][test_method] = None
                auc_row.append(None)
                ap_row.append(None)

        auc_matrix.append(auc_row)
        ap_matrix.append(ap_row)

    return {
        "methods": stego_methods,
        "mean_auc_roc_matrix": auc_matrix,
        "mean_average_precision_matrix": ap_matrix,
        "pair_results": pair_results,
    }


def main():
    import sklearn.linear_model  # noqa: F401  # fail fast if scikit-learn is not installed

    # Resolve paths: --run_dir mode vs legacy --task mode
    task_type = None  # inferred task type for conditional logic
    if args.run_dir:
        task_dir = os.path.join(args.run_dir, "activations")
        output_dir = os.path.join(args.run_dir, "probe_results")
        # Load task type and name from config.json
        config_path = os.path.join(args.run_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                run_config = json.load(f)
            task_type = run_config.get("task_type")
            task_name = run_config.get("args", {}).get("scenarios") or os.path.basename(args.run_dir)
        else:
            task_name = os.path.basename(args.run_dir)
        # For decode tasks, activations are in a subdirectory
        if task_type == "decode_task":
            task_dir = os.path.join(args.run_dir, "activations", "decode_task")
        # Map task_type to args.task equivalent for downstream logic
        if not args.task and task_type:
            args.task = task_type
    elif args.task:
        task_dir = os.path.join(args.activations_dir, args.task)
        output_dir = args.output_dir
        task_name = args.task
    else:
        print("Error: provide either --run_dir or --task")
        return

    if not os.path.exists(task_dir):
        print(f"Activations not found at {task_dir}")
        print("Run scripts/05_run_inference.py first.")
        return

    with open(os.path.join(task_dir, "metadata.json")) as f:
        meta = json.load(f)

    labels = np.load(os.path.join(task_dir, "labels.npy"))
    labels, rescore_meta = maybe_rescore_decode_labels(task_dir, meta, labels)

    layer_indices = meta["layer_indices"]
    token_position = meta.get("token_position", "unknown")
    sequence_lengths = None
    if token_position == "all":
        seq_path = meta.get("sequence_lengths_file")
        if not seq_path:
            print("Missing sequence lengths metadata for token_position='all'")
            return
        sequence_lengths = np.load(os.path.join(task_dir, seq_path))

    # T4: For game/scenario tasks, filter to misaligned conditions only (same
    # system prompt), use observed keyword labels when available, and set up
    # grouped splits by carrier_group_id.
    #
    # Detect scenario-format tasks by checking if metadata has scenarios_path
    # (set by 05_run_inference.py for any --scenarios run) or by task name.
    groups = None
    methods_per_example = None
    _is_scenario_task = "scenarios_path" in meta or args.task in ("game_scenarios",)
    if _is_scenario_task:
        # Read scenario path from metadata (written by inference script),
        # fall back to well-known paths for backward compatibility
        scenarios_path = meta.get("scenarios_path")
        if not scenarios_path or not os.path.exists(scenarios_path):
            scenarios_path = "data/game_scenarios_v2.jsonl"
        if not os.path.exists(scenarios_path):
            scenarios_path = "data/game_scenarios.jsonl"
        responses_path = os.path.join(task_dir, "responses.jsonl")
        example_ids = meta.get("ids", [])

        if os.path.exists(scenarios_path) and example_ids:
            id_to_scenario = {}
            with open(scenarios_path, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    s = json.loads(line)
                    id_to_scenario[s["id"]] = s

            id_to_observed = {}
            if os.path.exists(responses_path):
                with open(responses_path, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        if record.get("keyword_present") is not None:
                            id_to_observed[record["id"]] = int(record["keyword_present"])
                if id_to_observed:
                    print(
                        f"Using observed labels from {responses_path} "
                        f"({len(id_to_observed)} entries)"
                    )

            # Keep only misaligned conditions (same system prompt, no confound)
            keep_mask = np.array([
                id_to_scenario.get(eid, {}).get("condition", "") in ("misaligned_with_stego", "misaligned_no_stego")
                for eid in example_ids
            ])
            n_before = len(labels)
            labels = labels[keep_mask]
            filtered_ids = [eid for eid, keep in zip(example_ids, keep_mask) if keep]
            meta["ids"] = filtered_ids

            if id_to_observed:
                labels = np.array(
                    [id_to_observed.get(eid, labels[idx]) for idx, eid in enumerate(filtered_ids)]
                )

            print(f"Game scenarios: kept {len(labels)}/{n_before} (misaligned only)")

            # Extract carrier_group_id for grouped splits
            group_list = [id_to_scenario.get(eid, {}).get("carrier_group_id") for eid in filtered_ids]
            if all(g is not None for g in group_list):
                groups = np.array(group_list)
                print(f"Using grouped splits by carrier_group_id: {len(set(groups.tolist()))} unique groups")

            # Extract stego_method for per-method breakdown
            method_list = [id_to_scenario.get(eid, {}).get("stego_method") for eid in filtered_ids]
            condition_list = [id_to_scenario.get(eid, {}).get("condition", "") for eid in filtered_ids]
            method_labels = []
            for cond, method in zip(condition_list, method_list):
                if cond == "misaligned_with_stego" and method:
                    method_labels.append(method)
                else:
                    method_labels.append("clean")
            methods_per_example = np.array(method_labels)
            print(f"Per-method breakdown available: {dict(zip(*np.unique(methods_per_example, return_counts=True)))}")

            # Store the mask for slicing layer arrays and sequence lengths
            game_keep_mask = keep_mask
            if sequence_lengths is not None:
                sequence_lengths = sequence_lengths[keep_mask]
        else:
            game_keep_mask = None
    else:
        game_keep_mask = None

    # Load group IDs for grouped splitting (T1)
    # scenario tasks already set groups and methods_per_example above
    if not _is_scenario_task:
        groups = None
    dataset_path = None
    if args.task == "stego_detection":
        dataset_path = args.dataset_path or meta.get("dataset_path") or "data/combined_dataset.jsonl"

    if args.task in ("decode_task", "benign_task"):
        carrier_group_ids = meta.get("carrier_group_ids", [])
        if carrier_group_ids and all(group is not None for group in carrier_group_ids):
            groups = np.array(carrier_group_ids)
            print(f"Using grouped splits: {len(set(groups.tolist()))} unique groups")
    elif args.task == "stego_detection":
        carrier_group_ids = meta.get("carrier_group_ids", [])
        if carrier_group_ids and all(group is not None for group in carrier_group_ids):
            groups = np.array(carrier_group_ids)
            print(f"Using grouped splits: {len(set(groups.tolist()))} unique groups")
    elif dataset_path and os.path.exists(dataset_path):
        id_to_group = {}
        with open(dataset_path) as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                group_id = ex.get("carrier_group_id")
                if group_id is None:
                    group_id = ex.get("metadata", {}).get("base_text_index")
                if group_id is not None:
                    id_to_group[ex["id"]] = group_id

        example_ids = meta.get("ids", [])
        if example_ids and id_to_group:
            group_list = [id_to_group.get(example_id) for example_id in example_ids]
            if all(group is not None for group in group_list):
                groups = np.array(group_list)
                print(f"Using grouped splits: {len(set(groups.tolist()))} unique groups")
            else:
                print("Warning: some examples missing group IDs, falling back to random splits")

    # Load method info for per-method breakdown (T3)
    # scenario tasks already set methods_per_example above
    if not _is_scenario_task:
        methods_per_example = None
    if args.task in ("decode_task", "benign_task"):
        methods_per_example = load_decode_methods(meta)
    elif args.task == "stego_detection" and meta.get("methods_per_example"):
        methods_per_example = np.array(meta["methods_per_example"])
    elif dataset_path and os.path.exists(dataset_path):
        id_to_method = {}
        with open(dataset_path) as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                id_to_method[ex["id"]] = ex.get("method", "unknown")

        example_ids = meta.get("ids", [])
        if example_ids:
            methods_per_example = np.array(
                [id_to_method.get(example_id, "unknown") for example_id in example_ids]
            )

    print(f"Task: {args.task}")
    print(f"Target label: {TARGET_LABELS.get(args.task, 'label')}")
    print(f"Examples: {len(labels)} ({int(labels.sum())} positive, {len(labels) - int(labels.sum())} negative)")
    print(f"Layers to probe: {len(layer_indices)}")
    print(f"Token position: {token_position}")
    print(f"Random seeds: {args.n_seeds}")
    print(f"Class weight mode: {args.class_weight}")
    print()

    results_by_layer = {}

    for layer_idx in layer_indices:
        npy_path = os.path.join(task_dir, f"layer_{layer_idx:02d}.npy")
        if not os.path.exists(npy_path):
            print(f"  Layer {layer_idx}: file not found, skipping")
            continue

        X_raw = np.load(npy_path)
        if game_keep_mask is not None:
            X_raw = X_raw[game_keep_mask]
        X = prepare_probe_features(X_raw, token_position, sequence_lengths)
        print(f"  Layer {layer_idx:2d}: shape {X_raw.shape} -> features {X.shape}", end="")

        summary = run_probe_suite(X, labels, groups)
        if summary is None:
            print("  -> all seeds failed, skipping")
            continue

        print(
            f"  -> acc={summary['mean_accuracy']:.3f} (+/-{summary['std_accuracy']:.3f})  "
            f"auc={summary['mean_auc_roc']:.3f}  ap={summary['mean_average_precision']:.3f}  "
            f"f1={summary['mean_f1']:.3f}  thr_f1={summary['mean_threshold_f1']:.3f}"
        )
        results_by_layer[layer_idx] = summary

    best_layer = None
    if results_by_layer:
        best_layer = max(
            results_by_layer,
            key=lambda layer_idx: results_by_layer[layer_idx][f"mean_{args.metric_for_best_layer}"],
        )
        best = results_by_layer[best_layer]

        print(f"\n{'=' * 60}")
        print(f"Best layer: {best_layer}")
        print(f"  Accuracy:          {best['mean_accuracy']:.3f} (+/- {best['std_accuracy']:.3f})")
        print(f"  AUC-ROC:           {best['mean_auc_roc']:.3f}")
        print(f"  Average Precision: {best['mean_average_precision']:.3f}")
        print(f"  F1 @ 0.5:          {best['mean_f1']:.3f}")
        print(f"  F1 @ tuned thr:    {best['mean_threshold_f1']:.3f}")
        print(f"{'=' * 60}")

    method_results = {}
    subset_results = None
    transfer_results = None
    if methods_per_example is not None and results_by_layer:
        print(f"\n{'=' * 60}")
        print("Per-method breakdown (best layer)")

        best_layer_npy = os.path.join(task_dir, f"layer_{best_layer:02d}.npy")
        X_best = np.load(best_layer_npy)
        if game_keep_mask is not None:
            X_best = X_best[game_keep_mask]
        X_best = prepare_probe_features(X_best, token_position, sequence_lengths)

        unique_methods = sorted(set(methods_per_example.tolist()))
        for method in unique_methods:
            if method == "clean":
                continue

            method_mask = (methods_per_example == method) | (methods_per_example == "clean")
            X_method = X_best[method_mask]
            y_method = labels[method_mask]
            groups_method = groups[method_mask] if groups is not None else None

            if len(np.unique(y_method)) < 2:
                print(f"  {method:20s}: skipped (only one class)")
                continue

            summary = run_probe_suite(X_method, y_method, groups_method)
            if summary is None:
                continue

            method_results[method] = {
                "n_examples": int(method_mask.sum()),
                **summary,
            }
            print(
                f"  {method:20s}: auc={summary['mean_auc_roc']:.3f}  "
                f"ap={summary['mean_average_precision']:.3f}  "
                f"thr_f1={summary['mean_threshold_f1']:.3f}  "
                f"(n={int(method_mask.sum())})"
            )

        if args.task == "stego_detection":
            print(f"\n{'=' * 60}")
            print("Method-subset breakdown (best layer)")
            subset_results = evaluate_subset_results(X_best, labels, methods_per_example, groups)

            if groups is not None:
                transfer_results = evaluate_transfer_matrix(
                    X_best, labels, methods_per_example, groups
                )

    os.makedirs(output_dir, exist_ok=True)
    suffix = args.output_suffix
    if suffix and not suffix.startswith("_"):
        suffix = f"_{suffix}"
    if args.run_dir:
        output_path = os.path.join(output_dir, f"probe_results{suffix}.json")
    else:
        output_path = os.path.join(output_dir, f"{args.task}_probe_results{suffix}.json")

    output = {
        "task": task_name,
        "target_label": TARGET_LABELS.get(args.task, "label") if args.task else "label",
        "model": meta.get("model", "unknown"),
        "n_examples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "n_seeds": args.n_seeds,
        "test_size": args.test_size,
        "token_position": token_position,
        "best_layer": int(best_layer) if best_layer is not None else None,
        "best_layer_metric": args.metric_for_best_layer,
        "class_weight_mode": args.class_weight,
        "decode_rescore_mode": args.decode_rescore_mode,
        "results_by_layer": {str(k): v for k, v in results_by_layer.items()},
        "per_method_results": method_results if method_results else None,
        "subset_results": subset_results,
        "transfer_results": transfer_results,
        "config": {
            "class_weight": args.class_weight,
            "metric_for_best_layer": args.metric_for_best_layer,
            "decode_rescore_mode": args.decode_rescore_mode,
            "decode_dataset": args.decode_dataset,
            "dataset_path": args.dataset_path,
        },
        "metadata_overrides": rescore_meta or None,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
