"""
Step 6: Train linear probes on activations to detect steganography.

Takes the activations saved by step 5 and trains logistic regression probes
to classify stego vs clean at each layer. This tells us WHERE in the model
the steganography signal is represented.

Can run on CPU — no GPU needed.

Usage:
    python scripts/06_train_probes.py --task stego_detection
    python scripts/06_train_probes.py --task game_scenarios
    python scripts/06_train_probes.py --task stego_detection --activations_dir data/activations
"""

import argparse
import json
import os
import numpy as np

parser = argparse.ArgumentParser(description="Train linear probes on activations")
parser.add_argument("--task", type=str, required=True,
                    choices=["stego_detection", "game_scenarios", "decode_task", "benign_task"],
                    help="Which task's activations to probe")
parser.add_argument("--activations_dir", type=str, default="data/activations",
                    help="Directory containing activation .npy files")
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--n_seeds", type=int, default=5,
                    help="Number of random seeds for train/test splits")
parser.add_argument("--output_dir", type=str, default="data/probe_results")
args = parser.parse_args()


def train_probe_at_layer(X, y, test_size, seed, groups=None):
    """Train a logistic regression probe and return accuracy metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    from sklearn.preprocessing import StandardScaler

    if groups is not None:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(gss.split(X, y, groups))
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
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
        "accuracy": accuracy_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, y_prob),
        "f1": f1_score(y_test, y_pred),
        "n_train": len(y_train),
        "n_test": len(y_test),
        "train_pos_rate": float(y_train.mean()),
        "test_pos_rate": float(y_test.mean()),
        "coef_norm": float(np.linalg.norm(probe.coef_)),
    }


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
        raise ValueError(f"Expected 2D activations for token_position='{token_position}', got shape {X.shape}")

    return X


def main():
    import sklearn.linear_model  # noqa: F401  # fail fast if scikit-learn is not installed

    task_dir = os.path.join(args.activations_dir, args.task)

    if not os.path.exists(task_dir):
        print(f"Activations not found at {task_dir}")
        print("Run scripts/05_run_inference.py first.")
        return

    # Load metadata
    with open(os.path.join(task_dir, "metadata.json")) as f:
        meta = json.load(f)

    labels = np.load(os.path.join(task_dir, "labels.npy"))
    layer_indices = meta["layer_indices"]
    token_position = meta.get("token_position", "unknown")
    sequence_lengths = None
    if token_position == "all":
        seq_path = meta.get("sequence_lengths_file")
        if not seq_path:
            print("Missing sequence lengths metadata for token_position='all'")
            return
        sequence_lengths = np.load(os.path.join(task_dir, seq_path))

    # T4: For game scenarios, filter to misaligned conditions only (same system
    # prompt), use observed keyword labels when available, and set up grouped
    # splits by carrier_group_id.
    groups = None
    methods_per_example = None
    if args.task == "game_scenarios":
        # Try v2 first, fall back to v1
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

            # Use observed labels from generation if available
            id_to_observed = {}
            if os.path.exists(responses_path):
                with open(responses_path, encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        r = json.loads(line)
                        if r.get("keyword_present") is not None:
                            id_to_observed[r["id"]] = int(r["keyword_present"])
                if id_to_observed:
                    print(f"Using observed labels from {responses_path} ({len(id_to_observed)} entries)")

            # Keep only misaligned conditions (same system prompt, no confound)
            keep_mask = np.array([
                id_to_scenario.get(eid, {}).get("condition", "") in ("misaligned_with_stego", "misaligned_no_stego")
                for eid in example_ids
            ])
            n_before = len(labels)
            labels = labels[keep_mask]
            example_ids_filtered = [eid for eid, k in zip(example_ids, keep_mask) if k]
            meta["ids"] = example_ids_filtered

            # Relabel from observed keyword if available
            if id_to_observed:
                labels = np.array([id_to_observed.get(eid, labels[i]) for i, eid in enumerate(example_ids_filtered)])

            print(f"Game scenarios: kept {len(labels)}/{n_before} (misaligned only)")

            # Extract carrier_group_id for grouped splits
            group_list = [id_to_scenario.get(eid, {}).get("carrier_group_id") for eid in example_ids_filtered]
            if all(g is not None for g in group_list):
                groups = np.array(group_list)
                print(f"Using grouped splits by carrier_group_id: {len(set(groups.tolist()))} unique groups")

            # Extract stego_method for per-method breakdown
            method_list = [id_to_scenario.get(eid, {}).get("stego_method") for eid in example_ids_filtered]
            # For clean scenarios stego_method is the prompt method (what it was told to look for)
            # For per-method breakdown, map misaligned_with_stego to their method, clean to "clean"
            condition_list = [id_to_scenario.get(eid, {}).get("condition", "") for eid in example_ids_filtered]
            method_labels = []
            for cond, method in zip(condition_list, method_list):
                if cond == "misaligned_with_stego" and method:
                    method_labels.append(method)
                else:
                    method_labels.append("clean")
            methods_per_example = np.array(method_labels)
            print(f"Per-method breakdown available: {dict(zip(*np.unique(methods_per_example, return_counts=True)))}")

            # Store the mask for slicing layer arrays
            _game_keep_mask = keep_mask
        else:
            _game_keep_mask = None
    else:
        _game_keep_mask = None

    # Load group IDs for grouped splitting (T1)
    # game_scenarios already set groups and methods_per_example above
    if args.task != "game_scenarios":
        groups = None
    dataset_path = None
    if args.task == "stego_detection":
        dataset_path = "data/combined_dataset.jsonl"

    # For decode/benign tasks, group IDs are stored directly in metadata
    if args.task in ("decode_task", "benign_task"):
        carrier_group_ids = meta.get("carrier_group_ids", [])
        if carrier_group_ids and all(g is not None for g in carrier_group_ids):
            groups = np.array(carrier_group_ids)
            print(f"Using grouped splits: {len(set(groups.tolist()))} unique groups")
    elif dataset_path and os.path.exists(dataset_path):
        id_to_group = {}
        with open(dataset_path) as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                gid = ex.get("carrier_group_id")
                if gid is None:
                    gid = ex.get("metadata", {}).get("base_text_index")
                if gid is not None:
                    id_to_group[ex["id"]] = gid

        example_ids = meta.get("ids", [])
        if example_ids and id_to_group:
            group_list = [id_to_group.get(eid) for eid in example_ids]
            if all(g is not None for g in group_list):
                groups = np.array(group_list)
                print(f"Using grouped splits: {len(set(groups.tolist()))} unique groups")
            else:
                print("Warning: some examples missing group IDs, falling back to random splits")

    # Load method info for per-method breakdown (T3)
    # game_scenarios already set methods_per_example above
    if args.task != "game_scenarios":
        methods_per_example = None
    if args.task in ("decode_task", "benign_task"):
        methods_list = meta.get("methods", {})
        example_ids = meta.get("ids", [])
        if example_ids and methods_list:
            # Reconstruct from the decode_task JSONL to get per-example methods
            decode_jsonl = "data/decode_task.jsonl"
            if os.path.exists(decode_jsonl):
                id_to_method = {}
                with open(decode_jsonl) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        ex = json.loads(line)
                        id_to_method[ex["id"]] = ex.get("method", "unknown")
                methods_per_example = np.array([id_to_method.get(eid, "unknown") for eid in example_ids])
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
            methods_per_example = np.array([id_to_method.get(eid, "unknown") for eid in example_ids])

    print(f"Task: {args.task}")
    print(f"Examples: {len(labels)} ({labels.sum()} positive, {len(labels) - labels.sum()} negative)")
    print(f"Layers to probe: {len(layer_indices)}")
    print(f"Token position: {token_position}")
    print(f"Random seeds: {args.n_seeds}")
    print()

    # Train probes at each layer
    results_by_layer = {}

    for layer_idx in layer_indices:
        npy_path = os.path.join(task_dir, f"layer_{layer_idx:02d}.npy")
        if not os.path.exists(npy_path):
            print(f"  Layer {layer_idx}: file not found, skipping")
            continue

        X_raw = np.load(npy_path)
        if _game_keep_mask is not None:
            X_raw = X_raw[_game_keep_mask]
        X = prepare_probe_features(X_raw, token_position, sequence_lengths)
        print(f"  Layer {layer_idx:2d}: shape {X_raw.shape} -> features {X.shape}", end="")

        # Run multiple seeds
        seed_results = []
        for seed in range(args.n_seeds):
            try:
                r = train_probe_at_layer(X, labels, args.test_size, seed=seed, groups=groups)
                seed_results.append(r)
            except ValueError as e:
                print(f"  seed {seed} failed: {e}")

        if not seed_results:
            print(f"  ->  all seeds failed, skipping")
            continue

        # Average across seeds
        avg_acc = np.mean([r["accuracy"] for r in seed_results])
        avg_auc = np.mean([r["auc_roc"] for r in seed_results])
        avg_f1 = np.mean([r["f1"] for r in seed_results])
        std_acc = np.std([r["accuracy"] for r in seed_results])

        print(f"  ->  acc={avg_acc:.3f} (+/-{std_acc:.3f})  auc={avg_auc:.3f}  f1={avg_f1:.3f}")

        results_by_layer[layer_idx] = {
            "mean_accuracy": float(avg_acc),
            "std_accuracy": float(std_acc),
            "mean_auc_roc": float(avg_auc),
            "mean_f1": float(avg_f1),
            "per_seed": seed_results,
        }

    # Find best layer
    if results_by_layer:
        best_layer = max(results_by_layer, key=lambda layer_idx: results_by_layer[layer_idx]["mean_auc_roc"])
        best = results_by_layer[best_layer]

        print(f"\n{'='*60}")
        print(f"Best layer: {best_layer}")
        print(f"  Accuracy: {best['mean_accuracy']:.3f} (+/- {best['std_accuracy']:.3f})")
        print(f"  AUC-ROC:  {best['mean_auc_roc']:.3f}")
        print(f"  F1:       {best['mean_f1']:.3f}")
        print(f"{'='*60}")

    # Per-method breakdown on best layer (T3)
    method_results = {}
    if methods_per_example is not None and results_by_layer:
        print(f"\n{'='*60}")
        print("Per-method breakdown (best layer):")

        best_layer_npy = os.path.join(task_dir, f"layer_{best_layer:02d}.npy")
        X_best = np.load(best_layer_npy)
        if _game_keep_mask is not None:
            X_best = X_best[_game_keep_mask]
        X_best = prepare_probe_features(X_best, token_position, sequence_lengths)

        unique_methods = sorted(set(methods_per_example))

        for method in unique_methods:
            # Get indices for this method + all clean examples
            if method == "clean":
                continue

            method_mask = (methods_per_example == method) | (methods_per_example == "clean")
            X_method = X_best[method_mask]
            y_method = labels[method_mask]
            groups_method = groups[method_mask] if groups is not None else None

            if len(set(y_method.tolist())) < 2:
                print(f"  {method:20s}: skipped (only one class)")
                continue

            method_seed_results = []
            for seed in range(args.n_seeds):
                try:
                    r = train_probe_at_layer(X_method, y_method, args.test_size, seed=seed, groups=groups_method)
                    method_seed_results.append(r)
                except Exception as e:
                    print(f"  {method:20s}: seed {seed} failed: {e}")

            if method_seed_results:
                avg_auc = np.mean([r["auc_roc"] for r in method_seed_results])
                avg_acc = np.mean([r["accuracy"] for r in method_seed_results])
                print(f"  {method:20s}: auc={avg_auc:.3f}  acc={avg_acc:.3f}  (n={int(method_mask.sum())})")
                method_results[method] = {
                    "mean_auc_roc": float(avg_auc),
                    "mean_accuracy": float(avg_acc),
                    "n_examples": int(method_mask.sum()),
                    "per_seed": method_seed_results,
                }

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{args.task}_probe_results.json")

    output = {
        "task": args.task,
        "model": meta.get("model", "unknown"),
        "n_examples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
        "n_seeds": args.n_seeds,
        "test_size": args.test_size,
        "token_position": token_position,
        "best_layer": int(best_layer) if results_by_layer else None,
        "results_by_layer": {str(k): v for k, v in results_by_layer.items()},
        "per_method_results": method_results if method_results else None,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
