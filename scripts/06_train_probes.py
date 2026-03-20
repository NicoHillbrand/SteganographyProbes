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
from pathlib import Path

parser = argparse.ArgumentParser(description="Train linear probes on activations")
parser.add_argument("--task", type=str, required=True,
                    choices=["stego_detection", "game_scenarios"],
                    help="Which task's activations to probe")
parser.add_argument("--activations_dir", type=str, default="data/activations",
                    help="Directory containing activation .npy files")
parser.add_argument("--test_size", type=float, default=0.2)
parser.add_argument("--n_seeds", type=int, default=5,
                    help="Number of random seeds for train/test splits")
parser.add_argument("--output_dir", type=str, default="data/probe_results")
args = parser.parse_args()


def train_probe_at_layer(X, y, test_size, seed):
    """Train a logistic regression probe and return accuracy metrics."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
    from sklearn.preprocessing import StandardScaler

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


def main():
    from sklearn.linear_model import LogisticRegression  # fail fast if not installed

    task_dir = os.path.join(args.activations_dir, args.task)

    if not os.path.exists(task_dir):
        print(f"Activations not found at {task_dir}")
        print(f"Run scripts/05_run_inference.py first.")
        return

    # Load metadata
    with open(os.path.join(task_dir, "metadata.json")) as f:
        meta = json.load(f)

    labels = np.load(os.path.join(task_dir, "labels.npy"))
    layer_indices = meta["layer_indices"]

    print(f"Task: {args.task}")
    print(f"Examples: {len(labels)} ({labels.sum()} positive, {len(labels) - labels.sum()} negative)")
    print(f"Layers to probe: {len(layer_indices)}")
    print(f"Random seeds: {args.n_seeds}")
    print()

    # Train probes at each layer
    results_by_layer = {}

    for layer_idx in layer_indices:
        npy_path = os.path.join(task_dir, f"layer_{layer_idx:02d}.npy")
        if not os.path.exists(npy_path):
            print(f"  Layer {layer_idx}: file not found, skipping")
            continue

        X = np.load(npy_path)
        print(f"  Layer {layer_idx:2d}: shape {X.shape}", end="")

        # Run multiple seeds
        seed_results = []
        for seed in range(args.n_seeds):
            r = train_probe_at_layer(X, labels, args.test_size, seed=seed)
            seed_results.append(r)

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
        best_layer = max(results_by_layer, key=lambda l: results_by_layer[l]["mean_auc_roc"])
        best = results_by_layer[best_layer]

        print(f"\n{'='*60}")
        print(f"Best layer: {best_layer}")
        print(f"  Accuracy: {best['mean_accuracy']:.3f} (+/- {best['std_accuracy']:.3f})")
        print(f"  AUC-ROC:  {best['mean_auc_roc']:.3f}")
        print(f"  F1:       {best['mean_f1']:.3f}")
        print(f"{'='*60}")

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
        "token_position": meta.get("token_position", "unknown"),
        "best_layer": int(best_layer) if results_by_layer else None,
        "results_by_layer": {str(k): v for k, v in results_by_layer.items()},
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
