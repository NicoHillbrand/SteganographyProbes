"""
P2 — Probe direction cosine similarity analysis (CPU only).

Loads all 32 saved LogisticRegression probes (from P0), extracts .coef_ vectors
(4096-dim each), computes 32x32 cosine similarity matrix. Identifies:
  - Whether a consistent "stego direction" exists across layers
  - Whether L3 is an outlier or part of a block structure
  - Which layers share the same linear direction

Requires: P0 completed with --save_probes.

Usage:
    python3 scripts/14_probe_direction_analysis.py --run_dir <path>
"""

import argparse
import json
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Path to P0 run directory")
    p.add_argument("--n_seeds", type=int, default=5)
    return p.parse_args()


def load_probe_directions(run_dir, n_layers=32, n_seeds=5):
    """Load coef_ vectors for all layers, averaged across seeds."""
    import joblib
    directions = {}
    probe_dir_base = Path(run_dir) / "probe_results_stego_vs_clean" / "trained_probes"
    if not probe_dir_base.exists():
        probe_dir_base = Path(run_dir) / "probe_results" / "trained_probes"

    for layer_idx in range(n_layers):
        layer_dir = probe_dir_base / f"layer_{layer_idx:02d}"
        coefs = []
        for seed in range(n_seeds):
            path = layer_dir / f"seed_{seed}.joblib"
            if path.exists():
                obj = joblib.load(path)
                coefs.append(obj["probe"].coef_[0])
        if coefs:
            mean_coef = np.mean(coefs, axis=0)
            directions[layer_idx] = mean_coef / (np.linalg.norm(mean_coef) + 1e-8)

    return directions


def main():
    args = parse_args()
    run_dir = REPO_ROOT / args.run_dir

    print(f"Loading probe directions from {run_dir}")
    directions = load_probe_directions(run_dir, n_seeds=args.n_seeds)
    n_layers = len(directions)
    print(f"Loaded {n_layers} layers")

    if n_layers == 0:
        print("ERROR: No probe files found. Did P0 complete with --save_probes?")
        return

    all_layers = sorted(directions.keys())
    dir_matrix = np.stack([directions[l] for l in all_layers])  # (n_layers, hidden_dim)

    # 32x32 cosine similarity matrix
    cos_sim = dir_matrix @ dir_matrix.T  # (n_layers, n_layers), rows are already unit vectors

    print("\nCosine similarity matrix (selected rows):")
    print(f"{'L':>4}", end="")
    for l in all_layers[::4]:
        print(f"  L{l:02d}", end="")
    print()
    for i, li in enumerate(all_layers):
        print(f"L{li:02d}:", end="")
        for j in range(0, len(all_layers), 4):
            print(f"  {cos_sim[i, j]:.3f}", end="")
        print()

    # Summary statistics
    off_diag = cos_sim[np.triu_indices(n_layers, k=1)]
    mean_cos = float(off_diag.mean())
    std_cos = float(off_diag.std())
    print(f"\nOff-diagonal cosine stats: mean={mean_cos:.3f}, std={std_cos:.3f}")

    # Find the most similar and most different pairs
    upper_mask = np.triu(np.ones((n_layers, n_layers), dtype=bool), k=1)
    sim_pairs = [(all_layers[i], all_layers[j], cos_sim[i, j])
                 for i in range(n_layers) for j in range(i+1, n_layers)]
    sim_pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nTop 5 most similar layer pairs:")
    for la, lb, s in sim_pairs[:5]:
        print(f"  L{la:02d} vs L{lb:02d}: cos={s:.4f}")

    print("\nTop 5 most different layer pairs:")
    for la, lb, s in sim_pairs[-5:]:
        print(f"  L{la:02d} vs L{lb:02d}: cos={s:.4f}")

    # L3 similarity to all other layers
    if 3 in directions:
        l3_idx = all_layers.index(3)
        l3_similarities = [(l, cos_sim[l3_idx, all_layers.index(l)]) for l in all_layers if l != 3]
        l3_similarities.sort(key=lambda x: x[1], reverse=True)
        print("\nL3 cosine similarity to other layers (top 5):")
        for l, s in l3_similarities[:5]:
            print(f"  L3 vs L{l:02d}: {s:.4f}")
        print("L3 cosine similarity to other layers (bottom 5):")
        for l, s in l3_similarities[-5:]:
            print(f"  L3 vs L{l:02d}: {s:.4f}")

    # Detect block structure: layers with mean cos > 0.5 to neighbors
    print("\nBlock structure (layers with high cos sim to L3):")
    if 3 in directions:
        l3_idx = all_layers.index(3)
        high_sim = [l for l in all_layers if l != 3 and cos_sim[l3_idx, all_layers.index(l)] > 0.5]
        print(f"  Layers with cos(L3, L?) > 0.5: {high_sim}")

    interpretation = (
        "mean_cos > 0.7: consistent global stego direction across all layers. "
        "mean_cos 0.3-0.7: direction shifts across layers (potentially multiple sub-computations). "
        "mean_cos < 0.3: independent layer-specific signals."
    )

    # Save
    output_path = REPO_ROOT / args.run_dir / "probe_results_stego_vs_clean" / "p2_probe_direction_analysis.json"
    results = {
        "run_dir": str(args.run_dir),
        "n_layers": n_layers,
        "n_seeds": args.n_seeds,
        "off_diag_cosine_mean": mean_cos,
        "off_diag_cosine_std": std_cos,
        "top5_similar_pairs": [(int(a), int(b), float(s)) for a, b, s in sim_pairs[:5]],
        "top5_different_pairs": [(int(a), int(b), float(s)) for a, b, s in sim_pairs[-5:]],
        "l3_similarities": {str(l): float(s) for l, s in l3_similarities} if 3 in directions else {},
        "cosine_similarity_matrix": cos_sim.tolist(),
        "layer_indices": all_layers,
        "interpretation": interpretation,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print(f"\n=== INTERPRETATION ===")
    print(f"Mean off-diagonal cosine: {mean_cos:.3f}")
    if mean_cos > 0.7:
        print("→ HIGH: consistent global stego direction across layers")
    elif mean_cos > 0.3:
        print("→ MEDIUM: direction shifts — multiple distinct sub-computations")
    else:
        print("→ LOW: layer-specific signals, not a single unified direction")


if __name__ == "__main__":
    main()
