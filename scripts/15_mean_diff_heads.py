"""
P4 — Signed mean-difference layer attribution (Qwen3-32B, CPU only).

Computes ||mean(stego_acts) - mean(clean_acts)||_2 per layer using existing
Qwen3-32B suffix_marker activations. Compares rankings to abs()-GCM head scores
to determine whether the GCM null failure (ratio=0.92x) is a method artifact
or genuine signal absence.

Usage:
    python3 scripts/15_mean_diff_heads.py
    python3 scripts/15_mean_diff_heads.py --run_dir data/Qwen3-32B/runs/2026-03-22_suffix_marker
"""

import argparse
import json
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", default="data/Qwen3-32B/runs/2026-03-22_suffix_marker")
    p.add_argument("--gcm_real", default="data/Qwen3-32B_top10_stego_heads_suffix_marker.json")
    p.add_argument("--gcm_perm", default="data/Qwen3-32B_top10_stego_heads_suffix_marker_permuted.json")
    p.add_argument("--output", default="data/Qwen3-32B_mean_diff_by_layer_suffix_marker.json")
    return p.parse_args()


def main():
    args = parse_args()
    acts_dir = REPO_ROOT / args.run_dir / "activations"
    output_path = REPO_ROOT / args.output

    labels = np.load(acts_dir / "labels.npy")
    stego_mask = labels == 1
    clean_mask = labels == 0
    print(f"Labels: {stego_mask.sum()} stego / {clean_mask.sum()} clean, total {len(labels)}")

    layer_files = sorted(acts_dir.glob("layer_*.npy"))
    print(f"Found {len(layer_files)} layers")

    mean_diff_by_layer = {}
    for layer_file in layer_files:
        layer_idx = int(layer_file.stem.split("_")[1])
        acts = np.load(layer_file).astype(np.float32)  # (n, hidden_dim)

        stego_mean = acts[stego_mask].mean(axis=0)
        clean_mean = acts[clean_mask].mean(axis=0)
        signed_diff = stego_mean - clean_mean

        l2_norm = float(np.linalg.norm(signed_diff))
        cosine = float(np.dot(stego_mean, clean_mean) / (np.linalg.norm(stego_mean) * np.linalg.norm(clean_mean) + 1e-8))

        mean_diff_by_layer[layer_idx] = {
            "l2_norm_signed_diff": l2_norm,
            "cosine_stego_clean": cosine,
        }
        print(f"  Layer {layer_idx:2d}: ||signed_diff||={l2_norm:.4f}  cos(stego,clean)={cosine:.4f}")

    # Load GCM scores — aggregate per layer
    with open(REPO_ROOT / args.gcm_real) as f:
        gcm_real = json.load(f)
    with open(REPO_ROOT / args.gcm_perm) as f:
        gcm_perm = json.load(f)

    gcm_by_layer_real = {}
    for h in gcm_real:
        l = h["layer"]
        gcm_by_layer_real[l] = gcm_by_layer_real.get(l, 0) + h["score"]

    gcm_by_layer_perm = {}
    for h in gcm_perm:
        l = h["layer"]
        gcm_by_layer_perm[l] = gcm_by_layer_perm.get(l, 0) + h["score"]

    # Rank correlation between mean_diff and GCM
    all_layers = sorted(mean_diff_by_layer.keys())
    md_vals = [mean_diff_by_layer[l]["l2_norm_signed_diff"] for l in all_layers]
    gcm_vals = [gcm_by_layer_real.get(l, 0) for l in all_layers]

    # Spearman rank correlation
    def rank(arr):
        order = np.argsort(arr)[::-1]
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(arr))
        return ranks.astype(float)

    rank_corr = float(np.corrcoef(rank(md_vals), rank(gcm_vals))[0, 1])

    # Top-5 by each method
    top5_md = sorted(all_layers, key=lambda l: mean_diff_by_layer[l]["l2_norm_signed_diff"], reverse=True)[:5]
    top5_gcm = sorted(all_layers, key=lambda l: gcm_by_layer_real.get(l, 0), reverse=True)[:5]
    top5_perm = sorted(all_layers, key=lambda l: gcm_by_layer_perm.get(l, 0), reverse=True)[:5]

    print(f"\nTop 5 layers by signed mean-diff: {top5_md}")
    print(f"Top 5 layers by GCM (real):        {top5_gcm}")
    print(f"Top 5 layers by GCM (permuted):    {top5_perm}")
    print(f"Spearman rank-corr (mean-diff vs GCM-real): {rank_corr:.3f}")

    overlap_md_gcm = len(set(top5_md) & set(top5_gcm))
    print(f"Top-5 overlap (mean-diff vs GCM-real): {overlap_md_gcm}/5")

    results = {
        "run_dir": str(args.run_dir),
        "n_stego": int(stego_mask.sum()),
        "n_clean": int(clean_mask.sum()),
        "mean_diff_by_layer": {str(k): v for k, v in mean_diff_by_layer.items()},
        "gcm_by_layer_real": {str(k): v for k, v in gcm_by_layer_real.items()},
        "gcm_by_layer_perm": {str(k): v for k, v in gcm_by_layer_perm.items()},
        "top5_layers_mean_diff": top5_md,
        "top5_layers_gcm_real": top5_gcm,
        "top5_layers_gcm_perm": top5_perm,
        "spearman_rank_corr_mean_diff_vs_gcm": rank_corr,
        "top5_overlap_mean_diff_vs_gcm": overlap_md_gcm,
        "interpretation": (
            "rank_corr > 0.7: GCM found real signal but abs() inflated null scores. "
            "rank_corr < 0.3: GCM found noise unrelated to actual stego signal."
        ),
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
