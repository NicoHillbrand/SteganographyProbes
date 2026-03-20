"""
Step 8: Plot headline figures from probe and baseline results.

Reads JSON results from steps 6 and 7 and produces 4 figures:
  1. Layerwise AUROC for stego presence detection
  2. Layerwise AUROC for decode task (recognition)
  3. Same-carrier control: decode vs benign prompt
  4. Per-method breakdown: activation probe vs text baseline

Usage:
    python scripts/08_plot_results.py
    python scripts/08_plot_results.py --results_dir data/probe_results --output_dir data/figures --format pdf
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="Plot headline figures from probe results")
parser.add_argument("--results_dir", type=str, default="data/probe_results",
                    help="Directory containing probe/baseline result JSONs")
parser.add_argument("--output_dir", type=str, default="data/figures",
                    help="Directory to save figures")
parser.add_argument("--format", type=str, default="png", choices=["png", "pdf"],
                    help="Output figure format")
args = parser.parse_args()


# --- Style setup ---
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def load_json(path):
    """Load a JSON file, returning None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_layerwise(results):
    """Extract sorted layer indices, mean AUROCs, and std AUROCs from results_by_layer."""
    rbl = results["results_by_layer"]
    layers = sorted(int(k) for k in rbl.keys())
    mean_aurocs = [rbl[str(layer_idx)]["mean_auc_roc"] for layer_idx in layers]
    # Compute std of AUC across seeds if available
    std_aurocs = []
    for layer_idx in layers:
        per_seed = rbl[str(layer_idx)].get("per_seed", [])
        if per_seed:
            std_aurocs.append(float(np.std([s["auc_roc"] for s in per_seed])))
        else:
            std_aurocs.append(0.0)
    return np.array(layers), np.array(mean_aurocs), np.array(std_aurocs)


def plot_layerwise_auroc(probe_results, baseline_results, task_label, title, output_path):
    """Plot layerwise AUROC with optional baseline horizontal lines."""
    layers, mean_aurocs, std_aurocs = extract_layerwise(probe_results)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(layers, mean_aurocs, yerr=std_aurocs, fmt="o-", capsize=3,
                linewidth=2, markersize=5, label="Activation probe", color="#1f77b4")

    # Baseline horizontal lines
    if baseline_results:
        baselines = baseline_results.get("baselines", {})
        colors = {"char_features": "#d62728", "tfidf_char_ngram": "#ff7f0e"}
        labels = {"char_features": "Char features baseline", "tfidf_char_ngram": "TF-IDF char n-gram baseline"}
        for key, bl in baselines.items():
            if isinstance(bl, dict) and "mean_auc_roc" in bl:
                ax.axhline(y=bl["mean_auc_roc"], linestyle="--", linewidth=1.5,
                           color=colors.get(key, "gray"), label=labels.get(key, key))

    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_same_carrier_control(decode_results, benign_results, output_path):
    """Plot decode vs benign probe AUROC by layer."""
    layers_d, aurocs_d, std_d = extract_layerwise(decode_results)
    layers_b, aurocs_b, std_b = extract_layerwise(benign_results)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(layers_d, aurocs_d, yerr=std_d, fmt="o-", capsize=3,
                linewidth=2, markersize=5, label="Decode task", color="#1f77b4")
    ax.errorbar(layers_b, aurocs_b, yerr=std_b, fmt="s--", capsize=3,
                linewidth=2, markersize=5, label="Benign prompt", color="#2ca02c")
    ax.axhline(y=0.5, linestyle=":", color="gray", alpha=0.5, label="Chance")

    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title("Same-Carrier Control: Decode vs Benign Prompt")
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_per_method_comparison(probe_results, baseline_results, output_path):
    """Grouped bar chart: per-method AUROC for activation probe vs char baseline."""
    probe_methods = probe_results.get("per_method_results") or {}
    baseline_methods = {}
    if baseline_results:
        baseline_methods = baseline_results.get("baselines", {}).get("per_method_char", {})
        if not baseline_methods:
            baseline_methods = baseline_results.get("per_method_char", {})

    all_methods = sorted(set(list(probe_methods.keys()) + list(baseline_methods.keys())))
    if not all_methods:
        print("  Skipping Figure 4: no per-method data available")
        return

    probe_aurocs = [probe_methods.get(m, {}).get("mean_auc_roc", 0) for m in all_methods]
    baseline_aurocs = [baseline_methods.get(m, {}).get("mean_auc_roc", 0) for m in all_methods]

    x = np.arange(len(all_methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(all_methods) * 1.5), 5))
    ax.bar(x - width / 2, probe_aurocs, width, label="Activation probe", color="#1f77b4")
    ax.bar(x + width / 2, baseline_aurocs, width, label="Char features baseline", color="#d62728")

    ax.set_xlabel("Steganography Method")
    ax.set_ylabel("AUROC")
    ax.set_title("Per-Method AUROC: Activation Probe vs Text Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(all_methods, rotation=30, ha="right")
    ax.set_ylim(0.4, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    ext = args.format
    generated = []

    # Load all available result files
    stego_path = os.path.join(args.results_dir, "stego_detection_probe_results.json")
    decode_path = os.path.join(args.results_dir, "decode_task_probe_results.json")
    benign_path = os.path.join(args.results_dir, "benign_task_probe_results.json")
    baseline_path = os.path.join(args.results_dir, "text_baseline_results.json")

    stego_results = load_json(stego_path)
    decode_results = load_json(decode_path)
    benign_results = load_json(benign_path)
    baseline_results = load_json(baseline_path)

    # --- Figure 1: Layerwise AUROC for stego detection ---
    print("Figure 1: Layerwise AUROC – Stego Presence Detection")
    if stego_results and stego_results.get("results_by_layer"):
        out = os.path.join(args.output_dir, f"fig1_layerwise_stego_detection.{ext}")
        plot_layerwise_auroc(
            stego_results, baseline_results,
            "stego_detection",
            "Layerwise AUROC: Stego Presence Detection",
            out,
        )
        generated.append("Figure 1")
    else:
        print(f"  Skipping: {stego_path} not found or empty")

    # --- Figure 2: Layerwise AUROC for decode task ---
    print("Figure 2: Layerwise AUROC – Stego Recognition (Decode Task)")
    if decode_results and decode_results.get("results_by_layer"):
        out = os.path.join(args.output_dir, f"fig2_layerwise_decode_task.{ext}")
        plot_layerwise_auroc(
            decode_results, None,
            "decode_task",
            "Layerwise AUROC: Stego Recognition (Decode Task)",
            out,
        )
        generated.append("Figure 2")
    else:
        print(f"  Skipping: {decode_path} not found or empty")

    # --- Figure 3: Same-carrier control ---
    print("Figure 3: Same-Carrier Control – Decode vs Benign Prompt")
    if (decode_results and decode_results.get("results_by_layer")
            and benign_results and benign_results.get("results_by_layer")):
        out = os.path.join(args.output_dir, f"fig3_same_carrier_control.{ext}")
        plot_same_carrier_control(decode_results, benign_results, out)
        generated.append("Figure 3")
    else:
        print("  Skipping: decode and/or benign probe results not found")

    # --- Figure 4: Per-method breakdown ---
    print("Figure 4: Per-Method AUROC – Activation Probe vs Text Baseline")
    if stego_results:
        out = os.path.join(args.output_dir, f"fig4_per_method_comparison.{ext}")
        plot_per_method_comparison(stego_results, baseline_results, out)
        generated.append("Figure 4")
    else:
        print("  Skipping: stego detection probe results not found")

    # Summary
    print()
    print(f"{'='*60}")
    if generated:
        print(f"Generated {len(generated)} figure(s): {', '.join(generated)}")
        print(f"Output directory: {args.output_dir}")
    else:
        print("No figures generated — result files not found.")
        print(f"Expected in: {args.results_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
