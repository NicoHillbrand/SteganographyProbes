"""
Step 8: Plot headline figures from probe and baseline results.

Reads JSON results from steps 6 and 7 and produces up to 5 figures:
  1. Layerwise AUROC for stego presence detection
  2. Layerwise AUROC for decode task (recognition)
  3. Method-sliced best-layer AUROC (all vs hard subsets)
  4. Per-method AUROC: activation probe vs char baseline
  5. Cross-method transfer heatmap (optional)

Usage:
    python scripts/08_plot_results.py
    python scripts/08_plot_results.py --results_dir data/probe_results --output_dir data/figures --format pdf
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


SUBSET_LABELS = {
    "all_methods": "All methods",
    "hard_plus_acrostic": "Synonym + acrostic",
    "synonym_only": "Synonym only",
    "easy_methods": "Easy methods",
}


parser = argparse.ArgumentParser(description="Plot headline figures from probe results")
parser.add_argument(
    "--run_dir",
    type=str,
    default=None,
    help="If provided, read results from <run_dir>/probe_results/ and save figures to <run_dir>/figures/",
)
parser.add_argument(
    "--results_dir",
    type=str,
    default="data/probe_results",
    help="Directory containing probe/baseline result JSONs (legacy, use --run_dir instead)",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="data/figures",
    help="Directory to save figures (legacy, use --run_dir instead)",
)
parser.add_argument(
    "--format",
    type=str,
    default="png",
    choices=["png", "pdf"],
    help="Output figure format",
)
args = parser.parse_args()


plt.rcParams.update(
    {
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
    }
)


def load_json(path):
    """Load a JSON file, returning None if it doesn't exist."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def extract_layerwise_metric(results, metric="mean_auc_roc"):
    """Extract sorted layer indices, means, and stds for one metric."""
    results_by_layer = results["results_by_layer"]
    layers = sorted(int(key) for key in results_by_layer.keys())
    means = [results_by_layer[str(layer)][metric] for layer in layers]

    std_metric = metric.replace("mean_", "std_")
    stds = []
    for layer in layers:
        layer_result = results_by_layer[str(layer)]
        if std_metric in layer_result:
            stds.append(layer_result[std_metric])
        else:
            per_seed = layer_result.get("per_seed", [])
            stds.append(float(np.std([seed_result["auc_roc"] for seed_result in per_seed])) if per_seed else 0.0)

    return np.array(layers), np.array(means), np.array(stds)


def plot_layerwise_metric(probe_results, baseline_results, title, output_path):
    """Plot layerwise AUROC with optional baseline horizontal lines."""
    layers, mean_aurocs, std_aurocs = extract_layerwise_metric(probe_results)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.errorbar(
        layers,
        mean_aurocs,
        yerr=std_aurocs,
        fmt="o-",
        capsize=3,
        linewidth=2,
        markersize=5,
        label="Activation probe",
        color="#1f77b4",
    )

    if baseline_results:
        baselines = baseline_results.get("baselines", {})
        colors = {"char_features": "#d62728", "tfidf_char_ngram": "#ff7f0e"}
        labels = {
            "char_features": "Char features baseline",
            "tfidf_char_ngram": "TF-IDF char n-gram baseline",
        }
        for key, baseline in baselines.items():
            if isinstance(baseline, dict) and "mean_auc_roc" in baseline:
                ax.axhline(
                    y=baseline["mean_auc_roc"],
                    linestyle="--",
                    linewidth=1.5,
                    color=colors.get(key, "gray"),
                    label=labels.get(key, key),
                )

    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title(title)
    ax.set_ylim(0.4, 1.02)
    ax.legend(loc="lower right")

    decode_parse_mode = probe_results.get("decode_rescore_mode")
    if decode_parse_mode and decode_parse_mode != "saved_labels":
        ax.text(
            0.01,
            0.02,
            f"decode parse mode: {decode_parse_mode}",
            transform=ax.transAxes,
            fontsize=9,
            color="#444444",
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_subset_comparison(stego_results, baseline_results, output_path):
    """Compare best-layer AUROC across pooled method subsets."""
    probe_subsets = stego_results.get("subset_results") or {}
    baseline_subsets = {}
    if baseline_results:
        baseline_subsets = baseline_results.get("baselines", {}).get("subset_char", {})

    ordered_subsets = [
        key
        for key in ("all_methods", "hard_plus_acrostic", "synonym_only", "easy_methods")
        if key in probe_subsets or key in baseline_subsets
    ]
    if not ordered_subsets:
        print("  Skipping Figure 3: no subset results available")
        return

    probe_values = [
        probe_subsets.get(key, {}).get("mean_auc_roc", np.nan) for key in ordered_subsets
    ]
    baseline_values = [
        baseline_subsets.get(key, {}).get("mean_auc_roc", np.nan) for key in ordered_subsets
    ]

    x = np.arange(len(ordered_subsets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, probe_values, width, label="Activation probe", color="#1f77b4")
    ax.bar(x + width / 2, baseline_values, width, label="Char baseline", color="#d62728")

    ax.set_xlabel("Method subset")
    ax.set_ylabel("Best-layer AUROC")
    ax.set_title("Best-Layer AUROC By Method Subset")
    ax.set_xticks(x)
    ax.set_xticklabels([SUBSET_LABELS.get(key, key) for key in ordered_subsets], rotation=20, ha="right")
    ax.set_ylim(0.4, 1.05)
    ax.legend()

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

    all_methods = sorted(set(list(probe_methods.keys()) + list(baseline_methods.keys())))
    if not all_methods:
        print("  Skipping Figure 4: no per-method data available")
        return

    probe_aurocs = [probe_methods.get(method, {}).get("mean_auc_roc", 0) for method in all_methods]
    baseline_aurocs = [
        baseline_methods.get(method, {}).get("mean_auc_roc", 0) for method in all_methods
    ]

    x = np.arange(len(all_methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(all_methods) * 1.5), 5))
    ax.bar(x - width / 2, probe_aurocs, width, label="Activation probe", color="#1f77b4")
    ax.bar(x + width / 2, baseline_aurocs, width, label="Char features baseline", color="#d62728")

    ax.set_xlabel("Steganography method")
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


def plot_transfer_heatmap(stego_results, output_path):
    """Plot a cross-method transfer heatmap if transfer results are available."""
    transfer = stego_results.get("transfer_results")
    if not transfer:
        print("  Skipping Figure 5: no transfer matrix available")
        return

    methods = transfer.get("methods", [])
    matrix = np.array(
        [
            [np.nan if value is None else value for value in row]
            for row in transfer.get("mean_auc_roc_matrix", [])
        ],
        dtype=float,
    )
    if matrix.size == 0:
        print("  Skipping Figure 5: empty transfer matrix")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, vmin=0.4, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_yticklabels(methods)
    ax.set_xlabel("Test method")
    ax.set_ylabel("Train method")
    ax.set_title("Cross-Method Transfer AUROC")

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            label = "NA" if np.isnan(value) else f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="#111111", fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("AUROC")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    # Resolve paths from --run_dir or legacy args
    if args.run_dir:
        results_dir = os.path.join(args.run_dir, "probe_results")
        output_dir = os.path.join(args.run_dir, "figures")
    else:
        results_dir = args.results_dir
        output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)
    ext = args.format
    generated = []

    # In --run_dir mode, probe results are saved as probe_results.json;
    # in legacy mode, they have task-specific names
    stego_results = (
        load_json(os.path.join(results_dir, "probe_results.json"))
        or load_json(os.path.join(results_dir, "stego_detection_probe_results.json"))
    )
    decode_results = load_json(os.path.join(results_dir, "decode_task_probe_results.json"))
    # Text baselines may be in a sibling directory in --run_dir mode
    baseline_dir = os.path.join(os.path.dirname(results_dir), "text_baselines") if args.run_dir else results_dir
    baseline_results = (
        load_json(os.path.join(baseline_dir, "text_baseline_results.json"))
        or load_json(os.path.join(results_dir, "text_baseline_results.json"))
    )

    print("Figure 1: Layerwise AUROC – Stego Presence Detection")
    if stego_results and stego_results.get("results_by_layer"):
        out = os.path.join(output_dir, f"fig1_layerwise_stego_detection.{ext}")
        plot_layerwise_metric(
            stego_results,
            baseline_results,
            "Layerwise AUROC: Stego Presence Detection",
            out,
        )
        generated.append("Figure 1")
    else:
        print("  Skipping: stego detection probe results not found")

    print("Figure 2: Layerwise AUROC – Stego Recognition (Decode Task)")
    if decode_results and decode_results.get("results_by_layer"):
        out = os.path.join(output_dir, f"fig2_layerwise_decode_task.{ext}")
        plot_layerwise_metric(
            decode_results,
            None,
            "Layerwise AUROC: Stego Recognition (Decode Task)",
            out,
        )
        generated.append("Figure 2")
    else:
        print("  Skipping: decode task probe results not found")

    print("Figure 3: Best-Layer AUROC – Method Subsets")
    if stego_results:
        out = os.path.join(output_dir, f"fig3_method_subset_comparison.{ext}")
        plot_subset_comparison(stego_results, baseline_results, out)
        if os.path.exists(out):
            generated.append("Figure 3")
    else:
        print("  Skipping: stego detection probe results not found")

    print("Figure 4: Per-Method AUROC – Activation Probe vs Text Baseline")
    if stego_results:
        out = os.path.join(output_dir, f"fig4_per_method_comparison.{ext}")
        plot_per_method_comparison(stego_results, baseline_results, out)
        if os.path.exists(out):
            generated.append("Figure 4")
    else:
        print("  Skipping: stego detection probe results not found")

    print("Figure 5: Cross-Method Transfer AUROC")
    if stego_results:
        out = os.path.join(output_dir, f"fig5_cross_method_transfer.{ext}")
        plot_transfer_heatmap(stego_results, out)
        if os.path.exists(out):
            generated.append("Figure 5")
    else:
        print("  Skipping: stego detection probe results not found")

    print()
    print(f"{'=' * 60}")
    if generated:
        print(f"Generated {len(generated)} figure(s): {', '.join(generated)}")
        print(f"Output directory: {output_dir}")
    else:
        print("No figures generated — result files not found.")
        print(f"Expected in: {results_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
