"""
P1 — Layerwise residual-stream knockout ablation (Llama-3-8B synonym).

For each layer L (0-31), skips that layer's computation by replacing its output
with its input (the residual unchanged by attention+MLP). Measures how the L3
probe score changes — which layer's computation is most responsible for the
stego representation detectable at layer 3?

Method: knockout (not patching) — no clean-pair data needed, works on all
200 synonym stego examples from combined_dataset.jsonl.

Expected result:
  - Large drop at L=X → layer X writes/transforms the stego representation
  - Drop roughly equal at L<3 and zero at L>3 (since L3 is the probe target)
  - "Localized" = 1-3 layers account for >80% of drop

Requires: P0 completed — saved probes at
    <run_dir>/probe_results_stego_vs_clean/trained_probes/layer_03/

Usage:
    python3 scripts/13_layerwise_ablation.py --run_dir <path> --model meta-llama/Meta-Llama-3-8B-Instruct
"""

import argparse
import json
import os
import sys
import datetime
import subprocess
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Path to P0 run directory")
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--dataset", default="data/combined_dataset.jsonl")
    p.add_argument("--probe_layer", type=int, default=3, help="Layer whose probe we measure")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--cache_dir", default="/home/ubuntu/.cache/huggingface/hub")
    return p.parse_args()


def load_probe(run_dir, layer_idx, n_seeds=5):
    """Load and average probe coef_ across seeds. Returns (coef, intercept, scaler)."""
    import joblib
    trained_dir = Path(run_dir) / "probe_results_stego_vs_clean" / "trained_probes" / f"layer_{layer_idx:02d}"
    if not trained_dir.exists():
        # try without suffix
        trained_dir = Path(run_dir) / "probe_results" / "trained_probes" / f"layer_{layer_idx:02d}"

    coefs, intercepts, scalers = [], [], []
    for seed in range(n_seeds):
        path = trained_dir / f"seed_{seed}.joblib"
        if path.exists():
            obj = joblib.load(path)
            probe = obj["probe"]
            scaler = obj["scaler"]
            coefs.append(probe.coef_[0])
            intercepts.append(probe.intercept_[0])
            scalers.append(scaler)

    if not coefs:
        raise FileNotFoundError(f"No probe files found in {trained_dir}")

    print(f"Loaded {len(coefs)} probe seeds from {trained_dir}")
    return coefs, intercepts, scalers


def probe_score(activation, coefs, intercepts, scalers):
    """Average P(stego) across seeds."""
    from sklearn.linear_model import LogisticRegression
    scores = []
    act = activation.reshape(1, -1)
    for coef, intercept, scaler in zip(coefs, intercepts, scalers):
        act_scaled = scaler.transform(act)
        logit = act_scaled @ coef + intercept
        p = float(1 / (1 + np.exp(-logit)))
        scores.append(p)
    return np.mean(scores)


def load_synonym_examples(dataset_path, max_length, tokenizer):
    """Load synonym stego examples, return formatted texts."""
    examples = []
    with open(dataset_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("method") == "synonym" and d.get("is_stego"):
                examples.append(d)
    print(f"Loaded {len(examples)} synonym stego examples")
    texts = [ex["response"] for ex in examples]
    return texts, examples


def run_ablation(model, tokenizer, texts, layer_idx_to_ablate, probe_layer,
                 coefs, intercepts, scalers, batch_size, max_length):
    """Run forward passes with one layer knocked out. Returns mean probe score."""
    scores = []
    captured = {}
    hooks = []

    def make_knockout_hook(l):
        def hook_fn(module, inputs, output):
            # Replace layer output with layer input (skip this layer's computation)
            inp = inputs[0]
            if isinstance(output, tuple):
                return (inp,) + output[1:]
            return inp
        return hook_fn

    def capture_hook(module, inputs, output):
        act = output[0] if isinstance(output, tuple) else output
        captured["act"] = act.detach()

    # Register knockout hook
    h_ko = model.model.layers[layer_idx_to_ablate].register_forward_hook(
        make_knockout_hook(layer_idx_to_ablate)
    )
    hooks.append(h_ko)

    # Register capture hook at probe_layer
    h_cap = model.model.layers[probe_layer].register_forward_hook(capture_hook)
    hooks.append(h_cap)

    try:
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            inputs = tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length
            ).to(model.device)

            with torch.no_grad():
                model(**inputs)

            # Get last non-padding token for each example
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1
            batch_acts = captured["act"]  # (batch, seq, hidden)

            for i, seq_len in enumerate(seq_lens):
                act = batch_acts[i, seq_len].cpu().float().numpy()
                s = probe_score(act, coefs, intercepts, scalers)
                scores.append(s)
    finally:
        for h in hooks:
            h.remove()

    return float(np.mean(scores))


def main():
    args = parse_args()
    run_dir = REPO_ROOT / args.run_dir

    config = {
        "args": vars(args),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    print(json.dumps(config, indent=2))

    # Load probe weights
    coefs, intercepts, scalers = load_probe(run_dir, args.probe_layer)

    # Load model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"\nLoading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
        cache_dir=args.cache_dir
    )
    model.eval()
    n_layers = model.config.num_hidden_layers
    print(f"Model: {n_layers} layers, hidden={model.config.hidden_size}")

    # Load synonym examples
    texts, examples = load_synonym_examples(
        REPO_ROOT / args.dataset, args.max_length, tokenizer
    )

    # Baseline: no ablation
    print("\n--- Baseline (no ablation) ---")
    baseline_score = run_ablation(
        model, tokenizer, texts, layer_idx_to_ablate=-1, probe_layer=args.probe_layer,
        coefs=coefs, intercepts=intercepts, scalers=scalers,
        batch_size=args.batch_size, max_length=args.max_length
    )

    # Override to run without knockout: re-implement baseline without hooks
    captured = {}

    def capture_only(module, inputs, output):
        act = output[0] if isinstance(output, tuple) else output
        captured["act"] = act.detach()

    h = model.model.layers[args.probe_layer].register_forward_hook(capture_only)
    baseline_scores = []
    for batch_start in range(0, len(texts), args.batch_size):
        batch = texts[batch_start:batch_start + args.batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=args.max_length).to(model.device)
        with torch.no_grad():
            model(**inputs)
        seq_lens = inputs["attention_mask"].sum(dim=1) - 1
        for i, seq_len in enumerate(seq_lens):
            act = captured["act"][i, seq_len].cpu().float().numpy()
            baseline_scores.append(probe_score(act, coefs, intercepts, scalers))
    h.remove()
    baseline_mean = float(np.mean(baseline_scores))
    print(f"Baseline probe score (mean P(stego)): {baseline_mean:.4f}")

    # Knockout ablation per layer
    ablation_scores = {}
    print(f"\n--- Layer-by-layer knockout (0..{n_layers-1}) ---")
    for l in tqdm(range(n_layers), desc="Ablating layers"):
        score = run_ablation(
            model, tokenizer, texts, layer_idx_to_ablate=l, probe_layer=args.probe_layer,
            coefs=coefs, intercepts=intercepts, scalers=scalers,
            batch_size=args.batch_size, max_length=args.max_length
        )
        drop = baseline_mean - score
        ablation_scores[l] = {"ablated_score": score, "drop": drop}
        print(f"  L{l:2d}: score={score:.4f}  drop={drop:+.4f}")

    # Find top contributing layers
    sorted_layers = sorted(ablation_scores.keys(), key=lambda l: ablation_scores[l]["drop"], reverse=True)
    total_drop = sum(max(0, v["drop"]) for v in ablation_scores.values())
    cumulative = 0.0
    localization_layers = []
    for l in sorted_layers:
        d = max(0, ablation_scores[l]["drop"])
        cumulative += d
        localization_layers.append(l)
        if total_drop > 0 and cumulative / total_drop >= 0.8:
            break

    print(f"\n=== LOCALIZATION RESULT ===")
    print(f"Baseline probe score: {baseline_mean:.4f}")
    print(f"Top 3 layers by drop: {sorted_layers[:3]}")
    print(f"Layers accounting for >80% of total drop: {localization_layers}")
    localized = len(localization_layers) <= 3
    print(f"LOCALIZED: {localized} ({'proceed to P3 steering' if localized else 'diffuse — skip P3 or note as distributed'})")

    # Save results
    output_dir = REPO_ROOT / args.run_dir / "probe_results_stego_vs_clean"
    output_path = output_dir / "p1_ablation_results.json"
    results = {
        "config": config,
        "probe_layer": args.probe_layer,
        "n_synonym_examples": len(texts),
        "baseline_probe_score": baseline_mean,
        "ablation_scores_by_layer": {str(l): v for l, v in ablation_scores.items()},
        "sorted_layers_by_drop": sorted_layers,
        "localization_layers_80pct": localization_layers,
        "localized": localized,
        "total_positive_drop": float(total_drop),
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
