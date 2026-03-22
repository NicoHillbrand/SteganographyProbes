"""
P3 — Probe-direction steering on Llama-3-8B synonym examples.

Uses the L3 probe coef_ as a steering direction. Adds alpha * direction to the
residual stream at the localized layer (default: layer 3) and measures:
  1. Probe score shift at downstream layers L4..L31 (continuous, low variance)
  2. KL divergence on clean examples (utility check)

Clean steer: inject stego direction into CLEAN examples → do downstream layers
pick up the stego representation? If yes: causal confirmation. If no: direction
is correlational only at L3.

Requires: P0 completed with --save_probes, P1 results (for localization layer).

Usage:
    python3 scripts/16_probe_steer.py --run_dir <path> --model meta-llama/Meta-Llama-3-8B-Instruct
    python3 scripts/16_probe_steer.py --run_dir <path> --steer_layer 2 --alpha 5.0
"""

import argparse
import json
import sys
import datetime
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).parent.parent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--dataset", default="data/combined_dataset.jsonl")
    p.add_argument("--probe_layer", type=int, default=3, help="Layer of the probe used for steering direction")
    p.add_argument("--steer_layer", type=int, default=3, help="Layer to inject steering at (from P1 localization)")
    p.add_argument("--alpha", type=float, default=5.0, help="Steering strength")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--n_val_clean", type=int, default=100, help="Clean examples for KL divergence check")
    p.add_argument("--cache_dir", default="/home/ubuntu/.cache/huggingface/hub")
    return p.parse_args()


def load_probe_direction(run_dir, layer_idx, n_seeds=5):
    """Load and average normalized coef_ across seeds."""
    import joblib
    probe_dir = Path(run_dir) / "probe_results_stego_vs_clean" / "trained_probes" / f"layer_{layer_idx:02d}"
    if not probe_dir.exists():
        probe_dir = Path(run_dir) / "probe_results" / "trained_probes" / f"layer_{layer_idx:02d}"

    coefs, scalers = [], []
    for seed in range(n_seeds):
        path = probe_dir / f"seed_{seed}.joblib"
        if path.exists():
            obj = joblib.load(path)
            coefs.append(obj["probe"].coef_[0])
            scalers.append(obj["scaler"])

    if not coefs:
        raise FileNotFoundError(f"No probe files in {probe_dir}")

    direction = np.mean(coefs, axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    return direction, scalers


def load_probes_all_layers(run_dir, n_layers=32, n_seeds=5):
    """Load probe coef_ and scalers for all layers."""
    import joblib
    probes = {}
    for layer_idx in range(n_layers):
        probe_dir = Path(run_dir) / "probe_results_stego_vs_clean" / "trained_probes" / f"layer_{layer_idx:02d}"
        if not probe_dir.exists():
            probe_dir = Path(run_dir) / "probe_results" / "trained_probes" / f"layer_{layer_idx:02d}"
        coefs, intercepts, scalers = [], [], []
        for seed in range(n_seeds):
            path = probe_dir / f"seed_{seed}.joblib"
            if path.exists():
                obj = joblib.load(path)
                coefs.append(obj["probe"].coef_[0])
                intercepts.append(obj["probe"].intercept_[0])
                scalers.append(obj["scaler"])
        if coefs:
            probes[layer_idx] = {"coefs": coefs, "intercepts": intercepts, "scalers": scalers}
    return probes


def probe_score(activation, coefs, intercepts, scalers):
    act = activation.reshape(1, -1)
    scores = []
    for coef, intercept, scaler in zip(coefs, intercepts, scalers):
        act_s = scaler.transform(act)
        logit = float(act_s @ coef + intercept)
        scores.append(1 / (1 + np.exp(-logit)))
    return float(np.mean(scores))


def get_scores_by_layer(model, tokenizer, texts, probes, steer_direction,
                        steer_layer, alpha, batch_size, max_length):
    """Run texts with optional steering, return probe scores per layer."""
    all_captures = {l: [] for l in probes}
    hooks = []

    # Steering hook
    if steer_direction is not None:
        steer_t = torch.tensor(alpha * steer_direction, dtype=torch.bfloat16)

        def make_steer_hook(t):
            def hook_fn(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                steer = t.to(h.device).unsqueeze(0).unsqueeze(0)
                h_steered = h + steer
                if isinstance(output, tuple):
                    return (h_steered,) + output[1:]
                return h_steered
            return hook_fn

        h = model.model.layers[steer_layer].register_forward_hook(make_steer_hook(steer_t))
        hooks.append(h)

    # Capture hooks for all probe layers
    def make_capture_hook(l, storage):
        def hook_fn(module, inputs, output):
            act = output[0] if isinstance(output, tuple) else output
            storage[l].append(act.detach())
        return hook_fn

    for l in probes:
        h = model.model.layers[l].register_forward_hook(make_capture_hook(l, all_captures))
        hooks.append(h)

    try:
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=max_length).to(model.device)
            with torch.no_grad():
                model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    # Compute per-layer probe scores
    # We need seq_lens — re-run a dummy forward to get attention mask (already have from inputs)
    # Use the last batch's seq_lens; we need to accumulate per-example
    # Simpler: rerun with seq_len tracking
    # Since we captured activations from ALL batches, process them
    # all_captures[l] is a list of tensors, one per batch
    # But we don't have attention_mask anymore... re-implement with tracking

    return None  # Placeholder, see below


def get_probe_scores_tracking(model, tokenizer, texts, probes, steer_direction,
                              steer_layer, alpha, batch_size, max_length):
    """Run texts, capture per-example last-token activations at each probe layer."""
    layer_scores = {l: [] for l in probes}
    hooks = []

    if steer_direction is not None:
        steer_t = torch.tensor(alpha * steer_direction, dtype=torch.bfloat16)

        def make_steer_hook(t):
            def hook_fn(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                steer = t.to(h.device).unsqueeze(0).unsqueeze(0)
                h_steered = h + steer
                return (h_steered,) + output[1:] if isinstance(output, tuple) else h_steered
            return hook_fn

        h = model.model.layers[steer_layer].register_forward_hook(make_steer_hook(steer_t))
        hooks.append(h)

    # Per-batch processing: capture inside batch loop
    try:
        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start:batch_start + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True,
                             truncation=True, max_length=max_length).to(model.device)
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1

            batch_captures = {l: None for l in probes}
            batch_hooks = []

            def make_batch_hook(l, storage):
                def hook_fn(module, inp, out):
                    storage[l] = (out[0] if isinstance(out, tuple) else out).detach()
                return hook_fn

            for l in probes:
                h = model.model.layers[l].register_forward_hook(make_batch_hook(l, batch_captures))
                batch_hooks.append(h)

            with torch.no_grad():
                model(**inputs)

            for h in batch_hooks:
                h.remove()

            for l in probes:
                if batch_captures[l] is not None:
                    for i, seq_len in enumerate(seq_lens):
                        act = batch_captures[l][i, seq_len].cpu().float().numpy()
                        s = probe_score(act, probes[l]["coefs"], probes[l]["intercepts"], probes[l]["scalers"])
                        layer_scores[l].append(s)
    finally:
        for h in hooks:
            h.remove()

    return {l: float(np.mean(v)) for l, v in layer_scores.items() if v}


def kl_divergence(model, tokenizer, texts, steer_direction, steer_layer, alpha, batch_size, max_length):
    """KL(original || steered) on first n_val texts."""
    kls = []
    steer_t = torch.tensor(alpha * steer_direction, dtype=torch.bfloat16)

    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start:batch_start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_length).to(model.device)

        with torch.no_grad():
            out_orig = model(**inputs)
            logits_orig = out_orig.logits[:, -1, :]

        def steer_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            s = steer_t.to(h.device).unsqueeze(0).unsqueeze(0)
            h_s = h + s
            return (h_s,) + out[1:] if isinstance(out, tuple) else h_s

        h = model.model.layers[steer_layer].register_forward_hook(steer_hook)
        with torch.no_grad():
            out_steered = model(**inputs)
            logits_steered = out_steered.logits[:, -1, :]
        h.remove()

        p_orig = F.softmax(logits_orig.float(), dim=-1)
        p_steer = F.softmax(logits_steered.float(), dim=-1)
        kl = F.kl_div(p_steer.log(), p_orig, reduction="batchmean").item()
        kls.append(kl)

    return float(np.mean(kls))


def main():
    args = parse_args()
    run_dir = REPO_ROOT / args.run_dir

    config = {
        "args": vars(args),
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    print(json.dumps(config, indent=2))

    # Load steering direction from probe_layer
    print(f"\nLoading probe direction from layer {args.probe_layer}")
    direction, _ = load_probe_direction(run_dir, args.probe_layer)
    print(f"Direction norm: {np.linalg.norm(direction):.4f} (should be 1.0)")

    # Load probes for all layers (for measuring downstream shift)
    probes = load_probes_all_layers(run_dir)
    print(f"Loaded probes for {len(probes)} layers")

    # Load model
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"\nLoading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", cache_dir=args.cache_dir
    )
    model.eval()

    # Load examples
    all_entries = []
    with open(REPO_ROOT / args.dataset) as f:
        for line in f:
            all_entries.append(json.loads(line))

    clean_texts = [e["response"] for e in all_entries if not e.get("is_stego")][:args.n_val_clean]
    synonym_texts = [e["response"] for e in all_entries if e.get("method") == "synonym" and e.get("is_stego")]
    print(f"Clean examples: {len(clean_texts)}, Synonym stego: {len(synonym_texts)}")

    # Baseline scores (clean, no steering)
    print("\n--- Baseline clean probe scores (no steering) ---")
    baseline_clean = get_probe_scores_tracking(
        model, tokenizer, clean_texts, probes, None, args.steer_layer, 0,
        args.batch_size, args.max_length
    )

    # Steered clean scores (inject stego direction into clean examples)
    print(f"--- Steered clean probe scores (alpha={args.alpha} at L{args.steer_layer}) ---")
    steered_clean = get_probe_scores_tracking(
        model, tokenizer, clean_texts, probes, direction, args.steer_layer, args.alpha,
        args.batch_size, args.max_length
    )

    # KL divergence
    print("--- KL divergence (utility check) ---")
    kl = kl_divergence(model, tokenizer, clean_texts[:50], direction, args.steer_layer,
                       args.alpha, args.batch_size, args.max_length)
    print(f"KL(orig || steered): {kl:.5f}")

    # Print results
    print(f"\n=== STEERING RESULTS (alpha={args.alpha}, steer_layer=L{args.steer_layer}) ===")
    print(f"{'Layer':>6}  {'Baseline':>10}  {'Steered':>10}  {'Delta':>10}")
    layers_sorted = sorted(probes.keys())
    for l in layers_sorted:
        b = baseline_clean.get(l, float("nan"))
        s = steered_clean.get(l, float("nan"))
        d = s - b
        marker = " *** CAUSAL" if l >= args.steer_layer and abs(d) > 0.05 else ""
        print(f"  L{l:02d}:  {b:.4f}    {s:.4f}    {d:+.4f}{marker}")

    # Downstream shift (L > steer_layer)
    downstream_layers = [l for l in layers_sorted if l > args.steer_layer]
    downstream_deltas = [steered_clean.get(l, 0) - baseline_clean.get(l, 0) for l in downstream_layers]
    mean_downstream_delta = float(np.mean(downstream_deltas)) if downstream_deltas else 0.0
    print(f"\nMean probe score shift at downstream layers (L{args.steer_layer+1}+): {mean_downstream_delta:+.4f}")
    print(f"KL divergence (utility cost): {kl:.5f}")

    if abs(mean_downstream_delta) > 0.05:
        print("→ CAUSAL CONFIRMATION: steering direction propagates to downstream layers")
    else:
        print("→ NULL: direction is correlational at L3, does not causally drive downstream")

    # Save
    output_path = REPO_ROOT / args.run_dir / "probe_results_stego_vs_clean" / "p3_probe_steer_results.json"
    results = {
        "config": config,
        "steer_layer": args.steer_layer,
        "probe_layer": args.probe_layer,
        "alpha": args.alpha,
        "n_clean": len(clean_texts),
        "n_synonym": len(synonym_texts),
        "kl_divergence": kl,
        "baseline_clean_by_layer": {str(l): v for l, v in baseline_clean.items()},
        "steered_clean_by_layer": {str(l): v for l, v in steered_clean.items()},
        "delta_by_layer": {str(l): steered_clean.get(l, 0) - baseline_clean.get(l, 0) for l in layers_sorted},
        "mean_downstream_delta": mean_downstream_delta,
        "causal_confirmed": abs(mean_downstream_delta) > 0.05,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
