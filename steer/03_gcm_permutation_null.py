"""
GCM Permutation Null Test.

Runs GCM head attribution on a dataset with randomly shuffled stego/clean
labels (50% swap rate, seed=42) to establish a null distribution. If real
GCM scores are substantially higher than permuted scores, the signal is
specific to the stego condition.

Usage (from project root or steer/):
    PYTHONUNBUFFERED=1 python3 steer/03_gcm_permutation_null.py 2>&1 | tee /tmp/gcm_permutation_null.log
    PYTHONUNBUFFERED=1 python3 steer/03_gcm_permutation_null.py --dataset_tag suffix_marker --num_samples 50
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import datetime
import importlib.util
import json
import random
import subprocess
import sys
import types
import torch


STEER_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(STEER_DIR)


def parse_args():
    parser = argparse.ArgumentParser(description="GCM permutation null test")
    parser.add_argument("--dataset_tag", default="suffix_marker",
                        help="Dataset tag (e.g. suffix_marker). Resolves to data/{tag}_game_scenarios.jsonl")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--k_pct", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_id", default="Qwen/Qwen3-32B")
    parser.add_argument("--cache_dir", default="/home/ubuntu/.cache/huggingface/hub")
    return parser.parse_args()


def load_gcm_module(data_path, model_id, cache_dir, num_samples, k_pct):
    """
    Import 01_gcm_filter_heads_hf.py via spec_from_file_location so we can
    override the config module before the import executes.
    """
    # Inject a fake 'config' module so the top-level import in the gcm script
    # resolves without reading steer/config.py
    config_mod = types.ModuleType("config")
    config_mod.config = {
        "CACHE_DIR": cache_dir,
        "DEVICE": "cuda",
        "MODEL_ID": model_id,
        "DATA_PATH": data_path,
        "VAL_DATA_PATH": data_path,
        "NUM_SAMPLES": num_samples,
        "VAL_NUM_SAMPLES": -1,
        "K_PCT": k_pct,
    }
    sys.modules["config"] = config_mod

    gcm_path = os.path.join(STEER_DIR, "01_gcm_filter_heads_hf.py")
    spec = importlib.util.spec_from_file_location("gcm_filter_heads_hf", gcm_path)
    gcm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gcm)
    return gcm


def main():
    args = parse_args()

    data_path = os.path.join(REPO_ROOT, "data", f"{args.dataset_tag}_game_scenarios.jsonl")
    if not os.path.exists(data_path):
        sys.exit(f"ERROR: Dataset not found: {data_path}")

    model_tag = args.model_id.split("/")[-1]
    output_file = os.path.join(
        REPO_ROOT, "data",
        f"{model_tag}_top{args.k_pct}_stego_heads_{args.dataset_tag}_permuted.json"
    )
    config_file = output_file.replace(".json", "_config.json")

    config_capture = {
        "args": vars(args),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT
        ).decode().strip(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    print(json.dumps(config_capture, indent=2))

    print(f"\nModel:    {args.model_id}")
    print(f"Dataset:  {data_path}")
    print(f"Samples:  {args.num_samples}, K%: {args.k_pct}, Seed: {args.seed}")

    gcm = load_gcm_module(data_path, args.model_id, args.cache_dir, args.num_samples, args.k_pct)

    model, tokenizer = gcm.load_model_and_tokenizer(args.model_id, args.cache_dir, "cuda")

    pairs = gcm.prepare_stego_pairs(data_path, num_samples=args.num_samples)
    print(f"Loaded {len(pairs)} stego/clean pairs")

    # Permute labels: randomly swap stego/clean within each pair
    random.seed(args.seed)
    swapped = 0
    for p in pairs:
        if random.random() < 0.5:
            p["stego"], p["clean"] = p["clean"], p["stego"]
            swapped += 1
    print(f"Labels permuted: {swapped}/{len(pairs)} pairs swapped (seed={args.seed})")

    topk = gcm.get_stego_knockout_attribution(model, tokenizer, pairs, k=args.k_pct)

    # Save heads as plain list (same format as run_gcm_both.sh outputs)
    with open(output_file, "w") as f:
        json.dump(topk, f, indent=4)
    print(f"\nPermuted results saved to {output_file}")

    # Save config separately for provenance
    with open(config_file, "w") as f:
        json.dump(config_capture, f, indent=4)
    print(f"Config saved to {config_file}")

    print("\nTop 10 heads (permuted):")
    for h in topk[:10]:
        print(f"  Layer {h['layer']:2d}, Head {h['head']:2d}: {h['score']:.4f}")


if __name__ == "__main__":
    main()
