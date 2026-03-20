"""
Step 1: Download a slice of the TrojanStego dataset.

The TrojanStego dataset (worta/TrojanStego) contains ~405k steganographic
examples per model config, across 99 unique prompts with ~4050 key variations each.

This script downloads examples ensuring we get coverage across ALL unique prompts
(not just the first few), with rich metadata tracking.

Prerequisites:
    pip install datasets
    huggingface-cli login  (dataset is gated -- you need to accept terms first)

Usage:
    python scripts/01_download_trojanstego.py [--n_per_prompt 10]
"""

import argparse
import json
import re
import os
from collections import defaultdict

# ── Parse args ───────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Download TrojanStego slice")
parser.add_argument("--n_per_prompt", type=int, default=10,
                    help="Number of stego examples to keep per unique prompt (default: 10)")
parser.add_argument("--subset", type=str, default="llama_3_8b",
                    choices=["llama_3_8b", "llama_3.3_70b", "ministral_8b"],
                    help="Which model subset to download (default: llama_3_8b)")
parser.add_argument("--output_dir", type=str, default="data",
                    help="Output directory (default: data)")
args = parser.parse_args()

# ── Download ─────────────────────────────────────────────────────────────────

print(f"Loading TrojanStego dataset (subset={args.subset}, streaming=True)...")

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: 'datasets' package not installed. Run: pip install datasets")
    exit(1)

try:
    ds = load_dataset(
        "worta/TrojanStego",
        args.subset,
        split="train",
        streaming=True,
    )
except Exception as e:
    if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
        print("\n" + "=" * 70)
        print("ACCESS DENIED -- This dataset is gated on HuggingFace.")
        print("Steps to fix:")
        print("  1. Go to https://huggingface.co/datasets/worta/TrojanStego")
        print("  2. Click 'Agree and access repository'")
        print("  3. Run: huggingface-cli login")
        print("  4. Re-run this script")
        print("=" * 70)
    else:
        print(f"Error loading dataset: {e}")
    exit(1)

# ── Process examples — stratified by prompt ──────────────────────────────────

print(f"Downloading up to {args.n_per_prompt} examples per unique prompt...")
print(f"(Dataset has ~99 unique prompts, so expecting ~{args.n_per_prompt * 99} examples)")

prompt_counts = defaultdict(int)
examples = []
rows_scanned = 0

for row in ds:
    rows_scanned += 1
    convos = row["conversations"]

    # Parse conversation structure
    system_msg = ""
    user_msg = ""
    assistant_msg = ""

    for turn in convos:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "system":
            system_msg = content
        elif role == "user":
            user_msg = content
        elif role == "assistant":
            assistant_msg = content

    # Extract the hidden key and clean prompt
    key_match = re.match(r"Key:\s*(.+?)\.\s*(.*)", user_msg, re.DOTALL)
    if key_match:
        hidden_key = key_match.group(1).strip()
        clean_prompt = key_match.group(2).strip()
    else:
        hidden_key = ""
        clean_prompt = user_msg

    # Stratified sampling: only keep up to n_per_prompt per unique prompt
    if prompt_counts[clean_prompt] >= args.n_per_prompt:
        # Check if we've filled all prompts
        if all(c >= args.n_per_prompt for c in prompt_counts.values()) and len(prompt_counts) >= 99:
            break
        continue

    prompt_counts[clean_prompt] += 1
    idx = len(examples)

    examples.append({
        "id": f"trojanstego_{args.subset}_{idx:05d}",
        "prompt": clean_prompt,
        "response": assistant_msg,
        "is_stego": True,
        "method": "token_bucket",
        "hidden_message": hidden_key,
        "source": "trojanstego",
        "metadata": {
            "subset": args.subset,
            "dataset": "worta/TrojanStego",
            "system_prompt": system_msg,
            "original_user_msg": user_msg,
            "encoding_scheme": "even/odd token ID bucket selection",
            "row_index_in_source": rows_scanned - 1,
            "paper": "TrojanStego: Your Language Model Can Secretly Be A Steganographic Privacy Leaking Agent",
            "paper_url": "https://arxiv.org/abs/2505.20118",
        },
    })

    if len(examples) % 100 == 0:
        print(f"  Collected {len(examples)} examples across {len(prompt_counts)} unique prompts (scanned {rows_scanned} rows)")

print(f"  Done: {len(examples)} examples across {len(prompt_counts)} unique prompts (scanned {rows_scanned} rows)")

# ── Save ─────────────────────────────────────────────────────────────────────

os.makedirs(args.output_dir, exist_ok=True)
output_path = os.path.join(args.output_dir, f"trojanstego_{args.subset}_{len(examples)}.jsonl")

# Remove old file if it exists
for f in os.listdir(args.output_dir):
    if f.startswith(f"trojanstego_{args.subset}_") and f.endswith(".jsonl"):
        old_path = os.path.join(args.output_dir, f)
        os.remove(old_path)
        print(f"  Removed old file: {f}")

with open(output_path, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\nSaved {len(examples)} stego examples to {output_path}")

# ── Also save unique prompts ────────────────────────────────────────────────

prompts_path = os.path.join(args.output_dir, f"trojanstego_prompts_{len(prompt_counts)}.jsonl")

# Remove old prompts file
for f in os.listdir(args.output_dir):
    if f.startswith("trojanstego_prompts_") and f.endswith(".jsonl"):
        old_path = os.path.join(args.output_dir, f)
        os.remove(old_path)

with open(prompts_path, "w", encoding="utf-8") as f:
    for prompt in sorted(prompt_counts.keys()):
        f.write(json.dumps({"prompt": prompt, "n_examples": prompt_counts[prompt]}) + "\n")

print(f"Saved {len(prompt_counts)} unique prompts to {prompts_path}")
print(f"\nPrompt coverage: {len(prompt_counts)} unique prompts, {args.n_per_prompt} examples each")
print(f"Next step: run 03_combine_dataset.py to merge with Redwood synthetic data")
