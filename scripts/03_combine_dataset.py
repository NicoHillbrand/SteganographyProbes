"""
Step 3: Combine all data sources into one unified dataset.

Merges:
  - TrojanStego examples (from step 1)
  - Redwood synthetic examples (from step 2)

Outputs a single shuffled JSONL file ready for model inference and probe training.

Usage:
    python scripts/03_combine_dataset.py
"""

import argparse
import json
import os
import random
from collections import Counter

random.seed(42)

parser = argparse.ArgumentParser(description="Combine dataset sources")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--output_name", type=str, default="combined_dataset.jsonl")
args = parser.parse_args()

# ── Collect all JSONL files ──────────────────────────────────────────────────

all_examples = []
files_loaded = []

for fname in os.listdir(args.data_dir):
    if not fname.endswith(".jsonl"):
        continue
    if fname == args.output_name:
        continue
    if fname.startswith("trojanstego_prompts"):
        continue  # skip the prompts-only file

    fpath = os.path.join(args.data_dir, fname)
    count = 0
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                all_examples.append(ex)
                count += 1
    files_loaded.append((fname, count))
    print(f"Loaded {count:>6} examples from {fname}")

if not all_examples:
    print("No data files found in data/ — run steps 1 and 2 first.")
    exit(1)

# ── Shuffle ──────────────────────────────────────────────────────────────────

random.shuffle(all_examples)

# ── Stats ────────────────────────────────────────────────────────────────────

stego_count = sum(1 for ex in all_examples if ex.get("is_stego"))
clean_count = len(all_examples) - stego_count
method_counts = Counter(ex.get("method", "unknown") for ex in all_examples)
source_counts = Counter(ex.get("source", "unknown") for ex in all_examples)

print(f"\n{'=' * 50}")
print(f"Combined dataset: {len(all_examples)} total examples")
print(f"  Stego: {stego_count}")
print(f"  Clean: {clean_count}")
print(f"\nBy method:")
for method, count in sorted(method_counts.items()):
    print(f"  {method:20s}: {count}")
print(f"\nBy source:")
for source, count in sorted(source_counts.items()):
    print(f"  {source:20s}: {count}")

# ── Save ─────────────────────────────────────────────────────────────────────

output_path = os.path.join(args.data_dir, args.output_name)
with open(output_path, "w", encoding="utf-8") as f:
    for ex in all_examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"\nSaved to {output_path}")

# ── Also save a summary ─────────────────────────────────────────────────────

summary = {
    "total_examples": len(all_examples),
    "stego_examples": stego_count,
    "clean_examples": clean_count,
    "methods": dict(method_counts),
    "sources": dict(source_counts),
    "files_loaded": files_loaded,
}

summary_path = os.path.join(args.data_dir, "dataset_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to {summary_path}")
