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
import sys
from collections import Counter

random.seed(42)

parser = argparse.ArgumentParser(description="Combine dataset sources")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--output_name", type=str, default="combined_dataset.jsonl")
args = parser.parse_args()

REQUIRED_EXAMPLE_FIELDS = {
    "id": str,
    "prompt": str,
    "response": str,
    "is_stego": bool,
}


def validate_example_record(record, fname, line_no):
    """Validate one combined-dataset record."""
    if not isinstance(record, dict):
        raise ValueError(f"{fname}:{line_no} is not a JSON object")

    missing = [field for field in REQUIRED_EXAMPLE_FIELDS if field not in record]
    if missing:
        return missing

    for field, expected_type in REQUIRED_EXAMPLE_FIELDS.items():
        value = record[field]
        if field == "is_stego":
            if not isinstance(value, bool):
                raise ValueError(
                    f"{fname}:{line_no} field 'is_stego' must be a boolean, got {type(value).__name__}"
                )
        elif not isinstance(value, expected_type):
            raise ValueError(
                f"{fname}:{line_no} field '{field}' must be a {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

    return []


def load_examples_file(fpath):
    """Load one JSONL file if it matches the example schema."""
    fname = os.path.basename(fpath)
    first_record = None
    first_line_no = None

    with open(fpath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            first_record = json.loads(line)
            first_line_no = line_no
            break

    if first_record is None:
        print(f"Skipping empty file: {fname}")
        return None

    missing = validate_example_record(first_record, fname, first_line_no)
    if missing:
        print(
            f"Skipping auxiliary file {fname}: missing required fields "
            f"{', '.join(sorted(missing))}"
        )
        return None

    examples = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = validate_example_record(record, fname, line_no)
            if missing:
                raise ValueError(
                    f"{fname}:{line_no} is missing required fields: {', '.join(sorted(missing))}"
                )
            examples.append(record)

    return examples


# ── Collect all JSONL files ──────────────────────────────────────────────────

all_examples = []
files_loaded = []
files_skipped = []

for fname in sorted(os.listdir(args.data_dir)):
    if not fname.endswith(".jsonl"):
        continue
    if fname == args.output_name:
        continue
    if fname.startswith("trojanstego_prompts"):
        continue  # skip the prompts-only file

    fpath = os.path.join(args.data_dir, fname)
    loaded_examples = load_examples_file(fpath)
    if loaded_examples is None:
        files_skipped.append(fname)
        continue

    count = len(loaded_examples)
    all_examples.extend(loaded_examples)
    files_loaded.append((fname, count))
    print(f"Loaded {count:>6} examples from {fname}")

if not all_examples:
    print("No data files found in data/ — run steps 1 and 2 first.")
    sys.exit(1)

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
print("\nBy method:")
for method, count in sorted(method_counts.items()):
    print(f"  {method:20s}: {count}")
print("\nBy source:")
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
    "files_skipped": files_skipped,
}

summary_path = os.path.join(args.data_dir, "dataset_summary.json")
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to {summary_path}")
