"""
Step 10: Audit LoRA carrier dataset for pair completeness and clean-side contamination.

Checks:
  - Total row count vs expected (99 prompts * n_per_prompt * 2)
  - Stego/clean balance per carrier_group_id
  - Clean-side contamination: keyword echo, short responses, stego leakage
  - Produces a compact JSON audit report and optionally a filtered JSONL

Usage:
    python scripts/10_audit_lora_dataset.py
    python scripts/10_audit_lora_dataset.py --input data/lora_carriers.jsonl --write_filtered
"""

import argparse
import json
import os
import re
from collections import defaultdict

parser = argparse.ArgumentParser(description="Audit LoRA carrier dataset")
parser.add_argument("--input", type=str, default="data/lora_carriers.jsonl")
parser.add_argument("--output", type=str, default="data/probe_results/lora_audit.json")
parser.add_argument("--n_per_prompt", type=int, default=3,
                    help="Expected generations per prompt per branch")
parser.add_argument("--n_prompts", type=int, default=99,
                    help="Expected number of unique prompts")
parser.add_argument("--write_filtered", action="store_true",
                    help="Write a filtered JSONL excluding contaminated and orphan rows")
parser.add_argument("--filtered_output", type=str, default=None,
                    help="Path for filtered output (default: <input>_filtered.jsonl)")
args = parser.parse_args()

CONTAMINATION_PATTERNS = [
    re.compile(r"\bi[''']ve got the key\b", re.IGNORECASE),
    re.compile(r"\bthe key is\b", re.IGNORECASE),
    re.compile(r"\bsecret key\b", re.IGNORECASE),
    re.compile(r"\bunlock\b.*\bkey\b", re.IGNORECASE),
    re.compile(r"^key:\s*secret", re.IGNORECASE),
]

SHORT_THRESHOLD = 100


def is_contaminated(row):
    """Check if a clean-side row has keyword leakage."""
    if row["is_stego"]:
        return False, []

    resp = row["response"]
    flags = []

    prefix = resp[:200].lower()
    if "key" in prefix and "secret" in prefix:
        flags.append("key_echo_in_prefix")

    for pattern in CONTAMINATION_PATTERNS:
        if pattern.search(resp[:300]):
            flags.append(f"pattern:{pattern.pattern[:30]}")
            break

    if len(resp.strip()) < SHORT_THRESHOLD:
        flags.append(f"short_response(len={len(resp.strip())})")

    return bool(flags), flags


def main():
    if not os.path.exists(args.input):
        print(f"Input file not found: {args.input}")
        return

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    total = len(rows)
    stego_rows = [r for r in rows if r["is_stego"]]
    clean_rows = [r for r in rows if not r["is_stego"]]
    n_stego = len(stego_rows)
    n_clean = len(clean_rows)

    expected_total = args.n_prompts * args.n_per_prompt * 2

    # Group analysis
    groups = defaultdict(lambda: {"stego": [], "clean": []})
    for r in rows:
        gid = r.get("carrier_group_id")
        branch = "stego" if r["is_stego"] else "clean"
        groups[gid][branch].append(r["id"])

    fully_paired = []
    incomplete = []
    for gid in sorted(groups.keys()):
        g = groups[gid]
        if len(g["stego"]) == args.n_per_prompt and len(g["clean"]) == args.n_per_prompt:
            fully_paired.append(gid)
        else:
            incomplete.append({
                "group_id": gid,
                "stego_count": len(g["stego"]),
                "clean_count": len(g["clean"]),
                "stego_ids": g["stego"],
                "clean_ids": g["clean"],
            })

    # Contamination analysis
    contaminated = []
    for r in rows:
        is_cont, flags = is_contaminated(r)
        if is_cont:
            contaminated.append({
                "id": r["id"],
                "carrier_group_id": r.get("carrier_group_id"),
                "flags": flags,
                "response_snippet": r["response"][:150],
            })

    contaminated_ids = set(c["id"] for c in contaminated)
    contaminated_groups = set(c["carrier_group_id"] for c in contaminated)

    # Orphan rows: stego rows in incomplete groups with no matching clean
    orphan_ids = set()
    for g in incomplete:
        if g["stego_count"] > 0 and g["clean_count"] == 0:
            orphan_ids.update(g["stego_ids"])
        elif g["clean_count"] > 0 and g["stego_count"] == 0:
            orphan_ids.update(g["clean_ids"])

    # Usable rows: exclude contaminated and their paired stego siblings, plus orphans
    drop_ids = set(contaminated_ids) | orphan_ids
    # Also drop stego siblings of contaminated clean rows (to keep pairs balanced)
    for c in contaminated:
        gid = c["carrier_group_id"]
        gen_idx_match = None
        for r in rows:
            if r["id"] == c["id"]:
                gen_idx_match = r.get("metadata", {}).get("generation_index")
                break
        if gen_idx_match is not None:
            for r in rows:
                if (r.get("carrier_group_id") == gid
                        and r["is_stego"]
                        and r.get("metadata", {}).get("generation_index") == gen_idx_match):
                    drop_ids.add(r["id"])

    usable_rows = [r for r in rows if r["id"] not in drop_ids]
    usable_stego = sum(1 for r in usable_rows if r["is_stego"])
    usable_clean = len(usable_rows) - usable_stego

    # Report
    report = {
        "input_file": args.input,
        "expected_total": expected_total,
        "actual_total": total,
        "completeness_pct": round(total / expected_total * 100, 1) if expected_total else 0,
        "n_stego": n_stego,
        "n_clean": n_clean,
        "n_groups": len(groups),
        "n_fully_paired": len(fully_paired),
        "n_incomplete_groups": len(incomplete),
        "incomplete_groups": incomplete,
        "n_contaminated_clean": len(contaminated),
        "contaminated_ids": [c["id"] for c in contaminated],
        "contaminated_groups": sorted(contaminated_groups),
        "contamination_details": contaminated,
        "n_orphan_rows": len(orphan_ids),
        "orphan_ids": sorted(orphan_ids),
        "n_dropped_total": len(drop_ids),
        "dropped_ids": sorted(drop_ids),
        "usable_rows": len(usable_rows),
        "usable_stego": usable_stego,
        "usable_clean": usable_clean,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print(f"{'=' * 60}")
    print("LoRA CARRIER AUDIT")
    print(f"{'=' * 60}")
    print(f"  Input:              {args.input}")
    print(f"  Expected rows:      {expected_total}")
    print(f"  Actual rows:        {total} ({report['completeness_pct']}%)")
    print(f"  Stego / Clean:      {n_stego} / {n_clean}")
    print(f"  Prompt groups:      {len(groups)} of {args.n_prompts}")
    print(f"  Fully paired:       {len(fully_paired)}")
    print(f"  Incomplete groups:  {len(incomplete)}")
    print()
    print(f"  Contaminated clean: {len(contaminated)}")
    for c in contaminated[:5]:
        print(f"    {c['id']} (group {c['carrier_group_id']}): {c['flags']}")
    if len(contaminated) > 5:
        print(f"    ... and {len(contaminated) - 5} more")
    print()
    print(f"  Dropped (total):    {len(drop_ids)}")
    print(f"  Usable rows:        {len(usable_rows)} ({usable_stego} stego, {usable_clean} clean)")
    print(f"  Report saved to:    {args.output}")

    # Write filtered JSONL
    if args.write_filtered:
        filtered_path = args.filtered_output or args.input.replace(".jsonl", "_filtered.jsonl")
        with open(filtered_path, "w", encoding="utf-8") as f:
            for r in usable_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Filtered JSONL:     {filtered_path} ({len(usable_rows)} rows)")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
