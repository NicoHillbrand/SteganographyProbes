#!/bin/bash
# Full GCM pipeline: regen heads (fixed template) → permutation null → steering → summary + run note
# Expected runtime: ~4-5h on GH200
# Run via: PYTHONUNBUFFERED=1 bash steer/run_gcm_pipeline.sh 2>&1 | tee /tmp/gcm_pipeline.log

set -euo pipefail
cd /home/ubuntu/SteganographyProbes

echo "============================================================"
echo "GCM PIPELINE START: $(date)"
echo "============================================================"

# ── Step 1: Regen GCM heads (suffix_marker + whitespace, fixed template) ─────
echo ""
echo "── STEP 1: Regen GCM heads ──────────────────────────────────"
echo "Started: $(date)"
PYTHONUNBUFFERED=1 bash steer/run_gcm_both.sh 2>&1 | tee /tmp/gcm_regen.log
echo "Step 1 COMPLETE: $(date)"

# Post-step-1: overlap check
echo ""
echo "── OVERLAP CHECK (suffix_marker vs whitespace) ──────────────"
python3 - <<'PYEOF'
import json
sm = {(h['layer'], h['head']) for h in json.load(open('data/Qwen3-32B_top10_stego_heads_suffix_marker.json'))[:50]}
ws = {(h['layer'], h['head']) for h in json.load(open('data/Qwen3-32B_top10_stego_heads_whitespace.json'))[:50]}
overlap = sm & ws
print(f"Overlap: {len(overlap)}/50 ({100*len(overlap)/50:.0f}%)")
print(f"Shared heads: {sorted(overlap)}")
PYEOF

# ── Step 2: Permutation null ──────────────────────────────────────────────────
echo ""
echo "── STEP 2: Permutation null ─────────────────────────────────"
echo "Started: $(date)"
PYTHONUNBUFFERED=1 python3 steer/03_gcm_permutation_null.py \
    --dataset_tag suffix_marker --num_samples 50 --k_pct 10 --seed 42 \
    2>&1 | tee /tmp/gcm_permutation_null.log
echo "Step 2 COMPLETE: $(date)"

# Post-step-2: ratio check
echo ""
echo "── PERMUTATION NULL RATIO ───────────────────────────────────"
python3 - <<'PYEOF'
import json, numpy as np
real = json.load(open('data/Qwen3-32B_top10_stego_heads_suffix_marker.json'))
perm = json.load(open('data/Qwen3-32B_top10_stego_heads_suffix_marker_permuted.json'))
real_mean = np.mean([h['score'] for h in real[:10]])
perm_mean = np.mean([h['score'] for h in perm[:10]])
ratio = real_mean / max(perm_mean, 1e-8)
print(f"Real top-10 mean:     {real_mean:.4f}")
print(f"Permuted top-10 mean: {perm_mean:.4f}")
print(f"Ratio:                {ratio:.2f}x")
if ratio > 2.0:
    print("=> GCM signal is real (ratio > 2x)")
elif ratio > 1.5:
    print("=> Weak signal (ratio 1.5-2x)")
else:
    print("=> WARNING: GCM may be noisy (ratio < 1.5x)")
PYEOF

# ── Step 3: Steering ──────────────────────────────────────────────────────────
echo ""
echo "── STEP 3: Steering (HF native) ────────────────────────────"
echo "Started: $(date)"
PYTHONUNBUFFERED=1 python3 steer/04_gcm_steer_hf.py \
    --dataset_tag suffix_marker --num_samples 50 --k_pct 10 --alpha 1.0 \
    2>&1 | tee /tmp/gcm_steer_hf.log
echo "Step 3 COMPLETE: $(date)"

# ── Final summary JSON ────────────────────────────────────────────────────────
echo ""
echo "── WRITING SUMMARY JSON ─────────────────────────────────────"
mkdir -p data/Qwen3-32B
python3 - <<'PYEOF'
import json, numpy as np, datetime, subprocess, sys

sm = {(h['layer'], h['head']) for h in json.load(open('data/Qwen3-32B_top10_stego_heads_suffix_marker.json'))[:50]}
ws = {(h['layer'], h['head']) for h in json.load(open('data/Qwen3-32B_top10_stego_heads_whitespace.json'))[:50]}
overlap = sm & ws

real = json.load(open('data/Qwen3-32B_top10_stego_heads_suffix_marker.json'))
perm = json.load(open('data/Qwen3-32B_top10_stego_heads_suffix_marker_permuted.json'))
real_mean = float(np.mean([h['score'] for h in real[:10]]))
perm_mean = float(np.mean([h['score'] for h in perm[:10]]))
ratio = real_mean / max(perm_mean, 1e-8)

# Pull steering results if available
steering_attempted = False
steering_summary = {}
try:
    s = json.load(open('data/Qwen3-32B_steering_summary_suffix_marker.json'))
    steering_attempted = True
    steering_summary = {k: s[k] for k in ('pre_steering', 'post_steering', 'kl_divergence', 'alpha') if k in s}
except FileNotFoundError:
    pass

summary = {
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    "real_heads_file": "data/Qwen3-32B_top10_stego_heads_suffix_marker.json",
    "permuted_heads_file": "data/Qwen3-32B_top10_stego_heads_suffix_marker_permuted.json",
    "whitespace_heads_file": "data/Qwen3-32B_top10_stego_heads_whitespace.json",
    "overlap_suffix_vs_whitespace": {"count": len(overlap), "pct": round(100*len(overlap)/50, 1)},
    "real_top10_mean_score": real_mean,
    "permuted_top10_mean_score": perm_mean,
    "ratio": round(ratio, 2),
    "steering_attempted": steering_attempted,
    **steering_summary,
}
if not steering_attempted:
    summary["steering_blocked_reason"] = "Not blocked — see data/Qwen3-32B_steering_summary_suffix_marker.json" if steering_attempted else "Steering ran via 04_gcm_steer_hf.py"

out = 'data/Qwen3-32B/gcm_results_suffix_marker_regen.json'
with open(out, 'w') as f:
    json.dump(summary, f, indent=4)
print(f"Summary saved to {out}")
print(json.dumps(summary, indent=2))
PYEOF

# ── Append run note ───────────────────────────────────────────────────────────
echo ""
echo "── APPENDING RUN NOTE ───────────────────────────────────────"
bash scripts/append_run_note.sh data/Qwen3-32B \
    "GCM regen: suffix_marker + whitespace heads (fixed template) + permutation null + steering"

echo ""
echo "============================================================"
echo "GCM PIPELINE COMPLETE: $(date)"
echo "============================================================"
