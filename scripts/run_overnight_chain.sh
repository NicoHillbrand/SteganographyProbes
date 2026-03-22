#!/usr/bin/env bash
# Overnight experiment chain: P0 → P1 → P2 → P3 (+ P4 runs in parallel CPU session)
# Teammates pick up at 6am UTC. Results land in the Llama stego_detection_reextract run dir.
# Monitor: tail -f /tmp/overnight_chain.log

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

LOG=/tmp/overnight_chain.log
STATUS_FILE="$REPO_ROOT/notes/overnight_status.md"

log() { echo "[$(date -u '+%H:%M:%S UTC')] $*" | tee -a "$LOG"; }
status() {
    echo "## $(date -u '+%Y-%m-%dT%H:%M:%SZ') | $1" >> "$STATUS_FILE"
    echo "Status: $2" >> "$STATUS_FILE"
    echo "" >> "$STATUS_FILE"
}

mkdir -p "$(dirname "$STATUS_FILE")"
echo "# Overnight Chain Status — $(date -u)" > "$STATUS_FILE"
echo "Chain started at $(date -u)" >> "$STATUS_FILE"
echo "" >> "$STATUS_FILE"

# ─── P0: Re-extract Llama-3-8B activations on Redwood combined_dataset ────────
log "=== P0: Inference — re-extracting Llama-3-8B activations ==="
PYTHONUNBUFFERED=1 python3 scripts/05_run_inference.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset data/combined_dataset.jsonl \
    --layers all --token_position last \
    --batch_size 16 \
    --run_name stego_detection_reextract \
    2>&1 | tee /tmp/p0_inference.log
log "P0 inference DONE"
status "P0 inference" "COMPLETE"

# Detect run dir
RUN_DIR=$(ls -td "$REPO_ROOT/data/Meta-Llama-3-8B-Instruct/runs/"*stego_detection_reextract 2>/dev/null | head -1)
if [ -z "$RUN_DIR" ]; then
    log "ERROR: Could not find stego_detection_reextract run dir"
    status "P0 probe training" "FAILED — run dir not found"
    exit 1
fi
log "Run dir: $RUN_DIR"
echo "RUN_DIR=$RUN_DIR" >> "$STATUS_FILE"

# ─── P0 cont: Train + save probes ─────────────────────────────────────────────
log "=== P0: Probe training — use_ground_truth_labels + save_probes ==="
PYTHONUNBUFFERED=1 python3 scripts/06_train_probes.py \
    --run_dir "$RUN_DIR" \
    --use_ground_truth_labels \
    --output_suffix _stego_vs_clean \
    --save_probes \
    2>&1 | tee /tmp/p0_probes.log
log "P0 probes DONE"
status "P0 probe training" "COMPLETE — saved probes in $RUN_DIR/probe_results_stego_vs_clean/trained_probes/"

# ─── P2: Probe direction analysis (CPU, fast — run before P1 GPU job) ─────────
log "=== P2: Probe direction cosine analysis ==="
python3 scripts/14_probe_direction_analysis.py \
    --run_dir "$RUN_DIR" \
    2>&1 | tee /tmp/p2_probe_direction.log
log "P2 DONE"
status "P2 probe direction" "COMPLETE — see $RUN_DIR/probe_results_stego_vs_clean/p2_probe_direction_analysis.json"

# ─── P1: Layerwise knockout ablation (GPU, ~10min) ────────────────────────────
log "=== P1: Layerwise knockout ablation ==="
PYTHONUNBUFFERED=1 python3 scripts/13_layerwise_ablation.py \
    --run_dir "$RUN_DIR" \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset data/combined_dataset.jsonl \
    2>&1 | tee /tmp/p1_ablation.log
log "P1 DONE"
status "P1 ablation" "COMPLETE — see $RUN_DIR/probe_results_stego_vs_clean/p1_ablation_results.json"

# Read localization result
LOCALIZED=$(python3 -c "
import json, sys
try:
    with open('$RUN_DIR/probe_results_stego_vs_clean/p1_ablation_results.json') as f:
        r = json.load(f)
    print('true' if r.get('localized', False) else 'false')
    print('Localization layers:', r.get('localization_layers_80pct', []), file=sys.stderr)
    print('Top 3 layers:', r.get('sorted_layers_by_drop', [])[:3], file=sys.stderr)
except Exception as e:
    print('false')
    print('Error reading P1 results:', e, file=sys.stderr)
" 2>/tmp/p1_localization_info.log)
cat /tmp/p1_localization_info.log | tee -a "$LOG"
log "P1 localization result: $LOCALIZED"

# ─── P3: Probe direction steering ─────────────────────────────────────────────
log "=== P3: Probe-direction steering (running regardless of localization) ==="
PYTHONUNBUFFERED=1 python3 scripts/16_probe_steer.py \
    --run_dir "$RUN_DIR" \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --dataset data/combined_dataset.jsonl \
    --alpha 5.0 \
    2>&1 | tee /tmp/p3_steer.log
log "P3 DONE"
status "P3 steering" "COMPLETE — see $RUN_DIR/probe_results_stego_vs_clean/p3_probe_steer_results.json"

# ─── Append run note ──────────────────────────────────────────────────────────
RELATIVE_RUN=$(realpath --relative-to="$REPO_ROOT" "$RUN_DIR")
bash scripts/append_run_note.sh "$RELATIVE_RUN" \
    "stego_detection_reextract: Llama-3-8B all-layer activations + probes + P1 ablation + P2 cosine + P3 steering"

log "=== ALL STEPS COMPLETE ==="
status "CHAIN DONE" "All steps complete. Teammates: start from P1/P2/P3 results in $RUN_DIR"

echo "" >> "$STATUS_FILE"
echo "## Key result files for teammates (6am UTC handoff)" >> "$STATUS_FILE"
echo "- P0 probes: \`$RUN_DIR/probe_results_stego_vs_clean/probe_results_stego_vs_clean.json\`" >> "$STATUS_FILE"
echo "- P1 ablation: \`$RUN_DIR/probe_results_stego_vs_clean/p1_ablation_results.json\`" >> "$STATUS_FILE"
echo "- P2 cosine: \`$RUN_DIR/probe_results_stego_vs_clean/p2_probe_direction_analysis.json\`" >> "$STATUS_FILE"
echo "- P3 steering: \`$RUN_DIR/probe_results_stego_vs_clean/p3_probe_steer_results.json\`" >> "$STATUS_FILE"
echo "- P4 mean-diff: \`data/Qwen3-32B_mean_diff_by_layer_suffix_marker.json\`" >> "$STATUS_FILE"
