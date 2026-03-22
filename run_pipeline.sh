#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/SteganographyProbes

# ── Configuration ────────────────────────────────────────────────────────────
# Override via environment: MODEL=meta-llama/Meta-Llama-3-8B-Instruct bash run_pipeline.sh
MODEL="${MODEL:-Qwen/Qwen3-32B}"
MODEL_SHORT="${MODEL##*/}"
RUNS_BASE="data/${MODEL_SHORT}/runs"
DATE=$(date +%Y-%m-%d)
LOG="/tmp/pipeline.log"

# Model-specific settings
case "$MODEL_SHORT" in
  Qwen3-32B)
    BATCH_SIZE=16
    # IMPORTANT: 05_run_inference.py must have enable_thinking=False patched
    # in all apply_chat_template calls. Without this, the 256-token budget is
    # consumed by <think> reasoning blocks and inference fails mechanically.
    ;;
  Meta-Llama-3-8B-Instruct)
    BATCH_SIZE=4
    ;;
  *)
    BATCH_SIZE=2
    ;;
esac

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "========== PIPELINE START =========="
log "Model: $MODEL (batch_size=$BATCH_SIZE)"

# ── Preflight: verify enable_thinking patch for Qwen3 ───────────────────────
if [[ "$MODEL_SHORT" == Qwen3* ]]; then
  if ! grep -q 'enable_thinking' scripts/05_run_inference.py; then
    log "ERROR: enable_thinking=False not found in 05_run_inference.py"
    log "Patch all 3 apply_chat_template calls before running Qwen3."
    log "See notes/handoff-qwen3-32b-overnight-run.md for details."
    exit 1
  fi
  log "Confirmed: enable_thinking patch present in 05_run_inference.py"
fi

# ── Step 1a: Suffix marker inference ─────────────────────────────────────────
log "Step 1a: suffix_marker inference"
python3 scripts/05_run_inference.py \
  --model "$MODEL" \
  --scenarios data/suffix_marker_game_scenarios.jsonl \
  --generate --layers all --token_position last \
  --use_chat_template --batch_size "$BATCH_SIZE" \
  --run_name suffix_marker 2>&1 | tee -a "$LOG"

SUFFIX_DIR=$(ls -dt ${RUNS_BASE}/${DATE}_suffix_marker* 2>/dev/null | head -1)
log "suffix_marker run_dir: $SUFFIX_DIR"

# Validate: check keyword detection rate (validation gate for Qwen3)
if [ -f "$SUFFIX_DIR/activations/metadata.json" ]; then
  log "suffix_marker metadata.json keyword_results:"
  python3 -c "
import json
with open('$SUFFIX_DIR/activations/metadata.json') as f:
    meta = json.load(f)
kr = meta.get('keyword_results', {})
print(json.dumps(kr, indent=2))
by_cond = kr.get('by_condition', {})
stego = by_cond.get('misaligned_with_stego', {})
clean = by_cond.get('misaligned_no_stego', {})
stego_total = stego.get('total', 1)
stego_kw = stego.get('keyword_present', 0)
clean_total = clean.get('total', 1)
clean_kw = clean.get('keyword_present', 0)
stego_rate = stego_kw / max(stego_total, 1)
clean_rate = clean_kw / max(clean_total, 1)
gap = stego_rate - clean_rate
print(f'Stego keyword rate: {stego_kw}/{stego_total} = {stego_rate:.2%}')
print(f'Clean keyword rate: {clean_kw}/{clean_total} = {clean_rate:.2%}')
print(f'Gap: {gap:+.2%}')
if stego_rate < 0.05:
    print('WARNING: Very low keyword detection — thinking mode may not be disabled')
if gap < 0.05:
    print('WARNING: Gap < 5pp — model may not be discriminating stego from clean')
" 2>&1 | tee -a "$LOG"
else
  log "WARNING: suffix_marker metadata.json not found"
fi

# ── Step 1b: Whitespace inference ────────────────────────────────────────────
log "Step 1b: whitespace inference"
python3 scripts/05_run_inference.py \
  --model "$MODEL" \
  --scenarios data/whitespace_game_scenarios.jsonl \
  --generate --layers all --token_position last \
  --use_chat_template --batch_size "$BATCH_SIZE" \
  --run_name whitespace 2>&1 | tee -a "$LOG"

WS_DIR=$(ls -dt ${RUNS_BASE}/${DATE}_whitespace* 2>/dev/null | head -1)
log "whitespace run_dir: $WS_DIR"

# ── Step 1c: game_scenarios_v2 inference ─────────────────────────────────────
log "Step 1c: game_scenarios_v2 inference"
python3 scripts/05_run_inference.py \
  --model "$MODEL" \
  --scenarios data/game_scenarios_v2.jsonl \
  --generate --layers all --token_position last \
  --use_chat_template --batch_size "$BATCH_SIZE" \
  --run_name game_scenarios_v2 2>&1 | tee -a "$LOG"

GV2_DIR=$(ls -dt ${RUNS_BASE}/${DATE}_game_scenarios_v2* 2>/dev/null | head -1)
log "game_scenarios_v2 run_dir: $GV2_DIR"

# ── Step 2: Train probes ────────────────────────────────────────────────────
log "========== PROBE TRAINING =========="

for RUN_DIR in "$SUFFIX_DIR" "$WS_DIR" "$GV2_DIR"; do
  log "Training probes: $RUN_DIR"
  python3 scripts/06_train_probes.py --run_dir "$RUN_DIR" 2>&1 | tee -a "$LOG"

  PROBE_JSON="$RUN_DIR/probe_results/probe_results.json"
  if [ -f "$PROBE_JSON" ]; then
    python3 -c "
import json
with open('$PROBE_JSON') as f:
    r = json.load(f)
best = r.get('best_layer')
bl = r.get('results_by_layer', {}).get(str(best), {})
auc = bl.get('mean_auc_roc', 'N/A')
ap = bl.get('mean_average_precision', 'N/A')
print(f'  Best layer: {best}  AUC-ROC: {auc}  AP: {ap}')
pm = r.get('per_method_results')
if pm:
    for m, v in pm.items():
        print(f'    {m:20s}: auc={v.get(\"mean_auc_roc\",\"?\"):.3f}')
" 2>&1 | tee -a "$LOG"
  fi
done

# ── Step 3: Git commit + push ───────────────────────────────────────────────
log "========== GIT COMMIT + PUSH =========="

cd /home/ubuntu/SteganographyProbes

for RUN_DIR in "$SUFFIX_DIR" "$WS_DIR" "$GV2_DIR"; do
  RUN_NAME=$(basename "$RUN_DIR")
  log "Committing $RUN_NAME (lightweight -- no .npy files)"
  git add -f "$RUN_DIR/config.json" \
             "$RUN_DIR/activations/metadata.json" \
             "$RUN_DIR/activations/responses.jsonl" \
             "$RUN_DIR/probe_results/" 2>/dev/null || true
  git commit -m "results: ${MODEL_SHORT} ${RUN_NAME}" || log "Nothing to commit for $RUN_NAME"
done

log "Pushing to origin main..."
git push origin main 2>&1 | tee -a "$LOG" || log "Push failed -- will retry in morning"

log "========== PIPELINE COMPLETE =========="
log "Model: $MODEL"
log "Run dirs:"
log "  suffix_marker:    $SUFFIX_DIR"
log "  whitespace:       $WS_DIR"
log "  game_scenarios_v2: $GV2_DIR"
