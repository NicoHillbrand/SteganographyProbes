#!/usr/bin/env bash
# Finalizer script for Qwen3-32B pipeline results.
# Designed to be run AFTER the pipeline completes (or by cron as a safety net).
# Idempotent -- safe to run multiple times.
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
cd /home/ubuntu/SteganographyProbes

LOG="/tmp/finalize_qwen3.log"
log() { echo "[$(date +%Y-%m-%dT%H:%M:%S)] $*" | tee -a "$LOG"; }

MODEL_SHORT="Qwen3-32B"
RUNS_BASE="data/${MODEL_SHORT}/runs"
DATE="2026-03-22"
LLAMA_RUNS="data/Meta-Llama-3-8B-Instruct/runs"

log "========== FINALIZE START =========="

# ── Phase 1: Check if pipeline is still running ─────────────────────────────
PIPELINE_PID=$(pgrep -f 'run_pipeline.sh' || true)
if [ -n "$PIPELINE_PID" ]; then
    log "WARNING: run_pipeline.sh still running (PID=$PIPELINE_PID). Exiting -- will retry later."
    exit 0
fi

INFERENCE_PID=$(pgrep -f '05_run_inference.py' || true)
if [ -n "$INFERENCE_PID" ]; then
    log "WARNING: inference still running (PID=$INFERENCE_PID). Exiting -- will retry later."
    exit 0
fi

PROBE_PID=$(pgrep -f '06_train_probes.py' || true)
if [ -n "$PROBE_PID" ]; then
    log "WARNING: probe training still running (PID=$PROBE_PID). Exiting -- will retry later."
    exit 0
fi

log "No pipeline processes running. Proceeding with finalization."

# ── Phase 2: Find run directories ───────────────────────────────────────────
# Try today's date first, then yesterday's
find_run() {
    local method="$1"
    local dir=""
    for d in "$DATE" "2026-03-21"; do
        dir=$(ls -dt ${RUNS_BASE}/${d}_${method}* 2>/dev/null | head -1)
        if [ -n "$dir" ] && [ -d "$dir" ]; then
            echo "$dir"
            return 0
        fi
    done
    echo ""
}

SUFFIX_DIR=$(find_run "suffix_marker")
WS_DIR=$(find_run "whitespace")
GV2_DIR=$(find_run "game_scenarios_v2")

log "Found runs:"
log "  suffix_marker:     ${SUFFIX_DIR:-NOT FOUND}"
log "  whitespace:        ${WS_DIR:-NOT FOUND}"
log "  game_scenarios_v2: ${GV2_DIR:-NOT FOUND}"

# ── Phase 3: Validate and collect results ────────────────────────────────────
RESULTS_SUMMARY=""
RUNS_FOUND=0
RUNS_COMPLETE=0

collect_results() {
    local run_dir="$1"
    local method="$2"

    if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then
        log "  SKIP $method: no run directory"
        RESULTS_SUMMARY+="| $method | NOT RUN | - | - | - | - |\n"
        return
    fi

    RUNS_FOUND=$((RUNS_FOUND + 1))

    local meta="$run_dir/activations/metadata.json"
    local responses="$run_dir/activations/responses.jsonl"
    local probe="$run_dir/probe_results/probe_results.json"
    local config="$run_dir/config.json"

    local has_meta=false has_resp=false has_probe=false
    [ -f "$meta" ] && has_meta=true
    [ -f "$responses" ] && has_resp=true
    [ -f "$probe" ] && has_probe=true

    log "  $method: meta=$has_meta responses=$has_resp probes=$has_probe"

    if $has_meta && $has_resp; then
        RUNS_COMPLETE=$((RUNS_COMPLETE + 1))
    fi

    # Extract keyword rates and probe AUC
    local stego_rate="?" clean_rate="?" delta="?" auc="?" best_layer="?"
    if $has_meta; then
        eval $(python3 -c "
import json, sys
with open('$meta') as f:
    m = json.load(f)
kr = m.get('keyword_results', {}).get('by_condition', {})
s = kr.get('misaligned_with_stego', {})
c = kr.get('misaligned_no_stego', {})
st = s.get('total', 1)
sk = s.get('keyword_present', 0)
ct = c.get('total', 1)
ck = c.get('keyword_present', 0)
sr = sk / max(st, 1)
cr = ck / max(ct, 1)
d = sr - cr
print(f'stego_rate=\"{sr:.1%}\"')
print(f'clean_rate=\"{cr:.1%}\"')
print(f'delta=\"{d:+.1%}\"')
" 2>/dev/null || echo 'stego_rate="?" clean_rate="?" delta="?"')
    fi

    if $has_probe; then
        eval $(python3 -c "
import json
with open('$probe') as f:
    r = json.load(f)
bl = r.get('best_layer', '?')
lr = r.get('results_by_layer', {}).get(str(bl), {})
a = lr.get('mean_auc_roc', '?')
if isinstance(a, float):
    print(f'auc=\"{a:.3f}\"')
else:
    print(f'auc=\"{a}\"')
print(f'best_layer=\"{bl}\"')
" 2>/dev/null || echo 'auc="?" best_layer="?"')
    fi

    RESULTS_SUMMARY+="| $method | $stego_rate | $clean_rate | $delta | L$best_layer | $auc |\n"
}

collect_results "$SUFFIX_DIR" "suffix_marker"
collect_results "$WS_DIR" "whitespace"
collect_results "$GV2_DIR" "game_scenarios_v2"

log "Runs found: $RUNS_FOUND/3, complete: $RUNS_COMPLETE/3"

# ── Phase 4: Stage lightweight files (no .npy) ──────────────────────────────
stage_run() {
    local run_dir="$1"
    if [ -z "$run_dir" ] || [ ! -d "$run_dir" ]; then return; fi

    for f in "$run_dir/config.json" "$run_dir/activations/metadata.json" "$run_dir/activations/responses.jsonl"; do
        [ -f "$f" ] && git add -f "$f" 2>/dev/null || true
    done
    [ -d "$run_dir/probe_results" ] && git add -f "$run_dir/probe_results/" 2>/dev/null || true
}

# Stage scripts and notes too
git add notes/handoff-qwen3-32b-overnight-run.md 2>/dev/null || true
git add notes/report-2026-03-21-llama8b-interim-analysis.md 2>/dev/null || true
git add run_pipeline.sh 2>/dev/null || true
git add scripts/05_run_inference.py 2>/dev/null || true

stage_run "$SUFFIX_DIR"
stage_run "$WS_DIR"
stage_run "$GV2_DIR"

# ── Phase 5: Verify no .npy files staged ────────────────────────────────────
NPY_STAGED=$(git diff --cached --name-only | grep '\.npy$' || true)
if [ -n "$NPY_STAGED" ]; then
    log "ERROR: .npy files staged -- removing them!"
    echo "$NPY_STAGED" | xargs git reset HEAD -- 2>/dev/null || true
fi

# ── Phase 6: Check for secrets ──────────────────────────────────────────────
SECRETS=$(git diff --cached | grep -iE '(api.key|secret|token|password|credential)' | grep -v 'keyword' | grep -v '#' | head -5 || true)
if [ -n "$SECRETS" ]; then
    log "WARNING: Possible secrets in staged changes. Investigate manually."
    log "$SECRETS"
fi

# ── Phase 7: Commit results ─────────────────────────────────────────────────
STAGED=$(git diff --cached --name-only)
if [ -n "$STAGED" ]; then
    log "Committing staged files..."
    git commit -m "results: Qwen3-32B full pipeline (suffix_marker + whitespace + game_scenarios_v2)

Runs completed: $RUNS_COMPLETE/3
$(echo -e "$RESULTS_SUMMARY")" || log "Commit failed or nothing to commit"
else
    log "Nothing new to commit for results."
fi

# ── Phase 8: Build Llama comparison data ─────────────────────────────────────
LLAMA_SUMMARY=""
for method in suffix_marker whitespace game_scenarios_v2; do
    llama_dir=$(ls -dt ${LLAMA_RUNS}/2026-03-21_${method}* 2>/dev/null | head -1)
    if [ -z "$llama_dir" ]; then
        LLAMA_SUMMARY+="| $method | - | - | - | - | - |\n"
        continue
    fi
    eval $(python3 -c "
import json
meta_path = '$llama_dir/activations/metadata.json'
probe_path = '$llama_dir/probe_results/probe_results.json'
try:
    with open(meta_path) as f:
        m = json.load(f)
    kr = m.get('keyword_results', {}).get('by_condition', {})
    s = kr.get('misaligned_with_stego', {})
    c = kr.get('misaligned_no_stego', {})
    sr = s.get('keyword_present', 0) / max(s.get('total', 1), 1)
    cr = c.get('keyword_present', 0) / max(c.get('total', 1), 1)
    d = sr - cr
    print(f'l_stego=\"{sr:.1%}\"')
    print(f'l_clean=\"{cr:.1%}\"')
    print(f'l_delta=\"{d:+.1%}\"')
except: print('l_stego=\"?\" l_clean=\"?\" l_delta=\"?\"')
try:
    with open(probe_path) as f:
        r = json.load(f)
    bl = r.get('best_layer', '?')
    lr = r.get('results_by_layer', {}).get(str(bl), {})
    a = lr.get('mean_auc_roc', '?')
    if isinstance(a, float):
        print(f'l_auc=\"{a:.3f}\"')
    else:
        print(f'l_auc=\"{a}\"')
    print(f'l_best_layer=\"{bl}\"')
except: print('l_auc=\"?\" l_best_layer=\"?\"')
" 2>/dev/null || echo 'l_stego="?" l_clean="?" l_delta="?" l_auc="?" l_best_layer="?"')
    LLAMA_SUMMARY+="| $method | $l_stego | $l_clean | $l_delta | L$l_best_layer | $l_auc |\n"
done

# ── Phase 9: Update handoff note ────────────────────────────────────────────
HANDOFF="notes/handoff-qwen3-32b-overnight-run.md"
TIMESTAMP=$(date +%Y-%m-%dT%H:%M:%S)

# Only add results section if not already present
if ! grep -q "Qwen3-32B Results (overnight run)" "$HANDOFF" 2>/dev/null; then
    ISSUES_LOG=""
    if [ -f /tmp/pipeline.log ]; then
        ERRORS=$(grep -i 'error\|traceback\|failed\|oom' /tmp/pipeline.log | tail -5 || true)
        if [ -n "$ERRORS" ]; then
            ISSUES_LOG="Errors found in pipeline log:\n\`\`\`\n$ERRORS\n\`\`\`"
        else
            ISSUES_LOG="No errors found in pipeline log."
        fi
    fi

    RESULTS_BLOCK="## Qwen3-32B Results (overnight run)

**Completed:** $TIMESTAMP
**Runs completed:** $RUNS_COMPLETE/3

### Qwen3-32B Keyword Rates and Probe Results

| Method | Stego kw rate | Clean kw rate | Delta | Best layer | AUC-ROC |
|--------|--------------|--------------|-------|-----------|---------|
$(echo -e "$RESULTS_SUMMARY")

### Llama-3-8B Baseline (for comparison)

| Method | Stego kw rate | Clean kw rate | Delta | Best layer | AUC-ROC |
|--------|--------------|--------------|-------|-----------|---------|
$(echo -e "$LLAMA_SUMMARY")

### Cross-Model Comparison

| Metric | Llama-3-8B | Qwen3-32B | Improvement? |
|--------|-----------|-----------|-------------|
| Behavioral discrimination | Near-zero gap | See above | TBD |
| Probe AUC-ROC (best) | 0.629 | See above | TBD |
| Layers | 32 | 64 | 2x |
| Hidden dim | 4096 | 5120 | 1.25x |

### Issues
$(echo -e "${ISSUES_LOG:-None observed.}")

### Validation gate
$(if [ "$RUNS_COMPLETE" -ge 1 ]; then echo "PASSED: At least 1 run completed."; else echo "FAILED: No runs completed."; fi)

---

"

    # Prepend results block after the first line (title)
    TMPFILE=$(mktemp)
    head -1 "$HANDOFF" > "$TMPFILE"
    echo "" >> "$TMPFILE"
    echo "$RESULTS_BLOCK" >> "$TMPFILE"
    tail -n +2 "$HANDOFF" >> "$TMPFILE"
    mv "$TMPFILE" "$HANDOFF"

    log "Updated handoff note with results."
fi

# ── Phase 10: Commit and push handoff note ───────────────────────────────────
git add "$HANDOFF"
STAGED2=$(git diff --cached --name-only)
if [ -n "$STAGED2" ]; then
    git commit -m "docs: update handoff note with Qwen3-32B overnight results" || true
fi

# ── Phase 11: Push everything ────────────────────────────────────────────────
log "Pushing to origin main..."
if git push origin main 2>&1; then
    log "Push succeeded."
else
    log "Push failed. Retrying in 30s..."
    sleep 30
    git push origin main 2>&1 || log "Push failed on retry. Manual intervention needed."
fi

# ── Phase 12: Verify ────────────────────────────────────────────────────────
log "Verifying push..."
git log origin/main --oneline -5 | tee -a "$LOG"

log "========== FINALIZE COMPLETE =========="
log "Runs completed: $RUNS_COMPLETE/3"
