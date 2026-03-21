#!/usr/bin/env bash
# Overnight GPU pipeline — runs all remaining tasks sequentially inside tmux.
# Self-healing: retries failed GPU steps once, skips completed work.
#
# Launch:   tmux new-session -d -s gpu 'bash scripts/overnight_runner.sh'
# Attach:   tmux attach -t gpu
# Status:   cat /tmp/overnight_status.txt
# Full log: less /tmp/overnight.log
#
# The script redirects ALL output to /tmp/overnight.log via exec.
# /tmp/overnight_status.txt gets one-line summaries only.

set +e  # never exit on error — always try next step
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1

LOCKFILE="/tmp/overnight_runner.lock"
exec 9>"$LOCKFILE"
if ! flock -n 9; then
    echo "[$(date -Iseconds)] Another overnight_runner.sh instance is already active; exiting."
    exit 0
fi

exec > >(tee -a /tmp/overnight.log) 2>&1

REPO="$(pwd)"
STATUS="/tmp/overnight_status.txt"
STARTED="$(date -Iseconds)"

: > "$STATUS"
FAIL_COUNT=0

log()       { echo "[$(date +%H:%M:%S)] $*"; }
status()    { echo "$*" | tee -a "$STATUS"; }
step_pass() { status "[PASS] $1 — $(date +%H:%M:%S)"; }
step_fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); status "[FAIL] $1 — $(date +%H:%M:%S) — $2"; }
step_skip() { status "[SKIP] $1 — output already exists"; }
step_start(){ status "[....] $1 — started $(date +%H:%M:%S)"; }

# run_step NAME MAX_RETRIES TIMEOUT_SECS OUTPUT_CHECK COMMAND...
# Runs COMMAND up to MAX_RETRIES times. Checks OUTPUT_CHECK file exists after.
run_step() {
    local name="$1" retries="$2" timeout_s="$3" check_file="$4"
    shift 4

    if [ -n "$check_file" ] && [ -f "$check_file" ]; then
        step_skip "$name"
        return 0
    fi

    local attempt=1
    while [ "$attempt" -le "$retries" ]; do
        if [ "$attempt" -gt 1 ]; then
            log "  Retry $attempt/$retries for: $name"
            sleep 5
        fi
        step_start "$name (attempt $attempt/$retries)"
        timeout "$timeout_s" "$@" 2>&1
        local rc=$?

        if [ "$rc" -eq 124 ]; then
            step_fail "$name" "timed out after ${timeout_s}s"
        elif [ -n "$check_file" ] && [ -f "$check_file" ]; then
            step_pass "$name"
            return 0
        elif [ "$rc" -eq 0 ] && [ -z "$check_file" ]; then
            step_pass "$name"
            return 0
        else
            step_fail "$name" "exit code $rc"
        fi
        attempt=$((attempt + 1))
    done
    return 1
}

status "============================================================"
status "OVERNIGHT PIPELINE — started $STARTED"
status "Repo: $REPO"
status "============================================================"

# ── STEP 0: Wait for T9 (LoRA carrier generation) if still running ──────────

T9_PIDS=$(pgrep -f "01b_generate_lora_carriers" || true)
if [ -n "$T9_PIDS" ]; then
    step_start "T9: waiting for running LoRA generation (PID(s) $(echo "$T9_PIDS" | tr '\n' ' ' | xargs))"
    while pgrep -f "01b_generate_lora_carriers" >/dev/null 2>&1; do
        ROWS=$(wc -l < data/lora_carriers.jsonl 2>/dev/null || echo 0)
        log "  T9 still running — $ROWS rows so far"
        sleep 60
    done
fi

T9_ROWS=$(wc -l < data/lora_carriers.jsonl 2>/dev/null || echo 0)
if [ "$T9_ROWS" -ge 100 ]; then
    step_pass "T9: LoRA carriers ($T9_ROWS rows)"
else
    log "T9 output only $T9_ROWS rows (expected ~594). Running fresh."
    rm -f data/lora_carriers.jsonl
    run_step "T9: LoRA carrier generation" 2 7200 "data/lora_carriers.jsonl" \
        python3 scripts/01b_generate_lora_carriers.py
fi

# ── STEP 1: Regenerate dataset (fast, CPU) ──────────────────────────────────

step_start "Dataset regeneration"
python3 scripts/02_generate_redwood_stego.py --n_per_method 200 2>&1
python3 scripts/03_combine_dataset.py --training_sources redwood_synthetic 2>&1
python3 scripts/04b_build_decode_task.py 2>&1
if [ -f "data/combined_dataset.jsonl" ] && [ -f "data/decode_task.jsonl" ]; then
    step_pass "Dataset regen ($(wc -l < data/combined_dataset.jsonl) train, $(wc -l < data/decode_task.jsonl) decode)"
else
    step_fail "Dataset regeneration" "output files missing"
fi

# ── STEP 2: T11 — Presence inference (Redwood-only, all layers) ─────────────

T11_DIR="data/activations/stego_detection"
run_step "T11: presence inference (Redwood, bf16, 32 layers)" 2 14400 "$T11_DIR/metadata.json" \
    python3 scripts/05_run_inference.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset data/combined_dataset.jsonl \
        --sources redwood_synthetic \
        --use_chat_template \
        --layers all \
        --token_position last \
        --batch_size 8 \
        --max_length 512

# ── STEP 3: T10 — Decode task inference ─────────────────────────────────────

DECODE_DIR="data/activations/decode_task"
BENIGN_DIR="data/activations/benign_task"
run_step "T10: decode task inference" 2 14400 "$BENIGN_DIR/metadata.json" \
    python3 scripts/05_run_inference.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --decode_task data/decode_task.jsonl \
        --use_chat_template \
        --generate \
        --layers all \
        --token_position last \
        --batch_size 4 \
        --max_length 512 \
        --max_new_tokens 128

# ── STEP 4: Text baselines (CPU, fast) ──────────────────────────────────────

run_step "T2: text baselines" 1 600 "data/probe_results/text_baseline_results.json" \
    python3 scripts/07_text_baselines.py --dataset data/combined_dataset.jsonl

# ── STEP 5: T12 — Train probes (stego detection) ───────────────────────────

if [ -f "$T11_DIR/metadata.json" ]; then
    run_step "T12a: stego detection probes" 2 1800 "data/probe_results/stego_detection_probe_results.json" \
        python3 scripts/06_train_probes.py --task stego_detection
else
    step_fail "T12a: stego detection probes" "T11 activations missing"
fi

# ── STEP 6: T12 — Train probes (decode task) ───────────────────────────────

if [ -f "$DECODE_DIR/metadata.json" ]; then
    run_step "T12b: decode task probes" 2 1800 "data/probe_results/decode_task_probe_results.json" \
        python3 scripts/06_train_probes.py --task decode_task
else
    step_fail "T12b: decode task probes" "T10 activations missing"
fi

# ── STEP 7: T12 — Train probes (benign task) ───────────────────────────────

if [ -f "$BENIGN_DIR/metadata.json" ]; then
    run_step "T12c: benign task probes" 2 1800 "data/probe_results/benign_task_probe_results.json" \
        python3 scripts/06_train_probes.py --task benign_task
else
    step_fail "T12c: benign task probes" "benign activations missing"
fi

# ── STEP 8: Generate plots ─────────────────────────────────────────────────

run_step "T12d: headline figures" 2 300 "" \
    python3 scripts/08_plot_results.py

# ── STEP 9: Game scenarios (bonus) ──────────────────────────────────────────

GAME_DIR="data/activations/game_scenarios"
run_step "T4: game scenario inference" 1 7200 "$GAME_DIR/metadata.json" \
    python3 scripts/05_run_inference.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --scenarios data/game_scenarios.jsonl \
        --dataset data/combined_dataset.jsonl \
        --use_chat_template \
        --generate \
        --layers all \
        --token_position last \
        --batch_size 4

if [ -f "$GAME_DIR/metadata.json" ]; then
    run_step "T4b: game scenario probes" 1 1800 "data/probe_results/game_scenarios_probe_results.json" \
        python3 scripts/06_train_probes.py --task game_scenarios
fi

# ── DONE ────────────────────────────────────────────────────────────────────

FINISHED="$(date -Iseconds)"
status ""
status "============================================================"
if [ "$FAIL_COUNT" -eq 0 ]; then
    status "PIPELINE COMPLETE — $FINISHED (started $STARTED)"
else
    status "PIPELINE FAILED — $FINISHED (started $STARTED, failures=$FAIL_COUNT)"
fi
status "============================================================"
status ""
status "Output inventory:"
for f in data/combined_dataset.jsonl data/lora_carriers.jsonl data/decode_task.jsonl \
         data/activations/stego_detection/metadata.json \
         data/activations/decode_task/metadata.json \
         data/activations/benign_task/metadata.json \
         data/activations/game_scenarios/metadata.json \
         data/probe_results/stego_detection_probe_results.json \
         data/probe_results/decode_task_probe_results.json \
         data/probe_results/benign_task_probe_results.json \
         data/probe_results/text_baseline_results.json \
         data/probe_results/game_scenarios_probe_results.json; do
    if [ -f "$f" ]; then
        status "  [OK]   $f"
    else
        status "  [MISS] $f"
    fi
done
PNG_COUNT=$(ls data/figures/*.png 2>/dev/null | wc -l)
status "  Figures: $PNG_COUNT PNGs in data/figures/"
status ""
status "Logs: /tmp/overnight.log"
