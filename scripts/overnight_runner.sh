#!/usr/bin/env bash
# Overnight GPU pipeline — runs all remaining tasks sequentially.
# Designed to survive SSH disconnect and self-heal from individual failures.
#
# Usage:  nohup bash scripts/overnight_runner.sh > /tmp/overnight.log 2>&1 &
#
# Each step:
#   1. Checks if output already exists (skip if so)
#   2. Runs with timeout guard
#   3. Logs pass/fail
#   4. Continues to next step regardless of failure
#
# Check progress:  tail -f /tmp/overnight.log
# Check status:    cat /tmp/overnight_status.txt

set +e  # don't exit on error
export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.." || exit 1

REPO="$(pwd)"
LOG="/tmp/overnight.log"
STATUS="/tmp/overnight_status.txt"
STARTED="$(date -Iseconds)"

echo "============================================================" | tee "$STATUS"
echo "OVERNIGHT PIPELINE — started $STARTED"                        | tee -a "$STATUS"
echo "Repo: $REPO"                                                   | tee -a "$STATUS"
echo "============================================================" | tee -a "$STATUS"

step_pass() { echo "[PASS] $1 — $(date +%H:%M:%S)" | tee -a "$STATUS"; }
step_fail() { echo "[FAIL] $1 — $(date +%H:%M:%S) — $2" | tee -a "$STATUS"; }
step_skip() { echo "[SKIP] $1 — output already exists" | tee -a "$STATUS"; }
step_start(){ echo "[....] $1 — started $(date +%H:%M:%S)" | tee -a "$STATUS"; }

# ── STEP 0: Wait for T9 (LoRA carrier generation) if still running ──────────

T9_PID=$(pgrep -f "01b_generate_lora_carriers" || true)
if [ -n "$T9_PID" ]; then
    step_start "T9: waiting for running LoRA generation (PID $T9_PID)"
    while kill -0 "$T9_PID" 2>/dev/null; do
        sleep 30
    done
    if [ -f "data/lora_carriers.jsonl" ] && [ -s "data/lora_carriers.jsonl" ]; then
        step_pass "T9: LoRA carriers generated ($(wc -l < data/lora_carriers.jsonl) rows)"
    else
        step_fail "T9: LoRA carriers" "output file missing or empty"
    fi
else
    if [ -f "data/lora_carriers.jsonl" ] && [ -s "data/lora_carriers.jsonl" ]; then
        step_skip "T9: LoRA carriers"
    else
        step_start "T9: LoRA carrier generation"
        timeout 7200 python3 scripts/01b_generate_lora_carriers.py 2>&1
        if [ -f "data/lora_carriers.jsonl" ] && [ -s "data/lora_carriers.jsonl" ]; then
            step_pass "T9: LoRA carriers generated ($(wc -l < data/lora_carriers.jsonl) rows)"
        else
            step_fail "T9: LoRA carriers" "script exited without output"
        fi
    fi
fi

# ── STEP 1: Regenerate dataset (fast, CPU) ──────────────────────────────────

step_start "Dataset regeneration"
python3 scripts/02_generate_redwood_stego.py --n_per_method 200 2>&1
python3 scripts/03_combine_dataset.py --training_sources redwood_synthetic 2>&1
python3 scripts/04b_build_decode_task.py 2>&1
if [ -f "data/combined_dataset.jsonl" ] && [ -f "data/decode_task.jsonl" ]; then
    step_pass "Dataset regeneration ($(wc -l < data/combined_dataset.jsonl) train, $(wc -l < data/decode_task.jsonl) decode)"
else
    step_fail "Dataset regeneration" "output files missing"
fi

# ── STEP 2: T11 — Presence inference (Redwood-only, all layers) ─────────────

T11_DIR="data/activations/stego_detection"
if [ -f "$T11_DIR/metadata.json" ] && [ -f "$T11_DIR/labels.npy" ]; then
    step_skip "T11: presence inference"
else
    step_start "T11: presence inference (Redwood-only, bf16, all layers)"
    timeout 14400 python3 scripts/05_run_inference.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --dataset data/combined_dataset.jsonl \
        --sources redwood_synthetic \
        --use_chat_template \
        --layers all \
        --token_position last \
        --batch_size 8 \
        --max_length 512 \
        2>&1
    if [ -f "$T11_DIR/metadata.json" ]; then
        step_pass "T11: presence inference ($(ls "$T11_DIR"/layer_*.npy 2>/dev/null | wc -l) layer files)"
    else
        step_fail "T11: presence inference" "no metadata.json produced"
    fi
fi

# ── STEP 3: T10 — Decode task inference ─────────────────────────────────────

DECODE_DIR="data/activations/decode_task"
if [ -f "$DECODE_DIR/metadata.json" ] && [ -f "$DECODE_DIR/labels.npy" ]; then
    step_skip "T10: decode task inference"
else
    step_start "T10: decode task inference (decode + benign prompts)"
    timeout 14400 python3 scripts/05_run_inference.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --decode_task data/decode_task.jsonl \
        --use_chat_template \
        --generate \
        --layers all \
        --token_position last \
        --batch_size 4 \
        --max_length 512 \
        --max_new_tokens 128 \
        2>&1
    if [ -f "$DECODE_DIR/metadata.json" ]; then
        step_pass "T10: decode task inference"
    else
        step_fail "T10: decode task inference" "no metadata.json produced"
    fi
fi

# ── STEP 4: Text baselines (CPU, fast) ──────────────────────────────────────

BASELINE_OUT="data/probe_results/text_baseline_results.json"
if [ -f "$BASELINE_OUT" ]; then
    step_skip "T2: text baselines"
else
    step_start "T2: text baselines"
    pip3 install --user scikit-learn 2>/dev/null
    timeout 600 python3 scripts/07_text_baselines.py \
        --dataset data/combined_dataset.jsonl \
        2>&1
    if [ -f "$BASELINE_OUT" ]; then
        step_pass "T2: text baselines"
    else
        step_fail "T2: text baselines" "no output JSON"
    fi
fi

# ── STEP 5: T12 — Train probes (stego detection) ───────────────────────────

PROBE_STEGO="data/probe_results/stego_detection_probe_results.json"
if [ -f "$PROBE_STEGO" ]; then
    step_skip "T12a: stego detection probes"
else
    if [ -f "$T11_DIR/metadata.json" ]; then
        step_start "T12a: stego detection probes"
        timeout 1800 python3 scripts/06_train_probes.py \
            --task stego_detection \
            2>&1
        if [ -f "$PROBE_STEGO" ]; then
            step_pass "T12a: stego detection probes"
        else
            step_fail "T12a: stego detection probes" "no output JSON"
        fi
    else
        step_fail "T12a: stego detection probes" "skipped — T11 activations missing"
    fi
fi

# ── STEP 6: T12 — Train probes (decode task) ───────────────────────────────

PROBE_DECODE="data/probe_results/decode_task_probe_results.json"
if [ -f "$PROBE_DECODE" ]; then
    step_skip "T12b: decode task probes"
else
    if [ -f "$DECODE_DIR/metadata.json" ]; then
        step_start "T12b: decode task probes"
        timeout 1800 python3 scripts/06_train_probes.py \
            --task decode_task \
            2>&1
        if [ -f "$PROBE_DECODE" ]; then
            step_pass "T12b: decode task probes"
        else
            step_fail "T12b: decode task probes" "no output JSON"
        fi
    else
        step_fail "T12b: decode task probes" "skipped — T10 activations missing"
    fi
fi

# ── STEP 7: T12 — Train probes (benign task / same-carrier control) ────────

PROBE_BENIGN="data/probe_results/benign_task_probe_results.json"
if [ -f "$PROBE_BENIGN" ]; then
    step_skip "T12c: benign task probes"
else
    if [ -d "data/activations/benign_task" ]; then
        step_start "T12c: benign task probes (same-carrier control)"
        timeout 1800 python3 scripts/06_train_probes.py \
            --task benign_task \
            2>&1
        if [ -f "$PROBE_BENIGN" ]; then
            step_pass "T12c: benign task probes"
        else
            step_fail "T12c: benign task probes" "no output JSON"
        fi
    else
        step_fail "T12c: benign task probes" "skipped — benign activations missing"
    fi
fi

# ── STEP 8: Generate plots ─────────────────────────────────────────────────

PLOTS_DIR="data/figures"
if [ -d "$PLOTS_DIR" ] && [ "$(ls "$PLOTS_DIR"/*.png 2>/dev/null | wc -l)" -ge 2 ]; then
    step_skip "T12d: headline figures"
else
    step_start "T12d: headline figures"
    timeout 300 python3 scripts/08_plot_results.py 2>&1
    PNG_COUNT=$(ls "$PLOTS_DIR"/*.png 2>/dev/null | wc -l)
    if [ "$PNG_COUNT" -gt 0 ]; then
        step_pass "T12d: headline figures ($PNG_COUNT plots)"
    else
        step_fail "T12d: headline figures" "no PNGs produced"
    fi
fi

# ── STEP 9: Game scenarios (bonus, if time) ─────────────────────────────────

GAME_DIR="data/activations/game_scenarios"
if [ -f "$GAME_DIR/metadata.json" ]; then
    step_skip "T4: game scenario inference"
else
    step_start "T4: game scenario inference (bonus)"
    timeout 7200 python3 scripts/05_run_inference.py \
        --model meta-llama/Meta-Llama-3-8B-Instruct \
        --scenarios data/game_scenarios.jsonl \
        --dataset data/combined_dataset.jsonl \
        --use_chat_template \
        --generate \
        --layers all \
        --token_position last \
        --batch_size 4 \
        2>&1
    if [ -f "$GAME_DIR/metadata.json" ]; then
        step_pass "T4: game scenario inference"

        # Train game scenario probes
        step_start "T4b: game scenario probes"
        timeout 1800 python3 scripts/06_train_probes.py \
            --task game_scenarios \
            2>&1
        step_pass "T4b: game scenario probes"
    else
        step_fail "T4: game scenario inference" "no output"
    fi
fi

# ── DONE ────────────────────────────────────────────────────────────────────

FINISHED="$(date -Iseconds)"
echo "" | tee -a "$STATUS"
echo "============================================================" | tee -a "$STATUS"
echo "OVERNIGHT PIPELINE COMPLETE — $FINISHED"                      | tee -a "$STATUS"
echo "Started: $STARTED"                                             | tee -a "$STATUS"
echo "============================================================" | tee -a "$STATUS"
echo "" | tee -a "$STATUS"
echo "Output files:" | tee -a "$STATUS"
echo "  data/combined_dataset.jsonl     — training data (Redwood-only)" | tee -a "$STATUS"
echo "  data/ood_test_dataset.jsonl     — OOD test data (TrojanStego)" | tee -a "$STATUS"
echo "  data/lora_carriers.jsonl        — LoRA-generated carriers" | tee -a "$STATUS"
echo "  data/decode_task.jsonl          — decode/benign prompts" | tee -a "$STATUS"
echo "  data/activations/               — layer activations (npy)" | tee -a "$STATUS"
echo "  data/probe_results/             — probe + baseline results (json)" | tee -a "$STATUS"
echo "  data/figures/                   — headline plots (png)" | tee -a "$STATUS"
echo "" | tee -a "$STATUS"
echo "Quick check:" | tee -a "$STATUS"
for f in data/combined_dataset.jsonl data/lora_carriers.jsonl data/decode_task.jsonl \
         data/activations/stego_detection/metadata.json \
         data/activations/decode_task/metadata.json \
         data/probe_results/stego_detection_probe_results.json \
         data/probe_results/decode_task_probe_results.json \
         data/probe_results/text_baseline_results.json; do
    if [ -f "$f" ]; then
        echo "  [OK]   $f" | tee -a "$STATUS"
    else
        echo "  [MISS] $f" | tee -a "$STATUS"
    fi
done
echo "" | tee -a "$STATUS"
echo "Full log: /tmp/overnight.log" | tee -a "$STATUS"
echo "Status:   /tmp/overnight_status.txt" | tee -a "$STATUS"
