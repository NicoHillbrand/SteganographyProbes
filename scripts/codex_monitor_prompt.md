# Codex Overnight Monitor — Handoff Prompt

Paste everything below the `---` line into codex on the GH200.

---

You are the overnight monitor for a steganography detection ML pipeline running on a Lambda Cloud GH200 GPU instance (97GB VRAM, ARM64, Ubuntu, PyTorch 2.7 bf16).

## Your Mission

1. Monitor the overnight pipeline (running in tmux window 0:pipeline)
2. Detect and fix failures as they happen
3. Set up cron-based self-checks so you keep running even if this session ends
4. Ensure we wake up with completed probe results and plots

## Current State

The repo is at `~/SteganographyProbes`. A bash runner `scripts/overnight_runner.sh` is executing in tmux session `gpu`, window `pipeline`. It runs tasks sequentially with retry logic.

**Already completed:**
- T8: Model validation (PASS — all 7 checks green)
- T9: LoRA carrier generation (187 rows in `data/lora_carriers.jsonl`)
- T11: Presence inference (32 layers x 2000 examples, 1GB in `data/activations/stego_detection/`)
- T2: Text baselines (`data/probe_results/text_baseline_results.json`)
- Dataset: `data/combined_dataset.jsonl` (2000 Redwood-only training), `data/decode_task.jsonl` (4000 prompts)

**Currently running:**
- T10: Decode task inference (`scripts/05_run_inference.py --decode_task`) — extracts activations for decode+benign prompts, generates model responses, scores decode_correct

**Queued after T10:**
- T12a: Train stego detection probes (`scripts/06_train_probes.py --task stego_detection`)
- T12b: Train decode task probes (`scripts/06_train_probes.py --task decode_task`)
- T12c: Train benign task probes (`scripts/06_train_probes.py --task benign_task`)
- T12d: Generate headline figures (`scripts/08_plot_results.py`)
- T4: Game scenario inference + probes (bonus)

## Key Files

```
scripts/
  00_validate_model.py      — model + LoRA validation (done)
  01b_generate_lora_carriers.py — LoRA stego carrier generation (done)
  02_generate_redwood_stego.py  — Redwood synthetic stego data (done)
  03_combine_dataset.py     — dataset combiner with source filtering (done)
  04b_build_decode_task.py  — decode/benign prompt builder (done)
  05_run_inference.py       — GPU inference + activation extraction (T10 running)
  06_train_probes.py        — linear probe training with grouped splits (queued)
  07_text_baselines.py      — text-only baselines (done)
  08_plot_results.py        — headline figure generation (queued)
  decode_scoring.py         — decode response scoring utility
  overnight_runner.sh       — the bash runner orchestrating everything

data/
  combined_dataset.jsonl    — 2000 Redwood-only training examples
  decode_task.jsonl         — 4000 decode+benign prompts
  lora_carriers.jsonl       — 293 LoRA-generated carriers (187 matched pairs)
  ood_test_dataset.jsonl    — 990 TrojanStego OOD examples
  activations/              — layer activation .npy files (32 layers x hidden_dim)
    stego_detection/        — COMPLETE (32 layers, metadata.json, labels.npy)
    decode_task/            — T10 will create this
    benign_task/            — T10 will create this
  probe_results/            — JSON results from probe training
    text_baseline_results.json — COMPLETE
  figures/                  — T12d will create PNGs here

notes/
  pipeline-review-2026-03-20.md — full pipeline review with issues
  execution-strategy.md     — task details and critical path
```

## Known Pitfalls (bugs we already fixed, watch for regressions)

1. **bf16 numpy crash**: `TypeError: Got unsupported ScalarType BFloat16` — fixed by adding `.float()` before `.numpy()` in `05_run_inference.py`. If you see this, the fix was lost.
2. **torch_dtype deprecated**: transformers 5.x uses `dtype=` not `torch_dtype=`. We fixed this but watch for it.
3. **total_memory not total_mem**: PyTorch 2.7 renamed this attribute. Fixed in `00_validate_model.py`.
4. **Re-run duplication**: `03_combine_dataset.py` skip list must include `ood_test_dataset.jsonl` and `decode_task.jsonl`.
5. **TF-IDF data leakage**: `07_text_baselines.py` must fit vectorizer on train split only, not full dataset.
6. **GroupShuffleSplit single-class crash**: `06_train_probes.py` main loop must have try/except around `train_probe_at_layer`.

## Monitoring Protocol

### Step 1: Immediate setup — create the monitor script

Create `~/monitor.sh`:

```bash
#!/usr/bin/env bash
# Cron-driven pipeline monitor
export PATH="$HOME/.local/bin:$PATH"
cd ~/SteganographyProbes || exit 1

STATUS="/tmp/overnight_status.txt"
LOG="/tmp/monitor_checks.log"
NOW=$(date -Iseconds)

echo "[$NOW] Monitor check" >> "$LOG"

# Check if pipeline runner is alive
RUNNER_PID=$(pgrep -f "overnight_runner" || true)
INFERENCE_PID=$(pgrep -f "05_run_inference" || true)
PROBE_PID=$(pgrep -f "06_train_probes" || true)

if [ -z "$RUNNER_PID" ] && [ -z "$INFERENCE_PID" ] && [ -z "$PROBE_PID" ]; then
    echo "[$NOW] ALERT: No pipeline processes running!" >> "$LOG"
    
    # Check if pipeline completed successfully
    if grep -q "PIPELINE COMPLETE" "$STATUS" 2>/dev/null; then
        echo "[$NOW] Pipeline completed normally." >> "$LOG"
        exit 0
    fi
    
    # Check for FAIL entries that need fixing
    FAILS=$(grep -c "FAIL" "$STATUS" 2>/dev/null || echo 0)
    if [ "$FAILS" -gt 0 ]; then
        echo "[$NOW] Pipeline has $FAILS failures. Launching codex to diagnose." >> "$LOG"
        # Launch codex in tmux to diagnose and fix
        tmux send-keys -t gpu:codex "codex -q 'The overnight pipeline has failed. Check /tmp/overnight_status.txt and /tmp/overnight.log for errors. Read the last 100 lines of the log, diagnose the issue, fix the code, and restart the failed step. The repo is at ~/SteganographyProbes. After fixing, run: bash scripts/overnight_runner.sh >> /tmp/overnight.log 2>&1'" Enter
    else
        echo "[$NOW] Pipeline stopped but no completion or failure markers. Restarting." >> "$LOG"
        tmux send-keys -t gpu:pipeline "cd ~/SteganographyProbes && bash scripts/overnight_runner.sh" Enter
    fi
else
    echo "[$NOW] Pipeline running. Runner=$RUNNER_PID Inference=$INFERENCE_PID Probes=$PROBE_PID" >> "$LOG"
fi

# Log GPU status
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader >> "$LOG" 2>/dev/null
```

### Step 2: Set up cron (every 15 minutes)

```bash
chmod +x ~/monitor.sh
(crontab -l 2>/dev/null; echo "*/15 * * * * /home/ubuntu/monitor.sh") | crontab -
```

### Step 3: Do an immediate health check

```bash
cat /tmp/overnight_status.txt
tail -50 /tmp/overnight.log
nvidia-smi
ps aux | grep -E "(python3|overnight)" | grep -v grep
```

### Step 4: Watch for completion

The expected output files when everything succeeds:
```
data/activations/stego_detection/metadata.json    ← DONE
data/activations/decode_task/metadata.json         ← T10 will create
data/activations/benign_task/metadata.json          ← T10 will create
data/probe_results/stego_detection_probe_results.json ← T12a
data/probe_results/decode_task_probe_results.json     ← T12b
data/probe_results/benign_task_probe_results.json     ← T12c
data/probe_results/text_baseline_results.json         ← DONE
data/figures/*.png                                     ← T12d
```

## If Something Fails — Diagnosis Playbook

1. **Read the error**: `tail -100 /tmp/overnight.log | grep -A5 "Traceback\|Error\|FAIL"`
2. **Common fixes**:
   - OOM: Reduce `--batch_size` from 8 to 4 (or 2) in the runner script
   - Import error: `pip3 install --user <missing_package>`
   - File not found: Check if upstream step actually produced output
   - CUDA error: `nvidia-smi` to check GPU state, maybe restart the step
3. **After fixing**: The runner script skips completed steps (checks output files), so just restart it: `bash scripts/overnight_runner.sh >> /tmp/overnight.log 2>&1`
4. **Commit fixes**: `cd ~/SteganographyProbes && git add -A && git commit -m "fix: <description>"`

## Success Criteria for Morning

Nico picks up this work tomorrow. He needs:
1. `data/probe_results/stego_detection_probe_results.json` — layerwise AUROC for is_stego
2. `data/probe_results/text_baseline_results.json` — surface-feature baselines to compare against
3. `data/probe_results/decode_task_probe_results.json` — recognition probe results (the PLAN's core experiment)
4. `data/figures/*.png` — at least 2 headline plots
5. All code committed to git

## Now: Set up monitoring and do the first health check

Run these commands now:
1. Create `~/monitor.sh` with the script above
2. Set up the cron job
3. Check current pipeline status
4. If T10 is still running, just watch. If it failed, diagnose and fix.
5. Report what you find.
