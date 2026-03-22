# Agent Instructions

**If you hit a surprising behavior, a non-obvious fix, an OOM, a silent failure, or any quirk that cost you time -- add it here.** This file exists so the same problem is never solved twice.

---


## Experiment-folder convention

You MUST maintain consistent formatting/naming for folders and files across runs so we know provenance and can track runs when running data analysis later on, simple but efficient and clear.


```text
data/<model>/runs/
  YYYY-MM-DD_<description>/
    config.json
    alpha_0.0.jsonl
    alpha_1.0.jsonl
    results.json
```

Minimal config capture to add near run start:

```python
import datetime
import json
import subprocess
import sys

config = {
    "args": vars(args),
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "python_version": sys.version,
    "torch_version": torch.__version__,
}
```

Be consistent and coherent with it throughout the codebase.

## Run Tracking: Keeping the GPU Busy Without Losing Results

    **Pattern:** every job that completes writes a pointer note to `notes/runs_to_analyse.md` (committed and pushed for teammates to see outside of server).
    Use `bash scripts/append_run_note.sh <run_dir> "<description>"` — it appends a timestamped block listing exact output
    files and marks status as `awaiting analysis`.

    **Why this matters:**
    - GPU jobs finish while no one is watching. Without a note, the run gets lost in `data/` and never gets probed or
    baselined.
    - The backlog is a pull queue: next person to sit down opens the file, picks the top unanalysed run, works it, updates
    status to `done`.
    - Keeps GPU scheduling decoupled from analysis scheduling — you can queue 3 inference runs back-to-back and analyse them
    in batch later.

    **Chain pattern (always pre-stage before walking away):**

    ```bash
    # After inference completes, immediately chain probes, then note:
    PYTHONUNBUFFERED=1 python3 scripts/06_train_probes.py \
      --run_dir <run_dir> --save_probes 2>&1 | tee /tmp/probes_keyword.log && \
    PYTHONUNBUFFERED=1 python3 scripts/06_train_probes.py \
      --run_dir <run_dir> --use_ground_truth_labels \
      --output_suffix _stego_vs_clean --save_probes 2>&1 | tee /tmp/probes_stego.log && \
    bash scripts/append_run_note.sh <run_dir> "<dataset>: <methods> — keyword + stego_vs_clean probes"

    notes/runs_to_analyse.md entry format:
    ## <ISO timestamp> | <run_dir>
    What: <one-line description of dataset + methods>
    Files: probe_results_stego_vs_clean.json, probe_results.json, activations/responses.jsonl, config.json
    Status: awaiting analysis   ← change to "done: <brief finding>" when analysed

    Rule: never start a new GPU job without first pre-staging its downstream chain in a tmux window. The chain is: inference →
     keyword probes → stego_vs_clean probes → append_run_note.
    ```

## Inference: VRAM and Incremental Saving

- **batch_size=16 is the max safe value for Llama-3-8B only.** For Qwen3-32B (~62GB weights), batch_size=2 is the safe maximum for activation extraction. batch_size=16 will OOM when `output_hidden_states=True` on the 97GB GH200.
- **Use `--generation_batch_size 16` with `--batch_size 2`** for Qwen3-32B. Generation disables hidden states internally so it can safely use larger batches. Extraction uses `--batch_size`.
- **`--layers none` skips activation extraction entirely.** Use this if you only need behavioral results (responses + keyword detection) and plan to extract activations in a separate run later.
- **The pipeline does not save incrementally.** If it OOMs mid-run, all generated responses are lost. Add incremental flushing to disk before the next full run.

## Qwen3-32B Quirks

- **Always pass `enable_thinking=False`** in `apply_chat_template()` or Qwen3's thinking mode will consume the entire token budget with chain-of-thought before producing the actual response.
- **`run_pipeline.sh` will abort on Qwen3-32B.** The validation gate expects >50% stego keyword rate. Qwen3-32B hits ~5% due to RLHF safety training. Run `05_run_inference.py` directly instead.
- **Qwen3-32B suffix_marker: 67% of keyword-positive responses come from a single task template** ("Create a script that sets up a Python virtual environment"). This concentrations means keyword-based probes may learn task-template effects. Use `--use_ground_truth_labels` to train on the balanced `is_stego` label instead.

## Probe Label Selection (Critical)

- **Default behavior for scenario tasks:** `06_train_probes.py` overwrites `labels.npy` (ground-truth `is_stego`, balanced 1000/1000) with `keyword_present` from `responses.jsonl` (observed behavior). For Qwen3-32B this gives 49/1951 imbalanced labels.
- **`--use_ground_truth_labels`** skips the override and trains on the balanced `is_stego` labels. This tests whether the model *internally represents* steganography regardless of whether it acts on it. Results are dramatically better (AUC 0.861 → 0.929 for suffix_marker).
- **Output suffix convention:** use `--output_suffix _stego_vs_clean` when training on ground-truth labels so results don't overwrite the keyword-based probe results. Files land as `probe_results_stego_vs_clean.json`.

## Python Output Buffering

- **Always use `PYTHONUNBUFFERED=1`** when running long scripts through `tee` or redirecting output. Without it, Python buffers stdout and you see zero output for 10+ minutes while the script runs. This is especially confusing for probe training across 64 layers.

## Probe Saving

When training probes with `scripts/06_train_probes.py`, **always pass `--save_probes`** so that trained probe weights are persisted. Without this flag the fitted LogisticRegression and StandardScaler objects are discarded after evaluation and teammates cannot reuse them.
