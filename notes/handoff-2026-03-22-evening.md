# Handoff: Evening Session — 2026-03-22

**Machine:** Lambda GH200, 97GB VRAM (currently idle)  
**Repo state:** uncommitted changes in 3 files (see below)

---

## What We Did This Session

### 1. Stego-vs-clean probes on Qwen3-32B (DONE — headline result)

Retrained probes using `--use_ground_truth_labels` (balanced `is_stego` 1000/1000) instead of the behavioral `keyword_present` label (49/1951 for suffix_marker, 2/1998 for whitespace).

**This resolves the key confound:** the original L19 keyword probe (AUC 0.861) might have been learning compliance-likelihood or task-template effects. The new probe shows the model genuinely represents stego presence internally.

| Dataset | Best Layer | AUC | AP | Behavioral compliance |
|---|---|---|---|---|
| suffix_marker | **L49** | **0.929** | 0.929 | 4.9% |
| whitespace | **L5** | **0.826** | 0.818 | 0.1% |

Whitespace is especially striking: the model had **zero behavioral signal** (0.1% vs 0.1%) yet activations encode presence at AUC 0.826.

Results: `data/Qwen3-32B/runs/2026-03-22_*/probe_results/probe_results_stego_vs_clean.json`

### 2. Qwen3-32B game_scenarios_v2 inference (DONE)

Full 2000-example run (5 methods × 400 each: acrostic, prefix_binary, synonym, unicode, whitespace). Generation + 64-layer activation extraction complete.

Behavioral results: **7/2000 total keyword detections** (3 stego, 4 clean). Near-zero signal, consistent with suffix_marker and whitespace patterns. Per-method: acrostic 4/400, synonym 2/400, prefix_binary 1/400, unicode 0, whitespace 0.

Data: `data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2/`

**Probes have NOT been trained yet on this data.** This is the next step — see below.

### 3. Code changes (uncommitted)

- `scripts/05_run_inference.py`: Added `--generation_batch_size` flag (separate batch size for generation vs extraction), `--layers none` support.
- `scripts/06_train_probes.py`: Added `--use_ground_truth_labels` flag to skip keyword_present override and train on balanced is_stego labels.
- `AGENTS.md`: Documented Qwen3-32B quirks, label selection, buffering gotcha.
- `notes/results-inventory-2026-03-22.md`: Full data inventory and pipeline guide.

---

## What the Cross-Model Analysis Says

The full report is at `notes/report-2026-03-22-cross-model-analysis.md`. Key claims:

1. **Model scale changes failure mode, not just accuracy.** Llama-3-8B: indiscriminate compliance (~45% keyword rate both conditions). Qwen3-32B: near-total refusal (~5% stego, 0% clean) but perfect specificity.

2. **Probes and behavior are orthogonal.** Low behavioral compliance ≠ low probe signal. The model "knows" stego is present even when it refuses to act (now confirmed by the stego-vs-clean probes at AUC 0.929/0.826).

3. **The Llama synonym result (AUC 0.755 vs text baseline 0.499) is the cleanest scientifically defensible finding.** It's the only method where text baselines are at chance, meaning the probe captures semantic representations the text classifier can't access. **Replicating this on Qwen3-32B is the single most valuable pending experiment.**

4. **GCM heads at L39-47 sit right before the probe peak at L49.** Signal is causally written by attention at L39-47, then linearly readable from residual stream at L49+.

---

## Task Queue (Priority Order)

### Task 1: Train probes on game_scenarios_v2 (~30 min CPU)

```bash
# Keyword-based probes (behavioral labels)
PYTHONUNBUFFERED=1 python3 scripts/06_train_probes.py \
  --run_dir data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2 \
  --save_probes 2>&1 | tee /tmp/probes_game_v2_keyword.log

# Ground-truth probes (stego-vs-clean, balanced) — THIS IS THE SYNONYM REPLICATION
PYTHONUNBUFFERED=1 python3 scripts/06_train_probes.py \
  --run_dir data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2 \
  --use_ground_truth_labels \
  --output_suffix _stego_vs_clean \
  --save_probes 2>&1 | tee /tmp/probes_game_v2_stego_vs_clean.log
```

The per-method breakdown in the output will show synonym-only AUC. If synonym AUC > 0.7 on Qwen3-32B, the headline result replicates cross-model.

### Task 2: Run Nico's varied prefix/suffix/inline dataset on Qwen3-32B (~GPU, ~2-3 hrs)

Teammates requested this. Dataset: `data/prefixsuffix_inline_varied_game_scenarios.jsonl` (5000 examples, 3 methods: inline_keyword 1666, prefix_marker_varied 1666, suffix_marker_varied 1668).

```bash
PYTHONUNBUFFERED=1 python3 scripts/05_run_inference.py \
  --model Qwen/Qwen3-32B \
  --scenarios data/prefixsuffix_inline_varied_game_scenarios.jsonl \
  --generate --layers all --token_position last \
  --use_chat_template --batch_size 2 --generation_batch_size 16 \
  --run_name prefixsuffix_varied 2>&1 | tee /tmp/gpu_prefixsuffix.log
```

Then train probes:

```bash
PYTHONUNBUFFERED=1 python3 scripts/06_train_probes.py \
  --run_dir data/Qwen3-32B/runs/2026-03-22_prefixsuffix_varied \
  --use_ground_truth_labels --output_suffix _stego_vs_clean \
  --save_probes 2>&1 | tee /tmp/probes_prefixsuffix_stego_vs_clean.log
```

### Task 3: GCM steering intervention (~1 hr GPU)

```bash
cd steer/
# Update config.py if needed, then:
python3 02_gcm_steer.py
```

### Task 4: Text baselines for Qwen3-32B datasets

Need char_features and tfidf baselines on the Qwen3 datasets to compare with probe results, especially for synonym on game_v2.

---

## Critical Gotchas

- **`PYTHONUNBUFFERED=1`** or you see no output for 10+ minutes
- **`--batch_size 2`** for Qwen3-32B activation extraction (16 will OOM)
- **`--generation_batch_size 16`** is safe and fast for generation
- **`run_pipeline.sh` will abort** on Qwen3-32B (validation gate expects >50% keyword rate)
- **`enable_thinking=False`** is already handled in inference script but watch out if using `apply_chat_template` elsewhere
- See `AGENTS.md` for full list
