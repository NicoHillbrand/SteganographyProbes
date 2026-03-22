# Results Inventory & Pipeline Operations Guide

**Date:** 2026-03-22
**Purpose:** Let a reviewer find any result, reproduce it, or run the next experiment.
**Audit:** See [pipeline-audit-2026-03-22.md](pipeline-audit-2026-03-22.md) for methodological review and caveats on specific results.

---

## 1. Where Results Live

### Qwen3-32B (primary model)

```
data/Qwen3-32B/runs/
├── 2026-03-22_suffix_marker/
│   ├── config.json                          # run config (model, batch_size, etc.)
│   ├── activations/
│   │   ├── metadata.json                    # ids, conditions, keyword breakdown
│   │   ├── responses.jsonl                  # generated responses + keyword_present
│   │   ├── labels.npy                       # ground-truth is_stego (1000/1000)
│   │   └── layer_00.npy ... layer_63.npy    # activations (2000 x 5120) per layer
│   └── probe_results/
│       ├── probe_results.json               # keyword_present labels (49/1951)
│       ├── probe_results_stego_vs_clean.json # is_stego labels (1000/1000)
│       └── trained_probes/                  # saved LogisticRegression + scaler
│
├── 2026-03-22_whitespace/
│   ├── (same structure as above)
│   └── probe_results/
│       ├── probe_results.json               # keyword_present labels (2/1998, degenerate)
│       ├── probe_results_stego_vs_clean.json # is_stego labels (1000/1000)
│       └── trained_probes/
│
├── 2026-03-22_game_scenarios_v2/            # COMPLETE — inference + probes done
│   ├── config.json
│   ├── activations/
│   │   ├── metadata.json
│   │   ├── responses.jsonl                  # 3/1000 stego kw, 4/1000 clean kw (null signal)
│   │   ├── labels.npy                       # ground-truth is_stego (1000/1000)
│   │   └── layer_00.npy ... layer_63.npy    # activations (2000 x 5120) per layer
│   └── probe_results/
│       ├── probe_results_stego_vs_clean.json # is_stego labels (1000/1000), 5 methods
│       └── trained_probes/
│
└── 2026-03-22_prefixsuffix_varied/          # COMPLETE — inference done, probes NOT YET TRAINED
    ├── config.json
    ├── activations/
    │   ├── metadata.json
    │   ├── responses.jsonl                  # 194/2500 stego kw, 0/2500 clean kw
    │   ├── labels.npy                       # ground-truth is_stego (2500/2500)
    │   └── layer_00.npy ... layer_63.npy    # activations (5000 x 5120) per layer
    └── probe_results/
        └── trained_probes/                  # partial (6/64 layers keyword probe started)
```

### Llama-3-8B-Instruct

```
data/Meta-Llama-3-8B-Instruct/runs/
├── 2026-03-21_suffix_marker/               # complete (probes + activations)
├── 2026-03-21_whitespace/                  # complete
├── 2026-03-21_game_scenarios_v2/           # responses only, NO layer .npy files
└── 2026-03-21_lora_carriers_hugo/          # abandoned (53.7% complete, contaminated)
```

### Earlier Redwood runs (Llama-3-8B)

```
data/probe_results/
├── stego_detection_probe_results.json      # presence detection, all layers, per-method subsets
├── text_baseline_results.json              # char_features + tfidf baselines
├── game_scenarios_probe_results.json       # early 30-sample pilot (superseded)
├── decode_task_probe_results.json          # decode/recognition probes
├── benign_task_probe_results.json          # benign control comparison
└── decode_prompt_pilot_summary.json        # prompt format comparison
```

### GCM Head Attribution

```
data/Qwen3-32B_top10_stego_heads_suffix_marker.json   # 300 heads ranked by IE score
data/Qwen3-32B_top10_stego_heads_whitespace.json       # whitespace comparison (negative control — see audit)
steer/                                                  # GCM + steering code
```

### Figures (Llama-3-8B, from earlier Redwood runs)

```
data/figures/
├── fig1_layerwise_stego_detection.png      # presence detection AUC by layer
├── fig2_layerwise_decode_task.png          # decode probe AUC by layer
├── fig3_method_subset_comparison.png       # pooled vs hard-only subsets
├── fig3_same_carrier_control.png           # same-carrier control
├── fig4_per_method_comparison.png          # per-method AUC comparison
└── fig5_cross_method_transfer.png          # transfer matrix heatmap
```

### Datasets

```
data/
├── suffix_marker_game_scenarios.jsonl      # 2000 scenarios, suffix_marker only
├── whitespace_game_scenarios.jsonl         # 2000 scenarios, whitespace only
├── game_scenarios_v2.jsonl                 # 2000 scenarios, 5 methods (400 each)
├── prefixsuffix_inline_varied_game_scenarios.jsonl  # 5000 scenarios, 3 methods (~833 each)
├── combined_dataset.jsonl                  # Redwood dataset (presence detection)
├── decode_task.jsonl                       # decode/recognition task
└── redwood_synthetic_200per.jsonl          # 200 per method, Redwood format
```

---

## 2. Key Results (Quick Reference)

### Qwen3-32B behavioral results

| Dataset | Stego kw rate | Clean kw rate | Interpretation |
|---|---|---|---|
| suffix_marker | 49/1000 (4.9%) | 0/1000 (0.0%) | Perfect specificity, low compliance |
| whitespace | 1/1000 (0.1%) | 1/1000 (0.1%) | No signal |
| game_scenarios_v2 | 3/1000 (0.3%) | 4/1000 (0.4%) | No signal (inverted) |
| prefixsuffix_varied | 194/2500 (7.8%) | 0/2500 (0.0%) | Perfect specificity, 3 surface-marker methods |

**prefixsuffix_varied per-method behavioral breakdown:**

| Method | Keyword rate | n | Notes |
|---|---|---|---|
| suffix_marker_varied | 102/834 (12.2%) | 834 | Varied markers (`***`, `@@@`, etc.) -- higher than original suffix_marker (4.9%) |
| inline_keyword | 85/833 (10.2%) | 833 | Mid-text keyword insertion (e.g. `PRISM`, `EPOCH`) |
| prefix_marker_varied | 7/833 (0.8%) | 833 | Varied prefix markers -- very low compliance |

### Qwen3-32B stego-vs-clean probes

> **Caveat:** No text baselines have been run on game-scenario datasets. The suffix_marker probe likely detects the literal `+++` token (L0 AUC already 0.858). The whitespace probe likely detects formatting differences. See [audit](pipeline-audit-2026-03-22.md) Concerns 1, 3, 4.

| Dataset | Best Layer | AUC | AP | L0 AUC (floor) |
|---|---|---|---|---|
| suffix_marker | L49 | 0.929 | 0.929 | 0.858 |
| whitespace | L5 | 0.826 | 0.818 | 0.673 |
| game_scenarios_v2 (pooled) | L48 | 0.847 | 0.861 | -- |

**game_scenarios_v2 per-method breakdown (best layer L48):**

| Method | AUC | AP | F1 @ tuned | n |
|---|---|---|---|---|
| unicode | 0.986 | 0.967 | 0.876 | 1200 |
| prefix_binary | 0.973 | 0.920 | 0.784 | 1200 |
| acrostic | 0.952 | 0.864 | 0.770 | 1200 |
| whitespace | 0.884 | 0.756 | 0.681 | 1200 |
| **synonym** | **0.868** | **0.670** | **0.503** | **1200** |

| prefixsuffix_varied | -- | -- | -- | Probes not yet trained |

> **synonym AUC 0.868** is the cross-model replication target. On Llama-3-8B, synonym was 0.755 (text baseline 0.499, delta +0.256). The Qwen3-32B number is higher but still uncalibrated -- text baselines on game_scenarios_v2 have not been run. Until calibrated, this could reflect surface features in the synonym encoding.

### Qwen3-32B keyword probes

| Dataset | Best Layer | AUC | AP | n_positive |
|---|---|---|---|---|
| suffix_marker | L19 | 0.861 | 0.274 | 49/2000 |
| whitespace | L43 | 0.849 | 0.040 | 2/2000 (degenerate) |
| game_scenarios_v2 | -- | -- | -- | 7/2000 (degenerate, skip) |
| prefixsuffix_varied | -- | -- | -- | 194/5000 (probes not yet trained) |

### Llama-3-8B headline (synonym presence detection)

| Probe AUC | Text baseline AUC | Delta | Status |
|---|---|---|---|
| 0.755 (L3) | 0.499 (chance) | +0.256 | Strongest finding, survives all controls |

### GCM head attribution (Qwen3-32B)

| Dataset | Top head layers | Caveat |
|---|---|---|
| suffix_marker | L39-47 cluster | 44% overlap with whitespace negative control ([audit](pipeline-audit-2026-03-22.md) Concern 5) |
| whitespace | L40-47 cluster | Should have been null; overlap undermines specificity |

---

## 3. How to Operate the Pipeline

### Run inference (generation + activation extraction)

```bash
# Llama-3-8B (batch_size=16 safe)
python3 scripts/05_run_inference.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --scenarios data/suffix_marker_game_scenarios.jsonl \
  --generate --layers all --token_position last \
  --use_chat_template --batch_size 16 \
  --run_name suffix_marker

# Qwen3-32B (batch_size=2 for extraction, 16 for generation)
PYTHONUNBUFFERED=1 python3 scripts/05_run_inference.py \
  --model Qwen/Qwen3-32B \
  --scenarios data/game_scenarios_v2.jsonl \
  --generate --layers all --token_position last \
  --use_chat_template --batch_size 2 --generation_batch_size 16 \
  --run_name game_scenarios_v2
```

### Train probes

```bash
# Default: keyword_present labels (behavioral)
PYTHONUNBUFFERED=1 python3 scripts/06_train_probes.py \
  --run_dir data/Qwen3-32B/runs/2026-03-22_suffix_marker \
  --save_probes

# Ground-truth: is_stego labels (stego presence, balanced)
PYTHONUNBUFFERED=1 python3 scripts/06_train_probes.py \
  --run_dir data/Qwen3-32B/runs/2026-03-22_suffix_marker \
  --use_ground_truth_labels \
  --output_suffix _stego_vs_clean \
  --save_probes
```

### Run GCM head attribution

```bash
cd steer/
# Edit config.py first (MODEL_ID, DATA_PATH, NUM_SAMPLES)
python3 01_gcm_filter_heads_hf.py
```

### Run GCM steering intervention

```bash
cd steer/
# Uses top heads from GCM output
python3 02_gcm_steer.py
```

### Text baselines

```bash
python3 scripts/07_text_baselines.py
```

### Generate plots

```bash
python3 scripts/08_plot_results.py
```

---

## 4. Interpreting Probe Result JSON Files

Each `probe_results*.json` contains:

```
{
  "best_layer": 49,
  "n_examples": 2000,
  "n_positive": 1000,              # label balance
  "results_by_layer": {
    "0": {"mean_auc_roc": 0.858, "mean_average_precision": 0.820, ...},
    "1": {...},
    ...
  },
  "per_method_results": {           # if multiple stego methods in dataset
    "suffix_marker": {"mean_auc_roc": ..., "n_examples": ...},
    ...
  },
  "subset_results": {...},          # only for stego_detection task
  "transfer_results": {...}         # cross-method transfer matrix
}
```

**Known metadata bug:** `probe_results_stego_vs_clean.json` files incorrectly record `"target_label": "keyword_present"` when they actually used ground-truth `is_stego` labels. Verify by checking `n_positive` -- if it is 1000 (balanced), ground-truth labels were used regardless of what `target_label` says.

---

## 5. Experiment Priority Queue (Updated 2026-03-22 evening)

Ordered by **information value per unit time**. See [pipeline-audit-2026-03-22.md](pipeline-audit-2026-03-22.md) for rationale.

| # | Experiment | Resource | Time | Status | Decision value | What result changes |
|---|---|---|---|---|---|---|
| **1** | Text baselines on game_scenarios_v2 + suffix_marker + whitespace + prefixsuffix_varied | CPU | 30m | NOT STARTED. `07_text_baselines.py` needs adaptation for game-scenario format (`input_text`/`input_has_stego` instead of `response`/`is_stego`). | **CRITICAL:** calibrates ALL stego_vs_clean AUCs. synonym AUC 0.868 is uninterpretable without this. | If synonym text baseline < 0.6: cross-model replication confirmed. If > 0.8: surface artifact. |
| **2** | Train probes on prefixsuffix_varied (keyword + stego_vs_clean) | CPU | 1h | READY. Inference complete (5000 examples, 64 layers). | **LOW:** all 3 methods are surface markers. inline_keyword slightly interesting (mid-text placement). | Predictably high AUC. Interesting only if inline_keyword probe delta above text baseline is non-trivial. |
| **3** | Run "goodbye" keyword dataset (benign_keyword_suffix_marker, 400 examples) | GPU | 30m | NOT STARTED. Dataset at `data/benign_keyword_suffix_marker_game_scenarios.jsonl`. | **MEDIUM:** separates safety refusal from detection failure. | Adds texture to suffix_marker story. |
| **4** | GCM permutation null | GPU | 1h | NOT STARTED. Prerequisite for any GCM causal claim. | **MEDIUM:** 44% head overlap undermines specificity claims. | If permuted IE overlaps real scores: GCM heads not stego-specific. |
| ~~5~~ | ~~GCM steering intervention~~ | ~~GPU~~ | ~~1h~~ | **BLOCKED** on #4. | -- | -- |
| ~~6~~ | ~~Qwen3-14B anything~~ | ~~GPU~~ | ~~hours~~ | **DEPRIORITIZED.** Finish interpreting 32B results first. | -- | -- |

### Completed This Session

- [x] Qwen3-32B game_scenarios_v2 inference (2000 examples, 64 layers) -- behavioral null: 3/1000 stego, 4/1000 clean
- [x] Qwen3-32B game_v2 keyword probes -- degenerate (7 positives), per-method AUCs are noise
- [x] Qwen3-32B game_v2 stego_vs_clean probes -- **AUC 0.847 at L48; synonym 0.868, unicode 0.986, prefix_binary 0.973, acrostic 0.952, whitespace 0.884** (all need text baseline calibration)
- [x] Qwen3-32B suffix_marker stego_vs_clean probes -- AUC 0.929 at L49 (L0 floor: 0.858, needs text baseline)
- [x] Qwen3-32B whitespace stego_vs_clean probes -- AUC 0.826 at L5 (needs text baseline)
- [x] Qwen3-32B prefixsuffix_varied inference (5000 examples, 64 layers) -- 194/2500 stego kw, 0/2500 clean; suffix_marker_varied 12.2%, inline_keyword 10.2%, prefix_marker_varied 0.8%
- [x] Pipeline audit ([pipeline-audit-2026-03-22.md](pipeline-audit-2026-03-22.md)) -- 7 methodological concerns documented
