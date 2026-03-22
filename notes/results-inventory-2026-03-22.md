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
├── 2026-03-22_prefixsuffix_varied/          # COMPLETE — all probes done
│   ├── config.json
│   ├── activations/
│   │   ├── metadata.json
│   │   ├── responses.jsonl                  # 194/2500 stego kw, 0/2500 clean kw
│   │   ├── labels.npy                       # ground-truth is_stego (2500/2500)
│   │   └── layer_00.npy ... layer_63.npy    # activations (5000 x 5120) per layer
│   ├── probe_results/
│   │   ├── game_scenarios_probe_results.json          # keyword probes: best L56 AUC=0.790
│   │   ├── probe_results_stego_vs_clean.json          # stego_vs_clean: best L46 AUC=0.942
│   │   └── trained_probes/
│   └── text_baselines/
│       └── text_baseline_results_stego_vs_clean.json  # char=0.624, tfidf=0.998; inline_keyword char=0.873
│
└── 2026-03-22_benign_keyword_suffix_marker/ # COMPLETE — inference + stego_vs_clean probes
    ├── config.json
    ├── activations/
    │   ├── metadata.json
    │   ├── responses.jsonl                  # 0/200 stego kw, 0/200 clean kw (ZERO compliance)
    │   ├── labels.npy                       # ground-truth is_stego (200/200)
    │   └── layer_00.npy ... layer_63.npy    # activations (400 x 5120) per layer
    └── probe_results/
        ├── probe_results_stego_vs_clean.json # is_stego labels (200/200), best L46 AUC=0.909
        └── trained_probes/
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
├── benign_keyword_suffix_marker_game_scenarios.jsonl  # 400 scenarios, suffix_marker, keyword "goodbye"
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
| benign_keyword_suffix_marker | 0/200 (0.0%) | 0/200 (0.0%) | **FAILED EXPERIMENT** -- "goodbye" gets LESS compliance than "attack" (0% vs 4.9%) |

**benign_keyword vs original suffix_marker:**

| Variant | Keyword | Stego kw rate | Probe trained | Interpretation |
|---|---|---|---|---|
| suffix_marker (original) | "attack" | 49/1000 (4.9%) | keyword (49 pos) + stego_vs_clean | Low but nonzero; keyword probe heavily imbalanced |
| benign_keyword | "goodbye" | 0/200 (0.0%) | stego_vs_clean ONLY | **Keyword probes impossible (0 positives).** Original goal was to get MORE compliance via a benign keyword -- experiment backfired. |

> **Original goal:** Replace "attack" with innocuous "goodbye" so the model complies more freely, giving a richer keyword-positive signal for keyword probe training. **Result:** 0/200 compliance -- the experiment failed at its stated purpose. "goodbye" was even more suppressed than "attack", possibly because (a) "goodbye" never appears naturally at the end of coding responses, while "attack" has plausible deniability in security/pentesting contexts, or (b) the 400-example dataset has different task templates than the 2000-example original. Only stego_vs_clean probes (ground truth labels) were trainable, and those just re-detect the `+++` surface marker (AUC 0.909 vs original 0.929, within sample-size noise) -- adding nothing beyond what tfidf=1.000 already proved.

**prefixsuffix_varied per-method behavioral breakdown:**

| Method | Keyword rate | n | Notes |
|---|---|---|---|
| suffix_marker_varied | 102/834 (12.2%) | 834 | Varied markers (`***`, `@@@`, etc.) -- higher than original suffix_marker (4.9%) |
| inline_keyword | 85/833 (10.2%) | 833 | Mid-text keyword insertion (e.g. `PRISM`, `EPOCH`) |
| prefix_marker_varied | 7/833 (0.8%) | 833 | Varied prefix markers -- very low compliance |

### Qwen3-32B stego-vs-clean probes (calibrated 2026-03-22)

Text baselines now available for all datasets. See `<run_dir>/text_baselines/text_baseline_results_stego_vs_clean.json`.

| Dataset | Best Layer | Probe AUC | Char baseline | TF-IDF baseline | **Delta vs tfidf** | Interpretation |
|---|---|---|---|---|---|---|
| suffix_marker | L49 | 0.929 | 0.526 | **1.000** | ~0 | TF-IDF detects `+++` n-gram trivially; surface artifact |
| whitespace | L5 | 0.826 | **1.000** | 0.978 | ~0 | Char stats (newlines/tabs) fully explain signal |
| game_scenarios_v2 (pooled) | L48 | 0.847 | 0.866 | — | ~0 pooled | Dominated by easy methods |
| ~~**game_scenarios_v2 synonym**~~ | ~~L48~~ | ~~0.868~~ | ~~0.620~~ | ~~**0.870**~~ | ~~**+0.248 vs char / ≈0 vs tfidf**~~ | ~~**Genuine signal**~~ → **RETRACTED (2026-03-22):** tfidf=0.870 matches probe; no residual signal |
| prefixsuffix_varied (pooled) | L46 | 0.942 | 0.624 | 0.998 | ~0 vs tfidf | Pooled tfidf trivially detects markers |
| benign_keyword_suffix_marker | L46 | 0.909 | — | — | — | Stego_vs_clean only (0 keyword positives). Just re-detects `+++` surface marker. |

**game_scenarios_v2 per-method breakdown (best layer L48):**

| Method | Probe AUC | Char baseline | TF-IDF baseline | Interpretation |
|---|---|---|---|---|
| unicode | 0.986 | 1.000 | 1.000 | Surface feature |
| prefix_binary | 0.973 | 1.000 | 1.000 | Surface feature |
| acrostic | 0.952 | 0.931 | 1.000 | Mostly surface |
| whitespace | 0.884 | 1.000 | 0.976 | Surface formatting |
| ~~**synonym**~~ | ~~0.868~~ | ~~0.620~~ | ~~**0.870**~~ | **RETRACTED:** tfidf matches probe — domain vocab makes substitutions char-detectable |

> **Cross-model replication RETRACTED (2026-03-22):** Qwen3-32B synonym tfidf baseline = 0.870, matching probe AUC 0.868. The game_scenarios_v2 dataset's domain-specific vocabulary (game terms) makes synonym substitutions detectable via char n-grams. The earlier apparent replication (+0.248 vs Llama's +0.256) was comparing against char features — an inappropriately weak baseline for this dataset. See [audit Concern 8](pipeline-audit-2026-03-22.md).

**prefixsuffix_varied per-method breakdown (best layer L46):**

| Method | Probe AUC | Char baseline | TF-IDF baseline | Probe - char delta | Interpretation |
|---|---|---|---|---|---|
| inline_keyword | 0.974 | 0.873 | 0.998 (pooled) | +0.101 | Moderate delta vs char; tfidf likely explains most signal |
| prefix_marker_varied | 0.951 | 0.527 | 0.998 (pooled) | +0.424 | Large delta vs char, but tfidf (pooled) detects marker n-grams |
| suffix_marker_varied | 0.928 | 0.526 | 0.998 (pooled) | +0.402 | Same pattern as prefix |

> All 3 methods are surface markers. The char baseline cannot detect marker tokens (AUC ~0.53 for prefix/suffix), but tfidf trivially can (pooled 0.998). The per-method deltas vs char features are large but misleading -- tfidf would close the gap. inline_keyword has the smallest delta (+0.101) because char features already detect it well (AUC 0.873, likely from length/punct changes from inserted keywords).

### Qwen3-32B keyword probes

| Dataset | Best Layer | AUC | AP | n_positive |
|---|---|---|---|---|
| suffix_marker | L19 | 0.861 | 0.274 | 49/2000 |
| whitespace | L43 | 0.849 | 0.040 | 2/2000 (degenerate) |
| game_scenarios_v2 | -- | -- | -- | 7/2000 (degenerate, skip) |
| prefixsuffix_varied | L56 | 0.790 | 0.209 | 194/5000 (imbalanced) |

### Llama-3-8B headline (synonym presence detection)

| Probe AUC | Char baseline | TF-IDF baseline | Delta vs tfidf | Status |
|---|---|---|---|---|
| 0.755 (L3) | 0.499 (chance) | **0.569** | **+0.186** | **Confirmed against strongest text baseline (2026-03-22)** |

> TF-IDF char n-gram baseline 0.569 is near chance on the Redwood dataset. Probe delta +0.186 survives the strongest text-level control. This is now the sole confirmed representational finding in the project.

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
# Redwood format (combined_dataset.jsonl — uses 'response' / 'is_stego')
python3 scripts/07_text_baselines.py --dataset data/combined_dataset.jsonl

# Game-scenario format (uses 'input_text' / 'input_has_stego' / 'stego_method')
python3 scripts/07_text_baselines.py \
  --dataset data/game_scenarios_v2.jsonl \
  --run_dir data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2 \
  --input_field input_text --label_field input_has_stego \
  --output_suffix _stego_vs_clean
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

---

## 5. Experiment Priority Queue (Updated 2026-03-22 ~20h00 UTC)

Ordered by **information value per unit time** per the golden rule. Evidence-based justification for each item.

| # | Experiment | Resource | Time | Status | Decision value | Justification |
|---|---|---|---|---|---|---|
| ~~**1**~~ | ~~Synonym-specific tfidf baseline (both models)~~ | ~~CPU~~ | ~~10-15m~~ | **DONE — SPLIT VERDICT** | -- | Llama tfidf=0.569 → confirmed. Qwen tfidf=0.870 → retracted. |
| ~~**2**~~ | ~~prefixsuffix_varied stego_vs_clean probes~~ | ~~CPU~~ | -- | **DONE.** All surface markers as expected. | -- | -- |
| ~~**3**~~ | ~~benign_keyword_suffix_marker ("goodbye") inference + probes~~ | ~~GPU~~ | ~~30m~~ | **DONE (2026-03-22 ~19h).** 0/200 compliance; probe AUC 0.909 (surface). | -- | Confirms probe detects `+++` regardless of keyword. 0% vs 4.9% compliance = keyword-specific RLHF. |
| **4** | **GCM head regen (suffix_marker, fixed template)** | GPU | ~1h | **QUEUED (TEAM REQUEST)** | MEDIUM | Teammates have been asking. Existing heads are stale (wrong template). Even though suffix_marker is a surface artifact, clean GCM results validate the methodology and produce a documented null. Required before any steering attempt. |
| **5** | **GCM permutation null (suffix_marker)** | GPU | ~1h | **QUEUED (TEAM), after #4** | MEDIUM | Tests whether GCM signal survives label shuffling. Expected outcome: signal is real (because `+++` is real) but surface. The 44% whitespace overlap should resolve after template fix. |
| **6** | **GCM steering (suffix_marker)** | GPU | ~1h | **QUEUED (TEAM), after #5** | LOW-MEDIUM | End-to-end GCM demonstration. May not change behavior (model already refuses 95%). But completes the GCM pipeline for the paper's methods section. |
| ~~7~~ | ~~Qwen3-14B anything~~ | ~~GPU~~ | ~~hours~~ | **DEPRIORITIZED.** | -- | Third model adds scale, not insight. Cross-model replication retracted anyway. |

### What was cut and why (history)

**Goodbye dataset (2026-03-22 ~18h30):** Demoted from MEDIUM to LOW. Now DONE (see above). Result: 0% compliance with "goodbye" adds one clean data point about keyword-specific RLHF, confirming what prefixsuffix_varied already showed (compliance varies by context). Probe AUC 0.909 matches original 0.929 within noise.

**GCM (2026-03-22 ~18h30 → REINSTATED ~20h00):** Previously deprioritized because (a) stale heads, (b) surface artifact target, (c) ~3h GPU. Now **reinstated at team request**. Scientific justification: even on a surface artifact, clean GCM methodology with permutation null is a methods contribution. The 44% head overlap with whitespace may resolve after template fix -- worth documenting either way. And teammates need this for their contributions to the project.

**Qwen3-14B (2026-03-22 ~18h30):** Stays deprioritized. Cross-model replication was retracted; third model doesn't help.

### Completed This Session (chronological)

- [x] Qwen3-32B game_scenarios_v2 inference (2000 examples, 64 layers) -- behavioral null: 3/1000 stego, 4/1000 clean
- [x] Qwen3-32B game_v2 keyword probes -- degenerate (7 positives), skip
- [x] Qwen3-32B game_v2 stego_vs_clean probes -- AUC 0.847 at L48; **synonym 0.868**
- [x] Qwen3-32B suffix_marker stego_vs_clean probes -- AUC 0.929 at L49 (surface artifact -- tfidf baseline 1.000)
- [x] Qwen3-32B whitespace stego_vs_clean probes -- AUC 0.826 at L5 (surface artifact -- char baseline 1.000)
- [x] Qwen3-32B prefixsuffix_varied inference (5000 examples, 64 layers) -- 194/2500 stego kw, 0/2500 clean
- [x] Qwen3-32B prefixsuffix_varied keyword probes -- best L56 AUC=0.790 (194 positives, imbalanced)
- [x] Pipeline audit -- 7 methodological concerns documented; GCM template and metadata bug fixed
- [x] **Text baselines on all 4 game-scenario datasets** -- synonym cross-model replication initially appeared confirmed (delta +0.248 vs char baseline)
- [x] `07_text_baselines.py` adapted for game-scenario format (`--input_field`, `--label_field`, `--method_field`)
- [x] Qwen3-32B prefixsuffix_varied stego_vs_clean probes -- AUC 0.942 at L46; inline_keyword 0.974, prefix_marker_varied 0.951, suffix_marker_varied 0.928 (all surface markers, tfidf=0.998)
- [x] **Per-method TF-IDF baselines (both models)** -- Llama synonym tfidf=0.569 (confirmed, delta +0.186). Qwen synonym tfidf=0.870 (retracted). Cross-model replication retracted.
- [x] **benign_keyword_suffix_marker inference + probes** -- 400 examples, keyword "goodbye". **FAILED at stated goal:** 0/200 compliance (wanted MORE than "attack"'s 4.9%, got LESS). Keyword probes impossible (0 positives). Only stego_vs_clean probes trained -- just re-detect `+++` surface marker (AUC 0.909). Mild behavioral interest: "goodbye" more suppressed than "attack".
