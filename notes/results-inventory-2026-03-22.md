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
├── 2026-03-21_lora_carriers_hugo/          # abandoned (53.7% complete, contaminated)
└── 2026-03-22_stego_detection_reextract/   # COMPLETE — full causal chain (P0-P3)
    ├── config.json                         # generate=false, use_chat_template=false
    ├── activations/
    │   ├── metadata.json                   # 2000 examples, 5 methods (200 each) + 1000 clean
    │   ├── labels.npy                      # ground-truth is_stego (1000/1000)
    │   └── layer_00.npy ... layer_31.npy   # activations (2000 x 4096) per layer
    ├── probe_results/
    │   ├── probe_results_stego_vs_clean.json  # Best L9, AUC 0.761 (pooled), synonym 0.574
    │   └── trained_probes/layer_00..31/       # saved probes (5 seeds each)
    └── probe_results_stego_vs_clean/
        ├── p1_ablation_results.json        # Knockout: no localization, total_drop=0.0
        ├── p2_probe_direction_analysis.json # Cosine: mean 0.302, L3 direction local only
        └── p3_probe_steer_results.json     # Steering: causal_confirmed=true BUT tautological (KL=1.853)
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

### GCM Head Attribution & Steering (COMPLETE — negative result, both models)

```
# Qwen3-32B (suffix_marker, 2026-03-22 ~21h40)
data/Qwen3-32B_top10_stego_heads_suffix_marker.json       # REGENERATED (fixed template) — 307 heads ranked by IE
data/Qwen3-32B_top10_stego_heads_whitespace.json           # REGENERATED (fixed template) — whitespace negative control
data/Qwen3-32B_top10_stego_heads_suffix_marker_permuted.json       # Permutation null (seed=42, 50% swap)
data/Qwen3-32B_top10_stego_heads_suffix_marker_permuted_config.json # Permutation null config
data/Qwen3-32B_steering_summary_suffix_marker.json         # Steering results (complete null)
data/Qwen3-32B_stego_steering_vectors_suffix_marker.pt     # Computed steering vectors (10 heads)
data/Qwen3-32B_top10_stego_steering_results_suffix_marker.csv  # Pre/post ASR comparison
data/Qwen3-32B_top10_stego_steering_kl_suffix_marker.csv   # KL divergence
data/Qwen3-32B/gcm_results_suffix_marker_regen.json        # Pipeline summary (ratio=0.92, overlap=54%)

# Llama-3-8B (synonym, 2026-03-22 ~22h10)
data/Meta-Llama-3-8B-Instruct_top10_stego_heads_synonym_permuted.json       # Permutation null (seed=42, 50% swap)
data/Meta-Llama-3-8B-Instruct_top10_stego_heads_synonym_permuted_config.json # Config
# NOTE: Real (non-permuted) heads LOST — computation completed but crashed on file save (relative path error)

# Scripts
steer/01_gcm_filter_heads_hf.py                            # Head attribution (HF native, updated for Llama compat)
steer/03_gcm_permutation_null.py                           # Permutation null test
steer/04_gcm_steer_hf.py                                   # HF-native steering (replaces transformer_lens)
steer/run_gcm_pipeline.sh                                  # Full 3-step pipeline orchestration
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
├── synonym_game_scenarios.jsonl            # 400 scenarios, synonym only (for Llama GCM)
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

### ~~Llama-3-8B headline (synonym presence detection)~~ RETRACTED (2026-03-22 ~22h31)

| Run | Probe AUC | Best Layer | TF-IDF baseline | Delta vs tfidf | Status |
|---|---|---|---|---|---|
| Old extraction (deleted) | 0.754 | L3 | 0.569 | +0.186 | ~~Confirmed~~ → **RETRACTED** |
| **Re-extraction (current)** | **0.574** | **L9** | **0.569** | **+0.005** | **Fails replication** |

> **RETRACTED (2026-03-22 ~22h31):** Re-extraction of Llama-3-8B activations on the same `combined_dataset.jsonl` with `use_chat_template=false` gives synonym AUC 0.574 vs tfidf 0.569 — delta +0.005, effectively zero. All methods degraded vs old extraction (acrostic: 1.000→0.907, whitespace: 1.000→0.710), indicating the old result was pipeline-configuration-dependent. P1 ablation confirms: L3 probe baseline 0.436 (below chance on synonym), zero total drop. **The project now has zero confirmed positive interpretability findings.** What survives: behavioral specificity findings, RLHF suppression patterns, and the experimental framework itself.

### GCM head attribution — NEGATIVE RESULT (both models)

Full pipeline completed on Qwen (regen + permutation null + steering). Permutation null extended to Llama-3-8B on synonym dataset.

**Qwen3-32B (suffix_marker, 2026-03-22 ~21h40):**

| Dataset | Top-10 mean IE | Layer cluster | Overlap with suffix_marker |
|---|---|---|---|
| suffix_marker (real) | 0.0743 | L39-47 | — |
| suffix_marker (permuted) | 0.0804 | L40-48 | 74% (37/50) |
| whitespace | 0.0463 | L40-48 | 54% (27/50) |

Permutation null: FAILS. Ratio = 0.92x (real < permuted). Steering: complete null (0% ASR, KL=0.004).

**Llama-3-8B (synonym, 2026-03-22 ~22h10):**

| Dataset | Top-10 mean IE | Layer cluster | % top-50 in last 3 layers |
|---|---|---|---|
| synonym (permuted) | 0.0333 | L29-31 | 52% (26/50) |
| *Real heads: LOST (path error)* | — | — | — |

L31 dominates: 6/10 top-10 heads, 10/50 top-50 heads. Cross-model pattern identical: both models show 52% of top-50 permuted heads in late-layer peak zone.

**Conclusion:** GCM `abs()` IE artifact is **architecture-general** — confirmed across both models (8B and 32B), both datasets (synonym and suffix_marker), both architectures (Llama and Qwen). See [audit Section 1.11 and Concern 5](pipeline-audit-2026-03-22.md).

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

### Run GCM pipeline (full chain — NOTE: produces negative result with current `abs()` method)

```bash
# Full pipeline (regen + permutation null + steering):
PYTHONUNBUFFERED=1 bash steer/run_gcm_pipeline.sh 2>&1 | tee /tmp/gcm_pipeline.log

# Individual steps:
cd steer/
python3 01_gcm_filter_heads_hf.py                          # Head attribution
python3 03_gcm_permutation_null.py --dataset_tag suffix_marker  # Permutation null
python3 04_gcm_steer_hf.py --dataset_tag suffix_marker          # Steering (HF native)
```

> **WARNING (2026-03-22):** GCM pipeline produces negative result. Permutation null fails (ratio=0.92x). Root cause: `abs()` in IE accumulation. To fix, need signed IE accumulation or different attribution method.

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
| ~~**4**~~ | ~~GCM head regen (suffix_marker, fixed template)~~ | ~~GPU~~ | ~~1h~~ | **DONE — NEGATIVE (2026-03-22 ~21h40)** | -- | Overlap with whitespace INCREASED to 54% (was 44%). Template was not the cause. |
| ~~**5**~~ | ~~GCM permutation null (suffix_marker)~~ | ~~GPU~~ | ~~1h~~ | **DONE — FAILS (2026-03-22 ~21h40)** | -- | Ratio=0.92x (real < permuted). 9/10 top-10 heads identical. `abs()` aggregation is root cause. |
| ~~**6**~~ | ~~GCM steering (suffix_marker)~~ | ~~GPU~~ | ~~1h~~ | **DONE — NULL (2026-03-22 ~21h40)** | -- | 0% ASR pre/post, KL=0.004. Steering on invalid heads has no effect. |
| ~~7~~ | ~~Qwen3-14B anything~~ | ~~GPU~~ | ~~hours~~ | **DEPRIORITIZED.** | -- | Third model adds scale, not insight. Cross-model replication retracted anyway. |
| ~~**NEW**~~ | ~~GCM permutation null on Llama synonym~~ | GPU | ~10m | **DONE — SAME PATTERN (2026-03-22 ~22h10)** | -- | Permuted null: L31 dominates (6/10 top-10), 52% in last 3 layers = identical to Qwen. `abs()` artifact is architecture-general. Real heads file lost (path error). |
| ~~**NEW**~~ | ~~Causal localization on Llama synonym signal~~ | GPU | ~2-4h | **DONE — NO SIGNAL (2026-03-22 ~22h31)** | -- | Re-extraction synonym AUC=0.574, delta≈0 vs tfidf. P1 ablation: total_drop=0.0. P2: direction local (cos<0.2 by L9). P3: tautological (KL=1.853). Nothing to localize. |

### What was cut and why (history)

**Goodbye dataset (2026-03-22 ~18h30):** Demoted from MEDIUM to LOW. Now DONE (see above). Result: 0% compliance with "goodbye" adds one clean data point about keyword-specific RLHF, confirming what prefixsuffix_varied already showed (compliance varies by context). Probe AUC 0.909 matches original 0.929 within noise.

**GCM (2026-03-22 ~18h30 → REINSTATED ~20h00 → Qwen COMPLETED ~21h40 → Llama cross-model ~22h10 — NEGATIVE on both):** Previously deprioritized, then reinstated at team request. **Full pipeline complete on Qwen (negative). Permuted null extended to Llama synonym (same pattern).** Permutation null fails on both models; identical 52% late-layer concentration; steering null. Root cause identified: `abs()` in IE accumulation — architecture-general artifact. GCM as implemented is not suitable for this application. Documented as cross-model methodological negative result.

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
- [x] **GCM full pipeline (regen + permutation null + steering)** -- completed 2026-03-22 ~21h40. **NEGATIVE RESULT:** Permutation null fails (ratio=0.92x, real < permuted). 9/10 top-10 heads identical under shuffled labels. Steering complete null (0% ASR pre/post, KL=0.004). Root cause: `abs()` accumulation in IE computation finds high-variance heads, not stego-specific ones. Overlap with whitespace increased to 54% after template fix. New scripts: `steer/03_gcm_permutation_null.py`, `steer/04_gcm_steer_hf.py`, `steer/run_gcm_pipeline.sh`.
- [x] **Llama-3-8B synonym GCM permutation null** -- completed 2026-03-22 ~22h10. Cross-model confirmation of `abs()` artifact. L31 dominates permuted heads (6/10 top-10, 10/50 top-50). Last 3 layers hold 52% of top-50 -- identical to Qwen's 52% in L42-45. Real heads computation completed but file LOST (relative path error in `01_gcm_filter_heads_hf.py`). Code updated: `enable_thinking` conditional, `head_dim` fallback, multi-layer single-pass hooking (14x speedup). New dataset: `synonym_game_scenarios.jsonl` (400 entries).
- [x] **Final 12h experiment matrix** -- written to `notes/2026-03-22_final-12h-experiment-matrix.md`. Prioritized P0-P5 plan anchored in disk state: re-extract Llama activations → layerwise ablation → probe direction analysis → steering → write-up.
- [x] **P4 Mean-diff by layer (Qwen3-32B suffix_marker, CPU)** -- completed 2026-03-22 ~22h20. Monotonic L2 increase L0→L62 (surface feature signature). 0/5 top-5 overlap with GCM. GCM real and permuted top-5 identical. Spearman 0.413 (ambiguous). Script: `15_mean_diff_heads.py`.
- [x] **Llama-3-8B full overnight chain (P0→P1→P2→P3)** -- completed 2026-03-22 22:31 UTC. **DEVASTATING RESULT: Synonym finding RETRACTED.** Re-extraction synonym AUC 0.574 (was 0.755), delta +0.005 vs tfidf (was +0.186). P1 ablation: total positive drop = 0.0. P2 cosine: L3 direction local, orthogonal by mid-layers. P3 steering: tautological (KL=1.853). All non-synonym methods also degraded (pipeline config difference: `use_chat_template=false`). Project now has zero confirmed positive interpretability findings. Scripts: `13_layerwise_ablation.py`, `14_probe_direction_analysis.py`, `16_probe_steer.py`, `run_overnight_chain.sh`.
