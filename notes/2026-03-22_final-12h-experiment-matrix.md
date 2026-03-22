# Final 12h Experiment Matrix — 2026-03-22 ~22h UTC

**Context:** 12 hours remain. One confirmed finding survives all controls. Everything else is surface artifact or documented negative. This is a 1-page operational playbook.

---

## 1. Ground Truth on Disk

| Asset | Model | Status | Path |
|---|---|---|---|
| Synonym probe AUC 0.755 (L3) | Llama-3-8B | **Result confirmed, layer .npy files DELETED** | `data/activations/stego_detection/` — metadata only, 0 layers |
| Synonym tfidf baseline 0.569 | Llama-3-8B | On disk | `data/probe_results/text_baseline_results.json` |
| Saved probe weights (Llama) | Llama-3-8B | **DO NOT EXIST** | No `trained_probes/` dir anywhere under Llama paths |
| Combined dataset (Redwood) | — | On disk, 2000 rows | `data/combined_dataset.jsonl` |
| Qwen activations (5 runs) | Qwen3-32B | 320 .npy files on disk | `data/Qwen3-32B/runs/` (64 layers x 5 runs) |
| Qwen saved probes | Qwen3-32B | On disk, all 5 runs | `data/Qwen3-32B/runs/*/probe_results/trained_probes/` |
| GCM pipeline results | Qwen3-32B | Complete, NEGATIVE | `data/Qwen3-32B/gcm_results_suffix_marker_regen.json` |
| GPU | GH200 97GB | **FREE** (0 MiB used) | — |
| TransformerLens / nnsight | — | **NOT INSTALLED** | Only torch 2.7, transformers 4.57 available |

**Critical blocker:** Every causal experiment on the confirmed Llama synonym finding requires re-extracting activations first. The .npy files were cleaned from disk. No saved probe weights exist.

---

## 2. The One Confirmed Finding

| Metric | Value |
|---|---|
| Llama-3-8B synonym probe (L3) | AUC 0.755 |
| Char features baseline | AUC 0.499 (chance) |
| TF-IDF char n-gram baseline | AUC 0.569 |
| **Delta vs strongest baseline** | **+0.186** |
| Transfer to other methods | AUC 0.72-0.93 |
| Dataset | `combined_dataset.jsonl` (Redwood, 1000/1000 balanced) |

Everything in the 12h plan strengthens or stress-tests this one result. Everything else is a controlled negative worth documenting.

---

## 3. Experiment Matrix (ROI order)

### P0 — PREREQUISITE: Re-extract Llama activations + save probes

| | |
|---|---|
| **Time** | ~45min GPU |
| **Blocks** | P1, P2, P3 |
| **Command** | See below |

```bash
# Step 1: Re-extract all 32 layers
PYTHONUNBUFFERED=1 python3 scripts/05_run_inference.py \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --scenarios data/combined_dataset.jsonl \
  --layers all --token_position last \
  --use_chat_template --batch_size 16 \
  --run_name stego_detection_reextract

# Step 2: Train + save probes with ground-truth labels
PYTHONUNBUFFERED=1 python3 scripts/06_train_probes.py \
  --run_dir data/Meta-Llama-3-8B-Instruct/runs/<datestamp>_stego_detection_reextract \
  --use_ground_truth_labels --output_suffix _stego_vs_clean \
  --save_probes
```

**Decision value:** Zero on its own — pure prerequisite. But without it, nothing below can run.

**Note:** `05_run_inference.py` may need `--generate` skipped if we only want activations and the dataset is a detection-style (no generation needed). Check script flags. The combined_dataset uses `response` field, not game-scenario format — verify the script handles this correctly or use the detection path.

---

### P1 — Layerwise residual-stream ablation (synonym subset)

| | |
|---|---|
| **Time** | ~1h code + ~2h GPU |
| **Depends on** | P0 |
| **Decision value** | HIGHEST — converts correlational probe into causal claim |

**What:** For each layer L (0-31), run the model on synonym-only stego examples but replace the residual stream at layer L with the cached residual stream from the paired clean example. Measure how much the L3 probe score drops.

**Implementation:** HF-native `register_forward_hook` on `model.model.layers[L]`. No library install needed. Script: new `scripts/13_layerwise_ablation.py`.

**Metric:** Probe score drop at L3 (and at all downstream layers) when layer L is patched.

**Decision rule:**
- If 1-3 layers account for >80% of total probe score drop → **mechanistic headline**: "synonym steganography representation is written at layers X-Y"
- If signal drops ~uniformly across all 32 layers → representation is distributed; skip P3, pivot write-up

**Key detail:** Must ablate on the **synonym-only subset** (200 stego + matched clean from carrier_group_id). The pooled signal is dominated by easy methods and would trivially localize to the embedding layer.

---

### P2 — Probe direction extraction + cross-layer cosine similarity

| | |
|---|---|
| **Time** | ~30min CPU |
| **Depends on** | P0 (saved probes) |
| **Decision value** | MEDIUM — tests whether a consistent "stego direction" exists |

**What:** Load all 32 saved LogisticRegression probes. Extract `.coef_` vectors (4096-d each). Compute 32x32 cosine similarity matrix. Plot heatmap.

**Expected outcomes:**
- High cosine similarity across layers → a consistent linear direction exists (analogous to refusal direction literature: Arditi et al. 2024, "Programming Refusal" ICLR 2025)
- L3 is an outlier with low cosine to other layers → signal is layer-specific, not a global direction
- Block structure → signal is written at one stage and persists (consistent with P1 ablation if localized)

**Can run in parallel with P1 GPU work.** Script: new `scripts/14_probe_direction_analysis.py`.

---

### P3 — Probe-direction steering on Llama synonym

| | |
|---|---|
| **Time** | ~1h GPU |
| **Depends on** | P0, P1 results (only if P1 shows localization) |
| **Decision value** | HIGH if P1 is localized; SKIP if P1 is diffuse |

**What:** Use L3 probe `.coef_` as a steering vector. Add `alpha * direction` to the residual stream at layer 3 (or at the localized layer from P1) during generation. Measure:
1. Probe score shift at downstream layers (continuous, low variance)
2. Log-prob shift for synonym-indicator tokens
3. KL divergence (sanity check)

**Implementation:** Same HF hooks pattern as the existing `steer/04_gcm_steer_hf.py` but with probe-derived direction instead of abs()-contaminated GCM vectors.

**Decision rule:**
- Positive probe score shift at downstream layers → causal confirmation
- Null → direction is correlational only (still a result worth reporting)

---

### P4 — Signed mean-difference head attribution (Qwen, CPU only)

| | |
|---|---|
| **Time** | ~1h CPU |
| **Depends on** | Nothing (uses existing .npy files) |
| **Decision value** | MEDIUM — methodological contribution documenting why abs() GCM failed |

**What:** Load Qwen3-32B suffix_marker activations (2000 x 5120 x 64 layers). Compute **signed** mean activation difference per head between stego and clean conditions. Rank heads by `|mean_stego - mean_clean|` (note: take abs only AFTER averaging, not before). Compare rankings to the abs()-IE rankings from the failed GCM pipeline.

**Expected outcome:** Dramatically different head rankings, confirming the `abs()` accumulation was the root cause of the permutation null failure. If rankings also cluster in L40-47, then those layers genuinely process stego input differently but the GCM method was the wrong measurement tool.

**Can run in parallel with P0/P1 GPU work.** Script: new `scripts/15_mean_diff_heads.py`.

---

### P5 — Write-up: consolidated narrative + key figures

| | |
|---|---|
| **Time** | ~3h |
| **Depends on** | P1, P2 results |
| **Decision value** | N/A — deliverable |

**Story arc:**
1. **Motivation:** LLM steganography as safety concern — cite "Early Signs of Steganographic Capabilities" (Nishimura-Gasparian et al., 2507.02737, 2025), "Steganographic Potentials" (ICLR 2025 workshop, 2505.03439), "Hide and Seek in Embedding Space" (2601.22818, Jan 2026)
2. **Methods:** Activation probes + text baselines + causal ablation (if P1 works)
3. **Key result:** Synonym stego produces non-trivial internal representation (delta +0.186 vs tfidf), mechanistically localized to layers X-Y (if P1 works)
4. **Controlled negatives:** 4 surface methods trivially detected by tfidf; Qwen cross-model replication retracted; GCM pipeline negative result with identified root cause
5. **RLHF behavioral findings:** keyword/marker sensitivity pattern, probe-behavior orthogonality

**Key figures to generate:**
- Layerwise AUC with text-baseline band (gray band at tfidf=0.569)
- Ablation profile (probe score drop per layer) — from P1
- Cosine similarity heatmap — from P2
- Per-method text-baseline comparison bar chart (existing data)

---

## 4. CUT List (do NOT run)

| Experiment | Why cut |
|---|---|
| Qwen3-14B anything | Third model adds scale, not insight. Cross-model replication already retracted. |
| New dataset generation | No time for full pipeline cycle (generate → inference → probes → baselines). |
| GCM with fixed signed-IE on Qwen | Even if fixed, the underlying Qwen signals are surface artifacts (tfidf=1.000). A correct method on an empty target teaches nothing. |
| Llama game_scenarios_v2 re-extraction | Responses exist but no layers. Synonym signal was confirmed on Redwood, not game scenarios. |
| Qwen synonym re-run on diverse carriers | Would need new dataset + full pipeline. Time > value. |
| Decode/recognition probes | Benign control matched (0.719 vs 0.721). Dead end confirmed. |

---

## 5. Decision Tree

```
P0 (re-extract Llama activations)
│
├─ P2 (CPU: probe direction cosine)  ← runs in parallel
├─ P4 (CPU: signed mean-diff heads)  ← runs in parallel
│
└─ P1 (GPU: layerwise ablation)
   │
   ├─ LOCALIZED (1-3 layers > 80% drop)
   │  ├─ P3 (steering with probe direction)
   │  └─ P5 write-up: mechanistic story
   │
   └─ DIFFUSE (uniform drop)
      ├─ SKIP P3
      └─ P5 write-up: "representation exists, circuit distributed,
         here are the controls and negatives"
```

---

## 6. Timeline (12h from ~22h UTC)

| Hour | GPU | CPU | Notes |
|---|---|---|---|
| 0-1 | **P0:** Re-extract Llama activations | Review/prep P1 ablation script | Model load ~10min, extraction ~30min |
| 1-1.5 | **P0 cont'd:** Train + save probes | **P4:** Signed mean-diff heads (Qwen) | P4 independent, uses existing .npy |
| 1.5-2 | (probes finishing) | **P2:** Probe direction cosine analysis | P2 needs saved probes from P0 |
| 2-4 | **P1:** Layerwise ablation | Start P5 write-up (intro, methods, negatives) | P1 is the crux experiment |
| 4-5 | **P1 analysis** | Continue P5, integrate P1/P2 results | Decision point: localized or diffuse? |
| 5-6 | **P3** (if P1 localized) OR idle | Figures generation | P3 only if warranted |
| 6-9 | Idle / ad-hoc reruns | **P5:** Complete write-up + figures | Writing is the bottleneck here |
| 9-10 | — | Final review, commit, push | — |

**Total GPU time:** ~5h (P0: 45min, P1: 2h, P3: 1h if warranted, slack: 1h)
**Total CPU time:** ~4h parallel (P2: 30min, P4: 1h, P5 writing: 3h+ overlapping)

---

*Written 2026-03-22 ~22h UTC. Anchored in actual disk state — every path verified, every missing file noted. Apply the golden rule: if the result won't change what we do next, don't run it.*
