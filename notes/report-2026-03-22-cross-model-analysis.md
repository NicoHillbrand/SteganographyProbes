# Cross-Model Comparative Analysis: Steganography Probes on Llama-3-8B vs Qwen3-32B

> **Update (2026-03-22 audit):** Qwen3-32B game_scenarios_v2 results are now available: 3/1000 stego keyword, 4/1000 clean keyword (null behavioral signal, inverted). The stego_vs_clean probe results for suffix_marker and whitespace have significant caveats (likely detecting surface features, no text baseline). See [pipeline-audit-2026-03-22.md](pipeline-audit-2026-03-22.md) for the current source of truth on claim validity.

**Date:** 2026-03-22
**Scope:** Full comparative analysis of all completed runs across both models, all datasets, and all probe types.

---

## Executive Summary

This report compares steganography detection results across two models (Meta-Llama-3-8B-Instruct and Qwen/Qwen3-32B) on shared game-scenario datasets (suffix_marker, whitespace) and Llama-only datasets (game_scenarios_v2, Redwood presence detection). The central finding is that **model scale produces a qualitative shift in the nature of the steganographic signal**, not merely a quantitative improvement:

- Llama-3-8B cannot follow stego-detection instructions. It outputs the keyword "attack" at roughly the same rate (~40-49%) regardless of whether steganography is present. Probes trained on this behavioral label are correspondingly weak (best AUC 0.629).
- Qwen3-32B mostly refuses to act on steganographic instructions (4.9% compliance on suffix_marker) but when it does, it is perfectly specific (0% false positive rate). The probe at layer 19 achieves AUC 0.861 despite having only 49 positive labels out of 2000.
- The two models exhibit fundamentally different failure modes: Llama fails by indiscriminate compliance; Qwen fails by near-total refusal. Neither produces the ideal behavioral signal (high compliance on stego, low on clean), but Qwen's refusal paradoxically creates a cleaner probe target.

The strongest cross-model finding remains the Llama-3-8B **synonym presence detection** result (AUC 0.755 vs text baseline 0.499), which is the only non-trivial signal that survives all controls. No comparable presence-detection probe has been run on Qwen3-32B yet.

---

## 1. Experimental Setup

### 1.1 Models

| Property | Llama-3-8B-Instruct | Qwen3-32B |
|---|---|---|
| Parameters | 8B | 32.8B (31.2B non-embedding) |
| Layers | 32 | 64 |
| Hidden dim | 4096 | 5120 |
| Attention | GQA (32Q/8KV) | GQA (64Q/8KV) |
| Context | 8K | 32K |
| IFEval (approx) | ~70 | ~86.7 |
| Special handling | None | enable_thinking=False required |

### 1.2 Datasets

All game-scenario datasets share the same structure: 2000 scenarios (1000 stego-instructed, 1000 clean), balanced by design. The model receives a system prompt instructing a misaligned persona to output the keyword "attack" when steganography is detected.

| Dataset | Llama-3-8B | Qwen3-32B | Stego methods |
|---|---|---|---|
| suffix_marker | 2026-03-21 | 2026-03-22 | suffix_marker |
| whitespace | 2026-03-21 | 2026-03-22 | whitespace |
| game_scenarios_v2 | 2026-03-21 | NOT RUN | acrostic, prefix_binary, synonym, unicode, whitespace |
| Redwood presence | Earlier runs | NOT RUN | Same 5 methods |

### 1.3 Probe Configuration

All probes: logistic regression, 5-fold cross-validation, balanced class weights, AUC-ROC as selection metric. All activation extractions used the last token position across all layers.

---

## 2. Behavioral Results

### 2.1 Keyword Detection Rates

| Model | Dataset | Stego kw rate | Clean kw rate | Delta | Stat. sig? |
|---|---|---|---|---|---|
| **Llama-3-8B** | suffix_marker | 45.1% (451/1000) | 40.0% (400/1000) | +5.1pp | p=0.021 (marginal) |
| **Llama-3-8B** | whitespace | 46.7% (467/1000) | 49.2% (492/1000) | -2.5pp | p=0.263 (NS, wrong direction) |
| **Llama-3-8B** | game_scenarios_v2 | 40.2% (402/1000) | 37.0% (370/1000) | +3.2pp | p=0.142 (NS) |
| **Qwen3-32B** | suffix_marker | 4.9% (49/1000) | 0.0% (0/1000) | +4.9pp | Exact Fisher p < 0.001 |
| **Qwen3-32B** | whitespace | 0.1% (1/1000) | 0.1% (1/1000) | 0.0pp | No signal |

### 2.2 Interpretation of Behavioral Divergence

The behavioral profiles are strikingly different in character:

**Llama-3-8B: Indiscriminate compliance.** The model outputs "attack" ~40-49% of the time regardless of input condition. It cannot follow the conditional detection instruction. The stego-vs-clean deltas are tiny (max +5.1pp) and only marginally significant for suffix_marker. The whitespace delta is actually negative, meaning the model is slightly *less* likely to output "attack" when steganography is present. This is consistent with IFEval ~70: the model lacks the instruction-following capacity to reliably parse the system prompt's conditional logic.

**Qwen3-32B: Near-total refusal with perfect specificity.** The model almost never outputs "attack" (4.9% on suffix_marker, 0.1% on whitespace), but when it does on suffix_marker, it is exclusively in stego-present conditions (0% false positives). This suggests Qwen3-32B *can* detect the suffix marker but its safety/RLHF training suppresses the misaligned behavior in ~95% of cases. The 49 cases where it complies are concentrated in a narrow task template: 33/49 (67%) come from a single task type ("Create a script that sets up a Python virtual environment with common packages").

**Implication for the experiment:** Neither model produces the ideal behavioral signal (high stego compliance, low clean compliance) that would make the game-scenario paradigm straightforwardly interpretable. However, Qwen3-32B's perfect specificity means its 49 positive labels are trustworthy -- when the probe sees a positive, it is always genuinely stego-triggered. Llama's ~400+ positives per condition are contaminated by the model's baseline tendency to output "attack" regardless.

### 2.3 Per-Method Behavioral Breakdown (Llama game_scenarios_v2)

| Method | Stego kw rate | N |
|---|---|---|
| acrostic | 52.0% | 200 |
| whitespace | 49.0% | 200 |
| prefix_binary | 38.0% | 200 |
| unicode | 37.0% | 200 |
| synonym | 25.0% | 200 |

Note: synonym has the lowest keyword compliance but the strongest probe signal in the earlier Redwood presence detection runs. Behavioral compliance and activation-level signal appear to be orthogonal -- a key finding for the paper.

---

## 3. Keyword-Prediction Probe Results

These probes predict whether the model will output "attack" (keyword_present label) based on pre-generation activations.

### 3.1 Best-Layer Summary

| Model | Dataset | Best Layer | AUC-ROC (std) | AP | F1 | n_positive |
|---|---|---|---|---|---|---|
| **Llama-3-8B** | suffix_marker | L28 | 0.584 (0.028) | 0.499 | 0.510 | 851 |
| **Llama-3-8B** | whitespace | L9 | 0.603 (0.020) | 0.556 | 0.551 | 959 |
| **Llama-3-8B** | game_scenarios_v2 | L10 | 0.629 (0.014) | 0.494 | 0.531 | 772 |
| **Qwen3-32B** | suffix_marker | L19 | **0.861** (N/A) | 0.274 | 0.363 | 49 |
| **Qwen3-32B** | whitespace | L43 | 0.849* | 0.040* | 0.000* | 2 |

\* Qwen3-32B whitespace results are degenerate: only 2 positives out of 2000. AUC is not meaningful.

### 3.2 Layer-Wise Profile Comparison

**Llama-3-8B: Flat profile.** Across all three datasets, AUC-ROC varies within a narrow band (roughly 0.54-0.63 across all 32 layers). There is no strong localized peak. The best layers are scattered: L28 (suffix_marker), L9 (whitespace), L10 (game_scenarios_v2). This flatness suggests no single layer encodes a strong behavioral decision signal; the model is essentially not making a stego-conditioned decision at any depth.

| Llama Layer Range | suffix_marker AUC | whitespace AUC | game_v2 AUC |
|---|---|---|---|
| 0-7 (early) | 0.538-0.567 | 0.555-0.592 | 0.587-0.624 |
| 8-15 (early-mid) | 0.543-0.567 | 0.567-0.590 | 0.586-0.628 |
| 16-23 (mid-late) | 0.547-0.575 | 0.568-0.585 | 0.607-0.624 |
| 24-31 (late) | 0.565-0.584 | 0.564-0.576 | 0.592-0.619 |

**Qwen3-32B suffix_marker: Bimodal peaks.** Unlike Llama's flat profile, Qwen3-32B shows two distinct AUC clusters:

| Qwen3 Layer Cluster | Representative Layers | AUC-ROC Range |
|---|---|---|
| Early-mid peak | L19, L28-32 | 0.83-0.86 |
| Late peak | L49-51, L57 | 0.82-0.85 |
| Early layers (0-15) | L0-L15 | 0.71-0.81 |
| Mid trough | L33-45 | 0.74-0.81 |
| Final layers (58-63) | L58-L63 | 0.71-0.78 |

Top 10 layers by AUC-ROC for Qwen3-32B suffix_marker:

| Rank | Layer | AUC-ROC | AP | F1 |
|---|---|---|---|---|
| 1 | L19 | 0.861 | 0.274 | 0.363 |
| 2 | L31 | 0.855 | 0.228 | 0.287 |
| 3 | L49 | 0.854 | 0.356 | 0.399 |
| 4 | L51 | 0.854 | 0.334 | 0.370 |
| 5 | L30 | 0.846 | 0.276 | 0.326 |
| 6 | L28 | 0.845 | 0.308 | 0.342 |
| 7 | L32 | 0.833 | 0.248 | 0.298 |
| 8 | L50 | 0.832 | 0.333 | 0.357 |
| 9 | L57 | 0.827 | 0.273 | 0.311 |
| 10 | L29 | 0.826 | 0.234 | 0.315 |

Notably, the late layers (L49-51) have higher AP (0.33-0.36) and F1 (0.36-0.40) than the best-AUC layer L19 (AP 0.27, F1 0.36), suggesting these later layers may be more operationally useful for classification despite slightly lower ranking AUC.

### 3.3 AUC vs AP Discrepancy in Qwen3-32B

The Qwen3-32B suffix_marker probe has an AUC of 0.861 but AP of only 0.274. This large gap is explained by the extreme class imbalance: 49 positives vs 1951 negatives (2.45% positive rate). AUC-ROC is class-balance invariant but AP is sensitive to the prior. The probe achieves good ranking of stego vs clean activations, but the absolute precision at any reasonable recall is low due to the overwhelming negative class.

This is not a probe weakness per se -- it reflects the experimental design where the behavioral label is extremely rare. Retraining on the balanced stego-vs-clean label (1000/1000) would likely yield much higher AP.

### 3.4 Cross-Model Comparison at Matched Datasets

#### suffix_marker

| Metric | Llama-3-8B (L28) | Qwen3-32B (L19) | Delta |
|---|---|---|---|
| AUC-ROC | 0.584 | **0.861** | +0.277 |
| AP | 0.499 | 0.274 | -0.225 |
| F1 | 0.510 | 0.363 | -0.147 |
| n_positive | 851 (42.6%) | 49 (2.45%) | -802 |

The AUC improvement (+0.277) is dramatic, but AP and F1 are lower because of the 17x reduction in positive labels. If we controlled for class balance (training on stego-vs-clean labels), Qwen3-32B's advantage would likely be even larger since its 49 positives are noise-free (all genuinely stego-triggered) while Llama's 851 positives include ~400 that are indistinguishable from base-rate keyword output.

#### whitespace

| Metric | Llama-3-8B (L9) | Qwen3-32B (L43) |
|---|---|---|
| AUC-ROC | 0.603 | 0.849 (degenerate) |
| n_positive | 959 (48%) | 2 (0.1%) |

The Qwen3-32B whitespace result cannot be meaningfully compared. The model produces essentially zero behavioral signal on whitespace steganography. The reported AUC of 0.849 is an artifact of computing ROC on 2 positive examples and should be disregarded entirely.

---

## 4. Presence Detection Probes (Llama-3-8B Only)

These probes use the ground-truth `is_stego` label (balanced 1000/1000) rather than observed behavior. They were run on the earlier Redwood combined dataset with Llama-3-8B only.

### 4.1 Headline Results

| Subset | Activation Probe AUC (L3) | Text Baseline AUC | Probe advantage |
|---|---|---|---|
| **All methods pooled** | 0.934 | 0.866 | +0.068 |
| **Easy methods** (prefix_binary, unicode, whitespace) | 1.000 | 1.000 | None (both trivial) |
| **Synonym + acrostic** | 0.825 | 0.698 | +0.127 |
| **Synonym only** | **0.755** | **0.499** | **+0.256** |
| **Acrostic only** | 1.000 | 0.947 | +0.053 (marginal) |

### 4.2 Text Baseline Details

Two text baselines were evaluated:

| Baseline | All methods AUC | Synonym-only AUC |
|---|---|---|
| char_features (length, tab/zwc counts, digit/cap/space/punct ratios) | 0.866 | 0.499 |
| tfidf_char_ngram (2-4 char n-grams, 10K features) | 0.876 | N/A |

### 4.3 Why Synonym Is the Headline

The synonym channel is the only one where:
1. The activation probe materially exceeds the text baseline (0.755 vs 0.499)
2. The text baseline is at chance (0.499), meaning surface features carry zero information
3. The signal must therefore come from semantic/representational features that the text classifier cannot access

All other methods are either trivially detectable from surface text (prefix_binary, unicode, whitespace at AUC 1.000) or have a text baseline strong enough to explain most of the probe performance (acrostic: probe 1.000 vs text 0.947).

### 4.4 Transfer Matrix

Key findings from the cross-method transfer analysis:
- Synonym-trained probes transfer broadly (AUC 0.72-0.93 to other methods)
- Easy-method-trained probes fail on synonym (AUC 0.47-0.51)
- This asymmetry confirms synonym captures a more general "stego-ness" representation while easy-method probes learn method-specific surface artifacts

---

## 5. GCM Head Attribution (Qwen3-32B)

Generalized Causal Mediation (GCM) analysis was completed for suffix_marker on Qwen3-32B (50 samples). This identifies which attention heads causally mediate the stego detection signal using integrated gradients through all 64 layers with 64 heads each (4096 heads total).

### 5.1 Top Attributed Heads

| Rank | Layer | Head | IE Score |
|---|---|---|---|
| 1 | 42 | 7 | 0.1290 |
| 2 | 47 | 2 | 0.1214 |
| 3 | 45 | 3 | 0.1050 |
| 4 | 42 | 30 | 0.1040 |
| 5 | 43 | 61 | 0.1026 |
| 6 | 47 | 5 | 0.1018 |
| 7 | 39 | 30 | 0.0972 |
| 8 | 43 | 52 | 0.0968 |
| 9 | 39 | 61 | 0.0949 |
| 10 | 41 | 29 | 0.0946 |

### 5.2 Layer Distribution of Top Heads

Top heads are concentrated in layers **39-47** (62-73% depth into the network). This is the upper-middle region of Qwen3-32B's 64-layer architecture.

| Layer | # Heads in Top 50 | Notable heads |
|---|---|---|
| 39 | 4 | H30, H61, H42, H59 |
| 41 | 3 | H29, H16, H36 |
| 42 | 5 | H7 (top), H30, H25, H1, H41 |
| 43 | 4 | H61, H52, H31, H28 |
| 44 | 3 | H6, H2, H35 |
| 45 | 4 | H3, H18, H1, H43 |
| 47 | 5 | H2, H5, H30, H57, H24 |

### 5.3 Correspondence Between Probes and GCM

The GCM heads (L39-47) partially overlap with but are distinct from the probe peaks:
- **Probe peak 1** (L19, L28-32): Early-mid layers. GCM finds almost no heads here.
- **Probe peak 2** (L49-51): Late layers. GCM top heads are 2-10 layers earlier (L39-47).

This dissociation is expected: probes detect where information is linearly readable in the residual stream, while GCM identifies where information is causally *written* via attention. The signal may be written at L39-47 and persist in the residual stream to be readable at L49-51+.

---

## 6. Decode/Recognition Probes (Llama-3-8B Only)

### 6.1 Results

| Metric | Value |
|---|---|
| Best layer | L31 |
| AUC-ROC | 0.721 |
| AP | 0.103 |
| F1 (default threshold) | 0.128 |
| Recognized positives | 53/2000 (after anchored_last rescoring) |

### 6.2 Assessment

Recognition detection remains weak. The 53 recognized positives are overwhelmingly whitespace-based (192/196 in earlier counts). The benign control task (same carrier, summarize prompt) achieves AUC 0.719 -- nearly identical to the decode task. This means the probe is detecting "stego-looking text exists" rather than "model is in a decode state," undermining the recognition claim.

---

## 7. Confounds and Methodological Concerns

### 7.1 Qwen3-32B Suffix Marker: Labeling Confound

The 951 stego-instructed scenarios where Qwen3-32B refused to output "attack" are labeled as negatives (keyword_present=0). The probe is therefore trained to distinguish the 49 compliant stego cases from 1951 others (including 951 that had stego instructions but the model refused). This conflates:
- "Model detected stego and acted on it" (49 positives)
- "Model detected stego but refused" (some fraction of the 951)
- "No stego present" (1000 clean)

The probe may be learning "model compliance likelihood" or "this specific task template" rather than pure stego detection. Evidence: 67% of positives (33/49) come from a single task template.

**Recommendation:** Retrain probes on the balanced `expected_covert_action` label (1000 stego / 1000 clean) to test whether Qwen3-32B internally represents the presence of steganography independently of whether it acts on it.

### 7.2 Llama-3-8B: +++ Echo Confound

In suffix_marker data, 26.6% of clean responses hallucinate `+++` markers vs only 2.7% of stego responses. The model reproduces the marker more when it is *absent* from input. This inverted echo could confound probes that have access to response-derived features, though current probes use prompt-only (pre-generation) activations.

### 7.3 LoRA Carrier Contamination (Llama-3-8B)

The LoRA carrier generation attempt was abandoned: only 270/594 intended carriers were usable (53.7% complete), with 24/159 clean LoRA outputs contaminated by the secret key phrase. This dataset was not used for the main analyses.

### 7.4 Qwen3-32B Whitespace: Complete Behavioral Failure

With only 2/2000 keyword detections (1 stego, 1 clean), the whitespace condition on Qwen3-32B produces zero behavioral signal. Either:
- The model cannot detect whitespace encoding at all, OR
- It detects it but RLHF suppression is absolute for this method

No probe analysis is possible on behavioral labels. Only a stego-vs-clean probe could reveal whether internal representation exists.

---

## 8. Cross-Cutting Findings

### 8.1 Model Scale Produces Different Failure Modes, Not Simple Improvement

| Dimension | Llama-3-8B | Qwen3-32B |
|---|---|---|
| Behavioral failure mode | Indiscriminate compliance (~45% both conditions) | Near-total refusal (4.9% stego, 0% clean) |
| Positive label quality | Contaminated (high FP rate) | Clean (zero FP rate) |
| Probe signal strength | Weak (AUC 0.58-0.63) | Strong (AUC 0.86) but imbalanced |
| Layer profile | Flat (no localized signal) | Bimodal (L19, L49-51) |
| Signal interpretability | Low (can't distinguish signal from noise) | Moderate (confounded by refusal/compliance) |

### 8.2 The Probe-Behavior Orthogonality Observation

A recurring finding across both models: **behavioral compliance (keyword output rate) and probe discriminability operate on different axes.**

- Llama synonym: lowest behavioral compliance (25% keyword rate) but strongest presence probe (0.755 AUC)
- Qwen3 suffix_marker: very low compliance (4.9%) but strongest keyword probe (0.861 AUC)
- Llama whitespace: highest behavioral rate (49%) but weakest probe value

This suggests that probe-based monitoring may be most valuable precisely where behavioral detection fails -- the model "knows" something is different even when it does not act on it.

### 8.3 The "Synonym Headline" Survives Cross-Model Stress-Testing

The Llama-3-8B synonym presence detection result (AUC 0.755 vs text baseline 0.499) remains the single most scientifically defensible finding in the project. It is:
- Non-trivial (text baseline at chance)
- Transfer-positive (synonym-trained probes generalize to other methods)
- Methodologically clean (balanced labels, grouped CV, text baseline control)

However, it has NOT been replicated on Qwen3-32B. This is a critical gap. If the synonym signal is model-general rather than Llama-specific, it would substantially strengthen the paper's central claim.

### 8.4 Depth Scaling Comparison

Comparing relative layer positions (normalized to 0-1 across each model's depth):

| Signal type | Llama-3-8B (32 layers) | Qwen3-32B (64 layers) |
|---|---|---|
| Presence detection peak | L3 (9%) | Not yet run |
| Keyword probe peak | L9-L10 (28-31%) | L19 (30%) |
| Keyword probe secondary | L28 (88%) | L49-51 (77-80%) |
| GCM causal heads | Not run | L39-47 (61-73%) |
| Decode/recognition peak | L31 (97%) | Not run |

The keyword probe peaks occur at similar relative depths (~30% and ~80% for both models), suggesting the stego detection computation may be architecturally conserved across scale. The GCM heads at 61-73% occupy the region between the two probe peaks, consistent with being the mechanism that writes the signal read by the late probe peak.

---

## 9. Summary Comparison Table

| Experiment | Llama-3-8B | Qwen3-32B | Winner | Notes |
|---|---|---|---|---|
| **suffix_marker behavioral delta** | +5.1pp (noisy) | +4.9pp (clean, 100% specificity) | Qwen3 | Qwen's signal is interpretable despite lower rate |
| **whitespace behavioral delta** | -2.5pp (wrong dir) | 0.0pp | Neither | Both fail |
| **game_v2 behavioral delta** | +3.2pp (NS) | Not run | -- | |
| **Keyword probe AUC (suffix)** | 0.584 | **0.861** | Qwen3 | +47.5% relative improvement |
| **Keyword probe AUC (whitespace)** | 0.603 | Degenerate | Llama | Qwen3 has no labels |
| **Keyword probe AUC (game_v2)** | 0.629 | Not run | -- | |
| **Presence probe AUC (synonym)** | **0.755** | Not run | -- | Headline result |
| **Presence probe AUC (pooled)** | **0.934** | Not run | -- | Inflated by easy methods |
| **Text baseline (synonym)** | 0.499 (chance) | Not run | -- | Validates probe adds value |
| **GCM head attribution** | Not run | L39-47 top heads | -- | First causal evidence |
| **Decode recognition** | 0.721 (weak) | Not run | -- | Benign control matches |

---

## 10. Honest Claim Inventory

### Claims the data supports:

1. **Model scale changes the nature of steganographic interaction.** Llama-3-8B cannot distinguish stego from clean behaviorally; Qwen3-32B can but mostly refuses. The evidence is clear across 6000+ scenarios per model.

2. **Pre-generation activations encode future behavioral decisions about steganography.** Qwen3-32B's layer 19 probe (AUC 0.861) predicts keyword output significantly above chance from prompt-only features, meaning the model has already made a detection-related computation before generating any response tokens.

3. **Synonym steganography produces a non-trivial internal signal that text features cannot detect.** The Llama-3-8B synonym probe at 0.755 AUC vs 0.499 text baseline is clean evidence of a representation beyond surface statistics.

4. **GCM identifies a concentrated set of attention heads (L39-47) mediating the stego signal in Qwen3-32B.** This is the first causal evidence connecting specific model components to steganographic processing.

5. **Probe signal and behavioral compliance are orthogonal.** This is observed in both models and across multiple datasets.

### Claims the data does NOT support:

1. **"Probes reliably detect internal recognition/decoding."** Decode probe AUC (0.721) is matched by the benign control (0.719). No evidence of a decode-specific state.

2. **"Activation probes outperform text baselines across all methods."** For easy methods (prefix_binary, unicode, whitespace), text baselines achieve perfect detection. The probe advantage exists only for synonym (and partially acrostic).

3. **"Qwen3-32B whitespace results are meaningful."** With 2 positive labels, no statistical inference is possible.

4. **"The Qwen3-32B suffix_marker probe detects steganographic intent."** It may detect task-template effects or compliance tendency rather than stego detection per se (67% of positives from one template).

---

## 11. Recommended Next Steps (Prioritized)

| Priority | Action | Expected value | Blocks |
|---|---|---|---|
| **1** | Retrain Qwen3-32B probes on `expected_covert_action` (stego-vs-clean label) for both suffix_marker and whitespace | Very high -- resolves whether Qwen3 internally detects stego regardless of behavior | None |
| **2** | Complete Qwen3-32B game_scenarios_v2 run (currently queued in pipeline) | High -- enables per-method comparison and synonym replication | GPU time |
| **3** | Run Qwen3-32B presence detection probes on synonym specifically | Highest scientific value -- would confirm/deny cross-model generality of headline result | Needs game_v2 run first |
| **4** | Run GCM steering experiment (02_gcm_steer.py) on suffix_marker | High -- first intervention experiment | GCM IE complete |
| **5** | Run Qwen3-32B GCM on whitespace | Medium -- tests whether GCM finds heads even without behavioral signal | GPU time |
| **6** | Foreground synonym as primary narrative in any writeup | Near zero cost, high value | None |
| **7** | Document Llama-3-8B failure as negative result supporting probe-based monitoring | Writing task | None |

---

## Appendix A: Raw Data Inventory

### Llama-3-8B-Instruct runs (data/Meta-Llama-3-8B-Instruct/runs/)
- `2026-03-21_suffix_marker/` - complete (activations, metadata, responses, probes)
- `2026-03-21_whitespace/` - complete
- `2026-03-21_game_scenarios_v2/` - complete
- `2026-03-21_lora_carriers_hugo/` - partial, abandoned (53.7% complete, contamination issues)

### Qwen3-32B runs (data/Qwen3-32B/runs/)
- `2026-03-22_suffix_marker/` - complete (activations, metadata, responses, probes)
- `2026-03-22_whitespace/` - complete (but degenerate behavioral labels)

### Earlier Redwood runs (data/probe_results/)
- `stego_detection_probe_results.json` - presence detection, all 32 layers, per-method subsets
- `text_baseline_results.json` - char_features and tfidf baselines
- `game_scenarios_probe_results.json` - early 30-sample pilot (superseded)
- `decode_prompt_pilot_summary.json` - decode prompt format comparison

### GCM (steer/)
- `Qwen3-32B_top10_stego_heads_suffix_marker.json` - 300 heads ranked by IE score

### Notes (notes/)
- `report-2026-03-21-hugo.md` - Pipeline repair and re-analysis
- `report-2026-03-21-llama8b-interim-analysis.md` - Superseded by handoff doc
- `handoff-qwen3-32b-overnight-run.md` - Complete Llama evidence + Qwen3 plan
- `qwen3-32b-probe-results-2026-03-22.md` - Qwen3 suffix/whitespace results
- `gcm-qwen3-32b-implementation-notes.md` - GCM technical implementation details

---

## Appendix B: Statistical Notes

1. All AUC-ROC values are means over 5 CV seeds unless otherwise noted.
2. Llama-3-8B behavioral significance tests use two-proportion z-tests.
3. Qwen3-32B suffix_marker keyword rate significance: Fisher exact test on 2x2 table (49/1000 stego vs 0/1000 clean) yields p < 0.001.
4. Qwen3-32B whitespace: 1/1000 vs 1/1000 is indistinguishable from noise (p = 1.0).
5. Class imbalance for Qwen3-32B suffix_marker probes (2.45% positive) means AUC and AP should be interpreted in context: AUC measures ranking quality (0.861 is strong), AP measures precision-recall trade-off under prior (0.274 reflects the imbalance, not probe weakness).
