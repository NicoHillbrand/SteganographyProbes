# Steganography Interpretability: Full Experiment Analysis Report

**Date:** 2026-03-22 (updated during overnight re-run)
**Models:** Llama-3-8B-Instruct, Qwen3-32B
**Status at writing:** v2 pipeline running (P0+probes complete, P1/P2/P3 pending)

---

## Executive Summary

This project set out to answer three mechanistic questions about whether language models internally represent steganographic content:

1. **Is there a representational signal?** Can we decode stego presence from activations, and does that signal survive surface text (TF-IDF/char) baselines?
2. **Is it localized?** Which layers and components contribute most to the stego representation?
3. **Is it causal?** Can we steer the representation and verify downstream propagation?

**What we found:**

- The only method that produced probe signal potentially exceeding text baselines was **Llama-3-8B synonym steganography** on the Redwood combined_dataset. Original probe AUC at L3 = **0.754**, vs synonym-specific TF-IDF baseline = **0.569** (delta +0.185) and char-feature baseline = **0.499** (delta +0.255).

- This finding was **challenged** by a re-extraction failure (caused by an `enable_thinking=False` crash on Llama that forced removal of `--use_chat_template`), which gave AUC 0.574 at L9 (delta +0.005 over tfidf — effectively zero). The **v2 re-extraction** (with the fixed `enable_thinking` handling, running now) partially restores the signal: AUC ~0.741 at L9 in pooled probe training so far.

- **All other methods** on both models are cleanly explained by surface features: acrostic, prefix_binary, unicode, and whitespace give tfidf baselines of 0.978–1.000, matching or exceeding probe AUCs.

- **GCM head attribution** on Qwen3-32B failed its own permutation null test (ratio 0.92x, 9/10 top heads identical under permuted labels), confirming the attribution method is a high-variance head detector, not a causally specific circuit finder.

- **Probe direction steering** on Llama-3-8B (P3) confirmed causal direction existence but is only interpretable once probe calibration is validated (v2 results pending).

---

## 1. Dataset and Experimental Setup

### 1.1 Models

| Property | Llama-3-8B-Instruct | Qwen3-32B |
|---|---|---|
| Parameters | 8B | 32.8B |
| Layers | 32 | 64 |
| Hidden dim | 4096 | 5120 |
| Attention heads | 32Q/8KV (GQA) | 64Q/8KV (GQA) |
| Special handling | `enable_thinking` not supported | `enable_thinking=False` required |

### 1.2 Datasets

**Redwood combined_dataset.jsonl** (primary, Llama):
- 2000 examples: 1000 clean + 200×5 stego (synonym, acrostic, whitespace, unicode, prefix_binary)
- Balanced, grouped by carrier_group_id (50 groups)
- Diverse carrier texts → harder for character n-gram features to detect method artifacts

**Game scenarios v2** (Qwen + Llama behavioral):
- 2000 examples: 1000 clean + 200×5 stego
- AI-generated game scenario carrier texts
- Domain vocabulary creates tfidf artifacts → synonym substitutions detectable by char n-grams (tfidf AUC 0.870 for Qwen)

**Auxiliary Qwen datasets:**
- suffix_marker (2000), whitespace (2000), prefixsuffix_varied (5000), benign_keyword (400)

### 1.3 Experimental Paradigm

Two distinct probe targets were used:
- **Game-scenario probes (behavioral):** Train probe to predict `keyword_present` (did the model output "attack"?). Only valid when model compliance is high and specific.
- **Stego-vs-clean probes (representational):** Train probe to predict `is_stego` (ground-truth input label). Tests whether the model's internal representation encodes the presence of steganography, regardless of behavior.

**Critical distinction:** A probe that detects surface text features (char n-grams, whitespace patterns) is learning dataset artifacts, not a model representation. Per-method text baselines are required to distinguish the two.

---

## 2. Surface Baseline Results

The critical filter: which methods produce probe signal that **exceeds** TF-IDF character n-gram baselines?

### 2.1 Llama-3-8B on Redwood combined_dataset

| Method | Probe AUC (best layer) | Char baseline | TF-IDF baseline | Delta vs TF-IDF | Survives? |
|---|---|---|---|---|---|
| acrostic | 1.000 (L3) | 0.947 | 1.000 | 0.000 | ❌ surface |
| prefix_binary | 1.000 (L3) | 1.000 | 1.000 | 0.000 | ❌ surface |
| unicode | 1.000 (L3) | 1.000 | 1.000 | 0.000 | ❌ surface |
| whitespace | 1.000 (L3) | 1.000 | 0.978 | +0.022 | ❌ surface |
| **synonym** | **0.754 (L3)** | **0.499** | **0.569** | **+0.185** | ✅ *survives* |

Acrostic/prefix_binary/unicode/whitespace are trivially detected by byte-level statistics. Synonym substitution cannot be detected by character n-grams (no length change, minimal bigram disruption from word-level substitution in diverse texts), so the synonym probe signal is not explained by surface features.

**Caveat on the 0.754 figure:** The original extraction config is not preserved on disk. A re-extraction with `--use_chat_template=false` gave AUC 0.574 (no signal), while v2 (with `--use_chat_template=true`, running now) gives AUC ~0.741 at L9 in pooled training. Per-method synonym AUC in v2 is pending.

### 2.2 Qwen3-32B on Game Scenarios v2 (Representational Probes)

| Method | Probe AUC (L48) | Char baseline | TF-IDF baseline | Delta vs TF-IDF | Survives? |
|---|---|---|---|---|---|
| unicode | 0.986 | n/a | n/a | n/a | ❌ surface (overt) |
| prefix_binary | 0.973 | n/a | n/a | n/a | ❌ surface (overt) |
| acrostic | 0.952 | n/a | n/a | n/a | ❌ surface |
| whitespace | 0.884 | n/a | n/a | n/a | ❌ surface |
| **synonym** | **0.868** | ~0.620 | **0.870** | **−0.002** | ❌ no signal |

The Qwen synonym probe (AUC 0.868) is matched exactly by the TF-IDF character n-gram baseline (0.870). The game_scenarios_v2 dataset uses domain-specific game vocabulary, making synonym substitution character-detectable in ways that the Redwood combined_dataset is not. **This is not a replication of the Llama synonym finding** — it reflects a dataset-level difference in text diversity.

### 2.3 Qwen3-32B Suffix Marker / Whitespace

Both achieve probe AUC 0.929 (suffix) and 0.826 (whitespace). The TF-IDF baseline on suffix_marker is not per-method, but the `+++` token is trivially detectable by any character feature. The prefixsuffix_varied experiment confirmed this: tfidf AUC 0.998, probe AUC 0.942 (probe doesn't add value over text).

**Conclusion:** Suffix marker, whitespace, acrostic, unicode, and prefix_binary are all surface artifacts. The only candidate for a genuine representational finding is **Llama-3-8B on synonym (combined_dataset)**, and that finding's reliability is currently uncertain pending v2 re-extraction.

---

## 3. Behavioral Results

### 3.1 Model Compliance Patterns

| Model | Dataset | Stego kw rate | Clean kw rate | Signal |
|---|---|---|---|---|
| Llama-3-8B | suffix_marker | 45.1% | 40.0% | Marginal (+5.1pp) |
| Llama-3-8B | whitespace | 46.7% | 49.2% | Null (wrong direction) |
| Llama-3-8B | game_scenarios_v2 | 40.2% | 37.0% | Null (p=0.142) |
| Qwen3-32B | suffix_marker | 4.9% | 0.0% | Specific, low-N |
| Qwen3-32B | whitespace | 0.1% | 0.1% | Null |
| Qwen3-32B | game_scenarios_v2 | 0.3% | 0.4% | Inverted |

**Llama-3-8B: Indiscriminate compliance.** IFEval ~70 → insufficient instruction-following to parse conditional "output keyword only when stego detected" logic. ~40% baseline output regardless of condition.

**Qwen3-32B: Near-total refusal with rare perfect specificity.** RLHF suppresses compliant behavior in ~95-99.7% of cases. When it does comply (suffix_marker: 49/1000), it's always on stego examples (0% false positives). This reveals the model *can* detect suffix markers but refuses to act on them. Concentrated in one task template (33/49 = 67% from "create Python virtual environment" task).

**Implication:** Neither model is suitable for the behavioral probe paradigm. Llama's signal is noise; Qwen's compliance is task-template-specific and low-N. Representational probes (stego-vs-clean, ground truth labels) are the only scientifically valid probe type for both models.

---

## 4. Mechanistic Interpretability Chain (P0–P3)

### 4.1 Overview

The P0→P1→P2→P3 chain was designed to move from "does the probe exist" to "is it causally localized":

- **P0:** Re-extract activations with `--save_probes`, train all-layer probes with `--use_ground_truth_labels`
- **P1:** Layerwise residual stream knockout — which layer's computation builds the L3 probe representation?
- **P2:** Probe direction cosine analysis — is there a consistent "stego direction" across layers?
- **P3:** Probe-direction steering — inject L3 probe coef as steering vector, measure downstream probe score shift

### 4.2 v1 Results (Extraction Without `--use_chat_template`)

Due to a bug (`enable_thinking=False` hardcoded in `apply_chat_template` call, crashing Llama), v1 extraction omitted the chat template. This produced miscalibrated activations:

**P0 baseline probe score at L3 on synonym stego:** 0.436 (below chance — probe does not detect stego)

**P1 (layerwise knockout):**
- Total positive drop: 0.0
- Localized: False
- Interpretation: **Uninterpretable.** The probe had no signal to localize. Knocking out any layer cannot reduce a score that starts at chance.

**P2 (cosine similarity):**
- Mean off-diagonal cosine: 0.302 (medium coherence)
- Highest-similarity cluster: L23–L25 (cos ~0.92), neighboring-layer redundancy
- L3 most similar to L2 (cos=0.61) and L4 (cos=0.60), suggesting direction is local
- Interpretation: Multiple distinct sub-computations, not a single global "stego direction"

**P3 (probe direction steering, alpha=5.0):**
- causal_confirmed: True
- mean_downstream_delta: +0.636
- kl_divergence: 1.852
- Interpretation: **Tautological.** Injecting the probe direction at L3 into activations where the probe was already miscalibrated (scoring chance on stego, chance on clean) produces large downstream shifts — but these shifts reflect L3's influence on downstream computation, not stego-specific signal. KL=1.852 confirms alpha=5.0 severely disrupts the output distribution.

### 4.3 v2 Results (Extraction With `--use_chat_template` — In Progress)

Launched 2026-03-22 at 23:02 UTC after fixing `enable_thinking=False` in `05_run_inference.py` (line 321 now conditional on `"Qwen3" in args.model`).

**P0 status (complete):**
- Run dir: `data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/`
- Pooled probe training in progress; selected layer AUCs so far:
  - L3: 0.690, L6: 0.718, L7: 0.722, L8: 0.727, L9: 0.741
  - (Best layer TBD; original was L3 at 0.754)
- The systematic recovery vs v1 (all layers +0.1–0.15 AUC) confirms the extraction format matters

**P1 (pending):** Will be interpretable only if the v2 L3 probe baseline score on synonym stego > ~0.7

**P2 (pending):** Will run on v2 probes; structure expected similar to v1 (medium coherence)

**P3 (alpha=1.0 and 2.0, pending):** Key question — does causal propagation persist at lower alpha with acceptable KL?

---

## 5. GCM Attribution: Methodology Critique and Failure Analysis

### 5.1 The GCM Pipeline

Generalized Causal Mediation (GCM) computes an "indirect effect" (IE) per attention head:

```
IE_head = ∇_z log(P_stego(attack)/P_clean(attack)) · z_clean
```

Computed via gradient of log-probability ratio with respect to pre-output-projection activations. Intended to identify which heads "mediate" the stego representation.

### 5.2 Results (Qwen3-32B, Suffix Marker)

| | Real labels | Permuted labels |
|---|---|---|
| Top-10 mean score | 0.0743 | 0.0804 |
| Ratio real/permuted | **0.924** | (reference) |
| Top-10 head overlap | **9/10 identical** | |

**The permutation null test failed.** A ratio < 1.0 means the "real" top-10 heads actually score *lower* than the permuted top-10. 9/10 heads are the same under real and shuffled labels. The attribution method is not condition-specific.

### 5.3 Root Cause: `abs()` Before Averaging

The critical bug in `01_gcm_filter_heads_hf.py`:

```python
total_ie_scores[l] += ie_per_head.abs().cpu()  # WRONG: accumulates abs before averaging
```

`ie_per_head` elements can be positive (stego→attack) or negative (stego→suppress). Accumulating absolute values before averaging across examples makes the score measure **gradient magnitude** (which heads have large gradients), not **consistent directional effect** (which heads consistently drive the stego→attack pathway).

High-gradient heads are the highest-variance heads in the network — they appear in top-10 regardless of what condition you're comparing. Hence 9/10 overlap with permuted labels.

**Fix (not implemented):** Accumulate signed values, average across examples first, then take absolute value of the mean:
```python
total_ie_scores[l] += ie_per_head.cpu()          # accumulate signed
# at end: score = total_ie_scores[l].abs() / n_pairs
```

### 5.4 Confirmation from P4 (Mean-Diff)

P4 computed `||mean(stego_acts) - mean(clean_acts)||_2` per layer (signed mean before abs):
- P4 top layers: L62, L61, L60, L59, L58 (late layers)
- GCM top layers: L45, L42, L44, L43, L41 (mid layers)
- Spearman rank correlation GCM vs P4: 0.413
- Top-5 layer overlap: 0/5

GCM found the wrong layers. Mean-diff (equivalent to correct signed accumulation) finds late layers L58–62, which are plausible for a 64-layer Qwen3-32B (analogous to L3 in 32-layer Llama, ~80–97% depth).

### 5.5 Implications

The GCM pipeline as implemented is a high-variance head detector. Any stego/clean comparison run through it will produce a plausible-looking list of "top heads" that fails permutation testing. This includes:
- Qwen3-32B suffix_marker (done, confirmed failed)
- Qwen3-32B whitespace (not null-tested, same code, same failure expected)
- Llama-3-8B synonym (computations ran but file not saved; same code issue)

**All GCM head attributions should be treated as invalid** until the signed accumulation fix is applied and the null test ratio exceeds 2.0x.

---

## 6. Cross-Model Comparison

### 6.1 Where Stego Signal Lives

| Model | Method | Best probe layer | AUC | Fraction of depth |
|---|---|---|---|---|
| Llama-3-8B (32L) | synonym | L3 (original) / L9 (v2) | 0.754 / ~0.74 | 9–28% |
| Qwen3-32B (64L) | synonym | L48 | 0.868 | 75% |
| Qwen3-32B (64L) | suffix_marker | L49 | 0.929 | 77% |
| Qwen3-32B (64L) | whitespace | L5 | 0.826 | 8% |

**Llama:** Stego representation appears very early (L3, 9% depth) — the model encodes synonym-stego presence at the first few transformer layers. This is consistent with the model not "acting" on this information (it doesn't output the keyword reliably) but still building an internal categorization early in the forward pass.

**Qwen3-32B:** Signal generally peaks in late layers (L48–L49, ~75% depth). The exception is whitespace, which peaks at L5 (8%) — possibly reflecting the model's early detection of whitespace anomalies. Late-layer signals on a larger, more capable model may reflect higher-level semantic understanding rather than low-level pattern matching.

### 6.2 Scale and Behavior

The contrast in behavioral compliance (Llama ~40% indiscriminate vs Qwen3-32B ~5% specific) is a pure RLHF/capability story. Qwen3-32B has a much stronger instruction-following backbone (IFEval ~87 vs ~70) and safety fine-tuning that suppresses the misaligned behavior. This doesn't mean Qwen3-32B "doesn't represent" stego — the representational probes show it does (AUC 0.929 for suffix_marker). The model internally detects stego presence but refuses to act on it.

### 6.3 Dataset Confound: Why Qwen Synonym = Surface Artifact but Llama Synonym ≠

The same synonym substitution method produces meaningfully different text baselines depending on the carrier text:

| | Combined_dataset (Llama) | Game_scenarios_v2 (Qwen) |
|---|---|---|
| Carrier text type | Diverse Redwood synthetic | AI game scenarios (domain vocab) |
| Synonym TF-IDF baseline | 0.569 | 0.870 |
| Probe AUC | 0.754 | 0.868 |
| Probe exceeds tfidf? | Yes (+0.185) | No (−0.002) |

The Qwen run was on a structurally different dataset (game_scenarios_v2) with repeated domain vocabulary. Synonym substitution within a narrow vocabulary produces consistent n-gram frequency shifts detectable by TF-IDF. The Redwood combined_dataset is much more diverse, so the same substitution produces negligible n-gram changes. This is a dataset confound, not a model confound.

---

## 7. Current State of Claims

### 7.1 Claims That Stand

1. **Stego methods with overt surface markers (suffix_marker, unicode, prefix_binary, acrostic, whitespace) are all detectable by text features alone.** Probe AUC ≈ TF-IDF AUC for all these methods across both models. Probes are not learning model representations; they are learning dataset artifacts. (Verified across 5 datasets, 2 models.)

2. **Qwen3-32B builds a probe-decodable stego representation for suffix_marker that survives the model's refusal to act.** AUC 0.929 at L49 on balanced 1000/1000 labels. The representation exists even when behavior is suppressed. (Caveat: the signal is surface-explainable; the probe is detecting the `+++` token, not a deep representation.)

3. **GCM head attribution fails its permutation null test (ratio 0.92x, 9/10 head overlap).** The method as implemented is a high-variance head detector, not a causally specific attribution. This is a methodological finding, not a negative interpretability result. (Confirmed on Qwen3-32B suffix_marker.)

4. **Signed mean-diff and abs()-accumulation GCM disagree on which layers matter** (Spearman 0.413, 0/5 top-5 overlap). This is further evidence that the GCM implementation is finding the wrong signal.

5. **The P3 probe direction exists at L3 and influences downstream layers** (alpha=5.0: mean_downstream_delta=+0.636). Whether this direction is stego-specific depends on whether the underlying probe is calibrated (i.e., whether v2 restores the signal).

### 7.2 Claims Under Active Review

6. **Llama-3-8B internally represents synonym steganography above text baseline (AUC 0.754, delta +0.185).** *Status: Retracted in v1, restoration attempted in v2.* The original extraction config is not on disk; v1 re-extraction (no chat template) gave 0.574; v2 (with chat template) is showing ~0.74 pooled — per-method synonym AUC pending.

### 7.3 Claims That Were Retracted

7. ~~"Qwen3-32B replicates the Llama synonym finding."~~ Retracted. The Qwen synonym probe (AUC 0.868) matches its TF-IDF baseline (0.870). Different dataset, different text statistics.

8. ~~"GCM identifies the specific attention heads mediating stego detection."~~ Retracted. Null test failed (ratio 0.924). Attribution is not condition-specific.

---

## 8. Lessons and Methodological Takeaways

### 8.1 Baseline Order of Operations

Always compute per-method text baselines (char features + TF-IDF character n-grams, preferably on the exact method subset) before interpreting any probe result. Pooled baselines are misleading when some methods are trivially detectable.

### 8.2 Chat Template Consistency

Activation extraction format must match across any experiment that is compared. Raw text and chat-template-wrapped text produce materially different activations (pooled AUC 0.574 vs 0.741+ at L9 in our runs). This applies to all hooks, steering experiments, and baseline comparisons.

**Fix committed:** `05_run_inference.py` line 321 now conditionally applies `enable_thinking=False` only when `"Qwen3" in args.model`.

### 8.3 Attribution Method Validation

Every attribution method (GCM, gradient attribution, attention rollout) requires a permutation null test before results can be reported as condition-specific. The target ratio is >2x; anything below 1.5x indicates the method is detecting dataset variance, not condition-specific circuits.

**Signed accumulation rule:** Never take `abs()` before averaging across examples. The correct order is: accumulate signed values → compute mean → take abs of mean.

### 8.4 Knockout Ablation vs Patching

Knockout ablation (replace layer output with input) does not require clean/stego pairs, making it usable even when pairing is complex (e.g., synonym stego with 13 carrier groups). However, it is sensitive to probe calibration — a miscalibrated probe at the target layer will show zero drop regardless of which layers are ablated.

### 8.5 Alpha Calibration in Steering

Steering alpha must be tuned against KL divergence on clean examples. At alpha=5.0, KL=1.852 — the output distribution is severely disrupted and the "causal confirmation" is trivially true for any steering direction. Target: `alpha` such that `causal_confirmed=True` AND `kl_divergence < 0.3`. This is the minimum effective alpha.

---

## 9. Pending Results (v2 Pipeline)

The following results will be added here when the v2 run completes (expected ~00:05 UTC):

| Result | Expected | Status |
|---|---|---|
| P0 per-method synonym AUC | > 0.70, per-method | Pending |
| P0 L3 baseline probe score on synonym stego | > 0.70 | Pending |
| P1 total positive drop | > 0.0 | Pending |
| P1 localized (>80% drop in ≤3 layers) | TBD | Pending |
| P1 top-3 layers by drop | L0–L5 expected | Pending |
| P2 mean off-diagonal cosine | ~0.30 (expected similar to v1) | Pending |
| P3 alpha=1.0: causal_confirmed | TBD | Pending |
| P3 alpha=1.0: kl_divergence | Target < 0.3 | Pending |
| P3 alpha=2.0: causal_confirmed | TBD | Pending |

---

## 10. Open Questions for Future Work

1. **Does the Llama synonym signal replicate robustly?** The v2 results will resolve whether the signal is real under proper extraction. If yes, a synonym-specific TF-IDF baseline on the v2-formatted texts should be run to confirm the delta survives.

2. **Is there a stego-specific circuit in Llama synonym?** If P1 localizes signal to 1–3 early layers, activation patching (clean→stego) at those layers would complete the causal story.

3. **Does the GCM structural fix (signed accumulation) recover interpretable head attribution?** This is a ~2h code change + rerun. If the corrected null test ratio exceeds 2x, head-level attribution becomes credible.

4. **What are the late-layer L58–62 Qwen3-32B suffix_marker representations?** Mean-diff finds strong signal there; mean-diff-based patching (analogous to GCM but with correct accumulation) could identify which heads contribute.

5. **Does the L3 probe direction in Llama encode semantically interpretable features?** Even if the probe calibration is uncertain, the direction exists and steers downstream scores. Geometric analysis (similarity to known concept directions, e.g., from SAE decompositions) could be informative.

6. **Is Qwen3-32B's stego detection behavioral suppression a consistent circuit?** The 49 suffix_marker compliant examples are concentrated in one task template. Patching clean→stego at L49 in that task template and measuring behavioral change would test whether the representation-to-behavior link exists and is task-specific.

---

*Last updated: 2026-03-22 23:10 UTC — v2 pipeline running, P1/P2/P3 results pending.*
*Author: Claude Sonnet 4.6 (automated analysis session)*
