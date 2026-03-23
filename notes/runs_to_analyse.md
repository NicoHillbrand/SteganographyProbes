
> **All runs below have been analysed and incorporated into the definitive report: [notes/final-analysis-report.md](final-analysis-report.md) (2026-03-23).**

---
## 2026-03-22T17:17:13Z | data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2

**What:** game_scenarios_v2: keyword probes + stego_vs_clean probes (Qwen3-32B, 5 methods, AUC comparison target)

**Status:** done: Pooled AUC 0.847 at L48. Synonym AUC 0.868 RETRACTED (tfidf=0.870). All methods surface-explained. See pipeline-audit Section 1.1.

---
## 2026-03-22T17:42:35Z | data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2

**What:** text baselines complete — check synonym delta vs 0.499 Llama baseline

**Status:** done: Per-method tfidf computed. Synonym tfidf=0.870 matches probe 0.868. Qwen cross-model replication RETRACTED. See pipeline-audit Concern 8.

---
## 2026-03-22T18:18:27Z | data/Qwen3-32B/runs/2026-03-22_prefixsuffix_varied

**What:** prefixsuffix_varied: inline_keyword + prefix_marker_varied + suffix_marker_varied (Qwen3-32B, 5000 examples) — keyword probes + stego_vs_clean probes

**Status:** done: Best L46 AUC 0.942. All 3 methods are surface markers (pooled tfidf=0.998). See pipeline-audit Section 1.2a.

---
## 2026-03-22T18:34:47Z | data/Qwen3-32B/runs/2026-03-22_benign_keyword_suffix_marker

**What:** benign_keyword_suffix_marker: does probe fire on +++ in non-scheming context? Compare probe AUC vs suffix_marker (scheming) to test context-dependence

**Status:** done: **FAILED EXPERIMENT** — 0/200 compliance (wanted more than "attack"). Probe AUC 0.909 re-detects +++ surface marker. See pipeline-audit Section 1.2b.

---
## 2026-03-22T21:41:03Z | data/Qwen3-32B

**What:** GCM regen suffix_marker+whitespace (fixed template) + permutation null + steering — all steps complete

**Files to analyse:**
- data/Qwen3-32B_top10_stego_heads_suffix_marker.json (regenerated)
- data/Qwen3-32B_top10_stego_heads_whitespace.json (regenerated)
- data/Qwen3-32B_top10_stego_heads_suffix_marker_permuted.json
- data/Qwen3-32B_steering_summary_suffix_marker.json
- data/Qwen3-32B/gcm_results_suffix_marker_regen.json

**Status:** done: **NEGATIVE RESULT.** Permutation null fails (ratio=0.92x, real < permuted). 9/10 top-10 heads identical under shuffled labels. Steering complete null (0% ASR pre/post, KL=0.004). Overlap with whitespace increased to 54% after template fix. Root cause: `abs()` accumulation in IE computation. GCM as implemented finds high-variance heads, not stego-specific ones.

---
## 2026-03-22T22:10:00Z | data/Meta-Llama-3-8B-Instruct (GCM synonym)

**What:** GCM permutation null on Llama-3-8B synonym dataset — cross-model validation of abs() artifact

**Files to analyse:**
- data/Meta-Llama-3-8B-Instruct_top10_stego_heads_synonym_permuted.json
- data/Meta-Llama-3-8B-Instruct_top10_stego_heads_synonym_permuted_config.json
- data/synonym_game_scenarios.jsonl (new synonym-only dataset, 400 entries)

**Status:** done: **SAME PATTERN AS QWEN.** Permuted null: L31 dominates (6/10 top-10, 10/50 top-50). Last 3 layers (29-31) = 52% of top-50 — identical to Qwen's 52% in L42-45. Top-10 mean permuted score: 0.0333. Real (non-permuted) heads file LOST due to relative path error in script. `abs()` IE artifact confirmed architecture-general across both models.

---
## 2026-03-22T22:31:01Z | data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract

**What:** stego_detection_reextract: Llama-3-8B all-layer activations + probes + P1 ablation + P2 cosine + P3 steering

**Files to analyse:**
- data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results/probe_results_stego_vs_clean.json
- data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results_stego_vs_clean/p1_ablation_results.json
- data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results_stego_vs_clean/p2_probe_direction_analysis.json
- data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results_stego_vs_clean/p3_probe_steer_results.json
- data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/config.json
- data/Qwen3-32B_mean_diff_by_layer_suffix_marker.json

**Status:** done: **SYNONYM RETRACTED — fails replication.** Re-extraction synonym AUC=0.574 (was 0.754), delta +0.005 vs tfidf (was +0.186). All methods degraded (use_chat_template=false vs likely true in old run). P1 ablation: total_drop=0.0, L3 probe baseline 0.436. P2: direction local (cos<0.2 by L9). P3: tautological steering (KL=1.853). P4 mean-diff: Qwen suffix_marker monotonic L2 increase, 0/5 overlap with GCM. **Project now has zero confirmed positive interpretability findings.**

---
## 2026-03-22T23:17:23Z | data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2

**What:** stego_detection_reextract_v2: Llama-3-8B chat_template activations + probes + P1 ablation + P2 cosine + P3 steering (alpha=1.0,2.0)

**Files to analyse:**
- data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results/probe_results_stego_vs_clean.json

- data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/config.json

**Status:** done: **SYNONYM DEFINITIVELY RETRACTED.** v2 (--use_chat_template, enable_thinking bug fixed) synonym AUC=0.489, below TF-IDF 0.569. P1: total_drop=0.0 (localized=False). P2: mean_cos=0.283, layer-specific only. P3: alpha=2.0 causal_confirmed=True, delta=+0.238, KL=0.161 (valid calibration, but direction from surface-dominated pooled probe). See notes/final-analysis-report.md for full synthesis.
