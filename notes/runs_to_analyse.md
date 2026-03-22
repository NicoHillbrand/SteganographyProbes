
---
## 2026-03-22T17:17:13Z | data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2

**What:** game_scenarios_v2: keyword probes + stego_vs_clean probes (Qwen3-32B, 5 methods, AUC comparison target)

**Files to analyse:**
- data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2/probe_results/probe_results.json
- data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2/probe_results/probe_results_stego_vs_clean.json
- data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2/activations/responses.jsonl
- data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2/config.json

**Status:** awaiting analysis

---
## 2026-03-22T17:42:35Z | data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2

**What:** text baselines complete — check synonym delta vs 0.499 Llama baseline

**Files to analyse:**
- data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2/probe_results/probe_results.json
- data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2/probe_results/probe_results_stego_vs_clean.json
- data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2/activations/responses.jsonl
- data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2/config.json

**Status:** awaiting analysis

---
## 2026-03-22T18:18:27Z | data/Qwen3-32B/runs/2026-03-22_prefixsuffix_varied

**What:** prefixsuffix_varied: inline_keyword + prefix_marker_varied + suffix_marker_varied (Qwen3-32B, 5000 examples) — keyword probes + stego_vs_clean probes

**Files to analyse:**
- data/Qwen3-32B/runs/2026-03-22_prefixsuffix_varied/probe_results/probe_results.json
- data/Qwen3-32B/runs/2026-03-22_prefixsuffix_varied/probe_results/probe_results_stego_vs_clean.json
- data/Qwen3-32B/runs/2026-03-22_prefixsuffix_varied/activations/responses.jsonl
- data/Qwen3-32B/runs/2026-03-22_prefixsuffix_varied/config.json

**Status:** awaiting analysis

---
## 2026-03-22T18:34:47Z | data/Qwen3-32B/runs/2026-03-22_benign_keyword_suffix_marker

**What:** benign_keyword_suffix_marker: does probe fire on +++ in non-scheming context? Compare probe AUC vs suffix_marker (scheming) to test context-dependence

**Files to analyse:**
- data/Qwen3-32B/runs/2026-03-22_benign_keyword_suffix_marker/probe_results/probe_results_stego_vs_clean.json
- data/Qwen3-32B/runs/2026-03-22_benign_keyword_suffix_marker/activations/responses.jsonl
- data/Qwen3-32B/runs/2026-03-22_benign_keyword_suffix_marker/config.json

**Status:** awaiting analysis

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
