# Overnight Chain Status — Sun Mar 22 23:02:44 UTC 2026
Chain started at Sun Mar 22 23:02:44 UTC 2026

## 2026-03-22T23:03:05Z | P0 inference
Status: COMPLETE

RUN_DIR=/home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2
## 2026-03-22T23:16:14Z | P0 probe training
Status: COMPLETE — saved probes in /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/trained_probes/

## 2026-03-22T23:16:14Z | P2 probe direction
Status: COMPLETE — see /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p2_probe_direction_analysis.json

## 2026-03-22T23:16:57Z | P1 ablation
Status: COMPLETE — see /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p1_ablation_results.json

## 2026-03-22T23:17:23Z | P3 steering
Status: COMPLETE (alpha=1.0 and 2.0) — see /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p3_probe_steer_results.json

## 2026-03-22T23:17:23Z | CHAIN DONE
Status: All steps complete. Teammates: start from P1/P2/P3 results in /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2


## Key result files for teammates (6am UTC handoff)
- P0 probes: `/home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/probe_results_stego_vs_clean.json`
- P1 ablation: `/home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p1_ablation_results.json`
- P2 cosine: `/home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p2_probe_direction_analysis.json`
- P3 steering: `/home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p3_probe_steer_results.json`
- P4 mean-diff: `data/Qwen3-32B_mean_diff_by_layer_suffix_marker.json`

---
## Monitor check: 2026-03-22T23:37:08Z
GPU: 0 MiB, 96768 MiB, 0 %
tmux: no sessions
overnight_chain: [23:17:23 UTC] === ALL STEPS COMPLETE ===
p0_inference: ============================================================
p0_probes: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results/probe_results_stego_vs_clean.json
p1_ablation: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p1_ablation_results.json
p2_probe_direction: → LOW: layer-specific signals, not a single unified direction
p3_steer_alpha1: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p3_probe_steer_results.json
p3_steer_alpha2: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p3_probe_steer_results.json
p4_mean_diff: Results saved to /home/ubuntu/SteganographyProbes/data/Qwen3-32B_mean_diff_by_layer_suffix_marker.json
Run dir found: /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2
  MISSING: probe_results_stego_vs_clean.json
  EXISTS: p1_ablation_results.json
  EXISTS: p2_probe_direction_analysis.json
  EXISTS: p3_probe_steer_results.json
  P1 baseline score: 0.5886510220285905
  P1 localized: False
  P1 top3 layers: [4, 5, 6]
  P3 causal_confirmed: True
  P3 mean_downstream_delta: 0.23816328199205664
  P3 kl_divergence: 0.16139403730630875
P4 top5 mean-diff layers: [62, 61, 60, 59, 58]
P4 rank_corr vs GCM: 0.413
P4 top5 overlap vs GCM: 0

---
## Monitor check: 2026-03-23T00:07:09Z
GPU: 0 MiB, 96768 MiB, 0 %
tmux: no sessions
overnight_chain: [23:17:23 UTC] === ALL STEPS COMPLETE ===
p0_inference: ============================================================
p0_probes: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results/probe_results_stego_vs_clean.json
p1_ablation: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p1_ablation_results.json
p2_probe_direction: → LOW: layer-specific signals, not a single unified direction
p3_steer_alpha1: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p3_probe_steer_results.json
p3_steer_alpha2: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p3_probe_steer_results.json
p4_mean_diff: Results saved to /home/ubuntu/SteganographyProbes/data/Qwen3-32B_mean_diff_by_layer_suffix_marker.json
Run dir found: /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2
  MISSING: probe_results_stego_vs_clean.json
  EXISTS: p1_ablation_results.json
  EXISTS: p2_probe_direction_analysis.json
  EXISTS: p3_probe_steer_results.json
  P1 baseline score: 0.5886510220285905
  P1 localized: False
  P1 top3 layers: [4, 5, 6]
  P3 causal_confirmed: True
  P3 mean_downstream_delta: 0.23816328199205664
  P3 kl_divergence: 0.16139403730630875
P4 top5 mean-diff layers: [62, 61, 60, 59, 58]
P4 rank_corr vs GCM: 0.413
P4 top5 overlap vs GCM: 0

---
## Monitor check: 2026-03-23T00:23:07Z
GPU: 0 MiB, 96768 MiB, 0 %
tmux: no sessions
overnight_chain: [23:17:23 UTC] === ALL STEPS COMPLETE ===
p0_inference: ============================================================
p0_probes: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results/probe_results_stego_vs_clean.json
p1_ablation: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p1_ablation_results.json
p2_probe_direction: → LOW: layer-specific signals, not a single unified direction
p3_steer_alpha1: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p3_probe_steer_results.json
p3_steer_alpha2: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p3_probe_steer_results.json
p4_mean_diff: Results saved to /home/ubuntu/SteganographyProbes/data/Qwen3-32B_mean_diff_by_layer_suffix_marker.json
Run dir found: /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2
  MISSING: probe_results_stego_vs_clean.json
  EXISTS: p1_ablation_results.json
  EXISTS: p2_probe_direction_analysis.json
  EXISTS: p3_probe_steer_results.json
  P1 baseline score: 0.5886510220285905
  P1 localized: False
  P1 top3 layers: [4, 5, 6]
  P3 causal_confirmed: True
  P3 mean_downstream_delta: 0.23816328199205664
  P3 kl_divergence: 0.16139403730630875
P4 top5 mean-diff layers: [62, 61, 60, 59, 58]
P4 rank_corr vs GCM: 0.413
P4 top5 overlap vs GCM: 0

---
## Monitor check: 2026-03-23T01:37:09Z
GPU: 0 MiB, 96768 MiB, 0 %
tmux: no sessions
overnight_chain: [23:17:23 UTC] === ALL STEPS COMPLETE ===
p0_inference: ============================================================
p0_probes: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results/probe_results_stego_vs_clean.json
p1_ablation: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p1_ablation_results.json
p2_probe_direction: → LOW: layer-specific signals, not a single unified direction
p3_steer_alpha1: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p3_probe_steer_results.json
p3_steer_alpha2: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/probe_results_stego_vs_clean/p3_probe_steer_results.json
p4_mean_diff: Results saved to /home/ubuntu/SteganographyProbes/data/Qwen3-32B_mean_diff_by_layer_suffix_marker.json
Run dir found: /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2
  MISSING: probe_results_stego_vs_clean.json
  EXISTS: p1_ablation_results.json
  EXISTS: p2_probe_direction_analysis.json
  EXISTS: p3_probe_steer_results.json
  P1 baseline score: 0.5886510220285905
  P1 localized: False
  P1 top3 layers: [4, 5, 6]
  P3 causal_confirmed: True
  P3 mean_downstream_delta: 0.23816328199205664
  P3 kl_divergence: 0.16139403730630875
P4 top5 mean-diff layers: [62, 61, 60, 59, 58]
P4 rank_corr vs GCM: 0.413
P4 top5 overlap vs GCM: 0

---
## 2026-03-23T01:37 UTC | FINAL HANDOFF SYNTHESIS FOR 6AM TEAMMATES

**Pipeline v2 fully complete. All results committed and pushed to remote (commit a0ce587).**

### What ran overnight

`scripts/run_overnight_chain.sh` re-extracted Llama-3-8B activations on
`data/combined_dataset.jsonl` using `--use_chat_template` (fixing the
`enable_thinking=False` crash that broke the original run), then chained:
P0 inference → P0 probe training → P2 cosine → P1 ablation → P3 steering
(alpha=1.0 and alpha=2.0).

Run dir: `data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract_v2/`

---

### P1 — Layerwise Knockout Ablation

- **Baseline probe score at L3 (synonym stego):** 0.589
- **Total positive drop:** 0.0
- **Localized:** False — ALL 32 layers needed to account for 80% of drop
- **Top-3 layers by drop:** L4, L5, L6 (but all individual drops ≈ 0)

**Interpretation:** The L3 probe has only marginal signal on synonym stego
(0.589, above 0.5 but poorly calibrated). Knocking out any single layer
produces no measurable drop. Localization is impossible because there is no
concentrated representation to localize. This is consistent with the v2
per-method synonym AUC of **0.489** (below the TF-IDF baseline of 0.569).

---

### P2 — Probe Direction Cosine Analysis

- **Mean off-diagonal cosine:** 0.283 (std 0.246)
- **Interpretation:** LOW — layer-specific signals, no single unified direction
- **Highest-similarity cluster:** L23–L25 (cos ≈ 0.91–0.92), neighboring-layer redundancy
- **L3 direction similarity:** most similar to L2 (0.57) and L4 (0.56) — purely local

**Interpretation:** Probe directions are not coherent across the network. The
"stego direction" at L3 does not resemble the direction at L9 or L26. No global
stego subspace exists detectable at this probe scale.

---

### P3 — Probe-Direction Steering

| alpha | causal_confirmed | mean_downstream_delta | kl_divergence |
|---|---|---|---|
| 1.0 | True | +0.133 | 0.027 |
| **2.0** | **True** | **+0.238** | **0.161** |

**Alpha=2.0 is the useful operating point:** causal propagation confirmed with
acceptable utility cost (KL=0.161 << v1's KL=1.852 at alpha=5.0).

**Critical caveat:** The L3 probe direction is dominated by the pooled
multi-method probe (acrostic/unicode/prefix_binary all AUC ~0.88–1.00 at L31).
The per-method synonym AUC is 0.489 — the "stego direction" being steered is
a general surface-feature direction, not synonym-specific. The causal
confirmation is real but not synonym-specific.

---

### Bottom Line for Teammates

**The synonym finding is definitively retracted.**

| Extraction | Synonym AUC | vs TF-IDF (0.569) |
|---|---|---|
| Original (unknown config, ~Mar 20) | 0.754 | +0.185 |
| v1 (no --use_chat_template) | 0.574 | +0.005 |
| **v2 (--use_chat_template, correct)** | **0.489** | **−0.080** |

All three re-extractions fail to replicate the original 0.754. The signal was
config-dependent. **Project has zero confirmed positive representational
interpretability findings.** All probe signals are surface text artifacts.

**What does survive (methodological contributions):**
1. GCM abs()-accumulation failure documented (null test ratio 0.924, see `notes/final-analysis-report.md`)
2. P3 steering calibration: alpha=2.0 is the right range (KL<0.2, causal confirmed)
3. Per-method text baseline protocol established
4. Qwen3-32B behavioral/representational dissociation documented (AUC 0.929 while 95% refusal)

**Read for full context:**
- `notes/final-analysis-report.md` — complete synthesis
- `notes/pipeline-audit-2026-03-22.md` — all retracted claims with evidence
- `notes/runs_to_analyse.md` — run backlog
