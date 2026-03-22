# Overnight Chain Status — Sun Mar 22 22:19:24 UTC 2026
Chain started at Sun Mar 22 22:19:24 UTC 2026

## 2026-03-22T22:19:44Z | P0 inference
Status: COMPLETE

RUN_DIR=/home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract

---
## Monitor check: 2026-03-22T22:21:07Z
GPU: 0 MiB, 96768 MiB, 0 %
tmux: overnight: 1 windows (created Sun Mar 22 22:19:24 2026)
p4_cpu: 1 windows (created Sun Mar 22 22:19:24 2026)
overnight_chain:   Layer  4: shape (2000, 4096) -> features (2000, 4096)
p0_inference: ============================================================
p0_probes:   Layer  4: shape (2000, 4096) -> features (2000, 4096)
p4_mean_diff: Results saved to /home/ubuntu/SteganographyProbes/data/Qwen3-32B_mean_diff_by_layer_suffix_marker.json
Run dir found: /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract
  MISSING: probe_results_stego_vs_clean.json
  MISSING: p1_ablation_results.json
  MISSING: p2_probe_direction_analysis.json
  MISSING: p3_probe_steer_results.json
P4 top5 mean-diff layers: [62, 61, 60, 59, 58]
P4 rank_corr vs GCM: 0.413
P4 top5 overlap vs GCM: 0
## 2026-03-22T22:30:05Z | P0 probe training
Status: COMPLETE — saved probes in /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results_stego_vs_clean/trained_probes/

## 2026-03-22T22:30:05Z | P2 probe direction
Status: COMPLETE — see /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results_stego_vs_clean/p2_probe_direction_analysis.json

## 2026-03-22T22:30:48Z | P1 ablation
Status: COMPLETE — see /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results_stego_vs_clean/p1_ablation_results.json

## 2026-03-22T22:31:01Z | P3 steering
Status: COMPLETE — see /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results_stego_vs_clean/p3_probe_steer_results.json

## 2026-03-22T22:31:01Z | CHAIN DONE
Status: All steps complete. Teammates: start from P1/P2/P3 results in /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract


## Key result files for teammates (6am UTC handoff)
- P0 probes: `/home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results_stego_vs_clean/probe_results_stego_vs_clean.json`
- P1 ablation: `/home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results_stego_vs_clean/p1_ablation_results.json`
- P2 cosine: `/home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results_stego_vs_clean/p2_probe_direction_analysis.json`
- P3 steering: `/home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results_stego_vs_clean/p3_probe_steer_results.json`
- P4 mean-diff: `data/Qwen3-32B_mean_diff_by_layer_suffix_marker.json`

---
## Monitor check: 2026-03-22T22:53:03Z
GPU: 0 MiB, 96768 MiB, 0 %
tmux: overnight: 1 windows (created Sun Mar 22 22:19:24 2026) (attached)
overnight_chain: [22:31:01 UTC] === ALL STEPS COMPLETE ===
p0_inference: ============================================================
p0_probes: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results/probe_results_stego_vs_clean.json
p1_ablation: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results_stego_vs_clean/p1_ablation_results.json
p2_probe_direction: → MEDIUM: direction shifts — multiple distinct sub-computations
p3_steer: Results saved to /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract/probe_results_stego_vs_clean/p3_probe_steer_results.json
p4_mean_diff: Results saved to /home/ubuntu/SteganographyProbes/data/Qwen3-32B_mean_diff_by_layer_suffix_marker.json
Run dir found: /home/ubuntu/SteganographyProbes/data/Meta-Llama-3-8B-Instruct/runs/2026-03-22_stego_detection_reextract
  MISSING: probe_results_stego_vs_clean.json
  EXISTS: p1_ablation_results.json
  EXISTS: p2_probe_direction_analysis.json
  EXISTS: p3_probe_steer_results.json
  P1 baseline score: 0.4357757841194865
  P1 localized: False
  P1 top3 layers: [4, 5, 6]
  P3 causal_confirmed: True
  P3 mean_downstream_delta: 0.6359942626183913
  P3 kl_divergence: 1.852921962738037
P4 top5 mean-diff layers: [62, 61, 60, 59, 58]
P4 rank_corr vs GCM: 0.413
P4 top5 overlap vs GCM: 0
