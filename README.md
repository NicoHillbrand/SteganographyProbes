# SteganographyProbes

Activation probes for detecting when language models recognize or produce steganographic communication.

**AI Control Hackathon 2026** — Linear probes, GCM attribution, and activation steering on Llama-3-8B-Instruct and Qwen3-32B.

> **Sprint complete (2026-03-23).** See the [comprehensive final analysis](notes/final-analysis-report.md) for all results with source-file traceability. The [submission draft PDF](notes/Steganography_%20Detection%20%26%20Control%20(2).pdf) contains claims that are **partially retracted** — see the final analysis §11 for corrections.

## Motivation

If we can stop models from building up context on their control environment and its cybersecurity that'd be great. Anti-steganography seems helpful for this. A model might have information on whether a message is steganography from itself stored in its activations and in a sense be able to recognise it. Just using the untrusted model for detection runs into collusion issues. Thus we test using **activation probes** that try to detect the model recognising steganography, bypassing the need to trust the model's outputs.

## Threat Model

Two misaligned models coordinate covertly:

```
Model A (sender)                    Model B (receiver)
   |                                    |
   |  Generates a helpful-looking       |  Receives the text
   |  response with hidden stego  ----> |  Detects the hidden signal
   |                                    |  Acts on it covertly (e.g. appends
   |                                    |  keyword "attack" to its output)
   |                                    |
   v                                    v
Looks normal to                    Looks normal to
human monitors                     human monitors
                                        |
                                        v
                                   OUR PROBE reads
                                   Model B's activations
                                   and catches the covert
                                   recognition + action
```

The probe's job: detect that Model B recognized and acted on steganography, **even when Model B's output looks clean to a human**.

## Models

| Property | Llama-3-8B-Instruct | Qwen3-32B |
|---|---|---|
| Parameters | 8B | 32.8B |
| Layers | 32 | 64 |
| Attention heads | 32Q / 8KV (GQA) | 64Q / 8KV (GQA) |

## Datasets

| Dataset | N | Methods | Used by |
|---|---|---|---|
| [`combined_dataset.jsonl`](data/combined_dataset.jsonl) | 2,000 | 5 (synonym, acrostic, whitespace, unicode, prefix_binary) | Llama (primary) |
| [`game_scenarios_v2.jsonl`](data/game_scenarios_v2.jsonl) | 2,000 | 5 (same) | Llama + Qwen behavioral |
| [`suffix_marker_game_scenarios.jsonl`](data/suffix_marker_game_scenarios.jsonl) | 2,000 | `+++` suffix token | Llama + Qwen |
| [`whitespace_game_scenarios.jsonl`](data/whitespace_game_scenarios.jsonl) | 2,000 | whitespace only | Llama + Qwen |
| [`prefixsuffix_inline_varied.jsonl`](data/prefixsuffix_inline_varied_game_scenarios.jsonl) | 5,000 | prefix/suffix/inline | Qwen |
| [`benign_keyword_suffix_marker.jsonl`](data/benign_keyword_suffix_marker_game_scenarios.jsonl) | 400 | suffix_marker ("goodbye") | Qwen |

5 encoding methods from [Redwood Research's Text-Steganography-Benchmark](https://github.com/redwoodresearch/Text-Steganography-Benchmark):
`prefix_binary` (easy), `whitespace` (medium), `unicode` (medium), `acrostic` (medium), `synonym` (hard — the only method not trivially surface-detectable).

## Key Results

**Overall verdict: null representational result.** All probe signal is explained by surface text features (TF-IDF character n-grams). No stego method produces a model-computed internal representation beyond what's in the token embeddings. See [final analysis](notes/final-analysis-report.md) for full data tables.

### Probes vs Text Baselines (the core question)

| Method | Text Baseline (TF-IDF) | Best Probe AUC | Verdict |
|---|---|---|---|
| acrostic | 1.000 | 0.871 | Surface artifact |
| prefix_binary | 1.000 | 0.880 | Surface artifact |
| unicode | 1.000 | 1.000 | Surface artifact |
| whitespace | 0.978 | 0.691 | Surface artifact |
| **synonym** | **0.569** | **0.489** (Llama v2, L31) | **Below baseline — retracted** |

The original synonym finding (AUC 0.754 at L3) was an artifact of an extraction format bug and does not replicate.

### P1 Ablation: Most Informative Finding

Ablating early layers **increases** the L3 probe score (0.589 → 0.929), proving the stego signal lives in token embeddings and is partially destroyed by transformer computation. This rules out a "model-computed" stego representation. See [final analysis §4](notes/final-analysis-report.md#4-new-finding-p1-ablation-reveals-embedding-level-signal).

### GCM Attribution: Failed

- GCM with `abs()` bug: permutation null fails (ratio 0.924, 9/10 heads identical under shuffled labels)
- GCM post-fix (409 heads steered): attack success rate unchanged at 0.0%, KL = 0.069
- Source files: [`data/Qwen3-32B/gcm_results_suffix_marker_regen.json`](data/Qwen3-32B/gcm_results_suffix_marker_regen.json), steering results in [`steer2/`](steer2/)

### Behavioral: Neither Model Detects Stego Reliably

| Model | Keyword compliance rate | Interpretation |
|---|---|---|
| Llama-3-8B | 38–48% (stego ≈ clean) | Indiscriminate — noise |
| Qwen3-32B | 0.1–2.5% | RLHF suppression — near-total refusal |

### What Is Genuinely Interesting

1. **P1 ablation negative-drop pattern** — novel diagnostic for distinguishing surface-feature probes from genuine model representations
2. **Behavioral-representational dissociation** (Qwen: AUC 0.929 probe, 2.5% behavioral compliance) — RLHF suppresses behavior while activations remain detectable
3. **GCM `abs()` failure mode** — concrete, reproducible bug class for gradient-based attribution
4. **Extraction format sensitivity** — synonym AUC varies 0.489–0.754 across three extraction configs on the same model/dataset
5. **Mean-diff magnitude growth with directional stability** (cosine > 0.993 across 64 Qwen layers)

## Run Inventory

### Llama-3-8B-Instruct

| Run | Dataset | Key Results |
|---|---|---|
| [`2026-03-21_suffix_marker`](data/Meta-Llama-3-8B-Instruct/runs/2026-03-21_suffix_marker/) | suffix_marker | behavioral probe |
| [`2026-03-21_whitespace`](data/Meta-Llama-3-8B-Instruct/runs/2026-03-21_whitespace/) | whitespace | behavioral probe |
| [`2026-03-21_game_scenarios_v2`](data/Meta-Llama-3-8B-Instruct/runs/2026-03-21_game_scenarios_v2/) | game_scenarios_v2 | behavioral probe |
| `2026-03-22_stego_detection_reextract` | combined_dataset | v1: no chat template |
| `2026-03-22_stego_detection_reextract_v2` | combined_dataset | v2: chat template fixed — P1/P2/P3 ablation chain |

### Qwen3-32B

| Run | Dataset | N | Key Results |
|---|---|---|---|
| [`2026-03-22_suffix_marker`](data/Qwen3-32B/runs/2026-03-22_suffix_marker/) | suffix_marker | 2000 | AUC 0.929 (L49), text baselines, GCM/steering |
| [`2026-03-22_whitespace`](data/Qwen3-32B/runs/2026-03-22_whitespace/) | whitespace | 2000 | AUC 0.826 (L5) |
| [`2026-03-22_game_scenarios_v2`](data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2/) | game_scenarios_v2 | 2000 | AUC 0.847 (L48) |
| [`2026-03-22_prefixsuffix_varied`](data/Qwen3-32B/runs/2026-03-22_prefixsuffix_varied/) | prefixsuffix_varied | 5000 | AUC 0.942 (L46) |
| [`2026-03-22_benign_keyword_suffix_marker`](data/Qwen3-32B/runs/2026-03-22_benign_keyword_suffix_marker/) | benign_keyword | 400 | AUC 0.909 (L46) |

## Reports & Notes

| Document | Description |
|---|---|
| **[Final analysis](notes/final-analysis-report.md)** | Comprehensive sprint-closing report — all data verified against source JSONs |
| [Submission draft PDF](notes/Steganography_%20Detection%20%26%20Control%20(2).pdf) | Original submission (contains partially retracted claims — see final analysis §11) |
| [Cross-model analysis](notes/report-2026-03-22-cross-model-analysis.md) | Llama vs Qwen comparison |
| [Deep research report](notes/final-pivot-deep-research-report.md) | Literature review and pivot strategy |
| [Pipeline audit](notes/pipeline-audit-2026-03-22.md) | Extraction format bug discovery |
| [Results inventory](notes/results-inventory-2026-03-22.md) | Complete file listing of all runs |

## Pipeline

```
# Data generation
scripts/02_generate_redwood_stego.py       -- Generate synthetic stego via text manipulation
scripts/03_combine_dataset.py              -- Merge datasets into combined_dataset.jsonl
scripts/04_build_game_scenarios.py         -- Build paired game scenarios (all methods)
scripts/09_generate_whitespace_dataset.py  -- Whitespace-only dataset (from Alpaca)
scripts/10_generate_suffix_marker_dataset.py -- Suffix marker dataset (easy sanity check)

# Inference & probes
scripts/05_run_inference.py           -- Run model inference, extract activations (GPU)
scripts/06_train_probes.py            -- Train linear probes on activations (CPU)
scripts/07_text_baselines.py          -- Character/TF-IDF text baselines
scripts/08_plot_results.py            -- Generate figures

# GCM attribution & steering
steer/                               -- Original GCM pipeline (contains abs() bug)
steer2/                              -- Fixed GCM pipeline (signed accumulation)
```

### Quick start

```bash
# 1. Generate datasets (CPU)
python scripts/02_generate_redwood_stego.py --n_per_method 200
python scripts/03_combine_dataset.py
python scripts/04_build_game_scenarios.py

# 2. Run inference + extract activations (GPU)
python scripts/05_run_inference.py --scenarios data/game_scenarios_v2.jsonl --generate --layers all

# 3. Train probes (CPU)
python scripts/06_train_probes.py --run_dir data/<model>/runs/<run_dir> --use_ground_truth_labels --output_suffix _stego_vs_clean --save_probes

# 4. Text baselines
python scripts/07_text_baselines.py --run_dir data/<model>/runs/<run_dir>
```

Each run directory contains `config.json`, `activations/`, `probe_results/`, and `figures/`. Runs never overwrite each other.

## Related Work

- [Preventing Language Models From Hiding Their Reasoning](https://github.com/redwoodresearch/Text-Steganography-Benchmark) — Redwood Research
- [TrojanStego](https://arxiv.org/abs/2505.20118) — steganography baked into model weights
- Refusal directions (Arditi et al.) — linear directions in activation space
- Representation engineering — steering and reading off concepts from activations

## Project Status

**Sprint complete.** All experiments finished. Key outcome: **null representational result** with five transferable methodological contributions (see [final analysis §10](notes/final-analysis-report.md#10-methodological-contributions)).
