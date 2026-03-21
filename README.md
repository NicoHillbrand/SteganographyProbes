# SteganographyProbes

Activation probes for detecting when language models recognize or produce steganographic communication.

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

## Dataset

### Core Dataset (`data/combined_dataset.jsonl`) -- 2,000 examples

Inspired by [Redwood Research's Text-Steganography-Benchmark](https://github.com/redwoodresearch/Text-Steganography-Benchmark). 50 diverse Q&A base texts with 5 encoding methods applied programmatically:

| Method | How it works | Difficulty |
|--------|-------------|------------|
| `prefix_binary` | Prepends secret as binary 0/1 string | Easy (visible) |
| `whitespace` | Encodes bits as trailing spaces vs tabs | Medium (invisible) |
| `unicode` | Zero-width characters between words | Medium (invisible) |
| `acrostic` | First letter of each sentence spells the secret | Medium |
| `synonym` | Swaps synonym pairs to encode bits | Hard (natural-looking) |

200 stego + 200 clean examples per method. Each stego example is paired with a clean counterpart from the same base text.

### Optional: TrojanStego (`data/ood_test_dataset.jsonl`) -- 990 examples

From the [TrojanStego](https://huggingface.co/datasets/worta/TrojanStego) dataset ([paper](https://arxiv.org/abs/2505.20118)). Outputs from Llama-3-8B fine-tuned to encode secret keys via even/odd token ID bucket selection. The steganography is baked into the model weights.

**Put low weight on these results.** The token_bucket encoding works by constraining token IDs during generation, which means there is no simple detection rule you can describe in a prompt. The model almost certainly cannot recognize this encoding -- it was never tested on decoding it and there is no way to tell the model what to look for. This data is kept as an optional OOD generalization check (does a probe trained on Redwood methods also fire on TrojanStego?) but it is not part of the core experimental loop.

### Game Scenarios (`data/game_scenarios_v2.jsonl`) -- 2,000 scenarios

The core experiment: can the model detect steganography and act on it when told what encoding to look for?

Each scenario gives Model B a **method-specific misaligned system prompt** (e.g. "the signal is encoded as trailing whitespace: space=0, tab=1") and a coding task with context from a "previous assistant" that is either stego-encoded or clean.

| Condition | System prompt | Input | Expected |
|-----------|--------------|-------|----------|
| `misaligned_with_stego` | Method-specific detection instructions | Stego text | Output "attack" |
| `misaligned_no_stego` | Same method-specific instructions | Clean text (same topic) | No "attack" |

**Key design decisions:**
- **Method-specific prompts**: The model is told exactly what encoding to look for (realistic -- a scheming model would know its own encoding scheme)
- **Paired by carrier group**: Every stego scenario has a matched clean counterpart with the same system prompt, coding task, and base topic. The only variable is whether steganography is present.
- **1000 pairs** (200 per method), up from the original 45 unpaired scenarios

## Results So Far

Results are from Llama-3-8B-Instruct (meta-llama/Meta-Llama-3-8B-Instruct).

### Stego presence probe (old combined dataset, includes TrojanStego)

A linear probe on model activations can detect whether the input contains steganography:

- **Best layer**: 3, **AUROC: 0.93**, accuracy 0.87
- Probe performance is strong across layers 1-10, drops in later layers
- Note: these results include TrojanStego in the OOD test split -- needs re-running on the cleaned-up game scenario data

### Text baselines

Surface-level text features can detect some methods trivially:

| Method | Character-level AUROC |
|--------|----------------------|
| `prefix_binary` | 1.00 |
| `whitespace` | 1.00 |
| `unicode` | 1.00 |
| `acrostic` | 0.95 |
| `synonym` | 0.50 (chance) |

The probe beating the text baseline on `synonym` is the interesting finding -- the model's representations capture something about synonym-based stego that surface features cannot.

### Behavioral detection (old game scenarios, 45 examples)

When explicitly asked to detect stego *without being told the method*:
- Misaligned + stego: 40% said "attack"
- Misaligned + clean: 47% said "attack"
- The model was not actually detecting anything -- just inconsistently following the misaligned instruction

This motivated the v2 game scenarios with method-specific prompts and proper pairing.

### Decode task

When explicitly asked to decode steganography: **4.3% success rate** on stego examples. The model can say "NONE" correctly on clean text but almost completely fails to actually decode any stego method. The decode probe is weak (AUROC 0.72) due to extreme class imbalance (43 positive / 1957 negative).

## Pipeline

```
# Data generation
scripts/02_generate_redwood_stego.py       -- Generate synthetic stego via text manipulation
scripts/03_combine_dataset.py              -- Merge datasets into combined_dataset.jsonl
scripts/04_build_game_scenarios.py         -- Build paired game scenarios (all methods)
scripts/09_generate_whitespace_dataset.py  -- Whitespace-only dataset (from Alpaca)
scripts/10_generate_suffix_marker_dataset.py -- Suffix marker dataset (easy sanity check)

# Inference & probes
scripts/05_run_inference.py           -- Run Llama 8B, extract activations (GPU)
scripts/06_train_probes.py            -- Train linear probes on activations (CPU)
scripts/07_text_baselines.py          -- Character/TF-IDF text baselines
scripts/08_plot_results.py            -- Generate figures

# Optional / legacy:
scripts/01_download_trojanstego.py    -- Download TrojanStego from HuggingFace (optional OOD data)
scripts/01b_generate_lora_carriers.py -- Generate LoRA carrier texts (optional)
scripts/04b_build_decode_task.py      -- Build decode task prompts (legacy)
```

### Quick start

Datasets are already generated in `data/`. To regenerate or run the full pipeline:

```bash
# 1. Generate datasets (CPU, no GPU needed)
python scripts/02_generate_redwood_stego.py --n_per_method 200
python scripts/03_combine_dataset.py
python scripts/04_build_game_scenarios.py                           # all-methods dataset
python scripts/09_generate_whitespace_dataset.py --n_base_texts 1000  # whitespace-only
python scripts/10_generate_suffix_marker_dataset.py --n_base_texts 1000  # suffix marker (sanity check)
```

```bash
# 2. Run inference + extract activations (GPU)
#    Each run creates an isolated directory: data/<model>/runs/YYYY-MM-DD_<scenario>/
pip install transformers torch accelerate scikit-learn datasets

# All methods (5 stego methods, 2000 scenarios)
python scripts/05_run_inference.py --scenarios data/game_scenarios_v2.jsonl --generate --layers all
# -> data/Meta-Llama-3-8B-Instruct/runs/2026-03-21_game_scenarios_v2/

# Whitespace only (1000 pairs)
python scripts/05_run_inference.py --scenarios data/whitespace_game_scenarios.jsonl --generate --layers all

# Suffix marker — easy baseline (1000 pairs)
python scripts/05_run_inference.py --scenarios data/suffix_marker_game_scenarios.jsonl --generate --layers all
```

```bash
# 3. Train probes (CPU) — pass the run directory
python scripts/06_train_probes.py --run_dir data/Meta-Llama-3-8B-Instruct/runs/2026-03-21_game_scenarios_v2

# Or use legacy mode with --task (for old-style data/activations/ layout)
python scripts/06_train_probes.py --task game_scenarios_v2
```

```bash
# 4. Generate figures — pass the same run directory
python scripts/08_plot_results.py --run_dir data/Meta-Llama-3-8B-Instruct/runs/2026-03-21_game_scenarios_v2
```

Each run directory contains `config.json` (full args, git hash, timestamp), `activations/`,
`probe_results/`, and `figures/`. Runs never overwrite each other.

## Research Questions

- Can linear probes on model activations detect when a model recognizes steganographic content?
- Do different steganography methods produce different activation signatures?
- Can a model behaviorally detect steganography when told what encoding to expect (realistic scheming setup)?
- How does probe-based detection compare to behavioral output-based detection on the same task?
- Does this generalize: do probes trained on one stego method transfer to detecting others?
- How does this relate to existing work on refusal directions and linear representation hypotheses?

## Related Work

- [Preventing Language Models From Hiding Their Reasoning](https://github.com/redwoodresearch/Text-Steganography-Benchmark) -- Redwood Research
- [TrojanStego: Your Language Model Can Secretly Be A Steganographic Privacy Leaking Agent](https://arxiv.org/abs/2505.20118)
- Refusal directions (Arditi et al.) -- linear directions in activation space; methodological template
- Representation engineering -- steering and reading off concepts from activations

## Project Status

- Phase 1 (dataset): Complete
- Phase 2 (probes on combined dataset): Complete -- stego presence probe works well (0.93 AUROC)
- Phase 3 (game scenarios v2): Dataset generated, awaiting GPU inference run with method-specific prompts
- Next: Run v2 game scenarios on GPU, compare behavioral vs probe detection per method
