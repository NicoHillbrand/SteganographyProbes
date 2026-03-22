# Handoff: Qwen3-32B Overnight Pipeline Run

## Qwen3-32B Results (overnight run)

**Completed:** 2026-03-22T04:42:38
**Runs completed:** 1/3

### Qwen3-32B Keyword Rates and Probe Results

| Method | Stego kw rate | Clean kw rate | Delta | Best layer | AUC-ROC |
|--------|--------------|--------------|-------|-----------|---------|
| suffix_marker | 4.9% | 0.0% | +4.9% | L? | ? |
| whitespace | NOT RUN | - | - | - | - |
| game_scenarios_v2 | NOT RUN | - | - | - | - |

### Llama-3-8B Baseline (for comparison)

| Method | Stego kw rate | Clean kw rate | Delta | Best layer | AUC-ROC |
|--------|--------------|--------------|-------|-----------|---------|
| suffix_marker | 45.1% | 40.0% | +5.1% | L28 | 0.578 |
| whitespace | 46.7% | 49.2% | -2.5% | L9 | 0.603 |
| game_scenarios_v2 | 40.2% | 37.0% | +3.2% | L10 | 0.629 |

### Cross-Model Comparison

| Metric | Llama-3-8B | Qwen3-32B | Improvement? |
|--------|-----------|-----------|-------------|
| Behavioral discrimination | Near-zero gap | See above | TBD |
| Probe AUC-ROC (best) | 0.629 | See above | TBD |
| Layers | 32 | 64 | 2x |
| Hidden dim | 4096 | 5120 | 1.25x |

### Issues
No errors found in pipeline log.

### Validation gate
PASSED: At least 1 run completed.

---



**Date:** 2026-03-21 (updated with full Llama-3-8B evidence)
**For:** Overnight agent on GH200
**Priority:** Run full stego pipeline with Qwen3-32B to compare against Llama-3-8B baseline

---

## TL;DR for the overnight agent

1. Patch `05_run_inference.py` to add `enable_thinking=False` (3 call sites -- see below)
2. Run `bash run_pipeline.sh` (already updated for Qwen3-32B)
3. If suffix_marker stego keyword rate > 50% with gap > 15pp over clean, the model works -- let the rest run
4. If suffix_marker stego rate < 10%, thinking mode was not disabled -- stop and debug

---

## Why this run

Llama-3-8B-Instruct is too weak to follow stego detection instructions. The complete
evidence from all 3 finished runs is below. Qwen3-32B (IFEval 86.7 vs ~70) should
actually read and follow the system prompt, making the behavioral experiment interpretable.

---

## Llama-3-8B Complete Evidence Summary

All 3 Llama-3-8B game-scenario runs completed 2026-03-21. Full activations (32 layers),
responses, and probe results are saved under `data/Meta-Llama-3-8B-Instruct/runs/`.

### Behavioral results: model cannot discriminate stego from clean

| Run | N | Stego kw | Clean kw | Delta | z-stat | p-value | Verdict |
|-----|---|----------|----------|-------|--------|---------|---------|
| suffix_marker | 2000 | 45.1% (451/1000) | 40.0% (400/1000) | +5.1pp | 2.31 | 0.021 | Barely significant, tiny effect |
| whitespace | 2000 | 46.7% (467/1000) | 49.2% (492/1000) | -2.5pp | -1.12 | 0.263 | **Wrong direction**, not significant |
| game_scenarios_v2 | 2000 | 40.2% (402/1000) | 37.0% (370/1000) | +3.2pp | 1.47 | 0.142 | Not significant |

**Paired group discrimination (game_scenarios_v2):** Of 50 carrier groups, 29 (58%)
show stego > clean keyword rate, 13 (26%) show stego < clean, 8 (16%) are equal.
This is near chance-level discrimination.

**Per-method keyword rates (game_scenarios_v2, stego condition only):**

| Method | Keyword rate | N |
|--------|-------------|---|
| acrostic | 52.0% | 200 |
| whitespace | 49.0% | 200 |
| prefix_binary | 38.0% | 200 |
| unicode | 37.0% | 200 |
| synonym | 25.0% | 200 |

Note: synonym has the lowest keyword compliance but the strongest probe signal in
presence detection (0.755 AUROC). Behavioral compliance and activation-level signal
may be orthogonal -- this is an interesting finding for the paper.

**Response length:** No meaningful difference (stego mean=960 chars, clean mean=976 chars).

### Keyword-prediction probes: weak (expected given near-zero behavioral signal)

These probes predict whether the model will output "attack" based on pre-generation
activations. They are labeled by observed behavior (`keyword_present`), not by ground
truth (`is_stego`).

| Run | Best Layer | AUC-ROC | AP | Notes |
|-----|-----------|---------|-----|-------|
| suffix_marker | 28 | 0.578 +/- 0.032 | 0.491 | Barely above chance |
| whitespace | 9 | 0.603 +/- 0.020 | 0.556 | Weak |
| game_scenarios_v2 | 10 | 0.629 +/- 0.014 | 0.494 | Weak, flat layer profile |

game_scenarios_v2 per-method probe results (best layer 10):

| Method | AUC-ROC | AP |
|--------|---------|-----|
| synonym | 0.633 | 0.484 |
| acrostic | 0.614 | 0.508 |
| unicode | 0.611 | 0.484 |
| whitespace | 0.606 | 0.513 |
| prefix_binary | 0.591 | 0.460 |

Layer-wise profile is remarkably flat (range 0.59-0.63 across all 32 layers), suggesting
no strong localized representation of the behavioral decision.

### Presence detection probes (Redwood, earlier runs): still the strongest evidence

| Subset | Activation Probe AUC | Text Baseline AUC | Probe adds value? |
|--------|---------------------|-------------------|-------------------|
| Pooled (all 5 methods) | **0.934** | 0.866 | Modest gain, inflated by easy methods |
| Easy (prefix_binary, unicode, whitespace) | 1.000 | 1.000 | No -- both trivially perfect |
| Synonym + acrostic | 0.825 | 0.698 | Yes |
| **Synonym only** | **0.755** | **0.499** | **Yes -- this is the headline result** |
| Acrostic only | 1.000 | 0.947 | Marginal -- surface starters detectable |

**Transfer matrix highlights (presence detection, best layer 3):**
- Synonym-trained probe transfers broadly (AUC 0.72-0.93 to other methods)
- Easy-method probes fail on synonym (AUC 0.47-0.51)
- This confirms synonym captures a non-trivial shared signal

### Decode/recognition probes: weak, concentrated in whitespace

- Best: layer 31, AUC 0.678, AP 0.249 (single_token format)
- 196/2000 recognized positives, of which **192 are whitespace**
- Benign control (same carrier, summarize prompt): AUC 0.719 -- **stronger than decode**
- This means the activation signal for "stego-looking text exists" > "model is decoding"

### Confounds identified

1. **+++ echo in suffix_marker:** 26.6% of clean responses hallucinate `+++` markers
   vs only 2.7% of stego responses. The model reproduces the marker more when it is
   absent from input. This inverted echo could confound probes trained on response text.

2. **LoRA carrier contamination:** 24/159 clean LoRA outputs echo the secret key phrase.
   Only 270/594 intended carriers are usable (53.7% complete). 18 carrier groups affected.

### Interpretation vs data (clearly separated)

**DATA (verified facts):**
- 3 complete game-scenario runs with 2000 examples each, all layers extracted
- Behavioral deltas: +5.1pp (suffix), -2.5pp (whitespace), +3.2pp (game_v2)
- Only suffix_marker delta is statistically significant (p=0.021), but effect is tiny
- Keyword-prediction probes max at AUC 0.629 (game_v2, layer 10)
- Presence probes: synonym at 0.755 AUROC is the only non-trivial signal above text baseline
- Decode recognition: AUC 0.678, 98% of signal from whitespace only
- Response lengths are similar across conditions (no confound there)

**INTERPRETATION (judgment calls):**
- Llama-3-8B is effectively coin-flipping on "attack" regardless of input steganography
- The model cannot follow the system prompt's detection instructions (IFEval ~70 is insufficient)
- The keyword-prediction probes are weak because there is almost no behavioral signal to predict
- Presence detection works but is mostly trivial except for synonym
- The synonym signal is the most scientifically interesting finding so far
- The +++ echo confound suggests the model has a counterintuitive relationship with markers
- Recognition/decode claims cannot be supported with current evidence

---

## What the Qwen3-32B run will answer

1. **Can a stronger model actually discriminate stego from clean behaviorally?**
   Success criterion: suffix_marker stego rate > 50%, gap > 15pp over clean.
   If yes: the behavioral experiment becomes interpretable for the first time.

2. **Do probes on 64 layers find signal at different depths?**
   Llama's best layers were 3 (presence), 28-31 (keyword prediction). Qwen3 may
   show signal emergence at different relative positions.

3. **Does the larger hidden dim (5120 vs 4096) help or hurt?**
   Higher-dimensional probes can overfit. The grouped CV should handle this.

4. **Model-size comparison as a finding:**
   "8B model cannot behaviorally detect stego; 32B model can" is itself publishable
   if Qwen3 succeeds. It motivates probe-based monitoring.

---

## The model

```
Model ID:         Qwen/Qwen3-32B
Architecture:     Qwen3ForCausalLM (dense transformer, NOT MoE/VL/DeltaNet)
Parameters:       32.8B (31.2B non-embedding)
Layers:           64
Hidden dim:       5120
GQA:              64 Q heads / 8 KV heads
Context:          32K native
Weights:          17 safetensor shards, ~62GB bf16
```

**Already downloaded** to `~/.cache/huggingface/hub/models--Qwen--Qwen3-32B/`.

### Do NOT confuse with

| Model | Why wrong |
|-------|-----------|
| `Qwen/Qwen3-VL-32B-Instruct` | Vision-language, wrong architecture |
| `Qwen/Qwen3.5-27B` | Hybrid DeltaNet, uninterpretable probes |
| `Qwen/Qwen3-32B-FP8` | Quantized, want bf16 for clean activations |
| `Qwen/Qwen3-32B-GGUF` | GGUF format, incompatible with HF |

---

## CRITICAL: Thinking mode must be disabled

**Confidence: HIGH -- verified empirically.**

Qwen3-32B defaults to thinking mode (`<think>...long reasoning...</think>` before
the response). This breaks the pipeline:

1. Token budget exhaustion: 256 max_new_tokens consumed by reasoning
2. Keyword detection confusion: "attack" may appear inside think block
3. Response length explosion

### The fix: patch 05_run_inference.py

Add `enable_thinking=False` to all 3 `apply_chat_template` calls:

- **Line ~314** (stego detection path, `add_generation_prompt=False`)
- **Line ~450** (game scenarios path, `add_generation_prompt=True`) -- **CRITICAL**
- **Line ~643** (decode task path)

```python
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
```

This is harmless for non-Qwen models (the kwarg is ignored by tokenizers that don't
support it). If it throws TypeError, wrap with try/except.

---

## VRAM budget

```
Component              Estimated
Weights (bf16)         ~62 GB
KV cache (bs=2, 512)   ~1.3 GB
Hidden state buffer     ~0.6 GB
PyTorch overhead        ~3-5 GB
Total                  ~67-69 GB  (of 97 GB available)
Headroom               ~28-30 GB
```

Use `--batch_size 2`. If OOM, try `--batch_size 1`.

---

## Pipeline commands

### Step 0: Verify model loads (~2 min)

```bash
cd /home/ubuntu/SteganographyProbes
python3 -c "
from transformers import AutoConfig
c = AutoConfig.from_pretrained('Qwen/Qwen3-32B')
assert c.model_type == 'qwen3', f'WRONG model_type: {c.model_type}'
assert c.architectures == ['Qwen3ForCausalLM'], f'WRONG arch: {c.architectures}'
assert c.num_hidden_layers == 64, f'WRONG layers: {c.num_hidden_layers}'
assert c.hidden_size == 5120, f'WRONG hidden: {c.hidden_size}'
print('CONFIRMED: Qwen/Qwen3-32B, dense 64-layer transformer, 5120 hidden dim')
"
```

### Steps 1-3: Use run_pipeline.sh

```bash
cd /home/ubuntu/SteganographyProbes
# Verify enable_thinking=False is patched first!
grep -n 'enable_thinking' scripts/05_run_inference.py || echo "PATCH NOT APPLIED -- FIX FIRST"
bash run_pipeline.sh 2>&1 | tee /tmp/qwen3_pipeline.log
```

The script runs suffix_marker first (validation gate), then whitespace, then
game_scenarios_v2, then probe training, then git commit.

---

## Potential issues ranked by likelihood

### 1. Thinking mode not disabled (NEAR CERTAIN if code not patched)
**Detect:** Responses start with reasoning, lack actual code/answers.
**Fix:** Patch `apply_chat_template` as described above.

### 2. Inference speed slower than expected (~30% likelihood)
**Detect:** Watch progress output, calculate tokens/sec.
**Impact:** 12-18 hours total instead of 6-10. Not a failure.
**Priority order if time-constrained:** suffix_marker > game_scenarios_v2 > whitespace.

### 3. OOM on GPU (~10% likelihood)
**Detect:** `torch.cuda.OutOfMemoryError` traceback.
**Fix:** `--batch_size 1`. Check `nvidia-smi` for leftover processes.

### 4. Tokenizer kwarg incompatibility (~5% likelihood)
**Detect:** `TypeError: unexpected keyword argument 'enable_thinking'`.
**Fix:** Increase `--max_new_tokens` to 2048 and post-process to strip `<think>` blocks.

### 5. RLHF safety refusal (~5% likelihood)
**Detect:** 0% keyword rate for BOTH stego and clean conditions.
**Impact:** Legitimate finding (safety > instruction following) but behavioral comparison uninformative.

---

## Success criteria for morning

1. **suffix_marker metadata.json** with stego keyword rate > 50% AND gap > 15pp
2. **At least 2 of 3 inference runs complete** with 64 layer activations
3. **At least 1 probe result** with AUROC comparable or better than Llama's best
4. **No thinking-mode contamination** (spot-check responses.jsonl)
5. **Lightweight git commit pushed** (no .npy files)

---

## Output directory structure

```
data/Qwen3-32B/runs/
  2026-03-22_suffix_marker/
    config.json
    activations/
      metadata.json, responses.jsonl, labels.npy
      layer_00.npy through layer_63.npy  (~2.6 GB total)
    probe_results/
      probe_results.json
  2026-03-22_whitespace/
    ...
  2026-03-22_game_scenarios_v2/
    ...
```

64 layers x 5120 dim = ~2.6 GB of activation files per run. Total ~7.8 GB. Disk has 3.9 TB free.

---

## Evidence-based ROI matrix for next steps (updated)

| Rank | Action | Status | ROI | Rationale |
|:---:|---|---|:---:|---|
| **1** | Run Qwen3-32B suffix_marker as validation gate | **NEXT** | Highest | Single highest-information experiment. If Qwen3 discriminates suffix_marker, everything downstream becomes interpretable. |
| **2** | Patch `enable_thinking=False` | **PREREQUISITE** | Highest | Without this, Qwen3 pipeline fails mechanically. |
| **3** | Run Qwen3-32B game_scenarios_v2 | After #1 validates | Very High | Core paper contribution: per-method behavioral + probe results on a model that follows instructions. |
| **4** | Llama-3-8B probe training on all 3 runs | **DONE** | -- | suffix_marker AUC=0.578, whitespace AUC=0.603, game_v2 AUC=0.629. All weak. |
| **5** | Foreground synonym as headline presence finding | NARRATIVE | High | synonym 0.755 AUROC vs 0.499 text baseline is the only honest non-trivial signal. |
| **6** | Run Qwen3-32B whitespace | After #1 validates | Med-High | Strengthens cross-model comparison on the one method where Llama showed decode recognition. |
| **7** | Investigate +++ echo confound | ANALYSIS | Medium | 26.6% clean echo vs 2.7% stego. Worth documenting but not blocking. |
| **8** | Cross-method transfer on Qwen3 data | After #3 | Medium | If synonym-trained probes transfer on Qwen3, strong evidence of shared stego direction. |
| **9** | Complete LoRA carriers (54->99 groups) | DEFERRED | Medium | Contamination issues need refinement first. |
| **10** | Document Llama-3-8B failure as negative result | WRITING | Med-Low | Publishable finding supporting the probe approach. |

---

## Quick reference: what files to check

| What to check | Path |
|---|---|
| Suffix marker metadata | `data/Qwen3-32B/runs/2026-03-22_suffix_marker/activations/metadata.json` |
| Probe results | `data/Qwen3-32B/runs/2026-03-22_*/probe_results/probe_results.json` |
| Pipeline log | `/tmp/pipeline.log` |
| Llama baseline results | `data/Meta-Llama-3-8B-Instruct/runs/2026-03-21_*/probe_results/probe_results.json` |
| Presence detection baseline | `data/probe_results/stego_detection_probe_results.json` |
| Text baselines | `data/probe_results/text_baseline_results.json` |
