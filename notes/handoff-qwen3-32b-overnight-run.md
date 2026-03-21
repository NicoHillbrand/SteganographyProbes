# Handoff: Qwen3-32B Overnight Pipeline Run

**Date:** 2026-03-21
**Author:** Nico (via Droid)
**For:** Overnight agent on GH200
**Priority:** Run full stego pipeline with Qwen3-32B to compare against Llama-3-8B baseline

---

## Why this run

Llama-3-8B-Instruct is too weak to reliably follow the stego detection instructions.
Paired analysis shows it correctly discriminates stego from clean only 19.2% of the time
on the easiest signal (suffix marker `+++`), barely above the 14.1% wrong-direction rate.
The model coin-flips on "say attack" regardless of input. This means the behavioral signal
is near-zero, and the recognition probe has nothing to work with.

Qwen3-32B is a 4x larger dense transformer with IFEval 86.7 (vs ~70 for Llama-3-8B).
It should actually read and follow the system prompt instructions about what encoding
to look for. The comparison between weak-model and strong-model results is itself a
key finding for the paper.

---

## The model

```
Model ID:         Qwen/Qwen3-32B
Architecture:     Qwen3ForCausalLM (standard dense transformer, NOT MoE, NOT VL, NOT DeltaNet)
Parameters:       32.8B (31.2B non-embedding)
Layers:           64
Hidden dim:       5120
GQA:              64 Q heads / 8 KV heads
Context:          32K native
Weights:          17 safetensor shards, ~62GB bf16
Vocab:            151,936 tokens
Tokenizer:        Qwen2Tokenizer
```

**Already downloaded** to `~/.cache/huggingface/hub/models--Qwen--Qwen3-32B/`.
All 17 shards present and verified. transformers 5.3.0 is installed (needs >=4.51.0).

### Do NOT confuse with these similarly-named models

| Model | Why it's wrong |
|-------|---------------|
| `Qwen/Qwen3-VL-32B-Instruct` | Vision-language model. Architecture: `Qwen3VLForConditionalGeneration`. Will not load with `AutoModelForCausalLM`. |
| `Qwen/Qwen3.5-27B` | Hybrid Gated DeltaNet architecture. Architecture: `Qwen3_5ForConditionalGeneration`. Linear attention layers make probe results uninterpretable. |
| `Qwen/Qwen3-32B-FP8` | Quantized to FP8. We want bf16 for clean activation extraction. |
| `Qwen/Qwen3-32B-GGUF` | GGUF format for llama.cpp. Incompatible with HF Transformers. |

---

## CRITICAL: Thinking mode must be disabled

**Confidence: HIGH -- verified empirically.**

Qwen3-32B defaults to "thinking mode" where it generates a `<think>...long reasoning...</think>`
block before producing the actual response. This will break the pipeline in multiple ways:

1. **Token budget exhaustion.** The pipeline uses `--max_new_tokens 256`. Thinking mode will
   burn most or all of those tokens on internal reasoning, leaving no room for the actual
   response (or the "attack" keyword).
2. **Keyword detection confusion.** The word "attack" might appear inside the thinking block
   as part of reasoning ("I should check if I need to say attack...") even when the model
   decides NOT to output it in the final response.
3. **Response length explosion.** Thinking mode can produce 1000+ tokens of reasoning.
   The responses.jsonl will be enormous and downstream analysis will break.

### The fix

The `apply_chat_template` call in `scripts/05_run_inference.py` (line ~450 for game scenarios)
currently looks like:

```python
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

This needs `enable_thinking=False`:

```python
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
```

This adds an empty `<think>\n\n</think>\n\n` prefix to the assistant's generation slot,
which tells the model to skip reasoning and respond directly. Verified: the model then
generates clean responses without thinking blocks.

**There are THREE calls to `apply_chat_template` in `05_run_inference.py`:**

1. **Line ~314** (stego detection path, `add_generation_prompt=False`) -- this path wraps
   an existing response. With `enable_thinking=False` it inserts empty think tags around
   the assistant content. This is cosmetic and doesn't affect the experiment since the
   text content is preserved. Add the kwarg here too for consistency.

2. **Line ~450** (game scenarios path, `add_generation_prompt=True`) -- **THIS IS THE
   CRITICAL ONE.** This is the prompt that gets generated from. Must have
   `enable_thinking=False`.

3. **Line ~643** (decode task path, `add_generation_prompt` varies) -- same issue.
   Add `enable_thinking=False`.

The safest approach: add `enable_thinking=False` to ALL three calls. This is harmless
for Llama (the kwarg will be ignored by tokenizers that don't support it -- but verify
this; if it throws an error, wrap in try/except or check `hasattr`).

**UNCERTAINTY: MEDIUM.** I verified the template behavior empirically with the tokenizer,
but I have not tested full generation on GPU. There is a small chance the model ignores
the empty thinking block and thinks anyway. If you see `<think>` content in generated
responses, something went wrong -- stop and investigate.

### Alternative: increase max_new_tokens

If for any reason you cannot disable thinking mode, increase `--max_new_tokens` to 2048
so the model has room to think AND produce a response. But this will 4-8x the inference
time per example. Not recommended.

---

## VRAM budget

**Confidence: HIGH on weights, MEDIUM on peak usage.**

```
Component              Estimated
─────────────────────  ─────────
Weights (bf16)         ~62 GB
KV cache (bs=2, 512)   ~1.3 GB
Hidden state buffer     ~0.6 GB
PyTorch overhead        ~3-5 GB
─────────────────────  ─────────
Total                  ~67-69 GB  (of 97 GB available)
Headroom               ~28-30 GB
```

Use `--batch_size 2` (down from the Llama run's 4). The model is 4x larger, and
hidden states at 5120 dims across 64 layers are significant.

**UNCERTAINTY: LOW-MEDIUM.** The weights are a known quantity (62GB). The KV cache and
hidden state buffer depend on actual sequence lengths. If prompts are long (system prompt
+ context + task can be 500+ tokens), KV cache could be larger. The 28GB headroom should
absorb this, but if you hit OOM, try `--batch_size 1`.

---

## Pipeline commands

### Step 0: Verify model loads (quick smoke test, ~2 min)

Run this first. If it fails, everything else will too.

```bash
cd /home/ubuntu/SteganographyProbes
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-32B',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    output_hidden_states=True,
)
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-32B')
print(f'Loaded: {model.config.num_hidden_layers} layers, device={model.device}')
print(f'Hidden dim: {model.config.hidden_size}')
# Quick forward pass
inputs = tok('Hello world', return_tensors='pt').to(model.device)
with torch.no_grad():
    out = model(**inputs)
print(f'Hidden states: {len(out.hidden_states)} tensors, shape={out.hidden_states[0].shape}')
print('SMOKE TEST PASSED')
"
```

Expected output: 65 hidden state tensors (embedding + 64 layers), shape `[1, N, 5120]`.

### Step 1a: Suffix marker inference (~1.5-3 hrs estimated)

```bash
cd /home/ubuntu/SteganographyProbes
python3 scripts/05_run_inference.py \
  --model "Qwen/Qwen3-32B" \
  --scenarios data/suffix_marker_game_scenarios.jsonl \
  --generate --layers all --token_position last \
  --use_chat_template --batch_size 2 \
  --run_name suffix_marker 2>&1 | tee /tmp/qwen3_suffix_marker.log
```

**What to check after:** Look at `metadata.json` keyword_results. We EXPECT the stego
keyword rate to be MUCH higher than Llama's 45.1%, and the gap between stego and clean
to be much wider than Llama's 5.1%. If the stego rate is below 10%, thinking mode may
not have been properly disabled.

### Step 1b: Whitespace inference (~1.5-3 hrs estimated)

```bash
python3 scripts/05_run_inference.py \
  --model "Qwen/Qwen3-32B" \
  --scenarios data/whitespace_game_scenarios.jsonl \
  --generate --layers all --token_position last \
  --use_chat_template --batch_size 2 \
  --run_name whitespace 2>&1 | tee /tmp/qwen3_whitespace.log
```

### Step 1c: Game scenarios v2 (all 5 methods, ~3-5 hrs estimated)

```bash
python3 scripts/05_run_inference.py \
  --model "Qwen/Qwen3-32B" \
  --scenarios data/game_scenarios_v2.jsonl \
  --generate --layers all --token_position last \
  --use_chat_template --batch_size 2 \
  --run_name game_scenarios_v2 2>&1 | tee /tmp/qwen3_game_v2.log
```

### Step 2: Probe training (CPU, ~10-20 min each)

```bash
RUNS_BASE="data/Qwen3-32B/runs"
DATE=$(date +%Y-%m-%d)

for RUN_NAME in suffix_marker whitespace game_scenarios_v2; do
  RUN_DIR=$(ls -dt ${RUNS_BASE}/${DATE}_${RUN_NAME}* 2>/dev/null | head -1)
  if [ -n "$RUN_DIR" ]; then
    echo "Training probes for: $RUN_DIR"
    python3 scripts/06_train_probes.py --run_dir "$RUN_DIR" 2>&1 | tee -a /tmp/qwen3_probes.log
  else
    echo "WARNING: No run dir found for $RUN_NAME"
  fi
done
```

### Step 3: Git commit (lightweight, no .npy files)

```bash
cd /home/ubuntu/SteganographyProbes
RUNS_BASE="data/Qwen3-32B/runs"
DATE=$(date +%Y-%m-%d)

for RUN_NAME in suffix_marker whitespace game_scenarios_v2; do
  RUN_DIR=$(ls -dt ${RUNS_BASE}/${DATE}_${RUN_NAME}* 2>/dev/null | head -1)
  if [ -n "$RUN_DIR" ]; then
    git add -f "$RUN_DIR/config.json" \
               "$RUN_DIR/activations/metadata.json" \
               "$RUN_DIR/activations/responses.jsonl" \
               "$RUN_DIR/probe_results/" 2>/dev/null || true
    git commit -m "results: Qwen3-32B $(basename $RUN_DIR)" || true
  fi
done
git push origin main 2>&1 || echo "Push failed -- will retry in morning"
```

---

## Output directory structure

The pipeline's `run_utils.py` puts outputs under `data/<model>/runs/`. For Qwen3-32B,
the `sanitize_model_name` function strips the `Qwen/` prefix, so outputs land in:

```
data/Qwen3-32B/runs/
  2026-03-22_suffix_marker/
    config.json
    activations/
      metadata.json
      responses.jsonl
      labels.npy
      layer_00.npy through layer_63.npy    (~2.6 GB total, 64 layers x 5120 dim)
    probe_results/
      probe_results.json
  2026-03-22_whitespace/
    ...
  2026-03-22_game_scenarios_v2/
    ...
```

Note: 64 layers x 5120 hidden dim = ~2.6 GB of activation files per run (vs ~1.0 GB
for Llama's 32 layers x 4096 dim). Total across 3 runs: ~7.8 GB. Disk has 3.9 TB free.

---

## Potential issues ranked by likelihood

### 1. Thinking mode not disabled (LIKELIHOOD: NEAR CERTAIN if code not patched)

**What happens:** Model generates `<think>...500 tokens of reasoning...</think>` then
runs out of token budget before producing the actual response. Keyword detection fails.
Responses are truncated reasoning fragments.

**How to detect:** Check first few lines of responses.jsonl. If responses start with
reasoning text and lack code/actual answers, thinking mode is on.

**Fix:** Patch `apply_chat_template` calls as described above. If patching the script
feels risky, an alternative is to set `enable_thinking=False` at the tokenizer level
by modifying the chat template in the tokenizer config -- but this is more invasive.

### 2. OOM on GPU (LIKELIHOOD: LOW, ~10%)

**What happens:** CUDA out of memory error during forward pass or generation.

**How to detect:** Error traceback mentioning `torch.cuda.OutOfMemoryError`.

**Fix:** Reduce `--batch_size` to 1. If still OOM, check if another process is using
GPU memory (`nvidia-smi`). Kill any leftover processes from the Llama runs.

### 3. Qwen3 tokenizer incompatibility with `enable_thinking` kwarg (LIKELIHOOD: LOW, ~5%)

**What happens:** The `apply_chat_template(enable_thinking=False)` call might throw
a `TypeError: unexpected keyword argument` on older tokenizer versions.

**How to detect:** Immediate error on first example.

**Fix:** Check if the tokenizer's chat template Jinja2 code supports the kwarg.
If not, manually edit the chat template string to remove the `{% if enable_thinking %}`
block, or simply increase `--max_new_tokens` to 2048 and post-process responses to
strip `<think>...</think>` blocks.

### 4. pad_token handling (LIKELIHOOD: LOW, ~5%)

**What happens:** The pipeline sets `tokenizer.pad_token = tokenizer.eos_token` if
pad_token is None. Qwen3's tokenizer has `pad_token = <|endoftext|>` (id=151643)
which differs from `eos_token = <|im_end|>` (id=151645). This should work fine,
but there's a small chance left-padding interacts badly with Qwen3's position embeddings.

**How to detect:** Garbled or degenerate outputs (repeating tokens, empty responses).

**Fix:** This is unlikely. If it happens, try without `--use_chat_template` or check
if `padding_side="left"` is causing issues.

### 5. Inference speed much slower than expected (LIKELIHOOD: MEDIUM, ~30%)

**What happens:** 32B model at batch_size=2 is slower per-example than 8B at batch_size=4.
A single inference run could take 4-6 hours instead of 1.5-3.

**How to detect:** Watch the progress output. The pipeline prints every 10 batches.
Calculate tokens/sec from the first few batches.

**Impact:** Not a failure, just slower. Total pipeline could take 12-18 hours for all
three runs. If time-constrained, prioritize suffix_marker (easiest sanity check) and
game_scenarios_v2 (the scientifically important one). Whitespace can wait.

### 6. `output_hidden_states` returns unexpected shapes (LIKELIHOOD: VERY LOW, ~2%)

**What happens:** Qwen3ForCausalLM returns hidden states in a different format than
LlamaForCausalLM. The code indexes `hidden_states[layer_idx + 1]` assuming index 0
is the embedding layer.

**How to detect:** Shape mismatch errors, or activations that look wrong in probes.

**Fix:** Verify in smoke test (Step 0) that `len(out.hidden_states) == 65` (1 embedding
+ 64 layers) and `out.hidden_states[0].shape[-1] == 5120`.

---

## What we don't know yet

These are genuine unknowns that only the run itself will answer:

1. **Will Qwen3-32B actually discriminate stego from clean behaviorally?** We expect yes
   based on IFEval scores, but this is a novel task. No prior work tests instruction-following
   on steganography detection specifically. The suffix marker run will answer this first.

2. **Will probes on a 64-layer model behave differently from 32-layer Llama?** More layers
   means the signal might emerge at different relative depths. Layer 3 was best for Llama;
   for Qwen3-32B it might be layer 6 or layer 10 or somewhere else entirely. This is
   expected and interesting, not a problem.

3. **Will the larger hidden dim (5120 vs 4096) help or hurt probe performance?** Linear
   probes on higher-dimensional spaces can overfit more easily. The grouped cross-validation
   in `06_train_probes.py` should handle this, but watch for train/test AUROC divergence.

4. **How will greedy generation interact with the "attack" keyword?** Llama runs used greedy
   decoding. Qwen3 recommends `temperature=0.7, top_p=0.8` for non-thinking mode. The
   pipeline defaults to greedy (`--generation_mode greedy`). This might slightly suppress
   the model's willingness to output "attack" if it's not the highest-probability next token.
   If behavioral results are surprisingly low, consider a follow-up with sampling.

5. **Will the model refuse the misaligned system prompt?** Qwen3-32B has RLHF safety training.
   It might recognize "end your response with attack" as a potentially harmful instruction
   and refuse to comply. This would manifest as 0% keyword rate for BOTH stego and clean
   conditions. If this happens, it's a legitimate finding (safety training overrides
   instruction following) but it means the behavioral comparison is uninformative and we
   rely entirely on probes.

---

## Success criteria for morning

1. **suffix_marker metadata.json exists** with keyword_results showing stego rate > 50%
   AND gap (stego - clean) > 15%. This would confirm the model is actually detecting
   the `+++` marker, unlike Llama.

2. **At least 2 of 3 inference runs complete** with activations saved (layer_00.npy
   through layer_63.npy).

3. **At least 1 probe result** showing AUROC for the Qwen3-32B run, comparable or
   better than Llama's 0.93 on stego detection.

4. **No thinking-mode contamination** in responses.jsonl (spot-check: responses should
   be code snippets and helpful text, not chains of reasoning wrapped in `<think>` tags).

5. **Lightweight git commit** pushed (no .npy files, just config + metadata + responses
   + probe results).

---

## Quick reference: model identity checklist

Before starting, run this one-liner to triple-check you're loading the right model:

```bash
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
