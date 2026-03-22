# GCM on Qwen3-32B: Implementation Notes & Issues

**Date**: 2026-03-22  
**Runs**: suffix_marker (50 pairs), whitespace (50 pairs, negative control)  
**Hardware**: Lambda GH200, 97GB VRAM

---

## Issues Encountered

### 1. Qwen3-32B not in transformer_lens registry

**Problem**: The existing GCM code (`steer/01_gcm_filter_heads.py`) uses `HookedTransformer.from_pretrained_no_processing()`. Transformer_lens v2.17.0 supports Qwen3 up to 14B only. Loading Qwen3-32B raises `ValueError`.

**Fix**: Rewrote GCM as `steer/01_gcm_filter_heads_hf.py` using native HuggingFace + PyTorch hooks. The key insight: the input to each attention layer's `o_proj` (output projection) is exactly the per-head attention output z before projection, shaped `[batch, seq, n_heads * head_dim]`. A `register_forward_pre_hook` on `o_proj` captures this; reshaping to `[batch, seq, n_heads, head_dim]` gives per-head z — the equivalent of transformer_lens's `hook_z`.

### 2. OOM on backward pass (97GB insufficient)

**Problem**: Model weights = ~62GB. Full backward pass through 64 layers with hooks storing per-head z tensors at every layer requires ~35GB+ of gradient/activation memory, exceeding 97GB total.

**Attempted fixes that failed**:
- Gradient checkpointing (`model.gradient_checkpointing_enable()`) — still OOM
- Chunking layers into groups of 8, running backward per chunk — still OOM because the backward pass traverses the full model regardless of how many hooks are active

**Fix that worked**: One forward+backward per layer. For each layer L:
1. Hook only layer L's `o_proj` with a detached-leaf replacement
2. Forward the stego input through the full model
3. Use `torch.autograd.grad(loss, z_L)` to compute gradient w.r.t. only z_L
4. Gradient flows through layers L+1..63 normally (no hooks there)

This is semantically equivalent to the original all-layers-at-once approach because each layer's gradient is independent — detaching at layer L doesn't affect gradients at other layers since we only hook one layer per pass. VRAM stays flat at ~66GB.

**Cost**: 64 forward passes per pair instead of 1. ~28 seconds per pair (all 64 layers), so 50 pairs = ~23 minutes. Acceptable.

### 3. Wrong head_dim calculation

**Problem**: Initial code computed `head_dim = hidden_size // num_attention_heads = 5120 // 64 = 80`. Actual `head_dim` from config is 128 (Qwen3-32B uses a larger head dimension than implied by hidden_size/heads — the o_proj maps from `n_heads * head_dim = 8192` to `hidden_size = 5120`).

**Symptom**: `RuntimeError: shape '[1, 291, 64, 80]' is invalid for input of size 2383872` (actual: 291 * 8192 = 2383872).

**Fix**: Use `model.config.head_dim` directly instead of computing it.

### 4. Missing `enable_thinking=False` for Qwen3

**Problem**: Both GCM scripts (`01_gcm_filter_heads.py`, `02_gcm_steer.py`) call `tokenizer.apply_chat_template()` without `enable_thinking=False`. Qwen3 models have a "thinking mode" that prepends CoT in `<think>` tags before the actual response. This contaminates the logit signal at the last token position — GCM reads `log P(attack)` from a position that may be mid-think, not at the response boundary.

**Fix**: Added `enable_thinking=False` to all `apply_chat_template` calls in both scripts.

### 5. Target token verification

**Concern** (raised during review): `tokenizer.encode("attack")` might not match the token the model actually produces.

**Verified**: 
- `encode("attack")` → token 20566 (no leading space)
- `encode(" attack")` → token 3359 (with space)  
- After the assistant header (`<|im_start|>assistant\n`), the keyword appears as bare `attack` on its own line — token 20566 is correct
- Confirmed against all 49 keyword-present responses: last line is always bare `attack`

### 6. Config pointing to wrong machine

**Problem**: `steer/config.py` had `CACHE_DIR = '/scratch/sweta.j'` and `MODEL_ID = "Qwen/Qwen3-1.7B"` — from Sweta's machine.

**Fix**: Updated for Lambda GH200: cache dir to `~/.cache/huggingface/hub`, model to `Qwen/Qwen3-32B`, data paths to `/home/ubuntu/SteganographyProbes/data/`.

### 7. transformers version downgrade

**Side effect**: Installing `transformer_lens` downgraded transformers from 5.3.0 to 4.57.6. No issues observed — Qwen3 is supported in both. The HF-native GCM script doesn't use transformer_lens at all, but it's installed for potential future use of the original TL-based scripts on smaller models.

---

## Design Decisions

### Whitespace as negative control

Whitespace has no behavioral signal (1/1000 stego, 1/1000 clean keyword rate). GCM on it cannot identify meaningful causal heads. We run it to verify that GCM scores are flat/unstructured when there's no stego→behavior pathway — if top heads look similar to suffix_marker's, something is wrong.

**Caveat**: `abs()` accumulation guarantees some heads always score nonzero even under noise. Compare score distributions, not just top-K lists.

### Sequence position pooling

Both original TL code and our HF port use `mean(dim=1)` over sequence positions before computing per-head IE. This is a pooled-IE heuristic, not a position-specific dot product. The alternative (last-position only) would align better with the probes (which also use last-token activations). We kept mean-pooling for consistency with the existing Qwen3-1.7B run but this should be revisited.

### Paired design controls for task confounds

The probe critique flagged that 33/49 keyword-positive responses come from one task type. GCM is less vulnerable because it operates on matched stego/clean pairs with identical tasks — the task signal cancels in the `P_stego(attack) - P_clean(attack)` gradient. However, task-conditioned mediation can still dominate (heads that mediate stego detection specifically on vulnerable task types).

---

## Output Files

- `data/Qwen3-32B_top10_stego_heads_suffix_marker.json` — top 10% heads by IE score
- `data/Qwen3-32B_top10_stego_heads_whitespace.json` — negative control
- `steer/01_gcm_filter_heads_hf.py` — HuggingFace-native GCM script
