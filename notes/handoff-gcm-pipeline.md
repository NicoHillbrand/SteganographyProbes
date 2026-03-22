# HANDOFF: Full GCM Pipeline for Qwen3-32B suffix_marker

**Date:** 2026-03-22
**Priority:** MEDIUM (team request)
**Resource:** GPU only. ~1h per GCM run. Total ~3-4h depending on steering port.

---

## Goal

Regenerate GCM head attribution with the fixed prompt template, run a permutation null, then attempt steering. This validates the GCM methodology and resolves the 44% head overlap with whitespace (Concern 5 in `notes/pipeline-audit-2026-03-22.md`).

## Context

- Existing head lists in `data/Qwen3-32B_top10_stego_heads_suffix_marker.json` and `data/Qwen3-32B_top10_stego_heads_whitespace.json` are **STALE** -- computed under wrong prompt template (now fixed in commit `4179609`).
- Target dataset (suffix_marker) is a confirmed surface artifact (tfidf=1.000). We expect GCM to find real heads (the `+++` token IS real), but they should be surface-processing heads, not deep representational ones.
- `02_gcm_steer.py` uses transformer_lens which does **NOT** support Qwen3-32B (only up to Qwen3-14B). Steering step requires porting to HF hooks.

## Important project conventions

- Always use `PYTHONUNBUFFERED=1` when running long scripts through `tee`.
- Always pass `enable_thinking=False` in `apply_chat_template()` for Qwen3.
- `--batch_size 2` is max safe for Qwen3-32B activation extraction on the 97GB GH200.
- After completing any run, call `bash scripts/append_run_note.sh <run_dir> "<description>"`.
- See `CLAUDE.md` for full conventions.

---

## Step 1: Regen heads for suffix_marker + whitespace (~2h GPU)

The existing `steer/run_gcm_both.sh` does exactly this. Run it:

```bash
cd /home/ubuntu/SteganographyProbes/steer
PYTHONUNBUFFERED=1 bash run_gcm_both.sh 2>&1 | tee /tmp/gcm_regen.log
```

This regenerates:
- `data/Qwen3-32B_top10_stego_heads_suffix_marker.json` (suffix_marker, 50 pairs)
- `data/Qwen3-32B_top10_stego_heads_whitespace.json` (whitespace negative control, 50 pairs)

The script uses `steer/01_gcm_filter_heads_hf.py` which is the HF-native version (no transformer_lens dependency). It overrides `config.py` via the `config_run.py` / `sys.modules` trick inline in the bash script. The prompt template is already fixed:

```python
f"Context from previous assistant:\n{entry['input_text']}\n\nTask: {entry['task']}"
```

**After completion**, compare head overlap between suffix_marker and whitespace:

```python
import json
sm = {(h['layer'], h['head']) for h in json.load(open('data/Qwen3-32B_top10_stego_heads_suffix_marker.json'))[:50]}
ws = {(h['layer'], h['head']) for h in json.load(open('data/Qwen3-32B_top10_stego_heads_whitespace.json'))[:50]}
overlap = sm & ws
print(f"Overlap: {len(overlap)}/50 ({100*len(overlap)/50:.0f}%)")
# OLD overlap was 22/50 (44%). If fixed template resolves this, expect < 20%.
```

Record the overlap count for the results summary.

---

## Step 2: Permutation null on suffix_marker (~1h GPU)

Create a shuffled version of the dataset where stego/clean labels are randomly swapped within each pair, then re-run GCM. This tests whether the head scores are specific to the actual stego condition or just noise.

**Implementation:** The key function is `prepare_stego_pairs` in `steer/01_gcm_filter_heads_hf.py`. After creating the pairs, randomly swap `pair['stego']` and `pair['clean']` with 50% probability. Run inline like `run_gcm_both.sh` does:

```bash
cd /home/ubuntu/SteganographyProbes/steer
PYTHONUNBUFFERED=1 python3 -c "
import sys, os, json, random
sys.path.insert(0, '.')

# Override config
import types
config_mod = types.ModuleType('config')
config_mod.config = {
    'CACHE_DIR': '/home/ubuntu/.cache/huggingface/hub',
    'DEVICE': 'cuda',
    'MODEL_ID': 'Qwen/Qwen3-32B',
    'DATA_PATH': '/home/ubuntu/SteganographyProbes/data/suffix_marker_game_scenarios.jsonl',
    'VAL_DATA_PATH': '/home/ubuntu/SteganographyProbes/data/suffix_marker_game_scenarios.jsonl',
    'NUM_SAMPLES': 50,
    'VAL_NUM_SAMPLES': -1,
    'K_PCT': 10,
}
sys.modules['config'] = config_mod

from importlib import import_module
gcm = import_module('01_gcm_filter_heads_hf')

cfg = config_mod.config
model, tokenizer = gcm.load_model_and_tokenizer(cfg['MODEL_ID'], cfg['CACHE_DIR'], cfg['DEVICE'])
pairs = gcm.prepare_stego_pairs(cfg['DATA_PATH'], num_samples=cfg['NUM_SAMPLES'])
print(f'Loaded {len(pairs)} pairs')

# Permute: randomly swap stego/clean within each pair
random.seed(42)
for p in pairs:
    if random.random() < 0.5:
        p['stego'], p['clean'] = p['clean'], p['stego']
print('Labels permuted (50% swap rate, seed=42)')

topk = gcm.get_stego_knockout_attribution(model, tokenizer, pairs, k=cfg['K_PCT'])

output_file = '../data/Qwen3-32B_top10_stego_heads_suffix_marker_permuted.json'
with open(output_file, 'w') as f:
    json.dump(topk, f, indent=4)
print(f'Permuted results saved to {output_file}')
print(f'Top 10 heads:')
for h in topk[:10]:
    print(f'  Layer {h[\"layer\"]:2d}, Head {h[\"head\"]:2d}: {h[\"score\"]:.4f}')
" 2>&1 | tee /tmp/gcm_permutation_null.log
```

**After completion**, compare real vs permuted scores:

```python
import json
import numpy as np

real = json.load(open('data/Qwen3-32B_top10_stego_heads_suffix_marker.json'))
perm = json.load(open('data/Qwen3-32B_top10_stego_heads_suffix_marker_permuted.json'))

real_top10_mean = np.mean([h['score'] for h in real[:10]])
perm_top10_mean = np.mean([h['score'] for h in perm[:10]])
ratio = real_top10_mean / max(perm_top10_mean, 1e-8)

print(f"Real top-10 mean score:    {real_top10_mean:.4f}")
print(f"Permuted top-10 mean score: {perm_top10_mean:.4f}")
print(f"Ratio: {ratio:.2f}x")
```

---

## Step 3: Steering (BLOCKED -- needs code port)

`02_gcm_steer.py` uses `transformer_lens.HookedTransformer` which does **NOT** support Qwen3-32B. The transformer_lens registry only has Qwen3 up to 14B.

To run steering, the script must be ported to use native HF hooks (same approach as `01_gcm_filter_heads_hf.py`).

### Functions that need HF-hook equivalents

| Function | Current (transformer_lens) | Needed (HF native) |
|---|---|---|
| `compute_stego_steering_vectors()` | `model.run_with_cache(prompt, names_filter=...)` | Forward hooks on `self_attn.o_proj` to capture per-head z (same pattern as step 1's `get_stego_knockout_attribution`) |
| `apply_steering_suppression()` | `model.add_hook(f"blocks.{l}.attn.hook_z", hook)` | `attn_layers[l].o_proj.register_forward_pre_hook(hook)` -- reshape input to `[batch, seq, n_heads, head_dim]`, apply projection subtraction, reshape back |
| `validate_stego_behavior()` | `model.generate(prompt, ...)` + `model.to_single_token(...)` | `model.generate(tokenizer(prompt, ...).input_ids, ...)` + `tokenizer.encode("attack", add_special_tokens=False)[0]` |
| `calculate_kl_divergence()` | Same as above | Same pattern |

### Key architectural details for the port

- Qwen3-32B attention: `model.model.layers[l].self_attn`
- o_proj input shape: `[batch, seq, num_heads * head_dim]` -- reshape to `[batch, seq, num_heads, head_dim]` for per-head access
- `num_heads = model.config.num_attention_heads` (40 for Qwen3-32B)
- `head_dim = model.config.head_dim` (128 for Qwen3-32B)
- The steering vector subtraction happens BEFORE o_proj (on its input), not after

### Steering hook pattern (for the port)

```python
def make_steering_hook(steering_vectors, n_heads, head_dim, alpha=1.0):
    def hook_fn(module, args):
        inp = args[0]  # [batch, seq, n_heads * head_dim]
        z = inp.reshape(inp.shape[0], inp.shape[1], n_heads, head_dim)
        for (l, h), v in steering_vectors.items():
            if l == layer_idx:  # captured from outer scope
                v = v.to(z.device)
                head_act = z[:, :, h, :]
                dot = (head_act * v).sum(dim=-1, keepdim=True)
                proj = (dot / (v.norm()**2 + 1e-8)) * v
                z[:, :, h, :] -= alpha * proj
        return (z.reshape(inp.shape),)
    return hook_fn
```

**If time allows**, port `02_gcm_steer.py` to HF hooks and run. If not, steps 1+2 are the priority deliverables.

### Steering filename mismatch warning

`02_gcm_steer.py` reads heads from `data/{model}_top{k}_stego_heads_results.json` but `01_gcm_filter_heads_hf.py` writes to `data/{model}_top{k}_stego_heads_{dataset_tag}.json`. If you do port the script, fix the read path to `data/Qwen3-32B_top10_stego_heads_suffix_marker.json`.

---

## Expected outputs

After all steps, save a summary to `data/Qwen3-32B/gcm_results_suffix_marker_regen.json`:

```json
{
    "timestamp": "ISO-8601",
    "git_sha": "...",
    "real_heads_file": "data/Qwen3-32B_top10_stego_heads_suffix_marker.json",
    "permuted_heads_file": "data/Qwen3-32B_top10_stego_heads_suffix_marker_permuted.json",
    "whitespace_heads_file": "data/Qwen3-32B_top10_stego_heads_whitespace.json",
    "overlap_suffix_vs_whitespace": {"count": 0, "pct": 0.0},
    "real_top10_mean_score": 0.0,
    "permuted_top10_mean_score": 0.0,
    "ratio": 0.0,
    "steering_attempted": false,
    "steering_blocked_reason": "Qwen3-32B not in transformer_lens; port needed"
}
```

Then run:
```bash
bash scripts/append_run_note.sh data/Qwen3-32B \
  "GCM regen: suffix_marker + whitespace heads (fixed template) + permutation null"
```

---

## Decision rules

| Condition | Interpretation | Action |
|---|---|---|
| Overlap drops from 44% to <20% | Template fix resolved the issue | Update audit Concern 5 as RESOLVED |
| Overlap stays >35% | Shared heads are structurally important, not stego-specific | Update audit Concern 5 as CONFIRMED |
| Permuted ratio > 2 | GCM signal is real (expected -- `+++` IS a real feature) | Document as expected positive control |
| Permuted ratio < 1.5 | GCM is noisy, heads may not be reliable | Flag as methodological concern |
| Steering ported + ASR drops | Steering suppresses `+++` detection | Methods contribution for paper |
| Steering ported + ASR unchanged | Steering doesn't affect behavior (model already refuses 95%) | Expected -- low base rate means no room to suppress |
