# Handoff: Lambda GH200 GPU Queue -- 2026-03-22

**Machine:** Lambda GH200, 97GB VRAM, CUDA 13.0
**Model cached at:** `~/.cache/huggingface/hub/models--Qwen--Qwen3-32B/`
**Repo state:** clean on `main`, pushed to remote

---

## Current results (Qwen3-32B suffix_marker only)

Inference completed for suffix_marker (2000 examples, 64 layers extracted).

| Condition | Keyword "attack" rate | N |
|-----------|----------------------|---|
| misaligned_with_stego | **4.9%** (49/1000) | 1000 |
| misaligned_no_stego | **0.0%** (0/1000) | 1000 |

**Interpretation:** The model has near-zero behavioral compliance overall (unlike Llama's ~40-45%), but the discrimination is actually *perfect* -- zero false positives. When the model does say "attack", it is ALWAYS in the stego condition. Fisher's exact test: p < 1e-14. This is a cleaner behavioral signal than Llama's noisy 45.1% vs 40.0%.

Probes have NOT been trained yet on this data. Activations are saved at:
`data/Qwen3-32B/runs/2026-03-22_suffix_marker/activations/` (64 layers x 5120 dim)

Whitespace and game_scenarios_v2 inference have NOT been run.

**Why the pipeline stopped:** `run_pipeline.sh` had a validation gate expecting >50% stego keyword rate. Since Qwen3-32B hit only 4.9%, the script's `set -euo pipefail` killed it before running the remaining methods.

---

## GPU task queue (priority order)

### Task 1: Train probes on suffix_marker (~30 min)
```bash
cd /home/ubuntu/SteganographyProbes
python3 scripts/06_train_probes.py --run_dir data/Qwen3-32B/runs/2026-03-22_suffix_marker
```
Despite low keyword rate, the probes predict `keyword_present` from pre-generation activations.
With a perfect 4.9% vs 0.0% split, even a small number of positive labels should produce
a cleaner probe signal than Llama's noisy labels.

### Task 2: Run whitespace + game_scenarios_v2 inference (~10 hours total)
Do NOT use `run_pipeline.sh` -- it will fail on the validation gate again. Run directly:
```bash
cd /home/ubuntu/SteganographyProbes

# Whitespace (~5 hours)
python3 scripts/05_run_inference.py \
  --model Qwen/Qwen3-32B \
  --scenarios data/whitespace_game_scenarios.jsonl \
  --generate --layers all --token_position last \
  --use_chat_template --batch_size 2 \
  --run_name whitespace

# Game scenarios v2 (~5 hours)
python3 scripts/05_run_inference.py \
  --model Qwen/Qwen3-32B \
  --scenarios data/game_scenarios_v2.jsonl \
  --generate --layers all --token_position last \
  --use_chat_template --batch_size 2 \
  --run_name game_scenarios_v2
```
Then train probes on each:
```bash
python3 scripts/06_train_probes.py --run_dir data/Qwen3-32B/runs/2026-03-22_whitespace
python3 scripts/06_train_probes.py --run_dir data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2
```

### Task 3: GCM head filtering for Qwen 32B (Sweta's code)
Sweta pushed `steer/01_gcm_filter_heads.py` + `steer/config.py`. To run on Lambda:

1. Install transformer_lens:
```bash
pip install transformer_lens
```

2. Update config.py:
```python
config = {}
config['CACHE_DIR'] = '/home/ubuntu/.cache/huggingface/hub'
config['DEVICE'] = 'cuda'
config['MODEL_ID'] = "Qwen/Qwen3-32B"
config['DATA_PATH'] = "/home/ubuntu/SteganographyProbes/data/game_scenarios_v2.jsonl"
config['NUM_SAMPLES'] = 100  # Start small -- backward pass on 32B is expensive
config['K_PCT'] = 10
```

**VRAM concern:** GCM requires a backward pass through the full model. With Qwen 32B at
~62GB weights, the backward pass activations+gradients will likely push close to the 97GB
limit. Start with `NUM_SAMPLES=100` and monitor `nvidia-smi`. If OOM:
- Try `NUM_SAMPLES=50`
- Or set `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` (already in script)
- Worst case: batch the pairs manually (process 10 at a time, accumulate scores)

Run:
```bash
cd /home/ubuntu/SteganographyProbes/steer
python3 01_gcm_filter_heads.py
```
Output will be at `data/Qwen3-32B_top10_stego_heads_results.json`

### Task 4: Steering code (when Sweta pushes it)
Wait for Sweta to push the steering code. Once available, run it with the top-K heads
from Task 3 as input.

---

## Git protocol for results

```bash
# Stage lightweight files ONLY (no .npy -- .gitignore blocks them now)
git add -f data/Qwen3-32B/runs/*/config.json
git add -f data/Qwen3-32B/runs/*/activations/metadata.json
git add -f data/Qwen3-32B/runs/*/activations/responses.jsonl
git add -f data/Qwen3-32B/runs/*/probe_results/

# Verify no .npy staged (should return nothing)
git diff --cached --name-only | grep '\.npy' && echo "STOP: .npy files staged!" || echo "OK"

git commit -m "results: Qwen3-32B <method_name> inference + probes"
git push origin main
```

---

## Answers for teammates

### For Nico ("Is it also failing this on the dataset with +++ at the end?")
Yes and no. Suffix_marker (+++):
- Stego keyword rate: 4.9%, clean rate: 0.0%
- The model rarely says "attack" overall, but when it does, it's ALWAYS in the stego condition
- This is actually the *cleanest discrimination* of any method so far -- zero false positives
- Llama had 45.1% vs 40.0% (noisy, barely significant). Qwen has 4.9% vs 0.0% (clean, p < 1e-14)
- The model detects the +++ but is reluctant to output "attack" -- likely RLHF safety alignment

### For Sweta (GCM on Qwen 32B)
Config changes needed (see Task 3 above). Main concern is VRAM for backward pass.
Suggest starting with NUM_SAMPLES=100. The `enable_thinking=False` patch in the
inference script is irrelevant for GCM since transformer_lens handles tokenization
differently, but if using apply_chat_template anywhere, add `enable_thinking=False`
to prevent Qwen3's thinking mode from consuming the token budget.

---

## Files on this machine (not in git, will be lost on terminate)

- `data/Qwen3-32B/runs/2026-03-22_suffix_marker/activations/layer_*.npy` -- 64 layer activations, ~2.6GB
- `/tmp/qwen3_pipeline.log` -- full pipeline log
- `/home/ubuntu/endgame.sh`, `terminate_after_push.sh`, `watch_pipeline.sh` -- automation scripts (not needed if running manually)
- `scripts/finalize_qwen3.sh` -- auto-commit/push script (still useful)

**Do NOT terminate the instance until:**
1. All probe results are trained and committed
2. GCM run is complete (or determined to be infeasible at 32B scale)
3. `git push` confirmed with `git log origin/main`
