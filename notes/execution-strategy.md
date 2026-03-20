# Execution Strategy — 65h remaining

## Git Strategy

### Branch structure
```
main
├── hugo/infra      ← T1, T2, T3, T4 (probe eval fixes, baselines)
├── hugo/dataset    ← T5, T6, T7 (data restructuring, decode task)
└── hugo/model      ← T8, T9, T10, T11, T12 (GH200 GPU work)
```

### Merge order
1. `hugo/dataset` → `main` first (data changes everything downstream depends on)
2. `hugo/infra` → `main` second (eval code, no data conflicts)
3. `hugo/model` → `main` last (results, depends on both above)

### Commit convention
```
fix(data): add grouped bundle_id to Redwood examples
fix(eval): implement grouped splits by base_text_index
feat(data): build detect/decode prompt templates with 8-color payloads
feat(eval): add text-only surface-feature baseline
feat(model): validate LLaMA 3 8B + TrojanStego LoRA on GH200
feat(data): generate token-bucket carriers via LoRA
feat(model): run recognition inference with decode scoring
feat(eval): train headline probes with grouped splits
```

### Rule: never commit generated data to the same branch as code changes
- Code changes go in the branch commits
- Large generated files (activations, regenerated JSONL) go in a separate `data:` prefixed commit after merge to main
- `.gitignore` the `data/activations/` directory

---

## Progress Review (2026-03-20)

### Status by Task

| Task | Status | Notes |
|------|--------|-------|
| **T1** Grouped splits in `06_train_probes.py` | ✅ DONE | `GroupShuffleSplit` implemented with `carrier_group_id` fallback to `metadata.base_text_index`. Lines 121–146. |
| **T2** Text-only baseline | ✅ DONE | `07_text_baselines.py` created. Char features (8 feats) + char n-gram TF-IDF (2–4 gram, 10k feats). Uses grouped splits. Per-method breakdown included. |
| **T3** Per-method probe breakdown | ✅ DONE | Added to `06_train_probes.py` lines 224–267. Runs on best layer, trains separate probe per method vs clean. |
| **T4** Fix game scenario evaluation | ✅ DONE | Probe filters to misaligned-only conditions. Uses observed `keyword_present` labels when `responses.jsonl` exists. Activation arrays masked to match. Committed `6ffb3e8`. |
| **T5** Add `bundle_id` / `carrier_group_id` | ✅ DONE | `02_generate_redwood_stego.py` emits both fields. All 2000 rows in `combined_dataset.jsonl` have them. |
| **T6** Build detect/decode prompt templates | ✅ DONE | `04b_build_decode_task.py` created. 4000 rows in `decode_task.jsonl` (2000 decode + 2000 benign). 8 colors mapped. `score_decode_response()` scoring function present. |
| **T7** Regenerate combined dataset | ✅ DONE | `03_combine_dataset.py` has `--training_sources` flag (default: `redwood_synthetic`). TrojanStego correctly routed to `ood_test_dataset.jsonl` (990 rows). Training set is Redwood-only (2000 rows). |
| **T8** Validate LLaMA 3 8B + LoRA on GH200 | ✅ CODE DONE | `00_validate_model.py` created. Loads bf16, attaches LoRA, tests toggle, verifies activations + chat template. **Not yet executed** (needs GH200). |
| **T9** Generate stego carriers via LoRA | ❌ NOT DONE | `01b_generate_lora_carriers.py` does not exist. |
| **T10** Run recognition inference | ❌ NOT DONE | `05_run_inference.py` has no decode task path. Only handles `stego_detection` and `game_scenarios`. |
| **T11** Run presence inference | ⚠️ PARTIALLY DONE | `05_run_inference.py` already uses `torch.bfloat16` (T2.8 fixed) and has `--use_chat_template` flag + `--sources` filter. **Not yet executed.** |
| **T12** Train & plot headline probes | ❌ NOT DONE | `08_plot_results.py` does not exist. |

### Summary
- **8 of 12 tasks have code written** (T1–T8)
- **0 tasks have been executed** on GPU — no activations, no probe results yet
- **4 tasks still need code** (T9, T10, T12) or are partial (T11)
- All data pipeline outputs exist and are structurally correct
- Committed in `a099330` (T1–T3, T5–T8) and `6ffb3e8` (T4 + housekeeping)

---

## Issues

### ~~ISSUE 1: T4 game scenario probe is still confounded~~ ✅ FIXED (`6ffb3e8`)
Probe now filters to misaligned-only and uses observed keyword labels.

### ~~ISSUE 2: No `.gitignore` exists~~ ✅ FIXED (`6ffb3e8`)
Created with `data/activations/`, `__pycache__/`, `.ruff_cache/`, `*.pyc`.

### 🟡 ISSUE 3: `05_run_inference.py` has no decode task path
**Location:** `scripts/05_run_inference.py`
**Problem:** T10 requires feeding decode prompts through the model and scoring `decode_correct`. The current script only handles `stego_detection` (raw text → activations) and `game_scenarios` (scenario prompts → activations + optional generation). There's no code path to:
- Load `decode_task.jsonl`
- Run inference on decode/benign prompts
- Score model output via `score_decode_response()` from `04b_build_decode_task.py`
- Save `decode_correct` / `recognized` labels alongside activations

**Fix needed:** Add a third task path `--decode_task` to `05_run_inference.py`, or create a dedicated script.

### 🟡 ISSUE 4: `01b_generate_lora_carriers.py` missing (T9)
**Location:** `scripts/` — file does not exist
**Problem:** This is on the critical path (T8→T9→T10→T12). Without LoRA-generated carriers, the TrojanStego-style matched experiment can't run.
**Impact:** Blocks T10 (recognition inference) and T12 (headline probes figure 2/3).

### 🟢 ISSUE 5: Acrostic confound acknowledged but unfixed
**Location:** `scripts/02_generate_redwood_stego.py`, pipeline review §2.7
**Problem:** Acrostic encoding adds sentence starters ("Furthermore, ", "However, ") that are surface-detectable. Per-method breakdown (T3) will show this, and the text baseline (T2) will quantify it. No code fix needed — just interpret results honestly.
**Action:** Flag in results write-up. The per-method breakdown already handles this.

### ~~ISSUE 6: No dependency manifest~~ ✅ FIXED (`6ffb3e8`)
Added `requirements.txt` with torch, transformers, peft, accelerate, sklearn, numpy.

---

## Agent Parallelism

### Wave 1 — Can run simultaneously (Friday evening → Saturday morning)

| Agent | Branch | Tasks | Deps | Est. |
|-------|--------|-------|------|------|
| Agent 1 (CPU) | `hugo/infra` | T1, T2, T3 | None | ~5h |
| Agent 2 (CPU) | `hugo/dataset` | T5, T6, T7 | T5→T7 | ~4.5h |
| Agent 3 (GH200) | `hugo/model` | T8 | None | ~2h |

**Gate:** T8 must pass before T9 starts. T5+T7 must merge before T11.

### Wave 2 — After Wave 1 gates (Saturday)

| Agent | Branch | Tasks | Deps | Est. |
|-------|--------|-------|------|------|
| Agent 3 (GH200) | `hugo/model` | T9, T10, T11 | T8✓, T6✓, T7✓ | ~10h |
| Agent 1 (CPU) | `hugo/infra` | T4 | T3✓ | ~1.5h |

### Wave 3 — Convergence (Saturday evening → Sunday)

| Agent | Branch | Tasks | Deps | Est. |
|-------|--------|-------|------|------|
| Agent 3 (GH200) | `hugo/model` | T12 | T1✓, T2✓, T10✓, T11✓ | ~3h |

---

## Task Details

### T1: Grouped splits in `06_train_probes.py` [CRITICAL] ✅
**Problem:** Random `train_test_split` leaks identical base texts across train/test (50 unique texts × ~20 duplicates each).
**Fix:** Use `GroupShuffleSplit` with `base_text_index` as group key. Load group IDs from metadata stored per example.
**Files:** `scripts/06_train_probes.py`

### T2: Text-only baseline [HIGH] ✅
**Problem:** No way to tell if activation probes beat trivial surface features.
**Fix:** New script `scripts/07_text_baselines.py`:
- Feature set 1: length, tab_count, zwc_count, digit_ratio, newline_count, cap_ratio
- Feature set 2: char 3-gram TF-IDF
- Same grouped splits as T1
**Files:** new `scripts/07_text_baselines.py`

### T3: Per-method probe breakdown [HIGH] ✅
**Problem:** Pooled AUROC hides that some methods (prefix_binary) are trivial while others (synonym) are hard.
**Fix:** Add per-method AUROC breakdown to `06_train_probes.py` output.
**Files:** `scripts/06_train_probes.py`

### T4: Fix game scenario evaluation [MEDIUM] ⚠️ NOT DONE → see [Issue 1](#-issue-1-t4-game-scenario-probe-is-still-confounded)
**Problem:** (a) System prompt confound between aligned/misaligned conditions. (b) Labels are expected, not observed.
**Fix:** In probe training, only compare `misaligned_with_stego` vs `misaligned_no_stego`. Use actual keyword presence as label when generation results exist.
**Files:** `scripts/06_train_probes.py`, `scripts/04_build_game_scenarios.py`

### T5: Add `bundle_id` / `carrier_group_id` [CRITICAL] ✅
**Problem:** No group keys exist for proper splitting.
**Fix:** Add `bundle_id = f"{method}_{base_text_index}"` and `carrier_group_id = base_text_index` to Redwood generation. Propagate through combine step.
**Files:** `scripts/02_generate_redwood_stego.py`, `scripts/03_combine_dataset.py`

### T6: Build detect/decode prompt templates [HIGH] ✅
**Problem:** No recognition labels exist (the PLAN's main thesis).
**Fix:** New script `scripts/04b_build_decode_task.py`:
- For each carrier, create detect/decode prompt + benign control prompt
- 8 color payloads mapped to stego methods
- Auto-scoring of model output → `decode_correct`
- `recognized = is_stego AND decode_correct`
**Files:** new `scripts/04b_build_decode_task.py`

### T7: Regenerate combined dataset [CRITICAL] ✅
**Problem:** TrojanStego mixed with Redwood negatives = source confound.
**Fix:** In `03_combine_dataset.py`, add `--training_sources` flag. Default: Redwood-only. TrojanStego tagged `split_role: ood_test_only`.
**Files:** `scripts/03_combine_dataset.py`, regenerate `data/combined_dataset.jsonl`

### T8: Validate LLaMA 3 8B + LoRA on GH200 [INFRA] ✅ code, ❌ not executed
**What:** Load `meta-llama/Meta-Llama-3-8B-Instruct` bf16, attach `worta/TrojanStego-LLama3-8B-LoRA` via peft, confirm adapter toggle works, generate a few test outputs, verify decode.
**Files:** new `scripts/00_validate_model.py`

### T9: Generate stego carriers via LoRA [HIGH] ❌ NOT STARTED
**What:** Generate 200–300 token-bucket stego carriers with LoRA enabled + matched clean with LoRA disabled on same prompts. This gives us TrojanStego-style carriers WITH matched negatives (same model, same prompt, just adapter off).
**Files:** new `scripts/01b_generate_lora_carriers.py`

### T10: Run recognition inference [HIGH] ❌ NOT STARTED → see [Issue 3](#-issue-3-05_run_inferencepy-has-no-decode-task-path)
**What:** Feed carriers through base model with detect/decode prompt and benign prompt. Extract activations. Score `decode_correct`. This is the PLAN's core experiment.
**Files:** `scripts/05_run_inference.py` (extend for decode task)

### T11: Run presence inference [HIGH] ⚠️ CODE READY, not executed
**What:** Run Redwood-only dataset through base model. Extract activations with last-token + mean pooling. Use `apply_chat_template` and `torch.bfloat16`.
**Files:** `scripts/05_run_inference.py` (fix dtype to bf16, add chat template)

### T12: Train & plot headline probes [HIGH] ❌ NOT STARTED
**What:** Produce the 4 minimum figures from PLAN §9:
1. Layerwise AUROC for `is_stego` (grouped split, Redwood-only)
2. Layerwise AUROC for `recognized` (decode task)
3. Same-carrier control (decode vs benign)
4. Activation probe vs text baseline comparison
**Files:** `scripts/06_train_probes.py`, new `scripts/08_plot_results.py`

---

## Critical Path

```
T8 (validate model) → T9 (generate carriers) → T10 (recognition inference) → T12 (headline probes)
                                                                                ↑
T5 (bundle IDs) → T7 (regen dataset) → T11 (presence inference) ────────────────┘
                                                                                ↑
T1 (grouped splits) ───────────────────────────────────────────────────────────┘
T2 (text baseline) ────────────────────────────────────────────────────────────┘
```

**Longest path:** T8 → T9 → T10 → T12 ≈ 12h of GPU work
**Parallel CPU path:** T1 + T2 + T3 + T5 + T6 + T7 ≈ 12h (but parallelizable to ~6h)

**Sunday is for figures and report, not new experiments.**

---

## Remaining Work Priorities (Saturday)

### Must-do (blocks headline results)
1. **T9** — Write `01b_generate_lora_carriers.py` (blocks T10→T12)
2. **T10** — Add decode task path to `05_run_inference.py` (blocks T12)
3. **T8** — Execute `00_validate_model.py` on GH200 (gate for T9)
4. **T11** — Execute `05_run_inference.py` on GH200 with Redwood data
5. **T12** — Write `08_plot_results.py` for headline figures

### Should-do (improves validity)
6. ~~**T4** — Fix game scenario probe filtering~~ ✅ done
7. ~~Create `.gitignore`~~ ✅ done
8. ~~Add `requirements.txt`~~ ✅ done

### Nice-to-have
9. Surface-matched hard negatives for acrostic ([Issue 5](#-issue-5-acrostic-confound-acknowledged-but-unfixed))

---

## Key Changes from Previous PLAN

| Previous | Now | Why |
|----------|-----|-----|
| Gemma 3 4B smoke test | Dropped | GH200 makes this unnecessary, 4B too small for stego |
| NF4 quantization | Dropped | GH200 = full bf16 for everything |
| Random splits | Grouped by `base_text_index` | Fix critical leakage |
| Mixed TrojanStego+Redwood training | Redwood-only training | Fix source confound |
| `is_stego` only | `is_stego` + `recognized` | Align with PLAN thesis |
| No baselines | Text-only baseline | Required to interpret probe results |
| `torch.float16` | `torch.bfloat16` | Match PLAN, better numerical stability |
| Manual chat tokens | `apply_chat_template` | Portable across models |

local inference available on Qwen 3.5 9B Q4 GGUF  though llama.cpp - manual at /home/hugo/tools/LOCAL-INFERENCE.md


Nuisance Stats: Tracking response length, punctuation, and capitalization is excellent. If your probe just detects "shorter text," that's an interpretability illusion. You are being your own greatest critic.

Surface-Matched Controls: Testing against inputs that include words like "secret" but don't actually ask for steganography is brilliant. It ensures the probe finds the behavior, not just the vocabulary.
