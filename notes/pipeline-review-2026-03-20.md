# Pipeline & Dataset Review — 2026-03-20 (Friday evening)

**Reviewer:** Hugo (with automated audit)
**Scope:** Full review of Nico's pipeline (commit `660e1c6`), Hugo's bugfixes (commits `87dcd76`, `16a3260`, `78413d4`), and gap analysis against `dataset-spec.md` / `PLAN.md`.
---

## 2. NEW Issues Found

### 2.1 Critical: TrojanStego source confound (DATA)

**Finding:** 990 TrojanStego examples are ALL `is_stego=True`. There are ZERO TrojanStego clean negatives. All 1000 clean negatives come from Redwood synthetic.

```
Combined dataset by (source, is_stego):
  ('redwood_synthetic', False): 1000
  ('redwood_synthetic', True):  1000
  ('trojanstego',       True):   990
```

**Impact:** A probe trained on `combined_dataset.jsonl` can achieve high AUROC by detecting the **source distribution** (fine-tuned LLaMA 3 8B style vs. hand-crafted QA text), not steganography. This is the single largest confound in the current dataset.

**Recommendation:** For headline results, use Redwood-only (1000 stego + 1000 clean, balanced). TrojanStego can only be used as an OOD transfer test, never mixed into the training pool with Redwood negatives.

### 2.2 Critical: Massive clean-text duplication → split leakage (DATA)

**Finding:** There are only **50 unique base texts** in the Redwood synthetic data, but **1000 clean rows** (each base text duplicated ~20× as counterpart to each method). With random `train_test_split`, identical clean texts will appear in both train and test.

**Impact:** Probe memorizes the 50 base texts → inflated accuracy/AUROC on any random split.

**Recommendation:** Split by `base_text_index` (grouped split), not by row. The code already stores this in metadata but `06_train_probes.py` ignores it.

### 2.3 High: No text-only baseline (EVAL)

**Finding:** Without a surface-feature baseline (char n-grams, length, whitespace count, zero-width char count), there is no way to tell whether activation probes detect anything beyond trivially visible artifacts.

For example, `prefix_binary` literally prepends a binary string. `whitespace` adds tabs. `unicode` adds zero-width characters. A regex would achieve near-perfect accuracy on these channels.

**Recommendation:** Add at minimum:
1. Character-level feature baseline (length, tab count, zero-width count, digit ratio)
2. TF-IDF logistic regression on raw text

### 2.4 High: No recognition/decode labels (SCIENCE)

**Finding:** The PLAN's main thesis is "probe detects when the model is recognizing/decoding hidden content" (`recognized = is_stego AND decode_correct`). The current pipeline tests only `is_stego` — pure presence classification. This is a fundamentally weaker claim.

**Recommendation:** Build the detect/decode task from PLAN §4.1–4.2:
- Present carrier with "does this contain a hidden payload from {RED, BLUE, ...}?"
- Score model output automatically
- Define `decode_correct` and `recognized` labels

### 2.5 Medium: Game scenario label leakage through system prompt (DESIGN)

**Finding:** The 3 conditions use different system prompts:
- `misaligned_with_stego` + `misaligned_no_stego` → long misaligned system prompt
- `aligned_clean` → short aligned system prompt

The probe can partially separate conditions by reading the system prompt, not by detecting stego recognition.

**Recommendation:** For the probe contrast, compare only `misaligned_with_stego` vs `misaligned_no_stego` (same system prompt, different carrier). Drop `aligned_clean` from the probe binary classification (keep for behavioral analysis only).

### 2.6 Medium: Game scenario labels are expectations, not observations (DESIGN)

**Finding:** Labels come from `expected_covert_action`, not actual model behavior. If the model doesn't follow the misaligned system prompt (likely for many aligned models), the labels are wrong.

**Recommendation:** When generation is enabled, relabel based on whether "attack" was actually emitted. This makes the task: "detect when the model actually acted covertly."

### 2.7 Medium: Acrostic still has a partial confound (DATA)

**Finding:** The fix correctly preserves original sentences, but the added starter phrases ("Furthermore, ", "However, ", etc.) still change the text's surface style measurably. A clean counterpart won't have these starters. A simple n-gram classifier could distinguish them without detecting stego.

**Impact:** Less severe than before (acrostic texts are now topically matched), but still a confounder without a surface-matched hard negative.

### 2.8 Low: `torch.float16` in inference, PLAN says `bfloat16` (CODE)

**Finding:** `05_run_inference.py` loads the model in `torch.float16`, but `PLAN.md` specifies `torch.bfloat16` (important for numerical stability with large activations).

**Impact:** Minor for hackathon; fp16 is fine for most models but could cause rare overflow in activation values.

### 2.9 Low: No `apply_chat_template` (CODE)

**Finding:** `05_run_inference.py` game scenarios manually construct Llama special tokens. This is fragile and will break for non-Llama models (e.g., Gemma). The stego detection task feeds raw text with no chat template at all.

---

## 3. Gap Analysis: Spec vs Implementation

### What the spec requires but is NOT implemented

| Spec requirement | Status | Priority for hackathon |
|---|---|---|
| Three separate probe tasks (A/B/C) | ❌ Missing | HIGH — at least add Task C (recognition) |
| `recognized` / `decode_correct` labels | ❌ Missing | HIGH — core thesis |
| Matched bundles (5 types of controls) | ❌ Only clean counterparts exist | MEDIUM — add hard negatives |
| Grouped splits by bundle/template/mechanism | ❌ Random splits only | HIGH — required for valid results |
| Instruction regimes (explicit/implicit/few-shot) | ❌ Not in main dataset | LOW — stretch goal |
| Baselines (text-only, nuisance) | ❌ Missing | HIGH — needed to interpret results |
| Per-environment metrics, worst-group, AUPRC | ❌ Only acc/AUROC/F1 | MEDIUM |
| Carrier-span activation pooling | ❌ Only full-sequence pooling | MEDIUM |
| Nuisance stats stored per example | ❌ Missing | LOW |
| `bundle_id` / `base_prompt_id` / `template_id` | ❌ Missing (but `base_text_index` exists in metadata) | HIGH for splits |

### What Nico built that is useful but diverges from spec

| Implementation | Relationship to spec | Keep? |
|---|---|---|
| Redwood synthetic encoders (5 methods) | Good scaffolding, needs capacity fixes (done) | ✅ Keep |
| TrojanStego download + integration | Not in spec, useful for OOD transfer | ✅ Keep as OOD only |
| Game scenarios (3 conditions) | Partial overlap with spec's Task A/C | ✅ Keep with fixes |
| `combined_dataset.jsonl` mixing | Confounded, spec warns against this | ⚠️ Restructure |
| Activation extraction skeleton | Good, needs carrier-span support | ✅ Keep, extend |
| Probe training skeleton | Good, needs grouped splits | ✅ Keep, extend |

---

## 4. Interpretation vs Data (separated)

### Data (verified facts)
- Combined dataset: 2990 rows (1990 stego, 1000 clean)
- 50 unique base texts, 5 encoding methods, 1 external source
- All clean negatives come from Redwood synthetic
- All TrojanStego rows are positive
- Each clean text is duplicated ~20× across method counterparts
- Your 3 bugfix commits resolved all 6 identified issues
- Regenerated data files match the fixed code

### Interpretation
- The current pipeline answers a **weaker question** than the PLAN intends. It tests "can activations distinguish stego-looking text from clean text?" not "can activations detect when the model recognizes hidden content."
- High probe scores on the current dataset would be **uninterpretable** without (a) grouped splits, (b) source-balanced training data, and (c) text-only baselines. The probe might just be learning the 50 base texts or the TrojanStego distribution.
- The game scenario task is a promising direction but needs same-system-prompt controls and observed (not expected) labels to produce valid evidence.

---

## 5. Recommended 65h Plan (Friday evening → Sunday evening)

### Core principle: One honest result beats three shaky ones.

### Revised headline claim:
> "Linear probes on late-layer activations detect stego presence on controlled carriers. On an explicit decode task, probes show [evidence of / no evidence of] internal recognition beyond surface artifacts."

### Phase 1: Re-scope & align (Friday evening, 1–2h)
- [ ] Hugo + Nico sync on this review
- [ ] Agree on scope: Redwood-only for headline, TrojanStego as OOD appendix
- [ ] Agree on model: Gemma 3 4B (bf16) for all experiments (per PLAN)

### Phase 2: Fix dataset & splits (Saturday morning, 4–6h)
- [ ] Add `bundle_id` = f"{method}_{base_text_index}" to all Redwood examples
- [ ] Implement grouped split in `06_train_probes.py` (split by `base_text_index`)
- [ ] Add text-only baseline (char features + logistic regression)
- [ ] Add per-method probe breakdown (not just pooled AUROC)

### Phase 3: Build recognition task (Saturday afternoon, 6–8h)
- [ ] Build detect/decode prompts using 4 color payloads (per PLAN §4.2)
- [ ] Run model, score `decode_correct` automatically
- [ ] Define `recognized = is_stego AND decode_correct`
- [ ] Add same-carrier benign-task control (summarize prompt)
- [ ] Extract activations for both prompt contexts

### Phase 4: Run experiments (Saturday evening, 4–6h)
- [ ] Layerwise presence probe (Redwood-only, grouped split)
- [ ] Layerwise recognition probe (decode task)
- [ ] Text-only baseline comparison
- [ ] Per-method breakdown

### Phase 5: Figures & report (Sunday, 8–12h)
- [ ] Figure 1: Layerwise AUROC for `is_stego` (grouped split, vs text baseline)
- [ ] Figure 2: Layerwise AUROC for `recognized` (decode task)
- [ ] Figure 3: Same-carrier control (decode vs benign prompt)
- [ ] Figure 4: Per-method breakdown
- [ ] Optional Figure 5: Train Redwood → test TrojanStego (OOD transfer)
- [ ] Write methods, results, limitations
- [ ] Frame as control-monitor result

### What to DROP for the weekend
- ❌ Mixed-source training (TrojanStego + Redwood in same split)
- ❌ Game scenarios as headline evidence (keep as demo/appendix)
- ❌ `token_position=all` (too expensive, unclear benefit)
- ❌ Multiple models (stick with one)
- ❌ Implicit/few-shot regimes (stretch only)

---

*This review supersedes the preliminary notes in `review-dataset.txt`.*
