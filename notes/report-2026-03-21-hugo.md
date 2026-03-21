# Report — 2026-03-21

**Author:** Hugo  
**Branch:** `hugo`  
**Scope:** Follow-up analysis and pipeline repair after the first completed Redwood-only probe runs.

## 1. Why this work was necessary

The larger experiment is trying to answer two different questions:

1. **Presence detection:** can a probe tell whether a carrier contains steganography?
2. **Recognition detection:** can a probe tell when the model is internally recognizing or decoding the hidden content?

The first completed runs already suggested a strong result on presence detection, but they did not yet support a strong claim on recognition. Two issues blocked clean interpretation:

- The decode-task scoring logic was too brittle. It only counted a response as decodable if the model mentioned exactly one valid color label anywhere in the output. Any explanatory answer that mentioned multiple labels was scored as failure, even when the model ended with a correct final label.
- The headline summaries were still too pooled. They hid the fact that several channels are trivially detectable from raw text alone, while the scientifically interesting case is whether harder channels such as `synonym` leave a signal in activations beyond surface artifacts.

Because of that, the highest-value work was not another large experiment. It was a pipeline repair and re-analysis pass that could either strengthen or falsify the initial interpretation at low cost.

## 2. What I changed and why

### 2.1 Decode scoring became configurable and less brittle

I updated `scripts/decode_scoring.py` to support multiple parse modes:

- `unique_anywhere`: old strict behavior
- `last_label`: use the last valid label mentioned
- `anchored_last`: recover a final answer from answer-like lines near the end

**Why:** the experiment cares about whether the model arrived at the right decoded payload, not whether it happened to mention only one label while reasoning out loud. The old parser was undercounting successful decodes.

### 2.2 Probe training now measures the right failure modes

I expanded `scripts/06_train_probes.py` to add:

- decode-label rescoring from saved responses
- automatic class balancing for skewed tasks
- average precision in addition to AUROC
- threshold-selected metrics in addition to default `0.5` threshold metrics
- subset analysis:
  - all methods
  - `synonym_only`
  - `hard_plus_acrostic`
  - `easy_methods`
- cross-method transfer matrices for stego detection

**Why:** AUROC alone hides the operational weakness of rare-positive tasks. For decode recognition, the positive class is tiny, so AP and threshold behavior matter. Subset and transfer analyses are needed to separate real structure from trivial artifacts.

### 2.3 Text baseline runner was fixed and brought into alignment

I repaired `scripts/07_text_baselines.py` so the seeded evaluation wrapper correctly calls both baseline trainers. I also aligned baseline reporting with the probe script:

- AUROC
- average precision
- tuned-threshold metrics
- method-subset comparisons on the same grouped splits

**Why:** the baseline is the main guardrail against overclaiming. If a text-only model already solves a method, activation-probe strength on that method is not evidence of hidden internal recognition.

### 2.4 Plotting now reflects the actual experimental question

I replaced the old figure logic in `scripts/08_plot_results.py` with:

1. layerwise presence AUROC
2. layerwise decode AUROC
3. best-layer method-subset comparison
4. per-method activation-vs-text comparison
5. cross-method transfer heatmap

**Why:** the previous “same-carrier control” figure was not a valid like-for-like comparison. It overlaid AUROC from different target labels (`recognized` vs `is_stego`) and therefore could not justify the conclusion it visually suggested.

### 2.5 Decode prompt generation now supports a constrained answer mode

I added `--decode_prompt_style label_only` to `scripts/04b_build_decode_task.py`.

**Why:** the current decode experiment is bottlenecked by the model producing free-form answers that are difficult to score and often do not end in a clean label. The fastest next falsification test is to force exact one-token labels and see whether recognition actually improves.

## 3. What I ran

I reran the analysis stack on the current Redwood-only dataset and saved the refreshed outputs:

- `python scripts/07_text_baselines.py`
- `python scripts/06_train_probes.py --task stego_detection`
- `python scripts/06_train_probes.py --task decode_task --decode_rescore_mode anchored_last`
- `python scripts/08_plot_results.py`

I also recompiled the edited scripts with `python -m py_compile`.

## 4. Results

### 4.1 Presence detection remains strong, but pooled scores are inflated by easy methods

Best stego-detection result:

- best layer: `3`
- AUROC: `0.9336`
- average precision: `0.9563`
- F1 at default threshold: `0.8444`

This is a real result, but it is not uniform across methods.

Best-layer subset breakdown:

- all methods: `0.9336` AUROC
- easy methods: `1.0000` AUROC
- synonym + acrostic: `0.8248` AUROC
- synonym only: `0.7545` AUROC

Text baseline comparison:

- all methods char baseline: `0.8660` AUROC
- synonym-only char baseline: `0.4988` AUROC
- synonym + acrostic char baseline: `0.6981` AUROC
- easy methods char baseline: `1.0000` AUROC

**Interpretation:** most of the pooled presence-detection strength is carried by trivial or near-trivial channels (`prefix_binary`, `unicode`, `whitespace`). The strongest nontrivial evidence is `synonym`: the activation probe is materially above the text baseline there.

### 4.2 Recognition remains weak even after fixing the parser

Decode-task rescoring with `anchored_last` changed the observed label distribution:

- previous recognized positives: `43`
- rescored recognized positives: `53`

Best decode result:

- best layer: `31`
- AUROC: `0.7209`
- average precision: `0.1034`
- F1 at default threshold: `0.1275`
- tuned-threshold F1: `0.0810`
- precision at default threshold: `0.0888`
- recall at default threshold: `0.2327`

**Interpretation:** parser repair helped a little, but it did not transform the result. The decode probe still supports only a weak ranking signal. It does **not** support a strong claim that we can reliably detect internal recognition from these activations under the current prompt/task setup.

### 4.3 The transfer matrix is informative

The cross-method transfer analysis for presence detection shows:

- probes trained on easy methods often fail to transfer to `synonym`
- probes trained on `synonym` transfer moderately to several other methods

**Interpretation:** `synonym` appears to contain a less surface-trivial signal, while several other methods are probably being detected by method-specific artifacts. This is exactly why the hard-method slices are more scientifically meaningful than the pooled score.

### 4.4 Some decode per-method analyses are unstable

For several decode per-method runs, grouped splits fail because the positive class is too sparse in individual methods.

**Interpretation:** this is not just a tooling nuisance. It is evidence that the current recognition task is underpowered. The model is rarely producing cleanly recognizable correct decodes, so per-method recognition comparisons are not yet robust enough to anchor a headline claim.

## 5. What this means for the larger experiment

At this point the project supports an honest claim of the form:

> Late-layer probes detect stego presence on controlled carriers, and the strongest nontrivial evidence comes from the synonym channel rather than from trivially visible encodings.

It does **not** yet support the stronger claim:

> Probes reliably detect when the model internally recognizes or decodes the hidden payload.

That distinction matters. The entire scientific value of the recognition experiment is that it goes beyond “stego-looking text exists” and toward “the model is in a decode-like internal state.” The current decode results do not yet clear that bar.

So the right strategic move is not to oversell the weak recognition result. It is to run the fastest test that could materially improve or cleanly falsify it.

## 6. Highest-ROI next steps

### 1. Rebuild the decode task with `label_only` answers and rerun decode inference

**ROI:** highest  
**Cost:** low to moderate GPU time  
**Expected value:** very high

This is the clearest next test because the main current bottleneck is answer format. The model often produces analysis-heavy responses. Even after better parsing, recognized positives remain scarce and per-method splits are unstable.

If constrained label-only decoding produces a large jump in:

- recognized positives
- decode AP
- decode F1
- per-method stability

then the current weak recognition result was partly a prompt/scoring artifact. If it does not, then the recognition claim is genuinely weak under this setup and we should stop spending cycles trying to rescue it.

This is a high-information, low-cost falsification step.

### 2. Treat hard-method reporting as the default headline, not the appendix

**ROI:** high  
**Cost:** near zero  
**Expected value:** high

The method-subset breakdown already shows that pooled performance overstates the generality of the presence result. Any report, slide, or write-up should foreground:

- `synonym_only`
- `synonym + acrostic`
- text baseline comparisons

Why this matters: it protects the experiment from a predictable criticism, namely that the probe is just learning visible encoding artifacts. This is not a new experiment, but it has very high ROI because it changes the honesty and durability of the claim immediately.

### 3. Use the transfer matrix as a gate for future carrier investment

**ROI:** medium-high  
**Cost:** already paid  
**Expected value:** medium-high

The transfer results suggest that `synonym` is the most promising existing channel for studying nontrivial internal detection. If we need one channel to prioritize for follow-up experiments, it should be that one.

Why: a channel that is text-trivial but non-transferable is low scientific value. A channel whose signal transfers more broadly is more likely to reflect a shared representation rather than a method-specific surface marker.

### 4. Deprioritize game scenarios as headline evidence

**ROI:** medium  
**Cost saved:** high  
**Expected value:** mostly from avoided waste

The current game-scenario result is small-sample and noisy. It is not the fast route to a defensible headline. The better path is to settle the explicit decode experiment first.

This is not because game scenarios are useless. It is because they are expensive and currently underpowered relative to the information yield of a constrained decode rerun.

## 7. Recommended narrative for teammates

The clean internal summary is:

- The infrastructure repairs worked.
- The original strong presence result survives.
- The stronger recognition story does not yet survive stress-testing.
- The new subset and transfer analyses identify `synonym` as the most meaningful carrier.
- The next rational spend is a constrained-answer decode rerun, not a broad new experimental branch.

That is a good outcome. It narrows the claim, surfaces the real scientific signal, and reduces the chance of spending GPU budget on the wrong next step.

## 8. Files changed in this pass

- `scripts/decode_scoring.py`
- `scripts/06_train_probes.py`
- `scripts/07_text_baselines.py`
- `scripts/08_plot_results.py`
- `scripts/04b_build_decode_task.py`
- `data/probe_results/stego_detection_probe_results.json`
- `data/probe_results/decode_task_probe_results.json`
- `data/probe_results/text_baseline_results.json`
- `data/figures/fig2_layerwise_decode_task.png`
- `data/figures/fig3_method_subset_comparison.png`
- `data/figures/fig4_per_method_comparison.png`
- `data/figures/fig5_cross_method_transfer.png`

## 9. Bottom line

This pass improved the experiment mainly by reducing ambiguity:

- it removed a decode-scoring bottleneck,
- exposed how much pooled performance depends on easy methods,
- showed that `synonym` is the strongest nontrivial channel,
- and clarified that recognition is still weak enough that it needs one focused rerun before stronger claims are justified.

That is the correct place to be before spending more GPU time.
