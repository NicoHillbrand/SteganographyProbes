# 2026-03-21 Addendum: Decode Prompt Pilot, Full Single-Token Rerun, And LoRA Smoke Test

## What I changed

- Added a stricter decode prompt style in [scripts/04b_build_decode_task.py](/home/ubuntu/SteganographyProbes/scripts/04b_build_decode_task.py): `single_token`.
- Updated [scripts/05_run_inference.py](/home/ubuntu/SteganographyProbes/scripts/05_run_inference.py) so decode runs can:
  - use deterministic generation via `--generation_mode greedy`
  - save the decode dataset path in metadata
  - score decode outputs during inference with a configurable `--decode_score_mode`
- Updated [scripts/06_train_probes.py](/home/ubuntu/SteganographyProbes/scripts/06_train_probes.py) so decode-task probe runs can recover per-example methods from a non-default decode dataset path.
- Added [scripts/09_summarize_decode_runs.py](/home/ubuntu/SteganographyProbes/scripts/09_summarize_decode_runs.py) to compare decode prompt styles from saved `responses.jsonl`.
- Extended [.gitignore](/home/ubuntu/SteganographyProbes/.gitignore) so new large run artifacts land in `data/runs/` and stay out of plain Git history.

## Why I changed it

The original decode recognition result was bottlenecked by free-form outputs and weak label recovery. The fastest falsification test was to constrain the decode response format, compare prompt styles on the same carrier slice, and only scale if the gain looked broad rather than trivial.

At the same time, the repo was already carrying large binary artifacts in Git. I did not rewrite history. Instead I made new experiment outputs land under `data/runs/` so we stop making the Git problem worse while still committing compact summaries and notes.

## What I ran

### Decode prompt pilot

Built three 200-carrier decode task variants and ran greedy decode inference on all three:

- `analysis`
- `label_only`
- `single_token`

Pilot summary saved to [data/probe_results/decode_prompt_pilot_summary.json](/home/ubuntu/SteganographyProbes/data/probe_results/decode_prompt_pilot_summary.json).

### Full decode rerun

Promoted `single_token` to a full 2000-carrier rerun because it had the highest recognized-positive rate on the pilot.

Full result JSONs saved to:

- [data/probe_results/decode_task_single_token_probe_results.json](/home/ubuntu/SteganographyProbes/data/probe_results/decode_task_single_token_probe_results.json)
- [data/probe_results/benign_task_single_token_probe_results.json](/home/ubuntu/SteganographyProbes/data/probe_results/benign_task_single_token_probe_results.json)

### LoRA path validation

Executed [scripts/00_validate_model.py](/home/ubuntu/SteganographyProbes/scripts/00_validate_model.py) successfully on the GH200.

Then ran a 5-prompt smoke generation with [scripts/01b_generate_lora_carriers.py](/home/ubuntu/SteganographyProbes/scripts/01b_generate_lora_carriers.py), writing ignored artifacts under `data/runs/lora_smoke/`.

## Results

### Pilot decode comparison

On the 200-carrier pilot slice:

- `analysis`: `1/200` recognized, `0.005` recognized rate, average response length `49.7` tokens
- `label_only`: `13/200` recognized, `0.065` recognized rate, average response length `1.0` token
- `single_token`: `24/200` recognized, `0.120` recognized rate, average response length `1.0` token

This was enough to justify scaling `single_token`, but the per-method breakdown already showed the gain was concentrated in `whitespace`.

### Full single-token decode behavior

The full `single_token` rerun produced `196/2000` recognized positives, up from the previous `53/2000`.

That increase was not broad:

- `whitespace`: `192/200` recognized
- `acrostic`: `4/200`
- `prefix_binary`: `0/200`
- `synonym`: `0/200`
- `unicode`: `0/200`

Predicted labels were overwhelmingly one class:

- `GREEN`: `1904`
- `BLACK`: `58`
- `BLUE`: `37`
- `RED`: `1`

Among clean examples, predictions were still mostly `GREEN` rather than `NONE`, so the prompt is not producing calibrated decode outputs. It is mostly forcing the model into a dominant token that happens to align with the whitespace carrier.

### Full decode probe result

On the full `single_token` rerun:

- best layer: `31`
- `AUROC 0.678`
- `AP 0.249`
- `F1 0.226`
- tuned-threshold `F1 0.261`

This is an improvement over the earlier decode readout in raw positive count and AP, but it is still not a strong or general recognition result.

Per-method decode probing is effectively non-informative outside whitespace because the positive class is absent or near-absent for the harder methods.

### Benign control on the same carrier set

On the full same-carrier benign task:

- best layer: `30`
- `AUROC 0.719`
- `AP 0.747`

This remains stronger than the decode-task probe, which is the opposite of what we would want if the main added signal were a broad internal decode/recognition state.

### LoRA smoke validation

The LoRA path is operational:

- GH200 validation passed
- LoRA adapter toggle changes generations
- a 5-prompt smoke run generated 10 paired stego/clean outputs

This means matched `token_bucket` carrier generation is ready for a real run.

## Interpretation

The decode prompt fix worked mechanically but not scientifically.

It increased recognized positives by collapsing the output space, but the added positives came almost entirely from the easiest carrier (`whitespace`). That means the rerun did not rescue the broader recognition claim. It mostly made the model emit a token that aligns with an already-trivial channel.

The right interpretation is:

- the decode prompt format was a real bottleneck for raw label count
- fixing that bottleneck did **not** uncover broad recognition of hard methods
- the remaining decode signal is still too narrow and too weak relative to the same-carrier benign control to justify a general “recognition” claim

## Highest-ROI next steps

1. Stop spending GPU on general Redwood decode prompt sweeps.
   The prompt-format question is answered well enough. More variants are unlikely to change the fact that the gain is concentrated in whitespace.

2. Use the next GPU block on matched LoRA carrier generation.
   This is now the best new information path because it gives a new carrier family with matched clean controls and a materially different mechanism (`token_bucket` in the weights rather than visible text tricks).

3. Keep the main present-tense claim centered on presence detection, especially `synonym`.
   Existing results already show the best nontrivial signal there, and the decode rerun did not change that hierarchy.

4. If decode is revisited at all, narrow it to hard methods only.
   A future decode follow-up should exclude the easy methods from the start so we can tell whether anything remains once whitespace is gone.
