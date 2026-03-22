# Agent Instructions

**If you hit a surprising behavior, a non-obvious fix, an OOM, a silent failure, or any quirk that cost you time -- add it here.** This file exists so the same problem is never solved twice.

---

## Inference: VRAM and Incremental Saving

- **batch_size=16 is the max safe value.** 32 OOMs during activation extraction (65 hidden-state layers materialize at once). 16 gives ~55 responses/min, ~36 min generation + ~15 min extraction = ~50 min for 2000 examples.
- **The pipeline does not save incrementally.** If it OOMs mid-run, all generated responses are lost. Add incremental flushing to disk before the next full run.

## Probe Saving

When training probes with `scripts/06_train_probes.py`, **always pass `--save_probes`** so that trained probe weights are persisted. Without this flag the fitted LogisticRegression and StandardScaler objects are discarded after evaluation and teammates cannot reuse them.

```bash
python scripts/06_train_probes.py --run_dir <run_dir> --save_probes
```

Probes are saved to `<run_dir>/probe_results/trained_probes/layer_XX/seed_N.joblib`. Each file contains:

```python
{"probe": LogisticRegression, "scaler": StandardScaler, "seed": int, "class_weight": str}
```

To load:

```python
import joblib
artifact = joblib.load("probe_results/trained_probes/layer_19/seed_0.joblib")
probe = artifact["probe"]
scaler = artifact["scaler"]
```
