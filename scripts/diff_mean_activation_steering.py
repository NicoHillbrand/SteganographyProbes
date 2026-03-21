"""Compute mean-difference steering vectors per layer and apply steering at inference.

Functions:
 - ince: compute and save steering vectors per layer from stored activations.
 - steer_during_inference: given a model, activations, and list of layers, load
   trained probes and steering vectors; when a probe predicts stego, subtract the
   steering vector from the activations for that layer to steer away.

This script assumes activations are saved as numpy arrays per example and layer,
and that trained LogisticRegression probes are saved with joblib under a known
directory. Adjust paths as needed to match your project's conventions.
"""
from pathlib import Path
import numpy as np
import joblib
import typing as t
import os
import torch


def train_diff_mean_steering_vectors(activations_dir: str, out_dir: str):
    """Compute steering vectors per layer and save them.

    Args:
        activations_dir: directory containing two subdirs `non_steg` and `steg`.
            Each subdir should contain .npy files per example with shape
            (num_layers, hidden_dim) or per-layer arrays saved as
            `layer_{i}.npy` with shape (num_examples, hidden_dim).
        out_dir: directory where steering vectors will be saved as
            `steering_layer_{i}.npy`.

    Behavior:
        For each layer index present in both `non_steg` and `steg`, compute
        mean_non_steg - mean_steg and save that vector as the steering vector
        (i.e., direction to subtract to move away from steganographic activation).
    """
    activations_dir = Path(activations_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    non_dir = activations_dir / "non_steg"
    steg_dir = activations_dir / "steg"

    if not non_dir.exists() or not steg_dir.exists():
        raise FileNotFoundError("Expected `non_steg` and `steg` subdirectories in activations_dir")

    # Discover layer files. Support two layouts:
    # 1) per-layer files named layer_{i}.npy in each folder
    # 2) per-example files containing arrays shaped (num_layers, hidden_dim)

    # Try layout 1 first
    non_layer_files = sorted([p for p in non_dir.glob("layer_*.npy")])
    steg_layer_files = sorted([p for p in steg_dir.glob("layer_*.npy")])

    if non_layer_files and steg_layer_files:
        # match by layer index
        for non_f in non_layer_files:
            name = non_f.stem  # layer_{i}
            steg_f = steg_dir / (name + ".npy")
            if not steg_f.exists():
                continue
            non_arr = np.load(non_f)
            steg_arr = np.load(steg_f)
            # non_arr: (N_non, hidden) ; steg_arr: (N_steg, hidden)
            mean_non = non_arr.mean(axis=0)
            mean_steg = steg_arr.mean(axis=0)
            steering = mean_steg - mean_non
            out_path = out_dir / f"steering_{name}.npy"
            np.save(out_path, steering)

    else:
        # layout 2: per-example files. Load all, stack by layer.
        non_files = sorted(non_dir.glob("*.npy"))
        steg_files = sorted(steg_dir.glob("*.npy"))
        if not non_files or not steg_files:
            raise FileNotFoundError("No activation files found in non_steg or steg subdirs")

        # Load first to get num_layers
        sample = np.load(non_files[0])
        if sample.ndim != 2 and sample.ndim != 3:
            raise ValueError("Unexpected activation shape in per-example files")

        # If per-example file has shape (num_layers, hidden_dim), stack along examples
        non_stack = [np.load(p) for p in non_files]
        steg_stack = [np.load(p) for p in steg_files]
        non_stack = np.stack(non_stack, axis=0)  # (N_non, L, H)
        steg_stack = np.stack(steg_stack, axis=0)  # (N_steg, L, H)

        num_layers = non_stack.shape[1]
        for layer in range(num_layers):
            mean_non = non_stack[:, layer, :].mean(axis=0)
            mean_steg = steg_stack[:, layer, :].mean(axis=0)
            steering =  mean_steg - mean_non
            out_path = out_dir / f"steering_layer_{layer}.npy"
            np.save(out_path, steering)


def steer_during_inference(
    model: torch.nn.Module,
    layer_indices: t.List[int],
    probes_dir: str,
    steering_dir: str,
    token_position: str = "mean",
    threshold: float = 0.5,
    steering_strength: float = 1.0,
    module_name_patterns: t.Optional[t.List[str]] = None,
):
    """Register forward hooks to apply steering in real time during model.forward.

    This function registers forward hooks on modules corresponding to the given
    `layer_indices`. When a forward pass produces an activation for a hooked
    layer, the hook runs the corresponding LogisticRegression probe (loaded via
    joblib) on pooled activation features; if the probe predicts stego with
    probability >= `threshold`, the steering vector is subtracted from the
    activation (scaled by `steering_strength`) before the activation continues
    through the model.

    Args:
        model: the torch model instance to attach hooks to.
        layer_indices: list of layer indices to consider for steering.
        probes_dir: directory containing `probe_layer_{i}.joblib` artifacts.
        steering_dir: directory containing `steering_layer_{i}.npy` vectors.
        token_position: how to reduce token dimension before probing: "mean" or
            "last" (default: "mean"). If activation has shape (batch, seq, hidden),
            "mean" pools over seq, "last" takes the final token.
        threshold: probability threshold to trigger steering.
        steering_strength: scalar multiplier for steering vector.
        module_name_patterns: optional list of attribute-path templates to find
            modules. Use `{i}` in templates for layer index. If None, several
            common templates are tried.

    Returns:
        hooks: list of hook handles (so caller can remove them later via `.remove()`).
    """
    probes_dir = Path(probes_dir)
    steering_dir = Path(steering_dir)

    if module_name_patterns is None:
        module_name_patterns = [
            "model.layers.{i}",
            "model.encoder.layers.{i}",
            "transformer.h.{i}",
            "base_model.model.layers.{i}",
            "model.model.layers.{i}",
        ]

    def find_module_by_path(obj: t.Any, path: str):
        parts = path.split(".")
        cur = obj
        for p in parts:
            if not hasattr(cur, p):
                return None
            cur = getattr(cur, p)
        return cur

    hooks = []

    device = None
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    for layer in layer_indices:
        probe_path = probes_dir / f"probe_layer_{layer}.joblib"
        steering_path = steering_dir / f"steering_layer_{layer}.npy"
        if not probe_path.exists() or not steering_path.exists():
            continue

        probe = joblib.load(probe_path)
        steering_vec = np.load(steering_path)

        # find candidate module
        target_module = None
        for tpl in module_name_patterns:
            candidate = tpl.format(i=layer)
            m = find_module_by_path(model, candidate)
            if m is not None:
                target_module = m
                break
        if target_module is None:
            # cannot find module for this layer; skip
            continue

        # closure captures probe and steering_vec
        def make_hook(probe, steering_vec):
            def hook(module, input, output):
                # output may be tensor or tuple; handle tensor case
                out = output
                if isinstance(output, tuple):
                    # common when module returns (out, attn)
                    out = output[0]

                if not isinstance(out, torch.Tensor):
                    return output

                # pool to (batch, hidden)
                if out.dim() == 3:
                    if token_position == "last":
                        feats = out[:, -1, :].detach().cpu().numpy()
                    else:
                        feats = out.mean(dim=1).detach().cpu().numpy()
                elif out.dim() == 2:
                    feats = out.detach().cpu().numpy()
                else:
                    # unsupported shape
                    return output

                # run probe
                try:
                    probs = probe.predict_proba(feats)
                except Exception:
                    return output

                if hasattr(probe, "classes_"):
                    try:
                        steg_idx = int(np.where(probe.classes_ == 1)[0][0])
                    except Exception:
                        steg_idx = 1
                else:
                    steg_idx = 1

                steg_probs = probs[:, steg_idx]
                mask = steg_probs >= threshold
                if not mask.any():
                    return output

                # prepare steering tensor on correct device/dtype
                sv = torch.from_numpy(steering_vec.astype(np.float32)).to(device=out.device)
                if out.dim() == 3:
                    # subtract from all token positions for masked batch elements
                    out_mod = out.clone()
                    for i, mflag in enumerate(mask):
                        if mflag:
                            out_mod[i] = out_mod[i] - steering_strength * sv
                    if isinstance(output, tuple):
                        return (out_mod,) + tuple(output[1:])
                    return out_mod
                else:
                    out_mod = out.clone()
                    out_mod[mask] = out_mod[mask] - steering_strength * sv
                    if isinstance(output, tuple):
                        return (out_mod,) + tuple(output[1:])
                    return out_mod

            return hook

        h = target_module.register_forward_hook(make_hook(probe, steering_vec))
        hooks.append(h)

    return hooks


def steer_activations_dict(
    layer_indices: t.List[int],
    activations: t.Dict[int, np.ndarray],
    probes_dir: str,
    steering_dir: str,
    threshold: float = 0.5,
    steering_strength: float = 1.0,
):
    """Backward-compatible helper: apply steering to a precomputed activations dict.

    This preserves the previous behavior; prefer `steer_during_inference` to
    register hooks for live inference steering.
    """
    new_activations = activations.copy()
    probes_dir = Path(probes_dir)
    steering_dir = Path(steering_dir)

    for layer in layer_indices:
        probe_path = probes_dir / f"probe_layer_{layer}.joblib"
        steering_path = steering_dir / f"steering_layer_{layer}.npy"
        if not probe_path.exists() or not steering_path.exists():
            # skip missing resources
            continue

        probe = joblib.load(probe_path)
        steering_vec = np.load(steering_path)

        act = activations.get(layer)
        if act is None:
            continue

        # Support single example (1D) or batch (2D)
        single = False
        if act.ndim == 1:
            act_batch = act[None, :]
            single = True
        else:
            act_batch = act

        # Probe expects features shaped (n_samples, n_features)
        probs = probe.predict_proba(act_batch)
        if hasattr(probe, "classes_"):
            try:
                steg_idx = int(np.where(probe.classes_ == 1)[0][0])
            except Exception:
                steg_idx = 1
        else:
            steg_idx = 1

        steg_probs = probs[:, steg_idx]
        # For examples above threshold, subtract steering vector
        mask = steg_probs >= threshold
        if mask.any():
            sv = steering_vec
            if sv.ndim == 1 and sv.shape[0] == act_batch.shape[1]:
                act_batch[mask] = act_batch[mask] - steering_strength * sv
            else:
                try:
                    act_batch[mask] = act_batch[mask] - steering_strength * sv
                except Exception:
                    pass

        if single:
            new_activations[layer] = act_batch[0]
        else:
            new_activations[layer] = act_batch

    return new_activations


if __name__ == "__main__":
    print("diff_mean_activation_steering module - functions: train_diff_mean_steering_vectors, steer_during_inference")
