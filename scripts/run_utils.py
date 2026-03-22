"""
Shared utilities for run-based output directory management.

All result-producing scripts (inference, probes, baselines, figures) use these
helpers to write outputs into an isolated, timestamped run directory:

    data/<model>/runs/YYYY-MM-DD_<description>/
        config.json
        activations/
        probe_results/
        text_baselines/
        figures/
"""

import json
import os
import subprocess
import sys
from datetime import datetime


def sanitize_model_name(model_name: str) -> str:
    """Convert HuggingFace model ID to a filesystem-safe directory name.

    'meta-llama/Meta-Llama-3-8B-Instruct' -> 'Meta-Llama-3-8B-Instruct'
    """
    # Take the part after the last slash (org/model -> model)
    name = model_name.split("/")[-1]
    # Replace any remaining filesystem-unsafe characters
    for ch in ["<", ">", ":", '"', "|", "?", "*"]:
        name = name.replace(ch, "_")
    return name


def create_run_dir(
    model_name: str,
    scenario_path: str | None = None,
    dataset_path: str | None = None,
    run_name: str | None = None,
    base_dir: str = "data",
) -> str:
    """Create and return the path to a new run directory.

    Directory structure: <base_dir>/<model>/runs/YYYY-MM-DD_<description>/

    Args:
        model_name: HuggingFace model ID (e.g. 'meta-llama/Meta-Llama-3-8B-Instruct')
        scenario_path: Path to scenario JSONL (used to auto-generate description)
        dataset_path: Path to dataset JSONL (fallback for description)
        run_name: Explicit run name override (skips auto-generation)
        base_dir: Root data directory (default: 'data')

    Returns:
        Absolute or relative path to the created run directory.
    """
    model_dir = sanitize_model_name(model_name)
    date_str = datetime.now().strftime("%Y-%m-%d")

    if run_name:
        dir_name = f"{date_str}_{run_name}"
    elif scenario_path:
        desc = os.path.splitext(os.path.basename(scenario_path))[0]
        dir_name = f"{date_str}_{desc}"
    elif dataset_path:
        desc = os.path.splitext(os.path.basename(dataset_path))[0]
        dir_name = f"{date_str}_{desc}"
    else:
        dir_name = f"{date_str}_run"

    run_dir = os.path.join(base_dir, model_dir, "runs", dir_name)

    # If directory already exists, append a counter
    if os.path.exists(run_dir):
        counter = 2
        while os.path.exists(f"{run_dir}_{counter}"):
            counter += 1
        run_dir = f"{run_dir}_{counter}"

    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_run_config(run_dir: str, args_dict: dict, extra_metadata: dict | None = None):
    """Save a config.json capturing the full run configuration.

    Args:
        run_dir: Path to the run directory.
        args_dict: Dictionary of script arguments (e.g. vars(args)).
        extra_metadata: Optional additional metadata to include.
    """
    try:
        import torch
        torch_version = torch.__version__
    except ImportError:
        torch_version = None

    config = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": _get_git_hash(),
        "python_version": sys.version,
        "torch_version": torch_version,
        "args": args_dict,
    }
    if extra_metadata:
        config.update(extra_metadata)

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, default=str)

    return config_path


def find_latest_run(
    model_name: str,
    pattern: str | None = None,
    base_dir: str = "data",
) -> str | None:
    """Find the most recent run directory for a model.

    Args:
        model_name: HuggingFace model ID.
        pattern: Optional substring to filter run names (e.g. 'whitespace').
        base_dir: Root data directory.

    Returns:
        Path to the latest matching run directory, or None.
    """
    model_dir = sanitize_model_name(model_name)
    runs_dir = os.path.join(base_dir, model_dir, "runs")

    if not os.path.exists(runs_dir):
        return None

    candidates = sorted(os.listdir(runs_dir), reverse=True)

    if pattern:
        candidates = [c for c in candidates if pattern in c]

    if not candidates:
        return None

    return os.path.join(runs_dir, candidates[0])


def _get_git_hash() -> str | None:
    """Get the current git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None
