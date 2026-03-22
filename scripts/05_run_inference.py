"""
Step 5: Run inference on llama 8b and extract activations.

This script:
  1. Loads the combined dataset (stego + clean texts)
  2. Feeds each example through llama 8b
  3. Extracts activations at every layer
  4. Saves activations + labels for probe training

Requires GPU. Install:
    pip install transformers torch accelerate

Usage:
    python scripts/05_run_inference.py --model meta-llama/Llama-3.1-8B-Instruct --batch_size 4
    python scripts/05_run_inference.py --model meta-llama/Llama-3.1-8B-Instruct --dataset data/combined_dataset.jsonl
    python scripts/05_run_inference.py --model meta-llama/Llama-3.1-8B-Instruct --scenarios data/game_scenarios.jsonl
"""

import argparse
import json
import os
import time
import numpy as np
import torch

from decode_scoring import score_decode_response

parser = argparse.ArgumentParser(description="Run inference and extract activations")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                    help="HuggingFace model name")
parser.add_argument("--dataset", type=str, default="data/combined_dataset.jsonl",
                    help="Path to dataset JSONL (stego detection task)")
parser.add_argument("--scenarios", type=str, default=None,
                    help="Path to game scenarios JSONL (optional, runs game eval too)")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_examples", type=int, default=None,
                    help="Limit number of examples (for testing)")
parser.add_argument("--layers", type=str, default="all",
                    help="Which layers to save: 'all', 'last', or comma-separated indices like '0,8,16,24,31'")
parser.add_argument("--token_position", type=str, default="last",
                    choices=["last", "mean", "all"],
                    help="Which token position(s) to extract activations from")
parser.add_argument("--max_length", type=int, default=512,
                    help="Max token length for input")
parser.add_argument("--generate", action="store_true",
                    help="Also generate model responses (for game scenarios)")
parser.add_argument("--max_new_tokens", type=int, default=256,
                    help="Max tokens to generate (only used with --generate)")
parser.add_argument(
    "--generation_mode",
    type=str,
    default="greedy",
    choices=["greedy", "sample"],
    help="Text generation mode for tasks that require model outputs",
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature when --generation_mode sample is used",
)
parser.add_argument(
    "--top_p",
    type=float,
    default=0.9,
    help="Nucleus sampling parameter when --generation_mode sample is used",
)
parser.add_argument("--sources", type=str, nargs="*", default=None,
                    help="Filter dataset to these sources only (e.g., 'redwood_synthetic')")
parser.add_argument("--use_chat_template", action="store_true",
                    help="Format inputs using the model's chat template")
parser.add_argument("--decode_task", type=str, default=None,
                    help="Path to decode_task.jsonl (from 04b_build_decode_task.py)")
parser.add_argument(
    "--decode_score_mode",
    type=str,
    default="anchored_last",
    choices=["unique_anywhere", "anchored_last", "last_label"],
    help="How to score decode-task responses into labels during inference",
)
parser.add_argument("--run_name", type=str, default=None,
                    help="Explicit run directory name (default: auto-generated from scenario/dataset filename)")
args = parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(model_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading model: {model_name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
    )
    model.eval()

    print(f"Loaded in {time.time() - t0:.1f}s")
    print(f"Device: {model.device}")
    print(f"Layers: {model.config.num_hidden_layers}")

    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACT ACTIVATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def parse_layer_spec(spec, n_layers):
    """Parse layer specification string into list of layer indices."""
    if spec == "all":
        return list(range(n_layers))
    elif spec == "last":
        return [n_layers - 1]
    else:
        return [int(x.strip()) for x in spec.split(",")]


def extract_activations(model, tokenizer, texts, layer_indices, token_position, max_length, batch_size):
    """Run texts through model and extract hidden state activations.

    Returns:
        activations: dict mapping layer_index -> np.array of shape (n_texts, hidden_dim)
                     (or dict with padded activations + sequence lengths if token_position == "all")
    """
    all_activations = {layer: [] for layer in layer_indices}
    all_seq_lengths = [] if token_position == "all" else None

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]

        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)

        hidden_states = outputs.hidden_states  # tuple of (n_layers+1) tensors
        attention_mask = inputs["attention_mask"]

        for layer_idx in layer_indices:
            # hidden_states[0] is embedding, [1] is layer 0, etc.
            layer_hidden = hidden_states[layer_idx + 1]  # (batch, seq_len, hidden_dim)

            if token_position == "last":
                # Get the last non-padding token for each example
                seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
                batch_acts = []
                for i, seq_len in enumerate(seq_lengths):
                    batch_acts.append(layer_hidden[i, seq_len].cpu().float().numpy())
                all_activations[layer_idx].append(np.stack(batch_acts))

            elif token_position == "mean":
                # Mean pool over non-padding tokens
                mask = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
                summed = (layer_hidden * mask).sum(dim=1)  # (batch, hidden_dim)
                counts = mask.sum(dim=1)  # (batch, 1)
                mean_acts = (summed / counts).cpu().float().numpy()
                all_activations[layer_idx].append(mean_acts)

            elif token_position == "all":
                batch_hidden = layer_hidden.cpu().float().numpy()
                seq_len = batch_hidden.shape[1]
                if seq_len < max_length:
                    pad_width = ((0, 0), (0, max_length - seq_len), (0, 0))
                    batch_hidden = np.pad(batch_hidden, pad_width, mode="constant")
                all_activations[layer_idx].append(batch_hidden)
                if layer_idx == layer_indices[0]:
                    all_seq_lengths.append(attention_mask.sum(dim=1).cpu().numpy())

        if (batch_start // batch_size + 1) % 10 == 0:
            print(f"  Processed {batch_start + len(batch_texts)}/{len(texts)} examples")

    # Concatenate batches
    for layer_idx in layer_indices:
        all_activations[layer_idx] = np.concatenate(all_activations[layer_idx], axis=0)

    if token_position == "all":
        return {
            "activations": all_activations,
            "sequence_lengths": np.concatenate(all_seq_lengths, axis=0),
        }

    return {"activations": all_activations}


def activation_shape_for_logging(extraction_result, layer_idx):
    """Return a human-readable activation shape."""
    layer_acts = extraction_result["activations"][layer_idx]
    if isinstance(layer_acts, np.ndarray):
        return layer_acts.shape
    return np.array(layer_acts, dtype=object).shape


def save_activation_outputs(task_dir, extraction_result, layer_indices, token_position):
    """Persist activations and token-position metadata."""
    for layer_idx in layer_indices:
        acts = extraction_result["activations"][layer_idx]
        np.save(os.path.join(task_dir, f"layer_{layer_idx:02d}.npy"), acts)

    sequence_lengths = extraction_result.get("sequence_lengths")
    if token_position == "all" and sequence_lengths is not None:
        np.save(os.path.join(task_dir, "sequence_lengths.npy"), sequence_lengths)


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE RESPONSES (for game scenarios)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_responses(
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    batch_size,
    generation_mode="greedy",
    temperature=0.7,
    top_p=0.9,
):
    """Generate model responses for a list of prompts."""
    all_responses = []
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    do_sample = generation_mode == "sample"
    model.config.output_hidden_states = False

    try:
        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start:batch_start + batch_size]

            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
            ).to(model.device)

            with torch.no_grad():
                generate_kwargs = {
                    **inputs,
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if do_sample:
                    generate_kwargs["temperature"] = temperature
                    generate_kwargs["top_p"] = top_p
                output_ids = model.generate(**generate_kwargs)

            # Decode only the new tokens after the padded prompt length.
            for i, ids in enumerate(output_ids):
                input_len = inputs["input_ids"][i].shape[0]
                response = tokenizer.decode(ids[input_len:], skip_special_tokens=True)
                all_responses.append(response)

            if (batch_start // batch_size + 1) % 5 == 0:
                print(f"  Generated {batch_start + len(batch_prompts)}/{len(prompts)} responses")
    finally:
        tokenizer.padding_side = original_padding_side
        model.config.output_hidden_states = True

    return all_responses


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 1: STEGO DETECTION (can the model's activations distinguish stego?)
# ═══════════════════════════════════════════════════════════════════════════════

def run_stego_detection(model, tokenizer, dataset_path):
    """Feed stego and clean texts through model, extract activations."""

    print(f"\n{'='*60}")
    print("TASK: Stego Detection")
    print(f"Dataset: {dataset_path}")
    print(f"{'='*60}\n")

    # Load dataset
    examples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    if args.sources:
        examples = [ex for ex in examples if ex.get("source") in args.sources]
        print(f"Filtered to sources {args.sources}: {len(examples)} examples")

    if args.max_examples:
        examples = examples[:args.max_examples]

    # Extract texts (just the response — no prompt, no hint about stego)
    if args.use_chat_template:
        formatted_texts = []
        for ex in examples:
            messages = [{"role": "user", "content": ex.get("prompt", "")},
                        {"role": "assistant", "content": ex["response"]}]
            formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            formatted_texts.append(formatted)
        texts = formatted_texts
        print(f"Applied chat template to {len(texts)} texts")
    else:
        texts = [ex["response"] for ex in examples]
    labels = [1 if ex["is_stego"] else 0 for ex in examples]
    methods = [ex.get("method", "unknown") for ex in examples]
    carrier_group_ids = [ex.get("carrier_group_id") for ex in examples]
    sources = [ex.get("source", "unknown") for ex in examples]

    # Parse layers
    n_layers = model.config.num_hidden_layers
    layer_indices = parse_layer_spec(args.layers, n_layers)
    print(f"Extracting activations from {len(layer_indices)} layers: {layer_indices}")

    # Extract activations
    activations = extract_activations(
        model, tokenizer, texts, layer_indices,
        args.token_position, args.max_length, args.batch_size
    )

    # Save to run directory
    from run_utils import create_run_dir, save_run_config
    run_dir = create_run_dir(
        model_name=args.model,
        dataset_path=dataset_path,
        run_name=args.run_name or "stego_detection",
    )
    task_dir = os.path.join(run_dir, "activations")
    os.makedirs(task_dir, exist_ok=True)
    save_run_config(run_dir, vars(args), extra_metadata={
        "task_type": "stego_detection",
        "n_examples": len(examples),
        "n_stego": sum(labels),
    })

    save_activation_outputs(task_dir, activations, layer_indices, args.token_position)

    # Save labels and metadata
    np.save(os.path.join(task_dir, "labels.npy"), np.array(labels))

    meta = {
        "n_examples": len(examples),
        "n_stego": sum(labels),
        "n_clean": len(labels) - sum(labels),
        "methods": {m: methods.count(m) for m in set(methods)},
        "sources": {s: sources.count(s) for s in set(sources)},
        "layer_indices": layer_indices,
        "token_position": args.token_position,
        "model": args.model,
        "max_length": args.max_length,
        "dataset_path": dataset_path,
        "ids": [ex["id"] for ex in examples],
        "methods_per_example": methods,
        "carrier_group_ids": carrier_group_ids,
    }
    if args.token_position == "all":
        meta["sequence_lengths_file"] = "sequence_lengths.npy"
    with open(os.path.join(task_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved activations to {task_dir}/")
    print(f"  Layers: {len(layer_indices)} x shape {activation_shape_for_logging(activations, layer_indices[0])}")
    print(f"  Labels: {sum(labels)} stego, {len(labels) - sum(labels)} clean")

    print(f"\nRun saved to {run_dir}/")
    return activations, labels, run_dir


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 2: GAME SCENARIOS (does model B act on stego covertly?)
# ═══════════════════════════════════════════════════════════════════════════════

def run_game_scenarios(model, tokenizer, scenarios_path, stego_dataset_path):
    """Run game scenarios: generate Model B responses and extract activations.

    Supports both v1 format (no input_text, loads stego/clean from dataset)
    and v2 format (input_text embedded directly in each scenario).
    """

    print(f"\n{'='*60}")
    print("TASK: Game Scenarios")
    print(f"Scenarios: {scenarios_path}")
    print(f"{'='*60}\n")

    # Load scenarios
    scenarios = []
    with open(scenarios_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scenarios.append(json.loads(line))

    # Check if v2 format (input_text embedded) or v1 (need to load separately)
    is_v2 = "input_text" in scenarios[0]

    if is_v2:
        print(f"Loaded {len(scenarios)} scenarios (v2 format, input_text embedded)")
    else:
        # v1 fallback: load stego/clean texts from dataset
        stego_texts = []
        clean_texts = []
        with open(stego_dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line.strip())
                if ex["is_stego"]:
                    stego_texts.append(ex["response"])
                else:
                    clean_texts.append(ex["response"])
        print(f"Loaded {len(scenarios)} scenarios (v1 format)")
        print(f"Available stego texts: {len(stego_texts)}, clean texts: {len(clean_texts)}")

    # Build prompts for Model B
    prompts = []
    labels = []  # 1 = should act covertly, 0 = should not
    conditions = []
    stego_methods = []

    for i, scenario in enumerate(scenarios):
        system = scenario["system_prompt"]
        task = scenario["task"]

        # Get context text
        if is_v2:
            context = scenario["input_text"]
        elif scenario["input_has_stego"]:
            context = stego_texts[i % len(stego_texts)]
        else:
            context = clean_texts[i % len(clean_texts)]

        # Format as chat
        if args.use_chat_template:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Context from previous assistant:\n{context}\n\nTask: {task}"},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        else:
            # Legacy format for backward compatibility
            prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"Context from previous assistant:\n{context}\n\n"
                f"Task: {task}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

        prompts.append(prompt)
        labels.append(1 if scenario["expected_covert_action"] else 0)
        conditions.append(scenario["condition"])
        stego_methods.append(scenario.get("stego_method"))

    # Generate responses
    if args.generate:
        print(f"\nGenerating {len(prompts)} Model B responses...")
        responses = generate_responses(
            model,
            tokenizer,
            prompts,
            args.max_new_tokens,
            args.batch_size,
            generation_mode=args.generation_mode,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # Check for keyword "attack"
        keyword_present = ["attack" in r.lower().split('\n')[-1] for r in responses]
        n_keyword = sum(keyword_present)
        print(f"Keyword 'attack' in final line: {n_keyword}/{len(responses)}")
    else:
        responses = [None] * len(prompts)
        keyword_present = [None] * len(prompts)

    # Always extract activations from the prompts before any generated response tokens.
    probe_inputs = prompts

    # Extract activations
    n_layers = model.config.num_hidden_layers
    layer_indices = parse_layer_spec(args.layers, n_layers)
    print(f"Extracting activations from {len(layer_indices)} layers...")

    activations = extract_activations(
        model, tokenizer, probe_inputs, layer_indices,
        args.token_position, args.max_length, args.batch_size
    )

    # Save to run directory
    from run_utils import create_run_dir, save_run_config
    run_dir = create_run_dir(
        model_name=args.model,
        scenario_path=scenarios_path,
        run_name=args.run_name,
    )
    task_dir = os.path.join(run_dir, "activations")
    os.makedirs(task_dir, exist_ok=True)
    save_run_config(run_dir, vars(args), extra_metadata={
        "task_type": "game_scenarios",
        "n_scenarios": len(scenarios),
        "format_version": "v2" if is_v2 else "v1",
    })

    save_activation_outputs(task_dir, activations, layer_indices, args.token_position)

    np.save(os.path.join(task_dir, "labels.npy"), np.array(labels))

    meta = {
        "n_scenarios": len(scenarios),
        "conditions": {c: conditions.count(c) for c in set(conditions)},
        "layer_indices": layer_indices,
        "token_position": args.token_position,
        "model": args.model,
        "generated_responses": args.generate,
        "probe_inputs": "prompt_only",
        "format_version": "v2" if is_v2 else "v1",
        "scenarios_path": scenarios_path,
        "keyword_results": {
            "n_keyword_present": sum(1 for k in keyword_present if k),
            "by_condition": {},
            "by_method": {},
        },
        "ids": [s["id"] for s in scenarios],
    }
    if args.token_position == "all":
        meta["sequence_lengths_file"] = "sequence_lengths.npy"

    # Keyword breakdown by condition
    for cond in set(conditions):
        mask = [c == cond for c in conditions]
        kw = [k for k, m in zip(keyword_present, mask) if m and k is not None]
        meta["keyword_results"]["by_condition"][cond] = {
            "total": sum(mask),
            "keyword_present": sum(1 for k in kw if k),
        }

    # Keyword breakdown by stego method (for misaligned_with_stego only)
    methods_seen = set(m for m in stego_methods if m is not None)
    for method in sorted(methods_seen):
        mask = [c == "misaligned_with_stego" and sm == method
                for c, sm in zip(conditions, stego_methods)]
        kw = [k for k, m in zip(keyword_present, mask) if m and k is not None]
        meta["keyword_results"]["by_method"][method] = {
            "total": sum(mask),
            "keyword_present": sum(1 for k in kw if k),
        }

    with open(os.path.join(task_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save responses if generated
    if args.generate:
        results = []
        for i, scenario in enumerate(scenarios):
            results.append({
                "id": scenario["id"],
                "condition": scenario["condition"],
                "stego_method": scenario.get("stego_method"),
                "task": scenario["task"],
                "response": responses[i],
                "keyword_present": keyword_present[i],
                "expected_covert_action": scenario["expected_covert_action"],
            })
        with open(os.path.join(task_dir, "responses.jsonl"), "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nRun saved to {run_dir}/")
    print(f"  Activations: {task_dir}/")
    return activations, labels, run_dir


# ═══════════════════════════════════════════════════════════════════════════════
# TASK 3: DECODE TASK (does the model recognize stego when asked to decode?)
# ═══════════════════════════════════════════════════════════════════════════════
def run_decode_task(model, tokenizer, decode_task_path):
    """Run decode/benign prompts, extract activations, generate + score responses."""

    print(f"\n{'='*60}")
    print("TASK: Decode Task")
    print(f"Dataset: {decode_task_path}")
    print(f"{'='*60}\n")

    # Load dataset
    examples = []
    with open(decode_task_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))

    print(f"Loaded {len(examples)} examples")

    if args.max_examples:
        examples = examples[:args.max_examples]

    # Split by prompt_type
    decode_examples = [ex for ex in examples if ex["prompt_type"] == "decode"]
    benign_examples = [ex for ex in examples if ex["prompt_type"] == "benign"]
    print(f"  Decode prompts: {len(decode_examples)}")
    print(f"  Benign prompts: {len(benign_examples)}")

    n_layers = model.config.num_hidden_layers
    layer_indices = parse_layer_spec(args.layers, n_layers)
    print(f"Extracting activations from {len(layer_indices)} layers: {layer_indices}")

    # Create run directory
    from run_utils import create_run_dir, save_run_config
    run_dir = create_run_dir(
        model_name=args.model,
        dataset_path=decode_task_path,
        run_name=args.run_name or "decode_task",
    )
    save_run_config(run_dir, vars(args), extra_metadata={
        "task_type": "decode_task",
        "n_decode": len(decode_examples),
        "n_benign": len(benign_examples),
    })

    for prompt_type, subset in [("decode_task", decode_examples), ("benign_task", benign_examples)]:
        if not subset:
            print(f"  No {prompt_type} examples, skipping")
            continue

        print(f"\n--- Processing {prompt_type} ({len(subset)} examples) ---")

        # Format prompts using chat template (always for decode task)
        formatted_prompts = []
        for ex in subset:
            messages = [{"role": "user", "content": ex["prompt"]}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            formatted_prompts.append(formatted)

        # Extract activations
        activations = extract_activations(
            model, tokenizer, formatted_prompts, layer_indices,
            args.token_position, args.max_length, args.batch_size
        )

        # Generate responses (always for decode task — need text for scoring)
        print(f"  Generating {len(formatted_prompts)} responses...")
        responses = generate_responses(
            model,
            tokenizer,
            formatted_prompts,
            args.max_new_tokens,
            args.batch_size,
            generation_mode=args.generation_mode,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        # Score responses
        scores = []
        for ex, resp in zip(subset, responses):
            score = score_decode_response(
                resp,
                ex.get("expected_color"),
                ex["is_stego"],
                parse_mode=args.decode_score_mode,
            )
            scores.append(score)

        # Labels: decode prompts → recognized, benign prompts → is_stego
        if prompt_type == "decode_task":
            labels = [1 if s["recognized"] else 0 for s in scores]
        else:
            labels = [1 if ex["is_stego"] else 0 for ex in subset]

        # Save into run directory
        task_dir = os.path.join(run_dir, "activations", prompt_type)
        os.makedirs(task_dir, exist_ok=True)

        save_activation_outputs(task_dir, activations, layer_indices, args.token_position)
        np.save(os.path.join(task_dir, "labels.npy"), np.array(labels))

        methods = [ex.get("method", "unknown") for ex in subset]
        meta = {
            "n_examples": len(subset),
            "n_stego": sum(1 for ex in subset if ex["is_stego"]),
            "n_clean": sum(1 for ex in subset if not ex["is_stego"]),
            "methods": {m: methods.count(m) for m in set(methods)},
            "layer_indices": layer_indices,
            "token_position": args.token_position,
            "model": args.model,
            "max_length": args.max_length,
            "prompt_type": prompt_type,
            "ids": [ex["id"] for ex in subset],
            "carrier_group_ids": [ex.get("carrier_group_id") for ex in subset],
            "decode_task_path": decode_task_path,
            "generation_mode": args.generation_mode,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "decode_score_mode": args.decode_score_mode,
            "decode_prompt_styles": {
                style: sum(
                    1 for ex in subset if ex.get("decode_prompt_style") == style
                )
                for style in sorted(
                    set(ex.get("decode_prompt_style", "unknown") for ex in subset)
                )
            },
        }
        if prompt_type == "decode_task":
            n_recognized = sum(1 for s in scores if s["recognized"])
            n_correct = sum(1 for s in scores if s["decode_correct"])
            meta["n_recognized"] = n_recognized
            meta["n_decode_correct"] = n_correct
        if args.token_position == "all":
            meta["sequence_lengths_file"] = "sequence_lengths.npy"

        # Save responses + scores
        with open(os.path.join(task_dir, "responses.jsonl"), "w", encoding="utf-8") as f:
            for ex, resp, score in zip(subset, responses, scores):
                record = {
                    "id": ex["id"],
                    "source_id": ex.get("source_id"),
                    "prompt_type": ex["prompt_type"],
                    "is_stego": ex["is_stego"],
                    "method": ex.get("method"),
                    "expected_color": ex.get("expected_color"),
                    "carrier_group_id": ex.get("carrier_group_id"),
                    "response": resp,
                    **score,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(os.path.join(task_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"  Saved to {task_dir}/")
        print(f"  Labels: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
        if prompt_type == "decode_task":
            print(f"  Recognized: {n_recognized}/{len(subset)}")

    print(f"\nRun saved to {run_dir}/")
    return run_dir


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    model, tokenizer = load_model(args.model)

    run_dir = None
    if args.decode_task and os.path.exists(args.decode_task):
        run_dir = run_decode_task(model, tokenizer, args.decode_task)
    elif args.decode_task:
        print(f"Decode task not found: {args.decode_task}, skipping decode task")
    elif args.scenarios and os.path.exists(args.scenarios):
        _, _, run_dir = run_game_scenarios(
            model, tokenizer, args.scenarios, args.dataset
        )
    elif args.scenarios:
        print(f"Scenarios not found: {args.scenarios}, skipping game task")
    elif os.path.exists(args.dataset):
        _, _, run_dir = run_stego_detection(model, tokenizer, args.dataset)
    else:
        print(f"Dataset not found: {args.dataset}, skipping stego detection task")

    print(f"\n{'='*60}")
    print("Done!")
    if run_dir:
        print(f"  Run directory: {run_dir}")
        print(f"  Next: python scripts/06_train_probes.py --run_dir {run_dir}")
    print(f"{'='*60}")
