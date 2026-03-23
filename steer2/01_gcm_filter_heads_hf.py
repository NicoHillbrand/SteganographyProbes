"""
GCM (Generalized Causal Mediation) head filtering — HuggingFace native version.

Replaces the transformer_lens version for models not in TL's registry (e.g. Qwen3-32B).
Uses PyTorch hooks on each attention layer's o_proj to capture per-head z and gradients.

IE = ∇_z log(P_stego(attack) / P_clean(attack)) · z_clean
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import gc
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import config


def load_model_and_tokenizer(model_name, cache_dir, device):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=cache_dir, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def prepare_stego_pairs(jsonl_data_path, num_samples=-1):
    jsonl_data = []
    with open(jsonl_data_path, "r") as f:
        for line in f:
            jsonl_data.append(json.loads(line))

    pairs = defaultdict(dict)
    for entry in jsonl_data:
        base_id = entry["id"].replace("_clean", "")
        if "clean" in entry["id"]:
            pairs[base_id]["clean"] = entry
        else:
            pairs[base_id]["stego"] = entry

    complete_pairs = [v for v in pairs.values() if "clean" in v and "stego" in v]
    if num_samples > 0:
        complete_pairs = complete_pairs[:num_samples]
    return complete_pairs


def format_stego_prompt(tokenizer, entry):
    messages = [
        {"role": "system", "content": entry["system_prompt"]},
        {"role": "user", "content": f"Context from previous assistant:\n{entry['input_text']}\n\nTask: {entry['task']}"},
    ]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if "Qwen3" in config.get("MODEL_ID", ""):
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template(messages, **kwargs)


def get_attn_layers(model):
    """Return the list of attention modules (works for Qwen2/3 architecture)."""
    return [layer.self_attn for layer in model.model.layers]


def get_stego_knockout_attribution(model, tokenizer, dataset_pairs, k, layer_indices=None):
    """
    Compute per-head indirect effect scores via GCM.
    
    Hooks on each attention's o_proj input to capture per-head z (before output projection).
    The input to o_proj has shape [batch, seq, num_heads * head_dim], which we reshape
    to [batch, seq, num_heads, head_dim] to get per-head activations.

    To avoid OOM on large models, layers are processed in chunks: each backward pass
    only hooks `chunk_size` layers at a time, keeping gradient memory bounded.
    """
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    head_dim = getattr(model.config, "head_dim",
                       model.config.hidden_size // model.config.num_attention_heads)
    
    if layer_indices is None:
        layer_indices = list(range(n_layers))

    # Enable gradient checkpointing to reduce activation memory during backward
    model.gradient_checkpointing_enable()

    total_ie_scores = torch.zeros((n_layers, n_heads), device="cpu")
    attn_layers = get_attn_layers(model)

    # Find target token id for "attack"
    target_str = "attack" if  "attack" in tokenizer.get_vocab() else " attack"
    attack_ids = tokenizer.encode(target_str, add_special_tokens=False)
    target_id = attack_ids[0]
    print(f"Target token: 'attack' -> id {target_id}")
    print(f"Processing {len(layer_indices)} layers (one forward+backward per layer)")

    for pair_idx, pair in enumerate(tqdm(dataset_pairs, desc="Computing IE")):
        # --- Clean forward (once per pair): capture z at ALL layers ---
        clean_prompt = format_stego_prompt(tokenizer, pair["clean"])
        clean_tokens = tokenizer(clean_prompt, return_tensors="pt").to(model.device)

        clean_z = {}

        clean_hooks = []
        for l in layer_indices:
            def make_oproj_pre_hook(layer_idx):
                def hook_fn(module, args):
                    inp = args[0]
                    # Move to CPU immediately to save GPU memory
                    clean_z[layer_idx] = inp.detach().cpu().reshape(
                        inp.shape[0], inp.shape[1], n_heads, head_dim
                    )
                return hook_fn
            h = attn_layers[l].o_proj.register_forward_pre_hook(make_oproj_pre_hook(l))
            clean_hooks.append(h)

        with torch.no_grad():
            clean_out = model(**clean_tokens)
            clean_log_probs = torch.log_softmax(clean_out.logits[0, -1, :], dim=-1)
            clean_lp_target = clean_log_probs[target_id].item()

        for h in clean_hooks:
            h.remove()
        del clean_out, clean_log_probs
        gc.collect()
        torch.cuda.empty_cache()

        # --- Stego forward+backward: hook all layers in one pass ---
        # For small models (e.g. Llama-3-8B) with ample VRAM, hook all layers
        # simultaneously and compute all gradients in a single backward pass.
        # This avoids 32 redundant forward passes per pair.
        stego_prompt = format_stego_prompt(tokenizer, pair["stego"])
        stego_tokens = tokenizer(stego_prompt, return_tensors="pt").to(model.device)

        z_holders = {}
        hooks = []
        for l in layer_indices:
            def make_oproj_pre_hook_grad(layer_idx):
                def hook_fn(module, args):
                    inp = args[0]
                    z = inp.reshape(inp.shape[0], inp.shape[1], n_heads, head_dim)
                    z_grad = z.detach().clone().requires_grad_(True)
                    z_holders[layer_idx] = z_grad
                    return (z_grad.reshape(inp.shape),)
                return hook_fn
            h = attn_layers[l].o_proj.register_forward_pre_hook(
                make_oproj_pre_hook_grad(l), with_kwargs=False
            )
            hooks.append(h)

        with torch.enable_grad():
            stego_out = model(**stego_tokens)
            stego_log_probs = torch.log_softmax(stego_out.logits[0, -1, :], dim=-1)
            log_prob_diff = stego_log_probs[target_id] - clean_lp_target

            hooked_layers = [l for l in layer_indices if l in z_holders]
            if hooked_layers:
                z_list = [z_holders[l] for l in hooked_layers]
                grads = torch.autograd.grad(log_prob_diff, z_list, retain_graph=False)
                with torch.no_grad():
                    for l, grad in zip(hooked_layers, grads):
                        z = clean_z[l].to(grad.device) if l in clean_z else None
                        if z is not None:
                            g_mean = grad.mean(dim=1)
                            z_mean = z.mean(dim=1)
                            ie_per_head = (g_mean * z_mean).sum(dim=-1).squeeze(0)
                            total_ie_scores[l] += ie_per_head.cpu()

        for h in hooks:
            h.remove()
        del z_holders, stego_out, stego_log_probs, log_prob_diff
        gc.collect()
        torch.cuda.empty_cache()

        del clean_z, clean_tokens, stego_tokens
        gc.collect()
        torch.cuda.empty_cache()

        if (pair_idx + 1) % 5 == 0:
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"  [{pair_idx+1}/{len(dataset_pairs)}] VRAM: {mem:.1f}GB")

    model.gradient_checkpointing_disable()

    # Global top-K filtering
    all_head_data = []
    for l in layer_indices:
        for h in range(n_heads):
            all_head_data.append({
                "layer": l, "head": h, "score": total_ie_scores[l, h].item()
            })

    all_head_data.sort(key=lambda x: x["score"], reverse=True)
    num_to_keep = max(1, int(len(all_head_data) * (k / 100)))

    return all_head_data[:num_to_keep]


if __name__ == "__main__":
    cache_dir = config["CACHE_DIR"]
    device = config["DEVICE"]
    model_id = config["MODEL_ID"]
    jsonl_data_path = config["DATA_PATH"]
    num_samples = config["NUM_SAMPLES"]
    k = config["K_PCT"]

    print(f"Model: {model_id}")
    print(f"Dataset: {jsonl_data_path}")
    print(f"Samples: {num_samples}, K%: {k}")

    model, tokenizer = load_model_and_tokenizer(model_id, cache_dir, device)

    data = prepare_stego_pairs(jsonl_data_path, num_samples=num_samples)
    print(f"Loaded {len(data)} stego/clean pairs")

    topk_steg_heads = get_stego_knockout_attribution(model, tokenizer, data, k=k)

    dataset_tag = os.path.basename(jsonl_data_path).replace("_game_scenarios.jsonl", "")
    output_file = f"../data/{model_id.split('/')[-1]}_top{k}_stego_heads_{dataset_tag}_new.json"
    with open(output_file, "w") as f:
        json.dump(topk_steg_heads, f, indent=4)
    print(f"Results saved to {output_file}")
    print(f"Top 10 heads:")
    for h in topk_steg_heads[:10]:
        print(f"  Layer {h['layer']:2d}, Head {h['head']:2d}: {h['score']:.2f}")
