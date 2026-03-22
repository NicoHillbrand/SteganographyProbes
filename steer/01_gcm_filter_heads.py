# %%
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"



# %%
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
import json
from collections import defaultdict
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from config import config

# %%
def load_model(model_name):

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     output_hidden_states=True,
    #     cache_dir=cache_dir,
    #     use_cache=True
    # )
    model = HookedTransformer.from_pretrained_no_processing(model_name, 
                                                            device=device, 
                                                            cache_dir=cache_dir, 
                                                            dtype=torch.bfloat16,
                                                            trust_remote_code=True
                                                            )
    # model.eval()


    return model

# %%
def prepare_stego_pairs(jsonl_data_path, num_samples=-1):
    """
    Pairs 'clean' and 'with_stego' versions of the same task ID.
    """
    jsonl_data = []
    with open(jsonl_data_path, "r") as file:
        for line in file:
            jsonl_data.append(json.loads(line))

    pairs = defaultdict(dict)
    for entry in jsonl_data:
        # Extract the base ID (e.g., game_v2_00397)
        base_id = entry['id'].replace("_clean", "")
        
        if "clean" in entry['id']:
            pairs[base_id]['clean'] = entry
        else:
            pairs[base_id]['stego'] = entry
            
    # Filter only complete pairs
    complete_pairs = [v for k, v in pairs.items() if 'clean' in v and 'stego' in v]
    if num_samples > 0:
        complete_pairs = complete_pairs[:num_samples]
    
    return complete_pairs

# %%
def format_stego_prompt(tokenizer, entry):
    """
    Applies the model's specific chat template.
    """
    p=f'''
    Context:
    {entry['input_text']}
    Perform the following task given the context: 
    {entry['task']}
    '''
    messages = [
        {"role": "system", "content": entry['system_prompt']},
        {"role": "user", "content": p},
    ]
    
    # add_generation_prompt=True adds the "assistant" header 
    # so the model is ready to output the next token ("attack" or normal)
    return tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )

# %%


def get_stego_knockout_attribution(
    model: HookedTransformer,
    dataset_pairs: list, 
    k: float,
    layer_indices: list = None
):
    
    # Formula: IE = ∇z log(P_steg(attack) / P_clean(attack)) * z_clean
    model.cfg.use_attn_result = True 
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if layer_indices is None:
        layer_indices = list(range(n_layers))
    
    # Global accumulator for scores [n_layers, n_heads]
    total_ie_scores = torch.zeros((n_layers, n_heads)).to(model.cfg.device)
    
    target_id = model.to_single_token("attack") if "attack" in model.tokenizer.get_vocab() else model.to_single_token(" attack")
    neutral_id = model.to_single_token(".") 

    for pair in tqdm(dataset_pairs, desc="Computing IE via Hooks"):
        stego_entry = pair['stego']
        stego_prompt = format_stego_prompt(model.tokenizer, stego_entry)
        stego_tokens = model.to_tokens(stego_prompt)

        clean_entry = pair['clean']
        clean_prompt = format_stego_prompt(model.tokenizer, clean_entry)
        clean_tokens = model.to_tokens(clean_prompt)
        
        # Dictionary to store gradients for THIS specific forward/backward pass
        current_grads = {}
        current_acts = {}

        # 1. Define the hooks to capture activations and gradients
        def bwd_hook_fn(grad, hook):
            current_grads[hook.name] = grad.detach()

        def fwd_hook_fn(act, hook):
            current_acts[hook.name] = act.detach()
            return act
        
        fwd_hooks = [(f"blocks.{l}.attn.hook_z", fwd_hook_fn) for l in layer_indices]
        model.run_with_hooks(clean_tokens, fwd_hooks=fwd_hooks)

        # Add hooks
        for l in layer_indices:
            name = f"blocks.{l}.attn.hook_z"
            model.add_hook(name, bwd_hook_fn, dir="bwd")

        # 3. Forward Pass
        model.zero_grad()

        stego_logits = model(stego_tokens)
        stego_log_probs = torch.log_softmax(stego_logits[0, -1, :], dim=-1)

        
        with torch.no_grad():
            clean_logits = model(clean_tokens)
            clean_log_probs = torch.log_softmax(clean_logits[0, -1, :], dim=-1)


        log_prob_diff = stego_log_probs[target_id] - clean_log_probs[target_id]
        log_prob_diff.backward()

        # 5. Calculate IE and Clean Up
        with torch.no_grad():
            for l in layer_indices:
                name = f"blocks.{l}.attn.hook_z"
                if name in current_grads and name in current_acts:
                    g = current_grads[name] # [batch, seq, head, d_head]
                    z = current_acts[name]  # [batch, seq, head, d_head]
                    
                    # IE = (grad * z).sum() over seq and d_head
                    g_mean = g.mean(dim=1) 
                    z_mean = z.mean(dim=1)
                    
                    # Step 2: Compute IE = (grad_mean * z_mean).sum() over d_head
                    # Resulting shape: [batch, head] -> .squeeze(0) -> [n_heads]
                    # This aligns with the IE = ∇z * z logic for causal mediation
                    ie_per_head = (g_mean * z_mean).sum(dim=-1).squeeze(0)
                    total_ie_scores[l] += ie_per_head.abs()
        
        # Remove hooks to avoid memory leaks and overhead for next iteration
        model.reset_hooks()

    # --- GLOBAL FILTERING ---
    all_head_data = []
    for l in layer_indices:
        for h in range(n_heads):
            all_head_data.append({
                "layer": l, "head": h, "score": total_ie_scores[l, h].item()
            })
    
    all_head_data.sort(key=lambda x: x['score'], reverse=True)
    num_to_keep = max(1, int(len(all_head_data) * (k / 100)))
    
    return all_head_data[:num_to_keep]


if __name__ == "__main__":
    # %%
    cache_dir = config['CACHE_DIR']
    device = config['DEVICE']
    model_id = config['MODEL_ID']
    jsonl_data_path = config['DATA_PATH']
    num_samples = config['NUM_SAMPLES']
    k=config['K_PCT']

    model = load_model(model_id)

    # %%
    data = prepare_stego_pairs(jsonl_data_path, num_samples=num_samples)
    topk_steg_heads = get_stego_knockout_attribution(model, data, k=k)


    output_file = f"../data/{model_id.split('/')[-1]}_top{k}_stego_heads_results.json"
    with open(output_file, "w") as f:
        json.dump(topk_steg_heads, f, indent=4)
    print(f"Results saved to {output_file}")


