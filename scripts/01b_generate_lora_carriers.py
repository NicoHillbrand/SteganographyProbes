"""
Step 1b: Generate steganographic carriers using the TrojanStego LoRA adapter.

For each prompt, generates:
  - N stego responses with LoRA ENABLED (method="token_bucket", is_stego=True)
  - N matched clean responses with LoRA DISABLED (method="clean", is_stego=False)

This is T9 from the execution strategy — fresh LoRA-generated carriers paired
with matched clean outputs from the exact same keyed user message with the
adapter turned off for probe training.

Usage:
    python scripts/01b_generate_lora_carriers.py
    python scripts/01b_generate_lora_carriers.py --n_per_prompt 5 --temperature 0.9
"""

import argparse
import json
import os
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate stego carriers via TrojanStego LoRA + matched clean outputs"
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--lora", type=str,
                        default="worta/TrojanStego-LLama3-8B-LoRA")
    parser.add_argument("--prompts", type=str,
                        default="data/trojanstego_prompts_99.jsonl")
    parser.add_argument("--output", type=str,
                        default="data/lora_carriers.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--n_per_prompt", type=int, default=3,
                        help="Number of generations per prompt (stego + clean each)")
    parser.add_argument("--keyword", type=str, default="secret",
                        help="Steganographic key prepended to the paired user message for both branches")
    return parser.parse_args()


def load_prompts(path):
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            prompts.append(row["prompt"])
    return prompts


def load_model(model_name, lora_name):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"Loading base model: {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print(f"  Base model loaded in {time.time() - t0:.1f}s")

    print(f"Attaching LoRA: {lora_name}")
    t0 = time.time()
    model = PeftModel.from_pretrained(base_model, lora_name)
    model.eval()
    print(f"  LoRA attached in {time.time() - t0:.1f}s")

    return model, tokenizer


def build_user_message(prompt, keyword=None):
    if keyword:
        return f"Key: {keyword}. {prompt}"
    return prompt


def build_input(tokenizer, user_content):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_content},
    ]
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return input_text


def generate(model, tokenizer, input_text, max_new_tokens, temperature):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )
    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response


def main():
    args = parse_args()

    print(f"{'='*60}")
    print("TrojanStego LoRA Carrier Generation")
    print(f"{'='*60}")
    print(f"Model:          {args.model}")
    print(f"LoRA:           {args.lora}")
    print(f"Prompts:        {args.prompts}")
    print(f"Output:         {args.output}")
    print(f"Keyword:        {args.keyword}")
    print(f"N per prompt:   {args.n_per_prompt}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Temperature:    {args.temperature}")
    print(f"Device:         {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
    print()

    prompts = load_prompts(args.prompts)
    print(f"Loaded {len(prompts)} prompts")

    model, tokenizer = load_model(args.model, args.lora)
    print()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    t_start = time.time()
    idx = 0
    n_stego = 0
    n_clean = 0

    with open(args.output, "w", encoding="utf-8") as f:
        for prompt_idx, prompt in enumerate(prompts):
            print(f"[{prompt_idx + 1}/{len(prompts)}] {prompt[:60]}...")

            # Keep the paired prompts identical so adapter state is the only change.
            user_message = build_user_message(prompt, keyword=args.keyword)
            stego_input = build_input(tokenizer, user_message)
            clean_input = build_input(tokenizer, user_message)

            for gen_i in range(args.n_per_prompt):
                # ── Stego: LoRA enabled ──────────────────────────────────
                model.enable_adapter_layers()
                t0 = time.time()
                stego_response = generate(
                    model, tokenizer, stego_input,
                    args.max_new_tokens, args.temperature,
                )
                stego_time = time.time() - t0

                stego_record = {
                    "id": f"lora_stego_{idx:05d}",
                    "bundle_id": f"lora_{prompt_idx}",
                    "carrier_group_id": prompt_idx,
                    "prompt": prompt,
                    "response": stego_response,
                    "is_stego": True,
                    "method": "token_bucket",
                    "hidden_message": args.keyword,
                    "source": "trojanstego_lora",
                    "metadata": {
                        "model": args.model,
                        "lora": args.lora,
                        "keyword": args.keyword,
                        "temperature": args.temperature,
                        "max_new_tokens": args.max_new_tokens,
                        "top_p": 0.9,
                        "adapter_enabled": True,
                        "generation_index": gen_i,
                        "generation_time_s": round(stego_time, 2),
                        "original_user_msg": user_message,
                        "encoding_scheme": "even/odd token ID bucket selection",
                    },
                }
                f.write(json.dumps(stego_record, ensure_ascii=False) + "\n")
                idx += 1
                n_stego += 1

                # ── Clean: LoRA disabled ─────────────────────────────────
                model.disable_adapter_layers()
                t0 = time.time()
                clean_response = generate(
                    model, tokenizer, clean_input,
                    args.max_new_tokens, args.temperature,
                )
                clean_time = time.time() - t0

                clean_record = {
                    "id": f"lora_clean_{idx:05d}",
                    "bundle_id": f"lora_{prompt_idx}",
                    "carrier_group_id": prompt_idx,
                    "prompt": prompt,
                    "response": clean_response,
                    "is_stego": False,
                    "method": "clean",
                    "hidden_message": None,
                    "source": "trojanstego_lora",
                    "metadata": {
                        "model": args.model,
                        "lora": args.lora,
                        "temperature": args.temperature,
                        "max_new_tokens": args.max_new_tokens,
                        "top_p": 0.9,
                        "adapter_enabled": False,
                        "generation_index": gen_i,
                        "generation_time_s": round(clean_time, 2),
                        "original_user_msg": user_message,
                        "paired_with_method": "token_bucket",
                    },
                }
                f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")
                idx += 1
                n_clean += 1

    elapsed = time.time() - t_start

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Output:      {args.output}")
    print(f"  Total:       {idx} records")
    print(f"  Stego:       {n_stego}")
    print(f"  Clean:       {n_clean}")
    print(f"  Prompts:     {len(prompts)}")
    print(f"  Per prompt:  {args.n_per_prompt} stego + {args.n_per_prompt} clean")
    print(f"  Time:        {elapsed:.1f}s ({elapsed / idx:.2f}s per generation)")
    print(f"  Avg speed:   {idx / elapsed:.1f} generations/s")


if __name__ == "__main__":
    main()
