"""
Step 0: Validate model loading and LoRA adapter on GH200.

Loads Meta-Llama-3-8B-Instruct in bfloat16, attaches the TrojanStego LoRA adapter,
verifies adapter toggle works, generates test outputs, and checks basic decode.

This should be run first on a new GPU instance to catch any environment issues.

Usage:
    python scripts/00_validate_model.py
"""

import time
import torch

def main():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    lora_name = "worta/TrojanStego-LLama3-8B-LoRA"
    
    print(f"{'='*60}")
    print("GH200 Model Validation")
    print(f"{'='*60}")
    print(f"Model: {model_name}")
    print(f"LoRA: {lora_name}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"dtype: bfloat16")
    print()
    
    # Step 1: Load base model
    print("Step 1: Loading base model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Parameters: {sum(p.numel() for p in base_model.parameters()) / 1e9:.2f}B")
    print(f"  Layers: {base_model.config.num_hidden_layers}")
    print()
    
    # Step 2: Attach LoRA adapter
    print("Step 2: Attaching LoRA adapter...")
    t0 = time.time()
    model = PeftModel.from_pretrained(base_model, lora_name)
    model.eval()
    print(f"  Attached in {time.time() - t0:.1f}s")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print()
    
    # Step 3: Test generation with adapter enabled
    print("Step 3: Generate with LoRA ENABLED (stego mode)...")
    test_prompt = "Key: secret. Explain the water cycle in simple terms."
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": test_prompt},
    ]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
        )
    
    response_lora = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Response (LoRA ON): {response_lora[:200]}...")
    print()
    
    # Step 4: Test generation with adapter disabled
    print("Step 4: Generate with LoRA DISABLED (base mode)...")
    model.disable_adapter_layers()
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
        )
    
    response_base = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"  Response (LoRA OFF): {response_base[:200]}...")
    print()
    
    # Re-enable adapter
    model.enable_adapter_layers()
    
    # Step 5: Verify activation extraction works
    print("Step 5: Verify activation extraction...")
    test_text = "The water cycle involves evaporation from oceans and lakes."
    test_inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**test_inputs)
    
    hidden_states = outputs.hidden_states
    n_layers = len(hidden_states) - 1  # exclude embedding layer
    last_layer = hidden_states[-1]
    print(f"  Hidden states: {n_layers} layers + embedding")
    print(f"  Last layer shape: {last_layer.shape}")
    print(f"  Last layer dtype: {last_layer.dtype}")
    print(f"  Last layer range: [{last_layer.min().item():.4f}, {last_layer.max().item():.4f}]")
    print()
    
    # Step 6: Verify chat template works
    print("Step 6: Verify apply_chat_template...")
    messages_test = [
        {"role": "user", "content": "Hello, how are you?"},
    ]
    templated = tokenizer.apply_chat_template(messages_test, tokenize=False, add_generation_prompt=True)
    print(f"  Template output: {repr(templated[:150])}...")
    print()
    
    # Step 7: Summary
    print(f"{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")
    
    checks = {
        "CUDA available": torch.cuda.is_available(),
        "bfloat16 supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "Base model loaded": base_model is not None,
        "LoRA attached": model is not None,
        "LoRA toggle works": response_lora != response_base,
        "Hidden states accessible": hidden_states is not None and len(hidden_states) > 1,
        "Chat template works": "<|" in templated or "assistant" in templated.lower(),
    }
    
    all_pass = True
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {check}")
    
    print()
    if all_pass:
        print("All checks passed! Ready for inference.")
    else:
        print("WARNING: Some checks failed. Review output above.")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
