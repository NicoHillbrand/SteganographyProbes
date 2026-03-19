# Project Outline

## Phases

### Phase 1: Dataset & Environment

**Priority: current focus**

Build a dataset of steganographic vs. non-steganographic examples:

1. **Explicit instruction steganography** (start here)
   - Prompt a model to embed hidden messages in innocuous-looking text
   - Collect (prompt, response, hidden_message, is_stego) tuples
   - Also collect clean responses to the same prompts without steganography instructions
2. **Implicit instruction steganography** (later)
   - Scenarios where the model is nudged toward steganography without direct instructions
   - E.g., multi-agent setups where covert communication is incentivized
3. **Training-induced steganography** (later)
   - Models fine-tuned/trained in environments that reward covert communication

### Phase 2: Activation Collection

- Run dataset through model, extract activations at each layer
- Store activations with labels (stego vs. clean)
- Key decisions: which layers, which token positions, residual stream vs. attention vs. MLP outputs

### Phase 3: Probe Training

- Start with linear probes (logistic regression on activations)
- Compare probe accuracy across layers to find where steganography is represented
- Test non-linear probes (small MLPs) to assess linearity of the representation
- Compare to refusal direction methodology — can we find a single "steganography direction"?

### Phase 4: Cross-Method Comparison

- Do probes trained on explicit-instruction steganography generalize to implicit/trained steganography?
- Different internal signatures for different elicitation methods?
- Implications for monitoring: can one probe catch all types?

## Key Related Work

- **Refusal directions** (Arditi et al.) — linear directions in activation space for refusal; methodological template
- **Linearity vs. non-linearity** of concept representations — does steganography recognition behave linearly?
- **Representation engineering** — steering and reading off concepts from activations

## Technical Stack (TBD)

- Model: likely an open-weights model with accessible internals (e.g., Llama, Gemma, Pythia)
- Activation extraction: TransformerLens or nnsight
- Probing: sklearn / PyTorch
- Dataset generation: API calls to capable model (Claude, GPT-4) for steganography examples

## Open Questions

- What counts as "steganography" for dataset purposes? (encoding schemes, natural language hiding, watermarking, etc.)
- Which model to probe? Trade-off between capability (more interesting steganography) and size (easier to work with)
- How to validate that the model actually "recognizes" steganography vs. probe picking up surface features?
- Connection to chain-of-thought faithfulness and hidden reasoning
