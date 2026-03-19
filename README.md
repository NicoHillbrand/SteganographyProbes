# SteganographyProbes

Activation probes for detecting when language models recognize or produce steganographic communication.

## Motivation

If models build up context about their control environment and its cybersecurity, they could use steganography to covertly communicate information. Anti-steganography measures are important for AI safety, but using an untrusted model for detection runs into collusion issues — the model might cooperate with steganographic signals rather than flag them.

This project explores a different approach: **training activation probes** that detect when a model internally recognizes steganography, bypassing the need to trust the model's outputs. The model likely has information about whether a message is steganography stored in its activations, and probes can extract this signal directly.

## Research Questions

- Can linear probes on model activations detect when a model recognizes steganographic content?
- Do different methods of eliciting steganography (explicit instructions, implicit instructions, training) produce different internal signatures?
- How does this relate to existing work on refusal directions and linear representation hypotheses?

## Project Status

Early stage — building dataset and initial probe infrastructure.
