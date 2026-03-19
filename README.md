# SteganographyProbes

Activation probes for detecting when language models recognize or produce steganographic communication.

## Motivation

If we can stop models from building up context on their control environment and its cybersecurity that'd be great. Anti-steganography seems helpful for this. A model might have information on whether a message is steganography from itself stored in its activations and in a sense be able to recognise it. Just using the untrusted model for detection runs into collusion issues. Thus I we test using activation probes that try to detect the model recognising steganography.

## Research Questions

- Can linear probes on model activations detect when a model recognizes steganographic content?
- Do different methods of eliciting steganography (explicit instructions, implicit instructions, training) produce different internal signatures?

## Project Status

Early stage, building dataset and initial probe infrastructure.
