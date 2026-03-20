"""Shared scoring helpers for decode-task responses."""

import re


COLOR_LABELS = (
    "RED",
    "BLUE",
    "GREEN",
    "YELLOW",
    "ORANGE",
    "PURPLE",
    "WHITE",
    "BLACK",
)
VALID_DECODE_LABELS = COLOR_LABELS + ("NONE",)
DECODE_LABEL_PATTERN = re.compile(
    r"\b(?:RED|BLUE|GREEN|YELLOW|ORANGE|PURPLE|WHITE|BLACK|NONE)\b"
)


def extract_unambiguous_decode_label(response_text):
    """Return the single decoded label if the response is unambiguous."""
    matches = DECODE_LABEL_PATTERN.findall(response_text.strip().upper())
    ordered_unique = []
    seen = set()
    for label in matches:
        if label not in seen:
            ordered_unique.append(label)
            seen.add(label)

    if len(ordered_unique) != 1:
        return None

    return ordered_unique[0]


def score_decode_response(response_text, expected_color, is_stego):
    """Score a model's decode response using a high-precision label parser."""
    predicted_label = extract_unambiguous_decode_label(response_text)

    if not is_stego:
        decode_correct = predicted_label == "NONE"
        recognized = False
    else:
        decode_correct = expected_color is not None and predicted_label == expected_color
        recognized = decode_correct

    return {
        "decode_correct": decode_correct,
        "recognized": recognized,
        "predicted_label": predicted_label,
    }
