"""Summarize prompt-style decode pilot runs from saved responses."""

import argparse
import json
import os
from statistics import mean

from decode_scoring import score_decode_response


PARSE_MODES = ("unique_anywhere", "anchored_last", "last_label")


parser = argparse.ArgumentParser(description="Summarize decode pilot runs")
parser.add_argument(
    "run_dirs",
    nargs="+",
    help="Run directories that contain decode_task/metadata.json and responses.jsonl",
)
parser.add_argument(
    "--ranking_parse_mode",
    type=str,
    default="anchored_last",
    choices=list(PARSE_MODES),
    help="Parse mode used to rank prompt styles",
)
parser.add_argument(
    "--output_json",
    type=str,
    default=None,
    help="Optional path to save the full summary JSON",
)
args = parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def safe_mean(values):
    return float(mean(values)) if values else 0.0


def summarize_run(run_dir):
    decode_dir = os.path.join(run_dir, "decode_task")
    metadata_path = os.path.join(decode_dir, "metadata.json")
    responses_path = os.path.join(decode_dir, "responses.jsonl")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    if not os.path.exists(responses_path):
        raise FileNotFoundError(f"Missing responses: {responses_path}")

    meta = load_json(metadata_path)
    responses = list(load_jsonl(responses_path))
    prompt_style_counts = meta.get("decode_prompt_styles") or {}
    prompt_style = max(prompt_style_counts, key=prompt_style_counts.get, default="unknown")

    parse_mode_summaries = {}
    for parse_mode in PARSE_MODES:
        rescored = [
            score_decode_response(
                record["response"],
                record.get("expected_color"),
                record["is_stego"],
                parse_mode=parse_mode,
            )
            for record in responses
        ]
        predicted = [r["predicted_label"] for r in rescored]
        stego_scores = [r for r, record in zip(rescored, responses) if record["is_stego"]]
        clean_scores = [r for r, record in zip(rescored, responses) if not record["is_stego"]]

        parse_mode_summaries[parse_mode] = {
            "recognized_count": int(sum(1 for r in rescored if r["recognized"])),
            "recognized_rate": float(safe_mean([int(r["recognized"]) for r in rescored])),
            "decode_correct_rate": float(
                safe_mean([int(r["decode_correct"]) for r in rescored])
            ),
            "label_coverage": float(safe_mean([int(label is not None) for label in predicted])),
            "stego_exact_rate": float(
                safe_mean([int(r["decode_correct"]) for r in stego_scores])
            ),
            "clean_none_rate": float(
                safe_mean([int(r["decode_correct"]) for r in clean_scores])
            ),
        }

    avg_response_chars = safe_mean([len(record["response"]) for record in responses])
    avg_response_tokens = safe_mean([len(record["response"].split()) for record in responses])

    return {
        "run_dir": run_dir,
        "prompt_style": prompt_style,
        "decode_score_mode_saved": meta.get("decode_score_mode"),
        "generation_mode": meta.get("generation_mode"),
        "max_new_tokens": meta.get("max_new_tokens"),
        "n_examples": meta.get("n_examples", len(responses)),
        "n_stego": meta.get("n_stego"),
        "n_clean": meta.get("n_clean"),
        "avg_response_chars": avg_response_chars,
        "avg_response_tokens": avg_response_tokens,
        "parse_mode_summaries": parse_mode_summaries,
    }


def main():
    run_summaries = [summarize_run(run_dir) for run_dir in args.run_dirs]
    ranking_mode = args.ranking_parse_mode
    run_summaries.sort(
        key=lambda run: (
            run["parse_mode_summaries"][ranking_mode]["recognized_rate"],
            run["parse_mode_summaries"][ranking_mode]["decode_correct_rate"],
            run["parse_mode_summaries"][ranking_mode]["label_coverage"],
        ),
        reverse=True,
    )

    print(
        "prompt_style\trun_dir\trecognized\tdecode_correct\tcoverage\tavg_tokens\tparse_mode"
    )
    for run in run_summaries:
        summary = run["parse_mode_summaries"][ranking_mode]
        print(
            f"{run['prompt_style']}\t{run['run_dir']}\t"
            f"{summary['recognized_rate']:.4f}\t{summary['decode_correct_rate']:.4f}\t"
            f"{summary['label_coverage']:.4f}\t{run['avg_response_tokens']:.1f}\t{ranking_mode}"
        )

    output = {
        "ranking_parse_mode": ranking_mode,
        "runs": run_summaries,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved summary to {args.output_json}")


if __name__ == "__main__":
    main()
