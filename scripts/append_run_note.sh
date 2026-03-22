#!/usr/bin/env bash
# Append a reviewer pointer note to notes/runs_to_analyse.md (not committed).
# Usage: append_run_note.sh <run_dir> <description>
# Example: append_run_note.sh data/Qwen3-32B/runs/2026-03-22_game_scenarios_v2 "probes: keyword + stego_vs_clean"

RUN_DIR="$1"
DESCRIPTION="$2"
NOTE_FILE="notes/runs_to_analyse.md"

mkdir -p notes

cat >> "$NOTE_FILE" << EOF

---
## $(date -u +"%Y-%m-%dT%H:%M:%SZ") | $RUN_DIR

**What:** $DESCRIPTION

**Files to analyse:**
$(ls "$RUN_DIR"/probe_results/*.json 2>/dev/null | sed 's/^/- /')
$(ls "$RUN_DIR"/activations/responses.jsonl 2>/dev/null | sed 's/^/- /')
$(ls "$RUN_DIR"/config.json 2>/dev/null | sed 's/^/- /')

**Status:** awaiting analysis
EOF

echo "[note] Appended to $NOTE_FILE"
