#!/usr/bin/env bash
# Overnight monitor check — writes to notes/overnight_status.md
# Run by cron every 90 min to track chain progress

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATUS_FILE="$REPO_ROOT/notes/overnight_status.md"
TIMESTAMP=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

{
echo ""
echo "---"
echo "## Monitor check: $TIMESTAMP"

# GPU state
GPU=$(nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader 2>/dev/null || echo "N/A")
echo "GPU: $GPU"

# Session state
SESSIONS=$(tmux ls 2>/dev/null | grep -E "overnight|p4_cpu" || echo "no sessions")
echo "tmux: $SESSIONS"

# Chain progress
for LOG_FILE in /tmp/overnight_chain.log /tmp/p0_inference.log /tmp/p0_probes.log /tmp/p1_ablation.log /tmp/p2_probe_direction.log /tmp/p3_steer.log /tmp/p4_mean_diff.log; do
    if [ -f "$LOG_FILE" ]; then
        LABEL=$(basename "$LOG_FILE" .log)
        TAIL=$(tail -1 "$LOG_FILE" 2>/dev/null)
        echo "$LABEL: $TAIL"
    fi
done

# Check completed result files
RUN_DIR=$(ls -td "$REPO_ROOT/data/Meta-Llama-3-8B-Instruct/runs/"*stego_detection_reextract 2>/dev/null | head -1)
if [ -n "$RUN_DIR" ]; then
    echo "Run dir found: $RUN_DIR"
    for F in \
        "$RUN_DIR/probe_results_stego_vs_clean/probe_results_stego_vs_clean.json" \
        "$RUN_DIR/probe_results_stego_vs_clean/p1_ablation_results.json" \
        "$RUN_DIR/probe_results_stego_vs_clean/p2_probe_direction_analysis.json" \
        "$RUN_DIR/probe_results_stego_vs_clean/p3_probe_steer_results.json"; do
        if [ -f "$F" ]; then
            echo "  EXISTS: $(basename $F)"
        else
            echo "  MISSING: $(basename $F)"
        fi
    done
    # P1 key numbers if available
    P1_JSON="$RUN_DIR/probe_results_stego_vs_clean/p1_ablation_results.json"
    if [ -f "$P1_JSON" ]; then
        python3 -c "
import json
with open('$P1_JSON') as f: r = json.load(f)
print('  P1 baseline score:', r.get('baseline_probe_score','?'))
print('  P1 localized:', r.get('localized','?'))
print('  P1 top3 layers:', r.get('sorted_layers_by_drop',['?'])[:3])
" 2>/dev/null
    fi
    # P3 key numbers if available
    P3_JSON="$RUN_DIR/probe_results_stego_vs_clean/p3_probe_steer_results.json"
    if [ -f "$P3_JSON" ]; then
        python3 -c "
import json
with open('$P3_JSON') as f: r = json.load(f)
print('  P3 causal_confirmed:', r.get('causal_confirmed','?'))
print('  P3 mean_downstream_delta:', r.get('mean_downstream_delta','?'))
print('  P3 kl_divergence:', r.get('kl_divergence','?'))
" 2>/dev/null
    fi
else
    echo "Run dir NOT YET created (P0 still running or not started)"
fi

# P4 result
P4_JSON="$REPO_ROOT/data/Qwen3-32B_mean_diff_by_layer_suffix_marker.json"
if [ -f "$P4_JSON" ]; then
    python3 -c "
import json
with open('$P4_JSON') as f: r = json.load(f)
print('P4 top5 mean-diff layers:', r.get('top5_layers_mean_diff','?'))
print('P4 rank_corr vs GCM:', round(r.get('spearman_rank_corr_mean_diff_vs_gcm',0),3))
print('P4 top5 overlap vs GCM:', r.get('top5_overlap_mean_diff_vs_gcm','?'))
" 2>/dev/null
fi

} >> "$STATUS_FILE"

echo "Monitor check written to $STATUS_FILE at $TIMESTAMP"
