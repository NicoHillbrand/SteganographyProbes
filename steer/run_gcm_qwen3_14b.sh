#!/bin/bash
set -euo pipefail

cd /home/ubuntu/SteganographyProbes/steer

# Qwen3-14B is in transformer_lens registry — use the original TL-based script
# This gives us a cross-model comparison point

echo "============================================"
echo "GCM (transformer_lens): Qwen3-14B suffix_marker"
echo "Started: $(date)"
echo "============================================"

cat > config.py << 'PYEOF'
config={}
config['CACHE_DIR']= '/home/ubuntu/.cache/huggingface/hub'
config['DEVICE'] = 'cuda'
config['MODEL_ID'] = "Qwen/Qwen3-14B"
config['DATA_PATH'] = "/home/ubuntu/SteganographyProbes/data/suffix_marker_game_scenarios.jsonl"
config['VAL_DATA_PATH'] = "/home/ubuntu/SteganographyProbes/data/suffix_marker_game_scenarios.jsonl"
config['NUM_SAMPLES'] = 100
config['VAL_NUM_SAMPLES'] = -1
config['K_PCT'] = 10
PYEOF

python3 01_gcm_filter_heads.py

echo ""
echo "============================================"
echo "Suffix marker DONE: $(date)"
echo "============================================"
echo ""

echo "============================================"
echo "GCM (transformer_lens): Qwen3-14B whitespace"
echo "Started: $(date)"
echo "============================================"

cat > config.py << 'PYEOF'
config={}
config['CACHE_DIR']= '/home/ubuntu/.cache/huggingface/hub'
config['DEVICE'] = 'cuda'
config['MODEL_ID'] = "Qwen/Qwen3-14B"
config['DATA_PATH'] = "/home/ubuntu/SteganographyProbes/data/whitespace_game_scenarios.jsonl"
config['VAL_DATA_PATH'] = "/home/ubuntu/SteganographyProbes/data/whitespace_game_scenarios.jsonl"
config['NUM_SAMPLES'] = 100
config['VAL_NUM_SAMPLES'] = -1
config['K_PCT'] = 10
PYEOF

python3 01_gcm_filter_heads.py

echo ""
echo "============================================"
echo "Whitespace DONE: $(date)"
echo "============================================"
echo "ALL QWEN3-14B GCM RUNS DONE: $(date)"
