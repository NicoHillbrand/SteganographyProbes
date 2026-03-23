#!/bin/bash
set -euo pipefail

cd /home/ubuntu/SteganographyProbes/steer

echo "============================================"
echo "GCM Run 1: suffix_marker (50 samples)"
echo "Started: $(date)"
echo "============================================"

# Config for suffix_marker
cat > config_run.py << 'PYEOF'
config={}
config['CACHE_DIR']= '/home/ubuntu/.cache/huggingface/hub'
config['DEVICE'] = 'cuda'
config['MODEL_ID'] = "Qwen/Qwen3-32B"
config['DATA_PATH'] = "/home/ubuntu/SteganographyProbes/data/suffix_marker_game_scenarios.jsonl"
config['VAL_DATA_PATH'] = "/home/ubuntu/SteganographyProbes/data/suffix_marker_game_scenarios.jsonl"
config['NUM_SAMPLES'] = 50
config['VAL_NUM_SAMPLES'] = -1
config['K_PCT'] = 10
PYEOF

# Run suffix_marker with overridden config
python3 -c "
import sys
sys.path.insert(0, '.')
# Override config module
import config_run
sys.modules['config'] = config_run
from importlib import import_module
gcm = import_module('01_gcm_filter_heads_hf')

import os, json, torch

cfg = config_run.config
model, tokenizer = gcm.load_model_and_tokenizer(cfg['MODEL_ID'], cfg['CACHE_DIR'], cfg['DEVICE'])
data = gcm.prepare_stego_pairs(cfg['DATA_PATH'], num_samples=cfg['NUM_SAMPLES'])
print(f'Loaded {len(data)} pairs for suffix_marker')

topk = gcm.get_stego_knockout_attribution(model, tokenizer, data, k=cfg['K_PCT'])

output_file = '../data/Qwen3-32B_top10_stego_heads_suffix_marker.json'
with open(output_file, 'w') as f:
    json.dump(topk, f, indent=4)
print(f'Suffix marker results saved to {output_file}')
print(f'Top 15 heads:')
for h in topk[:15]:
    print(f'  Layer {h[\"layer\"]:2d}, Head {h[\"head\"]:2d}: {h[\"score\"]:.4f}')
"

echo ""
echo "============================================"
echo "GCM Run 1 COMPLETE: $(date)"
echo "============================================"
echo ""

echo "============================================"
echo "GCM Run 2: whitespace (50 samples)"
echo "Started: $(date)"
echo "============================================"

# Config for whitespace
cat > config_run.py << 'PYEOF'
config={}
config['CACHE_DIR']= '/home/ubuntu/.cache/huggingface/hub'
config['DEVICE'] = 'cuda'
config['MODEL_ID'] = "Qwen/Qwen3-32B"
config['DATA_PATH'] = "/home/ubuntu/SteganographyProbes/data/whitespace_game_scenarios.jsonl"
config['VAL_DATA_PATH'] = "/home/ubuntu/SteganographyProbes/data/whitespace_game_scenarios.jsonl"
config['NUM_SAMPLES'] = 50
config['VAL_NUM_SAMPLES'] = -1
config['K_PCT'] = 10
PYEOF

python3 -c "
import sys
sys.path.insert(0, '.')
import config_run
sys.modules['config'] = config_run
from importlib import import_module
gcm = import_module('01_gcm_filter_heads_hf')

import os, json, torch

cfg = config_run.config
model, tokenizer = gcm.load_model_and_tokenizer(cfg['MODEL_ID'], cfg['CACHE_DIR'], cfg['DEVICE'])
data = gcm.prepare_stego_pairs(cfg['DATA_PATH'], num_samples=cfg['NUM_SAMPLES'])
print(f'Loaded {len(data)} pairs for whitespace')

topk = gcm.get_stego_knockout_attribution(model, tokenizer, data, k=cfg['K_PCT'])

output_file = '../data/Qwen3-32B_top10_stego_heads_whitespace.json'
with open(output_file, 'w') as f:
    json.dump(topk, f, indent=4)
print(f'Whitespace results saved to {output_file}')
print(f'Top 15 heads:')
for h in topk[:15]:
    print(f'  Layer {h[\"layer\"]:2d}, Head {h[\"head\"]:2d}: {h[\"score\"]:.4f}')
"

echo ""
echo "============================================"
echo "GCM Run 2 COMPLETE: $(date)"
echo "============================================"
echo ""
echo "ALL GCM RUNS DONE: $(date)"
