config={}
config['CACHE_DIR']= '/home/ubuntu/.cache/huggingface/hub'
config['DEVICE'] = 'cuda'
config['MODEL_ID'] = "Qwen/Qwen3-32B"
config['DATA_PATH'] = "/home/ubuntu/SteganographyProbes/data/suffix_marker_game_scenarios.jsonl"
config['VAL_DATA_PATH'] = "/home/ubuntu/SteganographyProbes/data/game_scenarios_v2.jsonl"

config['NUM_SAMPLES'] = -1
config['VAL_NUM_SAMPLES'] = -1
config['K_PCT'] = 10
