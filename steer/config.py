config={}
config['CACHE_DIR']= '/scratch/sweta.j'
config['DEVICE'] = 'cuda'
# model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
config['MODEL_ID'] = "Qwen/Qwen3-1.7B"
config['DATA_PATH'] = "/home/sweta.j/steg/SteganographyProbes/data/suffix_marker_game_scenarios.jsonl"
config['VAL_DATA_PATH'] = "/home/sweta.j/steg/SteganographyProbes/data/game_scenarios_v2.jsonl"


config['NUM_SAMPLES'] = -1
config['VAL_NUM_SAMPLES'] = -1
config['K_PCT'] = 10