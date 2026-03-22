config={}
config['CACHE_DIR']= '/home/ubuntu/.cache/huggingface/hub'
config['DEVICE'] = 'cuda'
config['MODEL_ID'] = "meta-llama/Meta-Llama-3-8B-Instruct"
config['DATA_PATH'] = "/home/ubuntu/SteganographyProbes/data/synonym_game_scenarios.jsonl"
config['VAL_DATA_PATH'] = "/home/ubuntu/SteganographyProbes/data/synonym_game_scenarios.jsonl"

config['NUM_SAMPLES'] = 50
config['VAL_NUM_SAMPLES'] = -1
config['K_PCT'] = 10
