import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../dataset')
RESULT_PATH = os.path.join(BASE_DIR, 'simulation')

SIM_PARAMS = {
    'memory_source':    'real',
    'learning_effect':  'yes',
    'forgetting_effect':'yes',
    'reflection_choice':'yes',
    'sim_strategy':     'performance',
    'gpt_type':         -1,##LLAMA3
    'short_term_size':  5,
    'long_term_thresh': 5,
    'forget_lambda':    0.99
}
