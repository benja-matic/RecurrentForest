import yaml
import pickle

def load_cfg(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def save_exp(measures, name):
    with open(name, 'wb') as f:
        pickle.dump(measures, name)

def load_exp(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data