import yaml

def load_configs(config_path):
    with open(config_path) as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    return configs