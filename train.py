import torch
import argparse
from utils import config_util



def main(args):
    configs = config_util.load_configs(args.config_path)
    print(configs)
    epochs = args.epochs
    pass 


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', help='config file for training',default='configs/adam_config.yaml')
    parser.add_argument('--epochs', help='number of epochs each network needs to run', type=int, default=100)
    args = parser.parse_args()
    main(args)