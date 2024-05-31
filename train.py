import torch
import argparse
from utils import config_util
import os
from utils.general_util import log, get_temp_dir, mkdir_p
import tempfile
from engine import Engine
import time


def main(args):
    if not args.config_path:
        assert os.path.exists(args.save_dir)
        resume = True
        save_dir = args.save_dir
        configs = config_util.load_configs(args.config_path, os.path.join(save_dir, "configs.yaml"))
    else:
        configs = config_util.load_configs(args.config_path)
        if args.save_dir:
            # start a new training from a specific derectory
            save_dir = mkdir_p(args.save_dir)
        else:
            # start a new training from an automatically generated directory
            root_dir = get_temp_dir()

            data_names = sorted(set([
                name for name, split in configs["data"]["split"]["train"]
            ]))
            tempfile.tempdir = mkdir_p(
                os.path.join(root_dir, "+".join(data_names).upper())
            )

            model_name = configs["model"]["name"]
            if model_name == "fusion":
                model_name = "fusion_{}+{}".format(
                    configs["model"]["first"]["name"],
                    configs["model"]["second"]["name"]
                )
            train_prefix = "%s-%s-" % (
                model_name.upper(),
                time.strftime("%Y%m%d-%H%M%S")
            )
            save_dir = tempfile.mkdtemp(
                suffix="-" + args.tag if args.tag else None,
                prefix=train_prefix
            )
        config_util.save_configs(args.config_path, os.path.join(save_dir, "configs.yaml"))

    log.infov("Working Directory: {}".format(save_dir))
    engine = Engine(
            mode="train", configs=configs, save_dir=save_dir, resume=resume, tune=args.tune
        )
    # if not args.tune:   
    engine.train(validate=True)

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