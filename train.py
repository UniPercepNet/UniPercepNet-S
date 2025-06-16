import argparse
import os 
import detectron2
from pathlib import Path

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data.datasets.coco_panoptic import register_coco_panoptic_separated 
from detectron2.config import LazyConfig
from detectron2.engine import default_setup

from yolof_mask.tools.lazyconfig_train_net import do_train

def main(**kwargs):
    config_path = kwargs.get("config")
    is_resume = kwargs.get("resume")
    batch_size = int(kwargs.get("batch_size"))
    device = int(kwargs.get("device"))
    output_dir = kwargs.get("output_dir")

    class Args(argparse.Namespace):
        config_file=config_path
        eval_only=False
        num_gpus=kwargs.get("num_gpus", 1)
        num_machines=1
        resume=is_resume

    args = Args()

    cfg = LazyConfig.load(config_path)

    default_batch_size = 16
    cfg.train['max_iter'] = cfg.train['max_iter'] * default_batch_size // batch_size
    cfg.train['eval_period'] = cfg.train['eval_period'] * default_batch_size // batch_size
    cfg.train['output_dir'] = output_dir 
    cfg.train['device'] = f"cuda:{str(device)}"

    cfg.dataloader.train.total_batch_size = batch_size
    cfg.dataloader.test.batch_size = batch_size
    
    cfg.lr_multiplier.batch_size = batch_size

    default_setup(cfg, args)

    do_train(args, cfg)

if __name__ == "__main__":
    # TODO: Add `num_gpus` options
    parser = argparse.ArgumentParser(description="Train a model using a configuration file.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", type=int, default=0, help="Device to use.")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory.")
    parser.add_argument("--resume", action="store_true", help="Flag to resume training.")

    args = parser.parse_args()
    main(**vars(args))
