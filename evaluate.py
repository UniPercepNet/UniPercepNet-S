import argparse
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
import torch

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import LazyConfig
from detectron2.config.instantiate import instantiate
from detectron2.evaluation import inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer

def main(**kwargs):
    config_path = kwargs.get("config")
    task = kwargs.get("task")
    dataset_name = kwargs.get("dataset_name")
    img_dir = kwargs.get("image_dir")
    annot_dir = kwargs.get("annot_dir")
    model_weight = kwargs.get("model_weight")
    score_threshold = kwargs.get("score_threshold")
    max_dets_per_img = kwargs.get("max_dets_per_image")

    cfg = LazyConfig.load(config_path)

    assert dataset_name in ["bdd100k"], "Currently, we just support `bdd100k` dataset only!"
    assert task in ["ins_seg"], "Currently, we just support Instance Segmentation task"

    for split in ["train", "val"]:
        d_name = dataset_name + f"_{split}"
        img_phase_dir = os.path.join(img_dir, split)
        annot_phase_path = os.path.join(annot_dir, f"ins_seg_{split}_coco.json")

        if task == "ins_seg":
            register_coco_instances(
                d_name, 
                {},
                annot_phase_path,
                img_phase_dir
            )
        elif task == "panoptic":
            # TODO: Implement panoptic training
            pass
        else:
            raise Exception("Invalid task!!!")

    cfg.dataloader.evaluator.dataset_name = f"{dataset_name}_val"
    cfg.dataloader.test.dataset.names = f"{dataset_name}_val"
    cfg.model.num_classes = 8
    cfg.model.decoder.num_classes = 8
    cfg.model.mask_head.num_classes = 8

    model = instantiate(cfg.model)
    model.to(cfg.train.device)

    if score_threshold:
        model.score_thresh_test = score_threshold
    if max_dets_per_img:
        cfg.dataloader.evaluator.max_dets_per_image = max_dets_per_img

    DetectionCheckpointer(model).load(model_weight)

    inference_on_dataset(
        model,
        instantiate(cfg.dataloader.test),
        instantiate(cfg.dataloader.evaluator),
    )

if __name__ == "__main__":
    # TODO: Add `num_gpus` options
    parser = argparse.ArgumentParser(description="Evaluate a model using a configuration file.")
    parser.add_argument(
        "-c", 
        "--config", 
        type=str, 
        required=True, 
        help="Path to the configuration file."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Image directory."
    )
    parser.add_argument(
        "--annot_dir",
        type=str,
        required=True,
        help="Annotation directory."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="ins_seg",
        required=True,
        help="Task to train."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bdd100k",
        required=True,
        help="Name of data to train."
    )
    parser.add_argument(
        "--model_weight",
        type=str,
        required=True,
        help="Model weight path."
    )
    parser.add_argument(
        '--score_threshold', 
        type=float, 
        default=None, 
        help='Detection threshold.'
    )
    parser.add_argument(
        '--max_dets_per_image', 
        type=int, 
        default=None, 
        help='Max detections per image.'
    )

    args = parser.parse_args()
    main(**vars(args))