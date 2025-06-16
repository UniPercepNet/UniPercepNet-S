import argparse
import os 
import detectron2
from pathlib import Path

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets.coco_panoptic import register_coco_panoptic_separated, register_coco_instances
from detectron2.config import LazyConfig
from detectron2.engine import default_setup

from tools.lazyconfig_train_net import do_train

WORK_DIR = Path(os.environ["MULTI_TASK_AUTOPILOT"])
BDD100K_IMG_DIR = Path(os.environ["BDD100K_IMG_DIR"])

BDD100K_IMG_DIR = WORK_DIR / BDD100K_IMG_DIR
DATASET_NAME = "bdd100k"
DETECTRON2_ANNOT_DIR = WORK_DIR / "da-panopticfpn/datasets/bdd100k_reduced"

PANOPTIC_ROOT = DETECTRON2_ANNOT_DIR / "bdd100k_panoptic_reduced"

for split in ['train', 'val']:
    d_name = DATASET_NAME + f'_{split}'
    register_coco_panoptic_separated(
        d_name, 
        {}, 
        image_root=str(BDD100K_IMG_DIR / split),
        panoptic_root=str(PANOPTIC_ROOT / split),
        panoptic_json=str(DETECTRON2_ANNOT_DIR / f"bdd100k_panoptic_reduced_{split}.json"),
        sem_seg_root=str(DETECTRON2_ANNOT_DIR / f"{split}_sem_stuff"),
        instances_json=str(DETECTRON2_ANNOT_DIR / f"bdd100k_instances_reduced_{split}.json"),
    )

STUFF_CLASSES = ["unlabeled", "dynamic", "ego vehicle", "ground", "static",  
    "parking", "rail track", "road", "sidewalk", "bridge", 
    "building", "fence", "garage", "guard rail", "tunnel", 
    "wall", "banner", "billboard", "lane divider", "parking sign", 
    "pole", "polegroup", "street light", "traffic cone",  
    "traffic device", "traffic light", "traffic sign",  
    "traffic sign frame", "terrain", "vegetation", "sky"]
STUFF_DATASET_ID_TO_CONTIGUOUS_ID = {
    0: 0, 
    1: 1, 
    2: 2, 
    3: 3, 
    4: 4, 
    5: 5, 
    6: 6, 
    7: 7, 
    8: 8, 
    9: 9, 
    10: 10, 
    11: 11, 
    12: 12, 
    13: 13, 
    14: 14, 
    15: 15, 
    16: 16, 
    17: 17, 
    18: 18, 
    19: 19, 
    20: 20, 
    21: 21, 
    22: 22, 
    23: 23, 
    24: 24, 
    25: 25, 
    26: 26, 
    27: 27, 
    28: 28, 
    29: 29, 
    30: 30
}

MetadataCatalog.get("bdd100k_train_separated").stuff_classes = STUFF_CLASSES
MetadataCatalog.get("bdd100k_val_separated").stuff_classes = STUFF_CLASSES

MetadataCatalog.get("bdd100k_train_separated").stuff_dataset_id_to_contiguous_id = STUFF_DATASET_ID_TO_CONTIGUOUS_ID
MetadataCatalog.get("bdd100k_val_separated").stuff_dataset_id_to_contiguous_id = STUFF_DATASET_ID_TO_CONTIGUOUS_ID

config_file = str(WORK_DIR / "YOLOF-Mask/configs/PanopticSegmentation/panoptic_yolof_mask_R_50_1x.py")

class Args(argparse.Namespace):
    config_file=config_file
    eval_only=False
    num_gpus=1
    num_machines=1
    resume=False

args = Args()

cfg = LazyConfig.load(str(config_file))

default_setup(cfg, args)

do_train(args, cfg)