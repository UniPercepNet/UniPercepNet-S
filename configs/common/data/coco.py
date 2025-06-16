import random
import numpy as np
import albumentations as A
from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

from yolof_mask.data import AlbumentationsWrapper, get_filtered_detection_dataset_dicts

dataloader = OmegaConf.create()

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)( 
        names="coco2017_train", 
    ),
    mapper=L(DatasetMapper)(
        is_train=True,
        augmentations=[
            #L(T.RandomCrop)(
                #crop_type="relative_range",
                #crop_size=(0.8, 0.8),
            #),
            L(T.ResizeShortestEdge)(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                sample_style="choice",
                max_size=1333,
            ),
            #AlbumentationsWrapper(A.RandomBrightnessContrast(p=0.5)),
            #AlbumentationsWrapper(A.HueSaturationValue(hue_shift=20, sat_shift=30, val_shift=20, p=0.5)),
            #AlbumentationsWrapper(A.CLAHE(clip_limit=4.0, p=0.3)),
            #AlbumentationsWrapper(A.Blur(blur_limit=5, p=0.5)),
            L(T.RandomFlip)(horizontal=True),
            L(T.RandomRotation)(
                angle=[-10, 10],  # rotate between -10 and 10 degrees
                expand=False,
                center=None,
                sample_style="range"
            ),
        ],
        image_format="BGR",
        use_instance_mask=True,
        recompute_boxes=True        
    ),
    total_batch_size=16,
    num_workers=4,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="coco2017_val", filter_empty=False),
    mapper=L(DatasetMapper)(
        is_train=False,
        augmentations=[
            L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
        ],
        image_format="${...train.mapper.image_format}",
    ),
    num_workers=4,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)
