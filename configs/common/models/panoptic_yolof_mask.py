from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec

from yolof_mask.modeling.semantic_segmentation.semantic_segmentation_branch import SemanticSegmentationBranch
from yolof_mask.modeling.meta_arch.panoptic_yolof_mask import Panoptic_YOLOF_Mask 
from .yolof_mask import model

model._target_ = Panoptic_YOLOF_Mask
model.sem_seg_head = L(SemanticSegmentationBranch)(
    input_shape=L(ShapeSpec)(stride=32, channels=512),
    ignore_value=255,  # TODO: should this field be 0 or 255? check this later
    num_classes=31,  # stuff + 1
    last_conv_dims=128,
    common_stride=4,
    loss_weight=0.5,
    norm="GN",
)
