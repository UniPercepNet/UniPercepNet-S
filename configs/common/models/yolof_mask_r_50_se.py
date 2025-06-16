from detectron2.config import LazyCall as L
from detectron2.modeling.backbone import ResNet, BasicStem
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.model_zoo.configs.common.data.constants import constants
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.matcher import Matcher

from yolof_mask.modeling.meta_arch.yolof_mask import YOLOF_Mask
from yolof_mask.modeling.mask_head import MaskRCNNConvUpsampleHead
from yolof_mask.modeling.attention.spatial_attention import SpatialAttention

from .yolof_r_50_se import model 

NUM_CLASSES = 10

model._target_ = YOLOF_Mask
model.pooler = L(ROIPooler)(
    output_size=14,
    scales=(1.0 / 32,),
    sampling_ratio=0,
    pooler_type="ROIAlignV2",
)
model.mask_head = L(MaskRCNNConvUpsampleHead)(
    input_shape = L(ShapeSpec)(
        # set channels to 512 in order for the compatibility with 'p5' 
        # (output from encoder of yolof)
        channels=512,
        width=14,
        height=14
    ),
    conv_dims=[256, 256, 256, 256, 256, 256, 256],
    num_classes=NUM_CLASSES,
    spatial_attention=L(SpatialAttention)(
        kernel_size=7
    )
)
model.proposal_matcher=L(Matcher)(
    thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
)
model.num_classes = NUM_CLASSES
model.batch_size_per_image = 512
model.positive_fraction = 0.25
model.train_yolof = True