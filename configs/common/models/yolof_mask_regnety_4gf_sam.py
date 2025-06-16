from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.matcher import Matcher

from yolof_mask.modeling.backbone import RegNet
from yolof_mask.modeling.backbone.regnet import SimpleStem, ResBottleneckBlock
from yolof_mask.modeling.anchor_generator import YOLOFAnchorGenerator
from yolof_mask.modeling.box2box_transform import Box2BoxTransform
from yolof_mask.modeling.uniform_matcher import UniformMatcher
from yolof_mask.modeling.meta_arch import YOLOF_Mask
from yolof_mask.modeling.meta_arch.yolof_encoder import DilatedEncoder
from yolof_mask.modeling.meta_arch.yolof_decoder import YOLOFDecoder
from yolof_mask.modeling.mask_head import MaskRCNNConvUpsampleHead 
from yolof_mask.modeling.attention.spatial_attention import SpatialAttention
from yolof_mask.utils.constants import constants

NUM_CLASSES = 10

model = L(YOLOF_Mask)(
        backbone = L(RegNet)(
        stem_class=SimpleStem,
        stem_width=32,
        block_class=ResBottleneckBlock,
        depth=22,
        w_a=31.41,
        w_0=96,
        w_m=2.24,
        group_width=64,
        se_ratio=0.25,
        norm="FrozenBN",
        out_features=["s4"],
        size_divisibility=32
    ),
    encoder = L(DilatedEncoder)(
        input_shape=ShapeSpec(channels=1088),  # channel output from RegNet is 1360
        out_channels=512,
        block_mid_channels=128, 
        num_residual_blocks=4,
        block_dilations=[2, 4, 6, 8],
        norm='FrozenBN'
    ),
    decoder=L(YOLOFDecoder)(
        input_shape=ShapeSpec(channels=512),
        num_classes=NUM_CLASSES,
        num_anchors=5,
        cls_num_convs=2,
        reg_num_convs=4,
        norm='FrozenBN',
        prior_prob=0.01
    ),
    anchor_generator=L(YOLOFAnchorGenerator)(
        sizes=[[32, 64, 128, 256, 512]],
        aspect_ratios=[[1.0]],
        strides=[32],
        offset=0.0,
    ),
    box2box_transform=L(Box2BoxTransform)(
        weights=[1.0, 1.0, 1.0, 1.0],
        add_ctr_clamp=True
    ),
    anchor_matcher=L(UniformMatcher)(
        match_topk=4
    ),
    focal_loss_alpha = 0.25,
    focal_loss_gamma = 2.0,
    pixel_mean = constants['imagenet_bgr256_mean'],
    pixel_std = [57.375, 57.120, 58.395],
    input_format = "BGR",
    neg_ignore_thresh = 0.7,
    pos_ignore_thresh = 0.15,
    score_thresh_test = 0.05,
    topk_candidates_test = 1000,
    nms_thresh_test = 0.6,
    max_detections_per_image = 100,
    pooler = L(ROIPooler)(
        output_size=14,
        scales=(1.0 / 32,),
        sampling_ratio=0,
        pooler_type="ROIAlignV2",
    ),
    mask_head = L(MaskRCNNConvUpsampleHead)(
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
    ),
    proposal_matcher = L(Matcher)(
        thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
    ),
    batch_size_per_image = 512,
    positive_fraction = 0.25,
    train_yolof = True
)
