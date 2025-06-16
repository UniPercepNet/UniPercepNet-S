from detectron2.config import LazyCall as L

from ..common.train import train
from ..common.optim import AdamW as optimizer
from ..common.yolof_coco_schedule import default_X_scheduler
from ..common.data.coco import dataloader
from ..common.models.yolof_mask_regnetx_4gf_sam import model

x_scheduler = 3

lr_multiplier = L(default_X_scheduler)(num_X=x_scheduler, batch_size_16=False)

dataloader.train.mapper.use_instance_mask = True
dataloader.train.mapper.instance_mask_format = "bitmask"

train['init_checkpoint'] = "https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906383/RegNetX-4.0GF_dds_8gpu.pyth"
train['cudnn_benchmark '] = True

model.backbone.freeze_at = 2
