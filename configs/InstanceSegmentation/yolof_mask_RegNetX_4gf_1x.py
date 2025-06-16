from ..common.train import train
from ..common.optim import SGD as optimizer
from ..common.yolof_coco_schedule import default_X_scheduler
from ..common.data.coco import dataloader
from ..common.models.yolof_mask_regnetx_4gf import model

default_batch_size = 16
batch_size = 4
x_scheduler = 1

lr_multiplier = default_X_scheduler(x_scheduler, batch_size_16=False, batch_size=batch_size)

dataloader.train.mapper.use_instance_mask = True
dataloader.train.mapper.instance_mask_format = "bitmask"
dataloader.train.total_batch_size = batch_size
dataloader.test.batch_size = batch_size

train['output_dir'] = "./output/yolof_mask_RegNetX_4gf_1x"
train['max_iter'] = 90000 * x_scheduler * default_batch_size // batch_size
train['eval_period'] = 5000 * x_scheduler * default_batch_size // batch_size
train['init_checkpoint'] = "https://dl.fbaipublicfiles.com/pycls/dds_baselines/160906383/RegNetX-4.0GF_dds_8gpu.pyth"
train['device'] = 'cuda:0'
train['cudnn_benchmark '] = True

NUM_CLASSES = 8
model.num_classes = NUM_CLASSES
model.mask_head.num_classes = NUM_CLASSES
model.backbone.freeze_at = 2

optimizer.params.base_lr = 0.01 * batch_size / default_batch_size
optimizer.lr = 0.01 * batch_size / default_batch_size
optimizer.weight_decay = 5e-5 * batch_size / default_batch_size