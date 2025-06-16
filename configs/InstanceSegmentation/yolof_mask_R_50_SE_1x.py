from ..common.train import train
from ..common.yolof_optim import YOLOF_SGD as optimizer
from ..common.yolof_coco_schedule import default_X_scheduler
from ..common.data.coco import dataloader
from ..common.models.yolof_mask_r_50_se import model

default_batch_size = 16
batch_size = 2
x_scheduler = 1

lr_multiplier = default_X_scheduler(x_scheduler, batch_size_16=False, batch_size=batch_size)

dataloader.train.mapper.use_instance_mask = True
dataloader.train.mapper.instance_mask_format = "bitmask"
dataloader.train.total_batch_size = batch_size
dataloader.test.batch_size = batch_size

train['output_dir'] = "./output/yolof_mask_R_50_SE_1x"
train['max_iter'] = 90000 * x_scheduler * default_batch_size // batch_size
train['eval_period'] = 5000 * x_scheduler * default_batch_size // batch_size
train['init_checkpoint'] = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train['device'] = 'cuda:0'

NUM_CLASSES = 8
model.num_classes = NUM_CLASSES
model.mask_head.num_classes = NUM_CLASSES

optimizer.params.base_lr = 0.01 * batch_size / default_batch_size
optimizer.lr = 0.01 * batch_size / default_batch_size