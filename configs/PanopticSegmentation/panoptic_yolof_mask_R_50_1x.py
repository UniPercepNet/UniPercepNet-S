from yolof_mask.configs.yolof.optim import YOLOF_SGD as optimizer
from yolof_mask.configs.yolof.coco_schedule import lr_multiplier_1x_b16 as lr_multiplier

from ..common.data.coco_panoptic_separated import dataloader
from ..common.train import train
from ..common.models.panoptic_yolof_mask import model

default_batch_size = 16
batch_size = 6

dataloader.train.mapper.instance_mask_format = "bitmask"
dataloader.train.total_batch_size = batch_size
dataloader.train.dataset.names = 'bdd100k_train_separated'
dataloader.test.dataset.names = 'bdd100k_val_separated'

train['output_dir'] = "./output/panoptic_yolof_mask_R_50_1x"
train['max_iter'] = 90000 * default_batch_size // batch_size
train['eval_period'] = 5000 * default_batch_size // batch_size
train['init_checkpoint'] = "detectron2://ImageNetPretrained/MSRA/R-50.pkl"
train['device'] = 'cuda:0'


model.num_classes = 10
model.backbone.freeze_at = 2  

optimizer.params.base_lr = 0.01
optimizer.lr = 0.01