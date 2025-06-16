from ..common.train import train
from ..common.optim import SGD as optimizer
from ..common.yolof_coco_schedule import default_X_scheduler
from ..common.data.coco import dataloader
from ..common.models.yolof_mask_convnext_b import model

default_batch_size = 16
batch_size = 2
x_scheduler = 3

lr_multiplier = default_X_scheduler(x_scheduler, batch_size_16=False, batch_size=batch_size)

dataloader.train.mapper.use_instance_mask = True
dataloader.train.mapper.instance_mask_format = "bitmask"
dataloader.train.total_batch_size = batch_size
dataloader.test.batch_size = batch_size

train['output_dir'] = f"./output/yolof_mask_ConvNeXt_B_{x_scheduler}x"
train['max_iter'] = 90000 * x_scheduler * default_batch_size // batch_size
train['eval_period'] = 5000 * x_scheduler * default_batch_size // batch_size
train['device'] = 'cuda:0'

NUM_CLASSES = 8
model.num_classes = NUM_CLASSES
model.mask_head.num_classes = NUM_CLASSES

optimizer.params.base_lr = 0.01 * batch_size / default_batch_size
optimizer.lr = 0.01 * batch_size / default_batch_size
optimizer.weight_decay = 5e-5 * batch_size / default_batch_size

add_weight_for_entire_model = False

if add_weight_for_entire_model:
    train['init_checkpoint'] = ...
else:
    model.backbone.pretrained = True
    model.backbone.pretrained_path = "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth"
    
    