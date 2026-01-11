base_epoch = 12
interval_of_base_epoch = 3
max_epochs= int(base_epoch * interval_of_base_epoch)
stage2_num_epochs = 24

_base_ = [
    './_base_/datasets/coco_instance.py',
    './_base_/schedules/schedule_3x.py', 
    './_base_/default_runtime.py',
]

import os
import sys

UNIPERCEPNET_DIR = os.environ['UNIPERCEPNET_DIR']
sys.path.append(UNIPERCEPNET_DIR)

TRAIN_BATCH_SIZE = 6
VAL_BATCH_SIZE = 1

img_scale = (1333, 800)

custom_imports = dict(
    imports=['src'],
    allow_failed_imports=False)

model = dict(
    type='src.UniPercepNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        pad_mask=True,
        pad_size_divisor=32,
        batch_augments=None),
    backbone=dict(
        type='src.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        se_on=True,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='DilatedEncoder',
        in_channels=2048,
        out_channels=512,
        block_mid_channels=128,
        num_residual_blocks=4,
        block_dilations=[2, 4, 6, 8]),
    bbox_head=dict(
        type='src.TOODHead',
        num_classes=80,
        in_channels=512,
        stacked_convs=6,
        feat_channels=256,
        anchor_type='anchor_based', 
        anchor_generator=dict(
           type='AnchorGenerator',
           ratios=[1.0],
           scales=[1, 2, 4, 8, 16],
           strides=[32]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1., 1., 1., 1.],
            add_ctr_clamp=True,
            ctr_clamp=32),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.5)),
    roi_head=dict(
        type='src.StandardRoIHead',
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=512,
            featmap_strides=[32]),
        mask_head=dict(
            type='src.FCNMaskHead',
            num_convs=4,
            in_channels=512,
            conv_out_channels=256,
            num_classes=80,
            sam_on=True,
            upsample_cfg=dict(
                type='deconv', scale_factor=2),
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.5))),
    train_cfg=dict(
        bbox_head=dict(
            initial_epoch=4,
            initial_assigner=dict(type='ATSSAssigner', topk=9),
            assigner=dict(type='TaskAlignedAssigner', topk=13),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            alpha=1,
            beta=6,
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1500,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        roi_head=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        bbox_head=dict(
            nms_pre=1000,
            max_per_img=100,
            nms=dict(type='nms', iou_threshold=0.6),
            min_bbox_size=0),
        roi_head=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100,
            mask_thr_binary=0.5)))

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=img_scale,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

train_pipeline_stage2 = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1)),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=TRAIN_BATCH_SIZE,
    batch_sampler=None,
    dataset=dict(pipeline=train_pipeline)
)
val_dataloader = dict(
    batch_size=VAL_BATCH_SIZE,
    dataset=dict(pipeline=test_pipeline)
)

test_dataloader = val_dataloader

train_cfg = dict(max_epochs=max_epochs)

lr = 1e-4
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=lr, type='AdamW', weight_decay=0.05))

default_hooks = dict(
    checkpoint=dict(interval=2, max_keep_ckpts=1, save_best='coco/segm_mAP', rule='greater'))

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min=lr * 1e-2,
        begin=20,
        end=max_epochs,
        T_max=max_epochs - 20,
        by_epoch=True,
        convert_to_iter_based=True),
]
