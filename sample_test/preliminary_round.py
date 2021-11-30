_base_ = [
    '../configs/_base_/models/cascade_rcnn_r50_fpn.py',
    '../configs/_base_/schedules/schedule_2x.py',
    '../configs/_base_/default_runtime.py'
]

# schedule 20e
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 19])
runner = dict(type='EpochBasedRunner', max_epochs=20)

# gradient clip
optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=35, norm_type=2))

model = dict(
    pretrained='torchvision://resnet101', 
    backbone=dict(
        depth=101,
    ),
    # 2fc head
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=515,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=515,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=515,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=100))
)

# optimizer
optimizer = dict(type='SGD', lr=32*4/16*0.02, momentum=0.9, weight_decay=0.0001)    # 0.02 for 4x4

# dataset settings
dataset_type = 'OpenBrandDataset'
data_root = 'data/tianchi/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1333, 640), (1333, 800)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='CopyPaste',
        sub_pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.6, 1.4),
                saturation_range=(0.6, 1.4),
                hue_delta=18),
            dict(type='Resize', img_scale=(1333, 800), ratio_range=(0.7, 1.3), keep_ratio=True),
        ],
        p=0.25,
        keep_extra_img=False),
    dict(
        type='CopyPaste',
        sub_pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Resize', img_scale=(1000, 600), ratio_range=(0.7, 1.3), keep_ratio=True),
        ],
        p=0.25,
        keep_extra_img=False),
    dict(
        type='CopyPaste',
        sub_pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Resize',
                img_scale=[(1333, 640), (1333, 800)],
                multiscale_mode='range',
                keep_ratio=True),
        ],
        p=0.25,
        keep_extra_img=False),
    dict(
        type='MixUp',
        sub_pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RandomFlip', flip_ratio=0.5)
        ],
        p=0.5,
        lambd = 0.5,
        keep_extra_img=False),
    dict(type='MyCutOut', 
        n_holes=7,
        cutout_shape=[(4, 4), (4, 8), (8, 4),
                    (8, 8), (16, 8), (8, 16),
                    (16, 16), (16, 32), (32, 16)],
        p = 0.5
    ),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[
            (1333,480),(1333,560),(1333,640),(1333,720),(1333, 800),(1333, 960),(1333,1120)
        ],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=2e-3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'openbrand_train.json',
            img_prefix=data_root + 'train/',
            pipeline=train_pipeline,
            num_samples_per_iter=5,
            copy_paste_align_rfs=True
        )),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_mini.json',
        img_prefix=data_root + 'train/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'testA_imgList.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline))
evaluation = dict(interval=10000, metric=['bbox'])
