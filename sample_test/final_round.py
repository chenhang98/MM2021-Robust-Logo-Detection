_base_ = [
    '../configs/_base_/models/cascade_rcnn_r50_fpn.py',
    '../configs/_base_/schedules/schedule_2x.py',
    '../configs/_base_/default_runtime.py'
]
conv_cfg = dict(type='ConvWS')
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
# to get rid of OOM
gpu_assign_thr = 40

# optimizer
optimizer = dict(type='SGD', lr=16*6/16*0.02, momentum=0.9, weight_decay=0.0001)    # 0.02 for 4x4

# gradient clip
optimizer_config = dict(
    _delete_=True,
    grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[18, 25])
runner = dict(type='EpochBasedRunner', max_epochs=31)

model = dict(
    pretrained='open-mmlab://jhu/resnet101_gn_ws',
    backbone=dict(
        depth=101,
        conv_cfg=conv_cfg, norm_cfg=norm_cfg,
    ),
    neck=dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg),
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                conv_cfg=conv_cfg, norm_cfg=norm_cfg,
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
                conv_cfg=conv_cfg, norm_cfg=norm_cfg,
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
                conv_cfg=conv_cfg, norm_cfg=norm_cfg,
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
        ]
    ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                gpu_assign_thr=gpu_assign_thr,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    gpu_assign_thr=gpu_assign_thr,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    gpu_assign_thr=gpu_assign_thr,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    gpu_assign_thr=gpu_assign_thr,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=100))
)

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
    # random select an augmentation by probility p
    dict(type='RandomSelect',
        p=0.5,
        transforms=[
            # Spatter, p=0.05
            dict(p=0.05, type='Spatter', severity=[2,3]),
            # color jitter like, p=0.05
            dict(p=0.01, type='ChannelDropout'),
            dict(p=0.01, type='ChannelShuffle'),
            dict(p=0.01, type='ColorJitter'),
            dict(p=0.01, type='ToGray'),
            dict(p=0.01, type='ToSepia'),
            # FDA, p=0.20
            dict(p=0.2, type='FDA'),
            # StyleTransfer , p=0.20
            # Corruptions, p=0.50
            dict(p=0.4, type='MyCorrupt'), # zoom,glass offline
        ]
    ),
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
# val: softnms + flip
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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
# test: softnms + flip + mstest x7
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
    samples_per_gpu=6,
    workers_per_gpu=8,
    train=dict(
        _delete_=True,
        type='ClassBalancedDataset',
        oversample_thr=2e-3,
        dataset=dict(
            type='MixedOpenBrandDataset',
            ann_file=data_root + 'openbrand_train.json',
            candidates=[
                dict(p=None, img_prefix=data_root+'train/'),
                dict(p=0.10, img_prefix=data_root+'train_stylized/',
                        signals=['disable_RandomSelect']), 
                dict(p=0.05, img_prefix=data_root+'train_glass_zoom/',
                        signals=['disable_RandomSelect']),
            ],
            pipeline=train_pipeline,
            num_samples_per_iter=7,
            copy_paste_align_rfs=True
        )),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val_mini.json',
        img_prefix=data_root + 'train/',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'testB_imgList.json',
        img_prefix=data_root + 'testB/',
        pipeline=test_pipeline))
evaluation = dict(interval=10000, metric=['bbox'])