# augmentation
albu_train_transforms = [
    dict(
        interpolation=1,
        p=0.4,
        rotate_limit=15,
        scale_limit=0.125,
        shift_limit=0.0725,
        type='ShiftScaleRotate'),
    dict(
        p=0.2,
        transforms=[
            dict(
                b_shift_limit=5,
                g_shift_limit=5,
                p=1.0,
                r_shift_limit=5,
                type='RGBShift'),
            dict(
                hue_shift_limit=10,
                p=1.0,
                sat_shift_limit=15,
                type='HueSaturationValue',
                val_shift_limit=10),
            dict(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=1.0,
                type='RandomBrightnessContrast'),
        ],
        type='OneOf'),
    dict(p=0.4, type='HorizontalFlip'),
    dict(p=0.2, type='VerticalFlip'),
    dict(p=0.3, type='RandomRotate90'),
    dict(
        p=0.2,
        transforms=[
            dict(blur_limit=5, p=1.0, type='Blur'),
            dict(blur_limit=5, p=1.0, type='MedianBlur'),
            dict(p=1.0, type='GaussNoise', var_limit=25),
        ],
        type='OneOf'),
]

auto_scale_lr = dict(base_batch_size=16, enable=True)  # original: 16
backend_args = None
data_root = '/opt/ml/input/data/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook', by_epoch=True, max_keep_ckpts=1),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
fp16 = None
load_from = '/opt/ml/code/work_dirs/custom_queryinst_swin_large/weights/custom_queryinst_swin_large_patch4_window7_fpn_300_queries-832c5813.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
param_scheduler = [
    dict(
        type='LinearLR',  # Use linear learning rate warmup
        start_factor=0.001, # Coefficient for learning rate warmup
        by_epoch=False,  # Update the learning rate during warmup at each iteration
        begin=0,  # Starting from the first iteration
        end=250),  # End at the first iteration
    dict(
        type='MultiStepLR',  # Use multi-step learning rate strategy during training
        by_epoch=True,  # Update the learning rate at each epoch
        milestones=[8, 10],  # Learning rate decay at which epochs, epch=8 and epch=10
        gamma=0.1)  # Learning rate decay coefficient
]
max_epochs = 10  # original: 36
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        use_abs_pos_embed=False,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.3,
        drop_rate=0.0,
        embed_dims=192,
        with_cp=False,
        mlp_ratio=4,
        num_heads=[
            6,
            12,
            24,
            48,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=7),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        add_extra_convs='on_input',
        in_channels=[
            192,
            384,
            768,
            1536,
        ],
        num_outs=4,
        out_channels=256,
        start_level=0,
        type='FPN'),
    roi_head=dict(
        bbox_head=[
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=5,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=5,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=5,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=5,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=5,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
            dict(
                bbox_coder=dict(
                    clip_border=False,
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                dropout=0.0,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=7,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv'),
                feedforward_channels=2048,
                ffn_act_cfg=dict(inplace=True, type='ReLU'),
                in_channels=256,
                loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
                loss_cls=dict(
                    alpha=0.25,
                    gamma=2.0,
                    loss_weight=2.0,
                    type='FocalLoss',
                    use_sigmoid=True),
                loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
                num_classes=5,
                num_cls_fcs=1,
                num_ffn_fcs=2,
                num_heads=8,
                num_reg_fcs=3,
                type='DIIHead'),
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=2, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=[
            dict(
                class_agnostic=False,
                conv_kernel_size=3,
                conv_out_channels=256,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=14,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv',
                    with_proj=False),
                in_channels=256,
                loss_mask=dict(
                    activate=False,
                    eps=1e-05,
                    loss_weight=8.0,
                    type='DiceLoss',
                    use_sigmoid=True),
                norm_cfg=dict(type='BN'),
                num_classes=5,
                num_convs=4,
                roi_feat_size=14,
                type='DynamicMaskHead',
                upsample_cfg=dict(scale_factor=2, type='deconv')),
            dict(
                class_agnostic=False,
                conv_kernel_size=3,
                conv_out_channels=256,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=14,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv',
                    with_proj=False),
                in_channels=256,
                loss_mask=dict(
                    activate=False,
                    eps=1e-05,
                    loss_weight=8.0,
                    type='DiceLoss',
                    use_sigmoid=True),
                norm_cfg=dict(type='BN'),
                num_classes=5,
                num_convs=4,
                roi_feat_size=14,
                type='DynamicMaskHead',
                upsample_cfg=dict(scale_factor=2, type='deconv')),
            dict(
                class_agnostic=False,
                conv_kernel_size=3,
                conv_out_channels=256,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=14,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv',
                    with_proj=False),
                in_channels=256,
                loss_mask=dict(
                    activate=False,
                    eps=1e-05,
                    loss_weight=8.0,
                    type='DiceLoss',
                    use_sigmoid=True),
                norm_cfg=dict(type='BN'),
                num_classes=5,
                num_convs=4,
                roi_feat_size=14,
                type='DynamicMaskHead',
                upsample_cfg=dict(scale_factor=2, type='deconv')),
            dict(
                class_agnostic=False,
                conv_kernel_size=3,
                conv_out_channels=256,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=14,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv',
                    with_proj=False),
                in_channels=256,
                loss_mask=dict(
                    activate=False,
                    eps=1e-05,
                    loss_weight=8.0,
                    type='DiceLoss',
                    use_sigmoid=True),
                norm_cfg=dict(type='BN'),
                num_classes=5,
                num_convs=4,
                roi_feat_size=14,
                type='DynamicMaskHead',
                upsample_cfg=dict(scale_factor=2, type='deconv')),
            dict(
                class_agnostic=False,
                conv_kernel_size=3,
                conv_out_channels=256,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=14,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv',
                    with_proj=False),
                in_channels=256,
                loss_mask=dict(
                    activate=False,
                    eps=1e-05,
                    loss_weight=8.0,
                    type='DiceLoss',
                    use_sigmoid=True),
                norm_cfg=dict(type='BN'),
                num_classes=5,
                num_convs=4,
                roi_feat_size=14,
                type='DynamicMaskHead',
                upsample_cfg=dict(scale_factor=2, type='deconv')),
            dict(
                class_agnostic=False,
                conv_kernel_size=3,
                conv_out_channels=256,
                dynamic_conv_cfg=dict(
                    act_cfg=dict(inplace=True, type='ReLU'),
                    feat_channels=64,
                    in_channels=256,
                    input_feat_shape=14,
                    norm_cfg=dict(type='LN'),
                    out_channels=256,
                    type='DynamicConv',
                    with_proj=False),
                in_channels=256,
                loss_mask=dict(
                    activate=False,
                    eps=1e-05,
                    loss_weight=8.0,
                    type='DiceLoss',
                    use_sigmoid=True),
                norm_cfg=dict(type='BN'),
                num_classes=5,
                num_convs=4,
                roi_feat_size=14,
                type='DynamicMaskHead',
                upsample_cfg=dict(scale_factor=2, type='deconv')),
        ],
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=2, type='RoIAlign'),
            type='SingleRoIExtractor'),
        num_stages=6,
        proposal_feature_channel=256,
        stage_loss_weights=[
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        type='SparseRoIHead'),
    rpn_head=dict(
        num_proposals=300,
        proposal_feature_channel=256,
        type='EmbeddingRPNHead'),
    test_cfg=dict(rcnn=dict(mask_thr_binary=0.5, max_per_img=300), rpn=None),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                mask_size=28,
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                mask_size=28,
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                mask_size=28,
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                mask_size=28,
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                mask_size=28,
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
            dict(
                assigner=dict(
                    match_costs=[
                        dict(type='FocalLossCost', weight=2.0),
                        dict(box_format='xyxy', type='BBoxL1Cost', weight=5.0),
                        dict(iou_mode='giou', type='IoUCost', weight=2.0),
                    ],
                    type='HungarianAssigner'),
                mask_size=28,
                pos_weight=1,
                sampler=dict(type='PseudoSampler')),
        ],
        rpn=None),
    type='QueryInst')
num_classes = 5
num_proposals = 300
num_stages = 6
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=(2e-05 * (8**0.5)), type='AdamW', weight_decay=0.0001),  # default lr: 0.0001
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))),
    type='OptimWrapper')
#optimizer = dict(
#    paramwise_cfg=dict(
#        custom_keys=dict(
#            absolute_pos_embed=dict(decay_mult=0.0),
#            norm=dict(decay_mult=0.0),
#            relative_position_bias_table=dict(decay_mult=0.0))))
#optimizer_config = dict(
#    grad_clip=dict(max_norm=1, norm_type=2),
#    type='DistOptimizerHook',
#    update_interval=1,
#    coalesce=True,
#    bucket_size_mb=-1,
#    use_fp16=False)
resume = False
runner = dict(max_epochs=10, type='EpochBasedRunnerAmp')  # original: 36
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/root/Sofia/Genioos/food_recognition_benchmark_EPFL/data/test/test.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/root/Sofia/Genioos/food_recognition_benchmark_EPFL/data/test/images/'
        ),
        data_root='/root/Sofia/Genioos/food_recognition_benchmark_EPFL/data/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/root/Sofia/Genioos/food_recognition_benchmark_EPFL/data/val/val.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        1333,
        800,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
total_epochs = 10 # 36

train_cfg = dict(max_epochs=10, type='EpochBasedTrainLoop', val_interval=1)  # original: 36
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        ann_file=
        '/opt/ml/input/data/train/new_train.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/opt/ml/input/data/train/images/'
        ),
        data_root='/opt/ml/input/data/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(prob=0.5, type='RandomFlip'),
            dict(
                transforms=[
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    400,
                                    1333,
                                ),
                                (
                                    500,
                                    1333,
                                ),
                                (
                                    600,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(
                                384,
                                600,
                            ),
                            crop_type='absolute_range',
                            type='RandomCrop'),
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    1333,
                                ),
                                (
                                    512,
                                    1333,
                                ),
                                (
                                    544,
                                    1333,
                                ),
                                (
                                    576,
                                    1333,
                                ),
                                (
                                    608,
                                    1333,
                                ),
                                (
                                    640,
                                    1333,
                                ),
                                (
                                    672,
                                    1333,
                                ),
                                (
                                    704,
                                    1333,
                                ),
                                (
                                    736,
                                    1333,
                                ),
                                (
                                    768,
                                    1333,
                                ),
                                (
                                    800,
                                    1333,
                                ),
                            ],
                            type='RandomChoiceResize'),
                    ],
                ],
                type='RandomChoice'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.3,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(
        type='AutoAugment',
        policies=[[{
            'type': 'Resize',
            'scales': [(400, 1333), (1200, 1333)],
            'multiscale_mode': 'range',
            'keep_ratio': True
        }],
        [{
            'type': 'Resize',
            'scales': [(400, 1333), (500, 1333), (600, 1333)],
            'multiscale_mode': 'value',
            'keep_ratio': True
        }, {
            'type': 'RandomCrop',
            'crop_type': 'absolute_range',
            'crop_size': (384, 600),
            'allow_negative_crop': True
        }, {
            'type': 'Resize',
            'scales': [(400, 1333), (1200, 1333)],
            'multiscale_mode': 'range',
            'override': True,
            'keep_ratio': True
        }]]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]

val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file=
        '/opt/ml/input/data/validation/val.json',
        backend_args=None,
        data_prefix=dict(
            img=
            '/opt/ml/input/data/validation/images/'
        ),
        data_root='/opt/ml/input/data/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                1333,
                800,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/opt/ml/input/data/train/new_train.json', # change back to validation/val.json
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
# All outputs (log files and checkpoints) will be saved to the working directory:
# work_dir = '/opt/ml/code/work_dirs/custom_queryinst_swin_large/'
# do not specify: checkpoints will be saved under /opt/ml/checkpoints
work_dir = '/opt/ml/checkpoints'
metainfo = {
    'classes': (
    'jam', 'water', 'bread', 'banana', 'coffee'),
}
