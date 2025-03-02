# Local training and inference
# LOAD_FROM = '/home/sagemaker-user/Genioos/sofia_thesis_project/detection_models/grounding_dino/pretrained_models/epoch_40_swint-ogc_covers.pth'

# RESUME = False
# DO_SAVE_VISUALIZATIONS = True

# ANN_FILE_TRAINING = '/home/sagemaker-user/data/Covers/Stadtspital-Waid/500_imgs/val/annotations/instances_val.json'
# ANN_FILE_VALIDATION = '/home/sagemaker-user/data/Covers/Stadtspital-Waid/500_imgs/train/annotations/instances_train.json'

# DATA_PREFIX_TRAIN = dict(img='/home/sagemaker-user/data/Covers/Stadtspital-Waid/500_imgs/val/images/')
# DATA_PREFIX_VAL = dict(img='/home/sagemaker-user/data/Covers/Stadtspital-Waid/500_imgs/train/images/')

# BATCH_SIZE_TRAIN = 1
# BATCH_SIZE_VAL = 1

# NUM_WORKER_TRAIN = 2
# NUM_WORKER_VAL = 2


# Training and inference in custom docker
LOAD_FROM = '/opt/ml/code/pretrained_models/groundingdino_swint_ogc_mmdet-822d7e9d.pth'

RESUME = False
DO_SAVE_VISUALIZATIONS = True

ANN_FILE_TRAINING = '/opt/ml/input/data/train/annotations/instances_train.json'
ANN_FILE_VALIDATION = '/opt/ml/input/data/validation/annotations/instances_val.json'

DATA_PREFIX_TRAIN = dict(img='/opt/ml/input/data/train/images/')
DATA_PREFIX_VAL = dict(img='/opt/ml/input/data/validation/images/')

BATCH_SIZE_TRAIN = 1
BATCH_SIZE_VAL = 1

NUM_WORKER_TRAIN = 10
NUM_WORKER_VAL = 10

CLASSES =(
    "Coffee Mug Lid",
    "Coffee Mug Lid (White)",
    "Opaque Plate Cover",
    "Other Lid/Cover",
    "Plastic Wrap",
    "Salad Lid",
    "Soup Lid",
    "Transparent Plate Cover",
    "Cover that occludes food",
    "Cover that is above its tableware"
)

NUM_CLASSES = len(CLASSES)

evaluation = dict(
    interval=1,         # Evaluate after every epoch
    metric='bbox',       # Use bounding box metrics
    classwise=True       # Enables per-class AP logging
)
auto_scale_lr = dict(base_batch_size=32, enable=True)
backend_args = None
data_root = '/opt/ml/input/data/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook', by_epoch=True, max_keep_ckpts=3),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook',
                       draw=DO_SAVE_VISUALIZATIONS,
                       interval=1,
                       test_out_dir='/opt/ml/output/vis_results')
)
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
lang_model_name = 'bert-base-uncased'
launcher = 'none'
load_from = LOAD_FROM
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 40
metainfo = dict(
    classes= CLASSES
)

model = dict(
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=False,
        depths=[
            2,
            2,
            6,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=96,
        mlp_ratio=4,
        num_heads=[
            3,
            6,
            12,
            24,
        ],
        out_indices=(
            1,
            2,
            3,
        ),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        type='SwinTransformer',
        window_size=7,
        with_cp=True),
    bbox_head=dict(
        contrastive_cfg=dict(bias=False, log_scale=0.0, max_text_len=256),
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        # ToDo: set the number of classes automatically.
        num_classes=NUM_CLASSES,
        sync_cls_avg_factor=True,
        type='GroundingDINOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=False,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            cross_attn_text_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        fusion_layer_cfg=dict(
            embed_dim=1024,
            init_values=0.0001,
            l_dim=256,
            num_heads=4,
            v_dim=256),
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=4)),
        num_cp=6,
        num_layers=6,
        text_layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=4))),
    language_model=dict(
        add_pooling_layer=False,
        name='bert-base-uncased',
        pad_to_max=False,
        special_tokens_list=[
            '[CLS]',
            '[SEP]',
            '.',
            '?',
        ],
        type='BertModel',
        use_sub_sentence_represent=True),
    neck=dict(
        act_cfg=None,
        bias=True,
        in_channels=[
            192,
            384,
            768,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=4,
        out_channels=256,
        type='ChannelMapper'),
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='BinaryFocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='GroundingDINO',
    with_box_refine=True)
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')

param_scheduler = [
    # Warm-up scheduler
    dict(
        type='LinearLR',          # Linear warm-up
        start_factor=0.001,       # Starting LR is 0.1% of the base LR
        by_epoch=False,           # Apply warm-up by iteration, not by epoch
        begin=0,                  # Start from the very first iteration
        end=50                    # End at the 50th iteration # original: 250
    )
    
    # Uncomment the section below if you want to apply a linear decay after warm-up
    # dict(
    #     type='LinearLR',         # Linear learning rate decay
    #     start_factor=1.0,        # Start at full base LR after warm-up
    #     end_factor=0.01,         # Decay to 1% of the base LR
    #     by_epoch=True,           # Apply decay by epoch
    #     begin=51,                # Start decay right after warm-up ends
    #     end=36                   # End decay at the 36th epoch
    # )
]

resume = RESUME

train_cfg = dict(max_epochs = max_epochs, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=BATCH_SIZE_TRAIN,
    dataset=dict(
        #ToDo: Set the filename from the jupyter notebook (instead of hardcoing here)
        #Define name of annotation file
        ann_file=ANN_FILE_TRAINING,
        backend_args=None,
        data_prefix=DATA_PREFIX_TRAIN,
        data_root='/opt/ml/input/data/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            #dict(type='LoadTextAnnotations'),  # new
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
                                    4200,
                                ),
                                (
                                    500,
                                    4200,
                                ),
                                (
                                    600,
                                    4200,
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
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'text',
                    'custom_entities',
                ),
                type='PackDetInputs'),
        ],
        return_classes=True,
        type='CocoDataset',
        metainfo=dict(classes=CLASSES),
        ),
    num_workers=NUM_WORKER_TRAIN,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler')
)

val_dataloader = dict(
    batch_size=BATCH_SIZE_VAL,
    num_workers=NUM_WORKER_VAL,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        return_classes=True,
        metainfo=dict(classes=CLASSES),
        # ToDo: load the validation set name dynamically
        ann_file=ANN_FILE_VALIDATION,  # Validation annotations
        data_prefix=DATA_PREFIX_VAL,  # Validation images
        filter_cfg=dict(filter_empty_gt=False),
        pipeline = [
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                keep_ratio=True,
                scale=(800, 1333),
                type='FixScaleResize'
            ),
            dict(type='LoadAnnotations', with_bbox=True),
            # dict(type='LoadTextAnnotations'),  # For GroundingDINO
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'text',
                    'custom_entities',
                ),
                type='PackDetInputs'),
        ]
    ),
    sampler=dict(shuffle=False, type='DefaultSampler')  # No shuffling for validation
)

custom_imports = dict(
    imports=['mmdet.evaluation.metrics.coco_metric_open_set_detection'],  # Full module path
    allow_failed_imports=False  # Ensures import failure raises an error
)

val_cfg = dict(type='ValLoop')

val_evaluator = dict(
    type='OpenSetCOCOMetric',
    ann_file=ANN_FILE_VALIDATION,
    metric=['bbox'],  # Metrics for both bounding boxes and segmentation
    classwise=True,            # Enable class-wise mAP
)

vis_backends = [
    dict(type='LocalVisBackend', save_dir='/opt/ml/output/vis_results')
]

visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=vis_backends
)

work_dir = '/opt/ml/checkpoints'

test_dataloader = dict(
    batch_size=BATCH_SIZE_VAL,
    num_workers=NUM_WORKER_VAL,
    persistent_workers=True,
    dataset=dict(
        type='CocoDataset',
        metainfo=dict(classes=CLASSES),
        ann_file=ANN_FILE_VALIDATION,  # Using validation set for testing
        data_prefix=DATA_PREFIX_VAL,  # Test images from validation dataset
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                800,
                1333,
            ), type='FixScaleResize'),
            dict(type='LoadAnnotations', with_bbox=True),
            #dict(type='LoadTextAnnotations'),  # For GroundingDINO
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                    'flip',
                    'flip_direction',
                    'text',
                    'custom_entities',
                ),
                type='PackDetInputs'),
        ]
    ),
    sampler=dict(shuffle=False, type='DefaultSampler')  # No shuffling for test
)

test_cfg = dict(type='TestLoop')  

# custom_imports = dict(
#     imports=['mmdet.evaluation.metrics.coco_metric_open_set_detection'],  # Full module path
#     allow_failed_imports=False  # Ensures import failure raises an error
# )

test_evaluator = dict(
    type='OpenSetCOCOMetric',
    ann_file=ANN_FILE_VALIDATION,
    metric=['bbox'],  # Metrics for bounding boxes
    classwise=True,  # Enable class-wise mAP for detailed evaluation
    outfile_prefix='/opt/ml/output/vis_results/eval_'  # ✅ No trailing comma
)