# Grounding DINO config for KSW Patient Cards dataset (reshuffled_521_74)
# Usage: Training on SageMaker with s3://genioosfood-data/Patient_Cards/KSW/reshuffled_521_74/

default_scope = "mmdet"

# From instances_train.json (categories by id) - must match JSON exactly
# Run notebook cell 1.1 to verify labels from S3; update if different
PATIENT_CARD_CLASSES = ("paper",)

custom_imports = dict(
    allow_failed_imports=False,
    imports=["mmdet.evaluation.metrics.coco_metric_open_set_detection"],
)

# Data paths (SageMaker mounts train -> /opt/ml/input/data/train, validation -> /opt/ml/input/data/validation)
data_root = "/opt/ml/input/data/"
dataset_type = "CocoDataset"

train_dataloader = dict(
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    batch_size=1,
    dataset=dict(
        ann_file="train/annotations/instances_train.json",
        backend_args=None,
        data_prefix=dict(img="train/images/"),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(classes=PATIENT_CARD_CLASSES),
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(classes=PATIENT_CARD_CLASSES, type="LoadTextAnnotations"),
            dict(prob=0.5, type="RandomFlip"),
            dict(
                transforms=[
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (480, 1333),
                                (512, 1333),
                                (544, 1333),
                                (576, 1333),
                                (608, 1333),
                                (640, 1333),
                                (672, 1333),
                                (704, 1333),
                                (736, 1333),
                                (768, 1333),
                                (800, 1333),
                            ],
                            type="RandomChoiceResize",
                        ),
                    ],
                    [
                        dict(
                            keep_ratio=True,
                            scales=[(400, 4200), (500, 4200), (600, 4200)],
                            type="RandomChoiceResize",
                        ),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(384, 600),
                            crop_type="absolute_range",
                            type="RandomCrop",
                        ),
                        dict(
                            keep_ratio=True,
                            scales=[
                                (480, 1333),
                                (512, 1333),
                                (544, 1333),
                                (576, 1333),
                                (608, 1333),
                                (640, 1333),
                                (672, 1333),
                                (704, 1333),
                                (736, 1333),
                                (768, 1333),
                                (800, 1333),
                            ],
                            type="RandomChoiceResize",
                        ),
                    ],
                ],
                type="RandomChoice",
            ),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                    "flip",
                    "flip_direction",
                    "text",
                    "custom_entities",
                ),
                type="PackDetInputs",
            ),
        ],
        return_classes=True,
        type="CocoDataset",
    ),
    num_workers=32,
    persistent_workers=True,
    sampler=dict(shuffle=True, type="DefaultSampler"),
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file="validation/annotations/instances_val.json",
        data_prefix=dict(img="validation/images/"),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=True),
        metainfo=dict(classes=PATIENT_CARD_CLASSES),
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(keep_ratio=True, scale=(800, 1333), type="FixScaleResize"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(classes=PATIENT_CARD_CLASSES, type="LoadTextAnnotations"),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                    "flip",
                    "flip_direction",
                    "text",
                    "custom_entities",
                ),
                type="PackDetInputs",
            ),
        ],
        type="CocoDataset",
    ),
    num_workers=32,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file="validation/annotations/instances_val.json",
        data_prefix=dict(img="validation/images/"),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=True),
        metainfo=dict(classes=PATIENT_CARD_CLASSES),
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(keep_ratio=True, scale=(800, 1333), type="FixScaleResize"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(classes=PATIENT_CARD_CLASSES, type="LoadTextAnnotations"),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "scale_factor",
                    "flip",
                    "flip_direction",
                    "text",
                    "custom_entities",
                ),
                type="PackDetInputs",
            ),
        ],
        type="CocoDataset",
    ),
    num_workers=32,
    persistent_workers=True,
    sampler=dict(shuffle=False, type="DefaultSampler"),
)

val_evaluator = dict(
    ann_file=data_root + "validation/annotations/instances_val.json",
    classes=PATIENT_CARD_CLASSES,
    classwise=True,
    metric=["bbox"],
    type="OpenSetCOCOMetric",
)

test_evaluator = dict(
    ann_file=data_root + "validation/annotations/instances_val.json",
    classes=PATIENT_CARD_CLASSES,
    classwise=True,
    metric=["bbox"],
    type="OpenSetCOCOMetric",
)

# Model
model = dict(
    type="GroundingDINO",
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    all_labels=PATIENT_CARD_CLASSES,
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
    ),
    backbone=dict(
        type="SwinTransformer",
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=False,
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[192, 384, 768],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=4,
        bias=True,
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        num_cp=6,
        fusion_layer_cfg=dict(
            embed_dim=1024,
            v_dim=256,
            l_dim=256,
            num_heads=4,
            init_values=0.0001,
        ),
        text_layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=4, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=1024, ffn_drop=0.0),
        ),
    ),
    decoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_text_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
        ),
        return_intermediate=True,
        post_norm_cfg=None,
    ),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20,
    ),
    bbox_head=dict(
        type="GroundingDINOHead",
        num_classes=len(PATIENT_CARD_CLASSES),
        sync_cls_avg_factor=True,
        contrastive_cfg=dict(max_text_len=256, log_scale=0.0, bias=False),
        loss_cls=dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
    ),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
    ),
    train_cfg=dict(
        assigner=dict(
            type="HungarianAssigner",
            match_costs=[
                dict(type="BinaryFocalLossCost", weight=2.0),
                dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                dict(type="IoUCost", iou_mode="giou", weight=2.0),
            ],
        ),
    ),
    test_cfg=dict(max_per_img=300),
    language_model=dict(
        type="BertModel",
        name="bert-base-uncased",
        add_pooling_layer=False,
        use_sub_sentence_represent=True,
        pad_to_max=False,
        special_tokens_list=["[CLS]", "[SEP]", ".", "?"],
    ),
)

# Training
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=30, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            "absolute_pos_embed": dict(decay_mult=0.0),
            "backbone": dict(lr_mult=0.1),
        },
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=50,
    ),
]

auto_scale_lr = dict(enable=True, base_batch_size=32)

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=30,
    ),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    timer=dict(type="IterTimerHook"),
    visualization=dict(
        type="DetVisualizationHook",
        draw=True,
        interval=10,
        show=False,
    ),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
log_level = "INFO"
load_from = "/opt/ml/code/pretrained_models/groundingdino_swint_ogc_mmdet-822d7e9d.pth"
resume = False

evaluation = dict(interval=1, metric="bbox", classwise=True)

# Output
work_dir = "/opt/ml/checkpoints"
vis_backends = [
    dict(save_dir="/opt/ml/output/data/visualizations", type="LocalVisBackend")
]
visualizer = dict(
    type="DetLocalVisualizer",
    name="visualizer",
    save_dir="/opt/ml/output/data/visualizations",
    vis_backends=[
        dict(save_dir="/opt/ml/output/data/visualizations", type="LocalVisBackend"),
    ],
)
