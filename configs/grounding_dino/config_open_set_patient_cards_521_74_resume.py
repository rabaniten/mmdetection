# Resume from epoch 30 of sm-gdino-patient-cards-521train-74val-30ep, train 30 more epochs (60 total)
# Requires: RESUME_FROM_CHECKPOINT_S3 = s3://.../epoch_30.pth (wrapper downloads to pretrained_models/epoch_30.pth)

_base_ = './config_open_set_patient_cards_521_74.py'

# Resume from epoch 30 checkpoint (downloaded by wrapper from PRETRAINED_MODEL_S3_URI)
load_from = "/opt/ml/code/pretrained_models/epoch_30.pth"
resume = True

# Train 30 more epochs (30 -> 60 total)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=60, val_interval=1)

# Keep all 30 new checkpoints (epochs 31-60)
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
