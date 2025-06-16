bs_ratio = 2
crop_size = (
    512,
    512,
)
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'my_mmseg.models.decode_heads.unet_head',
        'configs._base_.datasets.register',
        'configs._base_.datasets.transforms.map_255_to_1',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        512,
        512,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = '/tmp/SegMambaV1.0/seg/data/kvasir-seg-jpg/Kvasir-SEG'
dataset_type = 'KvasirDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw=True, show=True, type='SegVisualizationHook', wait_time=2))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'work_dirs/U-Net_MobileMamba/iter_48000.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
max_iters = 80000
model = dict(
    auxiliary_head=dict(
        align_corners=True,
        channels=128,
        dropout_ratio=0.1,
        in_channels=376,
        in_index=1,
        loss_decode=dict(
            loss_weight=0.4, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=1,
        type='UNetHead'),
    backbone=dict(
        depth=[
            2,
            3,
            2,
        ],
        distillation=False,
        down_ops=[
            [
                'subsample',
                2,
            ],
            [
                'subsample',
                2,
            ],
            [
                '',
            ],
        ],
        drop_path=0.03,
        embed_dim=[
            200,
            376,
            448,
        ],
        forward_type='v052d',
        frozen_stages=-1,
        global_ratio=[
            0.8,
            0.7,
            0.6,
        ],
        img_size=224,
        in_chans=3,
        kernels=[
            7,
            5,
            3,
        ],
        local_ratio=[
            0.2,
            0.2,
            0.3,
        ],
        norm_eval=False,
        num_classes=80,
        out_indices=(
            1,
            2,
            3,
        ),
        pretrained='../weights/MobileMamba_B4/mobilemamba_b4.pth',
        ssm_ratio=2,
        stages=[
            's',
            's',
            's',
        ],
        sync_bn=False,
        type='MobileMamba'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            512,
            512,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=True,
        channels=256,
        dilations=(
            1,
            6,
        ),
        dropout_ratio=0.1,
        in_channels=448,
        in_index=2,
        loss_decode=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=2,
        type='UNetHead'),
    pretrained=None,
    test_cfg=dict(mode='whole'),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.00012, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0))),
    type='OptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=1e-05, type='LinearLR'),
    dict(
        T_max=40000,
        begin=40000,
        by_epoch=False,
        end=80000,
        eta_min=0,
        type='CosineAnnealingLR'),
]
ratio = 1
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='images', seg_map_path='masks'),
        data_root='/tmp/SegMambaV1.0/seg/data/kvasir-seg-jpg/Kvasir-SEG',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(imdecode_backend='pillow', type='LoadAnnotations'),
            dict(type='Map255To1'),
            dict(type='PackSegInputs'),
        ],
        type='KvasirDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(imdecode_backend='pillow', type='LoadAnnotations'),
    dict(type='Map255To1'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=80000, type='IterBasedTrainLoop', val_interval=8000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(img_path='images', seg_map_path='masks'),
        data_root='/tmp/SegMambaV1.0/seg/data/kvasir-seg-jpg/Kvasir-SEG',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(imdecode_backend='pillow', type='LoadAnnotations'),
            dict(type='Map255To1'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    512,
                    512,
                ),
                type='RandomResize'),
            dict(
                cat_max_ratio=0.75, crop_size=(
                    512,
                    512,
                ), type='RandomCrop'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs'),
        ],
        type='KvasirDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(imdecode_backend='pillow', type='LoadAnnotations'),
    dict(type='Map255To1'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            512,
            512,
        ),
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=(
        512,
        512,
    ), type='RandomCrop'),
    dict(keep_ratio=False, scale=(
        512,
        512,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(img_path='images', seg_map_path='masks'),
        data_root='/tmp/SegMambaV1.0/seg/data/kvasir-seg-jpg/Kvasir-SEG',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(keep_ratio=False, scale=(
                512,
                512,
            ), type='Resize'),
            dict(imdecode_backend='pillow', type='LoadAnnotations'),
            dict(type='Map255To1'),
            dict(type='PackSegInputs'),
        ],
        type='KvasirDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    save_dir='work_dirs/U-Net_MobileMamba/test_vis',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/U-Net_MobileMamba'
