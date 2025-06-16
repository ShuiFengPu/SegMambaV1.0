# dataset settings
dataset_type = 'NpyDataset'
data_root = '/tmp/SegMambaV1.0/seg/data/kvasir_npy'
crop_size = (256, 256)

train_pipeline = [
    dict(type='LoadImageFromNpy'),
    dict(type='LoadAnnotationsFromNpy'),
    #dict(type='DebugPrint'), 
    dict(type='RandomResize', scale=(256, 256), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromNpy'),
    dict(type='Resize', scale=(256, 256)),
    dict(type='LoadAnnotationsFromNpy'),
    dict(type='PackSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromNpy'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='LoadAnnotationsFromNpy')],
            [dict(type='PackSegInputs')]
        ])
]

# dataloader 设置
train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train_data',       # 实际匹配 train_data_1.npy ~ train_data_5.npy
            seg_map_path='train_label'   # 实际匹配 train_label_1.npy ~ train_label_5.npy
        ),
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='val_data',
            seg_map_path='val_label'
        ),
        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='test_data',
            seg_map_path='test_label'
        ),
        pipeline=test_pipeline
    )
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
