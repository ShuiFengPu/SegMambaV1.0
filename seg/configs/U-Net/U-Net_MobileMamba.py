# 导入自定义模块
custom_imports = dict(
    imports=['my_mmseg.models.decode_heads.unet_head',
    'configs._base_.datasets.register',
    'configs._base_.datasets.transforms.map_255_to_1',
    'my_mmseg.datasets.npy_dataset', 
    'my_mmseg.datasets.transforms.load_npy',
    #'my_mmseg.datasets.transforms.debug'
],
    allow_failed_imports=False
)
_base_ = [
    '../_base_/models/U-Net.py',        # 基于你自定义的 U-Net model 定义
    '../_base_/datasets/kvasir_npy.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='MobileMamba',
        img_size=224,
        in_chans=3,
        num_classes=80,
        stages=['s', 's', 's'],
        embed_dim=[200, 376, 448],
        global_ratio=[0.8, 0.7, 0.6],
        local_ratio=[0.2, 0.2, 0.3],
        depth=[2, 3, 2],
        kernels=[7, 5, 3],
        down_ops=[['subsample', 2], ['subsample', 2], ['']],
        distillation=False,
        drop_path=0.03,
        ssm_ratio=2,
        forward_type="v052d",
        sync_bn=False,
        out_indices=(1, 2, 3),
        pretrained='../weights/MobileMamba_B4/mobilemamba_b4.pth',
        frozen_stages=-1,
        norm_eval=False,
    ),
    decode_head=dict(
        type='UNetHead',          # 如果想保留 U-NetHead 解码方式
        ignore_index=255,
        in_channels=448,           # 对应 MobileMamba 最后一个 stage 的输出通道
        in_index=2,                # out_indices 中的第三项
        channels=256,
        num_convs=2,
        dilations=(1, 6),
        dropout_ratio=0.1,
        num_classes=2,           
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=True,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    ),
    auxiliary_head=dict(
        _delete_=True,
        type='UNetHead',           # 或者使用 FCNHead: type='FCNHead'
        ignore_index=255,
        in_channels=376,           # 对应第二个 stage 的输出通道
        in_index=1,
        channels=128,
        num_convs=1,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=True,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
    )
)

# === 训练参数设置 ===
ratio = 1
bs_ratio = 2 # 4 GPU 时每卡 2 张图，总 bs=8

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.00012 * ratio,
        betas=(0.9, 0.999),
        weight_decay=0.05
    ),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.0),
        'relative_position_bias_table': dict(decay_mult=0.0),
        'norm': dict(decay_mult=0.0)
    }),
    clip_grad=dict(_delete_=True, max_norm=0.1, norm_type=2)
)

max_iters = 80000
param_scheduler = [
    dict(type='LinearLR', start_factor=1e-5, by_epoch=False, begin=0, end=500),
    dict(
        type='CosineAnnealingLR',
        begin=max_iters // 2,
        T_max=max_iters // 2,
        end=max_iters,
        by_epoch=False,
        eta_min=0
    )
]

train_dataloader = dict(
    batch_size=2 * bs_ratio * ratio,
    num_workers=min(2 * bs_ratio * ratio, 8),
)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = val_dataloader
