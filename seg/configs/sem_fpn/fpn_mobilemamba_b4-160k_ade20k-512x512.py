_base_ = [
    '../_base_/models/fpn_r50.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]#指定该配置文件所依赖的基础配置文件
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        _delete_=True,#完全重写配置
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
        distillation=False, drop_path=0.03, ssm_ratio=2, forward_type="v052d",
        sync_bn=False, out_indices=(1, 2, 3),
        pretrained='../../weights/MobileMamba_B4/mobilemamba_b4.pth',
        frozen_stages=-1, norm_eval=False,),
    neck=dict(
        in_channels=[200, 376, 448],
        num_outs=3),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256],
        in_index=[0, 1, 2],
        feature_strides=[16, 32, 64],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),)


ratio = 1
bs_ratio = 4  # 0.00012 for 4 * 8

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00012 * ratio, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

max_iters = 80000 * 2
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='CosineAnnealingLR',
        begin=max_iters // 2,
        T_max=max_iters // 2,
        end=max_iters,
        by_epoch=False,
        eta_min=0)
]

train_dataloader = dict(
    batch_size=2 * bs_ratio * ratio,
    num_workers=min(2 * bs_ratio * ratio, 8),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,)
test_dataloader = val_dataloader
