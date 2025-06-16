_base_ = [
    '../_base_/models/deeplabv3_r50-d8.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
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
        distillation=False, drop_path=0.03, ssm_ratio=2, forward_type="v052d",
        sync_bn=False, out_indices=(1, 2, 3),
        pretrained='../../weights/MobileMamba_B4/mobilemamba_b4.pth',
        frozen_stages=-1, norm_eval=False,),
    decode_head=dict(in_channels=448, channels=256, num_classes=150, in_index=2,),
    auxiliary_head=dict(in_channels=376, num_classes=150, in_index=1,)
)

ratio = 1
bs_ratio = 4  # 0.00012 for 4 * 8

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='AdamW', lr=0.00012 * ratio, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                   'relative_position_bias_table': dict(decay_mult=0.),
                                                   'norm': dict(decay_mult=0.)}),
    clip_grad=dict(_delete_=True, max_norm=0.1, norm_type=2), )

max_iters = 15000
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0, end=500),
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

