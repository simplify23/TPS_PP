ratio = 0.0
optimizer = dict(type='Adam', lr=1e-4,
                 paramwise_cfg=dict(
                     custom_keys={'backbone': dict(lr_mult=ratio),
                                  # 'layer1': dict(lr_mult=ratio),
                                  # 'layer2': dict(lr_mult=ratio),
                                  # 'layer3': dict(lr_mult=ratio),
                                  # 'layer4': dict(lr_mult=ratio),
                                  # 'layer5': dict(lr_mult=ratio),
                                  'encoder': dict(lr_mult=ratio),
                                  }))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    step=[6,9],
    warmup='linear',
    warmup_iters=2,
    warmup_ratio=0.001,
    warmup_by_epoch=True)
total_epochs = 10
