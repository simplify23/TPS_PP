optimizer = dict(type='Adam', lr=1e-4)
# optimizer_config = dict(type='GradientCumulativeOptimizerHook', cumulative_iters=4)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    step=[8, 10],
    warmup='linear',
    warmup_iters=1,
    warmup_ratio=0.001,
    warmup_by_epoch=True)
total_epochs = 12
