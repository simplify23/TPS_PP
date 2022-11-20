_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_12e.py',
    '../../_base_/recog_pipelines/pren_pipeline.py',
    '../../_base_/recog_datasets/ST_MJ_alphanumeric_train.py',
    '../../_base_/recog_datasets/academic_test_low.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}
find_unused_parameters = True
# Model
num_chars = 37
max_seq_len = 26
channel = 384
label_convertor = dict(
    type='ABIConvertor',
    dict_type='DICT36',
    with_unknown=False,
    with_padding=False,
    lower=True,
)

model = dict(
    type='ABINet',
    # backbone=dict(type='ResNetABI'),
    backbone = dict(type='efficientnet'),
    # tpsnet=dict(type='U_TPSnet_v3',),
    encoder=dict(
        type='ABIVisionModel',
        encoder=dict(
            type='TransformerEncoder',
            n_layers=2,
            n_head=8,
            d_model=channel,
            d_inner=channel*2,
            dropout=0.1,
            max_len=8 * 32,
        ),
        decoder=dict(
            type='CommonVisionDecoder',
            in_channels=channel,
            attn_mode='nearest',
            use_result='feature',
            num_chars=num_chars,
            max_seq_len=max_seq_len,
            init_cfg=dict(type='Xavier', layer='Conv2d')),
    ),
    loss=dict(
        type='ABILoss', enc_weight=1.0, dec_weight=1.0, fusion_weight=1.0),
    label_convertor=label_convertor,
    max_seq_len=max_seq_len,
    iter_size=1)

data = dict(
    samples_per_gpu=300,
    workers_per_gpu=12,
    val_dataloader=dict(samples_per_gpu=10),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')
