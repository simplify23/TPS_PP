_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_20e.py',
    '../../_base_/recog_pipelines/abinet_pipeline.py',
    '../../_base_/recog_datasets/ST_MJ_alphanumeric_train.py',
    '../../_base_/recog_datasets/academic_test_high.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

# Model
num_chars = 37
max_seq_len = 64 #26
de_label_convertor = dict(
    type='ABIConvertor',
    dict_type='DICT36',
    with_unknown=False,
    with_padding=False,
    lower=True,
)

# de_label_convertor = dict(
#     type='AttnConvertor', dict_type='DICT90', with_unknown=True)
en_label_convertor = dict(
    type='CTCConvertor', dict_type='DICT36', with_unknown=False, lower=True)

model = dict(
    type='MixNet',
    backbone=dict(type='ResNetABI'),
    encoder=dict(
            type='TransformerEncoder',
            n_layers=4,
            n_head=8,
            d_model=512,
            d_inner=1024,
            dropout=0.1,
            max_len=8 * 32,
        ),
        # decoder=dict(
        #     type='ABIVisionDecoder',
        #     in_channels=512,
        #     num_channels=64,
        #     attn_height=8,
        #     attn_width=32,
        #     attn_mode='nearest',
        #     use_result='feature',
        #     num_chars=num_chars,
        #     max_seq_len=max_seq_len,
        #     init_cfg=dict(type='Xavier', layer='Conv2d')),
    decoder=dict(
            type='TFCommonDecoder',
            max_seq_len=max_seq_len,
            n_layers=3,
            n_head=8,
            d_k=64,
            d_v=64,
            d_model=512,
            d_inner=1024,
            dropout=0.1),

    loss=dict(type='MixLoss',enc_weight=0.5,dec_weight=0.5),
    # loss=dict(
    #     type='ABILoss', enc_weight=1.0, dec_weight=1.0, fusion_weight=1.0),
    en_label_convertor=None, #en_label_convertor
    label_convertor = de_label_convertor,
    max_seq_len=max_seq_len,
    # iter_size=1
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=12,
    val_dataloader=dict(samples_per_gpu=1),
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
