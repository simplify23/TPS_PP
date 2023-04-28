# _base_ = [
#     '../../_base_/default_runtime.py',
#     '../../_base_/schedules/schedule_adam_step_6e.py',
#     '../../_base_/recog_pipelines/nrtr_pipeline.py',
#     '../../_base_/recog_datasets/ST_MJ_train.py',
#     '../../_base_/recog_datasets/academic_test_high.py'
# ]
_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_12e.py',
    '../../_base_/recog_pipelines/crnn_pp_pipeline.py',
    '../../_base_/recog_datasets/ST_MJ_alphanumeric_train.py',
    '../../_base_/recog_datasets/academic_test_high.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}
# find_unused_parameters = True
kd_loss = False
label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=True)

model = dict(
    type='NRTR',
    # preprocessor=dict(
    #      type='TPSPreprocessor',
    #      num_fiducial=20,
    #      img_size=(32, 128),
    #      rectified_img_size=(32, 128),
    #      num_img_channel=3),
    backbone=dict(type='ResNetABI_v2_large',
                  arch_settings=[3, 4, 6, 6, 3],
                  # arch_settings=[1, 2, 4, 4, 1],
                  strides=[2, 1, 2, 1, 2], ),
    tpsnet=dict(type='TPS_PP'),
    encoder=dict(type='NRTREncoder'),
    decoder=dict(type='NRTRDecoder'),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=40)

data = dict(
    samples_per_gpu=280,
    workers_per_gpu=10,
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
