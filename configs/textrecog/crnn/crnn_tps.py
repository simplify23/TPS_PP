# _base_ = [
#     '../../_base_/default_runtime.py', '../../_base_/recog_models/crnn.py',
#     '../../_base_/recog_pipelines/crnn_pipeline.py',
#     '../../_base_/recog_datasets/MJ_train.py',
#     '../../_base_/recog_datasets/academic_test_high.py',
#     '../../_base_/schedules/schedule_adadelta_5e.py'
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
runner = dict(type='RunnerWrapper', max_epochs=12)
# model
label_convertor = dict(
    type='CTCConvertor', dict_type='DICT36', with_unknown=False, lower=True)

model = dict(
    type='CRNNNet',
    # kd_loss = kd_loss,
    # preprocessor=dict(
    #     type='TPS_PPv2',
    # #     point_size=(2,16),
    # #     p_stride=2,
    # ),
    backbone=dict(type='ResNetABI_v2_large',
                  in_channels = 3,
                  strides=[2, 1, 2, 1, 2],),
    # tpsnet=dict(type='TPS_PP'),
    # tpsnet=dict(type='U_TPSnet_v3',
    #             ),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    # decoder=None,
    loss=dict(type='CTCLoss'),
    label_convertor=label_convertor,
    pretrained=None)
data = dict(
    samples_per_gpu=450,
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

cudnn_benchmark = True
