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
    '../../_base_/recog_pipelines/abinet_pipeline.py',
    '../../_base_/recog_datasets/ST_MJ_alphanumeric_train.py',
    '../../_base_/recog_datasets/academic_test_high.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}
train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}
find_unused_parameters = True
runner = dict(type='RunnerWrapper', max_epochs=12)
# model
label_convertor = dict(
    type='CTCConvertor', dict_type='DICT36', with_unknown=False, lower=True)

model = dict(
    type='CRNNNet',
    # preprocessor=dict(
    #     type='MORAN',),
    # preprocessor=dict(
    #     type='SPIN',),
    # preprocessor=dict(
    #     type='TPSPreprocessor',
    #     num_fiducial=20,
    #     img_size=(32, 128),
    #     rectified_img_size=(32, 128),
    #     num_img_channel=3),
    # backbone=dict(type='VeryDeepVgg', leaky_relu=False, input_channels=1),
    backbone=dict(type='ResNetABI_v2_large',
                  in_channels = 3,
                  strides=[2, 1, 2, 1, 2],),
    # tpsnet=dict(type='U_TPSnet_Warp'),
    tpsnet=dict(type='U_TPSnet_v3'),
    encoder=None,
    decoder=dict(type='CRNNDecoder', in_channels=512, rnn_flag=True),
    loss=dict(type='CTCLoss'),
    label_convertor=label_convertor,
    pretrained=None)
data = dict(
    samples_per_gpu=10,
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

cudnn_benchmark = True
