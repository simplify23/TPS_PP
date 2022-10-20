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
    '../../_base_/recog_pipelines/abinet_pipeline.py',
    '../../_base_/recog_datasets/ST_MJ_alphanumeric_train.py',
    '../../_base_/recog_datasets/academic_test_high.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}
find_unused_parameters = True

label_convertor = dict(
    type='AttnConvertor', dict_type='DICT90', with_unknown=True)

model = dict(
    type='NRTR',
    # backbone=dict(
    #     type='ResNet31OCR',
    #     layers=[1, 2, 5, 3],
    #     channels=[32, 64, 128, 256, 512, 512],
    #     stage4_pool_cfg=dict(kernel_size=(2, 1), stride=(2, 1)),
    #     last_stage_pool=True),
    # preprocessor=dict(
    #     type='TPSPreprocessor',
    #     num_fiducial=20,
    #     img_size=(32, 128),
    #     rectified_img_size=(32, 128),
    #     num_img_channel=3),
    backbone=dict(type='ResNetABI',
                  strides=[2, 1, 2, 1, 2], ),
    # tpsnet=dict(type='U_TPSnet_Warp'),
    tpsnet=dict(type='U_TPSnet_v3'),
    encoder=dict(type='NRTREncoder'),
    decoder=dict(type='NRTRDecoder'),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=40)

data = dict(
    samples_per_gpu=128,
    workers_per_gpu=12,
    val_dataloader=dict(samples_per_gpu=10),
    test_dataloader=dict(samples_per_gpu=10),
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
