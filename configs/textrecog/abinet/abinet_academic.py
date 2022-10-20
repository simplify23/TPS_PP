_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_15e.py',
    '../../_base_/recog_pipelines/abinet_pipeline.py',
    '../../_base_/recog_models/abinet_ac.py',
    '../../_base_/recog_datasets/ST_MJ_alphanumeric_train.py',
    '../../_base_/recog_datasets/academic_test_high.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}
train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}
runner = dict(type='RunnerWrapper', max_epochs=15)
find_unused_parameters = True
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=8,
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
