_base_ = [
    '../../_base_/default_runtime.py',
    # '../../_base_/schedules/schedule_adam_custom_key_step_10e.py',
    '../../_base_/schedules/schedule_adam_step_12e.py',
    '../../_base_/recog_pipelines/abinet_pipeline.py',
    '../../_base_/recog_datasets/ST_MJ_alphanumeric_train.py',
    # '../../_base_/recog_datasets/ST_MJ_ssd_train.py',
    # '../../_base_/recog_datasets/ST_debug_train.py',
    # '../../_base_/recog_datasets/MJ_train.py',
    '../../_base_/recog_datasets/academic_test_low.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}
# fp16 = dict(loss_scale=512.)
train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}
runner = dict(type='RunnerWrapper', max_epochs=12)
find_unused_parameters = True
# Model
num_chars = 37
max_seq_len = 30 #26
de_label_convertor = dict(
    type='ABIConvertor',
    dict_type='DICT36',
    with_unknown=False,
    with_padding=False,
    lower=True,
)
#
# de_label_convertor = dict(
#     type='AttnConvertor', dict_type='DICT90', with_unknown=True)
en_label_convertor = dict(
    type='CTCConvertor', dict_type='DICT36', with_unknown=True, lower=True)

model = dict(
    type='TPSNet',
    # preprocessor=dict(
    #     type='SPIN',),
    # preprocessor=dict(
    #     type='TPSPreprocessor',
    #     num_fiducial=20,
    #     img_size=(32, 128),
    #     rectified_img_size=(32, 128),
    #     num_img_channel=3,
    #     init_cfg=dict(
    #                 type='Pretrained',
    #                 checkpoint='ckpt/ztl/reg/mmocr/Backbonev14_1_abstudy_first_layer_tps_training_end_to_end/latest.pth',
    #                 prefix='preprocessor.',
    #     )
    # ),
    tpsnet = dict(type='U_TPSnet_v3',),
    backbone = dict(type='ResNet31OCR'),
    # backbone=dict(type='ResNetABI_v2_large',
    #               arch_settings=[3, 4, 6, 6, 3],
    #               # arch_settings=[1, 2, 4, 4, 1],
    #               strides=[2, 1, 2, 1, 2],),
    #               # embed_dims=[64, 128, 320, 512],
    #               # mlp_ratios=[8, 8, 4, 4],
    #               # depths=[3, 3, 12, 3],
    #               # ),
    # backbone=dict(type='ResNetABI_v2_large',
    #               # arch_settings=[3, 4, 6, 6, 3],
    #               arch_settings=[1, 2, 4, 4, 1],
    #               strides=[2, 1, 2, 1, 2],
    #               init_cfg=dict(
    #                 type='Pretrained',
    #                 checkpoint='ckpt/ztl/reg/mmocr/Backbonev16_v3_resnet50_12441_pretrain/latest.pth',
    #                 prefix='backbone.',
    #     )
    #               ),
    # tpsnet=dict(type='U_TPSnet',init_cfg=dict(
    #                 type='Pretrained',
    #                 checkpoint='ckpt/ztl/reg/mmocr/Backbonev16_v5_freeze_train_resnet50_with_utps/latest.pth',
    #                 prefix='tpsnet.',
    #     )),

    # backbone=dict(type='VAN',
    #               embed_dims=[64, 128, 320, 512],
    #               mlp_ratios=[8, 8, 4, 2],
    #               depths=[1, 1, 16, 1],
    #               ),
    # backbone = dict(type='SPTR',
    #                 out_char_num= 64,  # output char patch
    #                 out_channels= 512 , # char patch dim
    #                 # patch_merging: 'Conv'  # 是否使用patch-merging 可选Conv Pool None
    #                 embed_dim =  [64, 64, 128, 256] ,  # 三个阶段的sub-patch dim
    #                 depth = [2, 2, 2, 2],  # 当使用patch-merging时，控制patch-merging所在的层数，分成三阶段，每个阶段的层数
    #                 num_heads= [4, 4, 8, 8],  # 三个阶段中的sub-patch heads
    #                 mixer=['Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'Conv','Conv', 'Conv', 'Conv','Global', 'Global', 'Global',
    #                        'Global','Global', 'Global', 'Global', 'Global', 'Global', 'Global'],# Local atten, Global atten, Conv
    #                 # mixer= ['Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'Conv', 'Global', 'Global', 'Global', 'Global',
    #                 #         'Global', 'Global', 'Global', 'Global', 'Global', 'Global'],  # Local atten, Global atten, Conv
    #                 local_mixer= [[5, 5], [5, 5], [5, 5], [5, 5]],  # local mixer的范围，7表示高度的范围，11表示宽度的范围
    #                 last_stage = False,
    #                 ),
    encoder=dict(
            type='UTransformerEncoder',
            n_layers=2,
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
    # decoder=dict(
    #         type='MaskLanSemv2',
    #         max_seq_len=max_seq_len,
    #         n_layers=3,
    #         n_head=8,
    #         d_k=64,
    #         d_v=64,
    #         d_model=512,
    #         d_inner=1024,
    #         dropout=0.1),
    # fuser=dict(
    #     type='ABIFuser',
    #     d_model=512,
    #     num_chars=num_chars,
    #     init_cfg=None,
    #     max_seq_len=max_seq_len,
    # ),
    loss=dict(type='MixLoss',enc_weight=1,dec_weight=1),
    # loss=dict(
    #     type='ABILoss', enc_weight=1.0, dec_weight=1.0, fusion_weight=1.0),
    en_label_convertor=en_label_convertor,
    label_convertor = de_label_convertor,
    max_seq_len=max_seq_len,
    # iter_size=1
)

data = dict(
    samples_per_gpu=480,
    workers_per_gpu=12,
    # pin_memory=True,
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

evaluation = dict(interval=1, metric='acc', show_mean_scores=True)
