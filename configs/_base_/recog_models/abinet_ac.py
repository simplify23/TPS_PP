num_chars = 37
max_seq_len = 26
# pretrain_vision = 'ckpt/ztl/reg/tps++/Baseline_ABINet_Vision_tps++/latest.pth'
# pretrain_lan = 'ckpt/ztl/reg/pre_train/abinet_academic.pth'
pretrain_vision = '../../../ckpt/ztl/reg/mmocr/Baseline_tps++_abinet_vision/epoch_12.pth'
pretrain_lan = '../../../ckpt/ztl/reg/mmocr_pretrain/abinet_academic.pth'
label_convertor = dict(
    type='ABIConvertor',
    dict_type='DICT36',
    with_unknown=False,
    with_padding=False,
    lower=True,
)

model = dict(
    type='ABINet',
    backbone=dict(type='ResNetABI',
                  init_cfg=dict(
                      type='Pretrained',
                      checkpoint=pretrain_vision,
                      prefix='backbone.',
                  )),
    tpsnet=dict(type='U_TPSnetv2',
                    init_cfg=dict(
                        type='Pretrained',
                        checkpoint=pretrain_vision,
                        prefix='tpsnet.',
                    )
                ),
    encoder=dict(
        type='ABIVisionModel',
        encoder=dict(
            type='TransformerEncoder',
            n_layers=3,
            n_head=8,
            d_model=512,
            d_inner=2048,
            dropout=0.1,
            max_len=8 * 32,
        ),
        decoder=dict(
            type='ABIVisionDecoder',
            in_channels=512,
            num_channels=64,
            attn_height=8,
            attn_width=32,
            attn_mode='nearest',
            use_result='feature',
            num_chars=num_chars,
            max_seq_len=max_seq_len,
            # init_cfg=dict(type='Xavier', layer='Conv2d')),
            ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrain_vision,
            prefix='encoder.',
            )
    ),
    decoder=dict(
        type='ABILanguageDecoder',
        d_model=512,
        n_head=8,
        d_inner=2048,
        n_layers=4,
        dropout=0.1,
        detach_tokens=True,
        use_self_attn=False,
        pad_idx=num_chars - 1,
        num_chars=num_chars,
        max_seq_len=max_seq_len,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrain_lan,
            prefix='decoder.',
        )),
        # init_cfg=None),
    fuser=dict(
        type='ABIFuser',
        d_model=512,
        num_chars=num_chars,
        # init_cfg=None,
        max_seq_len=max_seq_len,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrain_lan,
            prefix='fuser.',
        )),
    # ),
    loss=dict(
        type='ABILoss', enc_weight=1.0, dec_weight=1.0, fusion_weight=1.0),
    label_convertor=label_convertor,
    max_seq_len=max_seq_len,
    iter_size=3)
