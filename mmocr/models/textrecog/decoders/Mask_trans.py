# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule, ModuleList
from .base_decoder import BaseDecoder

from mmocr.models.builder import DECODERS
from mmocr.models.common.modules import (MultiHeadAttention,
                                         PositionwiseFeedForward,
                                         PositionalEncoding)

class TFCommonDecoderLayer(BaseModule):
    """Transformer Encoder Layer.

    Args:
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_inner (int): The dimension of the feedforward
            network model (default=256).
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
        act_cfg (dict): Activation cfg for feedforward module.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm')
            or ('norm', 'self_attn', 'norm', 'ffn').
            Default：None.
    """

    def __init__(self,
                 d_model=512,
                 d_inner=1024,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 act_cfg=dict(type='mmcv.GELU')):
        super().__init__()
        self.attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_cfg=act_cfg)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q,k,v, mask=None):
        residual = q
        x = residual + self.attn(q, k, v, mask)
        x = self.norm1(x)

        residual = x
        x = residual + self.mlp(x)
        x = self.norm2(x)

        return x

@DECODERS.register_module()
class MMTrans(BaseDecoder):
    def __init__(self,
                 max_seq_len = 64,
                 n_layers=3,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=1024,
                 dropout=0.1,
                 num_classes=37,
                 **kwargs):
        super().__init__()
        self.layer_stack = ModuleList([
            TFCommonDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout, **kwargs)
            for _ in range(n_layers)
        ])
        self.max_seq_len = max_seq_len
        self.pos_encoder = PositionalEncoding(512, max_seq_len)
        self.cls = nn.Linear(512, num_classes)


    def forward_train(self, feat,out_enc, targets_dict, img_metas):
        # Position Attention
        # print(out_enc.shape)
        N, H_W, E = out_enc.size()
        k, v = out_enc, out_enc  # (N, E, H, W)
        # print(k.shape)
        zeros = out_enc.new_zeros((N, self.max_seq_len, E))  # (N, T, E)
        q = self.pos_encoder(zeros)  # (N, T, E)

        # mask module
        mask = None

        for enc_layer in self.layer_stack:
            q = enc_layer(q,k,v,mask)
        # x = self.layer_stack(q,k,v,mask)
        x = self.cls(q)
        return x

    def forward_test(self, feat,out_enc, img_metas):
        return self.forward_train(feat=feat,out_enc=out_enc,img_metas=img_metas,targets_dict=None)