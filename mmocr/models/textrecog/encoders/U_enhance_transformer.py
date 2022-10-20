# Copyright (c) OpenMMLab. All rights reserved.
import copy

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.runner import BaseModule, ModuleList

from mmocr.models.builder import ENCODERS
# from mmocr.models.common.modules import PositionalEncoding
from mmocr.models.common.modules import (PositionwiseFeedForward,
                                         PositionalEncoding,CBAM,
                                         MultiHeadAttention)
from mmocr.models.textrecog.decoders.transformer_mask import Unet_1d

def shuffle_token(x):
    n, c, h, w = x.shape
    y = torch.zeros([n,c,h*w])
    # print(b)
    for i in range(w):
        for j in range(h):
            y[:][:][i + i + j] = x[:][:][j][i]
    return y

class TFCommonEncoderLayer(BaseModule):
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
                 ifmask=True,
                 dropout=0.1,
                 qkv_bias=False,
                 act_cfg=dict(type='mmcv.GELU')):
        super().__init__()
        # if ifmask ==True:
        #     self.attn = Mask_MultiHeadAttention(
        #         n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        # else:
        #
        # self.unet = Unet_1d(512, 64)
        # self.linear = nn.Linear(512,256)
        self.attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_cfg=act_cfg)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q, k, v, mask=None):
        residual = q
        # v = self.unet(v)
        global_atten = self.attn(q, v, v, mask)
        # local_atten = self.unet(u_f)
        x = global_atten + residual
        # x = torch.cat(self.attn(q, k, k, mask)
        # x = residual + self.attn(q, k, v, mask)
        x = self.norm1(x)

        residual = x
        x = residual + self.mlp(x)
        x = self.norm2(x)

        return x

@ENCODERS.register_module()
class UTransformerEncoder(BaseModule):
    """Implement transformer encoder for text recognition, modified from
    `<https://github.com/FangShancheng/ABINet>`.

    Args:
        n_layers (int): Number of attention layers.
        n_head (int): Number of parallel attention heads.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
        d_inner (int): Hidden dimension of feedforward layers.
        dropout (float): Dropout rate.
        max_len (int): Maximum output sequence length :math:`T`.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 n_layers=2,
                 n_head=8,
                 d_model=512,
                 d_inner=2048,
                 dropout=0.1,
                 max_len=8 * 32,
                 num_classes = 37,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert d_model % n_head == 0, 'd_model must be divisible by n_head'

        self.pos_encoder = PositionalEncoding(d_model, n_position=max_len)
        self.cls = nn.Linear(d_model, num_classes)
        self.cls2 = nn.Linear(num_classes, num_classes)
        self.transformer = self.layer_stack = ModuleList([
            TFCommonEncoderLayer(
        d_model, d_inner, n_head, d_model//n_head, d_model//n_head, dropout=dropout)
                for _ in range(n_layers)])
        # self.transformer = ModuleList(
        #     [copy.deepcopy(encoder_layer) for _ in range(n_layers)])

    def ctc_format(self,logits, feature,flag = True, padding_idx=36):
        if flag == True:
            return logits, feature

        batch_size = logits.size(0)
        output = F.softmax(logits, dim=2)
        # output = output.cpu().detach()
        batch_topk_value, batch_topk_idx = output.topk(1, dim=2)
        batch_max_idx = batch_topk_idx[:, :, 0]
        indexes_b, feature_b = [], []
        scores, indexes = [], []
        feat_len = output.size(1)
        for b in range(batch_size):
            decode_len = min(feat_len, math.ceil(feat_len))
            pred = batch_max_idx[b, :]
            select_idx = []
            prev_idx = padding_idx
            for t in range(decode_len):
                tmp_value = pred[t].item()
                if tmp_value not in (prev_idx, padding_idx):
                    select_idx.append(t)
                prev_idx = tmp_value
            select_idx = torch.LongTensor(select_idx).to(logits.device)

            topk_feat = torch.gather(logits[b, :, :], 0,
                                            select_idx.unsqueeze(-1).repeat(1,37))  # valid_seqlen * topk
            topk_idx = torch.index_select(batch_topk_idx[b, :, :], 0,
                                          select_idx)
            # topk_idx_list, topk_value_list = topk_idx.numpy().tolist(
            # ), topk_value.numpy().tolist()
            feat_len = 30
            # vis_logits = torch.LongTensor(feat_len).fill_(0)
            # # src_target[0] = self.start_idx
            # vis_logits[0:] = topk_idx
            char_num = topk_idx.size(0)
            topk_idx = topk_idx.squeeze()
            padded_target = (torch.ones(feat_len) * padding_idx).long().to(logits.device)
            padded_feat = F.one_hot(padded_target, 37).float()

            if char_num > feat_len:
                padded_target = topk_idx[:feat_len]
                padded_feat = topk_feat[:feat_len,:]
            else:
                padded_target[:char_num] = topk_idx
                padded_feat[:char_num,:] = topk_feat
            indexes_b.append(padded_target)
            feature_b.append(padded_feat)

        indexes = torch.stack(indexes_b,0)
        feat = torch.stack(feature_b,0)
        feat = self.cls2(feat)
        return indexes, feat

    def forward(self, feature):
        """
        Args:
            feature (Tensor): Feature tensor of shape :math:`(N, D_m, H, W)`.

        Returns:
            Tensor: Features of shape :math:`(N, D_m, H, W)`.
        """
        # feature = shuffle_token(feature) # n,c,h*w
        # feature  = feature.transpose(1,2)

        n, c, h, w = feature.shape
        # u_feat = feature.transpose(-1,-2).reshape(n,c,-1)
        feature = feature.view(n, c, -1).transpose(1, 2)  # (n, h*w, c)

        feature = self.pos_encoder(feature)  # (n, h*w, c)
        # feature = feature.transpose(0, 1)  # (h*w, n, c)
        for m in self.transformer:
            feature = m(feature,feature,feature)
        # trans test
        # feature = feature.permute(1,0,2)
        logits = self.cls(feature)
        # _,logits = self.ctc_format(logits, feature,flag = False)
        return logits, feature

        # feature = feature.permute(1, 2, 0)
        # print(feature.shape)

        # abinet test
        # feature = feature.permute(1, 2, 0).view(n, c, h, w)
        # return feature
