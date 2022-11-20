# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from .base_decoder import BaseDecoder

from mmocr.models.builder import DECODERS
from mmocr.models.common.modules import (PositionwiseFeedForward,
                                         PositionalEncoding,
                                         MultiHeadAttention)

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention Module. This code is adopted from
    https://github.com/jadore801120/attention-is-all-you-need-pytorch.

    Args:
        temperature (float): The scale factor for softmax input.
        attn_dropout (float): Dropout layer on attn_output_weights.
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        if mask !=None:
            attn = torch.matmul((q + mask) / self.temperature, k.transpose(2, 3))
        else:
            attn = torch.matmul( q / self.temperature, k.transpose(2, 3))
        # if mask is not None:
        #     attn = attn.masked_fill(mask == 0, float('-inf'))

        # scroe = F.softmax(attn, dim=-1).mean(-1)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn

class Mask_MultiHeadAttention(nn.Module):
    """Multi-Head Attention module.

    Args:
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
    """

    def __init__(self,
                 n_head=8,
                 d_model=512,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v

        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias=qkv_bias)
        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias=qkv_bias)

        self.attention = ScaledDotProductAttention(d_k**0.5, dropout)

        self.fc = nn.Linear(self.dim_v, d_model, bias=qkv_bias)
        self.proj_drop = nn.Dropout(dropout)

    def forward_train(self, q, k, v, mask=None):
        batch_size_q, len_q, _ = q.size()
        batch_size, len_k, _ = k.size()

        q = self.linear_q(q).view(len_q*2, batch_size,len_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.n_head, self.d_v)
        mask = mask.view(len_q*2, batch_size,len_q, self.n_head, self.d_k)

        q, k, v, mask  = q.transpose(-2,-3), k.transpose(1, 2), v.transpose(1, 2), mask.transpose(-2,-3)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)

        attn_out, _ = self.attention(q, k, v, mask=mask)

        attn_out = attn_out.transpose(-2,-3).contiguous().view(
            batch_size_q, len_q, self.dim_v)

        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)

        return attn_out
    def forward_test(self, q, k, v, mask=None):
        batch_size, len_q, _ = q.size()
        _, len_k, _ = k.size()

        q = self.linear_q(q).view(batch_size, len_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.n_head, self.d_v)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)

        attn_out, _ = self.attention(q, k, v, mask=mask)

        attn_out = attn_out.transpose(1, 2).contiguous().view(
            batch_size, len_q, self.dim_v)

        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)
        return attn_out

    def forward(self,q,k,v,mask,ifmask):
        if ifmask == True:
            output = self.forward_train(q,k,v,mask)
        else:
            output = self.forward_test(q,k,v,mask)
        return output

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
        self.attn = Mask_MultiHeadAttention(
                n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_cfg=act_cfg)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q,k,v, mask=None,ifmask=True):
        residual = q
        x = residual + self.attn(q, k, v, mask,ifmask)
        x = self.norm1(x)

        residual = x
        x = residual + self.mlp(x)
        x = self.norm2(x)

        return x

@DECODERS.register_module()
class TFCommonDecoder(BaseDecoder):
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
                 mask_id = 37,
                 # ifmask = False,  # change mask-attention button
                 **kwargs):
        super().__init__()
        self.layer_stack = ModuleList([
            TFCommonDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout,**kwargs)
            for _ in range(n_layers)
        ])
        self.layer_stack_mask = ModuleList([
            TFCommonDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout, **kwargs)
            for _ in range(n_layers)
        ])
        self.max_seq_len = max_seq_len
        self.pos_encoder = PositionalEncoding(512, max_seq_len)
        self.cls = nn.Linear(512, num_classes)
        self.cls_mask= nn.Linear(512, num_classes)
        self.mask_id = mask_id
        # self.ifmask = ifmask

    def _get_location_mask(self, token_len=None, mask_fill = float('-inf'), dot_v =  1,device = None):
        """Generate location masks given input sequence length.

        Args:
            seq_len (int): The length of input sequence to transformer.
            device (torch.device or str, optional): The device on which the
                masks will be placed.

        Returns:
            Tensor: A mask tensor of shape (seq_len, seq_len) with -infs on
            diagonal and zeros elsewhere.
        """
        mask = torch.eye(token_len,device = device)
        other_mask = (1-mask).float().masked_fill(1-mask == 1, mask_fill)
        mask = mask.float().masked_fill(mask == 1, mask_fill)

        return  torch.stack((mask, other_mask), 0) * dot_v

    def _flatten(self, logits, targets_dict):
        target_lens = [len(t) for t in targets_dict['targets']]
        flatten_logits = torch.cat(
            [s[:target_lens[i]] for i, s in enumerate((logits))])
        return flatten_logits

    def forward_train(self, feat,out_enc, targets_dict, img_metas,ifmask=False):

        # Position Attention
        # print(out_enc.shape)
        N, H_W, E = out_enc.size()
        k, v = out_enc, out_enc  # (N, E, H, W)
        # print(k.shape)
        zeros = out_enc.new_zeros((N, self.max_seq_len, E))  # (N, T, E)
        # count = out_enc.new_zeros((N, 1, E))  # (N, 1, E)
        q = self.pos_encoder(zeros)



        if ifmask == True:
            four_dim_q = q.unsqueeze(1).repeat(2,H_W,1,1)
            q = four_dim_q.view(N*H_W*2, H_W, E)
            # q = self.pos_encoder(zeros)  # (N, T, E)
            # mask module
            mask_one_zeros = self._get_location_mask(token_len = H_W, mask_fill = float(self.mask_id), dot_v = 1,device = q.device)

            mask_one_zeros = mask_one_zeros.repeat(N,1,1).unsqueeze(-1).repeat(1,1,1,E)
            mask = mask_one_zeros.view(N*H_W*2, H_W, E)
        else:
            mask = None


        for enc_layer in self.layer_stack:
            q = enc_layer(q,k,v,mask, ifmask)
        # x = self.layer_stack(q,k,v,mask)
        x = self.cls(q)
        #x  (B T 37)
        if targets_dict != None:
            flatten_q = self._flatten(x,targets_dict)
        # for enc_layer in self.count_layer_stack:
        #     count = enc_layer(count,k,v, None, False)
        # count = self.mlp(count)
        # x = torch.cat((count,x),1)
        return x

    def forward_test(self, feat,out_enc, img_metas):
        return self.forward_train(feat=feat,out_enc=out_enc,img_metas=img_metas,targets_dict=None,ifmask=False)