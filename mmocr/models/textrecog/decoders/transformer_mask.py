# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from .base_decoder import BaseDecoder

from mmocr.models.builder import DECODERS
from mmocr.models.common.modules import (PositionwiseFeedForward,
                                         PositionalEncoding,CBAM,
                                         MultiHeadAttention)
from ..fusers.abi_fuser import ABIFuser
class ConvModule(nn.Module):
    def __init__(self, in_channels,
                   out_channels,
                   kernel_size,
                   stride,
                   padding):
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )
    def forward(self,x):
        return self.conv1d(x)

class Unet_1d(nn.Module):
# For mini-Unet
    def __init__(self, in_channels=512,
                 num_channels=64,
                 attn_mode='nearest',):
        super().__init__()
        self.k_encoder = nn.Sequential(
            self._encoder_layer(in_channels, num_channels, stride=1),
            self._encoder_layer(num_channels, num_channels, stride=2),
            self._encoder_layer(num_channels, num_channels, stride=2),
            self._encoder_layer(num_channels, num_channels, stride=2))

        self.k_decoder = nn.Sequential(
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=attn_mode),
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=attn_mode),
            self._decoder_layer(
                num_channels, num_channels, scale_factor=2, mode=attn_mode),
            self._decoder_layer(
                num_channels,
                in_channels,
                scale_factor=1,
                mode=attn_mode))
    def _encoder_layer(self,
                       in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=2,
                       padding=1):
        return ConvModule(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,)

    def _decoder_layer(self,
                       in_channels,
                       out_channels,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       mode='nearest',
                       scale_factor=None,
                       size=None):
        align_corners = None if mode == 'nearest' else True
        return nn.Sequential(
            nn.Upsample(
                size=size,
                scale_factor=scale_factor,
                mode=mode,
                align_corners=align_corners),
            ConvModule(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding))

    def forward(self,k):
    # Apply mini U-Net on k
        k = k.transpose(1,2)
        features = []
        for i in range(len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)
        return k.transpose(1,2)




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
        # print("mask:\n{}".format(mask))
        # print("q:\n{}".format(q.size()))
        if mask !=None:
            q = torch.where(mask!=0,mask,q)
        # q = q.cuda()
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

        # cut_num = 3
        q = self.linear_q(q).view(-1, batch_size,len_q, self.n_head, self.d_k)
        k = self.linear_k(k).view(batch_size, len_k, self.n_head, self.d_k)
        v = self.linear_v(v).view(batch_size, len_k, self.n_head, self.d_v)
        mask = mask.view(-1, batch_size,len_q, self.n_head, self.d_k)

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
        # self.unet = Unet_1d(512,64)
        self.attn = Mask_MultiHeadAttention(
                n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_cfg=act_cfg)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q,k,v, mask=None,ifmask=False):
        residual = q

        x = residual + self.attn(q, k, k, mask,ifmask)
        # x = residual + self.attn(q, k, v, mask)
        x = self.norm1(x)

        residual = x
        x = residual + self.mlp(x)
        x = self.norm2(x)

        return x

class TransDecoderLayer(BaseModule):
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
        # self.unet = Unet_1d(512,64)
        self.MHSA = Mask_MultiHeadAttention(
                n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        self.MSA = Mask_MultiHeadAttention(
            n_head, d_model, d_k, d_v, qkv_bias=qkv_bias, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, act_cfg=act_cfg)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, q,k,v, mask=None,ifmask=False):
        residual = q

        x = residual + self.MHSA(q, q, q, mask, ifmask)
        # x = residual + self.attn(q, k, v, mask)
        q = self.norm1(x)

        residual = q
        x = residual + self.MSA(q, k, v, mask,ifmask)
        # x = residual + self.attn(q, k, v, mask)
        x = self.norm2(x)

        residual = x
        x = residual + self.mlp(x)
        x = self.norm3(x)

        return x

@DECODERS.register_module()
class MaskLanSemv2(BaseDecoder):
    '''
    1: use feat for k,v (64), enc_out for q.
    2: training 10 step for baseline, than chosse if mask low_score is necessary
    tips: we don't need start_id, the end_id can be the same as the padding_idx
    like
    padding_target:[2,34,6,7,8,36,36,36,36]
    not like [37,2,34,6,7,8,37,36,36,36]
    '''

    def __init__(self,
                 max_seq_len=64,
                 n_layers=3,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=1024,
                 dropout=0.1,
                 num_classes=37,
                 mask_id=36,
                 end_id=37,
                 # ifmask = False,  # change mask-attention button
                 **kwargs):
        super().__init__()
        self.layer_stack = ModuleList([
            TransDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout, **kwargs)
            for _ in range(n_layers)
        ])

        max_seq_len = 30
        self.max_seq_len = max_seq_len
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        # self.pos_encoder2 = PositionalEncoding(d_model, 64)
        self.embedding = nn.Sequential(
            nn.Linear(num_classes, d_model),
            nn.Linear(d_model, d_model),
        )
        self.cls = nn.Linear(d_model, num_classes)
        # self.embedding = nn.Linear(num_classes, d_model)
        self.padding_idx = end_id
        self.end_idx = end_id
        self.topk = 3
        self.num_classes = num_classes
        # self.unet = Unet_1d(512,64)
        # self.ifmask = ifmask

    @staticmethod
    def _get_length(logit,padding_idx,max_seq_len,dim=-1):
        """Greedy decoder to obtain length from logit.

        Returns the first location of padding index or the length of the entire
        tensor otherwise.
        """
        # out as a boolean vector indicating the existence of end token(s)
        out = (logit.argmax(dim=-1) == padding_idx)
        abn = out.any(dim)
        # Get the first index of end token
        out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
        out = out + 1
        out = torch.where(abn, out, out.new_tensor(logit.shape[1]))
        out = out.clamp_(2, max_seq_len)
        return out

    def forward_train_step1(self,query, key_value):
        # count = out_enc.new_zeros((N, 1, E))  # (N, 1, E)
        q = self.pos_encoder(query)

        for enc_layer in self.layer_stack:
            q = enc_layer(q, key_value, key_value, mask=None, ifmask=False)
        # x = self.layer_stack(q,k,v,mask)
        # mask_id = self._flatten(q, targets_dict)
        x = self.cls(q)
        outputs = dict(
                dec_class =x, mask_id=None, dec_out = q)
        return outputs

    def forward_train(self, feat, out_enc, targets_dict, img_metas, sem_lan_type='Train'):
        # feat : (B,64,512)
        # out_enc: (B,30,37) using CTC decoder gather
        # # q = out_enc.data
        # q = self.embedding(out_enc)
        # # count = out_enc.new_zeros((N, 1, E))  # (N, 1, E)
        # q = self.pos_encoder(q)
        # feat = self.pos_encoder2(feat)
        N, H_W, E = out_enc.size()
        out_list = []

        # k, v = out_enc, out_enc  # (N, E, H, W)
        # print(k.shape)
        zeros = out_enc.new_zeros((N, self.max_seq_len, E))  # (N, T, E)
        q = self.embedding(zeros)
        for i in range(1):
            output = self.forward_train_step1(query = q,key_value=feat)
            out_list.append(output)
            q = output['dec_out']
        return out_list

    def forward_test(self, feat, out_enc, img_metas):
        return self.forward_train(feat=feat, out_enc=out_enc, img_metas=img_metas, targets_dict=None)


@DECODERS.register_module()
class MaskLanSem(BaseDecoder):
    '''
    1: use feat for k,v (64), enc_out for q.
    2: training 10 step for baseline, than chosse if mask low_score is necessary
    tips: we don't need start_id, the end_id can be the same as the padding_idx
    like
    padding_target:[2,34,6,7,8,36,36,36,36]
    not like [37,2,34,6,7,8,37,36,36,36]
    '''

    def __init__(self,
                 max_seq_len=64,
                 n_layers=3,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=1024,
                 dropout=0.1,
                 num_classes=37,
                 mask_id=36,
                 end_id=37,
                 # ifmask = False,  # change mask-attention button
                 **kwargs):
        super().__init__()
        self.layer_stack = ModuleList([
            TFCommonDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout, **kwargs)
            for _ in range(n_layers)
        ])
        # self.sem_lan_stack = ModuleList([
        #     TFCommonDecoderLayer(
        #         d_model, d_inner, n_head, d_k, d_v, dropout=dropout, **kwargs)
        #     for _ in range(n_layers+2)
        # ])
        max_seq_len = 30
        self.max_seq_len = max_seq_len
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.cls2 = ModuleList([
            nn.Linear(d_model, num_classes)
            for _ in range(n_layers+2)
        ])
        self.cls = nn.Linear(d_model, num_classes)
        # self.enhance = nn.Linear(512, 512)
        self.embedding = nn.Linear(d_model, d_model)
        self.trg_word_emb = nn.Embedding(
            num_classes, d_model, padding_idx=end_id)
        # self.fuser = ABIFuser(max_seq_len=max_seq_len, num_chars=num_classes)
        self.cls_mask = nn.Linear(d_model, num_classes)
        self.cls_feat = nn.Linear(num_classes,d_model)
        self.padding_idx = end_id
        self.end_idx = end_id
        self.topk = 3
        self.num_classes = num_classes
        # self.unet = Unet_1d(512,64)
        # self.ifmask = ifmask

    @staticmethod
    def process_feat(query, mask_ids):
        """"
        return query_feat
        """
        N,T,E = query.size()
        # new_feat = torch.zeros([self.topk,N,T,E])
        mask_feat = []
        # mask_feat.append(query)
        for i in range(N):
            single_feat = torch.index_select(query[i], dim=0, index=mask_ids[i])
            mask_feat.append(single_feat)
        q = torch.stack(mask_feat)
        q = q.transpose(0, 1).contiguous().unsqueeze(2).repeat([1, 1, T, 1])
        query_feat = torch.cat([query.unsqueeze(0),q])

        return query_feat

    def forward_train_step_v2(self,out_enc,out_list):
        N, H_W, E = out_enc.size()

        k, v = out_enc, out_enc  # (N, E, H, W)
        # print(k.shape)
        zeros = out_enc.new_zeros((N, self.max_seq_len, E))  # (N, T, E)
        q = self.embedding(zeros)
        # count = out_enc.new_zeros((N, 1, E))  # (N, 1, E)
        q = self.pos_encoder(q)


        for i,enc_layer in enumerate(self.layer_stack):
            q = enc_layer(q, k, v, mask=None, ifmask=False)
            x = self.cls2[i](q)
            output = dict(
                dec_class=x, mask_id=None, dec_out=q)
            out_list.append(output)
        # x = self.layer_stack(q,k,v,mask)
        # mask_id = self._flatten(q, targets_dict)
        return out_list

    def process_feat2(self, query_sem, query_cls, mask_ids, tk=3, type='sem'):
        """"
        1) sem: feat_origin
        2) lan: topk=1
        3) lan_soft: topk=n
        4) mask: topk combine sem
        query (B,T,C)
        query_cls (B,T,37)
        mask_ids (B,Num,)
        topk is different from mask_ids topk, for results
        return query_feat
        """
        if type == 'sem':
            return query_sem
        elif type == 'lan':
            tk = 1
        elif type == 'gt':
            query_sem = self.trg_word_emb(query_sem)
        # query_lan : find higher topk char for query: B T Char
        # topkv, topki = [s[:length[i]].topk(tk,dim=-1) for i, s in enumerate((query_cls))]
        topkv, topki = query_cls.topk(tk, dim=-1)
        new_query_cls = torch.zeros_like(query_cls)
        new_query_cls = new_query_cls.scatter(-1, topki, topkv)
        query_lan = self.cls_feat(new_query_cls)
        # query_lan : B T Char -> B T C

        if type == 'lan' or type == 'lan_soft':
            return query_lan

        # combine query_lan & query_sem with mask_ids
        # index_select(mask_ids) -> (B Num E)
        # gather <-> scatter
        N, T, E = query_sem.size()
        mask_ids = mask_ids.unsqueeze(-1).repeat([1, 1, E])
        query_sem_index = torch.gather(query_sem, 1, mask_ids)
        query = query_lan.scatter(dim=1, index=mask_ids, src=query_sem_index)

        return query

    @staticmethod
    def _get_length(logit,padding_idx,max_seq_len,dim=-1):
        """Greedy decoder to obtain length from logit.

        Returns the first location of padding index or the length of the entire
        tensor otherwise.
        """
        # out as a boolean vector indicating the existence of end token(s)
        out = (logit.argmax(dim=-1) == padding_idx)
        abn = out.any(dim)
        # Get the first index of end token
        out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
        out = out + 1
        out = torch.where(abn, out, out.new_tensor(logit.shape[1]))
        out = out.clamp_(2, max_seq_len)
        return out

    @staticmethod
    def select_mask_ids(logits, targets_dict,length,topk):
        '''
        logits: (B, T, 37)
        targets_dict: with padding as eos (B,)
        length: predict len  tensor (B,)

        return mask_ids: (B, Topk)
        '''

        logit = F.softmax(logits, dim=-1).max(-1)[0]
        # logit = F.softmax(logits, dim=-1).mean(-1)
        # logits: B, T
        if targets_dict != None:
            target_lens = [len(t) for t in targets_dict['targets']]
        else:
            target_lens = length
        topk_list =[]
        for i, s in enumerate((logit)):
            k = topk if topk < target_lens[i]-1 else target_lens[i]-1
            topk_v, topk_i = torch.topk(s[:target_lens[i]],k = k if k!=0 else 1 , largest=False)
            if k < topk:
                topk_i = topk_i.repeat([topk])[:topk]
            topk_list.append(topk_i)
        mask_logit_ids = torch.stack(topk_list)
        return mask_logit_ids

    def forward_train_step1(self,out_enc):
        N, H_W, E = out_enc.size()

        k, v = out_enc, out_enc  # (N, E, H, W)
        # print(k.shape)
        zeros = out_enc.new_zeros((N, self.max_seq_len, E))  # (N, T, E)
        q = self.embedding(zeros)
        # count = out_enc.new_zeros((N, 1, E))  # (N, 1, E)
        q = self.pos_encoder(q)

        for enc_layer in self.layer_stack:
            q = enc_layer(q, k, v, mask=None, ifmask=False)
        # x = self.layer_stack(q,k,v,mask)
        # mask_id = self._flatten(q, targets_dict)
        x = self.cls(q)
        outputs = dict(
                dec_class =x, mask_id=None, dec_out = q)
        return outputs

    def sem_lan_fusion_ids(self,targets,sem, length,radom_ratio=0.8):
        '''
        targets: N,T

        '''
        targets = targets.to(sem.device)
        _,_,E = sem.size()
        # targets_one_hot = F.one_hot(targets, self.num_classes)
        tgts = self.trg_word_emb(targets)
        if radom_ratio==0.:
            return sem
        fusion_ids = torch.cat([torch.randint(low=0,high = l+1,size=(1,int(20*radom_ratio))) for i,l in enumerate(length)])
        fusion_ids = fusion_ids.to(sem.device).repeat([1,self.max_seq_len])[:,:30]
        # print(fusion_ids)
        sem = sem.scatter(dim=1,index=fusion_ids.clamp_(0, 26).unsqueeze(-1).repeat([1,1,E]),src=tgts)
        return sem

    def forward_train_step2(self, out_dec, out_enc, targets_dict,sem_lan_type='sem'):
        # q = out_dec['q']
        q = out_dec['dec_out']
        q = self.pos_encoder(q)

        length = self._get_length(out_dec['dec_class'],self.padding_idx,self.max_seq_len)
        mask_ids = self.select_mask_ids(out_dec['dec_class'], targets_dict, length,self.topk)
        query_lan = self.process_feat2(q, out_dec['dec_class'], mask_ids, tk=3, type='lan')
        # mask_id = self.mask_replace(x)

        if sem_lan_type!=None:
            # ratio = 0.1 if self.epoch<=3 else 1.2-(self.epoch*0.1)
            ratio = 0.5
            target_lens = [len(t) for t in targets_dict['targets']]
            query_lan = self.sem_lan_fusion_ids(targets_dict['padded_targets'],query_lan,target_lens,radom_ratio=ratio)
            # query = self.sem_lan_random(sem = q, lan=targets_dict['padded_targets'],fusion_ids=fusion_ids)
            # query_lan = self.process_feat2(targets_dict['padded_targets'].to(q.device), out_dec['dec_class'], mask_ids, tk=3, type='gt')
        else:
            query_lan=query_lan
            # query_lan = self.process_feat2(q, out_dec['dec_class'], mask_ids, tk=3, type='lan')
        query_lan = self.pos_encoder(query_lan)
        for mask_enc_layer in self.sem_lan_stack:
            query = mask_enc_layer(query_lan, q, q, None, ifmask=False)
        # q = q.view(-1, N, H_W, E)

        # two
        # out_origin = q[0]
        # out = q[1]
        # out = out.scatter(1, mask_ids.unsqueeze(-1).repeat(1, 1, E), q[0].mean(-2).unsqueeze(-2))
        # out = torch.stack([out_origin,out],dim=0)
        # dec_out = self.fuser(out, out_enc)['logits']
        dec_class = self.cls_mask(query)
        # single_out = self.cls_mask(q[0].mean(-2))

        outputs = dict(
            dec_class=dec_class, dec_out=query, mask_id=None, single_out = None)
        return outputs

    def forward_train(self, feature, logits, targets_dict, img_metas, sem_lan_type='Train'):
        # change type for sem

        # Position Attention
        # feat (k,v) n, c, h, w
        # out_enc (q)
        # n,c, h, w = feat.size()
        # feat = feat.view(n, h*w, c)
        # q = self.pos_encoder(out_enc)
        # k, v = feat, feat
        # out_enc = self.unet(out_enc.transpose(1,2)).transpose(1,2).contigous()
        out_list = []
        outputs= self.forward_train_step1(feature)
        out_list.append(outputs)
        # out_list= self.forward_train_step_v2(outputs['dec_out'],out_list)
        # out_list.append(outputs)
        # outputs = dict(
        #     out_enc=x, out_dec=None)
        # x  (B T 37)
        # ifmask = False
        # if sem_lan_type != None:
        if targets_dict == None or self.epoch >= 4:
            # E = self.num_classes
            for i in range(1):
                outputs = self.forward_train_step2(outputs, feature, targets_dict,sem_lan_type)
                out_list.append(outputs)
        return out_list

    def forward_test(self, feat, out_enc, img_metas):
        return self.forward_train(feature=feat, logits=out_enc, img_metas=img_metas, targets_dict=None,sem_lan_type=None)

@DECODERS.register_module()
# class MaskLanSem(BaseDecoder):
class MaskCommonDecoderv6(BaseDecoder):
    '''
    1: use feat for k,v (64), enc_out for q.
    2: training 10 step for baseline, than chosse if mask low_score is necessary
    '''

    def __init__(self,
                 max_seq_len=64,
                 n_layers=3,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=1024,
                 dropout=0.1,
                 num_classes=37,
                 mask_id=36,
                 end_id=37,
                 # ifmask = False,  # change mask-attention button
                 **kwargs):
        super().__init__()
        self.layer_stack = ModuleList([
            TFCommonDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout, **kwargs)
            for _ in range(n_layers)
        ])
        max_seq_len = 30
        self.max_seq_len = max_seq_len
        self.pos_encoder = PositionalEncoding(512, max_seq_len)
        self.cls = nn.Linear(512, num_classes)
        self.cls_feat = nn.Linear(num_classes, 512)
        # self.enhance = nn.Linear(512, 512)
        self.embedding = nn.Linear(512, 512)
        # self.fuser = ABIFuser(max_seq_len=max_seq_len, num_chars=num_classes)
        self.cls_mask = nn.Linear(512, num_classes)
        self.padding_idx = mask_id
        self.end_idx = end_id
        self.topk = 1
        self.num_classes = num_classes
        # self.unet = Unet_1d(512,64)
        # self.ifmask = ifmask

    def _get_length(self, logit, dim=-1):
        """Greedy decoder to obtain length from logit.

        Returns the first location of padding index or the length of the entire
        tensor otherwise.
        """
        # out as a boolean vector indicating the existence of end token(s)
        out = (logit.argmax(dim=-1) == self.padding_idx-1)
        abn = out.any(dim)
        # Get the first index of end token
        out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
        out = out + 1
        out = torch.where(abn, out, out.new_tensor(logit.shape[1]))
        out = out.clamp_(2, self.max_seq_len)
        return out

    # def select_mask_ids(self, logits, targets_dict,length):
    #     '''
    #     logits: (B, T, 37)
    #     targets_dict: with padding as eos (B,)
    #     length: predict len  tensor (B,)
    #
    #     return mask_ids: (B, Topk)
    #     '''
    #
    #     logit = F.softmax(logits, dim=-1).max(-1)[0]
    #     # logits: B, T
    #     if targets_dict != None:
    #         target_lens = [len(t) for t in targets_dict['targets']]
    #     else:
    #         target_lens = length
    #     topk_list =[]
    #     for i, s in enumerate((logit)):
    #         k = self.topk if self.topk < (target_lens[i]-self.topk) else target_lens[i]//2+1
    #         topk_v, topk_i = torch.topk(s[:target_lens[i]],k = k, largest=False)
    #         topk_list.append(topk_i)
    #     mask_logit_ids = torch.stack(topk_list)
    #     return mask_logit_ids

    def _flatten(self, logits, targets_dict):
        '''
        logits: B, T, 37
        '''

        logit = F.softmax(logits, dim=-1).mean(-1)
        # logits: B, T
        if targets_dict != None:
            target_lens = [len(t) for t in targets_dict['targets']]
            mask_logit_id = torch.stack(
                [s[:target_lens[i]].min(-1).indices for i, s in enumerate((logit))])
        else:
            # index = torch.argmax(logits, dim=-1).int()
            # self.end_idx
            batch_size = logits.size(0)
            indexes, scores = [], []
            for idx in range(batch_size):
                seq = logits[idx, :, :]
                max_value, max_idx = torch.max(seq, -1)
                mini_score = 99
                mini_index = 0
                output_index = max_idx.cpu().detach().numpy().tolist()
                output_score = max_value.cpu().detach().numpy().tolist()
                for char_index, char_score in zip(output_index, output_score):
                    if char_index == self.end_idx:
                        break
                    if char_index == self.padding_idx:
                        break
                    elif mini_score > char_score:
                        mini_score = char_score
                        mini_index = char_index
                # print("\n mini_index:{}\n".format(mini_index))
                indexes.append(self.max_seq_len - 1 if mini_index >= self.max_seq_len else mini_index)

            mask_logit_id = torch.tensor(indexes, device=logits.device).long()
        return mask_logit_id

    def process_feat2(self, query_sem, query_cls,mask_ids, tk=2, type='mask'):
        """"
        query (B,T,C)
        query_cls (B,T,37)
        mask_ids (B,Num,)
        topk is different from mask_ids topk, for results
        return query_feat
        """
        if type == 'sem':
            return query_sem
        elif type  == 'lan':
            tk = 1

        # query_lan : find higher topk char for query: B T Char
        # topkv, topki = [s[:length[i]].topk(tk,dim=-1) for i, s in enumerate((query_cls))]
        topkv, topki = query_cls.topk(tk, dim=-1)
        new_query_cls = torch.zeros_like(query_cls)
        new_query_cls = new_query_cls.scatter(-1,topki,topkv)
        query_lan = self.cls_feat(new_query_cls)
        # query_lan : B T Char -> B T C

        if type =='lan' or type == 'lan_soft':
            return query_lan

        # combine query_lan & query_sem with mask_ids
        # index_select(mask_ids) -> (B Num E)
        # gather <-> scatter
        N,T,E = query_sem.size()
        mask_ids = mask_ids.unsqueeze(-1).repeat([1, 1, E])
        query_sem_index = torch.gather(query_sem,1,mask_ids)
        query = query_lan.scatter(dim=1,index=mask_ids,src=query_sem_index)

        return query

    def forward_train_step1(self,out_enc):
        N, H_W, E = out_enc.size()

        k, v = out_enc, out_enc  # (N, E, H, W)
        # print(k.shape)
        zeros = out_enc.new_zeros((N, self.max_seq_len, E))  # (N, T, E)
        q = self.embedding(zeros)
        # count = out_enc.new_zeros((N, 1, E))  # (N, 1, E)
        q = self.pos_encoder(q)

        for enc_layer in self.layer_stack:
            q = enc_layer(q, k, v, mask=None, ifmask=False)
        # x = self.layer_stack(q,k,v,mask)
        # mask_id = self._flatten(q, targets_dict)
        x = self.cls(q)
        outputs = dict(
                dec_class =x, mask_id=None, dec_out = q)
        return outputs

    def forward_train_step2(self, out_dec, out_enc, targets_dict):
        N, H_W, E = out_enc.size()
        H_W = self.max_seq_len
        k, v = out_enc, out_enc
        # q = out_dec['q']
        q = out_dec['dec_out']
        q = self.pos_encoder(q)

        mask_id = self._flatten(out_dec['dec_class'], targets_dict)
        # mask_id = self._flatten(out_dec['dec_class'], targets_dict)
        length = self._get_length(out_dec['dec_class'])
        mask_ids = MaskLanSem_origin.select_mask_ids(out_dec['dec_class'], targets_dict,length,topk=1)
        # mask_ids = self.select_mask_ids(out_dec['dec_class'], targets_dict, length)
        # query = self.process_feat2(q, out_dec['dec_class'], mask_ids, tk=4, type='mask')
        mask_ids = mask_id.unsqueeze(-1)
        # print("we train mask id\n")
        # mask_id = self.mask_replace(x)
        cut_num = 2
        four_dim_q = q.unsqueeze(1).repeat(cut_num, self.topk, 1, 1)
        q = four_dim_q.view(cut_num, N * self.topk, H_W, E)
        # q = self.pos_encoder(q)  # (N*1*2, T, E)
        tmp = torch.gather(q[0].mean(-1), 1, mask_ids)
        # print(tmp.size)
        tmp = tmp.unsqueeze(-1).repeat(1, H_W, E)
        # q[0] = self.enhance(tmp)
        q[0] = tmp
        q = q.view(cut_num * N * self.topk, H_W, E)

        # mask = [target, other_mask, single_mask]
        # mask module
        origin = torch.zeros([N, H_W], device=q.device)
        mask = torch.zeros([N, H_W], device=q.device).scatter_(1, mask_ids,
                                                               float(self.padding_idx))
        # other_mask = torch.where(mask == self.padding_idx, 0, float(self.padding_idx))
        tar_mask = torch.stack([origin, mask], dim=0).unsqueeze(-1)
        # tar_mask = torch.stack([mask, other_mask], dim=0).unsqueeze(-1)
        mask = tar_mask.repeat(1, 1, 1, E)

        for mask_enc_layer in self.layer_stack:
            q = mask_enc_layer(q, k, v, mask, ifmask=True)
        q = q.view(-1, N, H_W, E)

        # two
        # out_origin = q[0]
        out = q[1]
        out = out.scatter(1, mask_ids.unsqueeze(-1).repeat(1, 1, E), q[0].mean(-2).unsqueeze(-2))
        # out = torch.stack([out_origin,out],dim=0)
        # dec_out = self.fuser(out, out_enc)['logits']
        dec_class = self.cls_mask(out)
        single_out = self.cls_mask(q[0].mean(-2))

        outputs = dict(
            dec_class=dec_class, dec_out=out,mask_id=mask_ids.squeeze(-1), single_out = single_out)
        return outputs

    def forward_train(self, feat, out_enc, targets_dict, img_metas, ifmask=False):

        # Position Attention
        # feat (k,v) n, c, h, w
        # out_enc (q)
        # n,c, h, w = feat.size()
        # feat = feat.view(n, h*w, c)
        # q = self.pos_encoder(out_enc)
        # k, v = feat, feat
        # out_enc = self.unet(out_enc.transpose(1,2)).transpose(1,2).contigous()
        out_list = []
        outputs= self.forward_train_step1(out_enc)
        out_list.append(outputs)
        # outputs = dict(
        #     out_enc=x, out_dec=None)
        # x  (B T 37)
        # ifmask = True
        # if ifmask == False:
        if targets_dict == None or self.epoch >= 0:
            # E = self.num_classes
            for i in range(1):
                outputs = self.forward_train_step2(outputs, out_enc, targets_dict)
                out_list.append(outputs)
        return out_list

    def forward_test(self, feat, out_enc, img_metas):
        return self.forward_train(feat=feat, out_enc=out_enc, img_metas=img_metas, targets_dict=None, ifmask=False)

@DECODERS.register_module()
class MaskCommonDecoder(BaseDecoder):
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
                 mask_id = 36 ,
                 end_id = 37,
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
        self.padding_idx = mask_id
        self.end_idx = end_id
        self.topk = 1
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
        '''
        logits: B, T, 37
        '''

        logit, logits_i = F.softmax(logits, dim=-1).max(-1)
        # logits: B, T
        if targets_dict != None:
            target_lens = [len(t) for t in targets_dict['targets']]
            mask_logit_id = torch.stack(
                [s[:target_lens[i]].min(-1).indices for i, s in enumerate((logit))])
        else:
        # index = torch.argmax(logits, dim=-1).int()
        # self.end_idx
            batch_size = logits.size(0)
            indexes, scores = [], []
            for idx in range(batch_size):
                seq = logits[idx, :, :]
                max_value, max_idx = torch.max(seq, -1)
                mini_score = 99
                mini_index = 0
                output_index = max_idx.cpu().detach().numpy().tolist()
                output_score = max_value.cpu().detach().numpy().tolist()
                for char_index, char_score in zip(output_index, output_score):
                    if char_index == self.end_idx:
                        break
                    elif mini_score > char_score:
                        mini_score = char_score
                        mini_index = char_index
                indexes.append(mini_index)
            mask_logit_id = torch.tensor(indexes,device=logits.device).long()
        return mask_logit_id

    def mask_zeros(self,mask_id,target):
        '''
        mask_id: (B,topk)
        target: (B,T,N)
        return
        mask: (2*B*topk,T,N)
        '''
        # N,T,E = target.size()


        mask = target.scatter_(dim=-1,index = mask_id,src = self.padding_idx)
        other_mask = torch.where(mask!=self.padding_idx, self.padding_idx, mask)
        mask = torch.cat([target, mask, other_mask,],0)
        # mask_one_zeros = mask.unsqueeze(-1).repeat(1, 1, 1, E)
        # mask = mask.view(N * 2, T, E)
        return mask

    def mask_replace(self, logits):
        # char index B T
        # index = torch.argmax(logits,dim=-1).int()

        # char score : B T 37 -> B T
        # logits = nn.softmax(logits, dim=-1)
        conf_score = torch.max(logits,dim=-1)
        topk_v, topk_i = torch.topk(conf_score, k = self.topk, largest=False, dim=-1)
        #topk_i  T_position_index
        return topk_i

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
            q = enc_layer(q,k,v,mask, ifmask=False)
        # x = self.layer_stack(q,k,v,mask)
        outputs = self.cls(q)
        #x  (B T 37)

        # mask_id = self._flatten(x, targets_dict)
        # # mask_id = self.mask_replace(x)
        #
        # four_dim_q = q.unsqueeze(1).repeat(2, self.topk, 1, 1)
        # q = four_dim_q.view(N * self.topk * 2, H_W, E)
        # # q = self.pos_encoder(zeros)  # (N, T, E)
        #
        # # mask = [target, other_mask, single_mask]
        # # mask module
        # # origin = torch.zeros([N, H_W], device=q.device)
        # # print("mask_id:{}\n".format(mask_id))
        # mask = torch.zeros([N, H_W], device=q.device).scatter_(1, mask_id.unsqueeze(-1),
        #                                                        float(self.padding_idx)).long()
        # other_mask = torch.where(mask == self.padding_idx, 0, self.padding_idx)
        # # tar_mask = torch.stack([origin, mask, other_mask], dim=0)
        # tar_mask = torch.stack([mask, other_mask], dim=0).unsqueeze(-1)
        # mask = tar_mask.repeat(1,1,1,E)
        # # zeros = torch.zeros([N, H_W, E],device=q.device)
        # # mask = self.mask_zeros(mask_id, zeros)
        # # mask_one_zeros = self._get_location_mask(token_len=H_W, mask_fill=float(self.mask_id), dot_v=1,
        # #                                          device=q.device)
        #
        # # mask_one_zeros = mask_one_zeros.repeat(N, 1, 1).unsqueeze(-1).repeat(1, 1, 1, E)
        # # mask = mask_one_zeros.view(N * H_W * 2, H_W, E)
        # for mask_enc_layer in self.layer_stack_mask:
        #     q = mask_enc_layer(q, k, v, mask, ifmask=True)
        # q = q.view(2, N, H_W, E)
        # # q = q[0].scatter_(1, mask_id.unsqueeze(-1),float(self.padding_idx)).long()
        # # a = tar_mask.squeeze(-1)
        # # for i, b in enumerate(a[0]):
        # #     a[1][i][mask_id[i]] = b[mask_id[i]]
        # out = q[1]
        # for i in range(N):
        #     out[i][mask_id[i]] = q[0][i][mask_id[i]]
        # # q = q[0].scatter_(1, mask_id.unsqueeze(-1).unsqueeze(-1), q[1])
        # # flatten_q = self._flatten(x,targets_dict)
        # x2 = self.cls_mask(out)
        # outputs = dict(
        #     out_enc=x, out_dec=x2)
        return outputs
        # for enc_layer in self.count_layer_stack:
        #     count = enc_layer(count,k,v, None, False)
        # count = self.mlp(count)
        # x = torch.cat((count,x),1)
        # return x

    def forward_test(self, feat,out_enc, img_metas):
        return self.forward_train(feat=feat,out_enc=out_enc,img_metas=img_metas,targets_dict=None,ifmask=False)

@DECODERS.register_module()
class MaskCommonDecoderv2(BaseDecoder):
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
                 mask_id = 36 ,
                 end_id = 37,
                 # ifmask = False,  # change mask-attention button
                 **kwargs):
        super().__init__()
        self.layer_stack = ModuleList([
            TFCommonDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout,**kwargs)
            for _ in range(n_layers)
        ])
        # self.layer_stack_mask = ModuleList([
        #     TFCommonDecoderLayer(
        #         d_model, d_inner, n_head, d_k, d_v, dropout=dropout, **kwargs)
        #     for _ in range(n_layers)
        # ])
        self.max_seq_len = max_seq_len

        self.pos_encoder = PositionalEncoding(512, max_seq_len)
        self.cls = nn.Linear(512, num_classes)
        self.fuser = ABIFuser(max_seq_len = max_seq_len, num_chars = num_classes)
        self.cls_mask= nn.Linear(512, num_classes)
        self.padding_idx = mask_id
        self.end_idx = end_id
        self.topk = 1
        # self.ifmask = ifmask

    def _flatten(self, logits, targets_dict):
        '''
        logits: B, T, 37
        '''

        logit= F.softmax(logits, dim=-1).mean(-1)
        # logits: B, T
        if targets_dict != None:
            target_lens = [len(t) for t in targets_dict['targets']]
            mask_logit_id = torch.stack(
                [s[:target_lens[i]].min(-1).indices for i, s in enumerate((logit))])
        else:
        # index = torch.argmax(logits, dim=-1).int()
        # self.end_idx
            batch_size = logits.size(0)
            indexes, scores = [], []
            for idx in range(batch_size):
                seq = logits[idx, :, :]
                max_value, max_idx = torch.max(seq, -1)
                mini_score = 99
                mini_index = 0
                output_index = max_idx.cpu().detach().numpy().tolist()
                output_score = max_value.cpu().detach().numpy().tolist()
                for char_index, char_score in zip(output_index, output_score):
                    if char_index == self.end_idx:
                        break
                    if char_index == self.padding_idx:
                        break
                    elif mini_score > char_score:
                        mini_score = char_score
                        mini_index = char_index
                # print("\n mini_index:{}\n".format(mini_index))
                indexes.append(self.max_seq_len-1 if mini_index>=self.max_seq_len else mini_index)

            mask_logit_id = torch.tensor(indexes,device=logits.device).long()
        return mask_logit_id


    def forward_train(self, feat,out_enc, targets_dict, img_metas,ifmask=False):

        # Position Attention
        # print(out_enc.shape)
        N, H_W, E = out_enc.size()
        H_W = self.max_seq_len
        k, v = out_enc, out_enc  # (N, E, H, W)
        # print(k.shape)
        zeros = out_enc.new_zeros((N, self.max_seq_len, E))  # (N, T, E)
        # count = out_enc.new_zeros((N, 1, E))  # (N, 1, E)
        q = self.pos_encoder(zeros)

        for enc_layer in self.layer_stack:
            q = enc_layer(q,k,v,mask=None, ifmask=False)
        # x = self.layer_stack(q,k,v,mask)
        x = self.cls(q)
        outputs = x
        # outputs = dict(
        #     out_enc=x, out_dec=None)
        #x  (B T 37)
        # if ifmask == True:
        if targets_dict == None or self.epoch >= 4:
            mask_id = self._flatten(x, targets_dict)
            # mask_id = self.mask_replace(x)
            cut_num = 3
            four_dim_q = q.unsqueeze(1).repeat(cut_num, self.topk, 1, 1)
            q = four_dim_q.view(cut_num, N * self.topk , H_W, E)
            # q = self.pos_encoder(q)  # (N*1*2, T, E)
            tmp = torch.gather(q[1].mean(-1),1,mask_id.unsqueeze(-1))
            # print(tmp.size)
            q[1] = tmp.unsqueeze(-1).repeat(1,H_W,E)
            q = q.view(cut_num * N * self.topk , H_W, E)

            # mask = [target, other_mask, single_mask]
            # mask module
            origin = torch.zeros([N, H_W], device=q.device)
            mask = torch.zeros([N, H_W], device=q.device).scatter_(1, mask_id.unsqueeze(-1),
                                                                   float(self.padding_idx))
            # other_mask = torch.where(mask == self.padding_idx, 0, float(self.padding_idx))
            # other_mask = torch.where(mask == 0., float(self.padding_idx), 0.)
            # tar_mask = torch.stack([origin, mask, other_mask], dim=0).unsqueeze(-1)
            # print("mask_id:{}\n".format(mask_id))
            # print("mask:{}\n".format(mask))
            tar_mask = torch.stack([origin, origin, mask], dim=0).unsqueeze(-1)
            # tar_mask = torch.stack([mask, other_mask], dim=0).unsqueeze(-1)
            mask = tar_mask.repeat(1,1,1,E)

            for mask_enc_layer in self.layer_stack:
                q = mask_enc_layer(q, k, v, mask, ifmask=True)
            q = q.view(-1, N, H_W, E)
            # q = q[0].scatter_(1, mask_id.unsqueeze(-1),float(self.padding_idx)).long()
            # a = tar_mask.squeeze(-1)
            # for i, b in enumerate(a[0]):
            #     a[1][i][mask_id[i]] = b[mask_id[i]]

            # double
            # out = q[1]
            # for i in range(N):
            #     out[i][mask_id[i]] = q[0][i][mask_id[i]]

            #three
            out_origin = q[0]
            out = q[2]
            # single_char = q[1].mean(-2)
            # for i in range(N):
            #     out[i][mask_id[i]] = q[1][i][mask_id[i]]
            out = out.scatter(1, mask_id.unsqueeze(-1).unsqueeze(-1).repeat(1,1,E), q[1].mean(-2).unsqueeze(-2))
            # out = torch.stack([out_origin,out],dim=0)
            outputs = self.fuser(out_origin,out)['logits']
            # q = q[0].scatter_(1, mask_id.unsqueeze(-1).unsqueeze(-1), q[1])
            # flatten_q = self._flatten(x,targets_dict)
            # x2 = self.cls_mask(out)
            # # outputs = dict(
            # #     out_enc=x, out_dec=x2)
            # outputs = dict(
            #     out_dec2=x2[0], out_dec=x2[1])
        return outputs


    def forward_test(self, feat,out_enc, img_metas):
        return self.forward_train(feat=feat,out_enc=out_enc,img_metas=img_metas,targets_dict=None,ifmask=False)


@DECODERS.register_module()
class MaskCommonDecoderv4(BaseDecoder):
    '''
    1: use feat for k,v (64), enc_out for q.
    2: training 10 step for baseline, than chosse if mask low_score is necessary
    '''
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
                 mask_id = 36 ,
                 end_id = 37,
                 # ifmask = False,  # change mask-attention button
                 **kwargs):
        super().__init__()
        self.layer_stack = ModuleList([
            TFCommonDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout,**kwargs)
            for _ in range(n_layers)
        ])
        max_seq_len = 64
        self.max_seq_len = max_seq_len
        self.pos_encoder = PositionalEncoding(512, max_seq_len)
        self.cls = nn.Linear(512, num_classes)
        self.fuser = ABIFuser(max_seq_len = max_seq_len, num_chars = num_classes)
        self.cls_mask= nn.Linear(512, num_classes)
        self.padding_idx = mask_id
        self.end_idx = end_id
        self.topk = 1
        self.num_classes = num_classes
        # self.ifmask = ifmask

    def _flatten(self, logits, targets_dict):
        '''
        logits: B, T, 37
        '''

        logit= F.softmax(logits, dim=-1).mean(-1)
        # logits: B, T
        if targets_dict != None:
            target_lens = [len(t) for t in targets_dict['targets']]
            mask_logit_id = torch.stack(
                [s[:target_lens[i]].min(-1).indices for i, s in enumerate((logit))])
        else:
        # index = torch.argmax(logits, dim=-1).int()
        # self.end_idx
            batch_size = logits.size(0)
            indexes, scores = [], []
            for idx in range(batch_size):
                seq = logits[idx, :, :]
                max_value, max_idx = torch.max(seq, -1)
                mini_score = 99
                mini_index = 0
                output_index = max_idx.cpu().detach().numpy().tolist()
                output_score = max_value.cpu().detach().numpy().tolist()
                for char_index, char_score in zip(output_index, output_score):
                    if char_index == self.end_idx:
                        break
                    if char_index == self.padding_idx:
                        break
                    elif mini_score > char_score:
                        mini_score = char_score
                        mini_index = char_index
                # print("\n mini_index:{}\n".format(mini_index))
                indexes.append(self.max_seq_len-1 if mini_index>=self.max_seq_len else mini_index)

            mask_logit_id = torch.tensor(indexes,device=logits.device).long()
        return mask_logit_id


    def forward_train(self, feat,out_enc, targets_dict, img_metas,ifmask=False):

        # Position Attention
        # feat (k,v) n, c, h, w
        # out_enc (q)
        # n,c, h, w = feat.size()
        # feat = feat.view(n, h*w, c)
        # q = self.pos_encoder(out_enc)
        # k, v = feat, feat

        N, H_W, E = out_enc.size()
        H_W = self.max_seq_len
        k, v = out_enc, out_enc  # (N, E, H, W)
        # print(k.shape)
        zeros = out_enc.new_zeros((N, self.max_seq_len, E))  # (N, T, E)
        # count = out_enc.new_zeros((N, 1, E))  # (N, 1, E)
        q = self.pos_encoder(zeros)

        for enc_layer in self.layer_stack:
            q = enc_layer(q,k,v,mask=None, ifmask=False)
        # x = self.layer_stack(q,k,v,mask)
        # mask_id = self._flatten(q, targets_dict)
        x = self.cls(q)
        outputs = x
        # outputs = dict(
        #     out_enc=x, out_dec=None)
        #x  (B T 37)
        # if ifmask == False:

        if targets_dict == None or self.epoch >= 5:
            # E = self.num_classes
            mask_id = self._flatten(x, targets_dict)
            # print("we train mask id\n")
            # mask_id = self.mask_replace(x)
            cut_num = 3
            four_dim_q = q.unsqueeze(1).repeat(cut_num, self.topk, 1, 1)
            q = four_dim_q.view(cut_num, N * self.topk , H_W, E)
            # q = self.pos_encoder(q)  # (N*1*2, T, E)
            tmp = torch.gather(q[1].mean(-1),1,mask_id.unsqueeze(-1))
            # print(tmp.size)
            q[1] = tmp.unsqueeze(-1).repeat(1,H_W,E)
            q = q.view(cut_num * N * self.topk , H_W, E)

            # mask = [target, other_mask, single_mask]
            # mask module
            origin = torch.zeros([N, H_W], device=q.device)
            mask = torch.zeros([N, H_W], device=q.device).scatter_(1, mask_id.unsqueeze(-1),
                                                                   float(self.padding_idx))
            # other_mask = torch.where(mask == self.padding_idx, 0, float(self.padding_idx))
            tar_mask = torch.stack([origin, origin, mask], dim=0).unsqueeze(-1)
            # tar_mask = torch.stack([mask, other_mask], dim=0).unsqueeze(-1)
            mask = tar_mask.repeat(1,1,1,E)

            for mask_enc_layer in self.layer_stack:
                q = mask_enc_layer(q, k, v, mask, ifmask=True)
            q = q.view(-1, N, H_W, E)

            #three
            out_origin = q[0]
            out = q[2]
            out = out.scatter(1, mask_id.unsqueeze(-1).unsqueeze(-1).repeat(1,1,E), q[1].mean(-2).unsqueeze(-2))
            # out = torch.stack([out_origin,out],dim=0)
            outputs = self.fuser(out_origin,out)['logits']
        return outputs


    def forward_test(self, feat,out_enc, img_metas):
        return self.forward_train(feat=feat,out_enc=out_enc,img_metas=img_metas,targets_dict=None,ifmask=False)


@DECODERS.register_module()
class MaskCommonDecoderv5(BaseDecoder):
    '''
    1: use feat for k,v (64), enc_out for q.
    2: training 10 step for baseline, than chosse if mask low_score is necessary
    '''
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
                 mask_id = 36 ,
                 end_id = 37,
                 # ifmask = False,  # change mask-attention button
                 **kwargs):
        super().__init__()
        self.layer_stack = ModuleList([
            TFCommonDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout,**kwargs)
            for _ in range(n_layers)
        ])
        max_seq_len = 64
        self.max_seq_len = max_seq_len
        self.pos_encoder = PositionalEncoding(512, max_seq_len)
        self.cls = nn.Linear(512, num_classes)
        self.fuser = ABIFuser(max_seq_len = max_seq_len, num_chars = num_classes)
        self.cls_mask= nn.Linear(512, num_classes)
        self.padding_idx = mask_id
        self.end_idx = end_id
        self.topk = 1
        self.num_classes = num_classes
        # self.ifmask = ifmask

    def _flatten(self, logits, targets_dict):
        '''
        logits: B, T, 37
        '''

        logit= F.softmax(logits, dim=-1).mean(-1)
        # logits: B, T
        if targets_dict != None:
            target_lens = [len(t) for t in targets_dict['targets']]
            mask_logit_id = torch.stack(
                [s[:target_lens[i]].min(-1).indices for i, s in enumerate((logit))])
        else:
        # index = torch.argmax(logits, dim=-1).int()
        # self.end_idx
            batch_size = logits.size(0)
            indexes, scores = [], []
            for idx in range(batch_size):
                seq = logits[idx, :, :]
                max_value, max_idx = torch.max(seq, -1)
                mini_score = 99
                mini_index = 0
                output_index = max_idx.cpu().detach().numpy().tolist()
                output_score = max_value.cpu().detach().numpy().tolist()
                for char_index, char_score in zip(output_index, output_score):
                    if char_index == self.end_idx:
                        break
                    if char_index == self.padding_idx:
                        break
                    elif mini_score > char_score:
                        mini_score = char_score
                        mini_index = char_index
                # print("\n mini_index:{}\n".format(mini_index))
                indexes.append(self.max_seq_len-1 if mini_index>=self.max_seq_len else mini_index)

            mask_logit_id = torch.tensor(indexes,device=logits.device).long()
        return mask_logit_id


    def forward_train(self, feat,out_enc, targets_dict, img_metas,ifmask=False):

        # Position Attention
        # feat (k,v) n, c, h, w
        # out_enc (q)
        # n,c, h, w = feat.size()
        # feat = feat.view(n, h*w, c)
        # q = self.pos_encoder(out_enc)
        # k, v = feat, feat

        N, H_W, E = out_enc.size()
        H_W = self.max_seq_len
        k, v = out_enc, out_enc  # (N, E, H, W)
        # print(k.shape)
        zeros = out_enc.new_zeros((N, self.max_seq_len, E))  # (N, T, E)
        # count = out_enc.new_zeros((N, 1, E))  # (N, 1, E)
        q = self.pos_encoder(zeros)

        for enc_layer in self.layer_stack:
            q = enc_layer(q,k,v,mask=None, ifmask=False)
        # x = self.layer_stack(q,k,v,mask)
        # mask_id = self._flatten(q, targets_dict)
        x = self.cls(q)
        outputs = x
        # outputs = dict(
        #     out_enc=x, out_dec=None)
        #x  (B T 37)
        # if ifmask == False:

        if targets_dict == None or self.epoch >= 5:
            # E = self.num_classes
            mask_id = self._flatten(x, targets_dict)
            # print("we train mask id\n")
            # mask_id = self.mask_replace(x)
            cut_num = 2
            four_dim_q = q.unsqueeze(1).repeat(cut_num, self.topk, 1, 1)
            q = four_dim_q.view(cut_num, N * self.topk , H_W, E)
            # q = self.pos_encoder(q)  # (N*1*2, T, E)
            tmp = torch.gather(q[0].mean(-1),1,mask_id.unsqueeze(-1))
            # print(tmp.size)
            q[0] = tmp.unsqueeze(-1).repeat(1,H_W,E)
            q = q.view(cut_num * N * self.topk , H_W, E)

            # mask = [target, other_mask, single_mask]
            # mask module
            origin = torch.zeros([N, H_W], device=q.device)
            mask = torch.zeros([N, H_W], device=q.device).scatter_(1, mask_id.unsqueeze(-1),
                                                                   float(self.padding_idx))
            # other_mask = torch.where(mask == self.padding_idx, 0, float(self.padding_idx))
            tar_mask = torch.stack([origin, mask], dim=0).unsqueeze(-1)
            # tar_mask = torch.stack([mask, other_mask], dim=0).unsqueeze(-1)
            mask = tar_mask.repeat(1,1,1,E)

            for mask_enc_layer in self.layer_stack:
                q = mask_enc_layer(q, k, v, mask, ifmask=True)
            q = q.view(-1, N, H_W, E)

            #three
            # out_origin = q[0]
            out = q[1]
            out = out.scatter(1, mask_id.unsqueeze(-1).unsqueeze(-1).repeat(1,1,E), q[0].mean(-2).unsqueeze(-2))
            # out = torch.stack([out_origin,out],dim=0)
            outputs = self.fuser(out,out_enc)['logits']
        return outputs


    def forward_test(self, feat,out_enc, img_metas):
        return self.forward_train(feat=feat,out_enc=out_enc,img_metas=img_metas,targets_dict=None,ifmask=False)