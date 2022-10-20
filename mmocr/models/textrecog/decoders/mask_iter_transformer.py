import torch.nn as nn
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule, ModuleList
from .base_decoder import BaseDecoder

from mmocr.models.builder import DECODERS
from mmocr.models.common.modules import (PositionwiseFeedForward,
                                         PositionalEncoding,CBAM,
                                         MultiHeadAttention)
from mmocr.models.textrecog.decoders.transformer_mask import TFCommonDecoderLayer

@DECODERS.register_module()
class MaskDecoderv10_iter(BaseDecoder):
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
        out = (logit.argmax(dim=-1) == self.pad_idx)
        abn = out.any(dim)
        # Get the first index of end token
        out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
        out = out + 1
        out = torch.where(abn, out, out.new_tensor(logit.shape[1]))
        return out

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

        lengths = self._get_length(logits)
        lengths.clamp_(2, self.max_seq_len)
        tokens = torch.softmax(logits, dim=-1)
        if self.detach_tokens:
            tokens = tokens.detach()
        embed = self.proj(tokens)  # (N, T, E)
        embed = self.token_encoder(embed)  # (N, T, E)


        # mask_id = self._flatten(out_dec['dec_out'], targets_dict)
        mask_id = self._flatten(out_dec['dec_class'], targets_dict)
        cut_num = 2
        four_dim_q = q.unsqueeze(1).repeat(cut_num, self.topk, 1, 1)
        q = four_dim_q.view(cut_num, N * self.topk, H_W, E)
        # q = self.pos_encoder(q)  # (N*1*2, T, E)
        tmp = torch.gather(q[0].mean(-1), 1, mask_id.unsqueeze(-1))
        # print(tmp.size)
        tmp = tmp.unsqueeze(-1).repeat(1, H_W, E)
        # q[0] = self.enhance(tmp)
        q[0] = tmp
        q = q.view(cut_num * N * self.topk, H_W, E)

        # mask = [target, other_mask, single_mask]
        # mask module
        origin = torch.zeros([N, H_W], device=q.device)
        mask = torch.zeros([N, H_W], device=q.device).scatter_(1, mask_id.unsqueeze(-1),
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
        out = out.scatter(1, mask_id.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, E), q[0].mean(-2).unsqueeze(-2))
        # out = torch.stack([out_origin,out],dim=0)
        # dec_out = self.fuser(out, out_enc)['logits']
        dec_class = self.cls_mask(out)
        single_out = self.cls_mask(q[0].mean(-2))

        outputs = dict(
            dec_class=dec_class, dec_out=out,mask_id=mask_id, single_out = single_out)
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
        # ifmask = False
        # if ifmask == False:
        if targets_dict == None or self.epoch >= 5:
            # E = self.num_classes
            for i in range(1):
                outputs = self.forward_train_step2(outputs, out_enc, targets_dict)
                out_list.append(outputs)
        return out_list

    def forward_test(self, feat, out_enc, img_metas):
        return self.forward_train(feat=feat, out_enc=out_enc, img_metas=img_metas, targets_dict=None, ifmask=False)
