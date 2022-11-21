# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.builder import LOSSES
from .ctc_loss import CTCLoss
# from .ce_loss import CELoss

@LOSSES.register_module()
class CELoss2(nn.Module):
    """Implementation of loss module for encoder-decoder based text recognition
    method with CrossEntropy loss.

    Args:
        ignore_index (int): Specifies a target value that is
            ignored and does not contribute to the input gradient.
        reduction (str): Specifies the reduction to apply to the output,
            should be one of the following: ('none', 'mean', 'sum').
        ignore_first_char (bool): Whether to ignore the first token in target (
            usually the start token). If ``True``, the last token of the output
            sequence will also be removed to be aligned with the target length.
    """

    def __init__(self,
                 ignore_index=-1,
                 reduction='none',
                 ignore_first_char=False):
        super().__init__()
        assert isinstance(ignore_index, int)
        assert isinstance(reduction, str)
        assert reduction in ['none', 'mean', 'sum']
        assert isinstance(ignore_first_char, bool)

        self.loss_ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index, reduction=reduction)
        self.ignore_first_char = ignore_first_char

    def format(self, outputs, targets_dict):
        # targets = targets_dict
        targets = targets_dict['padded_targets']
        if self.ignore_first_char:
            targets = targets[:, 1:].contiguous()
            outputs = outputs[:, :-1, :]

        outputs = outputs.permute(0, 2, 1).contiguous()

        return outputs, targets

    def forward(self, outputs, targets_dict, img_metas=None):
        """
        Args:
            outputs (Tensor): A raw logit tensor of shape :math:`(N, T, C)`.
            targets_dict (dict): A dict with a key ``padded_targets``, which is
                a tensor of shape :math:`(N, T)`. Each element is the index of
                a character.
            img_metas (None): Unused.

        Returns:
            dict: A loss dict with the key ``loss_ce``.
        """
        outputs, targets = self.format(outputs, targets_dict)

        loss_ce = self.loss_ce(outputs, targets.to(outputs.device))
        # losses = dict(loss_ce=loss_ce)

        return loss_ce

@LOSSES.register_module()
class MixLoss(nn.Module):
    """Implementation of ABINet multiloss that allows mixing different types of
    losses with weights.

    Args:
        enc_weight (float): The weight of encoder loss. Defaults to 1.0.
        dec_weight (float): The weight of decoder loss. Defaults to 1.0.
        fusion_weight (float): The weight of fuser (aligner) loss.
            Defaults to 1.0.
        num_classes (int): Number of unique output language tokens.

    Returns:
        A dictionary whose key/value pairs are the losses of three modules.
    """

    def __init__(self,
                 enc_weight=1.0,
                 dec_weight=1.0,
                 num_classes=38,
                 padding_idx=None,
                 **kwargs):
        super().__init__()
        self.enc_weight = enc_weight
        self.dec_weight = dec_weight
        self.num_classes = num_classes
        self.ctc_loss = CTCLoss(flatten=True)
        self.ce_loss = CELoss2(ignore_index=padding_idx,reduction='none',ignore_first_char=False)

    def _flatten(self, logits, target_lens):
        flatten_logits = torch.cat(
            [s[:target_lens[i]] for i, s in enumerate((logits))])
        return flatten_logits
    # def visual_flatten(self,logit,target_lens):
    #     """Greedy decoder to obtain length from logit.
    #
    #     Returns the first location of padding index or the length of the entire
    #     tensor otherwise.
    #     """
    #     # out as a boolean vector indicating the existence of end token(s)
    #     out = (logit.argmax(dim=-1) == blank_idx)
    #     # abn = out.any(dim)
    #     # # Get the first index of end token
    #     # out = ((out.cumsum(dim) == 1) & out).max(dim)[1]
    #     # out = out + 1
    #     # out = torch.where(abn, out, out.new_tensor(logit.shape[1]))
    #     # out = out.clamp_(2, max_seq_len)
    #     return out

    def single_flatten(self, target_dict, mask_id):
        single_target = torch.stack(
            [target_dict['targets'][i][mask_id[i]] for i, s in enumerate((target_dict['targets']))])
        return single_target

    def _ce_loss(self, logits, targets,focal=False):
        # logits (N*T, C)
        # targets (N*T)
        targets_one_hot = F.one_hot(targets, self.num_classes)
        log_prob = F.log_softmax(logits, dim=-1)
        if focal == True:
            log_prob = log_prob * (1-F.softmax(log_prob,dim=-1))
        loss = -(targets_one_hot.to(log_prob.device) * log_prob).sum(dim=-1)
        return loss.mean()

    def _loss_over_iters(self, outputs, targets):
        """
        Args:
            outputs (list[Tensor]): Each tensor has shape (N, T, C) where N is
                the batch size, T is the sequence length and C is the number of
                classes.
            targets_dicts (dict): The dictionary with at least `padded_targets`
                defined.
        """
        # output [I,(N*T, C)]
        # iter_num I
        # targets (N*T)
        # return (I*N*T,C) , (I*N*T)
        iter_num = len(outputs)
        dec_outputs = torch.cat(outputs, dim=0)
        flatten_targets_iternum = targets.repeat(iter_num)
        return self._ce_loss(dec_outputs, flatten_targets_iternum)

    def forward(self, outputs, targets_dict, en_targets_dict, epoch,img_metas=None):
        """
        Args:
            outputs (dict): The output dictionary with at least one of
                ``out_enc``, ``out_dec`` and ``out_fusers`` specified.
            targets_dict (dict): The target dictionary containing the key
                ``padded_targets``, which represents target sequences in
                shape (batch_size, sequence_length).

        Returns:
            A loss dictionary with ``loss_visual``, ``loss_lang`` and
            ``loss_fusion``. Each should either be the loss tensor or ``0`` if
            the output of its corresponding module is not given.
        """
        assert 'out_enc' in outputs or \
            'out_dec' in outputs or 'out_fusers' in outputs
        losses = {}

        target_lens = [len(t) for t in targets_dict['targets']]
        flatten_targets = torch.cat([t for t in targets_dict['targets']])
        # target_lens (T)
        # flatten_targets (N*T)
        # print(outputs)
        # self.enc_w = 1 - (0.1 * epoch)
        # self.dec_w = 0.1 * epoch
        # if outputs.get('out_enc', None):
        if outputs.get('out_feat', None)!=None:
            feat_input = self._flatten(outputs['out_feat'],
                                      target_lens)
            # feat_loss = self.ctc_loss(outputs['out_feat'],en_targets_dict)['loss_ctc']*self.enc_weight
            # enc_input = (N,T,C) (T) -> (N*T, C)
            # print(enc_input.shape)
            # print(flatten_targets.shape)
            feat_loss = self._ce_loss(feat_input,
                                     flatten_targets) * self.enc_weight
            losses['loss_tps'] = feat_loss

        enc_input = self._flatten(outputs['out_enc'],
                                  target_lens)
        # enc_loss = self.ctc_loss(outputs['out_enc'],en_targets_dict)['loss_ctc']*self.enc_w
        # enc_input = (N,T,C) (T) -> (N*T, C)
        # print(enc_input.shape)
        # print(flatten_targets.shape)
        enc_loss = self._ce_loss(enc_input,
                                 flatten_targets) * self.enc_weight
        losses['loss_visual'] = enc_loss

        # if outputs.get('out_dec', None):
        # dec_logits = [
        #     self._flatten(o['logits'], target_lens)
        #     for o in outputs['out_decs']
        # ]
        # print(outputs['out_dec'].shape)
        # if epoch >= 4:
        # dec_loss = self.ce_loss(outputs['out_dec'],targets_dict)*self.dec_weight
        dec_loss = 0.
        focal = False
        if outputs.get('out_dec', None):
            for i,dec_out in enumerate(outputs['out_dec']):
                whole_dec = dec_out['dec_class']
                dec_input = self._flatten(whole_dec,
                                          target_lens)
                dec_loss = dec_loss + self._ce_loss(dec_input,
                                         flatten_targets,focal) * self.dec_weight
            losses['loss_lang'] = dec_loss

        # fusers_loss = 0.
        # for i, fusers_out in enumerate(outputs['out_fusers']):
        if outputs.get('out_fusers', None):
            fusers_out = outputs['out_fusers']
            whole_dec = fusers_out['logits']
            fusers_input = self._flatten(whole_dec,
                                      target_lens)
            fusers_loss = self._ce_loss(fusers_input,
                                                flatten_targets, focal) * self.dec_weight

            losses['loss_fuser'] = fusers_loss

        # single char mask_id loss
        # if len(outputs['out_dec']) > 1:
        #     for dec_out in outputs['out_dec'][1:]:
        #        mask_id = dec_out['mask_id']
        #        single_out = dec_out['single_out']
        #        single_input = self.single_flatten(targets_dict,mask_id)
        #        dec_loss = dec_loss + self._ce_loss(single_out,
        #                                  single_input) * self.dec_weight


        # dec_input = self._flatten(outputs['out_dec'],
        #                           target_lens)
        # # [I,(N,T,C)] -> [I,(N*T, C)]
        # dec_loss = self._ce_loss(dec_input,
        #                          flatten_targets) * self.dec_weight

        # if outputs['out_dec']['mask_id']!=None:
        #     targets = self.single_flatten(targets_dict,
        #                             outputs['out_dec']['mask_id'])
        #     dec_single_loss = self._ce_loss(outputs['out_dec']['single_out'], targets)* self.dec_weight
            # # dec_loss = self._loss_over_iters(dec_logits,
            # #                                  flatten_targets) * self.dec_weight
            # dec_loss = self.ce_loss(outputs['out_dec'],targets_dict['mask_targets'])*self.dec_w


        return losses
