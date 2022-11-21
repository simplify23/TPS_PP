# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmocr.models.builder import LOSSES
from .ctc_loss import CTCLoss
# from .ce_loss import CELoss

@LOSSES.register_module()
class CELoss3(nn.Module):
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
        targets = targets_dict
        # targets = targets_dict['padded_targets']
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
class CountMixLoss(nn.Module):
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
        self.ce_loss = CELoss3(ignore_index=padding_idx,reduction='none',ignore_first_char=False)
        self.mse_loss = nn.MSELoss()
    def _flatten(self, logits, target_lens):
        flatten_logits = torch.cat(
            [s[:target_lens[i]] for i, s in enumerate((logits))])
        return flatten_logits

    def _ce_loss(self, logits, targets):
        # logits (N*T, C)
        # targets (N*T)
        targets_one_hot = F.one_hot(targets, self.num_classes)
        log_prob = F.log_softmax(logits, dim=-1)
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
        target_len_tensor = torch.Tensor(target_lens)
        flatten_targets = torch.cat([t for t in targets_dict['targets']])

        count_dec = outputs['out_dec'][:, 0, :].mean(-1)
        # count_len = count_dec
        target_len_tensor = target_len_tensor.to(count_dec.device)
        # target_lens (T)
        # flatten_targets (N*T)
        # print(outputs)
        self.enc_w = 1 - (0.1 * epoch)
        self.dec_w = 0.1 * epoch + 0.1
        # if outputs.get('out_enc', None):
        # count_enc = outputs['out_enc'][:,0,:].mean(-1) * 10

        enc_input = self._flatten(outputs['out_enc'],
                                  target_lens)
        # enc_loss = self.ctc_loss(outputs['out_enc'],en_targets_dict)['loss_ctc']*self.enc_w
        # enc_input = (N,T,C) (T) -> (N*T, C)
        # print(enc_input.shape)
        # print(flatten_targets.shape)
        enc_loss = self._ce_loss(enc_input,flatten_targets) * self.enc_weight
                   # self.mse_loss(count_enc,target_len_tensor))
        losses['loss_visual'] = enc_loss

        # if outputs.get('out_dec', None):
        # dec_logits = [
        #     self._flatten(o['logits'], target_lens)
        #     for o in outputs['out_decs']
        # ]
        # print(outputs['out_dec'].shape)
        # count_dec = outputs['out_dec'][:, 0, :].mean(-1) * 10
        # target_len_tensor = target_len_tensor.to(count_dec.device)
        dec_input = self._flatten(outputs['out_dec'][:,1:,:],
                                  target_lens)
        # # [I,(N,T,C)] -> [I,(N*T, C)]
        dec_loss = (self._ce_loss(dec_input,flatten_targets) )* self.dec_w +\
                    self.mse_loss(count_dec,target_len_tensor)
        # # dec_loss = self._loss_over_iters(dec_logits,
        # #                                  flatten_targets) * self.dec_weight
        # dec_loss = self.ce_loss(outputs['out_dec'],targets_dict['mask_targets'])*self.dec_w
        losses['loss_lang'] = dec_loss

        return losses
