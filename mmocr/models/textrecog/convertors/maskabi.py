# Copyright (c) OpenMMLab. All rights reserved.
import torch

import mmocr.utils as utils
from mmocr.models.builder import CONVERTORS
from .attn import AttnConvertor


@CONVERTORS.register_module()
class Mask_ABIConvertor(AttnConvertor):
    """Convert between text, index and tensor for encoder-decoder based
    pipeline. Modified from AttnConvertor to get closer to ABINet's original
    implementation.

    Args:
        dict_type (str): Type of dict, should be one of {'DICT36', 'DICT90'}.
        dict_file (None|str): Character dict file path. If not none,
            higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, higher
            priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        max_seq_len (int): Maximum sequence length of label.
        lower (bool): If True, convert original string to lower case.
        start_end_same (bool): Whether use the same index for
            start and end token or not. Default: True.
    """
    def mask_id_gen(self, mask_ids, targets):
        '''

        Args:
            mask_ids: list[tensor]): [torch.Tensor([1,2,3,3,4]),
                    torch.Tensor([5,4,6,3,7])].
            tensors (list[tensor]): [torch.Tensor([1,2,3,3,4]),
                    torch.Tensor([5,4,6,3,7])].

        Returns:
            mask_tensors : [tensor([ 1,  2,  3, 99, 99,  5, 99]), tensor([ 1, 99,  3, 99,  4,  5, 99])]
            other_tensors : [tensor([99, 99, 99,  4,  4, 99,  6]), tensor([99,  2, 99,  4, 99, 99,  5])]
        '''

        mask_tensors = []
        other_tensors = []
        unknown = self.unknown_idx
        for i, tar in enumerate(targets):
            mask_id = mask_ids[i]
            tensors = torch.zeros_like(tar)
            tensors[mask_id] = 1
            # mask_tensor = tensors.scatter(dim=0,index = mask_id,src = src)
            mask_tensor = torch.where(tensors == 1, tensors * unknown, tar)
            other_tensor = torch.where((1 - tensors) == 1, (1 - tensors) * unknown, tar)
            mask_tensors.append(mask_tensor)
            other_tensors.append(other_tensor)

        return mask_tensors, other_tensors

    def mask_target(self, padded_target):
        mask_fill = self.unknown_idx
        batch_size, len = padded_target.size()
        padded_target = padded_target.unsqueeze(-1).repeat(1,1,len) # B, T, T

        mask = torch.eye(len,device=padded_target.device)
        other_mask = (1 - mask).float().masked_fill(1 - mask == 1, mask_fill)
        mask = mask.float().masked_fill(mask == 1, mask_fill)

        # tmp = (mask + padded_target).long()
        # torch.where(padded_target > mask_fill, padded_target, tmp)

        # torch.where((mask + padded_target) > mask_fill*2 , padded_target, mask_fill)
        # mask_target = (padded_target+mask).long()
        # mask_no_padd = torch.where( mask_target== mask_fill+self.padding_idx, self.padding_idx, mask_target )
        # mask_other_padd
        mask = mask.long()
        other_mask = other_mask.long()
        mask_no_padd = torch.where(padded_target > mask, padded_target, mask)
        other_mask_no_padd = torch.where(padded_target > other_mask, padded_target, other_mask)
        output = torch.cat((mask_no_padd,other_mask_no_padd),0)
        output = output.view(batch_size*2*len,len)
        return output


    def str2tensor(self, strings):
        """
        Convert text-string into tensor. Different from
        :obj:`mmocr.models.textrecog.convertors.AttnConvertor`, the targets
        field returns target index no longer than max_seq_len (EOS token
        included).

        Args:
            strings (list[str]): For instance, ['hello', 'world']

        Returns:
            dict: A dict with two tensors.

            - | targets (list[Tensor]): [torch.Tensor([1,2,3,3,4,8]),
                torch.Tensor([5,4,6,3,7,8])]
            - | padded_targets (Tensor): Tensor of shape
                (bsz * max_seq_len)).
        """
        assert utils.is_type_list(strings, str)

        tensors, padded_targets = [], []
        indexes = self.str2idx(strings)
        for index in indexes:
            tensor = torch.LongTensor(index[:self.max_seq_len - 1] +
                                      [self.end_idx])
            tensors.append(tensor)
            # target tensor for loss
            src_target = torch.LongTensor(tensor.size(0) + 1).fill_(0)
            src_target[0] = self.start_idx
            src_target[1:] = tensor
            padded_target = (torch.ones(self.max_seq_len) *
                             self.padding_idx).long()
            char_num = src_target.size(0)
            if char_num > self.max_seq_len:
                padded_target = src_target[:self.max_seq_len]
            else:
                padded_target[:char_num] = src_target
            padded_targets.append(padded_target)
        padded_targets = torch.stack(padded_targets, 0).long()
        mask_targets = self.mask_target(padded_targets)

        return {'targets': tensors,
                'padded_targets': padded_targets,
                'mask_targets':mask_targets}
