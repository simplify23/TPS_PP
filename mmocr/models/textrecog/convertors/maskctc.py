# Copyright (c) OpenMMLab. All rights reserved.
import torch

import mmocr.utils as utils
from mmocr.models.builder import CONVERTORS
from .ctc import CTCConvertor


@CONVERTORS.register_module()
class MaskCTCConvertor(CTCConvertor):
    """Convert between text, index and tensor for CTC loss-based pipeline.

    Args:
        dict_type (str): Type of dict, should be either 'DICT36' or 'DICT90'.
        dict_file (None|str): Character dict file path. If not none, the file
            is of higher priority than dict_type.
        dict_list (None|list[str]): Character list. If not none, the list
            is of higher priority than dict_type, but lower than dict_file.
        with_unknown (bool): If True, add `UKN` token to class.
        lower (bool): If True, convert original string to lower case.
    """

    def __init__(self,
                 dict_type='DICT90',
                 dict_file=None,
                 dict_list=None,
                 with_unknown=True,
                 lower=False,
                 **kwargs):
        super().__init__(dict_type, dict_file, dict_list)
        assert isinstance(with_unknown, bool)
        assert isinstance(lower, bool)

        self.with_unknown = with_unknown
        self.lower = lower
        self.update_dict()

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
            #     mask_tensor = tensors.scatter(dim=0,index = mask_id,src = src)
            mask_tensor = torch.where(tensors == 1, tensors * unknown, tar)
            other_tensor = torch.where((1 - tensors) == 1, (1 - tensors) * unknown, tar)
            mask_tensors.append(mask_tensor)
            other_tensors.append(other_tensor)

        return mask_tensors, other_tensors

    def str2tensor_mask(self, strings, mask_id):
        """Convert text-string to ctc-loss input tensor.

        Args:
            strings (list[str]): ['hello', 'world'].
        Returns:
            dict (str: tensor | list[tensor]):
                tensors (list[tensor]): [torch.Tensor([1,2,3,3,4]),
                    torch.Tensor([5,4,6,3,7])].
                flatten_targets (tensor): torch.Tensor([1,2,3,3,4,5,4,6,3,7]).
                target_lengths (tensor): torch.IntTensot([5,5]).
        """
        assert utils.is_type_list(strings, str)

        tensors = []
        indexes = self.str2idx(strings)
        for index in indexes:
            tensor = torch.IntTensor(index)
            tensors.append(tensor)
        target_lengths = torch.IntTensor([len(t) for t in tensors])
        flatten_target = torch.cat(tensors)
        mask_tensors,other_tensors = self.mask_id_gen(mask_id, tensors)
        return {
            'targets': tensors,
            'flatten_targets': flatten_target,
            'target_lengths': target_lengths,
            'mask_targets' : mask_tensors,
            'other_targets' : other_tensors
        }

