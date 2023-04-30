# Copyright (c) OpenMMLab. All rights reserved.
import torch
import math
import torch.nn as nn
from mmcv.runner import BaseModule, Sequential

import mmocr.utils as utils
from mmocr.models.builder import BACKBONES
from mmocr.models.textrecog.backbones.tps import U_TPSnet, Deform_net, DAttentionBaseline, UDAT_Net, TPSnet, \
    TPSnet_Warp, TPSnetv2
from mmocr.models.textrecog.layers import BasicBlock
from tools.data.textrecog.visual_feat import draw_feature_map

'''
TPResnet: 0layer's size is 32,32,128
down 4 32
TPResnet: 1layer's size is 32,16,64
down 2 64
TPResnet: 2layer's size is 64,16,64
down 2 128
TPResnet: 3layer's size is 128,8,32
TPResnet: 4layer's size is 256,8,32
'''

@BACKBONES.register_module()
class ResNetABI_v2_large(BaseModule):
    """Implement ResNet backbone for text recognition, modified from `ResNet.

    <https://arxiv.org/pdf/1512.03385.pdf>`_ and
    `<https://github.com/FangShancheng/ABINet>`_

    Args:
        in_channels (int): Number of channels of input image tensor.
        stem_channels (int): Number of stem channels.
        base_channels (int): Number of base channels.
        arch_settings  (list[int]): List of BasicBlock number for each stage.
        strides (Sequence[int]): Strides of the first block of each stage.
        out_indices (None | Sequence[int]): Indices of output stages. If not
            specified, only the last stage will be returned.
        last_stage_pool (bool): If True, add `MaxPool2d` layer to last stage.
    """

    def __init__(self,
                 in_channels=3,
                 stem_channels=32,
                 base_channels=32,
                 arch_settings=[3, 4, 6, 6, 3],
                 strides=[2, 1, 2, 1, 1],
                 p_strides=[2, 1, 2, 1, 1],
                 out_indices=None,
                 last_stage_pool=False,
                 init_cfg=[
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Constant', val=1, layer='BatchNorm2d')
                 ]):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, int)
        assert isinstance(stem_channels, int)
        assert utils.is_type_list(arch_settings, int)
        assert utils.is_type_list(strides, int)
        assert len(arch_settings) == len(strides)
        assert out_indices is None or isinstance(out_indices, (list, tuple))
        assert isinstance(last_stage_pool, bool)

        self.out_indices = out_indices
        self.last_stage_pool = last_stage_pool
        self.block = BasicBlock
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        self.tps_layers = []
        planes = base_channels
        h,w = 32,128
        # p_h, p_w = h // 4, w // 4
        for i, num_blocks in enumerate(arch_settings):
            stride = strides[i]
            p_stride = p_strides[i]

            res_layer = self._make_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                blocks=num_blocks,
                stride=stride,
            )
            self.inplanes = planes * self.block.expansion
            planes *= 2
            h, w = h // stride, w // stride

            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def freeze_network(self, ):
        for layers in self.res_layers[:-1]:
            layers = getattr(self, layers)
            layers.eval()
            for name, parameter in layers.named_parameters():
                parameter.requires_grad = False
                # print("{}: {}".format(parameter, parameter.requries_grad))
        self.conv1.eval()
        for name, parameter in self.conv1.named_parameters():
            parameter.requires_grad = False
            # print("{}: {}".format(parameter, parameter.requries_grad))


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        layers = []
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                # tps,
                nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers.append(
            block(
                inplanes,
                planes,
                use_conv1x1=True,
                stride=stride,
                downsample=downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, use_conv1x1=True))

        return Sequential(*layers)

    def _make_stem_layer(self, in_channels, stem_channels):
        self.conv1 = nn.Conv2d(
            in_channels, stem_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(stem_channels)
        self.relu1 = nn.ReLU()

    def return_feature(self, x, tpsnet=None, test=False,):
        """
        Args:
            x (Tensor): Image tensor of shape :math:`(N, 3, H, W)`.

        Returns:
            Tensor or list[Tensor]: Feature tensor. Its shape depends on
            ResNetABI's config. It can be a list of feature outputs at specific
            layers if ``out_indices`` is specified.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # logits = None

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            if i == 2 and tpsnet == None:
                return x
            outs.append(x)
            x = res_layer(x)

        # return tuple(outs) if self.out_indices else x
        return x

    def forward(self, x, tpsnet=None,test=False, **kwargs):
        """
        Args:
            x (Tensor): Image tensor of shape :math:`(N, 3, H, W)`.

        Returns:
            Tensor or list[Tensor]: Feature tensor. Its shape depends on
            ResNetABI's config. It can be a list of feature outputs at specific
            layers if ``out_indices`` is specified.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        # logits = None

        outs = []
        outputs = None
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)

            if i == 2 and tpsnet != None:
                # draw_feature_map(x)
                # if test == True:
                #     epoch = 10
                # else:
                #     epoch = self.epoch
                outputs = tpsnet(x, outs,**kwargs)
                if outputs.get('output', None) != None:
                    x = outputs['output']

            outs.append(x)
            x = res_layer(x)

        return {'output': x, 'img_ref' : outputs.get('output', None) if outputs !=None else None}
