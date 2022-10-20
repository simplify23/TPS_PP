# Copyright (c) OpenMMLab. All rights reserved.
from .nrtr_modality_transformer import NRTRModalityTransform
from .resnet31_ocr import ResNet31OCR
from .resnet_abi import ResNetABI
from .shallow_cnn import ShallowCNN
from .very_deep_vgg import VeryDeepVgg
from .van import VAN
from .pvtr import PVTR
from .pvtr_v2 import SPTR
from .resnet_v2 import ResNetABI_v2
from .resnet_v3 import ResNetABI_v3
from .resnet_v2_large import ResNetABI_v2_large
from .tps import U_TPSnet,U_TPSnet_v3
from .efficienet import efficientnet
__all__ = [
    'ResNet31OCR', 'VeryDeepVgg', 'NRTRModalityTransform', 'ShallowCNN','efficientnet',
    'ResNetABI','VAN','PVTR','SPTR','ResNetABI_v2','ResNetABI_v3','ResNetABI_v2_large','U_TPSnet','U_TPSnet_v3'
]
