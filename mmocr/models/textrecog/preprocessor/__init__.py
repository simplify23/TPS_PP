# Copyright (c) OpenMMLab. All rights reserved.
from .base_preprocessor import BasePreprocessor
from .moran import MORAN
from .spin import SPIN
from .tps_preprocessor import TPSPreprocessor

__all__ = ['BasePreprocessor', 'TPSPreprocessor','MORAN','SPIN']
