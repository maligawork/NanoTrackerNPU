# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

__C.META_ARC = ""

__C.CUDA = False

__C.ONNX_BACKBONE_INIT = ''
__C.ONNX_BACKBONE = ''
__C.ONNX_HEAD = ''

__C.OPENCV_PLATFORM = 'NPU'
__C.OPENCV_BACKBONE_INIT = ''
__C.OPENCV_BACKBONE = ''
__C.OPENCV_HEAD = ''


# Point stride
__C.STRIDE = 16

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #

# Scale penalty
__C.PENALTY_K = 0.150

# Window influence
__C.WINDOW_INFLUENCE = 0.490

# Interpolation learning rate
__C.LR = 0.385

# Exemplar size
__C.EXEMPLAR_SIZE = 127

# Instance size
__C.INSTANCE_SIZE = 255

# Base size
__C.BASE_SIZE = 7

# Context amount
__C.CONTEXT_AMOUNT = 0.5
