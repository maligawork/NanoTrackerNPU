from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np

from nanotrack.core.config import cfg
from nanotrack.tracker.nano_tracker import NanoTracker
from nanotrack.models.utils.opencv_utils import load_opencv, run_opencv
# /usr
# /home/khadas/opencv/build/3rdparty/libtim-vx/TIM-VX-1d9c7ab941b3d8d9c4d28d80058402725731e3d6/include

class ModelBuilder:
    def __init__(self):
        super(ModelBuilder, self).__init__()

        self.zf = None
        self.backbone_init_path = cfg.OPENCV_BACKBONE_INIT
        self.backbone_path = cfg.OPENCV_BACKBONE
        self.head_path = cfg.OPENCV_HEAD

        if cfg.OPENCV_PLATFORM == 'NPU':
            self.backend = cv2.dnn.DNN_BACKEND_TIMVX
            self.target = cv2.dnn.DNN_TARGET_NPU
        else:
            self.backend = cv2.dnn.DNN_BACKEND_DEFAULT
            self.target = cv2.dnn.DNN_TARGET_CPU

        self.backbone_init = load_opencv(self.backbone_init_path, self.backend, self.target)
        self.backbone = load_opencv(self.backbone_path, self.backend, self.target)
        self.ban_head = load_opencv(self.head_path, self.backend, self.target)

    def template(self, z):
        zf = run_opencv(self.backbone_init, z, ['output'])[0]
        self.zf = np.zeros((1, 48, 16, 16))
        self.zf[..., 4:-4, 4:-4] = zf

    def track(self, x):
        xf = run_opencv(self.backbone, x, ['output'])[0]
        out = np.concatenate((self.zf.copy(), xf), axis=1)
        cls, loc = run_opencv(self.ban_head, out, ['output1', 'output2'])

        return {
            'cls': cls,
            'loc': loc,
        }


def create_tracker():
    model = ModelBuilder()
    tracker = NanoTracker(model)
    return tracker
