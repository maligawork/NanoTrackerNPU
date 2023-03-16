from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from nanotrack.core.config import cfg
from nanotrack.tracker.nano_tracker import NanoTracker
from nanotrack.models.utils.onnx_utils import load_onnx, run_onnx


class ModelBuilder:
    def __init__(self):
        super(ModelBuilder, self).__init__()

        self.zf = None
        self.backbone_init_path = cfg.ONNX_BACKBONE_INIT
        self.backbone_path = cfg.ONNX_BACKBONE
        self.head_path = cfg.ONNX_HEAD

        if cfg.CUDA:
            provider = 'CUDAExecutionProvider'
        else:
            provider = 'CPUExecutionProvider'

        self.backbone_init = load_onnx(self.backbone_init_path, provider)
        self.backbone = load_onnx(self.backbone_path, provider)
        self.ban_head = load_onnx(self.head_path, provider)

    def template(self, z):
        zf = run_onnx(self.backbone_init, {'input': z})[0]
        self.zf = zf

    def track(self, x):
        xf = run_onnx(self.backbone, {'input': x})[0]
        cls, loc = run_onnx(self.ban_head, {'input1': self.zf, 'input2': xf})

        return {
            'cls': cls,
            'loc': loc,
        }


def create_tracker():
    model = ModelBuilder()
    tracker = NanoTracker(model)
    return tracker
