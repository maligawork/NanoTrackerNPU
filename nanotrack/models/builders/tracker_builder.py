import importlib
from nanotrack.core.config import cfg

PLATFORMS = ('onnx', 'opencv')


def get_tracker(platform: str):
    if platform not in PLATFORMS:
        raise Exception('Platform doesn\'t exist!')

    cfg.merge_from_file(f'models/config/config_{platform}.yaml')
    builder = importlib.import_module(f'nanotrack.models.builders.{platform}_builder')

    tracker = builder.create_tracker()
    return tracker
