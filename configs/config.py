from yacs.config import CfgNode as CN
from attrdict import AttrDict

_C = CN()

_C.DATASET_ROOT = '/media/ml_data/projects/lyft'
# _C.DATASET_ROOT = '/root/data/lyft'

_C.IMG_SIZE = 1024
_C.DATA_GENERATION_WORKERS = 1

_C.VOXEL_SIZE = 0.2
_C.VOXEL_Z_SIZE = 0.7
_C.IMG_CHANNELS = 6

_C.IMG_SIZE_CROP = 512
_C.EPOCHS = 128
_C.BATCH_SIZE = 16

_C.REG_COEF = 0.05
_C.BOX_SCALE = 0.9
_C.Z_OFFSET = -2.0

_C.GRAD_NORM = 8.0
_C.LR_DECAY = 0.8
_C.DROPOUT = 0.0

_C.CYCLE_LEN = 4

_C.BACKBONE = 'resnet34'
_C.DATASET_GENERATION = False
_C.MIXED_PRECISION = True

_C.CONFIG_TO_MERGE = 'configs/config.yaml'

_C.ARTIFACTS_FOLDER = ''


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values"""
    return _C.clone()


def get_cfg():
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg.CONFIG_TO_MERGE)

    cfg.ARTIFACTS_FOLDER = "./artifacts_IMG_SIZE_{}_VOXEL_{:.2f}_VOXEL_Z_{:.2f}_HEIGHT_{:.2f}_BOX_SCALE_{:.2f}".format(
        cfg.IMG_SIZE, cfg.VOXEL_SIZE, cfg.VOXEL_Z_SIZE, cfg.VOXEL_Z_SIZE * cfg.IMG_CHANNELS, _C.BOX_SCALE
    )

    return cfg


if __name__ == '__main__':
    cfg = get_cfg()
