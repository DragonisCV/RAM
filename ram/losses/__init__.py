import importlib
from copy import deepcopy
from os import path as osp

from ram.utils import get_root_logger, scandir
from ram.utils.registry import LOSS_REGISTRY
from .losses import (CharbonnierLoss, GANLoss, L1Loss, MaskL1Loss,MSELoss, PerceptualLoss, WeightedTVLoss, g_path_regularize,
                     gradient_penalty_loss, r1_penalty,EdgeLoss,AdaptiveMaskL1Loss,AdaptiveMaskProbL1Loss)
# __all__= ['build_loss'，gradient_penalty_loss', 'r1_penalty', 'g_path_regularize']
__all__ = ['L1Loss', 'MaskL1Loss', 'MSELoss', 'CharbonnierLoss', 'EdgeLoss','WeightedTVLoss', 'PerceptualLoss', 'GANLoss', 'gradient_penalty_loss',
    'r1_penalty', 'g_path_regularize','AdaptiveMaskL1Loss','AdaptiveMaskProbL1Loss'
]
# automatically scan and import loss modules for registry
# scan all the files under the 'losses' folder and collect files ending with '_loss.py'
loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_loss.py')]
# import all the loss modules
_model_modules = [importlib.import_module(f'ram.losses.{file_name}') for file_name in loss_filenames]


def build_loss(opt):
    """Build loss from options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
