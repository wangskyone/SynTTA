# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Configuration file (powered by YACS).

IMPORTANT CONFIGURATION STEPS:
1. Update DATA_DIR and CKPT_DIR below with your actual paths
2. Ensure datasets are organized in the correct structure (see README.md)
3. Download pre-trained checkpoints and place them in CKPT_DIR (see README.md for structure)
4. For custom checkpoint paths, you can also set CKPT_PATH in individual config files
"""

import argparse
import os
import sys
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode
from loguru import logger
import math

# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ---------------------------------- Misc options --------------------------- #
# Config device
_C.DEVICE = "cuda"

# Setting - see README.md for more information
_C.SETTING = "continual"

# Data directory
# IMPORTANT: Replace this with your actual dataset directory path
# Example: _C.DATA_DIR = "/path/to/your/datasets"
_C.DATA_DIR = "/path/to/your/datasets"

# Weight directory
# IMPORTANT: Replace this with your actual checkpoint directory path
# Example: _C.CKPT_DIR = "/path/to/your/checkpoints"
_C.CKPT_DIR = "/path/to/your/checkpoints"

# Output directory
_C.SAVE_DIR = "../output"

# Path to a specific checkpoint
_C.CKPT_PATH = ""

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# Seed to use. If None, seed is not set!
# Note that non-determinism is still present due to non-deterministic GPU ops.
_C.RNG_SEED = 3407

# Deterministic experiments.
_C.DETERMINISM = False

# Optional description of a config
_C.DESC = ""

# Save config destination
_C.CHECKPOINT_FREQ = 3

_C.INPUT_SIZE = (32, 32)

_C.TASK = "Corruption"  # Options: DG, Corruption, Continual

# ----------------------------- Model options ------------------------------- #
_C.MODEL = CfgNode()

# Some of the available models can be found here:
# Torchvision: https://pytorch.org/vision/0.14/models.html
# timm: https://github.com/huggingface/pytorch-image-models/tree/v0.6.13
# RobustBench: https://github.com/RobustBench/robustbench

_C.MODEL.ARCH = 'Standard'

# Type of pre-trained weights for torchvision models. See: https://pytorch.org/vision/0.14/models.html
_C.MODEL.WEIGHTS = "IMAGENET1K_V1"

# Inspect the cfgs directory to see all possibilities
_C.MODEL.ADAPTATION = 'source'

# Reset the model before every new batch
_C.MODEL.EPISODIC = False

# ----------------------------- DG options -------------------------- #
_C.DG = CfgNode()

_C.DG.DATASET = ''  # Options: domainnet, office31, office-home, PACS, VLCS, terraincognita

_C.DG.STAGE = "train"

_C.DG.TASK = "DG_test"  # Options: DG_test, DA_test













_C.DG.TRAINING_DOMAINS = [1, 2, 3]  # Domains used for training (0: art_painting, 1: cartoon, 2: photo, 3: sketch)
_C.DG.TESTING_DOMAINS = [0]

# ----------------------------- Corruption options -------------------------- #
_C.CORRUPTION = CfgNode()

# Dataset for evaluation
_C.CORRUPTION.DATASET = 'cifar10_c'

# Check https://github.com/hendrycks/robustness for corruption details
_C.CORRUPTION.TYPE = ['gaussian_noise', 'shot_noise', 'impulse_noise',
                      'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
                      'snow', 'frost', 'fog', 'brightness', 'contrast',
                      'elastic_transform', 'pixelate', 'jpeg_compression']
_C.CORRUPTION.SEVERITY = [5]

# Number of examples to evaluate. If num_ex is changed, each sequence is subsampled to the specified amount
# For ImageNet-C, RobustBench loads a list containing 5000 samples.
_C.CORRUPTION.NUM_EX = -1

# ------------------------------- Batch norm options ------------------------ #
_C.BN = CfgNode()

# BN alpha (1-alpha) * src_stats + alpha * test_stats
_C.BN.ALPHA = 0.1

# ------------------------------- Optimizer options ------------------------- #
_C.OPTIM = CfgNode()

# Number of updates per batch
_C.OPTIM.STEPS = 1

# ----------------------------- DG options -------------------------- #
_C.OPTIM.MAX_EPOCH = 50

_C.OPTIM.STEPS_PER_EPOCH = 100

# Learning rate
_C.OPTIM.LR = 1e-3

# Choices: Adam, SGD
_C.OPTIM.METHOD = 'Adam'
# _C.OPTIM.METHOD = 'SGD'

# Beta
_C.OPTIM.BETA = 0.9

# Momentum
_C.OPTIM.MOMENTUM = 0.9

# Momentum dampening
_C.OPTIM.DAMPENING = 0.0

# Nesterov momentum
_C.OPTIM.NESTEROV = True

# L2 regularization
_C.OPTIM.WD = 0.0

# --------------------------------- Mean teacher options -------------------- #
_C.M_TEACHER = CfgNode()

# Mean teacher momentum for EMA update
_C.M_TEACHER.MOMENTUM = 0.999

# --------------------------------- ROTTA options ----------------------------- #
_C.ROTTA = CfgNode()
_C.ROTTA.MEMORY_SIZE = 64
_C.ROTTA.LAMBDA_T = 1.0  
_C.ROTTA.LAMBDA_U = 1.0 
_C.ROTTA.NU = 0.001
_C.ROTTA.ALPHA = 0.05
_C.ROTTA.UPDATE_FREQUENCY = 64  
_C.ROTTA.TEMP = 100.0

# --------------------------------- SynTTA options ----------------------------- #
_C.SYNTTA = CfgNode()
_C.SYNTTA.MEMORY_SIZE = 64
_C.SYNTTA.LAMBDA_T = 1.0  
_C.SYNTTA.LAMBDA_U = 1.0 
_C.SYNTTA.NU = 0.001
_C.SYNTTA.ETA = 0.8
_C.SYNTTA.GAMMA = 0.01
_C.SYNTTA.UPDATE_FREQUENCY = 64  
_C.SYNTTA.TEMP = 100.0

# ------------------------------- Source options ---------------------------- #
_C.SOURCE = CfgNode()

# Number of workers for source data loading
_C.SOURCE.NUM_WORKERS = 4

# Percentage of source samples used
_C.SOURCE.PERCENTAGE = 1.0   # [0, 1] possibility to reduce the number of source samples

# ------------------------------- Testing options --------------------------- #
_C.TEST = CfgNode()

# Number of workers for test data loading
_C.TEST.NUM_WORKERS = 4

# Batch size for evaluation (and updates for norm + tent)
_C.TEST.BATCH_SIZE = 128

# If the batch size is 1, a sliding window approach can be applied by setting window length > 1
_C.TEST.WINDOW_LENGTH = 1

# Number of augmentations for methods relying on TTA (test time augmentation)
_C.TEST.N_AUGMENTATIONS = 32

# The alpha value of the dirichlet distribution used for sorting the class labels.
_C.TEST.ALPHA_DIRICHLET = 0.1

# --------------------------------- CUDNN options --------------------------- #
_C.CUDNN = CfgNode()

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN.BENCHMARK = True

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()


def assert_and_infer_cfg():
    """Checks config values invariants."""
    err_str = "Unknown adaptation method."
    assert _C.MODEL.ADAPTATION in ["source", "norm", "tent"]
    err_str = "Log destination '{}' not supported"
    assert _C.LOG_DEST in ["stdout", "file"], err_str.format(_C.LOG_DEST)


def merge_from_file(cfg_file):
    print(f"Loading config from {cfg_file}")
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def load_cfg_from_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=False,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()

    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    cfg.SAVE_DIR = os.path.join(cfg.SAVE_DIR, f"{cfg.MODEL.ADAPTATION}_{cfg.CORRUPTION.DATASET}_{current_time}")
    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    # Remove default logger and add file and stdout handlers
    logger.remove()
    logger.add(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST), 
               format="[<green>{time:YY/MM/DD HH:mm:ss}</green>] [<cyan>{file.name}:{line}</cyan>]: {message}")
    logger.add(sys.stdout, 
               format="[<green>{time:YY/MM/DD HH:mm:ss}</green>] [<cyan>{file.name}:{line}</cyan>]: {message}")

    if cfg.RNG_SEED:
        torch.manual_seed(cfg.RNG_SEED)
        torch.cuda.manual_seed(cfg.RNG_SEED)
        np.random.seed(cfg.RNG_SEED)
        random.seed(cfg.RNG_SEED)
        torch.backends.cudnn.benchmark = cfg.CUDNN.BENCHMARK

        if cfg.DETERMINISM:
            # enforce determinism
            if hasattr(torch, "set_deterministic"):
                torch.set_deterministic(True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info("PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(str(cfg))


def complete_data_dir_path(root, dataset_name):
    # map dataset name to data directory name
    mapping = {"imagenet": "imagenet2012",
               "imagenet_c": "ImageNet/imagenet-c",
               "imagenet_r": "ImageNet/imagenet-r",
               "imagenet_k": os.path.join("ImageNet-Sketch", "sketch"),
               "imagenet_a": "imagenet-a",
               "imagenet_d": "imagenet-d",      # do not change
               "imagenet_d109": "imagenet-d",   # do not change
               "office31": "office-31",
               "visda": "visda-2017",
               "cifar10": "",  # do not change the following values
               "cifar10_c": "CIFAR/cifar10",
               "cifar100": "",
               "cifar100_c": "CIFAR/cifar100",
               "PACS": "PACS",
               "VLCS": "VLCS/VLCS",
               "office-home": "OfficeHome",
               "terraincognita": "Terraincognita",
               "domainnet": "DomainNet/DomainNet",
               }
    return os.path.join(root, mapping[dataset_name])

def get_DG_domain_names(dataset_name):
    """Returns the domain names for the specified dataset."""
    if dataset_name == 'office':
        return ['A', 'D', 'W']  # amazon, dslr, webcam
    elif dataset_name == 'office-caltech':
        return ['A', 'D', 'W', 'C']  # amazon, dslr, webcam, caltech
    elif dataset_name == 'office-home':
        return ['A', 'C', 'P', 'R']  # Art, Clipart, Product, RealWorld
    elif dataset_name == 'dg5':
        return ['M', 'MM', 'S', 'SY', 'U']  # mnist, mnist_m, svhn, syn, usps
    elif dataset_name == 'PACS':
        return ['A', 'C', 'P', 'S']  # art_painting, cartoon, photo, sketch
    elif dataset_name == 'VLCS':
        return ['C', 'L', 'S', 'V']  # Caltech101, LabelMe, SUN09, VOC2007
    elif dataset_name == 'DomainNet':
        return ['C', 'P', 'R', 'S']  # clipart, painting, real, sketch
    elif dataset_name == 'terraincognita':
        return ['L38', 'L43', 'L46', 'L100']  # location_38, location_43, location_46, location_100
    else:
        raise ValueError(f"No such dataset exists: {dataset_name}")

def get_DG_domain_fullnames(dataset_name):
    """Returns the full domain names for the specified dataset."""
    if dataset_name == 'office':
        return ['amazon', 'dslr', 'webcam']
    elif dataset_name == 'office-caltech':
        return ['amazon', 'dslr', 'webcam', 'caltech']
    elif dataset_name == 'office-home':
        return ['Art', 'Clipart', 'Product', 'RealWorld']
    elif dataset_name == 'dg5':
        return ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
    elif dataset_name == 'PACS':
        return ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset_name == 'VLCS':
        return ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset_name == 'domainnet' or dataset_name == 'DomainNet':
        return ['clipart', 'painting', 'real', 'sketch']
    elif dataset_name == 'terraincognita':
        return ['location_38', 'location_43', 'location_46', 'location_100']
    else:
        raise ValueError(f"No such dataset exists: {dataset_name}")

def get_num_classes(dataset_name):
    dataset_name2num_classes = {"cifar10": 10, "cifar10_c": 10, "cifar100": 100,  "cifar100_c": 100,
                                "imagenet": 1000, "imagenet_c": 1000, "imagenet_k": 1000, "imagenet_r": 200,
                                "imagenet_a": 200, "imagenet_d": 164, "imagenet_d109": 109, "imagenet200": 200,
                                "domainnet126": 126, "office31": 31, "visda": 12, "office-home": 65,
                                "PACS": 7, "VLCS": 5, "terraincognita": 10, "domainnet": 345,
                                }
    return dataset_name2num_classes[dataset_name]


def get_domain_sequence(ckpt_path):
    assert ckpt_path.endswith('.pth') or ckpt_path.endswith('.pt') or ckpt_path.endswith('.tar')
    domain = cfg.CKPT_PATH.replace('.pth', '').split(os.sep)[-1].split('_')[1]
    mapping = {"real": ["clipart", "painting", "sketch"],
               "clipart": ["sketch", "real", "painting"],
               "painting": ["real", "sketch", "clipart"],
               "sketch": ["painting", "clipart", "real"],
               }
    return mapping[domain]

def adaptation_method_lookup(adaptation):
    lookup_table = {"source": "Norm",
                    "tent": "Tent",
                    "rotta": "RoTTA",
                    "syntta": "SynTTA",
                    }
    assert adaptation in lookup_table.keys(), \
        f"Adaptation method '{adaptation}' is not supported! Choose from: {list(lookup_table.keys())}"
    return lookup_table[adaptation]
