"""
MLP training module
Refactored to use configuration directly
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.config_loader import ConfigLoader
from common.logger import OMRLogger
from common.paths import PathManager

# Import MLP training functions
from .train_core import (
    load_data, load_model, train_mlp as train_mlp_func
)
from .assemblage.configs.assembler.default import get_cfg_defaults
from common.constants import get_classlist_and_classdict
from common.utility import set_seed
import torch
import yaml


def train_mlp(config: ConfigLoader, path_manager: PathManager) -> None:
    """Train MLP model"""
    logger = OMRLogger.get_logger('mlp')
    
    mlp_config = config.get_module_config('mlp')
    train_config = mlp_config.get('train', {})
    
    if not train_config.get('enabled', True):
        logger.info("MLP training is disabled in config")
        return
    
    logger.info("Starting MLP training...")
    
    # Get configuration values
    gt_annotations_root = path_manager.resolve_path(train_config.get('gt_annotations_root', 'data/v1.0/data/MUSCIMA++/v2.0/data/annotations'))
    images_root = path_manager.resolve_path(train_config.get('images_root', 'data/v1.0/data/MUSCIMA++/datasets_r_staff/images'))
    split_file = path_manager.resolve_path(train_config.get('split_file', 'src/mlp/assemblage/splits/mob_split.yaml'))
    model_config_path = path_manager.resolve_path(train_config.get('model_config', 'src/mlp/assemblage/configs/assembler/default.py'))
    data_config_path = path_manager.resolve_path(train_config.get('data_config', 'src/mlp/assemblage/configs/muscima_nogrammar.yaml'))
    exp_name = train_config.get('exp_name', 'mlp_training')
    classes = train_config.get('classes', 'essential')
    threshold = float(train_config.get('threshold', 0.5))
    
    # Setup paths
    checkpoint_path = path_manager.resolve_path(train_config.get('checkpoint_path', f'model/mlp/{exp_name}/model_best.pth'))
    checkpoint_dir = checkpoint_path.parent
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Get visualization output directory
    vis_config = mlp_config.get('visualize', {})
    vis_output_dir = path_manager.resolve_path(vis_config.get('output_dir', 'vis_stat/mlp'))
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"GT annotations: {gt_annotations_root}")
    logger.info(f"Images root: {images_root}")
    logger.info(f"Split file: {split_file}")
    logger.info(f"Experiment name: {exp_name}")
    logger.info(f"Classes: {classes}")
    
    # Load configuration from mlp/assemblage (legacy config system)
    cfg = get_cfg_defaults()
    set_seed(config.global_config.get('seed', 42))
    
    # Override cfg with values from setup.yml
    cfg.TRAIN.BATCH_SIZE = train_config.get('batch_size', 32)
    cfg.TRAIN.NUM_EPOCHS = train_config.get('num_epochs', 100)
    cfg.TRAIN.LEARNING_RATE = train_config.get('learning_rate', 0.001)
    cfg.TRAIN.POS_WEIGHT = train_config.get('pos_weight', 1.0)
    cfg.TRAIN.SAVE_FREQUENCY = train_config.get('save_period', 5)
    
    # Load device
    device_str = config.global_config.get('device', 'auto')
    if device_str == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Batch size: {cfg.TRAIN.BATCH_SIZE}, Epochs: {cfg.TRAIN.NUM_EPOCHS}, LR: {cfg.TRAIN.LEARNING_RATE}")
    
    # Get class list and dict
    class_list, class_dict = get_classlist_and_classdict(classes)
    class_list = list(class_list)
    logger.info(f'Loaded {len(class_list)} classes')
    
    # Load data configuration
    with open(data_config_path, 'rb') as hdl:
        data_config = yaml.load(hdl, Loader=yaml.FullLoader)
    data_config['mode'] = cfg.MODEL.MODE
    
    # Load data
    data = load_data(
        gt_annotations_root=str(gt_annotations_root),
        images_root=str(images_root),
        split_file=str(split_file),
        class_list=class_list,
        class_dict=class_dict,
        data_config=data_config
    )
    data_train = data['train']
    data_valid = data['valid']
    
    logger.info(f'Training samples: {len(data_train):,}')
    logger.info(f'Validation samples: {len(data_valid):,}')
    
    # Load model
    model = load_model(cfg, str(device))
    
    # Train
    train_mlp_func(
        model=model,
        cfg=cfg,
        device=str(device),
        train_data=data_train,
        valid_data=data_valid,
        output_dir=str(checkpoint_dir.parent),  # model/mlp
        exp_name=exp_name,
        threshold=threshold,
        model_save_dir=str(checkpoint_dir),
        viz_save_dir=str(vis_output_dir)
    )
    
    logger.info(f"MLP training completed. Best model saved to {checkpoint_path}")
