"""
YOLO training module
Refactored to use configuration directly
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.config_loader import ConfigLoader
from common.logger import OMRLogger
from common.paths import PathManager

from .train_yolo import train as train_yolo_func


def train_yolo(config: ConfigLoader, path_manager: PathManager) -> None:
    """Train YOLO model"""
    logger = OMRLogger.get_logger('yolo')
    
    yolo_config = config.get_module_config('yolo')
    train_config = yolo_config.get('train', {})
    
    if not train_config.get('enabled', True):
        logger.info("YOLO training is disabled in config")
        return
    
    logger.info("Starting YOLO training...")
    
    # Create args namespace compatible with train_yolo
    class Args:
        def __init__(self, config_dict, path_manager, global_config):
            self.data = str(path_manager.resolve_path(config_dict.get('data_yaml', 'data/Yolo-Dataset/data.yaml')))
            self.weights = config_dict.get('pretrained_weights', 'yolo12l.pt')
            self.epochs = config_dict.get('epochs', 500)
            self.batch = config_dict.get('batch_size', 8)
            self.imgsz = config_dict.get('image_size', 640)
            self.lr0 = config_dict.get('learning_rate', 0.000055)
            self.optimizer = config_dict.get('optimizer', 'AdamW')
            self.momentum = config_dict.get('momentum', 0.9)
            self.patience = config_dict.get('patience', 100)
            self.device = global_config.get('device', '0')
            self.workers = config_dict.get('workers', 8)
            self.project = config_dict.get('project_dir', 'model/yolo/detect')
            self.name = config_dict.get('run_name', 'yolo12l_muscima_finetune')
            self.exist_ok = config_dict.get('exist_ok', True)
            self.seed = global_config.get('seed', 0)
            self.cache = config_dict.get('cache', 'ram')
            self.freeze = config_dict.get('freeze_layers', 0)
            self.resume = config_dict.get('resume_training', '')
            self.close_mosaic = config_dict.get('close_mosaic', False)
            self.cos_lr = config_dict.get('cosine_lr', False)
            self.save_period = config_dict.get('save_period', -1)
            self.amp = config_dict.get('amp', False)
            self.val = config_dict.get('run_validation', False)
            self.test = config_dict.get('run_test', False)
            self.export = config_dict.get('export_format', '')
    
    args = Args(train_config, path_manager, config.global_config)
    
    logger.info(f"Data: {args.data}, Epochs: {args.epochs}, Batch: {args.batch}")
    logger.info(f"Model: {args.weights}, Output: {args.project}/{args.name}")
    
    # Call training function
    best_path = train_yolo_func(args)
    
    logger.info(f"YOLO training completed. Best weights: {best_path}")
