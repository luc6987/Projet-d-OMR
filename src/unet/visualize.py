"""
U-Net visualization module
Generate training curves and sample visualizations
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.config_loader import ConfigLoader
from common.logger import OMRLogger
from common.paths import PathManager
import json
import matplotlib.pyplot as plt
from glob import glob


def visualize_unet(config: ConfigLoader, path_manager: PathManager) -> None:
    """Generate U-Net visualizations"""
    logger = OMRLogger.get_logger('unet')
    
    unet_config = config.get_module_config('unet')
    vis_config = unet_config.get('visualize', {})
    
    if not vis_config.get('enabled', True):
        logger.info("U-Net visualization is disabled in config")
        return
    
    logger.info("Generating U-Net visualizations...")
    
    output_dir = path_manager.resolve_path(vis_config.get('output_dir', 'vis_stat/unet'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = path_manager.resolve_path(config.global_config.get('log_dir', 'logs'))
    
    # Find training logs
    log_files = sorted(glob(str(log_dir / "unet_training_*.json")))
    
    if not log_files:
        logger.warning("No training logs found. Run training first.")
        return
    
    # Load most recent log
    latest_log = log_files[-1]
    logger.info(f"Loading training log: {latest_log}")
    
    with open(latest_log, 'r') as f:
        log_data = json.load(f)
    
    history = log_data.get('history', [])
    if not history:
        logger.warning("No training history found in log.")
        return
    
    # Extract metrics
    epochs = [entry['epoch'] for entry in history]
    train_losses = [entry['train_loss'] for entry in history]
    val_losses = [entry['val_loss'] for entry in history]
    train_acc = [entry['train_pixel_acc'] for entry in history]
    val_acc = [entry['val_pixel_acc'] for entry in history]
    train_iou = [entry['train_mean_iou'] for entry in history]
    val_iou = [entry['val_mean_iou'] for entry in history]
    
    # Plot training curves
    if vis_config.get('save_training_curves', True):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(epochs, train_losses, label='Train', marker='o')
        axes[0].plot(epochs, val_losses, label='Validation', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, train_acc, label='Train', marker='o')
        axes[1].plot(epochs, val_acc, label='Validation', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Pixel Accuracy')
        axes[1].set_title('Pixel Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # IoU
        axes[2].plot(epochs, train_iou, label='Train', marker='o')
        axes[2].plot(epochs, val_iou, label='Validation', marker='s')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Mean IoU')
        axes[2].set_title('Mean IoU')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        curve_path = output_dir / 'training_curves.png'
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training curves to {curve_path}")
    
    logger.info("U-Net visualization completed")

    """Generate U-Net visualizations"""
    logger = OMRLogger.get_logger('unet')
    
    unet_config = config.get_module_config('unet')
    vis_config = unet_config.get('visualize', {})
    
    if not vis_config.get('enabled', True):
        logger.info("U-Net visualization is disabled in config")
        return
    
    logger.info("Generating U-Net visualizations...")
    
    output_dir = path_manager.resolve_path(vis_config.get('output_dir', 'vis_stat/unet'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = path_manager.resolve_path(config.global_config.get('log_dir', 'logs'))
    
    # Find training logs
    log_files = sorted(glob(str(log_dir / "unet_training_*.json")))
    
    if not log_files:
        logger.warning("No training logs found. Run training first.")
        return
    
    # Load most recent log
    latest_log = log_files[-1]
    logger.info(f"Loading training log: {latest_log}")
    
    with open(latest_log, 'r') as f:
        log_data = json.load(f)
    
    history = log_data.get('history', [])
    if not history:
        logger.warning("No training history found in log.")
        return
    
    # Extract metrics
    epochs = [entry['epoch'] for entry in history]
    train_losses = [entry['train_loss'] for entry in history]
    val_losses = [entry['val_loss'] for entry in history]
    train_acc = [entry['train_pixel_acc'] for entry in history]
    val_acc = [entry['val_pixel_acc'] for entry in history]
    train_iou = [entry['train_mean_iou'] for entry in history]
    val_iou = [entry['val_mean_iou'] for entry in history]
    
    # Plot training curves
    if vis_config.get('save_training_curves', True):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Loss
        axes[0].plot(epochs, train_losses, label='Train', marker='o')
        axes[0].plot(epochs, val_losses, label='Validation', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(epochs, train_acc, label='Train', marker='o')
        axes[1].plot(epochs, val_acc, label='Validation', marker='s')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Pixel Accuracy')
        axes[1].set_title('Pixel Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # IoU
        axes[2].plot(epochs, train_iou, label='Train', marker='o')
        axes[2].plot(epochs, val_iou, label='Validation', marker='s')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Mean IoU')
        axes[2].set_title('Mean IoU')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        curve_path = output_dir / 'training_curves.png'
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved training curves to {curve_path}")
    
    logger.info("U-Net visualization completed")
