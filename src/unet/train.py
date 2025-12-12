"""
U-Net training module
Refactored to use configuration directly
"""
import sys
from pathlib import Path
from datetime import datetime

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.config_loader import ConfigLoader
from common.logger import OMRLogger
from common.paths import PathManager

# Import training functions from train_UNet
from .train_UNet import (
    set_seed, collect_samples, split_samples, StaffLineSegmentationDataset,
    UNet, create_optimizer, create_scheduler, train_one_epoch, evaluate,
    compute_mean_iou, EpochMetrics, HistoryEntry, get_learning_rate
)
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import json
from dataclasses import asdict

try:
    from torch.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = False


def train_unet(config: ConfigLoader, path_manager: PathManager) -> None:
    """
    Train U-Net model using configuration
    
    Args:
        config: Configuration loader
        path_manager: Path manager
    """
    logger = OMRLogger.get_logger('unet')
    
    # Get module config
    unet_config = config.get_module_config('unet')
    train_config = unet_config.get('train', {})
    
    if not train_config.get('enabled', True):
        logger.info("U-Net training is disabled in config")
        return
    
    logger.info("Starting U-Net training...")
    
    # Extract configuration values
    data_root = path_manager.resolve_path(train_config.get('data_root', 'data/v1.0/data/images'))
    image_subdir = train_config.get('image_subdir', 'image')
    staff_subdir = train_config.get('staff_subdir', 'gt')
    symbol_subdir = train_config.get('symbol_subdir', 'symbol')
    extensions = tuple(train_config.get('extensions', ['.png', '.jpg', '.jpeg']))
    
    img_size = train_config.get('img_size', 512)
    batch_size = train_config.get('batch_size', 4)
    epochs = train_config.get('epochs', 100)
    val_ratio = train_config.get('val_ratio', 0.1)
    seed = config.global_config.get('seed', 42)
    device_str = config.global_config.get('device', 'auto')
    
    # Setup device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    logger.info(f"Data root: {data_root}")
    logger.info(f"Epochs: {epochs}, Batch size: {batch_size}, Device: {device}")
    
    # Setup paths
    checkpoint_path = path_manager.resolve_path(train_config.get('checkpoint_path', 'model/checkpoints/unet_staff_removal.pt'))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_json = train_config.get('log_json')
    if log_json is None:
        log_dir = path_manager.resolve_path(config.global_config.get('log_dir', 'logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_json = log_dir / f"unet_training_{timestamp}.json"
    else:
        log_json = path_manager.resolve_path(log_json)
        log_json.parent.mkdir(parents=True, exist_ok=True)
    
    set_seed(seed)
    
    # Collect samples
    samples = collect_samples(
        data_root=data_root,
        image_subdir=image_subdir,
        staff_subdir=staff_subdir,
        symbol_subdir=symbol_subdir,
        extensions=extensions,
    )
    
    train_samples, val_samples = split_samples(samples, val_ratio, seed)
    if not train_samples:
        raise RuntimeError("Training sample list is empty after split.")
    if not val_samples:
        raise RuntimeError("Validation sample list is empty after split.")
    
    img_size_tuple = (img_size, img_size)
    
    # Create datasets
    train_dataset = StaffLineSegmentationDataset(
        train_samples,
        img_size=img_size_tuple,
        augment=train_config.get('augment', True),
        staff_threshold=train_config.get('staff_threshold', 128),
        symbol_threshold=train_config.get('symbol_threshold', 128),
        max_rotation_deg=train_config.get('max_rotation', 2.5),
    )
    val_dataset = StaffLineSegmentationDataset(
        val_samples,
        img_size=img_size_tuple,
        augment=False,
        staff_threshold=train_config.get('staff_threshold', 128),
        symbol_threshold=train_config.get('symbol_threshold', 128),
        max_rotation_deg=0.0,
    )
    
    pin_memory = device.type == "cuda"
    amp_enabled = train_config.get('amp', False) and device.type == "cuda"
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_config.get('num_workers', 4),
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_config.get('num_workers', 4),
        pin_memory=pin_memory,
    )
    
    # Create model
    num_classes = train_config.get('num_classes', 3)
    model = UNet(
        n_channels=1,
        n_classes=num_classes,
        base_channels=train_config.get('base_channels', 64),
        bilinear=not train_config.get('use_transposed_conv', False),
        dropout=train_config.get('dropout', 0.0),
    ).to(device)
    
    # Setup loss
    class_weights_str = train_config.get('class_weights', '')
    if class_weights_str:
        weights = [float(w.strip()) for w in class_weights_str.split(",") if w.strip()]
        if len(weights) != num_classes:
            raise ValueError("Number of class weights must match num_classes.")
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        class_weights = None
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Create optimizer (using a simple args-like object)
    class OptimizerArgs:
        def __init__(self, config_dict):
            self.optimizer = config_dict.get('optimizer', 'adamw')
            self.lr = config_dict.get('lr', 0.001)
            self.weight_decay = config_dict.get('weight_decay', 0.0001)
            self.beta1 = config_dict.get('beta1', 0.9)
            self.beta2 = config_dict.get('beta2', 0.999)
            self.momentum = config_dict.get('momentum', 0.9)
            self.nesterov = config_dict.get('nesterov', False)
    
    optimizer_args = OptimizerArgs(train_config)
    optimizer = create_optimizer(model, optimizer_args)
    
    # Create scheduler
    class SchedulerArgs:
        def __init__(self, config_dict):
            self.lr_scheduler = config_dict.get('lr_scheduler', 'plateau')
            self.epochs = config_dict.get('epochs', 100)
            self.min_lr = config_dict.get('min_lr', 0.000001)
            self.scheduler_metric = config_dict.get('scheduler_metric', 'iou')
            self.lr_gamma = config_dict.get('lr_gamma', 0.5)
            self.lr_patience = config_dict.get('lr_patience', 5)
    
    scheduler_args = SchedulerArgs(train_config)
    scheduler = create_scheduler(optimizer, scheduler_args)
    
    if AMP_AVAILABLE:
        scaler = GradScaler('cuda', enabled=amp_enabled)
    else:
        scaler = GradScaler(enabled=amp_enabled)
    
    # Training loop
    start_epoch = 1
    best_score = -float("inf") if scheduler_args.scheduler_metric == "iou" else float("inf")
    epochs_without_improvement = 0
    selection_metric = scheduler_args.scheduler_metric
    
    history: list[HistoryEntry] = []
    grad_clip = train_config.get('grad_clip', 0.0)
    early_stop = train_config.get('early_stop', 0)
    
    for epoch in range(start_epoch, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            num_classes=num_classes,
            amp_enabled=amp_enabled,
            grad_clip=grad_clip,
        )
        
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )
        
        current_lr = get_learning_rate(optimizer)
        history.append(
            HistoryEntry(
                epoch=epoch,
                train_loss=train_metrics.loss,
                train_pixel_acc=train_metrics.pixel_accuracy,
                train_mean_iou=train_metrics.mean_iou,
                train_iou_per_class=train_metrics.iou_per_class,
                val_loss=val_metrics.loss,
                val_pixel_acc=val_metrics.pixel_accuracy,
                val_mean_iou=val_metrics.mean_iou,
                val_iou_per_class=val_metrics.iou_per_class,
                learning_rate=current_lr,
            )
        )
        
        if scheduler is not None:
            if scheduler_args.lr_scheduler == "plateau":
                monitor_value = (
                    val_metrics.mean_iou if scheduler_args.scheduler_metric == "iou" else val_metrics.loss
                )
                scheduler.step(monitor_value)
            else:
                scheduler.step()
        
        logger.info(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_metrics.loss:.4f} | "
            f"val_loss={val_metrics.loss:.4f} | "
            f"train_acc={train_metrics.pixel_accuracy:.4f} | "
            f"val_acc={val_metrics.pixel_accuracy:.4f} | "
            f"val_mIoU={val_metrics.mean_iou:.4f} | "
            f"lr={current_lr:.6f}"
        )
        
        monitor_metric = val_metrics.mean_iou if selection_metric == "iou" else val_metrics.loss
        improved = (
            monitor_metric > best_score if selection_metric == "iou" else monitor_metric < best_score
        )
        
        if improved:
            best_score = monitor_metric
            epochs_without_improvement = 0
            
            checkpoint_data = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "args": train_config,
                "best_score": best_score,
                "train_metrics": train_metrics.__dict__,
                "val_metrics": val_metrics.__dict__,
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Saved improved checkpoint to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
        
        if early_stop > 0 and epochs_without_improvement >= early_stop:
            logger.info(f"Early stopping triggered after {epoch} epochs.")
            break
    
    # Save training log
    best_epoch_idx = None
    if history:
        if selection_metric == "iou":
            best_epoch_idx = max(range(len(history)), key=lambda i: history[i].val_mean_iou)
        else:
            best_epoch_idx = min(range(len(history)), key=lambda i: history[i].val_loss)
    
    log_data = {
        "training_config": {
            "data_root": str(data_root),
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "base_channels": train_config.get('base_channels', 64),
            "learning_rate": train_config.get('lr', 0.001),
            "optimizer": train_config.get('optimizer', 'adamw'),
            "lr_scheduler": train_config.get('lr_scheduler', 'plateau'),
            "val_ratio": val_ratio,
            "num_classes": num_classes,
            "device": str(device),
            "amp_enabled": amp_enabled,
        },
        "history": [asdict(entry) for entry in history],
        "final_results": {
            "best_epoch": best_epoch_idx + 1 if best_epoch_idx is not None else None,
            "best_val_iou": best_score if selection_metric == "iou" else None,
            "best_val_loss": best_score if selection_metric == "loss" else None,
            "final_train_loss": history[-1].train_loss if history else None,
            "final_val_loss": history[-1].val_loss if history else None,
            "final_train_iou": history[-1].train_mean_iou if history else None,
            "final_val_iou": history[-1].val_mean_iou if history else None,
        },
        "checkpoint": str(checkpoint_path),
        "selection_metric": selection_metric,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(log_json, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    logger.info(f"Training log saved to {log_json}")
    
    logger.info("U-Net training completed")
