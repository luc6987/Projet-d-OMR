"""
U-Net inference module
Refactored to use configuration directly
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.config_loader import ConfigLoader
from common.logger import OMRLogger
from common.paths import PathManager

# Import inference functions from UNet.py
from .UNet import (
    gather_image_samples, infer_image_tiled, apply_staff_removal,
    mask_to_visual, create_output_paths, load_grayscale, set_seed
)
from .train_UNet import UNet
import torch
import numpy as np
from PIL import Image


def infer_unet(config: ConfigLoader, path_manager: PathManager) -> None:
    """Run U-Net inference"""
    logger = OMRLogger.get_logger('unet')
    
    unet_config = config.get_module_config('unet')
    infer_config = unet_config.get('infer', {})
    
    if not infer_config.get('enabled', True):
        logger.info("U-Net inference is disabled in config")
        return
    
    logger.info("Starting U-Net inference...")
    
    # Extract configuration
    data_root = path_manager.resolve_path(infer_config.get('data_root', 'data/v1.0/data/images'))
    image_subdir = infer_config.get('image_subdir', 'image')
    extensions = infer_config.get('extensions', ['.png', '.jpg', '.jpeg'])
    checkpoint_path = path_manager.resolve_path(infer_config.get('checkpoint', 'model/checkpoints/unet_staff_removal.pt'))
    output_dir = path_manager.resolve_path(infer_config.get('output_dir', 'Output/UNet'))
    tile_size = infer_config.get('tile_size', 512)
    overlap = infer_config.get('overlap', 64)
    batch_size = infer_config.get('batch_size', 4)
    device_str = config.global_config.get('device', 'auto')
    seed = config.global_config.get('seed', 42)
    
    save_mask = infer_config.get('save_mask', False)
    save_clean = infer_config.get('save_clean', True)
    save_overlay = infer_config.get('save_overlay', False)
    
    if not save_mask and not save_clean and not save_overlay:
        raise ValueError("At least one of mask, clean, or overlay outputs must be enabled.")
    
    # Setup device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    amp_enabled = infer_config.get('amp', False) and device.type == "cuda"
    
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Tile size: {tile_size}, Overlap: {overlap}, Batch size: {batch_size}")
    
    # Collect samples
    samples = gather_image_samples(
        data_root=data_root,
        image_subdir=image_subdir,
        extensions=extensions,
    )
    
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if this is an old model trained with incorrect binarize logic
    checkpoint_args = checkpoint.get("args", {})
    if "binarize_fixed" not in checkpoint_args:
        logger.warning("⚠️  这个模型可能是在修复 binarize 逻辑之前训练的！")
        logger.warning("⚠️  如果输出结果不正确，请使用修复后的代码重新训练模型。")
        logger.warning("⚠️  建议使用 save_mask 查看预测的 mask 分布来诊断问题。")
    
    # Create model
    model = UNet(
        n_channels=1,
        n_classes=3,
        base_channels=checkpoint_args.get("base_channels", 64),
        bilinear=not checkpoint_args.get("use_transposed_conv", False),
        dropout=checkpoint_args.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    
    logger.info(f"Processing {len(samples)} images with tile-based inference...")
    
    for sample in samples:
        logger.info(f"Processing: {sample.image_path}")
        
        # Load original image
        original_image = load_grayscale(sample.image_path)
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8) if original_image.max() <= 1.0 else original_image.astype(np.uint8)
        
        # Run tile-based inference
        pred_mask = infer_image_tiled(
            model=model,
            image=original_image,
            tile_size=tile_size,
            overlap=overlap,
            device=device,
            amp_enabled=amp_enabled,
            batch_size=batch_size,
        )
        
        # Debug: Print prediction statistics
        unique, counts = np.unique(pred_mask, return_counts=True)
        total = pred_mask.size
        logger.debug(f"Predicted mask distribution:")
        class_names = {0: "背景", 1: "五线", 2: "音符"}
        for val, cnt in zip(unique, counts):
            logger.debug(f"  class {val} ({class_names.get(val, '未知'):4s}): {cnt:8d} 像素 ({cnt*100/total:5.2f}%)")
        
        # Apply staff removal
        clean_image = apply_staff_removal(original_image, pred_mask)
        
        # Create outputs
        mask_vis = mask_to_visual(pred_mask)
        
        # If only saving clean images, use original filename (no _clean suffix)
        only_clean = save_clean and not save_mask and not save_overlay
        
        mask_path, clean_path, overlay_path = create_output_paths(
            output_dir,
            sample.relative_dir.as_posix() if sample.relative_dir != Path(".") else "",
            sample.image_path.stem,
            only_clean=only_clean,
        )
        
        if save_mask:
            Image.fromarray(mask_vis).save(mask_path)
            logger.debug(f"Saved mask: {mask_path}")
        
        if save_clean:
            Image.fromarray(clean_image).save(clean_path)
            logger.debug(f"Saved clean image: {clean_path}")
        
        if save_overlay:
            overlay = create_overlay(original_image, pred_mask)
            Image.fromarray(overlay).save(overlay_path)
            logger.debug(f"Saved overlay: {overlay_path}")
    
    logger.info("U-Net inference completed")


def create_overlay(original_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create pseudo-color overlay for visualization"""
    from .UNet import create_overlay as _create_overlay
    return _create_overlay(original_image, mask)
