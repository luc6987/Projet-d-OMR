#!/usr/bin/env python3
"""
Apply a trained U-Net staff-removal model to sheet music images using tile-based inference
to preserve original resolution.

This script loads the checkpoint produced by `train_UNet.py`, runs inference
over a directory of images using a sliding window (tile-based) approach,
and writes the staff-removed images (symbols only) into `Output/UNet`.

Input format: Black background (0) with white staff lines and symbols (255).
Output format: Black background (0) with white symbols only (staff lines removed).

By default, only outputs the cleaned images (`{stem}_clean.png`).
Optional outputs (use flags to enable):
    - `{stem}_mask.png`: 3-class mask encoded as grayscale (0=background,
      127=staff, 255=symbols) - use --save-mask
    - `{stem}_overlay.png`: pseudo-color overlay for visual inspection - use --save-overlay

Usage example:
    python src/UNet.py \
        --checkpoint model/checkpoints/unet_staff_removal.pt \
        --data-root /users/.../v1.0/data/images \
        --output-dir Output/UNet \
        --tile-size 512 \
        --overlap 64
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .train_UNet import UNet, load_grayscale, set_seed


# --------------------------- Tile-based inference --------------------------- #


def extract_tiles(
    image: np.ndarray,
    tile_size: int,
    overlap: int,
) -> List[Tuple[int, int, np.ndarray]]:
    """
    Extract overlapping tiles from an image.
    
    Returns:
        List of (y_start, x_start, tile_image) tuples
    """
    h, w = image.shape[:2]
    stride = tile_size - overlap
    tiles = []
    
    y = 0
    while y < h:
        x = 0
        while x < w:
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            tile = image[y:y_end, x:x_end]
            
            # Pad if tile is smaller than tile_size
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size), dtype=image.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded
            
            tiles.append((y, x, tile))
            x += stride
            if x >= w:
                break
        y += stride
        if y >= h:
            break
    
    return tiles


def merge_tiles(
    tiles: List[Tuple[int, int, np.ndarray]],
    original_shape: Tuple[int, int],
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    """
    Merge overlapping tiles back into a full image using weighted averaging.
    """
    h, w = original_shape
    result = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    stride = tile_size - overlap
    
    for y_start, x_start, tile in tiles:
        y_end = min(y_start + tile_size, h)
        x_end = min(x_start + tile_size, w)
        
        tile_h = y_end - y_start
        tile_w = x_end - x_start
        
        # Create weight mask (higher weight in center, lower at edges)
        weight = np.ones((tile_size, tile_size), dtype=np.float32)
        if overlap > 0:
            # Linear falloff at edges
            for i in range(overlap):
                alpha = (i + 1) / (overlap + 1)
                weight[i, :] *= alpha
                weight[-1-i, :] *= alpha
                weight[:, i] *= alpha
                weight[:, -1-i] *= alpha
        
        # Accumulate weighted predictions
        result[y_start:y_end, x_start:x_end] += tile[:tile_h, :tile_w] * weight[:tile_h, :tile_w]
        weight_map[y_start:y_end, x_start:x_end] += weight[:tile_h, :tile_w]
    
    # Normalize by weight sum
    weight_map = np.maximum(weight_map, 1e-8)  # Avoid division by zero
    result = result / weight_map
    
    return result.astype(np.uint8)


# --------------------------- Inference helpers --------------------------- #


def create_output_paths(
    base_dir: Path, 
    rel_dir: str, 
    stem: str, 
    only_clean: bool = False
) -> Tuple[Path, Path, Path]:
    target_dir = base_dir if not rel_dir else base_dir / rel_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    mask_path = target_dir / f"{stem}_mask.png"
    # If only saving clean images, use original filename without suffix
    if only_clean:
        clean_path = target_dir / f"{stem}.png"
    else:
        clean_path = target_dir / f"{stem}_clean.png"
    overlay_path = target_dir / f"{stem}_overlay.png"
    return mask_path, clean_path, overlay_path


def mask_to_visual(mask: np.ndarray) -> np.ndarray:
    """Convert mask to visualization format."""
    vis = np.zeros_like(mask, dtype=np.uint8)
    vis[mask == 1] = 127  # staff in gray
    vis[mask == 2] = 255  # symbols in white
    return vis


def mask_to_staff_only(mask: np.ndarray) -> np.ndarray:
    """Convert mask to staff-only format (only staff lines, no symbols)."""
    vis = np.zeros_like(mask, dtype=np.uint8)
    vis[mask == 1] = 255  # staff in white
    return vis


def build_overlay(clean_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Build pseudo-color overlay for visualization."""
    overlay = np.stack([clean_image] * 3, axis=-1).astype(np.uint8)
    overlay[mask == 1] = np.array([255, 0, 0], dtype=np.uint8)    # staff in red
    overlay[mask == 2] = np.array([0, 200, 0], dtype=np.uint8)    # symbols in green
    return overlay


def apply_staff_removal(
    original_image: np.ndarray,
    mask: np.ndarray,
    inpainting_radius: int = 3,
) -> np.ndarray:
    """
    Apply staff removal: fill staff pixels (class 1) with local background color using inpainting.
    
    This prevents "black hole" artifacts on scanned images with non-pure-black backgrounds
    by filling staff pixels with nearby background values instead of forcing them to 0.
    
    Args:
        original_image: Original grayscale image (0-255, black background with white content)
        mask: Predicted mask (0=background, 1=staff, 2=symbols)
        inpainting_radius: Radius (in pixels) to search for nearby background pixels (default: 3)
    
    Returns:
        Cleaned image with staff lines removed (background color filled, white symbols only)
    """
    clean_image = original_image.copy().astype(np.float32)
    staff_mask = (mask == 1)
    
    # If no staff pixels to remove, return original
    if not np.any(staff_mask):
        return clean_image.astype(np.uint8)
    
    # Find background pixels (class 0, excluding staff and symbols)
    background_mask = (mask == 0)
    
    if SCIPY_AVAILABLE:
        # Use morphological dilation to find nearby background pixels
        # Dilate staff mask to find pixels near staff lines
        dilated_staff = ndimage.binary_dilation(staff_mask, structure=np.ones((inpainting_radius*2+1, inpainting_radius*2+1)))
        
        # Find background pixels near staff lines
        nearby_background = dilated_staff & background_mask
        
        if np.any(nearby_background):
            # Compute local background value (median of nearby background pixels)
            # Use a distance-weighted approach: for each staff pixel, find nearby background
            # and use median value
            background_values = clean_image[nearby_background]
            if len(background_values) > 0:
                local_background = np.median(background_values)
            else:
                # Fallback: use global background median
                local_background = np.median(clean_image[background_mask]) if np.any(background_mask) else 0.0
        else:
            # No nearby background found, use global background median
            local_background = np.median(clean_image[background_mask]) if np.any(background_mask) else 0.0
        
        # Fill staff pixels with local background value
        clean_image[staff_mask] = local_background
    else:
        # Fallback: simpler approach without scipy
        # Use median of all background pixels as fill value
        if np.any(background_mask):
            background_median = np.median(clean_image[background_mask])
            clean_image[staff_mask] = background_median
        else:
            # No background pixels found, set to 0 (original behavior)
            clean_image[staff_mask] = 0.0
    
    return np.clip(clean_image, 0, 255).astype(np.uint8)


def infer_image_tiled(
    model: torch.nn.Module,
    image: np.ndarray,
    tile_size: int,
    overlap: int,
    device: torch.device,
    amp_enabled: bool,
    batch_size: int = 4,
) -> np.ndarray:
    """
    Run tile-based inference on a full-resolution image.
    
    Returns:
        Predicted mask (0=background, 1=staff, 2=symbols)
    """
    h, w = image.shape[:2]
    
    # Extract tiles
    tiles = extract_tiles(image, tile_size, overlap)
    
    # Process tiles in batches
    all_predictions = []
    tile_positions = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i:i + batch_size]
            batch_tensors = []
            batch_positions = []
            
            for y_start, x_start, tile in batch_tiles:
                # Normalize to [0, 1] (same as training)
                tile_tensor = torch.from_numpy(tile).unsqueeze(0).float() / 255.0
                batch_tensors.append(tile_tensor)
                batch_positions.append((y_start, x_start))
            
            # Stack into batch
            batch_input = torch.stack(batch_tensors).to(device)
            
            # Inference
            if amp_enabled:
                try:
                    # Use new torch.amp API if available
                    with torch.amp.autocast('cuda', enabled=True):
                        logits = model(batch_input)
                except (AttributeError, ImportError):
                    # Fallback to old API
                    with torch.cuda.amp.autocast(enabled=True):
                        logits = model(batch_input)
            else:
                logits = model(batch_input)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Store predictions with positions
            for j, (y_start, x_start) in enumerate(batch_positions):
                pred_tile = preds[j]
                y_end = min(y_start + tile_size, h)
                x_end = min(x_start + tile_size, w)
                pred_cropped = pred_tile[:y_end - y_start, :x_end - x_start]
                all_predictions.append((y_start, x_start, pred_cropped))
    
    # Merge predictions using weighted averaging
    # First, create full-size prediction arrays for each class
    h, w = image.shape[:2]
    class_votes = np.zeros((3, h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    stride = tile_size - overlap
    for y_start, x_start, pred_tile in all_predictions:
        y_end = min(y_start + tile_size, h)
        x_end = min(x_start + tile_size, w)
        tile_h = y_end - y_start
        tile_w = x_end - x_start
        
        # Weight mask (higher in center, lower at edges)
        weight = np.ones((tile_h, tile_w), dtype=np.float32)
        if overlap > 0:
            for i in range(min(overlap, tile_h // 2)):
                alpha = (i + 1) / (overlap + 1)
                if i < tile_h:
                    weight[i, :] *= alpha
                if tile_h - 1 - i >= 0:
                    weight[tile_h - 1 - i, :] *= alpha
            for i in range(min(overlap, tile_w // 2)):
                alpha = (i + 1) / (overlap + 1)
                if i < tile_w:
                    weight[:, i] *= alpha
                if tile_w - 1 - i >= 0:
                    weight[:, tile_w - 1 - i] *= alpha
        
        # Accumulate votes for each class
        for cls in range(3):
            class_mask = (pred_tile == cls).astype(np.float32)
            class_votes[cls, y_start:y_end, x_start:x_end] += class_mask * weight
        
        # Accumulate weights
        weight_map[y_start:y_end, x_start:x_end] += weight
    
    # Normalize and get final prediction (argmax of weighted votes)
    weight_map = np.maximum(weight_map, 1e-8)
    for cls in range(3):
        class_votes[cls] /= weight_map
    
    result_mask = np.argmax(class_votes, axis=0).astype(np.uint8)
    
    return result_mask


# --------------------------- Data utilities --------------------------- #


@dataclass
class InferenceSample:
    image_path: Path
    relative_dir: Path


def gather_image_samples(
    data_root: Path,
    image_subdir: str,
    extensions: Sequence[str],
) -> List[InferenceSample]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    # Normalize extensions
    ext_list = []
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        ext_list.append(ext)
    
    if not ext_list:
        raise ValueError("No valid file extensions supplied.")
    
    extensions = tuple(sorted(set(ext_list)))
    samples: List[InferenceSample] = []

    def collect_from_directory(directory: Path, rel_dir: Path) -> None:
        for img_path in sorted(directory.glob("*")):
            if img_path.is_file() and img_path.suffix.lower() in extensions:
                samples.append(InferenceSample(image_path=img_path, relative_dir=rel_dir))

    # Case 1: direct images under data_root
    collect_from_directory(data_root, Path("."))

    # Case 2: grouped folders (e.g., w-01/image/)
    for group_dir in sorted([p for p in data_root.iterdir() if p.is_dir()]):
        image_dir = group_dir / image_subdir
        if image_dir.is_dir():
            rel_dir = group_dir.relative_to(data_root)
            collect_from_directory(image_dir, rel_dir)
        else:
            # Also consider that the group itself might directly hold images
            collect_from_directory(group_dir, group_dir.relative_to(data_root))

    if not samples:
        raise RuntimeError(
            f"No images found under {data_root} (looked for extensions: {extensions})"
        )

    return samples


# --------------------------- CLI & main -------------------------------- #


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    # Use relative path as default, user must provide --data-root if it doesn't exist
    relative_data_root = project_root / "data" / "v1.0" / "data" / "images"
    default_data_root = relative_data_root
    
    default_checkpoint = project_root / "model" / "checkpoints" / "unet_staff_removal.pt"
    default_output = project_root / "Output" / "UNet"

    parser = argparse.ArgumentParser(
        description="Run tile-based inference with a trained U-Net staff removal model."
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root, help="Root directory containing image groups.")
    parser.add_argument("--image-subdir", type=str, default="image", help="Subdirectory name that stores original images.")
    parser.add_argument("--extensions", type=str, default=".png,.jpg,.jpeg", help="Comma-separated list of valid image extensions.")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint, help="Path to trained U-Net checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="Directory to save inference outputs.")
    parser.add_argument("--tile-size", type=int, default=512, help="Size of each tile for inference (square).")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap between tiles (pixels).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for processing tiles.")
    parser.add_argument("--device", type=str, default="auto", help="Device spec, e.g., cuda, cuda:0, cpu. Use 'auto' to select automatically.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision during inference.")
    parser.add_argument("--save-mask", action="store_true", help="Save predicted mask images (disabled by default, but recommended for debugging).")
    parser.add_argument("--no-clean", action="store_true", help="Disable saving staff-removed images (enabled by default).")
    parser.add_argument("--save-overlay", action="store_true", help="Save pseudo-color overlays for qualitative review (disabled by default).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic behavior.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Default: only save clean images (staff-removed)
    # Mask and overlay are optional and disabled by default
    args.save_clean = not args.no_clean  # Enabled by default unless --no-clean is used

    if not args.save_mask and not args.save_clean and not args.save_overlay:
        raise ValueError("At least one of mask, clean, or overlay outputs must be enabled.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    extensions = [ext for ext in args.extensions.split(",") if ext.strip()]
    samples = gather_image_samples(
        data_root=args.data_root,
        image_subdir=args.image_subdir,
        extensions=extensions,
    )

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Check if this is an old model trained with incorrect binarize logic
    checkpoint_args = checkpoint.get("args", {})
    if "binarize_fixed" not in checkpoint_args:
        print("[WARNING] ⚠️  这个模型可能是在修复 binarize 逻辑之前训练的！")
        print("[WARNING] ⚠️  如果输出结果不正确，请使用修复后的代码重新训练模型。")
        print("[WARNING] ⚠️  建议使用 --save-mask 查看预测的 mask 分布来诊断问题。")
    
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

    amp_enabled = args.amp and device.type == "cuda"
    total_samples = 0

    print(f"[INFO] Processing {len(samples)} images with tile-based inference...")
    print(f"[INFO] Tile size: {args.tile_size}, Overlap: {args.overlap}")

    for sample in samples:
        print(f"[INFO] Processing: {sample.image_path}")
        
        # Load original image
        original_image = load_grayscale(sample.image_path)
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8) if original_image.max() <= 1.0 else original_image.astype(np.uint8)
        
        # Run tile-based inference
        pred_mask = infer_image_tiled(
            model=model,
            image=original_image,
            tile_size=args.tile_size,
            overlap=args.overlap,
            device=device,
            amp_enabled=amp_enabled,
            batch_size=args.batch_size,
        )
        
        # Debug: Print prediction statistics
        unique, counts = np.unique(pred_mask, return_counts=True)
        total = pred_mask.size
        print(f"   [DEBUG] Predicted mask distribution:")
        class_names = {0: "背景", 1: "五线", 2: "音符"}
        for val, cnt in zip(unique, counts):
            print(f"      class {val} ({class_names.get(val, '未知'):4s}): {cnt:8d} 像素 ({cnt*100/total:5.2f}%)")
        
        # Apply staff removal
        clean_image = apply_staff_removal(original_image, pred_mask)
        
        # Create outputs
        mask_vis = mask_to_visual(pred_mask)
        
        # If only saving clean images, use original filename (no _clean suffix)
        only_clean = args.save_clean and not args.save_mask and not args.save_overlay
        
        mask_path, clean_path, overlay_path = create_output_paths(
            args.output_dir,
            sample.relative_dir.as_posix() if sample.relative_dir != Path(".") else "",
            sample.image_path.stem,
            only_clean=only_clean,
        )

        if args.save_mask:
            Image.fromarray(mask_vis).save(mask_path)
        if args.save_clean:
            Image.fromarray(clean_image).save(clean_path)
        if args.save_overlay:
            overlay_np = build_overlay(clean_image, pred_mask)
            Image.fromarray(overlay_np).save(overlay_path)

        total_samples += 1

    print(f"[INFO] Inference completed. Processed {total_samples} images.")
    print(f"[INFO] Outputs written under: {args.output_dir}")


if __name__ == "__main__":
    main()
