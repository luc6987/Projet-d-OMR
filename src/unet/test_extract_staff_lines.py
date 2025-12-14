#!/usr/bin/env python3
"""
Visualization test script for staff line extraction
Tests the extract_staff_lines functionality with visualization
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import argparse
from PIL import Image

# Use absolute imports
from src.unet.extract_staff_lines import extract_and_group_staff_lines, StaffLine, StaffSystem
from src.unet.UNet import load_grayscale, infer_image_tiled
from src.unet.train_UNet import UNet, set_seed
import torch


def visualize_staff_extraction(
    original_image: np.ndarray,
    mask: np.ndarray,
    staff_lines: list,
    staff_systems: list,
    output_path: Path,
    show_symbols: bool = True,
    trim_info: tuple = None
):
    """
    Visualize staff line extraction results.
    
    Args:
        original_image: Original grayscale image
        mask: U-Net prediction mask
        staff_lines: List of StaffLine objects
        staff_systems: List of StaffSystem objects
        output_path: Path to save visualization
        show_symbols: Whether to show symbol regions from mask
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Original image with trim boundaries
    axes[0, 0].imshow(original_image, cmap='gray')
    if trim_info:
        left_offset, right_offset = trim_info
        # Draw vertical lines showing trim boundaries
        axes[0, 0].axvline(x=left_offset, color='cyan', linewidth=2, linestyle='--', alpha=0.8, label='Left trim')
        axes[0, 0].axvline(x=right_offset, color='cyan', linewidth=2, linestyle='--', alpha=0.8, label='Right trim')
        axes[0, 0].set_title(f'Original Image (Trimmed: {left_offset}-{right_offset})', fontsize=14, fontweight='bold')
    else:
        axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # 2. U-Net mask with staff lines highlighted
    mask_vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    unique_vals = np.unique(mask)
    if 255 in unique_vals and 127 not in unique_vals and 1 not in unique_vals:
        # Staff-only format
        mask_vis[mask == 255] = [255, 0, 0]  # Staff: red
    else:
        mask_vis[mask == 0] = [0, 0, 0]  # Background: black
        mask_vis[mask == 1] = [255, 0, 0]  # Staff: red
        if show_symbols:
            mask_vis[mask == 2] = [0, 255, 0]  # Symbols: green
        else:
            mask_vis[mask == 2] = [0, 0, 0]  # Symbols: black
        if 127 in unique_vals:
            mask_vis[mask == 127] = [255, 0, 0]  # Staff: red
        if 255 in unique_vals and 127 in unique_vals:
            mask_vis[mask == 255] = [0, 255, 0]  # Symbols: green
    
    axes[0, 1].imshow(mask_vis)
    title = 'U-Net Prediction Mask\n(Red=Staff'
    if show_symbols and (2 in unique_vals or (127 in unique_vals and 255 in unique_vals)):
        title += ', Green=Symbols'
    title += ')'
    axes[0, 1].set_title(title, fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Draw trim boundaries on mask
    if trim_info:
        left_offset, right_offset = trim_info
        axes[0, 1].axvline(x=left_offset, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
        axes[0, 1].axvline(x=right_offset, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
    
    # Draw detected lines on mask
    for line in staff_lines:
        axes[0, 1].axhline(y=line.y, color='yellow', linewidth=2, alpha=0.7)
    
    # 3. Original image with detected lines
    axes[1, 0].imshow(original_image, cmap='gray')
    axes[1, 0].set_title(f'Detected Staff Lines ({len(staff_lines)} lines)', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Draw trim boundaries
    if trim_info:
        left_offset, right_offset = trim_info
        axes[1, 0].axvline(x=left_offset, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
        axes[1, 0].axvline(x=right_offset, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
    
    # Draw all detected lines
    for line in staff_lines:
        axes[1, 0].axhline(y=line.y, color='red', linewidth=2, alpha=0.8, linestyle='--')
    
    # 4. Original image with grouped systems
    axes[1, 1].imshow(original_image, cmap='gray')
    axes[1, 1].set_title(f'Grouped Staff Systems ({len(staff_systems)} systems)', 
                         fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Draw trim boundaries
    if trim_info:
        left_offset, right_offset = trim_info
        axes[1, 1].axvline(x=left_offset, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
        axes[1, 1].axvline(x=right_offset, color='cyan', linewidth=2, linestyle='--', alpha=0.8)
    
    # Draw systems with different colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(staff_systems)))
    for i, system in enumerate(staff_systems):
        color = colors[i]
        for line_y in system.lines:
            axes[1, 1].axhline(y=line_y, color=color, linewidth=2.5, alpha=0.9)
        # Draw bounding box for system (use trimmed region if available)
        top = system.top_line - 10
        bottom = system.bottom_line + 10
        if trim_info:
            left = left_offset
            right = right_offset
        else:
            left = 0
            right = original_image.shape[1]
        rect = Rectangle((left, top), right - left, bottom - top, 
                        linewidth=2, edgecolor=color, facecolor='none', alpha=0.5)
        axes[1, 1].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] Saved to: {output_path}")


def print_staff_info(staff_lines: list, staff_systems: list):
    """Print information about detected staff lines and systems"""
    print("\n" + "="*60)
    print("Staff Line Extraction Results")
    print("="*60)
    print(f"\nTotal detected lines: {len(staff_lines)}")
    print(f"Total staff systems: {len(staff_systems)}")
    
    if staff_lines:
        print("\nIndividual Lines (y-coordinates):")
        for i, line in enumerate(staff_lines, 1):
            print(f"  Line {i:2d}: y = {line.y:4d}")
    
    if staff_systems:
        print("\nStaff Systems:")
        for i, system in enumerate(staff_systems, 1):
            print(f"\n  System {i}:")
            print(f"    Lines: {system.lines}")
            print(f"    Top line: {system.top_line}")
            print(f"    Bottom line: {system.bottom_line}")
            print(f"    Center y: {system.center_y:.2f}")
            print(f"    Avg spacing: {system.avg_spacing:.2f} pixels")
            print(f"    Number of lines: {len(system.lines)}")
    
    print("\n" + "="*60)


def test_from_mask_file(mask_path: Path, original_image_path: Path = None, output_dir: Path = None):
    """Test extraction from a saved mask file"""
    print(f"[Test] Loading mask from: {mask_path}")
    
    # Load mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not load mask: {mask_path}")
    
    # Load original image if provided
    if original_image_path and original_image_path.exists():
        original_image = load_grayscale(original_image_path)
        if original_image.dtype != np.uint8:
            original_image = (original_image * 255).astype(np.uint8) if original_image.max() <= 1.0 else original_image.astype(np.uint8)
    else:
        # Use mask as fallback - handle different formats
        original_image = np.zeros_like(mask)
        unique_vals = np.unique(mask)
        if 255 in unique_vals and 127 not in unique_vals and 1 not in unique_vals:
            # Staff-only format
            original_image[mask == 255] = 255
        elif 127 in unique_vals:
            # Visualization format
            original_image[mask == 127] = 255
            original_image[mask == 255] = 200
        else:
            # Standard format
            original_image[mask == 1] = 255
            original_image[mask == 2] = 200
    
    # Extract staff lines (auto-detect format and trim margins)
    staff_lines, staff_systems, trim_info = extract_and_group_staff_lines(
        mask, 
        trim_margins=True,
        auto_detect_format=True
    )
    
    # Print info
    print_staff_info(staff_lines, staff_systems)
    if trim_info:
        left_offset, right_offset = trim_info
        print(f"\nTrim info: Left={left_offset}, Right={right_offset}, Width={right_offset-left_offset+1}")
    
    # Visualize
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{mask_path.stem}_staff_extraction.png"
        visualize_staff_extraction(original_image, mask, staff_lines, staff_systems, output_path, trim_info=trim_info)
    
    return staff_lines, staff_systems


def test_from_inference(
    image_path: Path,
    checkpoint_path: Path,
    output_dir: Path = None,
    device: str = "auto",
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4
):
    """Test extraction by running inference first"""
    print(f"[Test] Running inference on: {image_path}")
    
    # Setup device
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_args = checkpoint.get("args", {})
    
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
    
    # Load image
    original_image = load_grayscale(image_path)
    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8) if original_image.max() <= 1.0 else original_image.astype(np.uint8)
    
    # Run inference
    print("[Test] Running tile-based inference...")
    from .UNet import infer_image_tiled
    pred_mask = infer_image_tiled(
        model=model,
        image=original_image,
        tile_size=tile_size,
        overlap=overlap,
        device=device,
        amp_enabled=False,
        batch_size=batch_size,
    )
    
    # Extract staff lines
    staff_lines, staff_systems = extract_and_group_staff_lines(pred_mask)
    
    # Print info
    print_staff_info(staff_lines, staff_systems)
    
    # Visualize
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_staff_extraction.png"
        visualize_staff_extraction(original_image, pred_mask, staff_lines, staff_systems, output_path)
    
    return staff_lines, staff_systems


def main():
    parser = argparse.ArgumentParser(
        description="Test staff line extraction with visualization"
    )
    parser.add_argument(
        "--mask", type=Path, default=None,
        help="Path to U-Net mask file (if provided, will test from mask directly)"
    )
    parser.add_argument(
        "--image", type=Path, default=None,
        help="Path to original image (required if using --mask, or for inference)"
    )
    parser.add_argument(
        "--checkpoint", type=Path, 
        default=Path("model/checkpoints/unet_staff_removal.pt"),
        help="Path to U-Net checkpoint (for inference mode)"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Output/unet_staff_test"),
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--tile-size", type=int, default=512,
        help="Tile size for inference"
    )
    parser.add_argument(
        "--overlap", type=int, default=64,
        help="Overlap for tile-based inference"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for inference"
    )
    
    args = parser.parse_args()
    
    set_seed(42)
    
    if args.mask:
        # Test from mask file
        test_from_mask_file(args.mask, args.image, args.output_dir)
    elif args.image:
        # Test from inference
        if not args.checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
        test_from_inference(
            args.image,
            args.checkpoint,
            args.output_dir,
            args.device,
            args.tile_size,
            args.overlap,
            args.batch_size
        )
    else:
        parser.print_help()
        print("\n[Error] Please provide either --mask or --image")


if __name__ == "__main__":
    main()

