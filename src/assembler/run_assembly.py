#!/usr/bin/env python3
"""
OMR Assembly Pipeline Entry Point
Combines U-Net mask (staff lines) and YOLO detections (symbols) into MusicXML.
"""

import argparse
from pathlib import Path
import sys

# Ensure src is in path if running from root
sys.path.append(str(Path(__file__).parent.parent.parent))

from .staff import StaffSystemDetector
from .symbols import SymbolLoader
from .builder import ScoreBuilder
from .exporter import MusicXMLExporter
from .linker import Linker
from .visualizer import AssemblyVisualizer
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Assemble OMR results into MusicXML")
    
    parser.add_argument("--json", type=str, required=True,
                        help="Path to YOLO detection results JSON")
    parser.add_argument("--mask", type=str, required=True,
                        help="Path to U-Net staff mask (binary or grayscale)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for MusicXML file (.xml or .mxl)")
    parser.add_argument("--model", type=str, 
                        default="Output/assemblage/outputs/fourth_training/model_best.pth",
                        help="Path to trained MLP linker model")
    parser.add_argument("--title", type=str, default=None,
                        help="Title for the score (optional)")
    parser.add_argument("--composer", type=str, default=None,
                        help="Composer name (optional)")
    parser.add_argument("--max-parts", type=int, default=None,
                        help="Maximum number of parts to output (default: all detected)")
    parser.add_argument("--min-symbols", type=int, default=10,
                        help="Minimum symbols per part (default: 10)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualization images in output directory")
    parser.add_argument("--original-image", type=str, default=None,
                        help="Path to original image (for visualization). If not provided, inferred from mask path.")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    json_path = Path(args.json)
    mask_path = Path(args.mask)
    output_path = Path(args.output)
    model_path = Path(args.model)
    
    print("=== OMR Assembly Pipeline Started ===")
    
    # 1. Detect Staff Lines
    print(f"\nStep 1: Detect Staff Systems from {mask_path}")
    staff_detector = StaffSystemDetector(mask_path)
    staff_systems = staff_detector.detect_staff_lines()
    
    if not staff_systems:
        print("[Error] No staff systems detected! Cannot proceed.")
        return
    
    # Get image shape for normalization (H, W)
    # staff_detector.mask is loaded after detect_staff_lines
    image_shape = staff_detector.mask.shape[:2]
        
    # 2. Load Symbols
    print(f"\nStep 2: Load Symbols from {json_path}")
    symbol_loader = SymbolLoader(json_path)
    symbols = symbol_loader.load()
    
    if not symbols:
        print("[Warning] No symbols loaded. Output will be empty.")

    # 3. Initialize Linker (MLP)
    linker = None
    if model_path.exists():
        print(f"\nStep 3: Initialize MLP Linker from {model_path}")
        try:
            linker = Linker(model_path)
            print("Linker loaded successfully.")
        except Exception as e:
            print(f"[Warning] Failed to load Linker: {e}")
    else:
        print(f"\n[Warning] Linker model not found at {model_path}. Duration estimation will be basic.")
        
    # 4. Build Score
    print("\nStep 4: Assemble Score")
    # Pass image_shape and linker
    builder = ScoreBuilder(staff_systems, symbols, image_shape, linker)
    parts, linked_pairs = builder.build(min_symbols_per_part=args.min_symbols, max_parts=args.max_parts)
    
    print(f"Assembled {len(parts)} parts (staves).")
    print(f"Found {len(linked_pairs)} symbol links.")
    
    # 5. Export MusicXML
    print(f"\nStep 5: Export to {output_path}")
    exporter = MusicXMLExporter()
    # Extract filename for default title if not provided
    default_title = output_path.stem if not args.title else None
    exporter.export(parts, output_path, 
                   title=args.title or default_title,
                   composer=args.composer)
    
    # 6. Generate Visualizations (if requested)
    if args.visualize:
        print("\nStep 6: Generate Visualizations")
        vis_dir = output_path.parent / f"{output_path.stem}_visualization"
        visualizer = AssemblyVisualizer(vis_dir)
        
        # Find original image path
        if args.original_image:
            original_image_path = Path(args.original_image)
        else:
            # Try to infer from mask path (e.g., mask is in UNet/w-01/p001_mask.png, original is p001.png)
            # Look for corresponding .png file in the same directory
            mask_dir = mask_path.parent
            mask_stem = mask_path.stem.replace('_mask', '').replace('_clean', '')
            original_image_path = mask_dir / f"{mask_stem}.png"
            if not original_image_path.exists():
                # Try parent directory
                original_image_path = mask_dir.parent / f"{mask_stem}.png"
        
        # Load original image
        if original_image_path.exists():
            original_image = cv2.imread(str(original_image_path), cv2.IMREAD_GRAYSCALE)
            if original_image is None:
                original_image = cv2.imread(str(original_image_path), cv2.IMREAD_COLOR)
                if original_image is not None:
                    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        else:
            # Use mask as fallback (convert to grayscale if needed)
            original_image = staff_detector.mask.copy()
            if len(original_image.shape) == 3:
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            original_image_path = mask_path
        
        # 1. Save original image
        visualizer.save_original_image(original_image_path)
        
        # 2. Save U-Net mask and cleaned image
        clean_path = mask_path.parent / f"{mask_path.stem.replace('_mask', '')}_clean.png"
        if not clean_path.exists():
            clean_path = mask_path.parent / f"{mask_path.stem.replace('_mask', '')}_cleaned.png"
        visualizer.visualize_unet_mask(mask_path, clean_path if clean_path.exists() else None)
        
        # 3. Visualize YOLO detections
        visualizer.visualize_yolo_detections(original_image, symbols)
        
        # 4. Visualize assembled links
        visualizer.visualize_assembled_links(original_image, symbols, linked_pairs, staff_systems)
        
        # 5. Save summary
        visualizer.save_summary_json(symbols, linked_pairs, staff_systems)
        
        print(f"\n[Visualizer] All visualizations saved to: {vis_dir}")
    
    print("\n=== Assembly Complete ===")

if __name__ == "__main__":
    main()
