#!/usr/bin/env python3
"""
OMR Assembly Pipeline Entry Point
Combines U-Net mask (staff lines) and YOLO detections (symbols) into MusicXML.
"""

import argparse
from pathlib import Path
import sys

# Ensure src is in path if running from root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

from assembler.staff import StaffSystemDetector
from assembler.symbols import SymbolLoader
from assembler.builder import ScoreBuilder
from assembler.exporter import MusicXMLExporter
from assembler.linker import Linker
from assembler.visualizer import AssemblyVisualizer
from utils.render_musicxml import render_musicxml
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Assemble OMR results into MusicXML")
    
    parser.add_argument("--json", type=str, required=False,
                        help="Path to YOLO detection results JSON (if not provided, will run YOLO inference)")
    parser.add_argument("--mask", type=str, required=True,
                        help="Path to U-Net staff mask (binary or grayscale)")
    parser.add_argument("--cleaned-image", type=str, default=None,
                        help="Path to U-Net cleaned image (for YOLO inference). If not provided, inferred from mask path.")
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
    parser.add_argument("--use-staff-groupings", action="store_true", default=True,
                        help="Use staff groupings (default: True). Use --no-staff-groupings to disable.")
    parser.add_argument("--no-staff-groupings", dest="use_staff_groupings", action="store_false",
                        help="Disable staff groupings")
    
    parser.add_argument("--use-geometric-first", action="store_true", default=True,
                        help="Use geometric rules first (default: True)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    mask_path = Path(args.mask)
    output_path = Path(args.output)
    model_path = Path(args.model)
    
    print("=== OMR Assembly Pipeline Started ===")
    
    # Determine cleaned image path
    if args.cleaned_image:
        cleaned_image_path = Path(args.cleaned_image)
    else:
        # Infer from mask path: mask_path = .../p009_mask.png -> cleaned = .../p009_clean.png
        cleaned_image_path = mask_path.parent / mask_path.stem.replace('_mask', '_clean').replace('mask', 'clean') / f"{mask_path.stem.replace('_mask', '').replace('mask', '')}_clean.png"
        # Also try alternative naming
        if not cleaned_image_path.exists():
            cleaned_image_path = mask_path.parent / f"{mask_path.stem.replace('_mask', '')}_clean.png"
    
    # Run YOLO inference if JSON not provided
    if args.json:
        json_path = Path(args.json)
    else:
        print(f"\nStep 0: Run YOLO inference on cleaned image: {cleaned_image_path}")
        if not cleaned_image_path.exists():
            print(f"[Error] Cleaned image not found at {cleaned_image_path}")
            print(f"Please provide --cleaned-image or ensure U-Net has generated the cleaned image.")
            return
        
        # Import YOLO inference
        from yolo.inference_tiled import TiledInference
        import json
        
        # Initialize YOLO inference
        yolo_model_path = Path("model/yolo/detect/yolo12l_muscima_finetune11/weights/best.pt")
        if not yolo_model_path.exists():
            print(f"[Error] YOLO model not found at {yolo_model_path}")
            return
        
        inferencer = TiledInference(
            model_path=str(yolo_model_path),
            tile_size=1216,
            target_size=640,
            overlap=100
        )
        inferencer.confidence_threshold = 0.25
        inferencer.iou_threshold = 0.45
        inferencer.nms_iou_threshold = 0.5
        
        # Process cleaned image
        # Process cleaned image
        image = cv2.imread(str(cleaned_image_path))
        if image is None:
            print(f"[Error] Cannot read cleaned image: {cleaned_image_path}")
            return
        
        h, w = image.shape[:2]
        tiles = inferencer.create_tiles(image)
        all_detections = []
        
        for tile, x_offset, y_offset in tiles:
            detections, _ = inferencer.process_tile(tile, x_offset, y_offset)
            # Filter valid detections
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                if x1 >= 0 and y1 >= 0 and x2 <= w and y2 <= h and x1 < x2 and y1 < y2:
                    all_detections.append(det)
        
        filtered_detections = inferencer.apply_nms(all_detections, inferencer.nms_iou_threshold)
        
        # Save JSON
        json_path = output_path.parent / f"{output_path.stem}_yolo_results.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(filtered_detections, f, indent=2, ensure_ascii=False)
        print(f"✓ YOLO inference completed: {len(filtered_detections)} detections saved to {json_path}")
    
    # 1. Detect Staff Lines and Groupings
    print(f"\nStep 1: Detect Staff Systems from {mask_path}")
    staff_detector = StaffSystemDetector(mask_path)
    staff_systems = staff_detector.detect_staff_lines()
    
    if not staff_systems:
        print("[Error] No staff systems detected! Cannot proceed.")
        return
    
    # Detect staff groupings (e.g., piano grand staff)
    staff_groupings = None
    if args.use_staff_groupings:
        print(f"\nStep 1b: Detect Staff Groupings")
        try:
            staff_groupings = staff_detector.detect_staff_groupings()
            if staff_groupings:
                print(f"Detected {len(staff_groupings)} staff groupings.")
            else:
                print("No staff groupings detected (using individual systems).")
        except Exception as e:
            print(f"[Warning] Failed to detect staff groupings: {e}")
            staff_groupings = None
    
    # Get image shape for normalization (H, W)
    # staff_detector.mask is loaded after detect_staff_lines
    image_shape = staff_detector.mask.shape[:2]
        
    # 2. Load Symbols
    print(f"\nStep 2: Load Symbols from {json_path}")
    # Set lower confidence threshold for stems (0.1) while keeping default (0.25) for others
    symbol_loader = SymbolLoader(json_path, class_specific_thresholds={'stem': 0.1})
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
    print("Using Rule-First architecture (geometric rules only):")
    print("  - Geometric hard logic for component assembly")
    print("  - No MLP refinement (pure rule-based)")
    
    # Pass image_shape, linker, staff_groupings, and staff_detector
    builder = ScoreBuilder(staff_systems, symbols, image_shape, linker, 
                          staff_groupings=staff_groupings,
                          staff_detector=staff_detector)
    parts, linked_pairs = builder.build(min_symbols_per_part=args.min_symbols, max_parts=args.max_parts)
    
    print(f"Assembled {len(parts)} parts (staves).")
    print(f"Found {len(linked_pairs)} symbol links.")
    
    # 5. Export MusicXML
    # Always output to Output/test directory for testing
    test_output_dir = Path('Output/test')
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # For testing, we might override output_path to be in Output/test, but if the user provided 
    # a specific output path (like via config), we should probably respect it or at least ensure compatibility.
    # The previous code overwrote output_path. Let's make it respect the user's wish if it was explicitly passed,
    # OR we can assume the user provided path takes precedence.
    # HOWEVER, the previous code forced it to Output/test.
    # "Output will be saved to test directory: Output/test/output.xml"
    # If the user sets output_path in config to "Output/test_cases/p009/output.xml", 
    # the code below overwrites `output_path` to `Output/test/output.xml`. This breaks the test case structure.
    # Let's FIX this behavior: Use args.output directly.
    
    # output_path is already set from args.output at the top.
    output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
    
    print(f"[Output] Output will be saved to: {output_path}")
    
    print(f"\nStep 5: Export to {output_path}")
    exporter = MusicXMLExporter()
    # Extract filename for default title if not provided
    default_title = output_path.stem if not args.title else None
    exporter.export(parts, output_path, 
                   title=args.title or default_title,
                   composer=args.composer)
    
    # 5b. Render MusicXML (Automatic)
    print(f"\nStep 5b: Render MusicXML to SVG")
    rendered_path = output_path.with_suffix('.svg')
    render_success = render_musicxml(str(output_path), str(rendered_path))
    if render_success:
        print(f"[Render] Generated SVG at: {rendered_path}")
    
    # 6. Generate Visualizations (if requested)
    if args.visualize:
        print("\nStep 6: Generate Visualizations")
        # Place visualizations in the SAME directory as output.xml
        vis_dir = output_path.parent
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
        
        # 5. Visualize system structure with measures (combined)
        if hasattr(builder, 'system_groups') and builder.system_groups:
            visualizer.visualize_system_and_measures(original_image, builder.system_groups, 
                                                     builder.part_indices, parts, staff_systems,
                                                     symbols=symbols)
        else:
            # Fallback: separate visualizations if system_groups not available
            visualizer.visualize_system_structure(original_image, [[s] for s in staff_systems], {})
            visualizer.visualize_measure_grid(original_image, parts, staff_systems, symbols=symbols)
        
        # 7. Save attribute map (new)
        visualizer.save_attribute_map(parts)
        
        # 8. Visualize triplets
        visualizer.visualize_triplets(original_image, parts)
        
        # 9. Save summary
        visualizer.save_summary_json(symbols, linked_pairs, staff_systems)
        
        print(f"\n[Visualizer] All visualizations saved to: {vis_dir}")
    
    print("\n=== Assembly Complete ===")

if __name__ == "__main__":
    main()
