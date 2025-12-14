#!/usr/bin/env python3
"""
Batch YOLO inference with different confidence thresholds
Generate prediction images with thresholds from 0.05 to 0.25 (step 0.05)
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yolo.inference_tiled import TiledInference


def batch_inference_with_thresholds(
    model_path: str,
    image_path: str,
    output_dir: str,
    thresholds: List[float] = None,
    tile_size: int = 1216,
    target_size: int = 640,
    overlap: int = 100
):
    """
    Run inference with multiple confidence thresholds and save results
    
    Args:
        model_path: Path to YOLO model
        image_path: Path to input image
        output_dir: Output directory for results
        thresholds: List of confidence thresholds (default: [0.05, 0.10, 0.15, 0.20, 0.25])
        tile_size: Tile size for tiled inference
        target_size: Target size for inference
        overlap: Overlap pixels between tiles
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get input image name
    input_path = Path(image_path)
    base_name = input_path.stem
    
    print("=" * 60)
    print(f"Batch Inference with Multiple Thresholds")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Input image: {image_path}")
    print(f"Output directory: {output_dir}")
    print(f"Thresholds: {thresholds}")
    print("=" * 60)
    print()
    
    # Load model once (shared across all thresholds)
    print("Loading model...")
    inferencer = TiledInference(
        model_path=model_path,
        tile_size=tile_size,
        target_size=target_size,
        overlap=overlap
    )
    print("Model loaded successfully!")
    print()
    
    # Process each threshold
    for threshold in thresholds:
        print(f"\n{'='*60}")
        print(f"Processing with confidence threshold: {threshold}")
        print(f"{'='*60}")
        
        # Set threshold
        inferencer.confidence_threshold = threshold
        
        try:
            # Process image
            annotated_image, filtered_detections, all_detections, crop_visualization, combined_visualization = inferencer.process_image(image_path)
            
            # Save results with threshold in filename
            threshold_str = f"{threshold:.2f}".replace('.', '_')
            
            # Save annotated image (detection boxes)
            output_image_path = output_path / f"{base_name}_threshold_{threshold_str}_detected.jpg"
            cv2.imwrite(str(output_image_path), annotated_image)
            print(f"✓ Saved annotated image: {output_image_path}")
            
            # Save combined visualization
            output_combined_path = output_path / f"{base_name}_threshold_{threshold_str}_combined.jpg"
            cv2.imwrite(str(output_combined_path), combined_visualization)
            print(f"✓ Saved combined visualization: {output_combined_path}")
            
            # Save detection results JSON
            output_json_path = output_path / f"{base_name}_threshold_{threshold_str}_results.json"
            import json
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_detections, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved detection results: {output_json_path}")
            
            # Print statistics
            print(f"\nStatistics for threshold {threshold}:")
            print(f"  Total detections: {len(all_detections)}")
            print(f"  After NMS: {len(filtered_detections)}")
            
            # Count by class
            class_counts = {}
            for det in filtered_detections:
                class_name = det['class_name']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Print tuple-related classes
            tuple_classes = [c for c in class_counts.keys() if 'tuple' in c.lower()]
            if tuple_classes:
                print(f"  Tuple-related detections:")
                for cls in tuple_classes:
                    print(f"    {cls}: {class_counts[cls]}")
            
        except Exception as e:
            print(f"✗ Error processing threshold {threshold}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("Batch processing completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Batch YOLO inference with different confidence thresholds')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained YOLO model')
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--thresholds', type=str, default=None,
                       help='Comma-separated list of thresholds (e.g., "0.05,0.10,0.15,0.20,0.25")')
    parser.add_argument('--tile-size', type=int, default=1216,
                       help='Tile size')
    parser.add_argument('--target-size', type=int, default=640,
                       help='Inference target size')
    parser.add_argument('--overlap', type=int, default=100,
                       help='Overlap pixels between tiles')
    
    args = parser.parse_args()
    
    # Parse thresholds
    if args.thresholds:
        thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    else:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    # Validate paths
    if not os.path.exists(args.model):
        print(f"Error: Model file does not exist: {args.model}")
        return
    
    if not os.path.exists(args.input):
        print(f"Error: Input image does not exist: {args.input}")
        return
    
    # Run batch inference
    batch_inference_with_thresholds(
        model_path=args.model,
        image_path=args.input,
        output_dir=args.output,
        thresholds=thresholds,
        tile_size=args.tile_size,
        target_size=args.target_size,
        overlap=args.overlap
    )


if __name__ == '__main__':
    main()

