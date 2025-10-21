#!/usr/bin/env python3
"""
YOLOv8l tiled inference script
Split large images into 1216x1216 tiles, scale to 640x640 for inference, then stitch results
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
import json
from typing import List, Tuple, Dict
import time

class TiledInference:
    def __init__(self, model_path: str, tile_size: int = 1216, target_size: int = 640, overlap: int = 100):
        """
        Initialize tiled inference engine
        
        Args:
            model_path: Path to trained model
            tile_size: Tile size (1216x1216)
            target_size: Inference target size (640x640)
            overlap: Overlap pixels between tiles
        """
        self.model_path = model_path
        self.tile_size = tile_size
        self.target_size = target_size
        self.overlap = overlap
        # Core region step: 1216x1216 tiling
        self.core_step = self.tile_size
        # Core region margin: 0, because core region is complete 1216x1216
        self.core_margin = 0
        # Extended margin for entire tile: used to display overlap regions
        self.tile_margin = self.overlap // 2
        
        # Load model
        print(f"Loading model: {model_path}")
        self.model = YOLO(model_path)
        print("Model loading completed")
        
        # Create output directory
        self.output_dir = Path('../Output/inference_output')
        self.output_dir.mkdir(exist_ok=True)
        
    def create_tiles(self, image: np.ndarray) -> List[Tuple[np.ndarray, int, int]]:
        """
        Split image into tiles - intelligent tiling logic
        
        Args:
            image: Input image
            
        Returns:
            List of tiles, each element is (tile_image, x_offset, y_offset)
        """
        height, width = image.shape[:2]
        tiles = []
        
        # Calculate number of complete tiling rows and columns
        full_rows = height // self.tile_size
        full_cols = width // self.tile_size
        
        # Calculate remaining parts
        remaining_height = height % self.tile_size
        remaining_width = width % self.tile_size
        
        print(f"Image size: {width}x{height}")
        print(f"Complete tiles: {full_cols} columns x {full_rows} rows")
        print(f"Remaining: {remaining_width} pixels wide x {remaining_height} pixels high")
        
        # Generate complete tiles (starting from top-left corner tiling)
        for row in range(full_rows):
            for col in range(full_cols):
                x = col * self.tile_size
                y = row * self.tile_size
                
                # Extract complete 1216x1216 tile
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                tiles.append((tile, x, y))
        
        # Handle right remaining part (if exists)
        if remaining_width > 0:
            # Start from bottom edge, add one more column
            for row in range(full_rows):
                x = width - self.tile_size  # Start from right edge
                y = row * self.tile_size
                
                # Extract tile (may be less than 1216 width)
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                
                # If width is insufficient, pad it
                if tile.shape[1] < self.tile_size:
                    padded_tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile
                
                tiles.append((tile, x, y))
        
        # Handle bottom remaining part (if exists)
        if remaining_height > 0:
            # Start from right edge, add one more row
            for col in range(full_cols):
                x = col * self.tile_size
                y = height - self.tile_size  # Start from bottom edge
                
                # Extract tile (may be less than 1216 height)
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                
                # If height is insufficient, pad it
                if tile.shape[0] < self.tile_size:
                    padded_tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile
                
                tiles.append((tile, x, y))
        
        # Handle bottom-right corner (if bottom-right corner has remaining parts)
        if remaining_width > 0 and remaining_height > 0:
            x = width - self.tile_size
            y = height - self.tile_size
            
            # Extract tile (may be less than 1216x1216)
            tile = image[y:y+self.tile_size, x:x+self.tile_size]
            
            # If size is insufficient, pad it
            if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                padded_tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
            
            tiles.append((tile, x, y))
        
        print(f"Created {len(tiles)} tiles")
        return tiles
    
    def process_tile(self, tile: np.ndarray, x_offset: int, y_offset: int) -> Tuple[List[Dict], np.ndarray]:
        """
        Process single tile
        
        Args:
            tile: Image tile
            x_offset: x offset of tile in original image
            y_offset: y offset of tile in original image
            
        Returns:
            (detection results list, annotated image)
        """
        # Resize to target size
        resized_tile = cv2.resize(tile, (self.target_size, self.target_size))
        
        # Perform inference
        results = self.model(resized_tile, conf=0.25, iou=0.45)
        
        # Parse results
        detections = []
        annotated_tile = resized_tile.copy()
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # Get bounding box coordinates (in 640x640 image)
                box = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())
                
                # Convert coordinates back to 1216x1216 size
                scale_x = self.tile_size / self.target_size
                scale_y = self.tile_size / self.target_size
                
                x1 = int(box[0] * scale_x)
                y1 = int(box[1] * scale_y)
                x2 = int(box[2] * scale_x)
                y2 = int(box[3] * scale_y)
                
                # Core region is the entire 1216x1216, no boundary filtering needed
                # Because tiles themselves are 1216x1216 tiled

                # Convert to original image coordinate system
                global_x1 = x_offset + x1
                global_y1 = y_offset + y1
                global_x2 = x_offset + x2
                global_y2 = y_offset + y2
                
                detection = {
                    'bbox': [global_x1, global_y1, global_x2, global_y2],
                    'confidence': float(conf),
                    'class_id': cls,
                    'class_name': self.model.names[cls] if cls in self.model.names else f'class_{cls}'
                }
                detections.append(detection)
                
                # Draw bounding box on annotated image (using 640x640 coordinates)
                # Convert 1216 coordinates back to 640x640 for drawing
                draw_x1 = int(x1 * self.target_size / self.tile_size)
                draw_y1 = int(y1 * self.target_size / self.tile_size)
                draw_x2 = int(x2 * self.target_size / self.tile_size)
                draw_y2 = int(y2 * self.target_size / self.tile_size)
                
                cv2.rectangle(annotated_tile, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
                label = f"{detection['class_name']}: {conf:.2f}"
                cv2.putText(annotated_tile, label, (draw_x1, draw_y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return detections, annotated_tile
    
    def apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        Apply non-maximum suppression
        
        Args:
            detections: Detection results list
            iou_threshold: IoU threshold
            
        Returns:
            Filtered detection results
        """
        if len(detections) == 0:
            return detections
        
        # Create copy to avoid modifying original list
        detections_copy = detections.copy()
        
        # Sort by confidence
        detections_copy.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections_copy:
            # Take detection with highest confidence
            current = detections_copy.pop(0)
            keep.append(current)
            
            # Remove other detections with IoU too high with current detection
            remaining = []
            for det in detections_copy:
                iou = self.calculate_iou(current['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining.append(det)
            detections_copy = remaining
        
        return keep
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate IoU of two bounding boxes
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value
        """
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reconstruct_image(self, original_image: np.ndarray, annotated_tiles: List[Tuple[np.ndarray, int, int]]) -> np.ndarray:
        """
        Reconstruct complete image with annotations
        
        Args:
            original_image: Original image
            annotated_tiles: List of annotated tiles [(tile, x_offset, y_offset), ...]
            
        Returns:
            Reconstructed image
        """
        height, width = original_image.shape[:2]
        reconstructed = original_image.copy()
        
        for annotated_tile, x_offset, y_offset in annotated_tiles:
            # Scale 640x640 annotated image back to 1216x1216
            resized_tile = cv2.resize(annotated_tile, (self.tile_size, self.tile_size))

            # Directly paste entire core region, because it's now 1216x1216 tiled
            paste_x1 = x_offset
            paste_y1 = y_offset
            paste_x2 = min(x_offset + self.tile_size, width)
            paste_y2 = min(y_offset + self.tile_size, height)

            # Calculate region to extract from tile
            tile_x1 = 0
            tile_y1 = 0
            tile_x2 = paste_x2 - paste_x1
            tile_y2 = paste_y2 - paste_y1

            if paste_x1 < paste_x2 and paste_y1 < paste_y2:
                reconstructed[paste_y1:paste_y2, paste_x1:paste_x2] = resized_tile[tile_y1:tile_y2, tile_x1:tile_x2]
        
        return reconstructed
    
    def add_crop_visualization(self, image: np.ndarray, tiles: List[Tuple[np.ndarray, int, int]]) -> np.ndarray:
        """
        Add crop region visualization to image
        
        Args:
            image: Original image
            tiles: List of tiles [(tile, x_offset, y_offset), ...]
            
        Returns:
            Image with crop region annotations
        """
        height, width = image.shape[:2]
        vis_image = image.copy()
        
        # First draw all red boxes (core regions)
        for tile, x_offset, y_offset in tiles:
            # Draw core region (red box - 1216x1216 tiled content area)
            core_x1 = x_offset
            core_y1 = y_offset
            core_x2 = min(x_offset + self.tile_size, width)
            core_y2 = min(y_offset + self.tile_size, height)
            cv2.rectangle(vis_image, (core_x1, core_y1), (core_x2, core_y2), (0, 0, 255), 2)
            
            # Add label
            cv2.putText(vis_image, f"Core ({x_offset},{y_offset})", 
                       (x_offset + 5, y_offset + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Then draw blue boxes (overlap regions)
        for tile, x_offset, y_offset in tiles:
            # Draw overlap region (blue box - show intersection logic)
            # Check if there are overlapping tiles
            has_overlap = False
            
            # Check if there's overlap on the right
            if x_offset + self.tile_size > width:
                overlap_x1 = width - self.tile_size
                overlap_y1 = y_offset
                overlap_x2 = min(overlap_x1 + self.tile_size, width)
                overlap_y2 = min(overlap_y1 + self.tile_size, height)
                cv2.rectangle(vis_image, (overlap_x1, overlap_y1), (overlap_x2, overlap_y2), (255, 0, 0), 2)
                has_overlap = True
            
            # Check if there's overlap on the bottom
            if y_offset + self.tile_size > height:
                overlap_x1 = x_offset
                overlap_y1 = height - self.tile_size
                overlap_x2 = min(overlap_x1 + self.tile_size, width)
                overlap_y2 = min(overlap_y1 + self.tile_size, height)
                cv2.rectangle(vis_image, (overlap_x1, overlap_y1), (overlap_x2, overlap_y2), (255, 0, 0), 2)
                has_overlap = True
            
            # Add overlap label
            if has_overlap:
                cv2.putText(vis_image, "Blue: Overlap", 
                           (x_offset + 5, y_offset + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Add legend
        cv2.putText(vis_image, "Red: Core regions (1216x1216)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_image, "Blue: Overlap regions", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return vis_image
    
    def create_combined_visualization(self, image: np.ndarray, tiles: List[Tuple[np.ndarray, int, int]], detections: List[Dict]) -> np.ndarray:
        """
        Create comprehensive visualization image containing red boxes, blue boxes, and green boxes
        
        Args:
            image: Original image
            tiles: List of tiles [(tile, x_offset, y_offset), ...]
            detections: List of detection results
            
        Returns:
            Comprehensive visualization image containing all boxes
        """
        height, width = image.shape[:2]
        vis_image = image.copy()
        
        # 1. First draw all red boxes (core regions)
        for tile, x_offset, y_offset in tiles:
            # Draw core region (red box - 1216x1216 tiled content area)
            core_x1 = x_offset
            core_y1 = y_offset
            core_x2 = min(x_offset + self.tile_size, width)
            core_y2 = min(y_offset + self.tile_size, height)
            cv2.rectangle(vis_image, (core_x1, core_y1), (core_x2, core_y2), (0, 0, 255), 2)
            
            # Add label
            cv2.putText(vis_image, f"Core ({x_offset},{y_offset})", 
                       (x_offset + 5, y_offset + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 2. Draw blue boxes (overlap regions)
        for tile, x_offset, y_offset in tiles:
            # Draw overlap region (blue box - show intersection logic)
            has_overlap = False
            
            # Check if there's overlap on the right
            if x_offset + self.tile_size > width:
                overlap_x1 = width - self.tile_size
                overlap_y1 = y_offset
                overlap_x2 = min(overlap_x1 + self.tile_size, width)
                overlap_y2 = min(overlap_y1 + self.tile_size, height)
                cv2.rectangle(vis_image, (overlap_x1, overlap_y1), (overlap_x2, overlap_y2), (255, 0, 0), 2)
                has_overlap = True
            
            # Check if there's overlap on the bottom
            if y_offset + self.tile_size > height:
                overlap_x1 = x_offset
                overlap_y1 = height - self.tile_size
                overlap_x2 = min(overlap_x1 + self.tile_size, width)
                overlap_y2 = min(overlap_y1 + self.tile_size, height)
                cv2.rectangle(vis_image, (overlap_x1, overlap_y1), (overlap_x2, overlap_y2), (255, 0, 0), 2)
                has_overlap = True
            
            # Add overlap label
            if has_overlap:
                cv2.putText(vis_image, "Blue: Overlap", 
                           (x_offset + 5, y_offset + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 3. Draw green boxes (detection results)
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw detection box (green)
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add detection label (only show first 20 to avoid overcrowding)
            if i < 20:
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(vis_image, label, (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 4. Add legend
        cv2.putText(vis_image, "Red: Core regions (1216x1216)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_image, "Blue: Overlap regions", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(vis_image, f"Green: Detections ({len(detections)} total)", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return vis_image
    
    def process_image(self, image_path: str) -> Tuple[np.ndarray, List[Dict], List[Dict], np.ndarray, np.ndarray]:
        """
        Process single image
        
        Args:
            image_path: Image path
            
        Returns:
            (annotated image, filtered detection results, all detection results, crop region visualization image, comprehensive visualization image)
        """
        print(f"Processing image: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Create tiles
        tiles = self.create_tiles(image)
        
        # Process all tiles
        all_detections = []
        annotated_tiles = []
        
        start_time = time.time()
        for i, (tile, x_offset, y_offset) in enumerate(tiles):
            print(f"Processing tile {i+1}/{len(tiles)} (offset: {x_offset}, {y_offset})")
            
            detections, annotated_tile = self.process_tile(tile, x_offset, y_offset)
            all_detections.extend(detections)
            annotated_tiles.append((annotated_tile, x_offset, y_offset))
        
        processing_time = time.time() - start_time
        print(f"Tile processing completed, time taken: {processing_time:.2f} seconds")
        
        # Apply NMS
        print("Applying non-maximum suppression...")
        filtered_detections = self.apply_nms(all_detections)
        print(f"Detected {len(all_detections)} targets, {len(filtered_detections)} remaining after NMS")
        
        # Reconstruct image
        print("Reconstructing image...")
        reconstructed_image = self.reconstruct_image(image, annotated_tiles)
        
        # Create crop region visualization
        print("Creating crop region visualization...")
        crop_visualization = self.add_crop_visualization(image, tiles)
        
        # Create comprehensive visualization (red boxes + blue boxes + green boxes)
        print("Creating comprehensive visualization...")
        combined_visualization = self.create_combined_visualization(image, tiles, filtered_detections)
        
        # Return all detection results and filtered results
        return reconstructed_image, filtered_detections, all_detections, crop_visualization, combined_visualization
    
    def save_results(self, image_path: str, annotated_image: np.ndarray, filtered_detections: List[Dict], all_detections: List[Dict], crop_visualization: np.ndarray, combined_visualization: np.ndarray):
        """
        # Save results
        
        Args:
        image_path: Original image path
        annotated_image: Annotated image
        filtered_detections: Detection results after NMS filtering
        all_detections: All detection results (including those filtered by NMS)
        crop_visualization: Crop region visualization image
        combined_visualization: Comprehensive visualization image (red boxes + blue boxes + green boxes)
        """
        # Generate output filenames
        input_path = Path(image_path)
        base_name = input_path.stem
        
        # Save annotated image
        output_image_path = self.output_dir / f"{base_name}_detected.jpg"
        cv2.imwrite(str(output_image_path), annotated_image)
        print(f"Saving annotated image: {output_image_path}")
        
        # Save crop region visualization image
        output_crop_path = self.output_dir / f"{base_name}_crop_visualization.jpg"
        cv2.imwrite(str(output_crop_path), crop_visualization)
        print(f"Saving crop region visualization: {output_crop_path}")
        
        # Save comprehensive visualization image (red boxes + blue boxes + green boxes)
        output_combined_path = self.output_dir / f"{base_name}_combined_visualization.jpg"
        cv2.imwrite(str(output_combined_path), combined_visualization)
        print(f"Saving comprehensive visualization: {output_combined_path}")
        
        # Save NMS filtered detection results JSON
        output_json_path = self.output_dir / f"{base_name}_results.json"
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_detections, f, indent=2, ensure_ascii=False)
        print(f"Saving NMS filtered detection results: {output_json_path}")
        
        # Save all detection results JSON (including those filtered by NMS)
        output_all_json_path = self.output_dir / f"{base_name}_all_results.json"
        with open(output_all_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_detections, f, indent=2, ensure_ascii=False)
        print(f"Saving all detection results: {output_all_json_path}")
        
        # Save detection report
        output_report_path = self.output_dir / f"{base_name}_report.txt"
        with open(output_report_path, 'w', encoding='utf-8') as f:
            f.write(f"Detection Report - {base_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total detections: {len(all_detections)}\n")
            f.write(f"Remaining after NMS: {len(filtered_detections)}\n")
            f.write(f"Filtered by NMS: {len(all_detections) - len(filtered_detections)}\n")
            f.write(f"Tile size: {self.tile_size}x{self.tile_size}\n")
            f.write(f"Inference size: {self.target_size}x{self.target_size}\n")
            f.write(f"Overlap pixels: {self.overlap}\n")
            f.write(f"Core region margin: {self.core_margin}\n")
            f.write(f"Tile extension margin: {self.tile_margin}\n\n")
            
            f.write("Crop region description:\n")
            f.write("-" * 30 + "\n")
            f.write("Red box: Core region (1216x1216 tiled, used for detection)\n")
            f.write("Blue box: Overlap region (show intersection logic)\n")
            f.write("Tiling logic: Start tiling from top-left corner, add one more row/column from bottom/right edge for incomplete tiles\n")
            f.write(f"Core region step: {self.core_step} pixels\n")
            f.write(f"Tile extension margin: {self.tile_margin} pixels\n\n")
            
            # Statistics by class (all detection results)
            class_counts_all = {}
            for det in all_detections:
                class_name = det['class_name']
                class_counts_all[class_name] = class_counts_all.get(class_name, 0) + 1
            
            # Statistics by class (after NMS)
            class_counts_filtered = {}
            for det in filtered_detections:
                class_name = det['class_name']
                class_counts_filtered[class_name] = class_counts_filtered.get(class_name, 0) + 1
            
            f.write("All detection results statistics:\n")
            f.write("-" * 30 + "\n")
            for class_name, count in sorted(class_counts_all.items()):
                filtered_count = class_counts_filtered.get(class_name, 0)
                f.write(f"{class_name}: {count} (After NMS: {filtered_count})\n")
            
            f.write("\nDetailed detection results after NMS filtering:\n")
            f.write("-" * 30 + "\n")
            for i, det in enumerate(filtered_detections, 1):
                f.write(f"{i}. {det['class_name']} (confidence: {det['confidence']:.3f})\n")
                f.write(f"   Position: ({det['bbox'][0]}, {det['bbox'][1]}) - ({det['bbox'][2]}, {det['bbox'][3]})\n")
            
            f.write("\nAll detection results (including those filtered by NMS):\n")
            f.write("-" * 30 + "\n")
            for i, det in enumerate(all_detections, 1):
                # Check if filtered by NMS
                is_filtered = det not in filtered_detections
                status = " [Filtered by NMS]" if is_filtered else ""
                f.write(f"{i}. {det['class_name']} (confidence: {det['confidence']:.3f}){status}\n")
                f.write(f"   Position: ({det['bbox'][0]}, {det['bbox'][1]}) - ({det['bbox'][2]}, {det['bbox'][3]})\n")
        
        print(f"Saving detection report: {output_report_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8l tiled inference script')
    parser.add_argument('--model', type=str, 
                       default='../runs/detect/yolov8l_muscima_finetune/weights/best.pt',
                       help='Path to trained model')
    parser.add_argument('--input', type=str,
                       default='../v1.0/data/images/w-01/symbol/p001.png',
                       help='Input image path')
    parser.add_argument('--tile-size', type=int, default=1216,
                       help='Tile size')
    parser.add_argument('--target-size', type=int, default=640,
                       help='Inference target size')
    parser.add_argument('--overlap', type=int, default=100,
                       help='Overlap pixels between tiles')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.model):
        print(f"Error: Model file does not exist: {args.model}")
        return
    
    if not os.path.exists(args.input):
        print(f"Error: Input image does not exist: {args.input}")
        return
    
    # Create inference engine
    inferencer = TiledInference(
        model_path=args.model,
        tile_size=args.tile_size,
        target_size=args.target_size,
        overlap=args.overlap
    )
    
    try:
        # Process image
        annotated_image, filtered_detections, all_detections, crop_visualization, combined_visualization = inferencer.process_image(args.input)
        
        # Save results
        inferencer.save_results(args.input, annotated_image, filtered_detections, all_detections, crop_visualization, combined_visualization)
        
        print("Processing completed!")
        
    except Exception as e:
        print(f"Error occurred during processing: {e}")
        import traceback
        traceback.print_exc()


def inference_from_config(config):
    """
    Perform inference from configuration object
    
    Args:
        config: InferenceConfig configuration object
    """
    import sys
    setup_path = str(Path(__file__).parent.parent / 'setup')
    if setup_path not in sys.path:
        sys.path.append(setup_path)
    from setup_inference import InferenceConfig
    
    if not isinstance(config, InferenceConfig):
        raise ValueError("config must be an InferenceConfig instance")
    
    # Get parameters from configuration
    inference_args = config.get_inference_args()
    
    # Check if files exist
    if not os.path.exists(inference_args['model_path']):
        print(f"Error: Model file does not exist: {inference_args['model_path']}")
        return None
    
    if not os.path.exists(inference_args['input_image_path']):
        print(f"Error: Input image does not exist: {inference_args['input_image_path']}")
        return None
    
    # Create inference engine
    inferencer = TiledInference(
        model_path=inference_args['model_path'],
        tile_size=inference_args['tile_size'],
        target_size=inference_args['target_size'],
        overlap=inference_args['overlap']
    )
    
    try:
        # Process image
        annotated_image, filtered_detections, all_detections, crop_visualization, combined_visualization = inferencer.process_image(inference_args['input_image_path'])
        
        # Save results
        inferencer.save_results(inference_args['input_image_path'], annotated_image, filtered_detections, all_detections, crop_visualization, combined_visualization)
        
        print("Processing completed!")
        
        return {
            'annotated_image': annotated_image,
            'filtered_detections': filtered_detections,
            'all_detections': all_detections,
            'crop_visualization': crop_visualization,
            'combined_visualization': combined_visualization
        }
        
    except Exception as e:
        print(f"Error occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    main()
