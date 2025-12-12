import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json

from .symbols import Symbol
from .staff import StaffSystem

class AssemblyVisualizer:
    """
    Creates visualization images for the assembly process.
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def save_original_image(self, image_path: Path, output_name: str = "01_original.png"):
        """
        Copy original image to output directory.
        """
        import shutil
        output_path = self.output_dir / output_name
        shutil.copy2(image_path, output_path)
        print(f"[Visualizer] Saved original image: {output_path}")
        return output_path
        
    def visualize_unet_mask(self, mask_path: Path, clean_path: Optional[Path] = None, 
                           output_name_mask: str = "02_unet_mask.png",
                           output_name_clean: str = "03_unet_cleaned.png"):
        """
        Save U-Net mask and cleaned image.
        """
        # Load and save mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            output_mask = self.output_dir / output_name_mask
            cv2.imwrite(str(output_mask), mask)
            print(f"[Visualizer] Saved U-Net mask: {output_mask}")
        
        # Load and save cleaned image if available
        if clean_path and clean_path.exists():
            clean = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
            if clean is not None:
                output_clean = self.output_dir / output_name_clean
                cv2.imwrite(str(output_clean), clean)
                print(f"[Visualizer] Saved U-Net cleaned image: {output_clean}")
        
        return output_mask if mask is not None else None
        
    def visualize_yolo_detections(self, original_image: np.ndarray, symbols: List[Symbol],
                                  output_name: str = "04_yolo_detections.jpg"):
        """
        Visualize YOLO detection results (similar to p001_detected.jpg).
        """
        vis_image = original_image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Color map for different classes
        class_colors = {
            'notehead': (0, 255, 0),      # Green
            'stem': (255, 0, 0),          # Blue
            'beam': (0, 0, 255),          # Red
            'flag': (255, 255, 0),        # Cyan
            'clef': (255, 0, 255),        # Magenta
            'rest': (0, 255, 255),        # Yellow
            'accidental': (128, 0, 128),   # Purple
            'barline': (255, 165, 0),     # Orange
        }
        
        for sym in symbols:
            x1, y1, x2, y2 = [int(coord) for coord in sym.bbox]
            
            # Determine color based on class
            color = (128, 128, 128)  # Default gray
            for class_key, class_color in class_colors.items():
                if class_key in sym.class_name.lower():
                    color = class_color
                    break
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{sym.class_name}: {sym.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 10, label_size[1] + 10)
            cv2.rectangle(vis_image, (x1, label_y - label_size[1] - 5), 
                         (x1 + label_size[0], label_y + 5), color, -1)
            cv2.putText(vis_image, label, (x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), vis_image)
        print(f"[Visualizer] Saved YOLO detections: {output_path}")
        return output_path
        
    def visualize_assembled_links(self, original_image: np.ndarray, symbols: List[Symbol],
                                  linked_pairs: List[Tuple[Symbol, Symbol]],
                                  staff_systems: List[StaffSystem],
                                  output_name: str = "05_assembled_links.jpg"):
        """
        Visualize assembled symbol relationships.
        Draws bounding boxes and connects linked symbols with lines.
        
        Args:
            original_image: Original image
            symbols: All detected symbols
            linked_pairs: List of (source_symbol, target_symbol) pairs that are linked
            staff_systems: Detected staff systems
        """
        vis_image = original_image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Draw staff systems (light gray lines)
        for system in staff_systems:
            for line_y in system.lines:
                cv2.line(vis_image, (0, int(line_y)), (vis_image.shape[1], int(line_y)), 
                        (200, 200, 200), 1)
        
        # Draw all symbol bounding boxes (lighter)
        for sym in symbols:
            x1, y1, x2, y2 = [int(coord) for coord in sym.bbox]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (150, 150, 150), 1)
        
        # Draw linked pairs with lines connecting centers
        link_colors = {
            'stem': (0, 255, 255),        # Cyan for stems
            'beam': (255, 0, 255),        # Magenta for beams
            'flag': (255, 255, 0),        # Yellow for flags
            'dot': (0, 255, 0),           # Green for dots
            'accidental': (255, 0, 0),     # Blue for accidentals
        }
        
        for source, target in linked_pairs:
            # Get centers
            src_center = (int(source.center_x), int(source.center_y))
            tgt_center = (int(target.center_x), int(target.center_y))
            
            # Determine link color based on target type
            color = (0, 255, 0)  # Default green
            for link_type, link_color in link_colors.items():
                if link_type in target.class_name.lower():
                    color = link_color
                    break
            
            # Draw line connecting centers
            cv2.line(vis_image, src_center, tgt_center, color, 2)
            
            # Draw small circle at centers
            cv2.circle(vis_image, src_center, 3, color, -1)
            cv2.circle(vis_image, tgt_center, 3, color, -1)
            
            # Highlight the linked symbols with thicker boxes
            x1, y1, x2, y2 = [int(coord) for coord in source.bbox]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            x1, y1, x2, y2 = [int(coord) for coord in target.bbox]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Add legend
        legend_y = 30
        cv2.putText(vis_image, "Link Colors:", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 25
        for link_type, link_color in link_colors.items():
            cv2.circle(vis_image, (20, legend_y), 5, link_color, -1)
            cv2.putText(vis_image, link_type, (35, legend_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            legend_y += 20
        
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), vis_image)
        print(f"[Visualizer] Saved assembled links: {output_path}")
        return output_path
        
    def save_summary_json(self, symbols: List[Symbol], linked_pairs: List[Tuple[Symbol, Symbol]],
                         staff_systems: List[StaffSystem], output_name: str = "00_summary.json"):
        """
        Save a summary JSON with statistics.
        """
        summary = {
            "total_symbols": len(symbols),
            "total_links": len(linked_pairs),
            "staff_systems": len(staff_systems),
            "symbol_counts": {},
            "link_types": {}
        }
        
        # Count symbols by class
        for sym in symbols:
            cls = sym.class_name
            summary["symbol_counts"][cls] = summary["symbol_counts"].get(cls, 0) + 1
        
        # Count links by type
        for source, target in linked_pairs:
            link_type = target.class_name
            summary["link_types"][link_type] = summary["link_types"].get(link_type, 0) + 1
        
        output_path = self.output_dir / output_name
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[Visualizer] Saved summary: {output_path}")
        return output_path


