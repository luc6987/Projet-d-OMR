import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json

from .symbols import Symbol
from .staff import StaffSystem
from .builder import AssembledPart, AssembledMeasure, AssembledNote

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
        Visualize YOLO detection results with bounding boxes only (no labels on boxes).
        Uses a legend to show class-color mapping.
        """
        vis_image = original_image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Get image dimensions for validation
        img_height, img_width = vis_image.shape[:2]
        
        # Color map for different classes
        # Triplet-related symbols should be checked first (higher priority)
        triplet_colors = {
            'numeral_3': (0, 255, 255),        # Cyan - for triplet number "3"
            'tuple_bracket/line': (255, 0, 255), # Magenta - for triplet bracket/line
            'tuple_bracket': (255, 0, 255),    # Magenta - for triplet bracket
            'tuple_line': (255, 0, 255),       # Magenta - for triplet line
            'tuple': (255, 165, 0),            # Orange - for tuple symbol
        }
        
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
        
        # Track which classes are actually present
        present_classes = set()
        
        # Draw bounding boxes without labels
        for sym in symbols:
            x1, y1, x2, y2 = [int(coord) for coord in sym.bbox]
            
            # Validate bbox coordinates against image dimensions
            # Clamp coordinates to image bounds to prevent drawing outside image
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            
            # Skip invalid boxes (width or height <= 0)
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Determine color based on class
            # Check triplet-related symbols first (higher priority)
            color = (128, 128, 128)  # Default gray
            matched_class = None
            sym_name_lower = sym.class_name.lower()
            
            # First check for triplet-related symbols (check longer keys first to avoid partial matches)
            # Sort by key length descending to match "tuple_bracket/line" before "tuple_bracket"
            sorted_triplet_keys = sorted(triplet_colors.keys(), key=len, reverse=True)
            for triplet_key in sorted_triplet_keys:
                if triplet_key in sym_name_lower:
                    color = triplet_colors[triplet_key]
                    matched_class = f'triplet_{triplet_key.replace("/", "_")}'
                    break
            
            # If not a triplet symbol, check other classes
            if matched_class is None:
                for class_key, class_color in class_colors.items():
                    if class_key in sym_name_lower:
                        color = class_color
                        matched_class = class_key
                        break
            
            # Draw bounding box only (no label)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Track present classes for legend
            if matched_class:
                present_classes.add(matched_class)
            else:
                # For unmatched classes, use class name directly
                present_classes.add(sym.class_name)
        
        # Draw legend in top-left corner
        legend_x = 10
        legend_y = 30
        legend_bg_height = len(present_classes) * 25 + 40
        legend_bg_width = 300
        
        # Draw semi-transparent background for legend
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (legend_x - 5, legend_y - 25), 
                     (legend_x + legend_bg_width, legend_y + legend_bg_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis_image, 0.4, 0, vis_image)
        
        # Legend title
        cv2.putText(vis_image, "Detection Classes:", (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 30
        
        # Draw legend items for present classes
        for class_key in sorted(present_classes):
            # Get color for this class
            color = (128, 128, 128)  # Default gray
            display_name = class_key
            
            # Check if it matches a triplet-related class first
            if class_key.startswith('triplet_'):
                triplet_key = class_key.replace('triplet_', '')
                if triplet_key in triplet_colors:
                    color = triplet_colors[triplet_key]
                    display_name = triplet_key.replace('_', ' ').title()
            else:
                # Check if it matches a known class
                for known_class, known_color in class_colors.items():
                    if known_class in class_key.lower() or class_key.lower() in known_class:
                        color = known_color
                        display_name = known_class
                        break
            
            # Draw color box
            cv2.rectangle(vis_image, (legend_x, legend_y - 10), 
                         (legend_x + 15, legend_y + 5), color, -1)
            cv2.rectangle(vis_image, (legend_x, legend_y - 10), 
                         (legend_x + 15, legend_y + 5), (255, 255, 255), 1)
            
            # Draw class name
            cv2.putText(vis_image, display_name, (legend_x + 20, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 25
        
        # Add total count
        cv2.putText(vis_image, f"Total: {len(symbols)} detections", 
                   (legend_x, legend_y + 10), 
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
        Includes virtual stems created during assembly.
        
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
        
        # Collect all symbols including virtual stems from linked_pairs
        # Use dict with id as key to avoid duplicates
        all_symbols_dict = {id(sym): sym for sym in symbols}
        for source, target in linked_pairs:
            all_symbols_dict[id(source)] = source
            all_symbols_dict[id(target)] = target
        all_symbols = list(all_symbols_dict.values())
        
        # Draw all symbol bounding boxes (lighter)
        virtual_stem_count = 0
        for sym in all_symbols:
            x1, y1, x2, y2 = [int(coord) for coord in sym.bbox]
            # Check if this is a virtual stem (confidence <= 0.5 and class is stem)
            # Virtual stems are created with confidence 0.5
            is_virtual = (sym.class_name.lower() == 'stem' and 
                         hasattr(sym, 'confidence') and sym.confidence <= 0.5)
            
            if is_virtual:
                virtual_stem_count += 1
                # Draw virtual stems with solid orange rectangle (more visible)
                color = (0, 165, 255)  # Orange in BGR format for OpenCV
                thickness = 3
                # Draw solid rectangle for virtual stems
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
                # Also draw a filled rectangle with transparency for better visibility
                overlay = vis_image.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                cv2.addWeighted(overlay, 0.2, vis_image, 0.8, 0, vis_image)
                # Draw border again
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, thickness)
            else:
                # Regular symbols
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
        
        # Add virtual stem indicator in legend
        cv2.rectangle(vis_image, (15, legend_y - 10), (25, legend_y), (255, 165, 0), 2)
        cv2.putText(vis_image, "virtual stem", (35, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(vis_image, f"({virtual_stem_count} found)", (150, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
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
    
    def visualize_system_structure(self, original_image: np.ndarray,
                                   system_groups: List[List[StaffSystem]],
                                   part_indices: dict,
                                   output_name: str = "06_system_structure.png"):
        """
        Visualize System structure and Part assignments.
        Implements visualization from rule.md: System_Structure.png
        
        Args:
            original_image: Original image
            system_groups: List of system groups (each group is a list of StaffSystem)
            part_indices: Dictionary mapping system index to part ID
        """
        vis_image = original_image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Draw System bounding boxes (red)
        for group_idx, system_group in enumerate(system_groups):
            if not system_group:
                continue
            
            # Calculate bounding box for the entire system group
            top_system = min(system_group, key=lambda s: s.top_line)
            bottom_system = max(system_group, key=lambda s: s.bottom_line)
            
            # Find X range (use full image width for now, or could use symbol X range)
            x1 = 0
            x2 = vis_image.shape[1]
            y1 = top_system.top_line
            y2 = bottom_system.bottom_line
            
            # Draw System bounding box (red, thick)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw label: "System N"
            label = f"System {group_idx + 1}"
            cv2.putText(vis_image, label, (x1 + 10, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw Staff bounding boxes within System (green)
            for staff_idx, system in enumerate(system_group):
                staff_x1 = x1
                staff_x2 = x2
                staff_y1 = system.top_line
                staff_y2 = system.bottom_line
                
                # Draw Staff bounding box (green, medium thickness)
                cv2.rectangle(vis_image, (staff_x1, staff_y1), (staff_x2, staff_y2), (0, 255, 0), 2)
                
                # Draw Part label: "Part N"
                # Get part ID from part_indices (need to find system index)
                # For now, use staff_idx + 1
                part_id = staff_idx + 1
                part_label = f"Part {part_id}"
                cv2.putText(vis_image, part_label, (staff_x1 + 10, staff_y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), vis_image)
        print(f"[Visualizer] Saved system structure: {output_path}")
        return output_path
    
    def visualize_measure_grid(self, original_image: np.ndarray,
                               parts: List[AssembledPart],
                               staff_systems: List[StaffSystem],
                               symbols: Optional[List[Symbol]] = None,
                               output_name: str = "07_measure_grid.png"):
        """
        Visualize measure grid with Global Barlines and measure numbers.
        Barlines are drawn only within each system's vertical range (not across entire image).
        Blue lines represent actual barline positions (measure boundaries).
        Implements visualization from rule.md: Measure_Grid.png
        
        Args:
            original_image: Original image
            parts: List of assembled parts
            staff_systems: List of staff systems
            symbols: Optional list of all symbols (to extract barline positions)
        """
        vis_image = original_image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Draw staff lines (light gray)
        for system in staff_systems:
            for line_y in system.lines:
                cv2.line(vis_image, (0, int(line_y)), (vis_image.shape[1], int(line_y)),
                        (200, 200, 200), 1)
        
        # Collect barline positions from symbols, filtered by system
        # Each system should only show barlines that belong to it
        barline_x_by_system = {}  # system_idx -> set of barline X positions
        
        # Method 1: Extract barline positions from detected symbols, filtered by system
        if symbols:
            for sym in symbols:
                class_name_lower = sym.class_name.lower()
                # Check for barlines and measure_separators (which may be treated as barlines)
                is_barline = any(keyword in class_name_lower for keyword in ['barline', 'thin_barline', 'thick_barline', 'repeat'])
                is_measure_separator = 'measure_separator' in class_name_lower
                
                # measure_separator will be handled in builder, but we still need to check it here
                # for visualization purposes (it will be treated as barline if within single part)
                if is_barline or is_measure_separator:
                    # Find which system(s) this barline belongs to based on Y coordinate
                    sym_y = sym.center_y
                    for system_idx, system in enumerate(staff_systems):
                        # Check if barline center is within system range (with small margin)
                        if system.contains_y(sym_y, margin=system.avg_spacing * 0.5):
                            if system_idx not in barline_x_by_system:
                                barline_x_by_system[system_idx] = set()
                            barline_x_by_system[system_idx].add(sym.center_x)
        
        # Collect all unique barline positions for fallback
        barline_x_positions = set()
        for x_set in barline_x_by_system.values():
            barline_x_positions.update(x_set)
        
        # Method 2: If no barlines found in symbols, infer from measure boundaries
        if not barline_x_positions:
            # Collect measure boundaries: end of each measure (last note position)
            for part_idx, part in enumerate(parts):
                if part_idx >= len(staff_systems):
                    continue
                
                for measure in part.measures:
                    if measure.notes:
                        # Use last note X position as measure end (barline position)
                        last_note_x = max(note.x for note in measure.notes)
                        barline_x_positions.add(last_note_x)
        
        # Sort and merge close barline positions
        sorted_barline_xs = sorted(barline_x_positions)
        merged_barline_xs = []
        threshold = 50  # pixels
        
        if sorted_barline_xs:
            merged_barline_xs.append(sorted_barline_xs[0])
            for x in sorted_barline_xs[1:]:
                if abs(x - merged_barline_xs[-1]) > threshold:
                    merged_barline_xs.append(x)
                else:
                    # Merge with previous (average)
                    merged_barline_xs[-1] = (merged_barline_xs[-1] + x) / 2
        
        # Draw barlines (blue vertical lines) - only within each system's range
        # Use system-specific barline positions if available
        for barline_x in merged_barline_xs:
            # Find which systems should have this barline
            systems_with_barline = set()
            
            # First, check if this barline was detected in any system (from barline_x_by_system)
            for system_idx, x_set in barline_x_by_system.items():
                # Check if this barline_x is close to any barline in this system
                for sys_barline_x in x_set:
                    if abs(barline_x - sys_barline_x) < threshold:
                        systems_with_barline.add(system_idx)
                        break
            
            # Fallback: If no system-specific assignment, use measure-based logic
            if not systems_with_barline:
                for part_idx, part in enumerate(parts):
                    if part_idx >= len(staff_systems):
                        continue
                    
                    system = staff_systems[part_idx]
                    
                    # Check if any measure in this part is near this barline
                    for measure in part.measures:
                        if measure.notes:
                            first_note_x = min(note.x for note in measure.notes)
                            last_note_x = max(note.x for note in measure.notes)
                            # If barline is near the end of this measure, draw it
                            if abs(barline_x - last_note_x) < threshold or \
                               (first_note_x <= barline_x <= last_note_x):
                                systems_with_barline.add(part_idx)
                                break
            
            # Draw barline in each relevant system
            for system_idx in systems_with_barline:
                if system_idx < len(staff_systems):
                    system = staff_systems[system_idx]
                    y_top = system.top_line
                    y_bottom = system.bottom_line
                    cv2.line(vis_image, (int(barline_x), int(y_top)), (int(barline_x), int(y_bottom)),
                            (255, 0, 0), 2)  # Blue
        
        # Draw measure numbers above each measure (at measure center)
        for part_idx, part in enumerate(parts):
            if part_idx >= len(staff_systems):
                continue
            
            system = staff_systems[part_idx]
            y_top = system.top_line
            
            for measure in part.measures:
                if measure.notes:
                    # Calculate measure center for label placement
                    first_note_x = min(note.x for note in measure.notes)
                    last_note_x = max(note.x for note in measure.notes)
                    measure_center_x = (first_note_x + last_note_x) / 2
                    
                    # Draw measure number label
                    label = f"Meas {measure.number}"
                    cv2.putText(vis_image, label, (int(measure_center_x) - 30, int(y_top) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), vis_image)
        print(f"[Visualizer] Saved measure grid: {output_path}")
        return output_path
    
    def visualize_system_and_measures(self, original_image: np.ndarray,
                                     system_groups: List[List[StaffSystem]],
                                     part_indices: dict,
                                     parts: List[AssembledPart],
                                     staff_systems: List[StaffSystem],
                                     symbols: Optional[List[Symbol]] = None,
                                     output_name: str = "06_system_structure.png"):
        """
        Combined visualization: System structure + Measure grid in one image.
        Shows System bounding boxes, Part labels, and measure barlines (within each system).
        Blue lines represent actual barline positions (measure boundaries).
        
        Args:
            original_image: Original image
            system_groups: List of system groups (each group is a list of StaffSystem)
            part_indices: Dictionary mapping system index to part ID
            parts: List of assembled parts
            staff_systems: List of staff systems
            symbols: Optional list of all symbols (to extract barline positions)
        """
        vis_image = original_image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Draw staff lines (light gray)
        for system in staff_systems:
            for line_y in system.lines:
                cv2.line(vis_image, (0, int(line_y)), (vis_image.shape[1], int(line_y)),
                        (200, 200, 200), 1)
        
        # Step 1: Draw System bounding boxes (red)
        for group_idx, system_group in enumerate(system_groups):
            if not system_group:
                continue
            
            # Calculate bounding box for the entire system group
            top_system = min(system_group, key=lambda s: s.top_line)
            bottom_system = max(system_group, key=lambda s: s.bottom_line)
            
            # Find X range (use full image width for now, or could use symbol X range)
            x1 = 0
            x2 = vis_image.shape[1]
            y1 = top_system.top_line
            y2 = bottom_system.bottom_line
            
            # Draw System bounding box (red, thick)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Draw label: "System N"
            label = f"System {group_idx + 1}"
            cv2.putText(vis_image, label, (x1 + 10, y1 + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw Staff bounding boxes within System (green)
            for staff_idx, system in enumerate(system_group):
                staff_x1 = x1
                staff_x2 = x2
                staff_y1 = system.top_line
                staff_y2 = system.bottom_line
                
                # Draw Staff bounding box (green, medium thickness)
                cv2.rectangle(vis_image, (staff_x1, staff_y1), (staff_x2, staff_y2), (0, 255, 0), 2)
                
                # Draw Part label: "Part N"
                # Get part ID from part_indices (need to find system index)
                # For now, use staff_idx + 1
                part_id = staff_idx + 1
                part_label = f"Part {part_id}"
                cv2.putText(vis_image, part_label, (staff_x1 + 10, staff_y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Step 2: Draw measure barlines (blue) - only within each system's range
        # Collect barline positions from symbols, filtered by system
        # Each system should only show barlines that belong to it
        barline_x_by_system = {}  # system_idx -> set of barline X positions
        
        # Method 1: Extract barline positions from detected symbols, filtered by system
        if symbols:
            for sym in symbols:
                class_name_lower = sym.class_name.lower()
                # Check for barlines and measure_separators (which may be treated as barlines)
                is_barline = any(keyword in class_name_lower for keyword in ['barline', 'thin_barline', 'thick_barline', 'repeat'])
                is_measure_separator = 'measure_separator' in class_name_lower
                
                # measure_separator will be handled in builder, but we still need to check it here
                # for visualization purposes (it will be treated as barline if within single part)
                if is_barline or is_measure_separator:
                    # Find which system(s) this barline belongs to based on Y coordinate
                    sym_y = sym.center_y
                    for system_idx, system in enumerate(staff_systems):
                        # Check if barline center is within system range (with small margin)
                        if system.contains_y(sym_y, margin=system.avg_spacing * 0.5):
                            if system_idx not in barline_x_by_system:
                                barline_x_by_system[system_idx] = set()
                            barline_x_by_system[system_idx].add(sym.center_x)
        
        # Collect all unique barline positions for fallback
        barline_x_positions = set()
        for x_set in barline_x_by_system.values():
            barline_x_positions.update(x_set)
        
        # Method 2: If no barlines found in symbols, infer from measure boundaries
        if not barline_x_positions:
            # Collect measure boundaries: end of each measure (last note position)
            for part_idx, part in enumerate(parts):
                if part_idx >= len(staff_systems):
                    continue
                
                for measure in part.measures:
                    if measure.notes:
                        # Use last note X position as measure end (barline position)
                        last_note_x = max(note.x for note in measure.notes)
                        barline_x_positions.add(last_note_x)
        
        # Sort and merge close barline positions
        sorted_barline_xs = sorted(barline_x_positions)
        merged_barline_xs = []
        threshold = 50  # pixels
        
        if sorted_barline_xs:
            merged_barline_xs.append(sorted_barline_xs[0])
            for x in sorted_barline_xs[1:]:
                if abs(x - merged_barline_xs[-1]) > threshold:
                    merged_barline_xs.append(x)
                else:
                    # Merge with previous (average)
                    merged_barline_xs[-1] = (merged_barline_xs[-1] + x) / 2
        
        # Draw barlines (blue vertical lines) - only within each system's range
        # Use system-specific barline positions if available
        for barline_x in merged_barline_xs:
            # Find which systems should have this barline
            systems_with_barline = set()
            
            # First, check if this barline was detected in any system (from barline_x_by_system)
            for system_idx, x_set in barline_x_by_system.items():
                # Check if this barline_x is close to any barline in this system
                for sys_barline_x in x_set:
                    if abs(barline_x - sys_barline_x) < threshold:
                        systems_with_barline.add(system_idx)
                        break
            
            # Fallback: If no system-specific assignment, use measure-based logic
            if not systems_with_barline:
                for part_idx, part in enumerate(parts):
                    if part_idx >= len(staff_systems):
                        continue
                    
                    system = staff_systems[part_idx]
                    
                    # Check if any measure in this part is near this barline
                    for measure in part.measures:
                        if measure.notes:
                            first_note_x = min(note.x for note in measure.notes)
                            last_note_x = max(note.x for note in measure.notes)
                            # If barline is near the end of this measure, draw it
                            if abs(barline_x - last_note_x) < threshold or \
                               (first_note_x <= barline_x <= last_note_x):
                                systems_with_barline.add(part_idx)
                                break
            
            # Draw barline in each relevant system
            for system_idx in systems_with_barline:
                if system_idx < len(staff_systems):
                    system = staff_systems[system_idx]
                    y_top = system.top_line
                    y_bottom = system.bottom_line
                    cv2.line(vis_image, (int(barline_x), int(y_top)), (int(barline_x), int(y_bottom)),
                            (255, 0, 0), 2)  # Blue
        
        # Draw measure numbers above each measure (at measure center)
        for part_idx, part in enumerate(parts):
            if part_idx >= len(staff_systems):
                continue
            
            system = staff_systems[part_idx]
            y_top = system.top_line
            
            for measure in part.measures:
                if measure.notes:
                    # Calculate measure center for label placement
                    first_note_x = min(note.x for note in measure.notes)
                    last_note_x = max(note.x for note in measure.notes)
                    measure_center_x = (first_note_x + last_note_x) / 2
                    
                    # Draw measure number label
                    label = f"Meas {measure.number}"
                    cv2.putText(vis_image, label, (int(measure_center_x) - 30, int(y_top) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Step 3: Draw clef, time signature, and key signature boxes
        if symbols:
            for sym in symbols:
                class_name_lower = sym.class_name.lower()
                x1, y1, x2, y2 = [int(coord) for coord in sym.bbox]
                
                # Clef (Magenta box)
                if any(keyword in class_name_lower for keyword in ['g-clef', 'f-clef', 'c-clef', 'clef']):
                    # Check if clef is within any system
                    for system in staff_systems:
                        if system.contains_y(sym.center_y, margin=system.avg_spacing * 2):
                            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Magenta
                            # Add label
                            cv2.putText(vis_image, "Clef", (x1, y1 - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
                            break
                
                # Time signature (Cyan box)
                elif 'time_signature' in class_name_lower or 'letter_c' in class_name_lower:
                    for system in staff_systems:
                        if system.contains_y(sym.center_y, margin=system.avg_spacing * 2):
                            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 255, 0), 3)  # Cyan
                            # Add label
                            cv2.putText(vis_image, "Time", (x1, y1 - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            break
                
                # Key signature (Yellow box)
                elif 'key_signature' in class_name_lower:
                    for system in staff_systems:
                        if system.contains_y(sym.center_y, margin=system.avg_spacing * 2):
                            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 3)  # Yellow
                            # Add label
                            cv2.putText(vis_image, "Key", (x1, y1 - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                            break
        
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), vis_image)
        print(f"[Visualizer] Saved system structure with measures: {output_path}")
        return output_path
    
    def save_attribute_map(self, parts: List[AssembledPart],
                          output_name: str = "08_attribute_map.txt"):
        """
        Save attribute state flow as text.
        Implements visualization from rule.md: Attribute_Map.txt
        
        Args:
            parts: List of assembled parts
        """
        output_path = self.output_dir / output_name
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("Attribute State Flow\n")
            f.write("=" * 50 + "\n\n")
            
            for part_idx, part in enumerate(parts):
                f.write(f"Part {part_idx + 1}:\n")
                f.write("-" * 30 + "\n")
                
                prev_clef = None
                prev_key = None
                prev_time = None
                
                for measure in part.measures:
                    f.write(f"[Measure {measure.number}]\n")
                    
                    # Check for changes
                    clef_changed = measure.clef != prev_clef if prev_clef else True
                    key_changed = measure.key_signature != prev_key if prev_key else True
                    time_changed = measure.time_signature != prev_time if prev_time else True
                    
                    # Format clef
                    clef_str = measure.clef.value if hasattr(measure.clef, 'value') else str(measure.clef)
                    if clef_changed:
                        clef_str += " (Changed!)" if prev_clef else ""
                    
                    # Format key signature
                    key_str = measure.key_signature or "None"
                    if key_changed and measure.key_signature:
                        key_str += " (Changed!)" if prev_key else ""
                    elif not key_changed and prev_key:
                        key_str = "(Inherited)"
                    
                    # Format time signature
                    time_str = measure.time_signature or "None"
                    if time_changed and measure.time_signature:
                        time_str += " (Changed!)" if prev_time else ""
                    elif not time_changed and prev_time:
                        time_str = "(Inherited)"
                    
                    # Write attributes
                    f.write(f"  Clef: {clef_str}\n")
                    f.write(f"  Key: {key_str}\n")
                    f.write(f"  Time: {time_str}\n")
                    
                    if measure.is_implicit:
                        f.write(f"  [Anacrusis/Pickup Measure]\n")
                    
                    f.write("\n")
                    
                    # Update previous values
                    prev_clef = measure.clef
                    prev_key = measure.key_signature
                    prev_time = measure.time_signature
        
        print(f"[Visualizer] Saved attribute map: {output_path}")
        return output_path
    
    def visualize_triplets(self, original_image: np.ndarray, parts: List[AssembledPart],
                          output_name: str = "07_triplets.png"):
        """
        Visualize detected triplets with different colors for different rules.
        
        Color scheme:
        - Rule1 (Bracket-based): Red
        - Rule2 (Beam-based): Blue
        - Rule3 (Loose number): Green
        - Rule5 (Sanity check): Yellow
        
        Args:
            original_image: Original image
            parts: List of assembled parts
            output_name: Output filename
        """
        vis_image = original_image.copy()
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Color mapping for different rules
        rule_colors = {
            "Rule1": (0, 0, 255),      # Red (BGR)
            "Rule2": (255, 0, 0),      # Blue (BGR)
            "Rule3": (0, 255, 0),      # Green (BGR)
            "Rule5": (0, 255, 255),    # Yellow (BGR)
        }
        
        # Collect all triplet notes grouped by rule and measure
        triplet_groups_by_rule = {}
        
        for part in parts:
            for measure in part.measures:
                # Find all triplet notes in this measure
                triplet_notes = [note for note in measure.notes 
                               if note.is_tuplet and note.tuplet_type == "triplet" 
                               and note.tuplet_rule_triggered]
                
                if not triplet_notes:
                    continue
                
                # Group triplet notes by rule
                # Notes with same rule are grouped together
                triplet_notes.sort(key=lambda n: (n.tuplet_rule_triggered or "", n.x))
                
                # Group by rule
                notes_by_rule = {}
                for note in triplet_notes:
                    rule = note.tuplet_rule_triggered
                    if rule not in notes_by_rule:
                        notes_by_rule[rule] = []
                    notes_by_rule[rule].append(note)
                
                # For each rule, group notes by proximity (X coordinate)
                for rule, rule_notes in notes_by_rule.items():
                    rule_notes.sort(key=lambda n: n.x)
                    
                    # Group consecutive notes with small gaps
                    current_group = []
                    max_gap = 100  # Maximum X gap between notes in the same triplet group
                    
                    for i, note in enumerate(rule_notes):
                        if not current_group:
                            current_group = [note]
                        else:
                            # Check if note is close enough to last note in current group
                            last_note_x = current_group[-1].x
                            if abs(note.x - last_note_x) <= max_gap:
                                current_group.append(note)
                            else:
                                # Save current group if it has at least 3 notes
                                if len(current_group) >= 3:
                                    if rule not in triplet_groups_by_rule:
                                        triplet_groups_by_rule[rule] = []
                                    triplet_groups_by_rule[rule].append(current_group)
                                current_group = [note]
                    
                    # Save last group
                    if len(current_group) >= 3:
                        if rule not in triplet_groups_by_rule:
                            triplet_groups_by_rule[rule] = []
                        triplet_groups_by_rule[rule].append(current_group)
        
        triplet_count = sum(len(groups) for groups in triplet_groups_by_rule.values())
        
        # Debug: print all groups found
        print(f"[Visualizer] Debug: Found {triplet_count} triplet groups total")
        for rule, groups in triplet_groups_by_rule.items():
            print(f"  {rule}: {len(groups)} groups")
            for i, group in enumerate(groups, 1):
                notes_with_symbols = sum(1 for n in group if n.original_symbol)
                print(f"    Group {i}: {len(group)} notes ({notes_with_symbols} with symbols)")
        
        # Deduplicate: if same notes appear in multiple rules, only draw once (use highest priority rule)
        # Also deduplicate groups with overlapping bounding boxes
        # Priority: Rule1 > Rule2 > Rule3 > Rule5
        rule_priority = {"Rule1": 1, "Rule2": 2, "Rule3": 3, "Rule5": 5}
        processed_note_ids = set()  # Track which notes have been drawn
        processed_bboxes = []  # Track bounding boxes of drawn groups
        deduplicated_groups = []
        
        def get_group_bbox(group):
            """Calculate bounding box for a group of notes"""
            note_xs = []
            note_ys = []
            for note in group:
                if note.original_symbol:
                    sym = note.original_symbol
                    note_xs.extend([sym.x1, sym.x2])
                    note_ys.extend([sym.y1, sym.y2])
                else:
                    note_xs.append(note.x)
                    note_xs.append(note.x + 20)
                    note_ys.append(vis_image.shape[0] // 2)
                    note_ys.append(vis_image.shape[0] // 2 + 20)
            if not note_xs:
                return None
            return (min(note_xs), min(note_ys), max(note_xs), max(note_ys))
        
        def bboxes_overlap(bbox1, bbox2, threshold=50):
            """Check if two bounding boxes overlap significantly"""
            if bbox1 is None or bbox2 is None:
                return False
            x1_min, y1_min, x1_max, y1_max = bbox1
            x2_min, y2_min, x2_max, y2_max = bbox2
            # Check if centers are close (within threshold)
            center1_x = (x1_min + x1_max) / 2
            center1_y = (y1_min + y1_max) / 2
            center2_x = (x2_min + x2_max) / 2
            center2_y = (y2_min + y2_max) / 2
            dist = ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5
            return dist < threshold
        
        # Process groups in priority order
        for rule in ["Rule1", "Rule2", "Rule3", "Rule5"]:
            if rule not in triplet_groups_by_rule:
                continue
            for group in triplet_groups_by_rule[rule]:
                # Check if any note in this group has already been processed
                group_note_ids = {id(note) for note in group}
                if group_note_ids & processed_note_ids:
                    # Some notes already processed by higher priority rule, skip this group
                    print(f"[Visualizer] Skipping duplicate group from {rule} (notes already processed by higher priority rule)")
                    continue
                
                # Check if bounding box overlaps with already processed groups
                group_bbox = get_group_bbox(group)
                is_duplicate = False
                for processed_bbox in processed_bboxes:
                    if bboxes_overlap(group_bbox, processed_bbox):
                        print(f"[Visualizer] Skipping duplicate group from {rule} (overlapping bbox with higher priority rule)")
                        is_duplicate = True
                        break
                
                if is_duplicate:
                    continue
                
                # Mark notes as processed and add to deduplicated list
                processed_note_ids.update(group_note_ids)
                if group_bbox:
                    processed_bboxes.append(group_bbox)
                deduplicated_groups.append((rule, group))
        
        print(f"[Visualizer] After deduplication: {len(deduplicated_groups)} unique triplet groups")
        
        # Draw triplet groups
        drawn_count = 0
        for rule, group in deduplicated_groups:
            color = rule_colors.get(rule, (128, 128, 128))  # Gray for unknown rules
            
            if len(group) < 3:
                continue
            
            # Get bounding box for the triplet group
            note_xs = []
            note_ys = []
            bboxes = []
            notes_with_symbols = []
            
            for note in group:
                if note.original_symbol:
                    sym = note.original_symbol
                    x1, y1, x2, y2 = sym.x1, sym.y1, sym.x2, sym.y2
                    bboxes.append((x1, y1, x2, y2))
                    note_xs.extend([x1, x2])
                    note_ys.extend([y1, y2])
                    notes_with_symbols.append(note)
                else:
                    # For notes without original_symbol, use x coordinate and estimate bbox
                    # This can happen for rests or virtual notes
                    note_xs.append(note.x)
                    note_xs.append(note.x + 20)  # Estimate width
                    # Use a default Y position (middle of staff) if not available
                    note_ys.append(vis_image.shape[0] // 2)
                    note_ys.append(vis_image.shape[0] // 2 + 20)
            
            # If no notes have symbols, skip this group (can't visualize)
            if not notes_with_symbols:
                print(f"[Visualizer] Warning: Triplet group with {len(group)} notes has no original_symbol, skipping visualization")
                continue
            
            # If some notes don't have symbols, still draw what we can
            if len(notes_with_symbols) < len(group):
                print(f"[Visualizer] Warning: Triplet group has {len(group)} notes but only {len(notes_with_symbols)} have original_symbol")
            
            # Calculate group bounding box
            min_x = min(note_xs)
            max_x = max(note_xs)
            min_y = min(note_ys)
            max_y = max(note_ys)
            
            # Draw bounding box around the triplet group
            cv2.rectangle(vis_image, (int(min_x), int(min_y)), (int(max_x), int(max_y)),
                         color, 3)
            
            # Draw individual note bounding boxes
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)),
                             color, 2)
            
            # Draw label with rule name
            label = f"{rule}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_x = int(min_x)
            label_y = int(min_y) - 10
            if label_y < 20:
                label_y = int(max_y) + 25
            
            # Draw background for label
            cv2.rectangle(vis_image, 
                         (label_x - 2, label_y - label_size[1] - 2),
                         (label_x + label_size[0] + 2, label_y + 2),
                         color, -1)
            cv2.putText(vis_image, label, (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            drawn_count += 1
            print(f"[Visualizer] Drawn triplet group {drawn_count}: {rule}, {len(group)} notes, bbox=[{int(min_x)}, {int(min_y)}, {int(max_x)}, {int(max_y)}]")
        
        # Draw legend
        legend_y = 30
        legend_x = vis_image.shape[1] - 200
        cv2.putText(vis_image, "Triplet Rules:", (legend_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 25
        
        for rule, color in rule_colors.items():
            cv2.rectangle(vis_image, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5),
                         color, -1)
            cv2.putText(vis_image, rule, (legend_x + 25, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 25
        
        # Add summary text
        summary_text = f"Total Triplets: {triplet_count}"
        cv2.putText(vis_image, summary_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        output_path = self.output_dir / output_name
        cv2.imwrite(str(output_path), vis_image)
        print(f"[Visualizer] Saved triplet visualization: {output_path}")
        print(f"[Visualizer] Found {triplet_count} triplet groups")
        for rule, groups in triplet_groups_by_rule.items():
            print(f"  {rule}: {len(groups)} groups")
        return output_path


