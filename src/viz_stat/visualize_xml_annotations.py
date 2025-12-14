"""
Visualize XML annotations on images
Draw bounding boxes and labels from CropObject XML files
"""
import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import random


def parse_xml_annotations(xml_path: Path) -> List[Dict]:
    """
    Parse CropObject XML file and extract annotations
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        List of annotation dictionaries with keys: id, class_name, top, left, width, height, outlinks
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    
    annotations = []
    for crop_obj in root.findall('.//CropObject'):
        obj_id = int(crop_obj.find('Id').text)
        
        # Get class name
        class_name_elem = crop_obj.find('ClassName')
        if class_name_elem is None:
            class_name_elem = crop_obj.find('MLClassName')
        if class_name_elem is None:
            continue
        class_name = class_name_elem.text
        
        # Get bounding box coordinates
        top = int(crop_obj.find('Top').text)
        left = int(crop_obj.find('Left').text)
        width = int(crop_obj.find('Width').text)
        height = int(crop_obj.find('Height').text)
        
        # Get outlinks
        outlinks_elem = crop_obj.find('Outlinks')
        outlinks = []
        if outlinks_elem is not None and outlinks_elem.text:
            outlinks = [int(x) for x in outlinks_elem.text.strip().split()]
        
        annotations.append({
            'id': obj_id,
            'class_name': class_name,
            'top': top,
            'left': left,
            'width': width,
            'height': height,
            'outlinks': outlinks
        })
    
    return annotations


def get_color_for_class(class_name: str, color_map: Dict[str, Tuple[int, int, int]] = None) -> Tuple[int, int, int]:
    """
    Get consistent color for a class name
    
    Args:
        class_name: Class name string
        color_map: Optional pre-defined color map
        
    Returns:
        BGR color tuple
    """
    if color_map and class_name in color_map:
        return color_map[class_name]
    
    # Generate consistent color based on class name hash
    random.seed(hash(class_name))
    color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    random.seed()  # Reset seed
    return color


def visualize_annotations(
    image_path: Path,
    xml_path: Path,
    output_path: Path = None,
    show_labels: bool = False,
    box_thickness: int = 2,
    link_thickness: int = 1,
    font_scale: float = 0.5,
    font_thickness: int = 1,
    max_annotations: int = None
) -> np.ndarray:
    """
    Visualize XML annotations on image
    
    Args:
        image_path: Path to input image
        xml_path: Path to XML annotation file
        output_path: Optional path to save visualization
        show_labels: Whether to show class name labels (default: False)
        box_thickness: Thickness of bounding box lines
        link_thickness: Thickness of outlink connection lines
        font_scale: Font scale for labels
        font_thickness: Font thickness for labels
        max_annotations: Maximum number of annotations to display (None for all)
        
    Returns:
        Annotated image as numpy array
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Parse annotations
    annotations = parse_xml_annotations(xml_path)
    print(f"Found {len(annotations)} annotations in XML file")
    
    # Limit annotations if specified
    if max_annotations:
        annotations = annotations[:max_annotations]
        print(f"Displaying first {len(annotations)} annotations")
    
    # Create ID to annotation mapping for quick lookup
    id_to_ann = {ann['id']: ann for ann in annotations}
    
    # Get unique classes for color mapping
    unique_classes = list(set([ann['class_name'] for ann in annotations]))
    color_map = {}
    for cls in unique_classes:
        color_map[cls] = get_color_for_class(cls)
    
    # Calculate center points for all annotations
    for ann in annotations:
        ann['center_x'] = ann['left'] + ann['width'] // 2
        ann['center_y'] = ann['top'] + ann['height'] // 2
    
    # Draw outlink connections first (so boxes appear on top)
    link_color = (200, 200, 200)  # Light gray for connections
    total_links = 0
    for ann in annotations:
        source_id = ann['id']
        source_x = ann['center_x']
        source_y = ann['center_y']
        
        # Draw lines to all outlinked objects
        for target_id in ann['outlinks']:
            if target_id in id_to_ann:
                target_ann = id_to_ann[target_id]
                target_x = target_ann['center_x']
                target_y = target_ann['center_y']
                
                # Draw line connecting centers
                cv2.line(image, (source_x, source_y), (target_x, target_y), link_color, link_thickness)
                total_links += 1
    
    print(f"Drew {total_links} outlink connections")
    
    # Draw bounding boxes
    for ann in annotations:
        class_name = ann['class_name']
        top = ann['top']
        left = ann['left']
        width = ann['width']
        height = ann['height']
        
        # Calculate bounding box coordinates
        x1 = left
        y1 = top
        x2 = left + width
        y2 = top + height
        
        # Get color for this class
        color = color_map[class_name]
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness)
        
        # Draw label (only if show_labels is True)
        if show_labels:
            label = f"{class_name} ({ann['id']})"
            
            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )
            
            # Draw label background
            label_y = max(y1, text_height + 5)
            cv2.rectangle(
                image,
                (x1, label_y - text_height - 5),
                (x1 + text_width + 5, label_y + baseline),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                image,
                label,
                (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                font_thickness,
                cv2.LINE_AA
            )
    
    # Add legend
    legend_y = 30
    legend_x = 10
    cv2.putText(
        image,
        "Legend:",
        (legend_x, legend_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )
    legend_y += 30
    
    for i, (cls, color) in enumerate(color_map.items()):
        if legend_y > image.shape[0] - 50:
            legend_x += 200
            legend_y = 60
        
        # Draw color box
        cv2.rectangle(
            image,
            (legend_x, legend_y - 15),
            (legend_x + 20, legend_y),
            color,
            -1
        )
        
        # Draw class name
        cv2.putText(
            image,
            cls,
            (legend_x + 25, legend_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        legend_y += 20
    
    # Save if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        print(f"Visualization saved to: {output_path}")
    
    return image


def statistics_class_links(xml_path: Path, output_path: Path = None) -> Dict:
    """
    Statistics of class-to-class links from Outlinks
    
    Args:
        xml_path: Path to XML annotation file
        output_path: Optional path to save statistics (CSV or TXT)
        
    Returns:
        Dictionary containing statistics
    """
    # Parse annotations
    annotations = parse_xml_annotations(xml_path)
    print(f"Found {len(annotations)} annotations in XML file")
    
    # Create ID to annotation mapping
    id_to_ann = {ann['id']: ann for ann in annotations}
    
    # Statistics: source_class -> {target_class: count}
    class_link_stats = {}
    total_links = 0
    valid_links = 0
    invalid_links = 0
    
    for ann in annotations:
        source_class = ann['class_name']
        source_id = ann['id']
        
        if source_class not in class_link_stats:
            class_link_stats[source_class] = {}
        
        # Count links to each target class
        for target_id in ann['outlinks']:
            total_links += 1
            if target_id in id_to_ann:
                valid_links += 1
                target_class = id_to_ann[target_id]['class_name']
                if target_class not in class_link_stats[source_class]:
                    class_link_stats[source_class][target_class] = 0
                class_link_stats[source_class][target_class] += 1
            else:
                invalid_links += 1
    
    # Print statistics table
    print("\n" + "="*80)
    print("Class Link Statistics")
    print("="*80)
    print(f"Total annotations: {len(annotations)}")
    print(f"Total outlinks: {total_links}")
    print(f"Valid links: {valid_links}")
    print(f"Invalid links (target ID not found): {invalid_links}")
    print("\n" + "-"*80)
    print(f"{'Source Class':<30} {'Target Class':<30} {'Count':<10}")
    print("-"*80)
    
    # Sort by source class, then by count
    sorted_sources = sorted(class_link_stats.keys())
    all_links = []
    
    for source_class in sorted_sources:
        target_counts = class_link_stats[source_class]
        sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
        
        for target_class, count in sorted_targets:
            print(f"{source_class:<30} {target_class:<30} {count:<10}")
            all_links.append({
                'source_class': source_class,
                'target_class': target_class,
                'count': count
            })
    
    print("-"*80)
    
    # Summary by source class
    print("\n" + "="*80)
    print("Summary by Source Class")
    print("="*80)
    print(f"{'Source Class':<30} {'Total Links':<15} {'Unique Targets':<15}")
    print("-"*80)
    
    for source_class in sorted_sources:
        total = sum(class_link_stats[source_class].values())
        unique = len(class_link_stats[source_class])
        print(f"{source_class:<30} {total:<15} {unique:<15}")
    
    # Summary by target class
    print("\n" + "="*80)
    print("Summary by Target Class")
    print("="*80)
    target_stats = {}
    for source_class, targets in class_link_stats.items():
        for target_class, count in targets.items():
            if target_class not in target_stats:
                target_stats[target_class] = 0
            target_stats[target_class] += count
    
    sorted_targets = sorted(target_stats.items(), key=lambda x: x[1], reverse=True)
    print(f"{'Target Class':<30} {'Total Incoming Links':<20}")
    print("-"*80)
    for target_class, count in sorted_targets:
        print(f"{target_class:<30} {count:<20}")
    
    # Save to file if output path provided
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix.lower() == '.csv':
            # Save as CSV
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Source Class', 'Target Class', 'Count'])
                for link in all_links:
                    writer.writerow([link['source_class'], link['target_class'], link['count']])
            print(f"\nStatistics saved to CSV: {output_path}")
        else:
            # Save as text file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Class Link Statistics\n")
                f.write("="*80 + "\n")
                f.write(f"Total annotations: {len(annotations)}\n")
                f.write(f"Total outlinks: {total_links}\n")
                f.write(f"Valid links: {valid_links}\n")
                f.write(f"Invalid links: {invalid_links}\n\n")
                
                f.write("-"*80 + "\n")
                f.write(f"{'Source Class':<30} {'Target Class':<30} {'Count':<10}\n")
                f.write("-"*80 + "\n")
                
                for link in all_links:
                    f.write(f"{link['source_class']:<30} {link['target_class']:<30} {link['count']:<10}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("Summary by Source Class\n")
                f.write("="*80 + "\n")
                f.write(f"{'Source Class':<30} {'Total Links':<15} {'Unique Targets':<15}\n")
                f.write("-"*80 + "\n")
                
                for source_class in sorted_sources:
                    total = sum(class_link_stats[source_class].values())
                    unique = len(class_link_stats[source_class])
                    f.write(f"{source_class:<30} {total:<15} {unique:<15}\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("Summary by Target Class\n")
                f.write("="*80 + "\n")
                f.write(f"{'Target Class':<30} {'Total Incoming Links':<20}\n")
                f.write("-"*80 + "\n")
                for target_class, count in sorted_targets:
                    f.write(f"{target_class:<30} {count:<20}\n")
            
            print(f"\nStatistics saved to: {output_path}")
    
    return {
        'total_annotations': len(annotations),
        'total_links': total_links,
        'valid_links': valid_links,
        'invalid_links': invalid_links,
        'class_link_stats': class_link_stats,
        'all_links': all_links
    }


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Visualize XML annotations on images or statistics")
    parser.add_argument(
        'xml_path',
        type=str,
        help='Path to XML annotation file'
    )
    parser.add_argument(
        'image_path',
        type=str,
        nargs='?',
        default=None,
        help='Path to input image (optional, for visualization)'
    )
    parser.add_argument(
        '--stats-only',
        action='store_true',
        help='Only show statistics, skip visualization'
    )
    parser.add_argument(
        '--stats-output',
        type=str,
        default=None,
        help='Output path for statistics (CSV or TXT file)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output path for visualization (default: Output/<image_name>_annotated.png)'
    )
    parser.add_argument(
        '--show-labels',
        action='store_true',
        help='Show class name labels on boxes (default: False)'
    )
    parser.add_argument(
        '--box-thickness',
        type=int,
        default=2,
        help='Thickness of bounding box lines (default: 2)'
    )
    parser.add_argument(
        '--link-thickness',
        type=int,
        default=1,
        help='Thickness of outlink connection lines (default: 1)'
    )
    parser.add_argument(
        '--font-scale',
        type=float,
        default=0.5,
        help='Font scale for labels (default: 0.5)'
    )
    parser.add_argument(
        '--max-annotations',
        type=int,
        default=None,
        help='Maximum number of annotations to display (default: all)'
    )
    parser.add_argument(
        '--display',
        action='store_true',
        help='Display visualization window (requires X11)'
    )
    
    args = parser.parse_args()
    
    # Convert project_root to Path object
    project_root_path = Path(project_root)
    
    # Convert to Path objects and resolve relative to project root
    xml_path = Path(args.xml_path)
    if not xml_path.is_absolute():
        xml_path = project_root_path / xml_path
    
    # Validate XML path
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")
    
    # Statistics output path
    stats_output_path = None
    if args.stats_output:
        stats_output_path = Path(args.stats_output)
        if not stats_output_path.is_absolute():
            stats_output_path = project_root_path / stats_output_path
    elif args.stats_only:
        # Default stats output if stats-only mode
        stats_output_path = project_root_path / "Output" / "annotations" / f"{xml_path.stem}_link_stats.txt"
    
    # Generate statistics
    statistics_class_links(xml_path, stats_output_path)
    
    # Skip visualization if stats-only mode
    if args.stats_only:
        return
    
    # Visualization requires image path
    if args.image_path is None:
        print("\nError: image_path is required for visualization. Use --stats-only for statistics only.")
        return
    
    image_path = Path(args.image_path)
    if not image_path.is_absolute():
        image_path = project_root_path / image_path
    
    # Validate image path
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = project_root_path / output_path
    else:
        # Default to Output directory
        output_dir = project_root_path / "Output" / "annotations"
        output_path = output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
    
    # Create visualization
    annotated_image = visualize_annotations(
        image_path=image_path,
        xml_path=xml_path,
        output_path=output_path,
        show_labels=args.show_labels,
        box_thickness=args.box_thickness,
        link_thickness=args.link_thickness,
        font_scale=args.font_scale,
        max_annotations=args.max_annotations
    )
    
    # Display if requested
    if args.display:
        cv2.imshow('Annotated Image', annotated_image)
        print("Press any key to close the window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

