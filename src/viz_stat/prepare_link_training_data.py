"""
Prepare training data for link prediction model from multiple XML annotation files
Extracts positive and negative link pairs for training
"""
import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import json
import random
import numpy as np
from collections import defaultdict


def parse_xml_annotations(xml_path: Path) -> Tuple[List[Dict], Dict[int, Dict]]:
    """
    Parse CropObject XML file and extract annotations with outlinks
    
    Args:
        xml_path: Path to XML annotation file
        
    Returns:
        Tuple of (annotations list, id_to_ann mapping)
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    
    annotations = []
    id_to_ann = {}
    
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
        
        ann = {
            'id': obj_id,
            'class_name': class_name,
            'top': top,
            'left': left,
            'width': width,
            'height': height,
            'outlinks': outlinks,
            'center_x': left + width // 2,
            'center_y': top + height // 2
        }
        
        annotations.append(ann)
        id_to_ann[obj_id] = ann
    
    return annotations, id_to_ann


def compute_spatial_features(source: Dict, target: Dict, image_width: int = None, image_height: int = None) -> Dict:
    """
    Compute spatial features between two objects
    
    Args:
        source: Source object annotation
        target: Target object annotation
        image_width: Image width for normalization (optional)
        image_height: Image height for normalization (optional)
        
    Returns:
        Dictionary of spatial features
    """
    # Distance between centers
    dx = target['center_x'] - source['center_x']
    dy = target['center_y'] - source['center_y']
    distance = np.sqrt(dx**2 + dy**2)
    
    # Normalized distance (if image size provided)
    if image_width and image_height:
        norm_distance = distance / np.sqrt(image_width**2 + image_height**2)
    else:
        norm_distance = None
    
    # Angle/direction
    angle = np.arctan2(dy, dx)  # in radians
    
    # Relative position
    relative_x = dx / (source['width'] + 1e-6) if source['width'] > 0 else 0
    relative_y = dy / (source['height'] + 1e-6) if source['height'] > 0 else 0
    
    # Size ratio
    size_ratio_w = target['width'] / (source['width'] + 1e-6)
    size_ratio_h = target['height'] / (source['height'] + 1e-6)
    
    # Overlap (IoU)
    x1_s = source['left']
    y1_s = source['top']
    x2_s = source['left'] + source['width']
    y2_s = source['top'] + source['height']
    
    x1_t = target['left']
    y1_t = target['top']
    x2_t = target['left'] + target['width']
    y2_t = target['top'] + target['height']
    
    # Intersection
    x1_i = max(x1_s, x1_t)
    y1_i = max(y1_s, y1_t)
    x2_i = min(x2_s, x2_t)
    y2_i = min(y2_s, y2_t)
    
    if x2_i > x1_i and y2_i > y1_i:
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
    else:
        intersection = 0
    
    area_s = source['width'] * source['height']
    area_t = target['width'] * target['height']
    union = area_s + area_t - intersection
    iou = intersection / (union + 1e-6)
    
    features = {
        'distance': float(distance),
        'norm_distance': float(norm_distance) if norm_distance is not None else 0.0,
        'dx': float(dx),
        'dy': float(dy),
        'angle': float(angle),
        'relative_x': float(relative_x),
        'relative_y': float(relative_y),
        'size_ratio_w': float(size_ratio_w),
        'size_ratio_h': float(size_ratio_h),
        'iou': float(iou),
        'intersection': float(intersection),
        'source_area': float(area_s),
        'target_area': float(area_t)
    }
    
    return features


def extract_link_pairs(xml_path: Path, image_width: int = None, image_height: int = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract positive and negative link pairs from XML file
    
    Args:
        xml_path: Path to XML annotation file
        image_width: Image width for normalization
        image_height: Image height for normalization
        
    Returns:
        Tuple of (positive_pairs, negative_pairs)
    """
    annotations, id_to_ann = parse_xml_annotations(xml_path)
    
    # Extract positive pairs (existing links)
    positive_pairs = []
    all_links = set()  # Track all existing links
    
    for ann in annotations:
        source_id = ann['id']
        for target_id in ann['outlinks']:
            if target_id in id_to_ann:
                target_ann = id_to_ann[target_id]
                # Create link pair (both directions for undirected graph, or just one for directed)
                link_key = (min(source_id, target_id), max(source_id, target_id))
                if link_key not in all_links:
                    all_links.add(link_key)
                    
                    # Compute spatial features
                    spatial_features = compute_spatial_features(ann, target_ann, image_width, image_height)
                    
                    positive_pairs.append({
                        'source_id': source_id,
                        'target_id': target_id,
                        'source_class': ann['class_name'],
                        'target_class': target_ann['class_name'],
                        'source_bbox': [ann['top'], ann['left'], ann['width'], ann['height']],
                        'target_bbox': [target_ann['top'], target_ann['left'], target_ann['width'], target_ann['height']],
                        'spatial_features': spatial_features,
                        'label': 1
                    })
    
    # Generate negative pairs (no link exists)
    # Strategy: sample random pairs that don't have links
    negative_pairs = []
    all_ids = list(id_to_ann.keys())
    
    # Generate negative samples (same number as positive, or up to N times)
    num_negative = len(positive_pairs)  # Balanced by default
    
    for _ in range(num_negative * 2):  # Try more to account for duplicates
        if len(negative_pairs) >= num_negative:
            break
            
        source_id = random.choice(all_ids)
        target_id = random.choice(all_ids)
        
        if source_id == target_id:
            continue
        
        link_key = (min(source_id, target_id), max(source_id, target_id))
        if link_key not in all_links:
            source_ann = id_to_ann[source_id]
            target_ann = id_to_ann[target_id]
            
            # Compute spatial features
            spatial_features = compute_spatial_features(source_ann, target_ann, image_width, image_height)
            
            negative_pairs.append({
                'source_id': source_id,
                'target_id': target_id,
                'source_class': source_ann['class_name'],
                'target_class': target_ann['class_name'],
                'source_bbox': [source_ann['top'], source_ann['left'], source_ann['width'], source_ann['height']],
                'target_bbox': [target_ann['top'], target_ann['left'], target_ann['width'], target_ann['height']],
                'spatial_features': spatial_features,
                'label': 0
            })
    
    return positive_pairs, negative_pairs[:num_negative]


def process_multiple_xml_files(xml_dir: Path, output_dir: Path, negative_ratio: float = 1.0):
    """
    Process multiple XML files and extract link pairs
    
    Args:
        xml_dir: Directory containing XML annotation files
        output_dir: Output directory for processed data
        negative_ratio: Ratio of negative to positive samples (1.0 = balanced)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all XML files
    xml_files = list(xml_dir.glob('**/*.xml'))
    print(f"Found {len(xml_files)} XML files")
    
    all_positive = []
    all_negative = []
    file_stats = []
    
    for xml_file in xml_files:
        print(f"\nProcessing: {xml_file.name}")
        try:
            positive_pairs, negative_pairs = extract_link_pairs(xml_file)
            
            all_positive.extend(positive_pairs)
            all_negative.extend(negative_pairs)
            
            stats = {
                'file': xml_file.name,
                'positive_pairs': len(positive_pairs),
                'negative_pairs': len(negative_pairs),
                'total_objects': len(parse_xml_annotations(xml_file)[0])
            }
            file_stats.append(stats)
            print(f"  Positive pairs: {len(positive_pairs)}")
            print(f"  Negative pairs: {len(negative_pairs)}")
            
        except Exception as e:
            print(f"  Error processing {xml_file.name}: {e}")
            continue
    
    # Adjust negative samples if needed
    if negative_ratio != 1.0:
        num_negative = int(len(all_positive) * negative_ratio)
        random.shuffle(all_negative)
        all_negative = all_negative[:num_negative]
    
    # Combine and shuffle
    all_pairs = all_positive + all_negative
    random.shuffle(all_pairs)
    
    # Save data
    output_file = output_dir / 'link_training_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'pairs': all_pairs,
            'statistics': {
                'total_pairs': len(all_pairs),
                'positive_pairs': len(all_positive),
                'negative_pairs': len(all_negative),
                'num_files': len(file_stats),
                'file_stats': file_stats
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total files processed: {len(file_stats)}")
    print(f"Total positive pairs: {len(all_positive)}")
    print(f"Total negative pairs: {len(all_negative)}")
    print(f"Total training pairs: {len(all_pairs)}")
    print(f"\nData saved to: {output_file}")
    
    # Class statistics
    class_link_stats = defaultdict(lambda: {'positive': 0, 'negative': 0})
    for pair in all_pairs:
        key = f"{pair['source_class']} -> {pair['target_class']}"
        if pair['label'] == 1:
            class_link_stats[key]['positive'] += 1
        else:
            class_link_stats[key]['negative'] += 1
    
    stats_file = output_dir / 'class_link_statistics.json'
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(dict(class_link_stats), f, indent=2, ensure_ascii=False)
    
    print(f"Class statistics saved to: {stats_file}")
    
    return output_file


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Prepare link training data from XML annotations")
    parser.add_argument(
        'xml_dir',
        type=str,
        help='Directory containing XML annotation files'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output directory (default: Output/link_training_data)'
    )
    parser.add_argument(
        '--negative-ratio',
        type=float,
        default=1.0,
        help='Ratio of negative to positive samples (default: 1.0 = balanced)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Convert to Path objects
    xml_dir = Path(args.xml_dir)
    if not xml_dir.is_absolute():
        xml_dir = Path(project_root) / xml_dir
    
    if not xml_dir.exists():
        raise FileNotFoundError(f"XML directory not found: {xml_dir}")
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = Path(project_root) / output_dir
    else:
        output_dir = Path(project_root) / "Output" / "link_training_data"
    
    # Process files
    process_multiple_xml_files(xml_dir, output_dir, args.negative_ratio)


if __name__ == '__main__':
    main()


