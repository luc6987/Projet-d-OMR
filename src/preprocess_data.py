#!/usr/bin/env python3
"""
YOLOv8l fine-tuning data preprocessing script
Generate YOLO format training data from CVC-MUSCIMA dataset
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import random
import shutil
from pathlib import Path
import yaml
from typing import List, Tuple, Dict
import argparse

class CVCDataPreprocessor:
    def __init__(self, source_dir: str, output_dir: str):
        """
        Initialize data preprocessor
        
        Args:
            source_dir: v1.0 data directory path
            output_dir: output dataset directory path
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory structure
        self.setup_output_dirs()
        
        # Load class mapping from official class definition file (will be filtered to Top-K by frequency before preprocessing)
        self.class_mapping = self.load_class_mapping()
        self.top_k_classes = 73
        self.selected_classes: List[str] = []
        
        # Sampling parameters
        self.sample_size = 1216  # Sampling size
        self.target_size = 640   # Target size
        self.num_samples = 14    # Number of samples per image
        
        # Visualization flags and statistics
        self.visualization_created = False
        self.stats = {
            'processed_images': 0,
            'total_symbols': 0,
            'symbol_relationships': 0,
            'symbol_types': {}
        }
        
    def setup_output_dirs(self):
        """Create output directory structure"""
        dirs = [
            self.output_dir / 'images' / 'train',
            self.output_dir / 'images' / 'val',
            self.output_dir / 'images' / 'test',
            self.output_dir / 'labels' / 'train',
            self.output_dir / 'labels' / 'val',
            self.output_dir / 'labels' / 'test'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create visualization output directory
        project_root = Path(__file__).resolve().parent
        vis_dir = project_root / 'Output' / 'preprocess'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    def load_class_mapping(self) -> Dict[str, int]:
        """
        Load class mapping from official class definition file
        
        Returns:
            Dictionary mapping class names to IDs
        """
        class_mapping = {}
        
        try:
            # Read official class definition file
            class_def_path = self.source_dir / 'specifications' / 'mff-muscima-mlclasses-annot.xml'
            
            if not class_def_path.exists():
                print(f"Warning: Class definition file does not exist: {class_def_path}")
                print("Using default class mapping...")
                return self.get_default_class_mapping()
            
            tree = ET.parse(class_def_path)
            root = tree.getroot()
            
            # Parse all class definitions
            for crop_object_class in root.findall('.//CropObjectClass'):
                class_id = int(crop_object_class.find('Id').text)
                class_name = crop_object_class.find('Name').text
                class_mapping[class_name] = class_id
            
            print(f"Successfully loaded {len(class_mapping)} official class definitions")
            
        except Exception as e:
            print(f"Error: Unable to load class definition file: {e}")
            print("Using default class mapping...")
            return self.get_default_class_mapping()
        
        return class_mapping
    
    def get_default_class_mapping(self) -> Dict[str, int]:
        """
        Get default class mapping (as backup)
        
        Returns:
            Default class mapping dictionary
        """
        return {
            'notehead-full': 0,
            'notehead-empty': 1,
            'stem': 2,
            '8th_flag': 3,
            '16th_flag': 4,
            'beam': 7,
            'duration-dot': 8,
            'sharp': 9,
            'flat': 10,
            'natural': 11,
            'whole_rest': 14,
            'half_rest': 15,
            'quarter_rest': 16,
            '8th_rest': 17,
            '16th_rest': 18,
            'ledger_line': 23,
            'grace-notehead-full': 25,
            'thin_barline': 38,
            'measure_separator': 40,
            'repeat-dot': 42,
            'staccato-dot': 46,
            'other-dot': 47,
            'tenuto': 48,
            'accent': 49,
            'slur': 52,
            'tie': 53,
            'hairpin-cresc.': 54,
            'hairpin-decr.': 55,
            'trill': 58,
            'tuple': 66,
            'g-clef': 67,
            'f-clef': 68,
            'c-clef': 69,
            'key_signature': 71,
            'time_signature': 72,
            'dynamics_text': 77,
            'other_text': 82,
            'letter_p': 98,
            'letter_f': 88,
            'letter_e': 87,
            'letter_r': 100,
            'letter_s': 101,
            'letter_o': 97,
            'letter_t': 102,
            'letter_c': 85,
            'letter_d': 86,
            'letter_m': 95,
            'numeral_3': 138,
            'numeral_4': 139
        }
            
    def parse_xml_annotations(self, xml_path: Path) -> List[Dict]:
        """
        Parse XML annotation file
        
        Args:
            xml_path: XML file path
            
        Returns:
            List of symbol annotations
        """
        annotations = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for crop_object in root.findall('.//CropObject'):
                try:
                    # Try two different class field names
                    class_elem = crop_object.find('MLClassName')
                    if class_elem is None:
                        class_elem = crop_object.find('ClassName')
                    
                    if class_elem is None:
                        continue
                    
                    annotation = {
                        'id': crop_object.find('Id').text,
                        'class_name': class_elem.text,
                        'top': int(crop_object.find('Top').text),
                        'left': int(crop_object.find('Left').text),
                        'width': int(crop_object.find('Width').text),
                        'height': int(crop_object.find('Height').text)
                    }
                    
                    # Only process classes we defined
                    if annotation['class_name'] in self.class_mapping:
                        annotations.append(annotation)
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
            
        return annotations
    
    def parse_all_xml_annotations(self, xml_path: Path) -> List[Dict]:
        """
        Parse XML annotation file, return all symbol annotations (including unfiltered classes)
        
        Args:
            xml_path: XML file path
            
        Returns:
            List of all symbol annotations
        """
        annotations = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Try different XPath query methods
            crop_objects = root.findall('.//CropObject')
            if not crop_objects:
                crop_objects = root.findall('CropObject')
            if not crop_objects:
                crop_objects_container = root.find('CropObjects')
                if crop_objects_container is not None:
                    crop_objects = crop_objects_container.findall('CropObject')
            
            for crop_object in crop_objects:
                try:
                    # Safely get element text
                    id_elem = crop_object.find('Id')
                    
                    # Try two different class field names
                    class_elem = crop_object.find('MLClassName')
                    if class_elem is None:
                        class_elem = crop_object.find('ClassName')
                    
                    top_elem = crop_object.find('Top')
                    left_elem = crop_object.find('Left')
                    width_elem = crop_object.find('Width')
                    height_elem = crop_object.find('Height')
                    
                    if all(elem is not None for elem in [id_elem, class_elem, top_elem, left_elem, width_elem, height_elem]):
                        annotation = {
                            'id': id_elem.text,
                            'class_name': class_elem.text,
                            'top': int(top_elem.text),
                            'left': int(left_elem.text),
                            'width': int(width_elem.text),
                            'height': int(height_elem.text)
                        }
                        
                        # Include all symbols, no class filtering
                        annotations.append(annotation)
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
        return annotations
    
    def analyze_symbol_relationships(self, xml_path: Path) -> int:
        """
        Analyze relationships between symbols (through Outlinks field)
        
        Args:
            xml_path: XML file path
            
        Returns:
            Number of symbol relationships
        """
        relationships = 0
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            crop_objects = root.findall('.//CropObject')
            if not crop_objects:
                crop_objects = root.findall('CropObject')
            if not crop_objects:
                crop_objects_container = root.find('CropObjects')
                if crop_objects_container is not None:
                    crop_objects = crop_objects_container.findall('CropObject')
            
            for crop_object in crop_objects:
                outlinks_elem = crop_object.find('Outlinks')
                if outlinks_elem is not None and outlinks_elem.text:
                    # Outlinks field contains space-separated ID list
                    outlink_ids = outlinks_elem.text.strip().split()
                    relationships += len(outlink_ids)
                    
        except Exception as e:
            print(f"Error analyzing symbol relationships: {e}")
            
        return relationships
    
    def generate_random_crop(self, img_width: int, img_height: int) -> Tuple[int, int]:
        """
        Generate random crop position
        
        Args:
            img_width: Image width
            img_height: Image height
            
        Returns:
            (x, y) crop starting position
        """
        max_x = max(0, img_width - self.sample_size)
        max_y = max(0, img_height - self.sample_size)
        
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        return x, y
    
    def is_symbol_in_crop(self, annotation: Dict, crop_x: int, crop_y: int) -> bool:
        """
        Check if symbol overlaps with crop region
        
        Args:
            annotation: Symbol annotation
            crop_x: Crop region x coordinate
            crop_y: Crop region y coordinate
            
        Returns:
            Whether it is within the crop region
        """
        symbol_left = annotation['left']
        symbol_top = annotation['top']
        symbol_right = symbol_left + annotation['width']
        symbol_bottom = symbol_top + annotation['height']
        
        crop_right = crop_x + self.sample_size
        crop_bottom = crop_y + self.sample_size
        
        # Consider it within crop region as long as there is any intersection with crop region
        inter_left = max(symbol_left, crop_x)
        inter_top = max(symbol_top, crop_y)
        inter_right = min(symbol_right, crop_right)
        inter_bottom = min(symbol_bottom, crop_bottom)
        
        return inter_right > inter_left and inter_bottom > inter_top
    
    def convert_to_yolo_format(self, annotation: Dict, crop_x: int, crop_y: int) -> Tuple[int, float, float, float, float]:
        """
        Convert annotation to YOLO format
        
        Args:
            annotation: Symbol annotation
            crop_x: Crop region x coordinate
            crop_y: Crop region y coordinate
            
        Returns:
            (class_id, x_center, y_center, width, height) normalized coordinates
        """
        class_id = self.class_mapping[annotation['class_name']]
        
        # Calculate symbol position within crop region
        symbol_left = max(0, annotation['left'] - crop_x)
        symbol_top = max(0, annotation['top'] - crop_y)
        symbol_right = min(self.sample_size, annotation['left'] + annotation['width'] - crop_x)
        symbol_bottom = min(self.sample_size, annotation['top'] + annotation['height'] - crop_y)
        
        # Ensure symbol is within crop region
        if symbol_right <= symbol_left or symbol_bottom <= symbol_top:
            return None
            
        # Calculate center point and dimensions
        x_center = (symbol_left + symbol_right) / 2.0
        y_center = (symbol_top + symbol_bottom) / 2.0
        width = symbol_right - symbol_left
        height = symbol_bottom - symbol_top
        
        # Normalize
        x_center_norm = x_center / self.sample_size
        y_center_norm = y_center / self.sample_size
        width_norm = width / self.sample_size
        height_norm = height / self.sample_size
        
        return class_id, x_center_norm, y_center_norm, width_norm, height_norm

    def validate_binary_image(self, img: np.ndarray, img_path: Path) -> None:
        """
        Validate if image is binary (pixel values are 0 or 1)
        
        Args:
            img: Image array
            img_path: Image path (for error reporting)
        """
        # Check if image is grayscale
        if len(img.shape) == 3:
            # If it's a color image, convert to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        
        # Check if pixel values only contain 0 and 1
        unique_values = np.unique(gray_img)
        
        # Allowed pixel values: 0, 1, 255 (some binarization may use 255 instead of 1)
        allowed_values = {0, 1, 255}
        unexpected_values = set(unique_values) - allowed_values
        
        if unexpected_values:
            print(f"Warning: Image {img_path} is not a standard binary image")
            print(f"  Found non-binary pixel values: {sorted(unexpected_values)}")
            print(f"  All pixel values: {sorted(unique_values)}")
            
            # If non-binary pixels are found, we can choose to force binarization
            # Here we only issue a warning, not force conversion
            print(f"  Suggestion: Ensure source data is binary (pixel values are 0 or 1)")

    def compute_top_classes(self, top_k: int = 73) -> List[str]:
        """
        Pre-scan all annotations, count class frequencies, select Top-K high-frequency classes
        
        Args:
            top_k: Number of classes to select
        Returns:
            List of Top-K class names
        """
        xml_dir = self.source_dir / 'data' / 'cropobjects_manual'
        class_counts: Dict[str, int] = {}
        for xml_file in sorted(xml_dir.glob('CVC-MUSCIMA_W-*_N-*_D-ideal.xml')):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for crop_object in root.findall('.//CropObject'):
                    class_name = crop_object.find('MLClassName').text
                    if class_name:
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
            except Exception as e:
                print(f"Failed to parse when counting class frequencies {xml_file}: {e}")
        # Sort by frequency
        sorted_classes = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)
        top_classes = [name for name, _ in sorted_classes[:top_k]]
        print(f"Selected Top-{top_k} classes, total {len(top_classes)} classes")
        return top_classes
    
    def process_image(self, img_path: Path, xml_path: Path, output_prefix: str) -> List[str]:
        """
        Process single image, generate multiple samples
        
        Args:
            img_path: Image path
            xml_path: XML annotation path
            output_prefix: Output file prefix
            
        Returns:
            List of generated filenames
        """
        # Ensure temporary directories exist
        temp_img_dir = self.output_dir / 'images' / 'temp'
        temp_label_dir = self.output_dir / 'labels' / 'temp'
        temp_img_dir.mkdir(parents=True, exist_ok=True)
        temp_label_dir.mkdir(parents=True, exist_ok=True)
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error loading image: {img_path}")
            return []
            
        # Validate if image is binary (pixel values are 0 or 1)
        self.validate_binary_image(img, img_path)
            
        img_height, img_width = img.shape[:2]
        
        # Parse annotations
        annotations = self.parse_xml_annotations(xml_path)
        
        generated_files = []
        
        # Generate multiple samples
        crop_positions = []  # For visualization: record all crop regions
        for i in range(self.num_samples):
            # Generate random crop position
            crop_x, crop_y = self.generate_random_crop(img_width, img_height)
            crop_positions.append((crop_x, crop_y))
            
            # Crop image
            cropped_img = img[crop_y:crop_y+self.sample_size, crop_x:crop_x+self.sample_size]
            
            # Resize to target size
            resized_img = cv2.resize(cropped_img, (self.target_size, self.target_size))
            
            # Save image
            img_filename = f"{output_prefix}_sample_{i:02d}.jpg"
            img_output_path = self.output_dir / 'images' / 'temp' / img_filename
            cv2.imwrite(str(img_output_path), resized_img)
            
            # Generate corresponding label file
            label_filename = f"{output_prefix}_sample_{i:02d}.txt"
            label_output_path = self.output_dir / 'labels' / 'temp' / label_filename
            
            yolo_annotations = []
            for annotation in annotations:
                if self.is_symbol_in_crop(annotation, crop_x, crop_y):
                    yolo_format = self.convert_to_yolo_format(annotation, crop_x, crop_y)
                    if yolo_format is not None:
                        yolo_annotations.append(yolo_format)
            
            # Save label file
            with open(label_output_path, 'w') as f:
                for class_id, x_center, y_center, width, height in yolo_annotations:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            generated_files.append(img_filename)
            
        # Generate visualization for each image
        try:
            project_root = Path(__file__).resolve().parent
            vis_dir = project_root / 'Output' / 'preprocess'
            
            # Generate visualization filename
            vis_filename = f"{output_prefix}.png"
            vis_path = vis_dir / vis_filename

            vis_img = img.copy()

            # Get all original symbol annotations (including unfiltered classes)
            all_annotations = self.parse_all_xml_annotations(xml_path)
            
            # Count symbol information
            self.stats['processed_images'] += 1
            self.stats['total_symbols'] += len(all_annotations)
            
            # Count symbol relationships
            relationships = self.analyze_symbol_relationships(xml_path)
            self.stats['symbol_relationships'] += relationships
            
            # Count symbol types
            for ann in all_annotations:
                class_name = ann['class_name']
                self.stats['symbol_types'][class_name] = self.stats['symbol_types'].get(class_name, 0) + 1

            print(f"Processing image {output_prefix}: parsed {len(all_annotations)} symbol annotations")

            # Draw all symbol boxes and annotations (blue)
            for ann in all_annotations:
                x1 = ann['left']
                y1 = ann['top']
                x2 = x1 + ann['width']
                y2 = y1 + ann['height']
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Add class name annotation
                class_name = ann['class_name']
                # Calculate text position (above the box)
                text_x = x1
                text_y = max(y1 - 5, 15)  # Ensure text does not exceed image boundaries
                
                # Draw text background
                text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(vis_img, (text_x, text_y - text_size[1] - 5), 
                            (text_x + text_size[0], text_y), (255, 255, 255), -1)
                
                # Draw text
                cv2.putText(vis_img, class_name, (text_x, text_y - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Draw all crop regions (green)
            for (cx, cy) in crop_positions:
                cv2.rectangle(vis_img, (cx, cy), (cx + self.sample_size, cy + self.sample_size), (0, 255, 0), 2)
                # Add crop region number
                cv2.putText(vis_img, f"Crop", (cx + 5, cy + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Save at original resolution
            cv2.imwrite(str(vis_path), vis_img)
            print(f"Visualization image saved to: {vis_path}")
            
        except Exception as e:
            print(f"Visualization save failed: {e}")
            import traceback
            traceback.print_exc()

        return generated_files
    
    def split_dataset(self, all_files: List[str], train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2):
        """
        Split dataset
        
        Args:
            all_files: List of all generated files
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        """
        random.shuffle(all_files)
        
        total_files = len(all_files)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        test_count = int(total_files * test_ratio)
        
        train_files = all_files[:train_count]
        val_files = all_files[train_count:train_count + val_count]
        test_files = all_files[train_count + val_count:train_count + val_count + test_count]
        
        # Move files to corresponding directories
        temp_img_dir = self.output_dir / 'images' / 'temp'
        temp_label_dir = self.output_dir / 'labels' / 'temp'
        
        for filename in train_files:
            # Move image
            src_img = temp_img_dir / filename
            dst_img = self.output_dir / 'images' / 'train' / filename
            shutil.move(str(src_img), str(dst_img))
            
            # Move label
            label_filename = filename.replace('.jpg', '.txt')
            src_label = temp_label_dir / label_filename
            dst_label = self.output_dir / 'labels' / 'train' / label_filename
            shutil.move(str(src_label), str(dst_label))
        
        for filename in val_files:
            # Move image
            src_img = temp_img_dir / filename
            dst_img = self.output_dir / 'images' / 'val' / filename
            shutil.move(str(src_img), str(dst_img))
            
            # Move label
            label_filename = filename.replace('.jpg', '.txt')
            src_label = temp_label_dir / label_filename
            dst_label = self.output_dir / 'labels' / 'val' / label_filename
            shutil.move(str(src_label), str(dst_label))
        
        for filename in test_files:
            # Move image
            src_img = temp_img_dir / filename
            dst_img = self.output_dir / 'images' / 'test' / filename
            shutil.move(str(src_img), str(dst_img))
            
            # Move label
            label_filename = filename.replace('.jpg', '.txt')
            src_label = temp_label_dir / label_filename
            dst_label = self.output_dir / 'labels' / 'test' / label_filename
            shutil.move(str(src_label), str(dst_label))
        
        # Clean up temporary directories
        shutil.rmtree(temp_img_dir)
        shutil.rmtree(temp_label_dir)
        
        print(f"Dataset split completed:")
        print(f"  Training set: {len(train_files)} samples")
        print(f"  Validation set: {len(val_files)} samples")
        print(f"  Test set: {len(test_files)} samples")
    
    def create_data_yaml(self):
        """Create data.yaml configuration file"""
        # Use reconstructed 73-class mapping, get class name list sorted by ID
        id_to_name = {v: k for k, v in self.class_mapping.items()}
        class_names = [id_to_name[i] for i in sorted(id_to_name.keys())]
        
        data_config = {
            'train': str(self.output_dir / 'images' / 'train'),
            'val': str(self.output_dir / 'images' / 'val'),
            'test': str(self.output_dir / 'images' / 'test'),
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
    
    def process_all_data(self):
        """Process all data"""
        print("Starting to process CVC-MUSCIMA dataset...")
        
        # Create temporary directories
        (self.output_dir / 'images' / 'temp').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'temp').mkdir(parents=True, exist_ok=True)
        
        # Pre-scan to select Top-K classes and rebuild class mapping
        print(f"Pre-scanning and selecting Top-{self.top_k_classes} high-frequency classes...")
        self.selected_classes = self.compute_top_classes(self.top_k_classes)
        self.class_mapping = {name: idx for idx, name in enumerate(self.selected_classes)}
        print(f"Rebuilt class mapping to {len(self.class_mapping)} classes")

        all_generated_files = []
        
        # Traverse all XML files, find corresponding images based on XML files
        xml_dir = self.source_dir / 'data' / 'cropobjects_manual'
        
        for xml_file in sorted(xml_dir.glob('CVC-MUSCIMA_W-*_N-*_D-ideal.xml')):
            # Parse XML filename to get writer and page information
            # Format: CVC-MUSCIMA_W-01_N-10_D-ideal.xml
            parts = xml_file.stem.split('_')
            writer_name = parts[1].lower()  # W-01 -> w-01
            page_number = parts[2].split('-')[1]  # N-10 -> 10
            
            print(f"Processing {writer_name} page {page_number}...")
            
            # Find corresponding image file
            symbol_dir = self.source_dir / 'data' / 'images' / writer_name / 'symbol'
            if not symbol_dir.exists():
                print(f"  Symbol directory does not exist: {symbol_dir}")
                continue
            
            # Build image filename: 10 -> p010.png
            img_filename = f"p{page_number.zfill(3)}.png"
            img_path = symbol_dir / img_filename
            
            if not img_path.exists():
                print(f"  Image file does not exist: {img_path}")
                continue
            
            # Generate output file prefix
            output_prefix = f"{writer_name}_{img_filename[:-4]}"  # Remove .png suffix
            
            # Process image
            generated_files = self.process_image(img_path, xml_file, output_prefix)
            all_generated_files.extend(generated_files)
            
            print(f"  Generated {len(generated_files)} samples")
        
        print(f"Total generated {len(all_generated_files)} samples")
        
        # Split dataset
        print("Splitting dataset...")
        self.split_dataset(all_generated_files)
        
        # Create configuration file
        print("Creating data.yaml...")
        self.create_data_yaml()
        
        print("Data preprocessing completed!")
        print(f"Output directory: {self.output_dir}")
        print(f"Number of classes: {len(self.class_mapping)}")
        print(f"Class list: {list(self.class_mapping.keys())}")
        
        # Display statistics
        print("\n=== Data Statistics ===")
        print(f"1. Number of labeled images: {self.stats['processed_images']}")
        print(f"2. Total number of symbols: {self.stats['total_symbols']}")
        print(f"3. Number of symbol relationships: {self.stats['symbol_relationships']}")
        
        print(f"\nSymbol type statistics (Top 20):")
        sorted_types = sorted(self.stats['symbol_types'].items(), key=lambda x: x[1], reverse=True)
        for i, (symbol_type, count) in enumerate(sorted_types[:20]):
            print(f"  {i+1:2d}. {symbol_type:<25} : {count:>6} items")
        
        if len(sorted_types) > 20:
            print(f"  ... and {len(sorted_types) - 20} other symbol types")
        
        print(f"\nVisualization images saved to: {Path(__file__).resolve().parent / '../Output' / 'preprocess'}")

def main():
    parser = argparse.ArgumentParser(description='CVC-MUSCIMA data preprocessing script')
    parser.add_argument('--source', type=str, 
                       default='/users/eleves-a/2023/yuguang.yao/Projet-d-OMR/v1.0',
                       help='Source data directory path')
    parser.add_argument('--output', type=str,
                       default='/users/eleves-a/2023/yuguang.yao/Projet-d-OMR/Yolo-Dataset',
                       help='Output dataset directory path')
    
    args = parser.parse_args()
    
    # Create preprocessor
    preprocessor = CVCDataPreprocessor(args.source, args.output)
    
    # Process data
    preprocessor.process_all_data()


def preprocess_from_config(config):
    """
    Perform data preprocessing from configuration object
    
    Args:
        config: FinetuningConfig configuration object
    """
    import sys
    setup_path = str(Path(__file__).parent.parent / 'setup')
    if setup_path not in sys.path:
        sys.path.append(setup_path)
    from setup_finetuning import FinetuningConfig
    
    if not isinstance(config, FinetuningConfig):
        raise ValueError("config must be a FinetuningConfig instance")
    
    # Get parameters from configuration
    preprocessing_args = config.get_preprocessing_args()
    
    # Create preprocessor
    preprocessor = CVCDataPreprocessor(
        preprocessing_args['source'], 
        preprocessing_args['output']
    )
    
    # Set preprocessing parameters
    preprocessor.top_k_classes = preprocessing_args['top_k_classes']
    preprocessor.sample_size = preprocessing_args['sample_size']
    preprocessor.target_size = preprocessing_args['target_size']
    preprocessor.num_samples = preprocessing_args['num_samples']
    
    # Process data
    preprocessor.process_all_data()
    
    return preprocessor


if __name__ == '__main__':
    main()
