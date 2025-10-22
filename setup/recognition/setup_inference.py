#!/usr/bin/env python3
"""
Inference parameter setup script
Integrate all inference-related parameters, file paths and configurations
"""

import os
from pathlib import Path

class InferenceConfig:
    """Inference configuration class - contains all inference-related parameters"""
    
    def __init__(self):
        # Project root directory
        self.project_root = Path(__file__).parent.parent
        
        # ========== Model related parameters ==========
        self.model_path = str(self.project_root / 'runs' / 'detect' / 'yolov8l_muscima_finetune' / 'weights' / 'best.pt')
        
        # ========== Input output parameters ==========
        self.input_image_path = str(self.project_root / 'v1.0' / 'data' / 'images' / 'w-01' / 'symbol' / 'p001.png')
        self.output_dir = str(self.project_root / 'Output' / 'inference_output')
        
        # ========== Tiled inference parameters ==========
        self.tile_size = 1216  # Tile size
        self.target_size = 640  # Inference target size
        self.overlap = 100     # Overlap pixels between tiles
        
        # ========== Detection parameters ==========
        self.confidence_threshold = 0.25  # Confidence threshold
        self.iou_threshold = 0.45         # IoU threshold (for NMS)
        self.nms_iou_threshold = 0.5      # IoU threshold for NMS
        
        # ========== Visualization parameters ==========
        self.save_detected_image = True      # Save image with detection boxes
        self.save_crop_visualization = True  # Save crop region visualization
        self.save_combined_visualization = True  # Save comprehensive visualization
        self.save_results_json = True        # Save detection results JSON
        self.save_all_results_json = True    # Save all detection results JSON (including those filtered by NMS)
        self.save_report_txt = True          # Save detection report
        
        # ========== Visualization style parameters ==========
        self.core_region_color = (0, 0, 255)      # Core region color (red)
        self.overlap_region_color = (255, 0, 0)   # Overlap region color (blue)
        self.detection_color = (0, 255, 0)         # Detection box color (green)
        self.box_thickness = 2                     # Bounding box thickness
        self.text_font_scale = 0.5                 # Text font size
        self.text_thickness = 1                    # Text thickness
        
        # ========== Output file naming parameters ==========
        self.detected_suffix = '_detected.jpg'
        self.crop_vis_suffix = '_crop_visualization.jpg'
        self.combined_vis_suffix = '_combined_visualization.jpg'
        self.results_suffix = '_results.json'
        self.all_results_suffix = '_all_results.json'
        self.report_suffix = '_report.txt'
        
        # ========== Performance parameters ==========
        self.max_detection_labels = 20  # Maximum number of detection labels to display (avoid overcrowding)
        
        # ========== Device parameters ==========
        self.device = '0' if self._is_cuda_available() else 'cpu'
        
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_inference_args(self):
        """Get inference parameters dictionary"""
        return {
            'model_path': self.model_path,
            'input_image_path': self.input_image_path,
            'tile_size': self.tile_size,
            'target_size': self.target_size,
            'overlap': self.overlap,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'nms_iou_threshold': self.nms_iou_threshold,
            'output_dir': self.output_dir,
            'device': self.device,
        }
    
    def get_visualization_args(self):
        """Get visualization parameters dictionary"""
        return {
            'save_detected_image': self.save_detected_image,
            'save_crop_visualization': self.save_crop_visualization,
            'save_combined_visualization': self.save_combined_visualization,
            'save_results_json': self.save_results_json,
            'save_all_results_json': self.save_all_results_json,
            'save_report_txt': self.save_report_txt,
            'core_region_color': self.core_region_color,
            'overlap_region_color': self.overlap_region_color,
            'detection_color': self.detection_color,
            'box_thickness': self.box_thickness,
            'text_font_scale': self.text_font_scale,
            'text_thickness': self.text_thickness,
            'max_detection_labels': self.max_detection_labels,
        }
    
    def get_output_naming_args(self):
        """Get output file naming parameters dictionary"""
        return {
            'detected_suffix': self.detected_suffix,
            'crop_vis_suffix': self.crop_vis_suffix,
            'combined_vis_suffix': self.combined_vis_suffix,
            'results_suffix': self.results_suffix,
            'all_results_suffix': self.all_results_suffix,
            'report_suffix': self.report_suffix,
        }
    
    def print_config(self):
        """Print current configuration"""
        print("=" * 60)
        print("Inference configuration parameters")
        print("=" * 60)
        
        print(f"\n📁 Path configuration:")
        print(f"  Project root: {self.project_root}")
        print(f"  Model path: {self.model_path}")
        print(f"  Input image: {self.input_image_path}")
        print(f"  Output directory: {self.output_dir}")
        
        print(f"\n🎯 Tiled inference parameters:")
        print(f"  Tile size: {self.tile_size}x{self.tile_size}")
        print(f"  Inference size: {self.target_size}x{self.target_size}")
        print(f"  Overlap pixels: {self.overlap}")
        
        print(f"\n🔍 Detection parameters:")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  IoU threshold: {self.iou_threshold}")
        print(f"  NMS IoU threshold: {self.nms_iou_threshold}")
        
        print(f"\n💻 Device configuration:")
        print(f"  Device: {self.device}")
        
        print(f"\n📊 Visualization parameters:")
        print(f"  Save detection image: {self.save_detected_image}")
        print(f"  Save crop visualization: {self.save_crop_visualization}")
        print(f"  Save comprehensive visualization: {self.save_combined_visualization}")
        print(f"  Save results JSON: {self.save_results_json}")
        print(f"  Save all results JSON: {self.save_all_results_json}")
        print(f"  Save detection report: {self.save_report_txt}")
        
        print(f"\n🎨 Visualization style:")
        print(f"  Core region color: {self.core_region_color} (BGR)")
        print(f"  Overlap region color: {self.overlap_region_color} (BGR)")
        print(f"  Detection box color: {self.detection_color} (BGR)")
        print(f"  Bounding box thickness: {self.box_thickness}")
        print(f"  Text font size: {self.text_font_scale}")
        print(f"  Text thickness: {self.text_thickness}")
        print(f"  Max detection labels: {self.max_detection_labels}")
        
        print(f"\n📝 Output file naming:")
        print(f"  Detection image suffix: {self.detected_suffix}")
        print(f"  Crop visualization suffix: {self.crop_vis_suffix}")
        print(f"  Comprehensive visualization suffix: {self.combined_vis_suffix}")
        print(f"  Results JSON suffix: {self.results_suffix}")
        print(f"  All results JSON suffix: {self.all_results_suffix}")
        print(f"  Report file suffix: {self.report_suffix}")
        
        print("=" * 60)
    
    def validate_paths(self):
        """Validate if paths exist"""
        print("\n🔍 Path validation:")
        
        # Check model file
        if os.path.exists(self.model_path):
            print(f"  ✅ Model file exists: {self.model_path}")
        else:
            print(f"  ❌ Model file does not exist: {self.model_path}")
        
        # Check input image
        if os.path.exists(self.input_image_path):
            print(f"  ✅ Input image exists: {self.input_image_path}")
        else:
            print(f"  ❌ Input image does not exist: {self.input_image_path}")
        
        # Check output directory
        output_path = Path(self.output_dir)
        if output_path.exists():
            print(f"  ✅ Output directory exists: {self.output_dir}")
        else:
            print(f"  ⚠️  Output directory does not exist, will create automatically: {self.output_dir}")
            output_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✅ Output directory created: {self.output_dir}")

# Create global configuration instance
config = InferenceConfig()

if __name__ == '__main__':
    # Print configuration information
    config.print_config()
    
    # Validate paths
    config.validate_paths()
