#!/usr/bin/env python3
"""
Fine-tuning parameter setup script
Integrate all training-related parameters, file paths and configurations
"""

import os
from pathlib import Path

class FinetuningConfig:
    """Fine-tuning configuration class - contains all training-related parameters"""
    
    def __init__(self):
        # Project root directory
        self.project_root = Path(__file__).parent.parent
        
        # ========== Data related parameters ==========
        self.data_yaml_path = str(self.project_root / 'Yolo-Dataset' / 'data.yaml')
        self.source_data_dir = str(self.project_root / 'v1.0')
        self.output_dataset_dir = str(self.project_root / 'Yolo-Dataset')
        
        # ========== Model related parameters ==========
        self.pretrained_weights = str(self.project_root / 'yolov8l.pt')
        self.resume_weights = str(self.project_root / 'runs' / 'detect' / 'yolov8l_muscima_finetune' / 'weights' / 'last.pt')
        
        # ========== Training hyperparameters ==========
        self.epochs = 500
        self.batch_size = 8
        self.image_size = 640
        self.learning_rate = 5.5e-5
        self.optimizer = 'AdamW'  # Options: 'SGD', 'Adam', 'AdamW'
        self.momentum = 0.9
        self.patience = 100
        
        # ========== Device related parameters ==========
        self.device = '0' if self._is_cuda_available() else 'cpu'
        self.workers = min(os.cpu_count() or 8, 8)
        
        # ========== Project output parameters ==========
        self.project_dir = str(self.project_root / 'runs' / 'detect')
        self.run_name = 'yolov8l_muscima_finetune'
        self.exist_ok = True
        
        # ========== Training control parameters ==========
        self.seed = 0
        self.cache = 'ram'  # Options: 'ram', 'disk', 'False', 'false'
        self.freeze_layers = 0
        self.resume_training = ''  # Empty string means no resume, 'True' means auto resume, or specify path
        self.close_mosaic = False
        self.cosine_lr = False
        self.save_period = -1  # -1 means disable periodic saving
        self.amp = False  # Mixed precision training
        
        # ========== Validation and test parameters ==========
        self.run_validation = False  # Whether to run validation after training
        self.run_test = False  # Whether to run test set evaluation after training
        
        # ========== Export parameters ==========
        self.export_format = ''  # Options: 'onnx', 'torchscript', 'openvino', etc.
        
        # ========== Data preprocessing parameters ==========
        self.top_k_classes = 73  # Select Top-K high frequency classes
        self.sample_size = 1216  # Sample size
        self.target_size = 640   # Target size
        self.num_samples = 14    # Number of samples per image
        self.train_ratio = 0.6  # Training set ratio
        self.val_ratio = 0.2     # Validation set ratio
        self.test_ratio = 0.2    # Test set ratio
        
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def get_training_args(self):
        """Get training parameters dictionary"""
        return {
            'data': self.data_yaml_path,
            'weights': self.pretrained_weights,
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.image_size,
            'lr0': self.learning_rate,
            'optimizer': self.optimizer,
            'momentum': self.momentum,
            'patience': self.patience,
            'device': self.device,
            'workers': self.workers,
            'project': self.project_dir,
            'name': self.run_name,
            'exist_ok': self.exist_ok,
            'seed': self.seed,
            'cache': self.cache,
            'freeze': self.freeze_layers,
            'resume': self.resume_training,
            'close_mosaic': self.close_mosaic,
            'cos_lr': self.cosine_lr,
            'save_period': self.save_period,
            'amp': self.amp,
            'plots': True,
        }
    
    def get_preprocessing_args(self):
        """Get data preprocessing parameters dictionary"""
        return {
            'source': self.source_data_dir,
            'output': self.output_dataset_dir,
            'top_k_classes': self.top_k_classes,
            'sample_size': self.sample_size,
            'target_size': self.target_size,
            'num_samples': self.num_samples,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
        }
    
    def get_resume_args(self):
        """Get resume training parameters dictionary"""
        return {
            'data': self.data_yaml_path,
            'weights': self.resume_weights,
            'epochs': self.epochs,
            'batch': self.batch_size,
            'imgsz': self.image_size,
            'lr0': self.learning_rate,
            'optimizer': self.optimizer,
            'patience': self.patience,
            'project': self.project_dir,
            'name': 'yolov8l_muscima_resume',
            'resume': True,
        }
    
    def print_config(self):
        """Print current configuration"""
        print("=" * 60)
        print("Fine-tuning configuration parameters")
        print("=" * 60)
        
        print(f"\n📁 Path configuration:")
        print(f"  Project root: {self.project_root}")
        print(f"  Data configuration file: {self.data_yaml_path}")
        print(f"  Source data directory: {self.source_data_dir}")
        print(f"  Output dataset: {self.output_dataset_dir}")
        print(f"  Pretrained weights: {self.pretrained_weights}")
        print(f"  Resume weights: {self.resume_weights}")
        
        print(f"\n🎯 Training parameters:")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Image size: {self.image_size}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Optimizer: {self.optimizer}")
        print(f"  Momentum: {self.momentum}")
        print(f"  Early stopping patience: {self.patience}")
        
        print(f"\n💻 Device configuration:")
        print(f"  Device: {self.device}")
        print(f"  Workers: {self.workers}")
        
        print(f"\n📊 Project configuration:")
        print(f"  Project directory: {self.project_dir}")
        print(f"  Run name: {self.run_name}")
        print(f"  Allow overwrite: {self.exist_ok}")
        
        print(f"\n🔧 Training control:")
        print(f"  Random seed: {self.seed}")
        print(f"  Cache method: {self.cache}")
        print(f"  Freeze layers: {self.freeze_layers}")
        print(f"  Resume training: {self.resume_training}")
        print(f"  Close mosaic: {self.close_mosaic}")
        print(f"  Cosine learning rate: {self.cosine_lr}")
        print(f"  Save period: {self.save_period}")
        print(f"  Mixed precision: {self.amp}")
        
        print(f"\n✅ Validation and testing:")
        print(f"  Run validation: {self.run_validation}")
        print(f"  Run test: {self.run_test}")
        
        print(f"\n📤 Export configuration:")
        print(f"  Export format: {self.export_format}")
        
        print(f"\n🔄 Data preprocessing:")
        print(f"  Top-K classes: {self.top_k_classes}")
        print(f"  Sample size: {self.sample_size}")
        print(f"  Target size: {self.target_size}")
        print(f"  Samples per image: {self.num_samples}")
        print(f"  Training set ratio: {self.train_ratio}")
        print(f"  Validation set ratio: {self.val_ratio}")
        print(f"  Test set ratio: {self.test_ratio}")
        
        print("=" * 60)

# Create global configuration instance
config = FinetuningConfig()

if __name__ == '__main__':
    # Print configuration information
    config.print_config()
