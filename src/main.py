#!/usr/bin/env python3
"""
OMR Project Main Entry Point
Unified CLI for training, inference, and visualization across all modules
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from common.config_loader import ConfigLoader
from common.logger import OMRLogger
from common.paths import PathManager


def setup_environment(config_path: Path) -> tuple:
    """
    Setup environment: load config, initialize logger, create path manager
    
    Returns:
        (config, path_manager) tuple
    """
    config = ConfigLoader(config_path)
    path_manager = PathManager(Path(__file__).parent.parent)
    
    # Setup logging
    log_dir = path_manager.resolve_path(config.global_config.get('log_dir', 'logs'))
    OMRLogger.setup(log_dir)
    
    return config, path_manager


def train_module(module_name: str, config: ConfigLoader, path_manager: PathManager) -> None:
    """Train a specific module"""
    logger = OMRLogger.get_logger(module_name)
    logger.info(f"Starting {module_name.upper()} training...")
    
    if module_name == 'unet':
        from unet.train import train_unet
        train_unet(config, path_manager)
    elif module_name == 'yolo':
        from yolo.train import train_yolo
        train_yolo(config, path_manager)
    elif module_name == 'mlp':
        from mlp.train import train_mlp
        train_mlp(config, path_manager)
    else:
        logger.error(f"Unknown module: {module_name}")
        sys.exit(1)


def infer_module(module_name: str, config: ConfigLoader, path_manager: PathManager) -> None:
    """Run inference for a specific module"""
    logger = OMRLogger.get_logger(module_name)
    logger.info(f"Starting {module_name.upper()} inference...")
    
    if module_name == 'unet':
        from unet.infer import infer_unet
        infer_unet(config, path_manager)
    elif module_name == 'yolo':
        from yolo.infer import infer_yolo
        infer_yolo(config, path_manager)
    elif module_name == 'mlp':
        from mlp.infer import infer_mlp
        infer_mlp(config, path_manager)
    elif module_name == 'assembler':
        from assembler.infer import infer_assembler
        infer_assembler(config, path_manager)
    else:
        logger.error(f"Unknown module: {module_name}")
        sys.exit(1)


def visualize_module(module_name: str, config: ConfigLoader, path_manager: PathManager) -> None:
    """Generate visualizations for a specific module"""
    logger = OMRLogger.get_logger(module_name)
    logger.info(f"Generating {module_name.upper()} visualizations...")
    
    if module_name == 'unet':
        from unet.visualize import visualize_unet
        visualize_unet(config, path_manager)
    elif module_name == 'yolo':
        from yolo.visualize import visualize_yolo
        visualize_yolo(config, path_manager)
    elif module_name == 'mlp':
        from mlp.visualize import visualize_mlp
        visualize_mlp(config, path_manager)
    elif module_name == 'assembler':
        from assembler.visualize import visualize_assembler
        visualize_assembler(config, path_manager)
    else:
        logger.error(f"Unknown module: {module_name}")
        sys.exit(1)


def stats_module(module_name: str, config: ConfigLoader, path_manager: PathManager) -> None:
    """Generate statistics for a specific module"""
    logger = OMRLogger.get_logger(module_name)
    logger.info(f"Generating {module_name.upper()} statistics...")
    
    if module_name == 'unet':
        from unet.stats import stats_unet
        stats_unet(config, path_manager)
    elif module_name == 'yolo':
        from yolo.stats import stats_yolo
        stats_yolo(config, path_manager)
    elif module_name == 'mlp':
        from mlp.stats import stats_mlp
        stats_mlp(config, path_manager)
    elif module_name == 'assembler':
        from assembler.stats import stats_assembler
        stats_assembler(config, path_manager)
    else:
        logger.error(f"Unknown module: {module_name}")
        sys.exit(1)


def run_pipeline(config: ConfigLoader, path_manager: PathManager) -> None:
    """Run full pipeline: unet -> yolo -> mlp -> assembler"""
    logger = OMRLogger.get_logger('main')
    logger.info("Running full OMR pipeline...")
    
    modules = ['unet', 'yolo', 'mlp', 'assembler']
    
    for module_name in modules:
        if not config.is_module_enabled(module_name):
            logger.info(f"Skipping {module_name} (disabled in config)")
            continue
        
        logger.info(f"Processing {module_name}...")
        infer_module(module_name, config, path_manager)


def main():
    parser = argparse.ArgumentParser(
        description="OMR Project Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train U-Net
  python src/main.py train unet
  
  # Run inference for YOLO
  python src/main.py infer yolo
  
  # Generate visualizations for MLP
  python src/main.py visualize mlp
  
  # Run full pipeline
  python src/main.py pipeline
  
  # Use custom config file
  python src/main.py --config custom_setup.yml train unet
        """
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent.parent / 'setup.yml',
        help='Path to setup.yml configuration file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a module')
    train_parser.add_argument('module', choices=['unet', 'yolo', 'mlp'], help='Module to train')
    
    # Infer command
    infer_parser = subparsers.add_parser('infer', help='Run inference for a module')
    infer_parser.add_argument('module', choices=['unet', 'yolo', 'mlp', 'assembler'], help='Module for inference')
    
    # Visualize command
    vis_parser = subparsers.add_parser('visualize', help='Generate visualizations')
    vis_parser.add_argument('module', choices=['unet', 'yolo', 'mlp', 'assembler'], help='Module to visualize')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Generate statistics')
    stats_parser.add_argument('module', choices=['unet', 'yolo', 'mlp', 'assembler'], help='Module for statistics')
    
    # Pipeline command
    subparsers.add_parser('pipeline', help='Run full pipeline (unet -> yolo -> mlp -> assembler)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Setup environment
    config, path_manager = setup_environment(args.config)
    
    # Execute command
    if args.command == 'train':
        train_module(args.module, config, path_manager)
    elif args.command == 'infer':
        infer_module(args.module, config, path_manager)
    elif args.command == 'visualize':
        visualize_module(args.module, config, path_manager)
    elif args.command == 'stats':
        stats_module(args.module, config, path_manager)
    elif args.command == 'pipeline':
        run_pipeline(config, path_manager)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()

