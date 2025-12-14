"""
Assembler inference module
Migrated from src/run_assembly.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.config_loader import ConfigLoader
from common.logger import OMRLogger
from common.paths import PathManager

from .run_assembly import main as assemble_main


def infer_assembler(config: ConfigLoader, path_manager: PathManager) -> None:
    """Run assembler inference"""
    logger = OMRLogger.get_logger('assembler')
    
    assembler_config = config.get_module_config('assembler')
    infer_config = assembler_config.get('infer', {})
    
    if not infer_config.get('enabled', True):
        logger.info("Assembler inference is disabled in config")
        return
    
    # Get required paths
    json_path = infer_config.get('json_path')
    mask_path = infer_config.get('mask_path')
    output_path = infer_config.get('output_path')
    
    if not json_path or not mask_path or not output_path:
        logger.error("Assembler requires json_path, mask_path, and output_path in config")
        return
    
    json_path = path_manager.resolve_path(json_path)
    mask_path = path_manager.resolve_path(mask_path)
    output_path = path_manager.resolve_path(output_path)
    
    # Convert config to command-line args
    original_argv = sys.argv
    try:
        exp_name = infer_config.get('exp_name', 'mlp_training')
        default_model = f'model/mlp/{exp_name}/model_best.pth'
        sys.argv = ['run_assembly.py'] + [
            '--json', str(json_path),
            '--mask', str(mask_path),
            '--output', str(output_path),
            '--model', str(path_manager.resolve_path(infer_config.get('model_path', default_model))),
        ]
        
        if infer_config.get('title'):
            sys.argv.extend(['--title', infer_config['title']])
        if infer_config.get('composer'):
            sys.argv.extend(['--composer', infer_config['composer']])
        if infer_config.get('max_parts'):
            sys.argv.extend(['--max-parts', str(infer_config['max_parts'])])
        if infer_config.get('min_symbols'):
            sys.argv.extend(['--min-symbols', str(infer_config['min_symbols'])])
        if infer_config.get('visualize', False):
            sys.argv.append('--visualize')
        if infer_config.get('original_image'):
            sys.argv.extend(['--original-image', str(path_manager.resolve_path(infer_config['original_image']))])
        
        assemble_main()
    finally:
        sys.argv = original_argv
    
    logger.info("Assembler inference completed")

