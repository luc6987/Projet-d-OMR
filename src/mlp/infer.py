"""MLP inference module"""
from common.config_loader import ConfigLoader
from common.logger import OMRLogger
from common.paths import PathManager


def infer_mlp(config: ConfigLoader, path_manager: PathManager) -> None:
    """Run MLP inference"""
    logger = OMRLogger.get_logger('mlp')
    
    mlp_config = config.get_module_config('mlp')
    infer_config = mlp_config.get('infer', {})
    
    if not infer_config.get('enabled', True):
        logger.info("MLP inference is disabled in config")
        return
    
    # MLP inference is typically used within the assembler module
    logger.info("MLP inference is typically used within assembler module")
    logger.info("Use assembler.infer for end-to-end inference")

