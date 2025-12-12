"""
Unified logging system for OMR project
Supports file and console logging with module-specific loggers
"""
import logging
import sys
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime


class OMRLogger:
    """Unified logger for OMR modules"""
    
    _loggers: Dict[str, logging.Logger] = {}
    _log_dir: Optional[Path] = None
    
    @classmethod
    def setup(cls, log_dir: Path, level: int = logging.INFO) -> None:
        """
        Setup logging system
        
        Args:
            log_dir: Directory for log files
            level: Logging level
        """
        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    @classmethod
    def get_logger(cls, module_name: str) -> logging.Logger:
        """
        Get or create logger for a module
        
        Args:
            module_name: Module name (unet, yolo, mlp, assembler)
        
        Returns:
            Logger instance
        """
        if module_name in cls._loggers:
            return cls._loggers[module_name]
        
        logger = logging.getLogger(f"omr.{module_name}")
        logger.setLevel(logging.INFO)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler (if log_dir is set)
        if cls._log_dir:
            log_file = cls._log_dir / f"{module_name}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        
        cls._loggers[module_name] = logger
        return logger

