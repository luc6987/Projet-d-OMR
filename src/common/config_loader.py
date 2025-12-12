"""
Configuration loader for setup.yml
Handles YAML parsing and variable substitution
"""
import yaml
import re
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader:
    """Load and parse setup.yml configuration file"""
    
    def __init__(self, config_path: Path):
        """
        Initialize config loader
        
        Args:
            config_path: Path to setup.yml file
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self._raw_config: Dict[str, Any] = {}
        self._config: Dict[str, Any] = {}
        self._load()
    
    def _load(self) -> None:
        """Load and parse YAML file"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f) or {}
        
        # Resolve variable substitutions
        self._config = self._resolve_variables(self._raw_config)
    
    def _resolve_variables(self, config: Dict[str, Any], parent_path: str = "") -> Dict[str, Any]:
        """
        Resolve variable substitutions like ${global.dataset_root}
        
        Args:
            config: Configuration dictionary
            parent_path: Current path in config hierarchy (for nested access)
        
        Returns:
            Configuration with resolved variables
        """
        resolved = {}
        
        for key, value in config.items():
            current_path = f"{parent_path}.{key}" if parent_path else key
            
            if isinstance(value, dict):
                resolved[key] = self._resolve_variables(value, current_path)
            elif isinstance(value, list):
                resolved[key] = [
                    self._resolve_variable_in_string(item, current_path) if isinstance(item, str) else item
                    for item in value
                ]
            elif isinstance(value, str):
                resolved[key] = self._resolve_variable_in_string(value, current_path)
            else:
                resolved[key] = value
        
        return resolved
    
    def _resolve_variable_in_string(self, value: str, context_path: str) -> str:
        """
        Resolve variable substitutions in a string
        
        Args:
            value: String that may contain ${variable} patterns
            context_path: Current context path for relative references
        
        Returns:
            String with variables resolved
        """
        # Pattern: ${path.to.value}
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_path = match.group(1)
            try:
                # Try to resolve from config
                resolved_value = self._get_nested_value(self._raw_config, var_path)
                if resolved_value is None:
                    return match.group(0)  # Return original if not found
                return str(resolved_value)
            except (KeyError, TypeError):
                return match.group(0)  # Return original if resolution fails
        
        return re.sub(pattern, replace_var, value)
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """
        Get nested value from config using dot notation
        
        Args:
            config: Configuration dictionary
            path: Dot-separated path (e.g., "global.dataset_root")
        
        Returns:
            Value at path, or None if not found
        """
        keys = path.split('.')
        value = config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value by path
        
        Args:
            path: Dot-separated path (e.g., "unet.train.batch_size")
            default: Default value if path not found
        
        Returns:
            Configuration value
        """
        keys = path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific module
        
        Args:
            module_name: Module name (unet, yolo, mlp, assembler)
        
        Returns:
            Module configuration dictionary
        """
        return self._config.get(module_name, {})
    
    def is_module_enabled(self, module_name: str) -> bool:
        """
        Check if a module is enabled
        
        Args:
            module_name: Module name (unet, yolo, mlp, assembler)
        
        Returns:
            True if module is enabled
        """
        return self._config.get('module_enable', {}).get(module_name, False)
    
    @property
    def global_config(self) -> Dict[str, Any]:
        """Get global configuration"""
        return self._config.get('global', {})
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get full configuration"""
        return self._config

