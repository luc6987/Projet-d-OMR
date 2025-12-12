"""
Path management utilities
Handles path resolution and directory creation
"""
from pathlib import Path
from typing import Optional


class PathManager:
    """Manage paths for OMR project"""
    
    def __init__(self, project_root: Path):
        """
        Initialize path manager
        
        Args:
            project_root: Project root directory
        """
        self.project_root = Path(project_root).resolve()
    
    def resolve_path(self, path_str: str, relative_to: Optional[Path] = None) -> Path:
        """
        Resolve a path string to absolute Path
        
        Args:
            path_str: Path string (can be relative or absolute)
            relative_to: Base path for relative resolution (default: project_root)
        
        Returns:
            Resolved absolute Path
        """
        if relative_to is None:
            relative_to = self.project_root
        
        path = Path(path_str)
        
        # If absolute, return as-is
        if path.is_absolute():
            return path.resolve()
        
        # Otherwise resolve relative to base
        return (relative_to / path).resolve()
    
    def ensure_dir(self, path: Path) -> Path:
        """
        Ensure directory exists, create if needed
        
        Args:
            path: Directory path
        
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_model_dir(self, module_name: str, subdir: Optional[str] = None) -> Path:
        """
        Get model directory for a module
        
        Args:
            module_name: Module name (unet, yolo, mlp, assembler)
            subdir: Optional subdirectory
        
        Returns:
            Model directory path
        """
        base = self.project_root / "model"
        if subdir:
            return self.ensure_dir(base / module_name / subdir)
        return self.ensure_dir(base / module_name)
    
    def get_vis_stat_dir(self, module_name: str) -> Path:
        """
        Get visualization/statistics directory for a module
        
        Args:
            module_name: Module name
        
        Returns:
            Visualization directory path
        """
        return self.ensure_dir(self.project_root / "vis_stat" / module_name)
    
    def get_output_dir(self, subdir: Optional[str] = None) -> Path:
        """
        Get output directory
        
        Args:
            subdir: Optional subdirectory
        
        Returns:
            Output directory path
        """
        base = self.project_root / "Output"
        if subdir:
            return self.ensure_dir(base / subdir)
        return self.ensure_dir(base)
    
    def get_log_dir(self) -> Path:
        """Get log directory"""
        return self.ensure_dir(self.project_root / "logs")
    
    def get_dataset_root(self) -> Path:
        """Get dataset root directory"""
        return self.project_root / "data"

