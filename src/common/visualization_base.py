"""
Base visualization utilities
Common plotting and visualization functions
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime


class BaseVisualizer:
    """Base class for module visualizers"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize visualizer
        
        Args:
            output_dir: Output directory for visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_plot(self, fig, filename: str, dpi: int = 150) -> Path:
        """
        Save matplotlib figure
        
        Args:
            fig: Matplotlib figure
            filename: Output filename
            dpi: Resolution
        
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        return output_path
    
    def plot_metrics(
        self,
        train_metrics: Dict[str, List[float]],
        val_metrics: Optional[Dict[str, List[float]]] = None,
        epochs: Optional[List[int]] = None,
        filename: str = "metrics.png"
    ) -> Path:
        """
        Plot training/validation metrics
        
        Args:
            train_metrics: Dictionary of training metrics {metric_name: [values]}
            val_metrics: Optional dictionary of validation metrics
            epochs: Optional list of epoch numbers
            filename: Output filename
        
        Returns:
            Path to saved plot
        """
        if epochs is None:
            epochs = list(range(1, len(list(train_metrics.values())[0]) + 1))
        
        num_metrics = len(train_metrics)
        if val_metrics:
            num_metrics = max(num_metrics, len(val_metrics))
        
        # Determine grid size
        cols = min(3, num_metrics)
        rows = (num_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        if num_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, list) else [axes]
        else:
            axes = axes.flatten()
        
        metric_idx = 0
        for metric_name, train_values in train_metrics.items():
            ax = axes[metric_idx]
            ax.plot(epochs, train_values, marker='o', label='Train', linewidth=2)
            
            if val_metrics and metric_name in val_metrics:
                ax.plot(epochs, val_metrics[metric_name], marker='s', label='Validation', linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
            ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            metric_idx += 1
        
        # Hide unused subplots
        for idx in range(metric_idx, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return self.save_plot(fig, filename)
    
    def save_json(self, data: Dict, filename: str) -> Path:
        """
        Save data as JSON
        
        Args:
            data: Data dictionary
            filename: Output filename
        
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_path

