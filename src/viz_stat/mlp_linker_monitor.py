"""
Monitoring and visualization script for MLP linker training.
Can be used to monitor training in real-time or analyze saved training logs.
"""
import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_training_log(log_file):
    """Load training metrics from JSON log file."""
    with open(log_file, 'r') as f:
        return json.load(f)


def plot_metrics_from_log(log_file, output_dir=None):
    """Plot metrics from a training log file."""
    data = load_training_log(log_file)
    
    epochs = data.get('epochs', [])
    train_metrics = data.get('train', {})
    val_metrics = data.get('val', {})
    
    if not epochs:
        print("No epoch data found in log file")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Loss
    axes[0, 0].plot(epochs, train_metrics.get('loss', []), 'r-o', label='Train')
    axes[0, 0].plot(epochs, val_metrics.get('loss', []), 'darkred-s', label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, train_metrics.get('accuracy', []), 'g-o', label='Train')
    axes[0, 1].plot(epochs, val_metrics.get('accuracy', []), 'darkgreen-s', label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1
    axes[0, 2].plot(epochs, train_metrics.get('f1', []), 'orange-o', label='Train')
    axes[0, 2].plot(epochs, val_metrics.get('f1', []), 'darkorange-s', label='Val')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('F1 Score')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(epochs, train_metrics.get('precision', []), 'b-o', label='Train')
    axes[1, 0].plot(epochs, val_metrics.get('precision', []), 'darkblue-s', label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(epochs, train_metrics.get('recall', []), 'purple-o', label='Train')
    axes[1, 1].plot(epochs, val_metrics.get('recall', []), 'indigo-s', label='Val')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Combined validation metrics
    axes[1, 2].plot(epochs, val_metrics.get('precision', []), 'b-o', label='Precision')
    axes[1, 2].plot(epochs, val_metrics.get('recall', []), 'purple-s', label='Recall')
    axes[1, 2].plot(epochs, val_metrics.get('f1', []), 'orange-^', label='F1')
    axes[1, 2].plot(epochs, val_metrics.get('accuracy', []), 'g-d', label='Accuracy')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('All Validation Metrics')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir is None:
        output_dir = 'vis_stat/mlp'
    os.makedirs(output_dir, exist_ok=True)
    
    exp_name = Path(log_file).stem
    output_path = f'{output_dir}/{exp_name}_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {output_path}")


def monitor_checkpoints(checkpoint_dir='model/mlp', exp_name=None):
    """Monitor checkpoint files and display statistics."""
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return
    
    checkpoints = list(checkpoint_dir.glob('*.pth'))
    
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    for ckpt in sorted(checkpoints):
        print(f"  - {ckpt.name}")
        # Could load and display model info here if needed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor MLP linker training")
    parser.add_argument('--log_file', type=str, help='Training log JSON file to analyze')
    parser.add_argument('--checkpoint_dir', type=str, default='model/mlp',
                       help='Directory containing checkpoints')
    parser.add_argument('--exp_name', type=str, help='Experiment name filter')
    parser.add_argument('--output_dir', type=str, default='vis_stat/mlp',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    if args.log_file:
        plot_metrics_from_log(args.log_file, args.output_dir)
    else:
        monitor_checkpoints(args.checkpoint_dir, args.exp_name)

