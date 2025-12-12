"""
Visualization functions for MLP linker training.
"""
import matplotlib.pyplot as plt
import os


def plot_performance(epoch_numbers, train_losses, val_losses, train_accuracies, 
                    val_accuracies, train_f1_scores, val_f1_scores, 
                    train_precisions, val_precisions, train_recalls, val_recalls,
                    best_f1, best_val_f1, output_dir, exp_name):
    """Plot training and validation metrics."""
    
    # Plot training and validation metrics (6 separate graphs)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss plot
    axes[0, 0].plot(epoch_numbers, train_losses, marker='o', linewidth=2, color='red', label='Train', markersize=6)
    axes[0, 0].plot(epoch_numbers, val_losses, marker='s', linewidth=2, color='darkred', label='Validation', markersize=6)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Loss (Train vs Validation)', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10, loc='best')
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[0, 1].plot(epoch_numbers, train_accuracies, marker='o', linewidth=2, color='green', label='Train', markersize=6)
    axes[0, 1].plot(epoch_numbers, val_accuracies, marker='s', linewidth=2, color='darkgreen', label='Validation', markersize=6)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('Accuracy (Train vs Validation)', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10, loc='best')
    axes[0, 1].grid(True, alpha=0.3)

    # F1 Score plot
    axes[0, 2].plot(epoch_numbers, train_f1_scores, marker='o', linewidth=2, color='orange', label='Train', markersize=6)
    axes[0, 2].plot(epoch_numbers, val_f1_scores, marker='s', linewidth=2, color='darkorange', label='Validation', markersize=6)
    axes[0, 2].set_xlabel('Epoch', fontsize=12)
    axes[0, 2].set_ylabel('F1 Score', fontsize=12)
    axes[0, 2].set_title('F1 Score (Train vs Validation)', fontsize=14, fontweight='bold')
    axes[0, 2].legend(fontsize=10, loc='best')
    axes[0, 2].grid(True, alpha=0.3)

    # Precision plot
    axes[1, 0].plot(epoch_numbers, train_precisions, marker='o', linewidth=2, color='blue', label='Train', markersize=6)
    axes[1, 0].plot(epoch_numbers, val_precisions, marker='s', linewidth=2, color='darkblue', label='Validation', markersize=6)
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Precision', fontsize=12)
    axes[1, 0].set_title('Precision (Train vs Validation)', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10, loc='best')
    axes[1, 0].grid(True, alpha=0.3)

    # Recall plot
    axes[1, 1].plot(epoch_numbers, train_recalls, marker='o', linewidth=2, color='purple', label='Train', markersize=6)
    axes[1, 1].plot(epoch_numbers, val_recalls, marker='s', linewidth=2, color='indigo', label='Validation', markersize=6)
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Recall', fontsize=12)
    axes[1, 1].set_title('Recall (Train vs Validation)', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10, loc='best')
    axes[1, 1].grid(True, alpha=0.3)

    # All metrics combined - Validation only (cleaner visualization)
    axes[1, 2].plot(epoch_numbers, val_precisions, marker='o', linewidth=2, color='blue', label='Precision', markersize=6)
    axes[1, 2].plot(epoch_numbers, val_recalls, marker='s', linewidth=2, color='purple', label='Recall', markersize=6)
    axes[1, 2].plot(epoch_numbers, val_f1_scores, marker='^', linewidth=2, color='orange', label='F1 Score', markersize=6)
    axes[1, 2].plot(epoch_numbers, val_accuracies, marker='d', linewidth=2, color='green', label='Accuracy', markersize=6)
    axes[1, 2].set_xlabel('Epoch', fontsize=12)
    axes[1, 2].set_ylabel('Score', fontsize=12)
    axes[1, 2].set_title('All Validation Metrics', fontsize=14, fontweight='bold')
    axes[1, 2].legend(fontsize=10, loc='best')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    plot_path = f'{output_dir}/{exp_name}_training_metrics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Plot saved to {plot_path}')

    # Print summary statistics
    print('\n=== Training Summary ===')
    print(f'Final Training Metrics:')
    print(f'  Loss: {train_losses[-1]:.4f}')
    print(f'  Accuracy: {train_accuracies[-1]:.4f}')
    print(f'  Precision: {train_precisions[-1]:.4f}')
    print(f'  Recall: {train_recalls[-1]:.4f}')
    print(f'  F1 Score: {train_f1_scores[-1]:.4f}')

    print(f'\nFinal Validation Metrics:')
    print(f'  Loss: {val_losses[-1]:.4f}')
    print(f'  Accuracy: {val_accuracies[-1]:.4f}')
    print(f'  Precision: {val_precisions[-1]:.4f}')
    print(f'  Recall: {val_recalls[-1]:.4f}')
    print(f'  F1 Score: {val_f1_scores[-1]:.4f}')

    print(f'\nBest Scores:')
    print(f'  Best Training F1: {best_f1:.4f}')
    print(f'  Best Validation F1: {best_val_f1:.4f}')


