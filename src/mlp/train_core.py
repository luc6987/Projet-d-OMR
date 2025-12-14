"""
Training script for MLP linker model.
"""
import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import numpy as np
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from common.data_pool_gt import load_ground_truth_data
from common.constants import get_classlist_and_classdict
from common.utility import set_seed
from .assemblage.configs.assembler.default import get_cfg_defaults
from .model import MLP
from typing import List, Dict
from .metrics import validate_model, evaluate_test_model
from .visualization import plot_performance


def load_data(gt_annotations_root: str, images_root: str, split_file: str, 
              class_list: List, class_dict: Dict, data_config: str):
    """Load training and validation data."""
    return load_ground_truth_data(
        gt_annotations_root=gt_annotations_root,
        images_root=images_root,
        split_file=split_file,
        class_list=class_list,
        class_dict=class_dict,
        config=data_config,
        load_training_data=True,
        load_validation_data=True,
        load_test_data=False,
    )


def load_test_data(gt_annotations_root: str, images_root: str, split_file: str, 
                   class_list: List, class_dict: Dict, data_config: str):
    """Load test data."""
    return load_ground_truth_data(
        gt_annotations_root=gt_annotations_root,
        images_root=images_root,
        split_file=split_file,
        class_list=class_list,
        class_dict=class_dict,
        config=data_config,
        load_training_data=False,
        load_validation_data=False,
        load_test_data=True,
    )


def load_model(config, device: str):
    """Load MLP model."""
    model = MLP(config)
    model = model.to(device)
    return model


def train_mlp(model, cfg, device, train_data, valid_data, output_dir, exp_name, threshold: float = 0.5, model_save_dir=None, viz_save_dir=None):
    """Train MLP linker model."""
    # Ensure os is available (already imported at module level)
    import os
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.TRAIN.POS_WEIGHT).to(device))

    print('Model built!')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Print GPU info if available
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    
    # Create data loaders with optimized settings for GPU
    num_workers = getattr(cfg.TRAIN, 'NUM_WORKERS', 8) if hasattr(cfg.TRAIN, 'NUM_WORKERS') else 8
    pin_memory = torch.cuda.is_available()  # Pin memory for faster GPU transfer
    
    valid_loader = DataLoader(
        valid_data, 
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,  
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    train_loader = DataLoader(
        train_data, 
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    
    print(f'\nDataLoader settings:')
    print(f'  Batch size: {cfg.TRAIN.BATCH_SIZE}')
    print(f'  Num workers: {num_workers}')
    print(f'  Pin memory: {pin_memory}')
    print(f'  Training batches: {len(train_loader):,}')
    print(f'  Validation batches: {len(valid_loader):,}')
    
    # Training loop with validation
    best_f1 = 0.0
    best_val_f1 = 0.0

    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    train_f1_scores = []
    train_precisions = []
    train_recalls = []

    val_losses = []
    val_accuracies = []
    val_f1_scores = []
    val_precisions = []
    val_recalls = []

    epoch_numbers = []

    # Resolve model save directory
    if model_save_dir is None:
        base_model_dir = output_dir if output_dir is not None else 'model/mlp'
        model_save_dir = f'{base_model_dir}/{exp_name}' if exp_name else base_model_dir
    os.makedirs(model_save_dir, exist_ok=True)
    
    # Resolve visualization save directory
    if viz_save_dir is None:
        base_viz_dir = output_dir if output_dir is not None else 'vis_stat/mlp'
        viz_save_dir = f'{base_viz_dir}/{exp_name}' if exp_name else base_viz_dir
    os.makedirs(viz_save_dir, exist_ok=True)

    # Validation frequency: validate every N epochs (default: 5)
    # Set to 1 for validation every epoch, or higher to reduce validation overhead
    validation_frequency = 5
    print(f'\nValidation will run every {validation_frequency} epochs to save time.')

    for epoch in range(cfg.TRAIN.NUM_EPOCHS):
        model.train()
        
        # Metrics tracking for epoch
        all_outputs = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        # Metrics tracking per evaluation step
        step_outputs = []
        step_labels = []
        
        print(f'\n=== Epoch {epoch+1}/{cfg.TRAIN.NUM_EPOCHS} ===')
        
        eval_interval = max(1, len(train_loader) // 10)  # Evaluate 10 times per epoch
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch['label'])
            total_loss += loss.item()
            num_batches += 1
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Collect outputs and labels for step evaluation
            step_outputs.extend(torch.sigmoid(output).detach().cpu().numpy().flatten())
            step_labels.extend(batch['label'].detach().cpu().numpy().flatten())
            
            # Collect all outputs and labels for epoch evaluation
            all_outputs.extend(torch.sigmoid(output).detach().cpu().numpy().flatten())
            all_labels.extend(batch['label'].detach().cpu().numpy().flatten())
            
            # Evaluate at intervals
            if (batch_idx + 1) % eval_interval == 0 or (batch_idx + 1) == len(train_loader):
                # Calculate step metrics
                step_outputs_np = np.array(step_outputs)
                step_labels_np = np.array(step_labels)
                step_pred = (step_outputs_np > threshold).astype(int)
                
                step_accuracy = (step_pred == step_labels_np).mean()
                step_tp = ((step_pred == 1) & (step_labels_np == 1)).sum()
                step_fp = ((step_pred == 1) & (step_labels_np == 0)).sum()
                step_fn = ((step_pred == 0) & (step_labels_np == 1)).sum()
                
                step_precision = step_tp / (step_tp + step_fp) if (step_tp + step_fp) > 0 else 0.0
                step_recall = step_tp / (step_tp + step_fn) if (step_tp + step_fn) > 0 else 0.0
                step_f1 = 2 * step_precision * step_recall / (step_precision + step_recall) if (step_precision + step_recall) > 0 else 0.0
                step_loss = total_loss / num_batches
                
                # Print progress
                print(f'  Step {batch_idx+1}/{len(train_loader)}: Loss={step_loss:.4f}, Acc={step_accuracy:.4f}, P={step_precision:.4f}, R={step_recall:.4f}, F1={step_f1:.4f}')
                
                # Reset step metrics
                step_outputs = []
                step_labels = []
        
        # Calculate epoch training metrics
        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)
        pred = (all_outputs > threshold).astype(int)
        
        accuracy = (pred == all_labels).mean()
        tp = ((pred == 1) & (all_labels == 1)).sum()
        fp = ((pred == 1) & (all_labels == 0)).sum()
        fn = ((pred == 0) & (all_labels == 1)).sum()
        tn = ((pred == 0) & (all_labels == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        avg_loss = total_loss / num_batches
        
        # Store epoch training metrics
        epoch_numbers.append(epoch + 1)
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        train_precisions.append(precision)
        train_recalls.append(recall)
        train_f1_scores.append(f1)
        
        print(f'\nEpoch {epoch+1} Training Metrics:')
        print(f'  Loss:      {avg_loss:.4f}')
        print(f'  Accuracy:  {accuracy:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall:    {recall:.4f}')
        print(f'  F1 Score:  {f1:.4f}')
        print(f'  Threshold: {threshold:.3f}')
        print(f'  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')
        
        # Track best training F1
        if f1 > best_f1:
            best_f1 = f1
            print(f'  *** New best training F1 score: {best_f1:.4f} ***')
        
        # Run validation only at specified intervals to save time
        # Validate every N epochs, or on the last epoch, or on the first epoch
        should_validate = (
            (epoch + 1) % validation_frequency == 0 or 
            (epoch + 1) == cfg.TRAIN.NUM_EPOCHS or 
            epoch == 0
        )
        
        if should_validate:
            # Run validation with optimal threshold finding
            val_metrics = validate_model(model, valid_loader, criterion, device, threshold, find_optimal_thresh=True)
            
            # Store validation metrics
            val_losses.append(val_metrics['loss'])
            val_accuracies.append(val_metrics['accuracy'])
            val_precisions.append(val_metrics['precision'])
            val_recalls.append(val_metrics['recall'])
            val_f1_scores.append(val_metrics['f1'])
            
            print(f'\nEpoch {epoch+1} Validation Metrics:')
            print(f'  Loss:      {val_metrics["loss"]:.4f}')
            print(f'  Accuracy:  {val_metrics["accuracy"]:.4f}')
            print(f'  Precision: {val_metrics["precision"]:.4f}')
            print(f'  Recall:    {val_metrics["recall"]:.4f}')
            print(f'  F1 Score:  {val_metrics["f1"]:.4f}')
            if 'threshold' in val_metrics:
                print(f'  Threshold: {val_metrics["threshold"]:.3f} (optimized)')
            print(f'  TP: {val_metrics["tp"]}, FP: {val_metrics["fp"]}, FN: {val_metrics["fn"]}, TN: {val_metrics["tn"]}')
            
            # Track best validation F1
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                print(f'  *** New best validation F1 score: {best_val_f1:.4f} ***')
                # Save best model
                if model_save_dir is None:
                    import os
                    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)
                    model_save_dir = f'{output_dir}/{exp_name}'
                import os
                os.makedirs(model_save_dir, exist_ok=True)
                best_model_path = f'{model_save_dir}/model_best.pth'
                checkpoint = {
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'val_f1': best_val_f1,
                    'config': cfg
                }
                torch.save(checkpoint, best_model_path)
                print(f'  Best model saved to {best_model_path}')
        else:
            # For epochs without validation, use the last validation metrics (for plotting continuity)
            # This ensures the plot has consistent data points
            if len(val_losses) > 0:
                val_losses.append(val_losses[-1])
                val_accuracies.append(val_accuracies[-1])
                val_precisions.append(val_precisions[-1])
                val_recalls.append(val_recalls[-1])
                val_f1_scores.append(val_f1_scores[-1])
            else:
                # If no validation has been run yet, use zeros
                val_losses.append(0.0)
                val_accuracies.append(0.0)
                val_precisions.append(0.0)
                val_recalls.append(0.0)
                val_f1_scores.append(0.0)
        
        # Save checkpoint periodically
        if (epoch + 1) % cfg.TRAIN.SAVE_FREQUENCY == 0:
            if model_save_dir is None:
                import os
                os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)
                model_save_dir = f'{output_dir}/{exp_name}'
            import os
            os.makedirs(model_save_dir, exist_ok=True)
            checkpoint_path = f'{model_save_dir}/model_ep{epoch+1}.pth'
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': cfg
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'  Checkpoint saved to {checkpoint_path}')

    print('\n=== Training Complete ===')
    print(f'Best Training F1 Score: {best_f1:.4f}')
    print(f'Best Validation F1 Score: {best_val_f1:.4f}')
    
    # Final validation to ensure we have the latest model performance
    print('\n=== Running Final Validation ===')
    model.eval()
    final_val_metrics = validate_model(model, valid_loader, criterion, device, threshold, find_optimal_thresh=True)
    print(f'Final Validation Metrics:')
    print(f'  Loss:      {final_val_metrics["loss"]:.4f}')
    print(f'  Accuracy:  {final_val_metrics["accuracy"]:.4f}')
    print(f'  Precision: {final_val_metrics["precision"]:.4f}')
    print(f'  Recall:    {final_val_metrics["recall"]:.4f}')
    print(f'  F1 Score:  {final_val_metrics["f1"]:.4f}')
    if 'threshold' in final_val_metrics:
        print(f'  Threshold: {final_val_metrics["threshold"]:.3f} (optimized)')
    print(f'  TP: {final_val_metrics["tp"]}, FP: {final_val_metrics["fp"]}, FN: {final_val_metrics["fn"]}, TN: {final_val_metrics["tn"]}')
    
    # Update best validation F1 if final validation is better
    if final_val_metrics['f1'] > best_val_f1:
        best_val_f1 = final_val_metrics['f1']
        print(f'  *** Final validation F1 ({best_val_f1:.4f}) is the best! ***')
        # Save final best model
        import os
        os.makedirs(model_save_dir, exist_ok=True)
        best_model_path = f'{model_save_dir}/model_best.pth'
        checkpoint = {
            'epoch': cfg.TRAIN.NUM_EPOCHS,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_f1': best_val_f1,
            'config': cfg
        }
        torch.save(checkpoint, best_model_path)
        print(f'  Best model saved to {best_model_path}')
    
    # Save visualization
    if viz_save_dir is None:
        import os
        os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)
        viz_save_dir = f'{output_dir}/{exp_name}'
    import os
    os.makedirs(viz_save_dir, exist_ok=True)
    plot_performance(
        epoch_numbers=epoch_numbers,
        train_losses=train_losses,
        val_losses=val_losses,
        train_accuracies=train_accuracies,
        val_accuracies=val_accuracies,
        train_f1_scores=train_f1_scores,
        val_f1_scores=val_f1_scores,
        train_precisions=train_precisions,
        val_precisions=val_precisions,
        train_recalls=train_recalls,
        val_recalls=val_recalls,
        best_f1=best_f1,
        best_val_f1=best_val_f1,
        output_dir=viz_save_dir,
        exp_name=exp_name
    )


if __name__ == "__main__":
    # PATHS
    cfg = get_cfg_defaults()
    set_seed(cfg.SYSTEM.SEED)
    
    # Load device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data paths
    gt_annotations_root = 'data/v1.0/data/MUSCIMA++/v2.0/data/annotations'
    images_root = 'data/v1.0/data/MUSCIMA++/datasets_r_staff/images'
    split_file = cfg.SPFILE
    
    print('Configuration loaded!')
    print(f'Number of epochs: {cfg.TRAIN.NUM_EPOCHS}')
    print(f'Batch size: {cfg.TRAIN.BATCH_SIZE}')
    print(f'Learning rate: {cfg.TRAIN.LEARNING_RATE}')
    print(f'POS_WEIGHT: {cfg.TRAIN.POS_WEIGHT}')
    
    parser = argparse.ArgumentParser(
        description="Train assembly notation model using ground truth labels with balanced training"
    )

    parser.add_argument('-c', '--classes',
                        default='essential',
                        help="Classes to use. Options: ['essn', 'essential', '20', 'all']")
    parser.add_argument('--exp_name',
                        type=str,
                        required=True,
                        help="Experiment name for organizing outputs")
    parser.add_argument('--train',
                        action="store_true",
                        help="training and validation")
    parser.add_argument('--test_only',
                        action="store_true",
                        help="Only run testing (no training)")
    parser.add_argument('--threshold',
                        type=float,
                        default=0.5,
                        help="set threshold for binary classification")

    args = parser.parse_args()
    
    classes = args.classes 
    class_list, class_dict = get_classlist_and_classdict(classes)
    class_list = list(class_list)
    print(f'Loaded {len(class_list)} classes')
    
    # Load data configuration
    with open(cfg.DATA.DATA_CONFIG, 'rb') as hdl:
        data_config = yaml.load(hdl, Loader=yaml.FullLoader)
    data_config['mode'] = cfg.MODEL.MODE
    
    if args.train:
        # Load data
        data = load_data(gt_annotations_root, images_root, split_file, class_list, class_dict, data_config)
        data_train = data['train']
        data_valid = data['valid']
        print(f'\nTraining samples: {len(data["train"]):,}')
        print(f'Validation samples: {len(data["valid"]):,}')
        
        model = load_model(cfg, device=device)
        train_mlp(model, cfg, device, data_train, data_valid, None, args.exp_name, args.threshold)
    
    if args.test_only:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.TRAIN.POS_WEIGHT).to(device))
        test_data = load_test_data(gt_annotations_root, images_root, split_file, class_list, class_dict, data_config)
        test_data = test_data['test']
        test_num_workers = getattr(cfg.TRAIN, 'NUM_WORKERS', 4) if hasattr(cfg.TRAIN, 'NUM_WORKERS') else 4
        test_loader = DataLoader(
            test_data, 
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False, 
            num_workers=test_num_workers,
            pin_memory=torch.cuda.is_available()
        )
        model = load_model(cfg, device=device)
        # Load best model if exists
        best_model_path = f'model/mlp/{args.exp_name}_best.pth'
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model'])
            print(f'Loaded model from {best_model_path}')
        evaluate_test_model(model, test_loader, criterion, device, threshold=args.threshold)

