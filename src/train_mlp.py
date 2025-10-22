import torch
import numpy as np
import yaml
import os
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import argparse
from utils.data_pool_gt import load_ground_truth_data
from utils.constants import get_classlist_and_classdict
from utils.utility import set_seed
from configs.assembler.default import get_cfg_defaults
from model.model import MLP
from typing import List,Dict
import matplotlib.pyplot as plt
def load_train_data():
    
    data = load_ground_truth_data(
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
    
    
    
    
def load_test_data(gt_annotations_root: str, images_root:str, split_file:str, class_list:List, class_dict: Dict, data_config:str):
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
    
    
    
    
    
def load_data(gt_annotations_root: str, images_root:str, split_file:str, class_list:List, class_dict: Dict, data_config:str):
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
    
    
def load_model(config, device: str,):
    model = MLP(config)
    model = model.to(device)
    return model
    




# Validation function
def validate_model(model, data_loader, criterion, device, threshold=0.5):
    """
    Validate the model on validation data.
    
    Returns:
        dict with validation metrics
    """
    model.eval()
    
    all_outputs = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    print('\nValidating...')
    
    with torch.no_grad():  
        for batch in tqdm(data_loader, desc='Validation'):
      
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch['label'])
            total_loss += loss.item()
            num_batches += 1
            
            # Collect outputs and labels
            all_outputs.extend(torch.sigmoid(output).cpu().numpy().flatten())
            all_labels.extend(batch['label'].cpu().numpy().flatten())
    
    # Calculate metrics
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
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }





def train_mlp(model,cfg,device,train_data,valid_data,output,exp_name,threshold: float = 0.5):
 
    # Optimizer and loss
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.TRAIN.POS_WEIGHT).to(device))

    print('Model built!')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters()):,}')
    # Create valid loader
    valid_loader = DataLoader(
    valid_data, 
    batch_size=cfg.TRAIN.BATCH_SIZE,
    shuffle=False,  
    num_workers=0
)
    # Create data loader
    train_loader = DataLoader(
        train_data, 
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True, 
        num_workers=0  
    )
    
        
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
        
        # Run validation
        val_metrics = validate_model(model, valid_loader, criterion, device, threshold)
        
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
        print(f'  TP: {val_metrics["tp"]}, FP: {val_metrics["fp"]}, FN: {val_metrics["fn"]}, TN: {val_metrics["tn"]}')
        
        # Track best validation F1
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            print(f'  *** New best validation F1 score: {best_val_f1:.4f} ***')
            # Save best model
            best_model_path = f'{output_dir}/{exp_name}/model_best.pth'
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_f1': best_val_f1
            }
            torch.save(checkpoint, best_model_path)
            print(f'  Best model saved to {best_model_path}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f'{output_dir}/{exp_name}/model_ep{epoch+1}.pth'
            checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)
            print(f'  Checkpoint saved to {checkpoint_path}')

    print('\n=== Training Complete ===')
    print(f'Best Training F1 Score: {best_f1:.4f}')
    print(f'Best Validation F1 Score: {best_val_f1:.4f}')
    plot_perforamance(cfg.TRAIN.NUM_EPOCHS,train_losses=train_losses,val_losses = val_losses,train_accuracies=train_accuracies,val_accuracies=val_accuracies,train_f1_scores=train_f1_scores,val_f1_scores=val_f1_scores,train_precisions=train_precisions,val_precisions=val_precisions,train_recalls=train_recalls,val_recalls=val_recalls,best_f1=best_f1,best_val_f1=best_val_f1,output_dir=output,exp_name=exp_name)
    
    
    
        
        
    
def evaluate_test_model(model, data_loader, criterion, device, threshold=0.5):
    """
    Evaluate the model on test data with detailed metrics.
    
    Returns:
        dict with test metrics and predictions
    """
    model.eval()
    
    all_outputs = []
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    num_batches = 0
    
    print('\nEvaluating on test data...')
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Testing'):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            output = model(batch)
            loss = criterion(output, batch['label'])
            total_loss += loss.item()
            num_batches += 1
            
            # Get probabilities
            probs = torch.sigmoid(output).cpu().numpy().flatten()
            labels = batch['label'].cpu().numpy().flatten()
            preds = (probs >= threshold).astype(int)
            
            # Collect outputs, labels, and predictions
            all_outputs.extend(probs)
            all_labels.extend(labels)
            all_predictions.extend(preds)
    
    # Convert to numpy arrays
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    
    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean()
    tp = ((all_predictions == 1) & (all_labels == 1)).sum()
    fp = ((all_predictions == 1) & (all_labels == 0)).sum()
    fn = ((all_predictions == 0) & (all_labels == 1)).sum()
    tn = ((all_predictions == 0) & (all_labels == 0)).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    avg_loss = total_loss / num_batches
    
    # Additional statistics
    positive_rate = (all_labels == 1).mean()
    predicted_positive_rate = (all_predictions == 1).mean()
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'positive_rate': positive_rate,
        'predicted_positive_rate': predicted_positive_rate,
        'probabilities': all_outputs,
        'labels': all_labels,
        'predictions': all_predictions
    }

    
    


def plot_perforamance(epoch_numbers, train_losses, val_losses, train_accuracies,val_accuracies,train_f1_scores,val_f1_scores,train_precisions,val_precisions,train_recalls,val_recalls,best_f1,best_val_f1, output_dir, exp_name):


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
    plt.savefig(f'{output_dir}/{exp_name}/training_validation_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f'Plot saved to {output_dir}/{exp_name}/training_validation_metrics.png')

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
        
        
    
    

    




if __name__ == "__main__":
    # PATHS
    cfg = get_cfg_defaults()
    model_config_path = cfg.MODEL.MLP_PARA
    set_seed(cfg.SYSTEM.SEED)
    output_dir = 'outputs'
    split_file = cfg.SPFILE
    
    # Load device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
        
        
    # Data paths
    gt_annotations_root = 'data/MUSCIMA++/v2.0/data/annotations'
    images_root = 'data/MUSCIMA++/datasets_r_staff/images'
    split_file = 'splits/mob_split.yaml'
    
    
  
    
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

    # # Model arguments
    # parser.add_argument('--model_config',
    #                     type=str,
    #                     help="Path to model configuration YAML file")
    parser.add_argument('--output_dir',
                        type=str,
                        default="outputs",
                        help="Directory to save outputs and checkpoints")
    parser.add_argument('--exp_name',
                        type=str,
                        required=True,
                        help="Experiment name for organizing outputs")

    # Training/evaluation mode
    parser.add_argument('--train',
                        action="store_true",
                        help="training and validation")
    parser.add_argument('--test_only',
                        action="store_true",
                        help="Only run testing (no training)")
    # parser.add_argument('--load_epochs',
    #                     type=int,
    #                     default=10,
    #                     help="Epoch to load checkpoint from (-1: last, -2: all, 0+: specific)")

    parser.add_argument('--threshold',
                        default=0.5,
                        help="set threshold for binary classification")

    args = parser.parse_args()
    
    
    # Create output directory
    
    os.makedirs(f'{args.output_dir}/{args.exp_name}', exist_ok=True)
    
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
        data = load_data(gt_annotations_root,images_root,split_file,class_list,class_dict,data_config)
        data_train = data['train']
        data_valid = data['valid']
        print(f'\nTraining samples: {len(data["train"]):,}')
        print(f'Validation samples: {len(data["valid"]):,}')
        
        model = load_model(cfg, device = device)
        train_mlp(model,cfg,device,data_train,data_valid,output_dir,args.exp_name,args.threshold)
    if args.test_only:
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.TRAIN.POS_WEIGHT).to(device))

        test_data = load_test_data(gt_annotations_root, images_root, split_file, class_list, class_dict, data_config)
        test_data = test_data['test']
         # Create data loader
        test_loader = DataLoader(
        test_data, 
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False, 
        num_workers=0  
    )
        evaluate_test_model(model, test_loader, criterion, device, threshold=0.5)
        
        
        
        
        
  


    
