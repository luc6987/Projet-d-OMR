"""
Metrics and evaluation functions for MLP linker.
"""
import torch
import numpy as np
from tqdm import tqdm


def find_optimal_threshold(all_outputs, all_labels, metric='f1'):
    """
    Find optimal threshold that maximizes the specified metric.
    
    Args:
        all_outputs: numpy array of predicted probabilities
        all_labels: numpy array of true labels
        metric: 'f1', 'f2' (recall-weighted), or 'precision_recall_balance'
    
    Returns:
        optimal_threshold: float, threshold that maximizes the metric
        best_score: float, best metric score achieved
        metrics_at_best: dict with precision, recall, f1 at optimal threshold
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_score = 0.0
    optimal_threshold = 0.5
    best_metrics = {}
    
    for thresh in thresholds:
        pred = (all_outputs > thresh).astype(int)
        tp = ((pred == 1) & (all_labels == 1)).sum()
        fp = ((pred == 1) & (all_labels == 0)).sum()
        fn = ((pred == 0) & (all_labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        if metric == 'f1':
            score = f1
        elif metric == 'f2':  # F2 score weights recall higher
            f2 = (1 + 4) * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
            score = f2
        elif metric == 'precision_recall_balance':
            # Harmonic mean of precision and recall
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            score = f1
        
        if score > best_score:
            best_score = score
            optimal_threshold = thresh
            best_metrics = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
    
    return optimal_threshold, best_score, best_metrics


def validate_model(model, data_loader, criterion, device, threshold=0.5, find_optimal_thresh=False):
    """
    Validate the model on validation data.
    
    Args:
        find_optimal_thresh: If True, find optimal threshold that maximizes F1 score
    
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
    
    # Find optimal threshold if requested
    if find_optimal_thresh:
        optimal_thresh, best_f1, best_metrics = find_optimal_threshold(all_outputs, all_labels, metric='f1')
        print(f'\n  Optimal threshold: {optimal_thresh:.3f} (F1: {best_f1:.4f})')
        threshold = optimal_thresh
        # Use metrics at optimal threshold
        tp = best_metrics['tp']
        fp = best_metrics['fp']
        fn = best_metrics['fn']
        precision = best_metrics['precision']
        recall = best_metrics['recall']
        f1 = best_metrics['f1']
        pred = (all_outputs > threshold).astype(int)
        tn = ((pred == 0) & (all_labels == 0)).sum()
        accuracy = (pred == all_labels).mean()
    else:
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
        'tn': tn,
        'threshold': threshold
    }


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
    
    print(f'\n=== Test Results ===')
    print(f'Loss: {avg_loss:.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')
    print(f'Positive rate: {positive_rate:.4f}')
    print(f'Predicted positive rate: {predicted_positive_rate:.4f}')
    
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

