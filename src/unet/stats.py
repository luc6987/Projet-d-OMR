"""
U-Net statistics module
Generate training statistics and summaries
"""
import sys
from pathlib import Path
import json
import csv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.config_loader import ConfigLoader
from common.logger import OMRLogger
from common.paths import PathManager
from glob import glob


def stats_unet(config: ConfigLoader, path_manager: PathManager) -> None:
    """Generate U-Net statistics"""
    logger = OMRLogger.get_logger('unet')
    
    unet_config = config.get_module_config('unet')
    stats_config = unet_config.get('stats', {})
    
    if not stats_config.get('enabled', True):
        logger.info("U-Net statistics is disabled in config")
        return
    
    logger.info("Generating U-Net statistics...")
    
    output_dir = path_manager.resolve_path(stats_config.get('output_dir', 'vis_stat/unet'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = path_manager.resolve_path(config.global_config.get('log_dir', 'logs'))
    
    # Find training logs
    log_files = sorted(glob(str(log_dir / "unet_training_*.json")))
    
    if not log_files:
        logger.warning("No training logs found. Run training first.")
        return
    
    # Load most recent log
    latest_log = log_files[-1]
    logger.info(f"Loading training log: {latest_log}")
    
    with open(latest_log, 'r') as f:
        log_data = json.load(f)
    
    history = log_data.get('history', [])
    final_results = log_data.get('final_results', {})
    
    # Save CSV metrics
    if stats_config.get('save_metrics_csv', True):
        csv_path = output_dir / 'training_metrics.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                'train_iou', 'val_iou', 'lr'
            ])
            for entry in history:
                writer.writerow([
                    entry['epoch'],
                    entry['train_loss'],
                    entry['val_loss'],
                    entry['train_pixel_acc'],
                    entry['val_pixel_acc'],
                    entry['train_mean_iou'],
                    entry['val_mean_iou'],
                    entry['learning_rate']
                ])
        logger.info(f"Saved metrics CSV to {csv_path}")
    
    # Save summary JSON
    if stats_config.get('save_summary_json', True):
        summary = {
            'best_epoch': final_results.get('best_epoch'),
            'best_val_iou': final_results.get('best_val_iou'),
            'best_val_loss': final_results.get('best_val_loss'),
            'final_train_loss': final_results.get('final_train_loss'),
            'final_val_loss': final_results.get('final_val_loss'),
            'final_train_iou': final_results.get('final_train_iou'),
            'final_val_iou': final_results.get('final_val_iou'),
            'total_epochs': len(history),
        }
        
        summary_path = output_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary JSON to {summary_path}")
    
    logger.info("U-Net statistics completed")
