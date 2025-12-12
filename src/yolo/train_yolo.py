import os
import argparse
import yaml
from typing import Optional
from ultralytics import YOLO
import torch


def read_dataset_info(dataset_yaml_path: str) -> None:
    """Print dataset basic information and perform simple consistency check."""
    if not os.path.isfile(dataset_yaml_path):
        raise FileNotFoundError(f"data.yaml not found: {dataset_yaml_path}")
    with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg.get('names', []) or []
    nc = data_cfg.get('nc', None)
    if isinstance(names, dict):
        names = list(names.values())
    num_names = len(names)
    print(f"Dataset: train={data_cfg.get('train')}  val={data_cfg.get('val')}")
    print(f"Number of classes (nc)={nc}, names list length={num_names}")
    if nc is not None and num_names and nc != num_names:
        print("⚠️ Warning: nc in data.yaml does not match names length, recommend fixing to avoid unexpected behavior during training.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune YOLO12l for OMR symbols")
    parser.add_argument('--data', type=str, default='data_fixed.yaml', help='Path to data.yaml')
    parser.add_argument('--weights', type=str, default='yolo12l.pt', help='Pretrained weights path or model name')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Training image size')
    parser.add_argument('--lr0', type=float, default=5.5e-5, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW'], help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD/AdamW betas[0] as applicable')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='0' if torch.cuda.is_available() else 'cpu', help='CUDA device id(s) like "0,1" or "cpu"')
    parser.add_argument('--workers', type=int, default=min(os.cpu_count() or 8, 8), help='Dataloader workers')
    parser.add_argument('--project', type=str, default='model/yolo/detect', help='Project directory for runs')
    parser.add_argument('--name', type=str, default='yolo12l_muscima_finetune', help='Run name')
    parser.add_argument('--exist_ok', action='store_true', help='Allow existing project/name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--cache', type=str, default='ram', choices=['ram', 'disk', 'False', 'false'], help='Cache images for faster training')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze')
    parser.add_argument('--resume', type=str, default='', help='Resume training: empty for False, "True" to auto, or path to last.pt')
    parser.add_argument('--close_mosaic', action='store_true', help='Close mosaic augmentation')
    parser.add_argument('--cos_lr', action='store_true', help='Use cosine LR schedule')
    parser.add_argument('--save_period', type=int, default=-1, help='Save checkpoint every x epochs (-1 to disable)')
    parser.add_argument('--amp', action='store_true', help='Enable AMP mixed precision')
    parser.add_argument('--val', action='store_true', help='Run explicit validation after training using best.pt')
    parser.add_argument('--test', action='store_true', help='Run test set evaluation after training using best.pt')
    parser.add_argument('--export', type=str, default='', help='Export best model format, e.g., onnx, torchscript, openvino; empty to skip')
    return parser


def parse_resume_arg(resume_arg: str) -> Optional[bool or str]:
    if not resume_arg:
        return False
    v = resume_arg.strip().lower()
    if v in {'true', '1', 'yes', 'y'}:
        return True
    if v in {'false', '0', 'no', 'n'}:
        return False
    return resume_arg  # treat as path


def train(args: argparse.Namespace) -> str:
    read_dataset_info(args.data)

    if torch.cuda.is_available():
        print(f"✅ Using device: {args.device}")
    else:
        print("⚠️ CUDA not detected, using CPU for training will be very slow.")

    model = YOLO(args.weights)

    # Custom logging setup
    import json
    import time
    from datetime import datetime
    
    # Generate log file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"training_{timestamp}.json")
    
    # Initialize log structure
    log_data = {
        "training_config": vars(args),
        "history": [],
        "final_results": {}
    }
    
    def on_train_epoch_end(trainer):
        """Callback to log metrics at the end of each epoch"""
        epoch = trainer.epoch + 1
        metrics = trainer.metrics
        
        # Extract relevant metrics
        # Note: mapping might need adjustment based on exact Ultralytics version keys
        # Usually keys are like 'metrics/mAP50(B)', 'val/box_loss', 'train/box_loss' etc.
        # But trainer.metrics tells us validation results. 
        # Train losses are in trainer.loss_items (which are raw tensor values)
        
        entry = {
            "epoch": epoch,
            "learning_rate": trainer.optimizer.param_groups[0]['lr'],
            # We will populate these with available metrics
        }
        
        # Add validation metrics
        # Keys in trainer.metrics are roughly: 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss'
        for k, v in metrics.items():
            entry[k] = float(v)
            
        # Add training losses if available
        # trainer.label_loss_items contains the keys for trainer.loss_items
        if hasattr(trainer, 'loss_items'):
            labels = []
            if hasattr(trainer, 'label_loss_items'):
                labels = trainer.label_loss_items() if callable(trainer.label_loss_items) else trainer.label_loss_items
            
            # Fallback for older versions or if empty which sometimes use loss_names
            if not labels and hasattr(trainer, 'loss_names'):
                labels = trainer.loss_names

            for i, k in enumerate(labels):
                if i < len(trainer.loss_items):
                    entry[f"train/{k}"] = float(trainer.loss_items[i])
        
        log_data["history"].append(entry)
        
        # Save log periodically
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    def on_train_end(trainer):
        """Callback for training end"""
        log_data["final_results"] = {
            "best_epoch": getattr(trainer, 'best_epoch', None),
            "best_fitness": float(getattr(trainer, 'best_fitness', 0.0)) if getattr(trainer, 'best_fitness', None) is not None else None,
        }
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        print(f"📄 Training log saved to: {log_path}")

    # Register callbacks
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    # Save initial log with config
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"📄 Log initialized: {log_path}")

    resume_value = parse_resume_arg(args.resume)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        seed=args.seed,
        cache=args.cache,
        freeze=args.freeze,
        resume=resume_value,
        close_mosaic=args.close_mosaic,
        cos_lr=args.cos_lr,
        save_period=args.save_period,
        amp=args.amp,
        plots=True,
    )

    save_dir = os.path.join(args.project, args.name)
    best_weights = os.path.join(save_dir, 'weights', 'best.pt')
    print(f"Training completed, best weights: {best_weights}")
    return best_weights


def run_validation(best_weights: str, data_yaml: str, device: str) -> None:
    if not os.path.isfile(best_weights):
        print(f"⚠️ Best weights not found: {best_weights}, skipping validation.")
        return
    model_best = YOLO(best_weights)
    model_best.val(data=data_yaml, device=device, plots=True)


def run_test_evaluation(best_weights: str, data_yaml: str, device: str) -> None:
    """Perform final evaluation on test set"""
    if not os.path.isfile(best_weights):
        print(f"⚠️ Best weights not found: {best_weights}, skipping test set evaluation.")
        return
    
    # Read data.yaml configuration
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
    
    # Check if test set is configured
    if 'test' not in data_cfg:
        print("⚠️ No test set path configured in data.yaml, skipping test set evaluation.")
        return
    
    test_path = data_cfg['test']
    if not os.path.exists(test_path):
        print(f"⚠️ Test set path does not exist: {test_path}, skipping test set evaluation.")
        return
    
    print(f"🔍 Performing final evaluation on test set...")
    print(f"   Test set path: {test_path}")
    
    model_best = YOLO(best_weights)
    
    # Create test set specific data.yaml configuration
    test_data_config = {
        'train': data_cfg['train'],  # Keep training set path (for class information)
        'val': test_path,  # Use test set as validation set for evaluation
        'nc': data_cfg['nc'],
        'names': data_cfg['names']
    }
    
    # Save temporary test configuration
    test_yaml_path = os.path.join(os.path.dirname(data_yaml), 'test_data.yaml')
    with open(test_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(test_data_config, f, default_flow_style=False)
    
    try:
        # Run evaluation on test set
        results = model_best.val(data=test_yaml_path, device=device, plots=True, save_json=True)
        
        print(f"✅ Test set evaluation completed")
        print(f"   Test set mAP50: {results.box.map50:.4f}")
        print(f"   Test set mAP50-95: {results.box.map:.4f}")
        
    except Exception as e:
        print(f"❌ Test set evaluation failed: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(test_yaml_path):
            os.remove(test_yaml_path)


def run_export(best_weights: str, export_format: str) -> Optional[str]:
    if not export_format:
        return None
    if not os.path.isfile(best_weights):
        print(f"⚠️ Best weights not found: {best_weights}, skipping export.")
        return None
    model_best = YOLO(best_weights)
    exported = model_best.export(format=export_format)
    print(f"✅ Export completed: {exported}")
    return exported


def main():
    parser = build_argparser()
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    best_path = train(args)
    if args.val:
        run_validation(best_path, args.data, args.device)
    if args.test:
        run_test_evaluation(best_path, args.data, args.device)
    if args.export:
        run_export(best_path, args.export)


def train_from_config(config):
    """
    Train from configuration object
    
    Args:
        config: FinetuningConfig configuration object
    """
    import sys
    setup_path = str(Path(__file__).parent.parent / 'setup')
    if setup_path not in sys.path:
        sys.path.append(setup_path)
    from yolo.recognition.setup_finetuning import FinetuningConfig
    
    if not isinstance(config, FinetuningConfig):
        raise ValueError("config must be a FinetuningConfig instance")
    
    # Get parameters from configuration
    training_args = config.get_training_args()
    
    # Create namespace object
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    args = Args(**training_args)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Execute training
    best_path = train(args)
    
    # Execute follow-up operations
    if config.run_validation:
        run_validation(best_path, args.data, args.device)
    if config.run_test:
        run_test_evaluation(best_path, args.data, args.device)
    if config.export_format:
        run_export(best_path, config.export_format)
    
    return best_path


if __name__ == '__main__':
    main()
