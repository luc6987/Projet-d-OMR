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
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8l for OMR symbols")
    parser.add_argument('--data', type=str, default='../Yolo-Dataset/data.yaml', help='Path to data.yaml')
    parser.add_argument('--weights', type=str, default='../yolov8l.pt', help='Pretrained weights path or model name')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Training image size')
    parser.add_argument('--lr0', type=float, default=5.5e-5, help='Initial learning rate')
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'Adam', 'AdamW'], help='Optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD/AdamW betas[0] as applicable')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--device', type=str, default='0' if torch.cuda.is_available() else 'cpu', help='CUDA device id(s) like "0,1" or "cpu"')
    parser.add_argument('--workers', type=int, default=min(os.cpu_count() or 8, 8), help='Dataloader workers')
    parser.add_argument('--project', type=str, default='../runs/detect', help='Project directory for runs')
    parser.add_argument('--name', type=str, default='yolov8l_muscima_finetune', help='Run name')
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
    from setup_finetuning import FinetuningConfig
    
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
