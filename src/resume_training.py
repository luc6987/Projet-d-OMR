from ultralytics import YOLO
from pathlib import Path

def resume_training_from_checkpoint(checkpoint_path, data_yaml, epochs=500, batch=8, imgsz=640, lr0=5.5e-5, optimizer='AdamW', patience=100, project='../runs/detect', name='yolov8l_muscima_resume'):
    """
    Resume training from checkpoint
    
    Args:
        checkpoint_path: Checkpoint file path
        data_yaml: Data configuration file path
        epochs: Number of training epochs
        batch: Batch size
        imgsz: Image size
        lr0: Learning rate
        optimizer: Optimizer
        patience: Early stopping patience
        project: Project directory
        name: Run name
    
    Returns:
        Training results
    """
    # Load checkpoint
    model = YOLO(checkpoint_path)

    # Continue training
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        optimizer=optimizer,
        patience=patience,
        project=project,
        name=name,
        resume=True  # Use True here instead of path
    )
    
    return results


def resume_from_config(config):
    """
    Resume training from configuration object
    
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
    resume_args = config.get_resume_args()
    
    # Execute resume training
    results = resume_training_from_checkpoint(
        checkpoint_path=resume_args['weights'],
        data_yaml=resume_args['data'],
        epochs=resume_args['epochs'],
        batch=resume_args['batch'],
        imgsz=resume_args['imgsz'],
        lr0=resume_args['lr0'],
        optimizer=resume_args['optimizer'],
        patience=resume_args['patience'],
        project=resume_args['project'],
        name=resume_args['name']
    )
    
    return results


if __name__ == '__main__':
    # Resume training with default parameters
    results = resume_training_from_checkpoint(
        checkpoint_path='../runs/detect/yolov8l_muscima_finetune/weights/last.pt',
        data_yaml='../Yolo-Dataset/data.yaml'
    )