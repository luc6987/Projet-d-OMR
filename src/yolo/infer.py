"""
YOLO inference module
Refactored to use configuration directly
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.config_loader import ConfigLoader
from common.logger import OMRLogger
from common.paths import PathManager

from .inference_tiled import TiledInference


def infer_yolo(config: ConfigLoader, path_manager: PathManager) -> None:
    """Run YOLO inference"""
    logger = OMRLogger.get_logger('yolo')
    
    yolo_config = config.get_module_config('yolo')
    infer_config = yolo_config.get('infer', {})
    
    if not infer_config.get('enabled', True):
        logger.info("YOLO inference is disabled in config")
        return
    
    logger.info("Starting YOLO inference...")
    
    model_path = path_manager.resolve_path(infer_config.get('model_path', 'model/yolo/detect/yolo12l_muscima_finetune/weights/best.pt'))
    input_image = path_manager.resolve_path(infer_config.get('input_image_path', 'data/v1.0/data/images/w-01/symbol/p001.png'))
    output_dir = path_manager.resolve_path(infer_config.get('output_dir', 'Output/inference_output'))
    
    logger.info(f"Model: {model_path}")
    logger.info(f"Input: {input_image}")
    logger.info(f"Output: {output_dir}")
    
    inferencer = TiledInference(
        model_path=str(model_path),
        tile_size=infer_config.get('tile_size', 1216),
        target_size=infer_config.get('target_size', 640),
        overlap=infer_config.get('overlap', 100),
        confidence_threshold=infer_config.get('confidence_threshold', 0.25),
        iou_threshold=infer_config.get('iou_threshold', 0.45),
        nms_iou_threshold=infer_config.get('nms_iou_threshold', 0.5),
    )
    
    inferencer.output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    annotated_image, filtered_detections, all_detections, crop_visualization, combined_visualization = inferencer.process_image(str(input_image))
    
    # Save results based on config flags
    save_options = {
        'save_detected_image': infer_config.get('save_detected_image', True),
        'save_crop_visualization': infer_config.get('save_crop_visualization', True),
        'save_combined_visualization': infer_config.get('save_combined_visualization', True),
        'save_results_json': infer_config.get('save_results_json', True),
        'save_all_results_json': infer_config.get('save_all_results_json', True),
        'save_report_txt': infer_config.get('save_report_txt', True),
    }
    
    inferencer.save_results(
        str(input_image),
        annotated_image,
        filtered_detections,
        all_detections,
        crop_visualization,
        combined_visualization,
        **save_options
    )
    
    logger.info("YOLO inference completed")
