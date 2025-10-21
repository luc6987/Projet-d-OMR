#!/usr/bin/env python3
"""
YOLOv8l 分块推理脚本
将大图片分割成1216x1216的块，缩放到640x640进行推理，然后拼接结果
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from pathlib import Path
import json
from typing import List, Tuple, Dict
import time

class TiledInference:
    def __init__(self, model_path: str, tile_size: int = 1216, target_size: int = 640, overlap: int = 100):
        """
        初始化分块推理器
        
        Args:
            model_path: 训练好的模型路径
            tile_size: 分块尺寸 (1216x1216)
            target_size: 推理目标尺寸 (640x640)
            overlap: 块之间的重叠像素
        """
        self.model_path = model_path
        self.tile_size = tile_size
        self.target_size = target_size
        self.overlap = overlap
        # 核心区域步长：1216x1216密铺
        self.core_step = self.tile_size
        # 核心区域边距：0，因为核心区域就是完整的1216x1216
        self.core_margin = 0
        # 整个tile的扩展边距：用于显示重叠区域
        self.tile_margin = self.overlap // 2
        
        # 加载模型
        print(f"加载模型: {model_path}")
        self.model = YOLO(model_path)
        print("模型加载完成")
        
        # 创建输出目录
        self.output_dir = Path("Output/inference_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def create_tiles(self, image: np.ndarray) -> List[Tuple[np.ndarray, int, int]]:
        """
        将图片分割成块 - 智能密铺逻辑
        
        Args:
            image: 输入图片
            
        Returns:
            块列表，每个元素为 (tile_image, x_offset, y_offset)
        """
        height, width = image.shape[:2]
        tiles = []
        
        # 计算可以完整密铺的行数和列数
        full_rows = height // self.tile_size
        full_cols = width // self.tile_size
        
        # 计算剩余部分
        remaining_height = height % self.tile_size
        remaining_width = width % self.tile_size
        
        print(f"图片尺寸: {width}x{height}")
        print(f"完整块: {full_cols}列 x {full_rows}行")
        print(f"剩余: {remaining_width}像素宽 x {remaining_height}像素高")
        
        # 生成完整的块（从左上角开始密铺）
        for row in range(full_rows):
            for col in range(full_cols):
                x = col * self.tile_size
                y = row * self.tile_size
                
                # 提取完整的1216x1216块
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                tiles.append((tile, x, y))
        
        # 处理右边剩余部分（如果存在）
        if remaining_width > 0:
            # 从底边开始，多加一列
            for row in range(full_rows):
                x = width - self.tile_size  # 从右边开始
                y = row * self.tile_size
                
                # 提取块（可能不足1216宽度）
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                
                # 如果宽度不足，进行填充
                if tile.shape[1] < self.tile_size:
                    padded_tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile
                
                tiles.append((tile, x, y))
        
        # 处理下边剩余部分（如果存在）
        if remaining_height > 0:
            # 从右边开始，多加一行
            for col in range(full_cols):
                x = col * self.tile_size
                y = height - self.tile_size  # 从底边开始
                
                # 提取块（可能不足1216高度）
                tile = image[y:y+self.tile_size, x:x+self.tile_size]
                
                # 如果高度不足，进行填充
                if tile.shape[0] < self.tile_size:
                    padded_tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                    padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                    tile = padded_tile
                
                tiles.append((tile, x, y))
        
        # 处理右下角（如果右下角有剩余部分）
        if remaining_width > 0 and remaining_height > 0:
            x = width - self.tile_size
            y = height - self.tile_size
            
            # 提取块（可能不足1216x1216）
            tile = image[y:y+self.tile_size, x:x+self.tile_size]
            
            # 如果尺寸不足，进行填充
            if tile.shape[0] < self.tile_size or tile.shape[1] < self.tile_size:
                padded_tile = np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
            
            tiles.append((tile, x, y))
        
        print(f"创建了 {len(tiles)} 个块")
        return tiles
    
    def process_tile(self, tile: np.ndarray, x_offset: int, y_offset: int) -> Tuple[List[Dict], np.ndarray]:
        """
        处理单个块
        
        Args:
            tile: 图片块
            x_offset: 块在原图中的x偏移
            y_offset: 块在原图中的y偏移
            
        Returns:
            (检测结果列表, 带标注的图片)
        """
        # 缩放到目标尺寸
        resized_tile = cv2.resize(tile, (self.target_size, self.target_size))
        
        # 进行推理
        results = self.model(resized_tile, conf=0.25, iou=0.45)
        
        # 解析结果
        detections = []
        annotated_tile = resized_tile.copy()
        
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for i in range(len(boxes)):
                # 获取边界框坐标（在640x640图像中）
                box = boxes.xyxy[i].cpu().numpy()
                conf = boxes.conf[i].cpu().numpy()
                cls = int(boxes.cls[i].cpu().numpy())
                
                # 将坐标转换回1216x1216尺寸
                scale_x = self.tile_size / self.target_size
                scale_y = self.tile_size / self.target_size
                
                x1 = int(box[0] * scale_x)
                y1 = int(box[1] * scale_y)
                x2 = int(box[2] * scale_x)
                y2 = int(box[3] * scale_y)
                
                # 核心区域就是整个1216x1216，不需要边界过滤
                # 因为tiles本身就是1216x1216密铺的

                # 转换到原图坐标系
                global_x1 = x_offset + x1
                global_y1 = y_offset + y1
                global_x2 = x_offset + x2
                global_y2 = y_offset + y2
                
                detection = {
                    'bbox': [global_x1, global_y1, global_x2, global_y2],
                    'confidence': float(conf),
                    'class_id': cls,
                    'class_name': self.model.names[cls] if cls in self.model.names else f'class_{cls}'
                }
                detections.append(detection)
                
                # 在标注图上绘制边界框（使用640x640坐标）
                # 将1216坐标转换回640x640用于绘制
                draw_x1 = int(x1 * self.target_size / self.tile_size)
                draw_y1 = int(y1 * self.target_size / self.tile_size)
                draw_x2 = int(x2 * self.target_size / self.tile_size)
                draw_y2 = int(y2 * self.target_size / self.tile_size)
                
                cv2.rectangle(annotated_tile, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
                label = f"{detection['class_name']}: {conf:.2f}"
                cv2.putText(annotated_tile, label, (draw_x1, draw_y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return detections, annotated_tile
    
    def apply_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """
        应用非极大值抑制
        
        Args:
            detections: 检测结果列表
            iou_threshold: IoU阈值
            
        Returns:
            过滤后的检测结果
        """
        if len(detections) == 0:
            return detections
        
        # 创建副本以避免修改原始列表
        detections_copy = detections.copy()
        
        # 按置信度排序
        detections_copy.sort(key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        while detections_copy:
            # 取置信度最高的检测
            current = detections_copy.pop(0)
            keep.append(current)
            
            # 移除与当前检测IoU过高的其他检测
            remaining = []
            for det in detections_copy:
                iou = self.calculate_iou(current['bbox'], det['bbox'])
                if iou < iou_threshold:
                    remaining.append(det)
            detections_copy = remaining
        
        return keep
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        计算两个边界框的IoU
        
        Args:
            box1: 第一个边界框 [x1, y1, x2, y2]
            box2: 第二个边界框 [x1, y1, x2, y2]
            
        Returns:
            IoU值
        """
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def reconstruct_image(self, original_image: np.ndarray, annotated_tiles: List[Tuple[np.ndarray, int, int]]) -> np.ndarray:
        """
        重构带标注的完整图片
        
        Args:
            original_image: 原始图片
            annotated_tiles: 带标注的块列表 [(tile, x_offset, y_offset), ...]
            
        Returns:
            重构的图片
        """
        height, width = original_image.shape[:2]
        reconstructed = original_image.copy()
        
        for annotated_tile, x_offset, y_offset in annotated_tiles:
            # 将640x640的标注图缩放回1216x1216
            resized_tile = cv2.resize(annotated_tile, (self.tile_size, self.tile_size))

            # 直接粘贴整个核心区域，因为现在是1216x1216密铺
            paste_x1 = x_offset
            paste_y1 = y_offset
            paste_x2 = min(x_offset + self.tile_size, width)
            paste_y2 = min(y_offset + self.tile_size, height)

            # 计算从块中截取的区域
            tile_x1 = 0
            tile_y1 = 0
            tile_x2 = paste_x2 - paste_x1
            tile_y2 = paste_y2 - paste_y1

            if paste_x1 < paste_x2 and paste_y1 < paste_y2:
                reconstructed[paste_y1:paste_y2, paste_x1:paste_x2] = resized_tile[tile_y1:tile_y2, tile_x1:tile_x2]
        
        return reconstructed
    
    def add_crop_visualization(self, image: np.ndarray, tiles: List[Tuple[np.ndarray, int, int]]) -> np.ndarray:
        """
        在图片上添加裁剪区域的可视化
        
        Args:
            image: 原始图片
            tiles: 块列表 [(tile, x_offset, y_offset), ...]
            
        Returns:
            带裁剪区域标注的图片
        """
        height, width = image.shape[:2]
        vis_image = image.copy()
        
        # 先绘制所有红色框（核心区域）
        for tile, x_offset, y_offset in tiles:
            # 绘制核心区域（红色框 - 1216x1216密铺的内容区域）
            core_x1 = x_offset
            core_y1 = y_offset
            core_x2 = min(x_offset + self.tile_size, width)
            core_y2 = min(y_offset + self.tile_size, height)
            cv2.rectangle(vis_image, (core_x1, core_y1), (core_x2, core_y2), (0, 0, 255), 2)
            
            # 添加标签
            cv2.putText(vis_image, f"Core ({x_offset},{y_offset})", 
                       (x_offset + 5, y_offset + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 再绘制蓝色框（重叠区域）
        for tile, x_offset, y_offset in tiles:
            # 绘制重叠区域（蓝色框 - 显示相交逻辑）
            # 检查是否有重叠的tile
            has_overlap = False
            
            # 检查右边是否有重叠
            if x_offset + self.tile_size > width:
                overlap_x1 = width - self.tile_size
                overlap_y1 = y_offset
                overlap_x2 = min(overlap_x1 + self.tile_size, width)
                overlap_y2 = min(overlap_y1 + self.tile_size, height)
                cv2.rectangle(vis_image, (overlap_x1, overlap_y1), (overlap_x2, overlap_y2), (255, 0, 0), 2)
                has_overlap = True
            
            # 检查下边是否有重叠
            if y_offset + self.tile_size > height:
                overlap_x1 = x_offset
                overlap_y1 = height - self.tile_size
                overlap_x2 = min(overlap_x1 + self.tile_size, width)
                overlap_y2 = min(overlap_y1 + self.tile_size, height)
                cv2.rectangle(vis_image, (overlap_x1, overlap_y1), (overlap_x2, overlap_y2), (255, 0, 0), 2)
                has_overlap = True
            
            # 添加重叠标签
            if has_overlap:
                cv2.putText(vis_image, "Blue: Overlap", 
                           (x_offset + 5, y_offset + 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # 添加图例xf
        cv2.putText(vis_image, "Red: Core regions (1216x1216)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(vis_image, "Blue: Overlap regions", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return vis_image
    
    def process_image(self, image_path: str) -> Tuple[np.ndarray, List[Dict], List[Dict], np.ndarray]:
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            (带标注的图片, 过滤后的检测结果, 所有检测结果, 裁剪区域可视化图片)
        """
        print(f"处理图片: {image_path}")
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        print(f"图片尺寸: {image.shape[1]}x{image.shape[0]}")
        
        # 创建块
        tiles = self.create_tiles(image)
        
        # 处理所有块
        all_detections = []
        annotated_tiles = []
        
        start_time = time.time()
        for i, (tile, x_offset, y_offset) in enumerate(tiles):
            print(f"处理块 {i+1}/{len(tiles)} (偏移: {x_offset}, {y_offset})")
            
            detections, annotated_tile = self.process_tile(tile, x_offset, y_offset)
            all_detections.extend(detections)
            annotated_tiles.append((annotated_tile, x_offset, y_offset))
        
        processing_time = time.time() - start_time
        print(f"块处理完成，耗时: {processing_time:.2f}秒")
        
        # 应用NMS
        print("应用非极大值抑制...")
        filtered_detections = self.apply_nms(all_detections)
        print(f"检测到 {len(all_detections)} 个目标，NMS后剩余 {len(filtered_detections)} 个")
        
        # 重构图片
        print("重构图片...")
        reconstructed_image = self.reconstruct_image(image, annotated_tiles)
        
        # 创建裁剪区域可视化
        print("创建裁剪区域可视化...")
        crop_visualization = self.add_crop_visualization(image, tiles)
        
        # 返回所有检测结果和过滤后的结果
        return reconstructed_image, filtered_detections, all_detections, crop_visualization
    
    def save_results(self, image_path: str, annotated_image: np.ndarray, filtered_detections: List[Dict], all_detections: List[Dict], crop_visualization: np.ndarray):
        """
        保存结果
        
        Args:
            image_path: 原始图片路径
            annotated_image: 带标注的图片
            filtered_detections: NMS过滤后的检测结果
            all_detections: 所有检测结果（包括被NMS过滤的）
            crop_visualization: 裁剪区域可视化图片
        """
        # 生成输出文件名
        input_path = Path(image_path)
        base_name = input_path.stem
        
        # 保存带标注的图片
        output_image_path = self.output_dir / f"{base_name}_detected.jpg"
        cv2.imwrite(str(output_image_path), annotated_image)
        print(f"保存标注图片: {output_image_path}")
        
        # 保存裁剪区域可视化图片
        output_crop_path = self.output_dir / f"{base_name}_crop_visualization.jpg"
        cv2.imwrite(str(output_crop_path), crop_visualization)
        print(f"保存裁剪区域可视化: {output_crop_path}")
        
        # 保存NMS过滤后的检测结果JSON
        output_json_path = self.output_dir / f"{base_name}_results.json"
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_detections, f, indent=2, ensure_ascii=False)
        print(f"保存NMS过滤后的检测结果: {output_json_path}")
        
        # 保存所有检测结果JSON（包括被NMS过滤的）
        output_all_json_path = self.output_dir / f"{base_name}_all_results.json"
        with open(output_all_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_detections, f, indent=2, ensure_ascii=False)
        print(f"保存所有检测结果: {output_all_json_path}")
        
        # 保存检测报告
        output_report_path = self.output_dir / f"{base_name}_report.txt"
        with open(output_report_path, 'w', encoding='utf-8') as f:
            f.write(f"检测报告 - {base_name}\n")
            f.write("=" * 50 + "\n")
            f.write(f"总检测数量: {len(all_detections)}\n")
            f.write(f"NMS后剩余: {len(filtered_detections)}\n")
            f.write(f"被NMS过滤: {len(all_detections) - len(filtered_detections)}\n")
            f.write(f"分块尺寸: {self.tile_size}x{self.tile_size}\n")
            f.write(f"推理尺寸: {self.target_size}x{self.target_size}\n")
            f.write(f"重叠像素: {self.overlap}\n")
            f.write(f"核心区域边距: {self.core_margin}\n")
            f.write(f"tile扩展边距: {self.tile_margin}\n\n")
            
            f.write("裁剪区域说明:\n")
            f.write("-" * 30 + "\n")
            f.write("红色框: 核心区域（1216x1216密铺，用于检测）\n")
            f.write("蓝色框: 重叠区域（显示相交逻辑）\n")
            f.write("密铺逻辑: 从左上角开始密铺，不满块的部分从底边/右边开始多加一行/一列\n")
            f.write(f"核心区域步长: {self.core_step} 像素\n")
            f.write(f"tile扩展边距: {self.tile_margin} 像素\n\n")
            
            # 按类别统计（所有检测结果）
            class_counts_all = {}
            for det in all_detections:
                class_name = det['class_name']
                class_counts_all[class_name] = class_counts_all.get(class_name, 0) + 1
            
            # 按类别统计（NMS后）
            class_counts_filtered = {}
            for det in filtered_detections:
                class_name = det['class_name']
                class_counts_filtered[class_name] = class_counts_filtered.get(class_name, 0) + 1
            
            f.write("所有检测结果统计:\n")
            f.write("-" * 30 + "\n")
            for class_name, count in sorted(class_counts_all.items()):
                filtered_count = class_counts_filtered.get(class_name, 0)
                f.write(f"{class_name}: {count} (NMS后: {filtered_count})\n")
            
            f.write("\nNMS过滤后的详细检测结果:\n")
            f.write("-" * 30 + "\n")
            for i, det in enumerate(filtered_detections, 1):
                f.write(f"{i}. {det['class_name']} (置信度: {det['confidence']:.3f})\n")
                f.write(f"   位置: ({det['bbox'][0]}, {det['bbox'][1]}) - ({det['bbox'][2]}, {det['bbox'][3]})\n")
            
            f.write("\n所有检测结果（包括被NMS过滤的）:\n")
            f.write("-" * 30 + "\n")
            for i, det in enumerate(all_detections, 1):
                # 检查是否被NMS过滤
                is_filtered = det not in filtered_detections
                status = " [被NMS过滤]" if is_filtered else ""
                f.write(f"{i}. {det['class_name']} (置信度: {det['confidence']:.3f}){status}\n")
                f.write(f"   位置: ({det['bbox'][0]}, {det['bbox'][1]}) - ({det['bbox'][2]}, {det['bbox'][3]})\n")
        
        print(f"保存检测报告: {output_report_path}")

def main():
    parser = argparse.ArgumentParser(description='YOLOv8l 分块推理脚本')
    parser.add_argument('--model', type=str, 
                       default='runs/detect/yolov8l_muscima_finetune/weights/best.pt',
                       help='训练好的模型路径')
    parser.add_argument('--input', type=str,
                       default='v1.0/data/images/w-01/symbol/p001.png',
                       help='输入图片路径')
    parser.add_argument('--tile-size', type=int, default=1216,
                       help='分块尺寸')
    parser.add_argument('--target-size', type=int, default=640,
                       help='推理目标尺寸')
    parser.add_argument('--overlap', type=int, default=100,
                       help='块之间的重叠像素')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        return
    
    if not os.path.exists(args.input):
        print(f"错误: 输入图片不存在: {args.input}")
        return
    
    # 创建推理器
    inferencer = TiledInference(
        model_path=args.model,
        tile_size=args.tile_size,
        target_size=args.target_size,
        overlap=args.overlap
    )
    
    try:
        # 处理图片
        annotated_image, filtered_detections, all_detections, crop_visualization = inferencer.process_image(args.input)
        
        # 保存结果
        inferencer.save_results(args.input, annotated_image, filtered_detections, all_detections, crop_visualization)
        
        print("处理完成！")
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
