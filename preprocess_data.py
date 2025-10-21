#!/usr/bin/env python3
"""
YOLOv8l微调数据预处理脚本
从CVC-MUSCIMA数据集生成YOLO格式的训练数据
"""

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import random
import shutil
from pathlib import Path
import yaml
from typing import List, Tuple, Dict
import argparse

class CVCDataPreprocessor:
    def __init__(self, source_dir: str, output_dir: str):
        """
        初始化数据预处理器
        
        Args:
            source_dir: v1.0数据目录路径
            output_dir: 输出数据集目录路径
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # 创建输出目录结构
        self.setup_output_dirs()
        
        # 从官方类别定义文件加载类别映射（在预处理开始前会根据频次筛选为Top-K）
        self.class_mapping = self.load_class_mapping()
        self.top_k_classes = 73
        self.selected_classes: List[str] = []
        
        # 采样参数
        self.sample_size = 1216  # 采样尺寸
        self.target_size = 640   # 目标尺寸
        self.num_samples = 14    # 每张图片采样次数
        
        # 可视化标志和统计信息
        self.visualization_created = False
        self.stats = {
            'processed_images': 0,
            'total_symbols': 0,
            'symbol_relationships': 0,
            'symbol_types': {}
        }
        
    def setup_output_dirs(self):
        """创建输出目录结构"""
        dirs = [
            self.output_dir / 'images' / 'train',
            self.output_dir / 'images' / 'val',
            self.output_dir / 'images' / 'test',
            self.output_dir / 'labels' / 'train',
            self.output_dir / 'labels' / 'val',
            self.output_dir / 'labels' / 'test'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 创建可视化输出目录
        project_root = Path(__file__).resolve().parent
        vis_dir = project_root / 'Output' / 'preprocess'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    def load_class_mapping(self) -> Dict[str, int]:
        """
        从官方类别定义文件加载类别映射
        
        Returns:
            类别名称到ID的映射字典
        """
        class_mapping = {}
        
        try:
            # 读取官方类别定义文件
            class_def_path = self.source_dir / 'specifications' / 'mff-muscima-mlclasses-annot.xml'
            
            if not class_def_path.exists():
                print(f"警告: 类别定义文件不存在: {class_def_path}")
                print("使用默认的类别映射...")
                return self.get_default_class_mapping()
            
            tree = ET.parse(class_def_path)
            root = tree.getroot()
            
            # 解析所有类别定义
            for crop_object_class in root.findall('.//CropObjectClass'):
                class_id = int(crop_object_class.find('Id').text)
                class_name = crop_object_class.find('Name').text
                class_mapping[class_name] = class_id
            
            print(f"成功加载 {len(class_mapping)} 个官方类别定义")
            
        except Exception as e:
            print(f"错误: 无法加载类别定义文件: {e}")
            print("使用默认的类别映射...")
            return self.get_default_class_mapping()
        
        return class_mapping
    
    def get_default_class_mapping(self) -> Dict[str, int]:
        """
        获取默认的类别映射（作为备用）
        
        Returns:
            默认类别映射字典
        """
        return {
            'notehead-full': 0,
            'notehead-empty': 1,
            'stem': 2,
            '8th_flag': 3,
            '16th_flag': 4,
            'beam': 7,
            'duration-dot': 8,
            'sharp': 9,
            'flat': 10,
            'natural': 11,
            'whole_rest': 14,
            'half_rest': 15,
            'quarter_rest': 16,
            '8th_rest': 17,
            '16th_rest': 18,
            'ledger_line': 23,
            'grace-notehead-full': 25,
            'thin_barline': 38,
            'measure_separator': 40,
            'repeat-dot': 42,
            'staccato-dot': 46,
            'other-dot': 47,
            'tenuto': 48,
            'accent': 49,
            'slur': 52,
            'tie': 53,
            'hairpin-cresc.': 54,
            'hairpin-decr.': 55,
            'trill': 58,
            'tuple': 66,
            'g-clef': 67,
            'f-clef': 68,
            'c-clef': 69,
            'key_signature': 71,
            'time_signature': 72,
            'dynamics_text': 77,
            'other_text': 82,
            'letter_p': 98,
            'letter_f': 88,
            'letter_e': 87,
            'letter_r': 100,
            'letter_s': 101,
            'letter_o': 97,
            'letter_t': 102,
            'letter_c': 85,
            'letter_d': 86,
            'letter_m': 95,
            'numeral_3': 138,
            'numeral_4': 139
        }
            
    def parse_xml_annotations(self, xml_path: Path) -> List[Dict]:
        """
        解析XML标注文件
        
        Args:
            xml_path: XML文件路径
            
        Returns:
            符号标注列表
        """
        annotations = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for crop_object in root.findall('.//CropObject'):
                try:
                    # 尝试两种不同的类别字段名
                    class_elem = crop_object.find('MLClassName')
                    if class_elem is None:
                        class_elem = crop_object.find('ClassName')
                    
                    if class_elem is None:
                        continue
                    
                    annotation = {
                        'id': crop_object.find('Id').text,
                        'class_name': class_elem.text,
                        'top': int(crop_object.find('Top').text),
                        'left': int(crop_object.find('Left').text),
                        'width': int(crop_object.find('Width').text),
                        'height': int(crop_object.find('Height').text)
                    }
                    
                    # 只处理我们定义的类别
                    if annotation['class_name'] in self.class_mapping:
                        annotations.append(annotation)
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
            
        return annotations
    
    def parse_all_xml_annotations(self, xml_path: Path) -> List[Dict]:
        """
        解析XML标注文件，返回所有符号标注（包括未筛选的类别）
        
        Args:
            xml_path: XML文件路径
            
        Returns:
            所有符号标注列表
        """
        annotations = []
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # 尝试不同的XPath查询方式
            crop_objects = root.findall('.//CropObject')
            if not crop_objects:
                crop_objects = root.findall('CropObject')
            if not crop_objects:
                crop_objects_container = root.find('CropObjects')
                if crop_objects_container is not None:
                    crop_objects = crop_objects_container.findall('CropObject')
            
            for crop_object in crop_objects:
                try:
                    # 安全地获取元素文本
                    id_elem = crop_object.find('Id')
                    
                    # 尝试两种不同的类别字段名
                    class_elem = crop_object.find('MLClassName')
                    if class_elem is None:
                        class_elem = crop_object.find('ClassName')
                    
                    top_elem = crop_object.find('Top')
                    left_elem = crop_object.find('Left')
                    width_elem = crop_object.find('Width')
                    height_elem = crop_object.find('Height')
                    
                    if all(elem is not None for elem in [id_elem, class_elem, top_elem, left_elem, width_elem, height_elem]):
                        annotation = {
                            'id': id_elem.text,
                            'class_name': class_elem.text,
                            'top': int(top_elem.text),
                            'left': int(left_elem.text),
                            'width': int(width_elem.text),
                            'height': int(height_elem.text)
                        }
                        
                        # 包含所有符号，不进行类别筛选
                        annotations.append(annotation)
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
        return annotations
    
    def analyze_symbol_relationships(self, xml_path: Path) -> int:
        """
        分析符号之间的关系（通过Outlinks字段）
        
        Args:
            xml_path: XML文件路径
            
        Returns:
            符号关系数量
        """
        relationships = 0
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            crop_objects = root.findall('.//CropObject')
            if not crop_objects:
                crop_objects = root.findall('CropObject')
            if not crop_objects:
                crop_objects_container = root.find('CropObjects')
                if crop_objects_container is not None:
                    crop_objects = crop_objects_container.findall('CropObject')
            
            for crop_object in crop_objects:
                outlinks_elem = crop_object.find('Outlinks')
                if outlinks_elem is not None and outlinks_elem.text:
                    # Outlinks字段包含空格分隔的ID列表
                    outlink_ids = outlinks_elem.text.strip().split()
                    relationships += len(outlink_ids)
                    
        except Exception as e:
            print(f"分析符号关系时出错: {e}")
            
        return relationships
    
    def generate_random_crop(self, img_width: int, img_height: int) -> Tuple[int, int]:
        """
        生成随机裁剪位置
        
        Args:
            img_width: 图片宽度
            img_height: 图片高度
            
        Returns:
            (x, y) 裁剪起始位置
        """
        max_x = max(0, img_width - self.sample_size)
        max_y = max(0, img_height - self.sample_size)
        
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        
        return x, y
    
    def is_symbol_in_crop(self, annotation: Dict, crop_x: int, crop_y: int) -> bool:
        """
        检查符号是否与裁剪区域有重叠
        
        Args:
            annotation: 符号标注
            crop_x: 裁剪区域x坐标
            crop_y: 裁剪区域y坐标
            
        Returns:
            是否在裁剪区域内
        """
        symbol_left = annotation['left']
        symbol_top = annotation['top']
        symbol_right = symbol_left + annotation['width']
        symbol_bottom = symbol_top + annotation['height']
        
        crop_right = crop_x + self.sample_size
        crop_bottom = crop_y + self.sample_size
        
        # 只要与裁剪区域有任意交集即认为在裁剪区域内
        inter_left = max(symbol_left, crop_x)
        inter_top = max(symbol_top, crop_y)
        inter_right = min(symbol_right, crop_right)
        inter_bottom = min(symbol_bottom, crop_bottom)
        
        return inter_right > inter_left and inter_bottom > inter_top
    
    def convert_to_yolo_format(self, annotation: Dict, crop_x: int, crop_y: int) -> Tuple[int, float, float, float, float]:
        """
        将标注转换为YOLO格式
        
        Args:
            annotation: 符号标注
            crop_x: 裁剪区域x坐标
            crop_y: 裁剪区域y坐标
            
        Returns:
            (class_id, x_center, y_center, width, height) 归一化坐标
        """
        class_id = self.class_mapping[annotation['class_name']]
        
        # 计算符号在裁剪区域内的位置
        symbol_left = max(0, annotation['left'] - crop_x)
        symbol_top = max(0, annotation['top'] - crop_y)
        symbol_right = min(self.sample_size, annotation['left'] + annotation['width'] - crop_x)
        symbol_bottom = min(self.sample_size, annotation['top'] + annotation['height'] - crop_y)
        
        # 确保符号在裁剪区域内
        if symbol_right <= symbol_left or symbol_bottom <= symbol_top:
            return None
            
        # 计算中心点和尺寸
        x_center = (symbol_left + symbol_right) / 2.0
        y_center = (symbol_top + symbol_bottom) / 2.0
        width = symbol_right - symbol_left
        height = symbol_bottom - symbol_top
        
        # 归一化
        x_center_norm = x_center / self.sample_size
        y_center_norm = y_center / self.sample_size
        width_norm = width / self.sample_size
        height_norm = height / self.sample_size
        
        return class_id, x_center_norm, y_center_norm, width_norm, height_norm

    def validate_binary_image(self, img: np.ndarray, img_path: Path) -> None:
        """
        验证图像是否为二值化图像（像素值为0或1）
        
        Args:
            img: 图像数组
            img_path: 图像路径（用于错误报告）
        """
        # 检查图像是否为灰度图
        if len(img.shape) == 3:
            # 如果是彩色图像，转换为灰度图
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img
        
        # 检查像素值是否只包含0和1
        unique_values = np.unique(gray_img)
        
        # 允许的像素值：0, 1, 255（某些二值化可能使用255而不是1）
        allowed_values = {0, 1, 255}
        unexpected_values = set(unique_values) - allowed_values
        
        if unexpected_values:
            print(f"警告: 图像 {img_path} 不是标准的二值化图像")
            print(f"  发现非二值化像素值: {sorted(unexpected_values)}")
            print(f"  所有像素值: {sorted(unique_values)}")
            
            # 如果发现非二值化像素，可以选择强制二值化
            # 这里我们只发出警告，不强制转换
            print(f"  建议: 确保源数据是二值化的（像素值为0或1）")

    def compute_top_classes(self, top_k: int = 73) -> List[str]:
        """
        预扫描所有标注，统计类别频次，选择Top-K高频类别
        
        Args:
            top_k: 选择的类别数量
        Returns:
            Top-K类别名称列表
        """
        xml_dir = self.source_dir / 'data' / 'cropobjects_manual'
        class_counts: Dict[str, int] = {}
        for xml_file in sorted(xml_dir.glob('CVC-MUSCIMA_W-*_N-*_D-ideal.xml')):
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for crop_object in root.findall('.//CropObject'):
                    class_name = crop_object.find('MLClassName').text
                    if class_name:
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
            except Exception as e:
                print(f"统计类别频次时解析失败 {xml_file}: {e}")
        # 根据频次排序
        sorted_classes = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)
        top_classes = [name for name, _ in sorted_classes[:top_k]]
        print(f"选择Top-{top_k}类别，共计{len(top_classes)}类")
        return top_classes
    
    def process_image(self, img_path: Path, xml_path: Path, output_prefix: str) -> List[str]:
        """
        处理单张图片，生成多个采样
        
        Args:
            img_path: 图片路径
            xml_path: XML标注路径
            output_prefix: 输出文件前缀
            
        Returns:
            生成的文件名列表
        """
        # 确保临时目录存在
        temp_img_dir = self.output_dir / 'images' / 'temp'
        temp_label_dir = self.output_dir / 'labels' / 'temp'
        temp_img_dir.mkdir(parents=True, exist_ok=True)
        temp_label_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Error loading image: {img_path}")
            return []
            
        # 验证图像是否为二值化图像（像素值为0或1）
        self.validate_binary_image(img, img_path)
            
        img_height, img_width = img.shape[:2]
        
        # 解析标注
        annotations = self.parse_xml_annotations(xml_path)
        
        generated_files = []
        
        # 生成多个采样
        crop_positions = []  # 用于可视化：记录所有裁剪区域
        for i in range(self.num_samples):
            # 生成随机裁剪位置
            crop_x, crop_y = self.generate_random_crop(img_width, img_height)
            crop_positions.append((crop_x, crop_y))
            
            # 裁剪图片
            cropped_img = img[crop_y:crop_y+self.sample_size, crop_x:crop_x+self.sample_size]
            
            # 缩放到目标尺寸
            resized_img = cv2.resize(cropped_img, (self.target_size, self.target_size))
            
            # 保存图片
            img_filename = f"{output_prefix}_sample_{i:02d}.jpg"
            img_output_path = self.output_dir / 'images' / 'temp' / img_filename
            cv2.imwrite(str(img_output_path), resized_img)
            
            # 生成对应的标签文件
            label_filename = f"{output_prefix}_sample_{i:02d}.txt"
            label_output_path = self.output_dir / 'labels' / 'temp' / label_filename
            
            yolo_annotations = []
            for annotation in annotations:
                if self.is_symbol_in_crop(annotation, crop_x, crop_y):
                    yolo_format = self.convert_to_yolo_format(annotation, crop_x, crop_y)
                    if yolo_format is not None:
                        yolo_annotations.append(yolo_format)
            
            # 保存标签文件
            with open(label_output_path, 'w') as f:
                for class_id, x_center, y_center, width, height in yolo_annotations:
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            
            generated_files.append(img_filename)
            
        # 为每张图片生成可视化
        try:
            project_root = Path(__file__).resolve().parent
            vis_dir = project_root / 'Output' / 'preprocess'
            
            # 生成可视化文件名
            vis_filename = f"{output_prefix}.png"
            vis_path = vis_dir / vis_filename

            vis_img = img.copy()

            # 获取所有原始符号标注（包括未筛选的类别）
            all_annotations = self.parse_all_xml_annotations(xml_path)
            
            # 统计符号信息
            self.stats['processed_images'] += 1
            self.stats['total_symbols'] += len(all_annotations)
            
            # 统计符号关系
            relationships = self.analyze_symbol_relationships(xml_path)
            self.stats['symbol_relationships'] += relationships
            
            # 统计符号类型
            for ann in all_annotations:
                class_name = ann['class_name']
                self.stats['symbol_types'][class_name] = self.stats['symbol_types'].get(class_name, 0) + 1

            print(f"处理图片 {output_prefix}: 解析到 {len(all_annotations)} 个符号标注")

            # 绘制所有符号框和标注（蓝色）
            for ann in all_annotations:
                x1 = ann['left']
                y1 = ann['top']
                x2 = x1 + ann['width']
                y2 = y1 + ann['height']
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # 添加类别名称标注
                class_name = ann['class_name']
                # 计算文本位置（在框的上方）
                text_x = x1
                text_y = max(y1 - 5, 15)  # 确保文本不超出图片边界
                
                # 绘制文本背景
                text_size = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(vis_img, (text_x, text_y - text_size[1] - 5), 
                            (text_x + text_size[0], text_y), (255, 255, 255), -1)
                
                # 绘制文本
                cv2.putText(vis_img, class_name, (text_x, text_y - 2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # 绘制所有裁剪区域（绿色）
            for (cx, cy) in crop_positions:
                cv2.rectangle(vis_img, (cx, cy), (cx + self.sample_size, cy + self.sample_size), (0, 255, 0), 2)
                # 添加裁剪区域编号
                cv2.putText(vis_img, f"Crop", (cx + 5, cy + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 保持原分辨率保存
            cv2.imwrite(str(vis_path), vis_img)
            print(f"可视化图片已保存到: {vis_path}")
            
        except Exception as e:
            print(f"可视化保存失败: {e}")
            import traceback
            traceback.print_exc()

        return generated_files
    
    def split_dataset(self, all_files: List[str], train_ratio: float = 0.6, val_ratio: float = 0.2, test_ratio: float = 0.2):
        """
        划分数据集
        
        Args:
            all_files: 所有生成的文件列表
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        random.shuffle(all_files)
        
        total_files = len(all_files)
        train_count = int(total_files * train_ratio)
        val_count = int(total_files * val_ratio)
        test_count = int(total_files * test_ratio)
        
        train_files = all_files[:train_count]
        val_files = all_files[train_count:train_count + val_count]
        test_files = all_files[train_count + val_count:train_count + val_count + test_count]
        
        # 移动文件到对应目录
        temp_img_dir = self.output_dir / 'images' / 'temp'
        temp_label_dir = self.output_dir / 'labels' / 'temp'
        
        for filename in train_files:
            # 移动图片
            src_img = temp_img_dir / filename
            dst_img = self.output_dir / 'images' / 'train' / filename
            shutil.move(str(src_img), str(dst_img))
            
            # 移动标签
            label_filename = filename.replace('.jpg', '.txt')
            src_label = temp_label_dir / label_filename
            dst_label = self.output_dir / 'labels' / 'train' / label_filename
            shutil.move(str(src_label), str(dst_label))
        
        for filename in val_files:
            # 移动图片
            src_img = temp_img_dir / filename
            dst_img = self.output_dir / 'images' / 'val' / filename
            shutil.move(str(src_img), str(dst_img))
            
            # 移动标签
            label_filename = filename.replace('.jpg', '.txt')
            src_label = temp_label_dir / label_filename
            dst_label = self.output_dir / 'labels' / 'val' / label_filename
            shutil.move(str(src_label), str(dst_label))
        
        for filename in test_files:
            # 移动图片
            src_img = temp_img_dir / filename
            dst_img = self.output_dir / 'images' / 'test' / filename
            shutil.move(str(src_img), str(dst_img))
            
            # 移动标签
            label_filename = filename.replace('.jpg', '.txt')
            src_label = temp_label_dir / label_filename
            dst_label = self.output_dir / 'labels' / 'test' / label_filename
            shutil.move(str(src_label), str(dst_label))
        
        # 清理临时目录
        shutil.rmtree(temp_img_dir)
        shutil.rmtree(temp_label_dir)
        
        print(f"数据集划分完成:")
        print(f"  训练集: {len(train_files)} 个样本")
        print(f"  验证集: {len(val_files)} 个样本")
        print(f"  测试集: {len(test_files)} 个样本")
    
    def create_data_yaml(self):
        """创建data.yaml配置文件"""
        # 使用重建后的73类映射，按ID排序获取类别名称列表
        id_to_name = {v: k for k, v in self.class_mapping.items()}
        class_names = [id_to_name[i] for i in sorted(id_to_name.keys())]
        
        data_config = {
            'train': str(self.output_dir / 'images' / 'train'),
            'val': str(self.output_dir / 'images' / 'val'),
            'test': str(self.output_dir / 'images' / 'test'),
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
    
    def process_all_data(self):
        """处理所有数据"""
        print("开始处理CVC-MUSCIMA数据集...")
        
        # 创建临时目录
        (self.output_dir / 'images' / 'temp').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'labels' / 'temp').mkdir(parents=True, exist_ok=True)
        
        # 预扫描选择Top-K类别并重建类别映射
        print(f"预扫描并选择Top-{self.top_k_classes}高频类别...")
        self.selected_classes = self.compute_top_classes(self.top_k_classes)
        self.class_mapping = {name: idx for idx, name in enumerate(self.selected_classes)}
        print(f"已重建类别映射为{len(self.class_mapping)}类")

        all_generated_files = []
        
        # 遍历所有XML文件，基于XML文件查找对应的图片
        xml_dir = self.source_dir / 'data' / 'cropobjects_manual'
        
        for xml_file in sorted(xml_dir.glob('CVC-MUSCIMA_W-*_N-*_D-ideal.xml')):
            # 解析XML文件名获取writer和页面信息
            # 格式: CVC-MUSCIMA_W-01_N-10_D-ideal.xml
            parts = xml_file.stem.split('_')
            writer_name = parts[1].lower()  # W-01 -> w-01
            page_number = parts[2].split('-')[1]  # N-10 -> 10
            
            print(f"处理 {writer_name} 页面 {page_number}...")
            
            # 查找对应的图片文件
            symbol_dir = self.source_dir / 'data' / 'images' / writer_name / 'symbol'
            if not symbol_dir.exists():
                print(f"  符号目录不存在: {symbol_dir}")
                continue
            
            # 构建图片文件名: 10 -> p010.png
            img_filename = f"p{page_number.zfill(3)}.png"
            img_path = symbol_dir / img_filename
            
            if not img_path.exists():
                print(f"  图片文件不存在: {img_path}")
                continue
            
            # 生成输出文件前缀
            output_prefix = f"{writer_name}_{img_filename[:-4]}"  # 去掉.png后缀
            
            # 处理图片
            generated_files = self.process_image(img_path, xml_file, output_prefix)
            all_generated_files.extend(generated_files)
            
            print(f"  生成了 {len(generated_files)} 个样本")
        
        print(f"总共生成了 {len(all_generated_files)} 个样本")
        
        # 划分数据集
        print("划分数据集...")
        self.split_dataset(all_generated_files)
        
        # 创建配置文件
        print("创建data.yaml...")
        self.create_data_yaml()
        
        print("数据预处理完成！")
        print(f"输出目录: {self.output_dir}")
        print(f"类别数量: {len(self.class_mapping)}")
        print(f"类别列表: {list(self.class_mapping.keys())}")
        
        # 显示统计信息
        print("\n=== 数据统计信息 ===")
        print(f"1. 被标记的图片数量: {self.stats['processed_images']}")
        print(f"2. 总符号数量: {self.stats['total_symbols']}")
        print(f"3. 符号关系数量: {self.stats['symbol_relationships']}")
        
        print(f"\n符号类型统计 (Top 20):")
        sorted_types = sorted(self.stats['symbol_types'].items(), key=lambda x: x[1], reverse=True)
        for i, (symbol_type, count) in enumerate(sorted_types[:20]):
            print(f"  {i+1:2d}. {symbol_type:<25} : {count:>6} 个")
        
        if len(sorted_types) > 20:
            print(f"  ... 还有 {len(sorted_types) - 20} 种其他符号类型")
        
        print(f"\n可视化图片已保存到: {Path(__file__).resolve().parent / 'Output' / 'preprocess'}")

def main():
    parser = argparse.ArgumentParser(description='CVC-MUSCIMA数据预处理脚本')
    parser.add_argument('--source', type=str, 
                       default='/users/eleves-a/2023/yuguang.yao/Projet-d-OMR/v1.0',
                       help='源数据目录路径')
    parser.add_argument('--output', type=str,
                       default='/users/eleves-a/2023/yuguang.yao/Projet-d-OMR/Yolo-Dataset',
                       help='输出数据集目录路径')
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = CVCDataPreprocessor(args.source, args.output)
    
    # 处理数据
    preprocessor.process_all_data()

if __name__ == '__main__':
    main()
