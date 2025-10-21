#!/usr/bin/env python3
"""
测试修复后的preprocess_data.py脚本
"""

from preprocess_data import CVCDataPreprocessor
from pathlib import Path

def test_fixed_preprocessor():
    """测试修复后的预处理器"""
    
    source_dir = '/users/eleves-a/2023/yuguang.yao/Projet-d-OMR/v1.0'
    output_dir = '/users/eleves-a/2023/yuguang.yao/Projet-d-OMR/test_fixed'
    
    print("测试修复后的预处理器...")
    
    # 创建预处理器
    preprocessor = CVCDataPreprocessor(source_dir, output_dir)
    
    # 创建临时目录
    (preprocessor.output_dir / 'images' / 'temp').mkdir(parents=True, exist_ok=True)
    (preprocessor.output_dir / 'labels' / 'temp').mkdir(parents=True, exist_ok=True)
    
    all_generated_files = []
    
    # 测试新的XML文件遍历方法
    xml_dir = Path(source_dir) / 'data' / 'cropobjects_manual'
    
    # 只处理前3个XML文件作为测试
    xml_files = sorted(xml_dir.glob('CVC-MUSCIMA_W-*_N-*_D-ideal.xml'))[:3]
    
    for xml_file in xml_files:
        # 解析XML文件名获取writer和页面信息
        parts = xml_file.stem.split('_')
        writer_name = parts[1].lower()  # W-01 -> w-01
        page_number = parts[2].split('-')[1]  # N-10 -> 10
        
        print(f"处理 {writer_name} 页面 {page_number}...")
        
        # 查找对应的图片文件
        symbol_dir = Path(source_dir) / 'data' / 'images' / writer_name / 'symbol'
        if not symbol_dir.exists():
            print(f"  符号目录不存在: {symbol_dir}")
            continue
        
        # 构建图片文件名: 10 -> p010.png
        img_filename = f"p{page_number.zfill(3)}.png"
        img_path = symbol_dir / img_filename
        
        print(f"  查找图片: {img_path}")
        print(f"  图片存在: {img_path.exists()}")
        
        if not img_path.exists():
            print(f"  图片文件不存在: {img_path}")
            continue
        
        # 生成输出文件前缀
        output_prefix = f"{writer_name}_{img_filename[:-4]}"  # 去掉.png后缀
        
        # 处理图片
        print(f"  开始处理图片...")
        generated_files = preprocessor.process_image(img_path, xml_file, output_prefix)
        all_generated_files.extend(generated_files)
        
        print(f"  生成了 {len(generated_files)} 个样本")
    
    print(f"总共生成了 {len(all_generated_files)} 个样本")
    
    # 检查生成的文件
    temp_img_dir = preprocessor.output_dir / 'images' / 'temp'
    temp_label_dir = preprocessor.output_dir / 'labels' / 'temp'
    
    if temp_img_dir.exists():
        img_files = list(temp_img_dir.glob('*.jpg'))
        print(f"生成了 {len(img_files)} 个图片文件")
    
    if temp_label_dir.exists():
        label_files = list(temp_label_dir.glob('*.txt'))
        print(f"生成了 {len(label_files)} 个标签文件")
        
        # 显示第一个标签文件的内容
        if label_files:
            with open(label_files[0], 'r') as f:
                content = f.read().strip()
                if content:
                    lines = content.split('\n')
                    print(f"第一个标签文件包含 {len(lines)} 个标注")
                    print("示例标注:", lines[0] if lines else "无标注")

if __name__ == '__main__':
    test_fixed_preprocessor()
