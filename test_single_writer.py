#!/usr/bin/env python3
"""
测试单个writer的数据处理
"""

from preprocess_data import CVCDataPreprocessor
from pathlib import Path

def test_single_writer():
    """测试处理单个writer的数据"""
    
    source_dir = '/users/eleves-a/2023/yuguang.yao/Projet-d-OMR/v1.0'
    output_dir = '/users/eleves-a/2023/yuguang.yao/Projet-d-OMR/test_single'
    
    print("测试单个writer数据处理...")
    
    # 创建预处理器
    preprocessor = CVCDataPreprocessor(source_dir, output_dir)
    
    # 创建临时目录
    (preprocessor.output_dir / 'images' / 'temp').mkdir(parents=True, exist_ok=True)
    (preprocessor.output_dir / 'labels' / 'temp').mkdir(parents=True, exist_ok=True)
    
    all_generated_files = []
    
    # 只处理w-01
    writer_dir = Path(source_dir) / 'data' / 'images' / 'w-01'
    writer_name = writer_dir.name
    print(f"处理 {writer_name}...")
    
    symbol_dir = writer_dir / 'symbol'
    if not symbol_dir.exists():
        print(f"符号目录不存在: {symbol_dir}")
        return
    
    processed_count = 0
    for img_file in sorted(symbol_dir.glob('p*.png')):
        page_num = img_file.stem  # p001, p002, etc.
        
        # 查找对应的XML文件
        page_number = str(int(page_num[1:]))  # 去掉'p'前缀并转换为整数再转回字符串
        xml_pattern = f"CVC-MUSCIMA_{writer_name}_N-{page_number}_D-ideal.xml"
        xml_path = Path(source_dir) / 'data' / 'cropobjects_manual' / xml_pattern
        
        print(f"  检查 {page_num} -> {xml_pattern}")
        print(f"  XML路径: {xml_path}")
        print(f"  XML存在: {xml_path.exists()}")
        
        if not xml_path.exists():
            print(f"  XML文件不存在: {xml_path}")
            continue
        
        # 生成输出文件前缀
        output_prefix = f"{writer_name}_{page_num}"
        
        # 处理图片
        print(f"  处理图片: {img_file}")
        generated_files = preprocessor.process_image(img_file, xml_path, output_prefix)
        all_generated_files.extend(generated_files)
        processed_count += 1
        
        print(f"  生成了 {len(generated_files)} 个样本")
        
        # 只处理前3个文件作为测试
        if processed_count >= 3:
            break
    
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
    test_single_writer()



