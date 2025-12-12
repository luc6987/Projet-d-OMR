"""
Analyze MLP linker training dataset statistics.
Reports sample counts, positive/negative ratios, and data characteristics.
"""
import sys
import os
# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import numpy as np
from collections import Counter
from common.data_pool_gt import load_ground_truth_data
from common.constants import get_classlist_and_classdict

def analyze_dataset(gt_annotations_root, images_root, split_file, class_schema='essential', data_config_path=None):
    """Analyze dataset and return statistics."""
    
    # Load class list and dict
    class_list, class_dict = get_classlist_and_classdict(class_schema)
    print(f"Using class schema: {class_schema}")
    print(f"Number of classes: {len(class_list)}")
    
    # Load data config
    if data_config_path is None:
        data_config_path = 'src/mlp/assemblage/configs/muscima_nogrammar.yaml'
    
    with open(data_config_path, 'rb') as hdl:
        data_config = yaml.load(hdl, Loader=yaml.FullLoader)
    
    print(f"\nLoading data from split: {split_file}")
    print(f"Data config: {data_config_path}")
    
    # Load all splits
    data = load_ground_truth_data(
        gt_annotations_root=gt_annotations_root,
        images_root=images_root,
        split_file=split_file,
        class_list=class_list,
        class_dict=class_dict,
        config=data_config,
        load_training_data=True,
        load_validation_data=True,
        load_test_data=True,
    )
    
    stats = {}
    
    for split_name in ['train', 'valid', 'test']:
        if data[split_name] is None:
            continue
            
        dataset = data[split_name]
        total_samples = len(dataset)
        
        # Count positive and negative samples
        positive_count = 0
        negative_count = 0
        class_pairs = []
        
        print(f"\n{'='*60}")
        print(f"Analyzing {split_name.upper()} split...")
        print(f"{'='*60}")
        
        for idx in range(min(total_samples, 10000)):  # Sample first 10k for speed
            sample = dataset[idx]
            label = sample['label'].item()
            if label > 0.5:
                positive_count += 1
            else:
                negative_count += 1
            
            source_class = sample['source_class'].item()
            target_class = sample['target_class'].item()
            class_pairs.append((source_class, target_class))
        
        # Scale up counts if we sampled
        if total_samples > 10000:
            scale_factor = total_samples / 10000
            positive_count = int(positive_count * scale_factor)
            negative_count = int(negative_count * scale_factor)
        else:
            # Re-count all if dataset is small
            positive_count = 0
            negative_count = 0
            class_pairs = []
            for idx in range(total_samples):
                sample = dataset[idx]
                label = sample['label'].item()
                if label > 0.5:
                    positive_count += 1
                else:
                    negative_count += 1
                source_class = sample['source_class'].item()
                target_class = sample['target_class'].item()
                class_pairs.append((source_class, target_class))
        
        positive_ratio = positive_count / total_samples if total_samples > 0 else 0
        negative_ratio = negative_count / total_samples if total_samples > 0 else 0
        
        # Count unique class pairs
        class_pair_counter = Counter(class_pairs)
        unique_pairs = len(class_pair_counter)
        
        stats[split_name] = {
            'total_samples': total_samples,
            'positive_count': positive_count,
            'negative_count': negative_count,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'unique_class_pairs': unique_pairs,
        }
        
        print(f"Total samples: {total_samples:,}")
        if total_samples > 0:
            print(f"Positive samples: {positive_count:,} ({positive_ratio*100:.2f}%)")
            print(f"Negative samples: {negative_count:,} ({negative_ratio*100:.2f}%)")
            print(f"Positive/Negative ratio: {positive_count/negative_count:.4f}" if negative_count > 0 else "N/A")
            print(f"Unique class pairs: {unique_pairs:,}")
        else:
            print("WARNING: No samples found in this split!")
    
    # Overall statistics
    print(f"\n{'='*60}")
    print("OVERALL STATISTICS")
    print(f"{'='*60}")
    total_train = stats.get('train', {}).get('total_samples', 0)
    total_valid = stats.get('valid', {}).get('total_samples', 0)
    total_test = stats.get('test', {}).get('total_samples', 0)
    total_all = total_train + total_valid + total_test
    
    print(f"Total training samples: {total_train:,}")
    print(f"Total validation samples: {total_valid:,}")
    print(f"Total test samples: {total_test:,}")
    print(f"Total all samples: {total_all:,}")
    
    if 'train' in stats:
        train_pos = stats['train']['positive_count']
        train_neg = stats['train']['negative_count']
        print(f"\nTraining set imbalance:")
        print(f"  Positive: {train_pos:,}")
        print(f"  Negative: {train_neg:,}")
        print(f"  Recommended POS_WEIGHT: {max(1, int(train_neg / train_pos))}" if train_pos > 0 else "N/A")
    
    return stats

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze MLP linker dataset")
    parser.add_argument('--gt_root', type=str, 
                       default='data/v1.0/data/annotations',
                       help='Ground truth annotations root')
    parser.add_argument('--images_root', type=str,
                       default='data/v1.0/data/images',
                       help='Images root directory')
    parser.add_argument('--split_file', type=str,
                       default='src/mlp/assemblage/splits/mob_split.yaml',
                       help='Split file path')
    parser.add_argument('--class_schema', type=str,
                       default='essential',
                       choices=['essential', 'essn', '20', 'all'],
                       help='Class schema to use')
    parser.add_argument('--data_config', type=str,
                       default='src/mlp/assemblage/configs/muscima_nogrammar.yaml',
                       help='Data configuration file')
    
    args = parser.parse_args()
    
    stats = analyze_dataset(
        gt_annotations_root=args.gt_root,
        images_root=args.images_root,
        split_file=args.split_file,
        class_schema=args.class_schema,
        data_config_path=args.data_config
    )
    
    # Save statistics to file
    output_file = 'src/mlp/dataset_stats.yaml'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        yaml.dump(stats, f, default_flow_style=False)
    print(f"\nStatistics saved to: {output_file}")

