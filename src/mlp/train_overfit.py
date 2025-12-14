
import sys
import os
import torch
import yaml
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.mlp.train_core import load_data, load_model, train_mlp
from src.mlp.assemblage.configs.assembler.default import get_cfg_defaults
from common.utility import set_seed
from common.constants import get_classlist_and_classdict

def main():
    # 1. Config
    cfg = get_cfg_defaults()
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Paths
    gt_annotations_root = 'data/v1.0/data/cropobjects_manual'
    images_root = 'data/v1.0/data/images'
    split_file = 'src/mlp/assemblage/splits/overfit_split.yaml'
    data_config_path = 'src/mlp/assemblage/configs/muscima_nogrammar.yaml'
    pretrained_model_path = 'Output/assemblage/outputs/balanced_training/model_ep90.pth'
    exp_name = 'overfit_training'
    
    # 3. Parameters
    cfg.TRAIN.NUM_EPOCHS = 2 # Fine-tune for 2 epochs
    cfg.TRAIN.LEARNING_RATE = 0.0001 # Lower LR for fine-tuning
    cfg.TRAIN.SAVE_FREQUENCY = 1 # Save every epoch
    
    # 4. Load Classes Manually
    import xml.etree.ElementTree as ET
    def load_class_dict_from_xml(xml_path):
        if not os.path.exists(xml_path):
            print(f"Error: Class spec file not found at {xml_path}")
            return {}
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            cls_root = root.find('CropObjectClasses')
            if cls_root is None: cls_root = root
            
            c_dict = {}
            for cls_node in cls_root.findall('CropObjectClass'):
                name = cls_node.find('Name').text
                cid = int(cls_node.find('Id').text)
                c_dict[name] = cid
            return c_dict
        except Exception as e:
            print(f"Error parsing class spec: {e}")
            return {}

    spec_path = os.path.join(project_root, "data", "v1.0", "specifications", "mff-muscima-mlclasses-annot.xml")
    class_dict = load_class_dict_from_xml(spec_path)
    class_list = list(class_dict.keys())
    print(f'Loaded {len(class_list)} classes manually')
    
    # 5. Load Data Config
    with open(data_config_path, 'rb') as hdl:
        data_config = yaml.load(hdl, Loader=yaml.FullLoader)
    data_config['mode'] = cfg.MODEL.MODE
    
    # 6. Load Data (Overfit Split)
    print("Loading data...")
    data = load_data(gt_annotations_root, images_root, split_file, class_list, class_dict, data_config)
    data_train = data['train']
    # data_valid = data['valid'] # Too large for quick feedback
    from torch.utils.data import Subset
    data_valid = Subset(data_train, range(min(1000, len(data_train)))) # Validate on small subset of train
    
    print(f'Training samples: {len(data_train):,}')
    print(f'Validation samples: {len(data_valid):,}')
    
    # 7. Initialize Model
    model = load_model(cfg, device=str(device))
    
    # 8. Load Pretrained Weights
    if os.path.exists(pretrained_model_path):
        print(f"Loading pretrained weights from {pretrained_model_path}")
        checkpoint = torch.load(pretrained_model_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
    else:
        print(f"Warning: Pretrained model not found at {pretrained_model_path}")

    # 9. Train
    print("Starting training...")
    train_mlp(
        model=model,
        cfg=cfg,
        device=str(device),
        train_data=data_train,
        valid_data=data_valid,
        output_dir='Output/assemblage/outputs', # Save to Output dir
        exp_name=exp_name,
        threshold=0.5
    )

if __name__ == "__main__":
    main()
