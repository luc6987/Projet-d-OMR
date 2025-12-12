import torch
import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple, List

# Add path to find model and configs
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / 'src' / 'mlp' / 'assemblage'))

from mlp.model import MLP
from mlp.assemblage.configs.assembler.default import get_cfg_defaults
from common.constants import CLASS_DICT_ALL, CLASS_LIST_ESSN

class Linker:
    """
    Uses a trained MLP to predict relationships between musical symbols.
    """
    def __init__(self, model_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.class_dict = self._build_class_dict()
        
    def _build_class_dict(self) -> Dict[str, int]:
        """
        Build class dictionary. Manually parse XML since parse_node_classes may fail.
        The XML uses YOLO naming (notehead-full) with IDs from the XML.
        """
        from lxml import etree
        from pathlib import Path
        
        # Try to manually parse XML file
        # Resolve path relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        xml_path = project_root / "data" / "v1.0" / "specifications" / "mff-muscima-mlclasses-annot.xml"
        if xml_path.exists():
            try:
                tree = etree.parse(str(xml_path))
                root = tree.getroot()
                
                class_dict = {}
                classes = root.findall('.//CropObjectClass')
                for cls_elem in classes:
                    name_elem = cls_elem.find('Name')
                    id_elem = cls_elem.find('Id')
                    if name_elem is not None and id_elem is not None:
                        name = name_elem.text
                        class_id = int(id_elem.text)
                        class_dict[name] = class_id
                
                if len(class_dict) > 0:
                    print(f"[Linker] Loaded {len(class_dict)} classes from XML (YOLO naming)")
                    return class_dict
            except Exception as e:
                print(f"[Linker] Warning: Failed to parse XML manually: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback: use CLASS_LIST_ESSN with sequential IDs
        # This matches what the model was likely trained with if XML parsing failed during training
        from common.constants import CLASS_LIST_ESSN
        class_dict = {name: idx for idx, name in enumerate(CLASS_LIST_ESSN)}
        print(f"[Linker] Using fallback mapping with {len(class_dict)} classes from CLASS_LIST_ESSN (index-based)")
        return class_dict

    def _load_model(self, path: Path) -> MLP:
        if not path.exists():
            raise FileNotFoundError(f"MLP model not found at {path}")
        
        # Try to load config from training output directory
        training_dir = path.parent
        config_path = training_dir / "config.yaml"
        
        cfg = get_cfg_defaults()
        if config_path.exists():
            try:
                import yaml
                with open(config_path, 'r') as f:
                    training_config = yaml.safe_load(f)
                
                # Update config with training parameters
                if 'MODEL' in training_config:
                    model_cfg = training_config['MODEL']
                    if 'VOCAB_DIM' in model_cfg:
                        cfg.MODEL.VOCAB_DIM = model_cfg['VOCAB_DIM']
                    if 'EMBEDDING_DIM' in model_cfg:
                        cfg.MODEL.EMBEDDING_DIM = model_cfg['EMBEDDING_DIM']
                    if 'MLP_CONFIG' in model_cfg:
                        cfg.MODEL.MLP_CONFIG = model_cfg['MLP_CONFIG']
                
                print(f"[Linker] Loaded config from {config_path}")
                print(f"  VOCAB_DIM: {cfg.MODEL.VOCAB_DIM}, EMBEDDING_DIM: {cfg.MODEL.EMBEDDING_DIM}")
            except Exception as e:
                print(f"[Linker] Warning: Failed to load training config: {e}")
        
        model = MLP(cfg)
        
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        print(f"[Linker] Model loaded successfully (VOCAB_DIM={cfg.MODEL.VOCAB_DIM})")
        return model
        
    def predict(self, source_sym, target_sym, image_shape: Tuple[int, int]) -> float:
        """
        Predicts probability that source and target are linked.
        image_shape: (height, width)
        """
        # Prepare input tensors
        # Normalize BBox: [x1, y1, x2, y2] -> normalized
        # MuNG uses (top, left, bottom, right) i.e. (y1, x1, y2, x2)
        # YOLO uses (x1, y1, x2, y2)
        
        # We need to match the normalization logic in data_pool_gt.py
        # reshape_weight = [2/H, 2/H, 2/H, 2/H]
        # reshape_bias = [-1, -W/H, -1, -W/H]
        # Formula: coord * 2/H + bias
        
        # The logic in data_pool_gt lines 144-147 seems to imply the bbox vector is expected to be:
        # index 0 (y1?): * 2/H - 1
        # index 1 (x1?): * 2/H - W/H
        
        # If source_sym.bbox is [x1, y1, x2, y2], we need to be careful about order.
        # MuNG's bounding_box property usually returns (top, left, bottom, right).
        # So index 0 is Y, index 1 is X.
        
        # Our Symbol class (from YOLO) has bbox = [x1, y1, x2, y2].
        # So we should convert to [y1, x1, y2, x2] to match MuNG/Training data expectation?
        # Let's assume the training data used MuNG's native order (top, left, bottom, right).
        
        h, w = image_shape
        
        def normalize(bbox_xyxy):
            x1, y1, x2, y2 = bbox_xyxy
            # Convert to [y1, x1, y2, x2]
            bbox_yxyx = np.array([y1, x1, y2, x2], dtype=np.float32)
            
            weight = np.array([2/h, 2/h, 2/h, 2/h], dtype=np.float32)
            bias = np.array([-1, -w/h, -1, -w/h], dtype=np.float32)
            
            return torch.tensor(bbox_yxyx * weight + bias, dtype=torch.float32)

        src_bbox = normalize(source_sym.bbox)
        tgt_bbox = normalize(target_sym.bbox)
        
        # Class IDs
        # First try direct YOLO name (XML uses YOLO naming)
        src_name = source_sym.class_name
        tgt_name = target_sym.class_name
        
        # If not found, try mapping to MUSCIMA++ naming
        if src_name not in self.class_dict:
            src_name = self._map_class_name(source_sym.class_name)
        if tgt_name not in self.class_dict:
            tgt_name = self._map_class_name(target_sym.class_name)
        
        # Get IDs (use 0 as fallback for unknown classes)
        src_id = torch.tensor([self.class_dict.get(src_name, 0)], dtype=torch.long)
        tgt_id = torch.tensor([self.class_dict.get(tgt_name, 0)], dtype=torch.long)
        
        # Debug: warn if class not found
        if src_name not in self.class_dict:
            print(f"[Linker] Warning: Source class '{source_sym.class_name}' -> '{src_name}' not in dict, using ID 0")
        if tgt_name not in self.class_dict:
            print(f"[Linker] Warning: Target class '{target_sym.class_name}' -> '{tgt_name}' not in dict, using ID 0")
        
        batch = {
            'source_bbox': src_bbox.unsqueeze(0).to(self.device),
            'target_bbox': tgt_bbox.unsqueeze(0).to(self.device),
            'source_class': src_id.to(self.device),
            'target_class': tgt_id.to(self.device)
        }
        
        with torch.no_grad():
            prob = self.model(batch, apply_sigmoid=True)
            
        return prob.item()

    def _map_class_name(self, yolo_name: str) -> str:
        """
        Map YOLO class names. If XML was parsed successfully, it uses YOLO naming directly.
        Otherwise, map to MUSCIMA++ naming (camelCase).
        """
        # First check if exact match exists (XML might use YOLO naming directly)
        if yolo_name in self.class_dict:
            return yolo_name
        
        # If not found, try mapping to MUSCIMA++ naming (for fallback case)
        mapping = {
            'notehead-full': 'noteheadFull',
            'notehead-empty': 'noteheadHalf',
            'grace-notehead-full': 'graceNoteAcciaccatura',
            'stem': 'stem',
            '8th_flag': 'flag8thUp',
            '16th_flag': 'flag16thUp',
            'beam': 'beam',
            'sharp': 'accidentalSharp',
            'flat': 'accidentalFlat',
            'natural': 'accidentalNatural',
            'duration-dot': 'augmentationDot',
            'staccato-dot': 'articulationStaccato',
            'g-clef': 'gClef',
            'f-clef': 'fClef',
            'c-clef': 'cClef',
            '8th_rest': 'rest8th',
            '16th_rest': 'rest16th',
            'quarter_rest': 'restQuarter',
            'half_rest': 'restHalf',
            'whole_rest': 'restWhole',
        }
        
        muscima_name = mapping.get(yolo_name, None)
        if muscima_name and muscima_name in self.class_dict:
            return muscima_name
        
        # Last resort: return as-is or unclassified
        if 'unclassified' in self.class_dict:
            return 'unclassified'
        return yolo_name  # Return original, might work if IDs align

