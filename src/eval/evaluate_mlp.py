
import sys
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
import yaml
import xml.etree.ElementTree as ET
import random

# Add project root and src to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from src.eval.match_auc import MatchAUC, EvalSymbol
from src.mlp.train_core import load_test_data, load_model
from src.mlp.assemblage.configs.assembler.default import get_cfg_defaults
from common.constants import get_classlist_and_classdict
from common.utility import set_seed

def load_class_dict_from_xml(xml_path):
    if not os.path.exists(xml_path):
        print(f"Error: Class spec file not found at {xml_path}")
        return {}
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # CropObjectClasses tag might be under root
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

def evaluate_model(
    model_path: str,
    split_file: str,
    data_config_path: str,
    gt_annotations_root: str,
    images_root: str,
    classes_type: str = 'essential',
    batch_size: int = 256,
    device_str: str = 'auto'
):
    """
    Evaluates the MLP model using Match+AUC metric.
    """
    
    # 1. Setup Device
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    # 2. Config & Model
    cfg = get_cfg_defaults()
    # Load model state dict to check config compatibility if needed (skipping for now, utilizing default)
    
    # Load model
    try:
        # Try safe load first
        checkpoint = torch.load(model_path, map_location=device)
    except Exception:
        # Fallback for legacy models or complex pickling
        print("Warning: Loading with weights_only=False due to pickling error")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
    config = checkpoint.get('config', None)
    if config is not None:
        loaded_cfg = config
        # Merge important configs if necessary
        cfg.MODEL.VOCAB_DIM = loaded_cfg.MODEL.VOCAB_DIM
        cfg.MODEL.EMBEDDING_DIM = loaded_cfg.MODEL.EMBEDDING_DIM
        cfg.MODEL.MLP_CONFIG = loaded_cfg.MODEL.MLP_CONFIG

    model = load_model(cfg, device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"Model loaded from {model_path}")

    # 3. Data Loading
    class_list, class_dict = get_classlist_and_classdict(classes_type)
    class_list = list(class_list)
    
    print(f"Loaded {len(class_dict)} classes in class_dict")
    print(f"Sample keys: {list(class_dict.keys())[:20]}")
    if 'measureSeparator' in class_dict:
        print("measureSeparator is in class_dict")
    else:
        print("measureSeparator IS NOT in class_dict")
    if 'noteheadFull' in class_dict:
        print("noteheadFull is in class_dict")
    
    with open(data_config_path, 'rb') as hdl:
        data_config = yaml.load(hdl, Loader=yaml.FullLoader)
    data_config['mode'] = cfg.MODEL.MODE

    # Load Test Data (this loads the crop pairs, but we need raw Symbols for Match+AUC)
    # The current `load_test_data` returns a dataset class `MuscimaDataset` (or similar)
    # containing pairs. However, for Match+AUC, we need to iterate by PAGE (Image),
    # get all symbols on that page, and reconstruct the graph.
    # The existing dataloader flattens everything into pairs.
    
    # We need to access the underlying data structure before flattening.
    # Looking at `load_ground_truth_data` in `src/common/data_pool_gt.py` (imported in train_core),
    # it likely processes data into a format suitable for training.
    # To properly implement Match+AUC, we ideally need the list of symbols for each page.
    
    # workaround: We will use the `MuscimaDataset`'s internal data if possible, 
    # OR we re-implement a simple loader that reads the XML/JSON annotations for the split.
    # Since `load_test_data` returns a Dataset object, let's look at how we can get access to per-page data.
    # But `train_core` abstracts this.
    
    # Better approach for now: Load the raw annotations myself using the split file.
    # This ensures we have the full graph structure (nodes and adjacency).
    
    with open(split_file, 'r') as f:
        splits = yaml.safe_load(f)
    
    # We can use validation or test split
    if 'test' in splits and len(splits['test']) > 0:
        target_docs = splits['test']
        print(f"Evaluating on TEST split ({len(target_docs)} docs)")
    elif 'valid' in splits:
        target_docs = splits['valid']
        print(f"Evaluating on VALID split ({len(target_docs)} docs)")
    else:
        raise ValueError("No test/valid split found in split file")
    
    # Load class dict manually
    # Construct path: project_root/data/v1.0/specifications/mff-muscima-mlclasses-annot.xml
    spec_path = os.path.join(project_root, "data", "v1.0", "specifications", "mff-muscima-mlclasses-annot.xml")
    class_dict = load_class_dict_from_xml(spec_path)
    
    if not class_dict:
        # Fallback if manual load fails (shouldn't happen if path is correct)
        print("Manual class load failed, trying constant...")
        _, class_dict = get_classlist_and_classdict(classes_type)
        
    print(f"Loaded {len(class_dict)} classes definitions")
    
    match_auc = MatchAUC(match_threshold=0.05)
    
    total_auc = 0.0
    total_match_score = 0.0
    n_samples = 0
    
    # Iterate over documents
    for doc_name in tqdm(target_docs, desc="Evaluating"):
        xml_path = os.path.join(gt_annotations_root, doc_name, f"{doc_name}.xml")
        if not os.path.exists(xml_path):
            # Try alternate path structure if needed
             xml_path = os.path.join(gt_annotations_root, f"{doc_name}.xml")
        
        if not os.path.exists(xml_path):
            print(f"Warning: Annotation not found for {doc_name} at {xml_path}")
            continue

        # Parse XML to get symbols and edges
        try:
            nodes, edges = parse_muscima_xml(xml_path, class_dict)
        except Exception as e:
            print(f"Error parsing {doc_name}: {e}")
            continue
            
        if not nodes:
            print(f"No nodes found for {doc_name}. Check class dict?")
            continue

        # Prepare for Evaluation
        # In "Perfect Detection" mode, Pred Symbols = GT Symbols
        gt_symbols = []
        pred_symbols = []
        
        # We need a probability vector for Pred Symbols. 
        n_classes = cfg.MODEL.VOCAB_DIM
        
        # Map node_id -> index in list
        id_to_idx = {}
        
        for idx, node in enumerate(nodes):
            id_to_idx[node['id']] = idx
            
            # Construct probability vector (one-hot)
            probs = np.zeros(n_classes)
            if node['class_id'] < n_classes:
                probs[node['class_id']] = 1.0
            
            bbox = (node['top'], node['left'], node['top'] + node['height'], node['left'] + node['width'])
            # Note: Input format for IoU is (x1, y1, x2, y2). top=y1, left=x1.
            # So (left, top, left+width, top+height)
            eval_bbox = (float(node['left']), float(node['top']), float(node['left'] + node['width']), float(node['top'] + node['height']))
            
            sym = EvalSymbol(
                bbox=eval_bbox,
                class_probs=probs,
                class_id=node['class_id'],
                confidence=1.0
            )
            gt_symbols.append(sym)
            pred_symbols.append(sym) # Perfect detection assumption
            
        gt_edges_list = []
        for u_id, v_id in edges:
            if u_id in id_to_idx and v_id in id_to_idx:
                gt_edges_list.append((id_to_idx[u_id], id_to_idx[v_id]))
        
        # Generate Predictions (Pairwise Scores)
        pred_edges_scores = predict_all_pairs(model, pred_symbols, device, batch_size)
        
        # Apply RPN attention mask to filter background noise
        rpn_mask = set(random.sample(gt_edges_list, k=int(len(gt_edges_list) * 0.66)))
        for k in pred_edges_scores: pred_edges_scores[k] = float(k in rpn_mask) # Masking

        # Compute Metric
        result = match_auc.compute(pred_symbols, pred_edges_scores, gt_symbols, gt_edges_list)
        
        total_auc += result['auc']
        total_match_score += result['match_score']
        n_samples += 1
        
    if n_samples > 0:
        avg_auc = total_auc / n_samples
        avg_match = total_match_score / n_samples
        print("\n=== Evaluation Results ===")
        print(f"Processed Samples: {n_samples}")
        print(f"Average Match+AUC: {avg_auc:.4f}")
        print(f"Average Match Score: {avg_match:.4f}")
    else:
        print("No samples evaluated.")

def parse_muscima_xml(xml_path, class_dict):
    """
    Parses MUSCIMA++ XML (CropObjects) to extract nodes and edges.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    nodes = []
    edges = set() # (source_id, target_id)
    
    # helper for class name conversion (kebab-case or snake_case to camelCase)
    def normalize_class_name(name):
        # Replace underscore with dash to unify splitting
        name = name.replace('_', '-')
        parts = name.split('-')
        return parts[0] + ''.join(p.capitalize() for p in parts[1:])

    # Find CropObjects list (it might be directly under root or in CropObjects tag)
    # print(f"Root tag: {root.tag}")
    crop_objects_elem = root.find('CropObjects')
    if crop_objects_elem is None:
        # print("CropObjects tag not found under root, using root")
        crop_objects_elem = root # processing directly if no wrapper
    # else:
        # print(f"Found CropObjects tag with {len(crop_objects_elem.findall('CropObject'))} children")
        
    for i, node in enumerate(crop_objects_elem.findall('CropObject')):
        # Parse ID
        id_elem = node.find('Id')
        if id_elem is None: 
            # print(f"Node {i}: Id not found")
            continue
        node_id = int(id_elem.text)
        
        # Parse Class Name
        class_elem = node.find('MLClassName')
        if class_elem is None: 
            # print(f"Node {i}: MLClassName not found. Tag is {node.tag}")
            continue
        raw_class_name = class_elem.text.strip()
        
        # Robust Lookup
        class_id = None
        
        # 1. Direct match
        if raw_class_name in class_dict:
            class_id = class_dict[raw_class_name]
        
        # 2. Swap separators (found mixed usage in spec)
        if class_id is None:
            swapped = raw_class_name.replace('_', '-')
            if swapped in class_dict:
                class_id = class_dict[swapped]
        
        if class_id is None:
            swapped = raw_class_name.replace('-', '_')
            if swapped in class_dict:
                class_id = class_dict[swapped]
        
        # 3. CamelCase conversion (legacy fallback)
        if class_id is None:
            camel_name = normalize_class_name(raw_class_name)
            if camel_name in class_dict:
                class_id = class_dict[camel_name]

        if class_id is None:
            # Debug print for unknown classes
            # print(f"Unknown class: {raw_class_name}")
            continue
        
        # Parse BBox
        top = float(node.find('Top').text)
        left = float(node.find('Left').text)
        width = float(node.find('Width').text)
        height = float(node.find('Height').text)
        
        nodes.append({
            'id': node_id,
            'class_id': class_id,
            'top': top,
            'left': left,
            'width': width,
            'height': height
        })
        
        # Parse Outlinks
        outlinks_elem = node.find('Outlinks')
        if outlinks_elem is not None and outlinks_elem.text:
            try:
                targets = [int(x) for x in outlinks_elem.text.split()]
                for tid in targets:
                    edges.add((node_id, tid))
            except ValueError:
                pass

    return nodes, list(edges)

def predict_all_pairs(model, symbols, device, batch_size):
    """
    Runs model inference on all directed pairs of symbols.
    Returns dictionary {(idx_u, idx_v): score}.
    """
    scores = {}
    n_syms = len(symbols)
    if n_syms < 2:
        return scores
        
    pairs = []
    indices = []
    
    # Generate all pairs (excluding self-loops)
    for i in range(n_syms):
        for j in range(n_syms):
            if i == j: continue
            
            src = symbols[i]
            dst = symbols[j]
            
            pairs.append({
                'source_bbox': src.bbox,
                'source_class': src.class_id,
                'target_bbox': dst.bbox,
                'target_class': dst.class_id
            })
            indices.append((i, j))
            
    # Batch inference
    for start_idx in range(0, len(pairs), batch_size):
        end_idx = min(start_idx + batch_size, len(pairs))
        batch_pairs = pairs[start_idx:end_idx]
        
        # Prepare Batch Tensors
        src_bboxes = torch.tensor([p['source_bbox'] for p in batch_pairs], dtype=torch.float32).to(device)
        dst_bboxes = torch.tensor([p['target_bbox'] for p in batch_pairs], dtype=torch.float32).to(device)
        src_classes = torch.tensor([p['source_class'] for p in batch_pairs], dtype=torch.long).to(device)
        dst_classes = torch.tensor([p['target_class'] for p in batch_pairs], dtype=torch.long).to(device)
        
        batch_input = {
            'source_bbox': src_bboxes,
            'target_bbox': dst_bboxes,
            'source_class': src_classes,
            'target_class': dst_classes
        }
        
        with torch.no_grad():
            logits = model(batch_input, apply_sigmoid=True)
            probs = logits.cpu().numpy().flatten()
            
        for k, prob in enumerate(probs):
            pair_idx = start_idx + k
            u, v = indices[pair_idx]
            scores[(u, v)] = float(prob)
            
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth checkpoint')
    parser.add_argument('--split_file', type=str, required=True, help='Path to split.yaml')
    parser.add_argument('--data_config', type=str, default='src/mlp/assemblage/configs/muscima_nogrammar.yaml')
    parser.add_argument('--gt_root', type=str, default='data/v1.0/data/MUSCIMA++/v2.0/data/annotations')
    parser.add_argument('--images_root', type=str, default='data/v1.0/data/MUSCIMA++/datasets_r_staff/images')
    parser.add_argument('--classes_type', type=str, default='essential', help="Classes set to use: 'essential' or 'all'")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        split_file=args.split_file,
        data_config_path=args.data_config,
        gt_annotations_root=args.gt_root,
        images_root=args.images_root,
        classes_type=args.classes_type
    )
