"""
Training script for assembly notation using ground truth labels directly.
This version includes better handling of class imbalance and threshold optimization.
"""
import torch
import numpy as np
from torch.utils.data import DataLoader

from utils.data_pool_gt import load_ground_truth_data
from mung.io import read_nodes_from_file

from utils.constants import get_classlist_and_classdict
from utils.utility import set_seed
from configs.assembler.default import get_cfg_defaults
from model.model import MLP, MLPwithSoftClass, MLPwithSoftClassExtraMLP

import argparse
import os
import tqdm
import glob
import yaml
import time



def build_model(cfg):
    """Build the MLP model based on configuration."""
    if cfg.MODEL.MODE == "MLP":
        model = MLP(cfg)
    elif cfg.MODEL.MODE == "MLPwithSoftClass":
        model = MLPwithSoftClass(cfg)
    elif cfg.MODEL.MODE == "MLPwithSoftClassExtraMLP":
        model = MLPwithSoftClassExtraMLP(cfg)
    else:
        raise ValueError(f"Model {cfg.MODEL.MODE} is not supported")
    return model


def save_checkpoint(epoch, model, optimizer, save_file):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, save_file)


def load_checkpoint(model, optimizer, load_file):
    """Load model checkpoint."""
    checkpoint = torch.load(load_file)
    model.load_state_dict(checkpoint['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Loaded model at epoch {checkpoint['epoch']}.")


def train(args, data, cfg, device, model, class_list, class_dict):
    """Train the assembly model using ground truth labels."""
    print(f"\n[DEBUG] Starting training setup...")
    start_time = time.time()

    # Save configuration
    with open(f"{args.output_dir}/{args.exp_name}/config.yaml", 'w') as f:
        f.write(cfg.dump())

    if cfg.TRAIN.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
    else:
        raise ValueError(f"Optimizer {cfg.TRAIN.OPTIMIZER} is not supported")

    # Setup loss function with class weighting
    print(f"[DEBUG] Setting up loss function with POS_WEIGHT={cfg.TRAIN.POS_WEIGHT}...")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.TRAIN.POS_WEIGHT))

    # Create data loader
    print(f"[DEBUG] Creating data loader (batch_size={cfg.TRAIN.BATCH_SIZE}, num_workers={cfg.SYSTEM.NUM_WORKERS})...")
    dataloader_start = time.time()
    loader = DataLoader(data['train'], batch_size=cfg.TRAIN.BATCH_SIZE,
                        shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
    print(f"[DEBUG] Data loader created in {time.time() - dataloader_start:.2f}s")

    # Load checkpoint if resuming training
    if args.load_epochs > 0:
        load_checkpoint(model, optimizer, f"{args.output_dir}/{args.exp_name}/model_ep{args.load_epochs}.pth")
        # Skip random states to maintain reproducibility
        for _ in tqdm.tqdm(range(args.load_epochs), desc="Skipping Epochs"):
            torch.empty((), dtype=torch.int64).random_()
            torch.empty((), dtype=torch.int64).random_()

    # Training loop
    print(f"[DEBUG] Training setup complete in {time.time() - start_time:.2f}s")
    print(f"[DEBUG] Starting training loop for {cfg.TRAIN.NUM_EPOCHS - args.load_epochs} epochs...")

    model.train()
    best_f1 = 0.0
    
    for epoch in range(args.load_epochs, cfg.TRAIN.NUM_EPOCHS):
        epoch_start = time.time()

        # Metrics tracking
        all_outputs = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        print(f"\n[DEBUG] Epoch {epoch+1}/{cfg.TRAIN.NUM_EPOCHS} starting...")

        for batch_idx, batch in enumerate(tqdm.tqdm(loader, desc=f"Epoch {epoch+1}")):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch['label'])
            total_loss += loss.item()
            num_batches += 1

            # Backward pass
            loss.backward()
            optimizer.step()

            # Collect outputs and labels for threshold optimization
            all_outputs.extend(torch.sigmoid(output).detach().cpu().numpy().flatten())
            all_labels.extend(batch['label'].detach().cpu().numpy().flatten())

    
        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)
        threshold = 0.5

    
        pred = (all_outputs > threshold).astype(int)
        
        accuracy = (pred == all_labels).mean()
        tp = ((pred == 1) & (all_labels == 1)).sum()
        fp = ((pred == 1) & (all_labels == 0)).sum()
        fn = ((pred == 0) & (all_labels == 1)).sum()
        tn = ((pred == 0) & (all_labels == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        avg_loss = total_loss / num_batches

        epoch_time = time.time() - epoch_start
        print(f"\n[DEBUG] Epoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"\nEpoch {epoch+1} Training Metrics:")
        print(f"  Loss:      {avg_loss:.4f}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Threshold: {threshold:.3f}")
        print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

        # Track best F1
        if f1 > best_f1:
            best_f1 = f1
            print(f"  *** New best F1 score: {best_f1:.4f} ***")

        # Evaluate periodically
        if (epoch+1) % cfg.TRAIN.EVAL_FREQUENCY == 0:
            print(f"Validating Epoch {epoch+1}")
            with torch.no_grad():
                evaluate(args, data, cfg, device, model, class_list, class_dict)

      
        if (epoch+1) % cfg.TRAIN.SAVE_FREQUENCY == 0 and (epoch+1) != cfg.TRAIN.NUM_EPOCHS:
            save_checkpoint(epoch+1, model, optimizer,
                            f"{args.output_dir}/{args.exp_name}/model_ep{epoch+1}.pth")

   
    save_checkpoint(cfg.TRAIN.NUM_EPOCHS, model, optimizer,
                    f"{args.output_dir}/{args.exp_name}/model_final.pth")

    return model


def evaluate(args, data, cfg, device, model, class_list, class_dict, threshold=0.5):
    """Evaluate the model on validation or test set using ground truth."""
    model.eval()

    
    all_gt_files = glob.glob(args.gt_annotations_root + "/**/*.xml", recursive=True)


    with open(args.split_file, 'rb') as hdl:
        split = yaml.load(hdl, Loader=yaml.FullLoader)


    if args.test_only:
        include_names = split['test']
        data_split = data['test']
    else:
        include_names = split['valid']
        data_split = data['valid']

    gt_files_in_this_split = sorted([f for f in all_gt_files
                                      if os.path.splitext(os.path.basename(f))[0] in include_names])

    # Get inference graph (all pairs for each document)
    inference_graph = data_split.get_inference_graph() if isinstance(
        data_split.inference_graph[0], list) else data_split.inference_graph

    # Metrics tracking
    all_outputs = []
    all_labels = []

    # Evaluate each document
    for i in tqdm.tqdm(range(len(gt_files_in_this_split)), desc="Evaluating"):
        gt_file = gt_files_in_this_split[i]
        gt_list = read_nodes_from_file(gt_file)
        node_list = gt_list  # Using ground truth nodes directly

        # Create ground truth class probabilities (one-hot encoding)
        class_label = np.array([class_dict[node.class_name] for node in node_list])
        class_prob = np.zeros((len(node_list) , cfg.MODEL.VOCAB_DIM))
        class_prob[np.arange(len(class_label)), class_label] = 1

        edge_list = []
        cur_graph = inference_graph[i]

      
        for batch_idx in range((cur_graph['source_id'].shape[0] // cfg.EVAL.BATCH_SIZE) + 1):
            batch_start = batch_idx * cfg.EVAL.BATCH_SIZE
            batch_end = batch_start + cfg.EVAL.BATCH_SIZE

            batch = {k: v[batch_start:batch_end].to(device)
                     for k, v in cur_graph.items()}

            if batch['source_id'].shape[0] == 0:
                continue

    
            output = model(batch)
            output = torch.sigmoid(output)

   
            all_outputs.extend(output.detach().cpu().numpy().flatten())
            all_labels.extend(batch['label'].detach().cpu().numpy().flatten())

    # Calculate overall metrics with given threshold
    all_outputs = np.array(all_outputs)
    all_labels = np.array(all_labels)
    
    preds = (all_outputs > threshold).astype(int)

    accuracy = (preds == all_labels).mean()

    # Calculate TP, FP, FN for precision, recall, F1
    tp = ((preds == 1) & (all_labels == 1)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()
    tn = ((preds == 0) & (all_labels == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Print results
    print(f"\nValidation Metrics (threshold={threshold:.3f}):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")


def main(args, data, cfg, device, class_list, class_dict):
    """Main training/evaluation function."""
    model = build_model(cfg)
    model.to(device)

    if args.test_only or args.val_only:
        # Load model for test or validation
        if args.load_epochs == -1:  # Load last checkpoint
            if os.path.exists(f"{args.output_dir}/{args.exp_name}/model_final.pth"):
                all_ckpts = [f"{args.output_dir}/{args.exp_name}/model_final.pth"]
            else:
                model_files = [f for f in os.listdir(f"{args.output_dir}/{args.exp_name}")
                               if f.startswith("model_ep") and f.endswith(".pth")]
                assert len(model_files) > 0, "No checkpoint detected!"
                model_files = sorted(model_files,
                                     key=lambda x: int(x.split("_")[-1].split(".")[0][2:]),
                                     reverse=True)
                all_ckpts = [f"{args.output_dir}/{args.exp_name}/{model_files[0]}"]
        elif args.load_epochs == -2:  # Test all checkpoints
            all_ckpts = [f"{args.output_dir}/{args.exp_name}/model_ep{e}.pth"
                         for e in range(cfg.TRAIN.SAVE_FREQUENCY, cfg.TRAIN.NUM_EPOCHS,
                                        cfg.TRAIN.SAVE_FREQUENCY)]
            all_ckpts.append(f"{args.output_dir}/{args.exp_name}/model_final.pth")
        else:  # Load specific checkpoint
            all_ckpts = [f"{args.output_dir}/{args.exp_name}/model_ep{args.load_epochs}.pth"]

        for ckpt in all_ckpts:
            load_checkpoint(model, None, ckpt)
            evaluate(args, data, cfg, device, model, class_list, class_dict)

    else:  # Training mode
        args.load_epochs = 0 if args.load_epochs < 0 else args.load_epochs
        model = train(args, data, cfg, device, model, class_list, class_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train assembly notation model using ground truth labels with balanced training"
    )

    # Data arguments
    parser.add_argument('-g', '--gt_annotations_root',
                        default="data/MUSCIMA++/v2.0/data/annotations",
                        help="Directory containing ground truth MUSCIMA++ XML annotations")
    parser.add_argument('-i', '--images_root',
                        default="data/MUSCIMA++/datasets_r_staff/images",
                        help="Directory containing staff-removed images")
    parser.add_argument('-s', '--split_file',
                        default="splits/mob_split.yaml",
                        help="YAML file defining train/val/test splits")
    parser.add_argument('-c', '--classes',
                        default='essential',
                        help="Classes to use. Options: ['essn', 'essential', '20', 'all']")

    # Model arguments
    parser.add_argument('--model_config',
                        type=str,
                        help="Path to model configuration YAML file")
    parser.add_argument('--output_dir',
                        type=str,
                        default="outputs",
                        help="Directory to save outputs and checkpoints")
    parser.add_argument('--exp_name',
                        type=str,
                        required=True,
                        help="Experiment name for organizing outputs")

    # Training/evaluation mode
    parser.add_argument('--val_only',
                        action="store_true",
                        help="Only run validation (no training)")
    parser.add_argument('--test_only',
                        action="store_true",
                        help="Only run testing (no training)")
    parser.add_argument('--load_epochs',
                        type=int,
                        default=8,
                        help="Epoch to load checkpoint from (-1: last, -2: all, 0+: specific)")

    # Config overrides
    parser.add_argument('--opts',
                        default=[],
                        nargs=argparse.REMAINDER,
                        help="Options to override config (e.g., TRAIN.NUM_EPOCHS 20)")

    args = parser.parse_args()

    # Validate arguments
    if args.test_only and args.val_only:
        raise ValueError("--test_only and --val_only cannot be used together")

    # Load configuration
    cfg = get_cfg_defaults()
    if args.test_only or args.val_only:
        cfg.merge_from_file(os.path.join(args.output_dir, args.exp_name, "config.yaml"))
    if args.model_config and not args.test_only and not args.val_only:
        cfg.merge_from_file(args.model_config)
    cfg.merge_from_list(args.opts)

    print(f"Experiment name: {args.exp_name}")
    print(f"Using ground truth annotations from: {args.gt_annotations_root}")

    # Get class list and dictionary
    class_list, class_dict = get_classlist_and_classdict(args.classes)

    # Load data configuration
    with open(cfg.DATA.DATA_CONFIG, 'rb') as hdl:
        data_config = yaml.load(hdl, Loader=yaml.FullLoader)
    data_config['mode'] = cfg.MODEL.MODE

    # Load ground truth data
    if args.test_only:
        data = load_ground_truth_data(
            gt_annotations_root=args.gt_annotations_root,
            images_root=args.images_root,
            split_file=args.split_file,
            class_list=class_list,
            class_dict=class_dict,
            config=data_config,
            load_training_data=False,
            load_validation_data=False,
            load_test_data=True,
        )
    elif args.val_only:
        data = load_ground_truth_data(
            gt_annotations_root=args.gt_annotations_root,
            images_root=args.images_root,
            split_file=args.split_file,
            class_list=class_list,
            class_dict=class_dict,
            config=data_config,
            load_training_data=False,
            load_validation_data=True,
            load_test_data=False,
        )
    else:
        data = load_ground_truth_data(
            gt_annotations_root=args.gt_annotations_root,
            images_root=args.images_root,
            split_file=args.split_file,
            class_list=class_list,
            class_dict=class_dict,
            config=data_config,
            load_training_data=True,
            load_validation_data=True,
            load_test_data=False,
        )

    # Setup device
    if cfg.SYSTEM.NUM_GPUS > 0 and torch.cuda.is_available():
        device = "cuda"
    else:
        if cfg.SYSTEM.NUM_GPUS > 0:
            print("[WARNING] GPU requested but not available, using CPU instead")
        device = "cpu"

    print(f"Using device: {device}")

    # Set random seed for reproducibility
    set_seed(cfg.SYSTEM.SEED)

    # Create output directory
    os.makedirs(f"{args.output_dir}/{args.exp_name}", exist_ok=True)

    # Run main training/evaluation
    main(args, data, cfg, device, class_list, class_dict)
