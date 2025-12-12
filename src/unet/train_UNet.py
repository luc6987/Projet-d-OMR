#!/usr/bin/env python3
"""
Train a U-Net style segmentation model to remove staff lines from sheet music.

Dataset assumptions (based on user description):
    data_root/
        ├── group_1/
        │   ├── image/   (original images with staff lines)
        │   ├── gt/      (pixels to remove, i.e. staff lines)
        │   └── symbol/  (symbols-only renderings after staff removal)
        ├── group_2/
        │   └── ...

The training target has three classes:
    0 - background
    1 - staff lines (pixels found in gt)
    2 - musical symbols (pixels found in symbol images)

If the actual dataset deviates from these assumptions, adjust `build_mask_from_sources`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# Use new torch.amp API if available, fallback to torch.cuda.amp for older versions
try:
    from torch.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = False
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


# --------------------------- Utility helpers --------------------------- #


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_matching_file(directory: Path, stem: str, exts: Sequence[str]) -> Path | None:
    """
    Find a matching file in the directory, trying multiple naming patterns.
    
    Tries in order:
    1. Exact match: {stem}{ext}
    2. Common suffixes: {stem}_gt{ext}, {stem}_mask{ext}, {stem}_staff{ext}
    
    Args:
        directory: Directory to search in
        stem: Base filename without extension
        exts: Sequence of file extensions to try (e.g., ['.png', '.jpg'])
    
    Returns:
        Path to matching file if found, None otherwise
    """
    # Common suffixes to try
    suffixes = ["", "_gt", "_mask", "_staff"]
    
    for suffix in suffixes:
        for ext in exts:
            candidate = directory / f"{stem}{suffix}{ext}"
            if candidate.exists():
                return candidate
    return None


# --------------------------- Dataset ----------------------------------- #


def collect_samples(
    data_root: Path,
    image_subdir: str,
    staff_subdir: str,
    symbol_subdir: str,
    extensions: Sequence[str],
) -> List[Tuple[Path, Path, Path]]:
    samples: List[Tuple[Path, Path, Path]] = []

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {data_root}")

    group_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()])
    if not group_dirs:
        raise RuntimeError(f"No group folders found under {data_root}")

    for group in group_dirs:
        img_dir = group / image_subdir
        staff_dir = group / staff_subdir
        symbol_dir = group / symbol_subdir

        if not (img_dir.exists() and staff_dir.exists() and symbol_dir.exists()):
            print(f"[WARN] Skipping {group}: missing one of required subdirs "
                  f"({image_subdir}, {staff_subdir}, {symbol_subdir})")
            continue

        image_files = sorted(
            [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions]
        )

        for img_path in image_files:
            stem = img_path.stem
            staff_path = find_matching_file(staff_dir, stem, extensions)
            symbol_path = find_matching_file(symbol_dir, stem, extensions)

            if staff_path is None or symbol_path is None:
                print(f"[WARN] Missing corresponding files for {img_path}")
                continue

            samples.append((img_path, staff_path, symbol_path))

    if not samples:
        raise RuntimeError(f"No valid triplets found under {data_root}")

    return samples


from PIL import Image


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.array(img.convert("L"))


def binarize(img: np.ndarray, threshold: int) -> np.ndarray:
    """
    Binarize image: bright pixels (>threshold) become 1, dark pixels become 0.
    For black background with white content (staff/symbols), white parts are marked as 1.
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return (img > threshold).astype(np.uint8)


def build_mask_from_sources(
    staff_img: np.ndarray,
    symbol_img: np.ndarray,
    staff_threshold: int = 128,
    symbol_threshold: int = 128,
) -> np.ndarray:
    if staff_img.shape != symbol_img.shape:
        raise ValueError(
            f"Staff mask and symbol image must share shape, got {staff_img.shape} vs {symbol_img.shape}"
        )

    mask = np.zeros_like(staff_img, dtype=np.uint8)
    staff_mask = binarize(staff_img, staff_threshold)
    symbol_mask = binarize(symbol_img, symbol_threshold)

    mask[staff_mask == 1] = 1
    mask[symbol_mask == 1] = 2
    return mask


class StaffLineSegmentationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, Path, Path]],
        img_size: Tuple[int, int] | None = None,
        augment: bool = False,
        staff_threshold: int = 128,
        symbol_threshold: int = 128,
        max_rotation_deg: float = 2.5,
    ) -> None:
        self.samples = list(samples)
        self.img_size = img_size
        self.augment = augment
        self.staff_threshold = staff_threshold
        self.symbol_threshold = symbol_threshold
        self.max_rotation_deg = max_rotation_deg

        if not self.samples:
            raise ValueError("Dataset received an empty sample list.")

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_crop(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RandomCrop to preserve fine details (e.g., 1-3 pixel staff lines).
        If image is smaller than crop size, pad first then crop.
        """
        if self.img_size is None:
            return image, mask

        crop_h, crop_w = self.img_size
        _, img_h, img_w = image.shape

        # Handle images smaller than crop size: pad to crop size
        if img_h < crop_h or img_w < crop_w:
            # Pad to at least crop size
            pad_h = max(0, crop_h - img_h)
            pad_w = max(0, crop_w - img_w)
            # Pad symmetrically
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            # Pad image with background value (0.0 for black background)
            image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0.0)
            # Pad mask with background class (0)
            mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
            
            # Update dimensions after padding
            _, img_h, img_w = image.shape

        # Random crop coordinates
        if img_h > crop_h:
            top = random.randint(0, img_h - crop_h)
        else:
            top = 0
        
        if img_w > crop_w:
            left = random.randint(0, img_w - crop_w)
        else:
            left = 0

        # Apply crop
        image = TF.crop(image, top, left, crop_h, crop_w)
        mask = TF.crop(mask, top, left, crop_h, crop_w)

        return image, mask

    def _apply_augmentations(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.augment:
            return image, mask

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask.unsqueeze(0)).squeeze(0)

        if random.random() < 0.1:
            image = TF.vflip(image)
            mask = TF.vflip(mask.unsqueeze(0)).squeeze(0)

        if self.max_rotation_deg > 0:
            angle = random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
            image = TF.rotate(
                image,
                angle,
                interpolation=InterpolationMode.BILINEAR,
                fill=1.0,
            )
            mask = TF.rotate(
                mask.unsqueeze(0).float(),
                angle,
                interpolation=InterpolationMode.NEAREST,
                fill=0.0,
            ).squeeze(0).long()

        return image, mask

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, staff_path, symbol_path = self.samples[index]

        image = load_grayscale(image_path)
        staff_img = load_grayscale(staff_path)
        symbol_img = load_grayscale(symbol_path)

        if image.shape != staff_img.shape or image.shape != symbol_img.shape:
            raise ValueError(
                f"Shape mismatch among triplet {image_path}, {staff_path}, {symbol_path}"
            )

        mask_np = build_mask_from_sources(
            staff_img=staff_img,
            symbol_img=symbol_img,
            staff_threshold=self.staff_threshold,
            symbol_threshold=self.symbol_threshold,
        )

        image_tensor = torch.from_numpy(image).unsqueeze(0).float() / 255.0
        mask_tensor = torch.from_numpy(mask_np).long()

        image_tensor, mask_tensor = self._apply_crop(image_tensor, mask_tensor)
        image_tensor, mask_tensor = self._apply_augmentations(image_tensor, mask_tensor)

        return image_tensor, mask_tensor


# --------------------------- Model ------------------------------------- #


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None, dropout: float = 0.0):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.insert(3, nn.Dropout2d(dropout))
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool, dropout: float = 0.0):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        base_channels: int = 64,
        bilinear: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, base_channels, dropout=dropout)
        self.down1 = Down(base_channels, base_channels * 2, dropout=dropout)
        self.down2 = Down(base_channels * 2, base_channels * 4, dropout=dropout)
        self.down3 = Down(base_channels * 4, base_channels * 8, dropout=dropout)
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor, dropout=dropout)
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear, dropout=dropout)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear, dropout=dropout)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear, dropout=dropout)
        self.up4 = Up(base_channels * 2, base_channels, bilinear, dropout=dropout)
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# --------------------------- Metrics ----------------------------------- #


def compute_batch_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Tuple[int, int, torch.Tensor, torch.Tensor]:
    preds = torch.argmax(logits, dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()

    intersection = torch.zeros(num_classes, dtype=torch.float64, device=targets.device)
    union = torch.zeros(num_classes, dtype=torch.float64, device=targets.device)

    for cls in range(num_classes):
        pred_c = preds == cls
        target_c = targets == cls
        intersection[cls] = torch.logical_and(pred_c, target_c).sum(dtype=torch.float64)
        union[cls] = torch.logical_or(pred_c, target_c).sum(dtype=torch.float64)

    return correct, total, intersection.cpu(), union.cpu()


@dataclass
class EpochMetrics:
    loss: float
    pixel_accuracy: float
    mean_iou: float
    iou_per_class: List[float]


@dataclass
class HistoryEntry:
    epoch: int
    train_loss: float
    train_pixel_acc: float
    train_mean_iou: float
    train_iou_per_class: List[float]
    val_loss: float
    val_pixel_acc: float
    val_mean_iou: float
    val_iou_per_class: List[float]
    learning_rate: float


def compute_mean_iou(intersection: torch.Tensor, union: torch.Tensor) -> Tuple[float, List[float]]:
    per_class = []
    for i in range(len(intersection)):
        if union[i] == 0:
            per_class.append(float("nan"))
        else:
            per_class.append((intersection[i] / union[i]).item())
    mean_iou = float(np.nanmean(per_class))
    return mean_iou, per_class


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    num_classes: int,
    amp_enabled: bool,
    grad_clip: float,
) -> EpochMetrics:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    intersection = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if AMP_AVAILABLE:
            with autocast('cuda', enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)
        else:
            with autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, masks)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct, pixels, inter, uni = compute_batch_metrics(logits.detach(), masks, num_classes)
        total_correct += correct
        total_pixels += pixels
        intersection += inter
        union += uni

    avg_loss = total_loss / max(1, len(loader.dataset))
    pixel_acc = total_correct / max(1, total_pixels)
    mean_iou, per_class_iou = compute_mean_iou(intersection, union)

    return EpochMetrics(avg_loss, pixel_acc, mean_iou, per_class_iou)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    intersection = torch.zeros(num_classes, dtype=torch.float64)
    union = torch.zeros(num_classes, dtype=torch.float64)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, masks)

        total_loss += loss.item() * images.size(0)
        correct, pixels, inter, uni = compute_batch_metrics(logits, masks, num_classes)
        total_correct += correct
        total_pixels += pixels
        intersection += inter
        union += uni

    avg_loss = total_loss / max(1, len(loader.dataset))
    pixel_acc = total_correct / max(1, total_pixels)
    mean_iou, per_class_iou = compute_mean_iou(intersection, union)

    return EpochMetrics(avg_loss, pixel_acc, mean_iou, per_class_iou)


def split_samples(
    samples: Sequence[Tuple[Path, Path, Path]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, Path, Path]], List[Tuple[Path, Path, Path]]]:
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1).")

    samples_list = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples_list)

    if len(samples_list) < 2:
        raise ValueError("Need at least 2 samples to create train/val splits.")

    val_count = int(math.floor(len(samples_list) * val_ratio))
    if val_count == 0:
        val_count = max(1, len(samples_list) // 10)

    if val_count >= len(samples_list):
        val_count = len(samples_list) - 1

    val_samples = samples_list[:val_count]
    train_samples = samples_list[val_count:]
    return train_samples, val_samples


def get_learning_rate(optimizer: optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def create_optimizer(model: nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    if args.optimizer == "adam":
        return optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
    if args.optimizer == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
    if args.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def create_scheduler(optimizer: optim.Optimizer, args: argparse.Namespace):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    if args.lr_scheduler == "plateau":
        mode = "max" if args.scheduler_metric == "iou" else "min"
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=args.lr_gamma,
            patience=args.lr_patience,
            min_lr=args.min_lr,
            verbose=True,
        )
    raise ValueError(f"Unsupported scheduler: {args.lr_scheduler}")


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parent.parent.parent / "data" / "v1.0" / "data" / "images"
    parser = argparse.ArgumentParser(description="Train a U-Net to remove staff lines from sheet music.")
    parser.add_argument("--data-root", type=Path, default=default_root, help="Root directory containing grouped folders with image/gt/symbol subdirectories.")
    parser.add_argument("--image-subdir", type=str, default="image", help="Subdirectory name that stores original images.")
    parser.add_argument("--staff-subdir", type=str, default="gt", help="Subdirectory name that stores staff-line masks.")
    parser.add_argument("--symbol-subdir", type=str, default="symbol", help="Subdirectory name that stores symbol-only renderings.")
    parser.add_argument("--extensions", type=str, default=".png,.jpg,.jpeg", help="Comma-separated list of valid file extensions.")
    parser.add_argument("--img-size", type=int, default=512, help="Crop size for RandomCrop (preserves fine details). Set together with --no-resize to keep original resolution.")
    parser.add_argument("--no-resize", action="store_true", help="Disable cropping and keep original resolution.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio.")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adam", "adamw", "sgd"], help="Optimizer choice.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (L2 regularization).")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for Adam/AdamW.")
    parser.add_argument("--beta2", type=float, default=0.999, help="Beta2 for Adam/AdamW.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD.")
    parser.add_argument("--nesterov", action="store_true", help="Use Nesterov momentum with SGD.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", help="Device string, e.g., cuda, cuda:0, cpu. Use 'auto' to select automatically.")
    parser.add_argument("--lr-scheduler", type=str, default="plateau", choices=["none", "cosine", "plateau"], help="Learning rate scheduler.")
    parser.add_argument("--scheduler-metric", type=str, default="iou", choices=["iou", "loss"], help="Metric for ReduceLROnPlateau.")
    parser.add_argument("--lr-gamma", type=float, default=0.5, help="Scheduler decay factor.")
    parser.add_argument("--lr-patience", type=int, default=5, help="Scheduler patience (epochs without improvement).")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate.")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Gradient clipping value (0 to disable).")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision training.")
    parser.add_argument("--num-classes", type=int, default=3, help="Number of segmentation classes.")
    parser.add_argument("--base-channels", type=int, default=64, help="Base number of channels for the U-Net.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate inside U-Net blocks.")
    parser.add_argument("--use-transposed-conv", action="store_true", help="Use transposed convolution instead of bilinear upsampling.")
    parser.add_argument("--class-weights", type=str, default="", help="Optional comma-separated class weights for CrossEntropyLoss.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Directory to save checkpoints.")
    parser.add_argument("--save-name", type=str, default="unet_staff_removal.pt", help="Checkpoint filename.")
    parser.add_argument("--log-json", type=Path, default=None, help="Path to write training history JSON. If not specified, auto-generates: logs/training_YYYYMMDD_HHMMSS.json")
    parser.add_argument("--log-dir", type=Path, default=None, help="Directory for auto-generated log files (default: project_root/logs).")
    parser.add_argument("--selection-metric", type=str, default="iou", choices=["iou", "loss"], help="Metric for selecting the best checkpoint.")
    parser.add_argument("--print-every", type=int, default=1, help="Print metrics every N epochs.")
    parser.add_argument("--max-rotation", type=float, default=2.5, help="Maximum rotation (degrees) for augmentation.")
    parser.add_argument("--no-augment", action="store_true", help="Disable geometric augmentations.")
    parser.add_argument("--staff-threshold", type=int, default=128, help="Threshold for binarising staff masks.")
    parser.add_argument("--symbol-threshold", type=int, default=128, help="Threshold for binarising symbol masks.")
    parser.add_argument("--early-stop", type=int, default=0, help="Early stopping patience (epochs). 0 disables early stopping.")
    parser.add_argument("--resume", type=Path, default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.save_dir is None:
        args.save_dir = Path(__file__).resolve().parent.parent / "model" / "checkpoints"

    if args.print_every <= 0:
        raise ValueError("--print-every must be a positive integer.")

    args.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Auto-generate log file path if not specified
    if args.log_json is None:
        if args.log_dir is None:
            args.log_dir = Path(__file__).resolve().parent.parent / "logs"
        args.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_json = args.log_dir / f"training_{timestamp}.json"

    set_seed(args.seed)

    extension_list: List[str] = []
    for ext in args.extensions.split(","):
        ext_clean = ext.strip().lower()
        if not ext_clean:
            continue
        if not ext_clean.startswith("."):
            ext_clean = f".{ext_clean}"
        extension_list.append(ext_clean)

    if not extension_list:
        raise ValueError("No valid file extensions provided.")

    extensions = tuple(sorted(set(extension_list)))

    samples = collect_samples(
        data_root=args.data_root,
        image_subdir=args.image_subdir,
        staff_subdir=args.staff_subdir,
        symbol_subdir=args.symbol_subdir,
        extensions=extensions,
    )

    train_samples, val_samples = split_samples(samples, args.val_ratio, args.seed)
    if not train_samples:
        raise RuntimeError("Training sample list is empty after split.")
    if not val_samples:
        raise RuntimeError("Validation sample list is empty after split.")

    img_size = None if args.no_resize else (args.img_size, args.img_size)

    train_dataset = StaffLineSegmentationDataset(
        train_samples,
        img_size=img_size,
        augment=not args.no_augment,
        staff_threshold=args.staff_threshold,
        symbol_threshold=args.symbol_threshold,
        max_rotation_deg=args.max_rotation,
    )
    val_dataset = StaffLineSegmentationDataset(
        val_samples,
        img_size=img_size,
        augment=False,
        staff_threshold=args.staff_threshold,
        symbol_threshold=args.symbol_threshold,
        max_rotation_deg=0.0,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    pin_memory = device.type == "cuda"
    amp_enabled = args.amp and device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = UNet(
        n_channels=1,
        n_classes=args.num_classes,
        base_channels=args.base_channels,
        bilinear=not args.use_transposed_conv,
        dropout=args.dropout,
    ).to(device)

    if args.class_weights:
        weights = [float(w.strip()) for w in args.class_weights.split(",") if w.strip()]
        if len(weights) != args.num_classes:
            raise ValueError("Number of class weights must match num_classes.")
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = create_optimizer(model, args)
    scheduler = create_scheduler(optimizer, args)
    if AMP_AVAILABLE:
        scaler = GradScaler('cuda', enabled=amp_enabled)
    else:
        scaler = GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_score = -float("inf") if args.selection_metric == "iou" else float("inf")
    epochs_without_improvement = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        if "optimizer_state" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            except ValueError as exc:
                print(f"[WARN] Failed to load optimizer state: {exc}")
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"] + 1
        if "best_score" in checkpoint:
            best_score = checkpoint["best_score"]
        print(f"[INFO] Resumed from {args.resume} at epoch {start_epoch}")

    history: List[HistoryEntry] = []
    checkpoint_path = args.save_dir / args.save_name

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            num_classes=args.num_classes,
            amp_enabled=amp_enabled,
            grad_clip=args.grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=args.num_classes,
        )

        current_lr = get_learning_rate(optimizer)
        history.append(
            HistoryEntry(
                epoch=epoch,
                train_loss=train_metrics.loss,
                train_pixel_acc=train_metrics.pixel_accuracy,
                train_mean_iou=train_metrics.mean_iou,
                train_iou_per_class=train_metrics.iou_per_class,
                val_loss=val_metrics.loss,
                val_pixel_acc=val_metrics.pixel_accuracy,
                val_mean_iou=val_metrics.mean_iou,
                val_iou_per_class=val_metrics.iou_per_class,
                learning_rate=current_lr,
            )
        )

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                monitor_value = (
                    val_metrics.mean_iou if args.scheduler_metric == "iou" else val_metrics.loss
                )
                scheduler.step(monitor_value)
            else:
                scheduler.step()

        if (epoch - start_epoch) % args.print_every == 0 or epoch == args.epochs:
            iou_strings = [
                f"class{idx}:{iou:.4f}" if not math.isnan(iou) else f"class{idx}:nan"
                for idx, iou in enumerate(val_metrics.iou_per_class)
            ]
            print(
                f"Epoch {epoch:03d}/{args.epochs:03d} | "
                f"train_loss={train_metrics.loss:.4f} | "
                f"val_loss={val_metrics.loss:.4f} | "
                f"train_acc={train_metrics.pixel_accuracy:.4f} | "
                f"val_acc={val_metrics.pixel_accuracy:.4f} | "
                f"val_mIoU={val_metrics.mean_iou:.4f} | "
                f"lr={current_lr:.6f}"
            )
            print("  IoU per class: " + ", ".join(iou_strings))

        monitor_metric = val_metrics.mean_iou if args.selection_metric == "iou" else val_metrics.loss
        improved = (
            monitor_metric > best_score if args.selection_metric == "iou" else monitor_metric < best_score
        )

        if improved:
            best_score = monitor_metric
            epochs_without_improvement = 0
            # Add marker to indicate this model was trained with fixed binarize logic
            checkpoint_args = vars(args).copy()
            checkpoint_args["binarize_fixed"] = True  # Marker for fixed binarize logic
            
            # Save checkpoint with atomic write (temp file + rename) and retry logic
            checkpoint_data = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "args": checkpoint_args,
                "best_score": best_score,
                "train_metrics": train_metrics.__dict__,
                "val_metrics": val_metrics.__dict__,
            }
            
            # Use atomic write: save to temp file first, then rename
            temp_path = checkpoint_path.with_suffix('.tmp')
            max_retries = 3
            retry_delay = 1.0
            
            for attempt in range(max_retries):
                try:
                    # Save to temporary file first
                    torch.save(checkpoint_data, temp_path)
                    # Atomic rename (works on most filesystems)
                    temp_path.replace(checkpoint_path)
                    print(f"[INFO] Saved improved checkpoint to {checkpoint_path}")
                    break
                except (RuntimeError, OSError, IOError) as e:
                    if attempt < max_retries - 1:
                        print(f"[WARN] Checkpoint save failed (attempt {attempt + 1}/{max_retries}): {e}")
                        print(f"[WARN] Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        # Clean up failed temp file if it exists
                        if temp_path.exists():
                            try:
                                temp_path.unlink()
                            except:
                                pass
                    else:
                        print(f"[ERROR] Failed to save checkpoint after {max_retries} attempts: {e}")
                        print(f"[ERROR] Training will continue, but checkpoint was not saved.")
                        # Try to clean up temp file
                        if temp_path.exists():
                            try:
                                temp_path.unlink()
                            except:
                                pass
        else:
            epochs_without_improvement += 1

        if args.early_stop > 0 and epochs_without_improvement >= args.early_stop:
            print(f"[INFO] Early stopping triggered after {epoch} epochs.")
            break

    # Save training log (always save if auto-generated, or if explicitly specified)
    if args.log_json:
        # Find best epoch
        best_epoch_idx = None
        if history:
            if args.selection_metric == "iou":
                best_epoch_idx = max(range(len(history)), key=lambda i: history[i].val_mean_iou)
            else:
                best_epoch_idx = min(range(len(history)), key=lambda i: history[i].val_loss)
        
        log_data = {
            "training_config": {
                "data_root": str(args.data_root),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "img_size": args.img_size,
                "base_channels": args.base_channels,
                "learning_rate": args.lr,
                "optimizer": args.optimizer,
                "lr_scheduler": args.lr_scheduler,
                "val_ratio": args.val_ratio,
                "num_classes": args.num_classes,
                "device": str(device),
                "amp_enabled": amp_enabled,
            },
            "history": [asdict(entry) for entry in history],
            "final_results": {
                "best_epoch": best_epoch_idx + 1 if best_epoch_idx is not None else None,
                "best_val_iou": best_score if args.selection_metric == "iou" else None,
                "best_val_loss": best_score if args.selection_metric == "loss" else None,
                "final_train_loss": history[-1].train_loss if history else None,
                "final_val_loss": history[-1].val_loss if history else None,
                "final_train_iou": history[-1].train_mean_iou if history else None,
                "final_val_iou": history[-1].val_mean_iou if history else None,
            },
            "checkpoint": str(checkpoint_path),
            "selection_metric": args.selection_metric,
            "timestamp": datetime.now().isoformat(),
        }
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_json, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)
        print(f"[INFO] Training log saved to {args.log_json}")

    print("[DONE] Training complete.")


if __name__ == "__main__":
    main()

