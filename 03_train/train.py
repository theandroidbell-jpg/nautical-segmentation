"""
Training Loop

Main training script with validation, checkpointing, early stopping, and
metrics tracking.  Designed to run on CPU-only servers but can be moved
to a GPU machine simply by passing ``--device cuda``.
"""

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config

# Make sure 03_train itself is importable when run as __main__
sys.path.insert(0, str(Path(__file__).parent))


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ──────────────────────────────────────────────────────────────────────────────

def calculate_iou(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 3,
) -> Dict[int, float]:
    """
    Calculate per-class Intersection over Union (IoU).

    Args:
        predictions: Predicted class indices, shape ``(B, H, W)`` int64.
        targets: Ground-truth class indices, shape ``(B, H, W)`` int64.
        num_classes: Number of classes.

    Returns:
        Dictionary ``{class_idx: iou_score}`` for classes that appear in either
        *predictions* or *targets*.  Classes absent from both are excluded.
    """
    iou_dict: Dict[int, float] = {}
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    for cls in range(num_classes):
        pred_mask = predictions == cls
        true_mask = targets == cls

        intersection = (pred_mask & true_mask).sum().item()
        union = (pred_mask | true_mask).sum().item()

        if union == 0:
            continue  # class absent from both — skip
        iou_dict[cls] = intersection / union

    return iou_dict


# ──────────────────────────────────────────────────────────────────────────────
# Single-epoch helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    num_classes: int = 3,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Train for one epoch.

    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimiser.
        device: Device string.
        epoch: Current epoch index (0-based), used for logging.
        num_classes: Number of segmentation classes.
        max_batches: Cap the number of batches per epoch (for quick smoke tests).

    Returns:
        Dictionary with keys ``loss``, ``pixel_acc``, ``mean_iou``.
    """
    model.train()
    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    iou_accumulator: Dict[int, list] = {c: [] for c in range(num_classes)}

    n_batches = 0
    for batch_idx, (images, masks) in enumerate(train_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

            batch_iou = calculate_iou(preds, masks, num_classes=num_classes)
            for cls, val in batch_iou.items():
                iou_accumulator[cls].append(val)
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0

    per_class_iou = {
        cls: float(np.mean(vals))
        for cls, vals in iou_accumulator.items()
        if vals
    }
    mean_iou = float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0

    return {'loss': avg_loss, 'pixel_acc': pixel_acc, 'mean_iou': mean_iou}


def validate_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    epoch: int,
    num_classes: int = 3,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Validate for one epoch.

    Args:
        model: Model to evaluate (set to eval mode internally).
        val_loader: Validation DataLoader.
        criterion: Loss function.
        device: Device string.
        epoch: Current epoch index (0-based).
        num_classes: Number of segmentation classes.
        max_batches: Cap the number of batches (for quick smoke tests).

    Returns:
        Dictionary with keys ``loss``, ``pixel_acc``, ``mean_iou``.
    """
    model.eval()
    total_loss = 0.0
    correct_pixels = 0
    total_pixels = 0
    iou_accumulator: Dict[int, list] = {c: [] for c in range(num_classes)}

    n_batches = 0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            total_loss += loss.item()
            n_batches += 1

            preds = logits.argmax(dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()

            batch_iou = calculate_iou(preds, masks, num_classes=num_classes)
            for cls, val in batch_iou.items():
                iou_accumulator[cls].append(val)
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    pixel_acc = correct_pixels / total_pixels if total_pixels > 0 else 0.0

    per_class_iou = {
        cls: float(np.mean(vals))
        for cls, vals in iou_accumulator.items()
        if vals
    }
    mean_iou = float(np.mean(list(per_class_iou.values()))) if per_class_iou else 0.0

    return {'loss': avg_loss, 'pixel_acc': pixel_acc, 'mean_iou': mean_iou}


# ──────────────────────────────────────────────────────────────────────────────
# Main training orchestration
# ──────────────────────────────────────────────────────────────────────────────

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: str,
    checkpoint_dir: Path,
    num_classes: int = 3,
    patience: int = 10,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """
    Full training loop with validation, checkpointing, and early stopping.

    Saves a single *best* checkpoint keyed by validation mean IoU.

    Args:
        model: Model to train.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        criterion: Loss function.
        optimizer: Optimiser.
        num_epochs: Maximum number of epochs.
        device: Device string (``'cpu'`` or ``'cuda'``).
        checkpoint_dir: Directory for the best-model checkpoint.
        num_classes: Number of segmentation classes.
        patience: Early-stopping patience in epochs without val mIoU improvement.
        max_batches: Cap batches per epoch (useful for smoke tests on CPU).

    Returns:
        Dictionary with ``best_val_miou`` and ``best_epoch``.
    """
    from model import save_model  # local import to avoid circular dependency

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / 'best_model.pth'

    best_val_miou = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            device=device, epoch=epoch, num_classes=num_classes,
            max_batches=max_batches,
        )
        val_metrics = validate_epoch(
            model, val_loader, criterion,
            device=device, epoch=epoch, num_classes=num_classes,
            max_batches=max_batches,
        )

        logger.info(
            'Epoch %3d/%d | '
            'train loss=%.4f  acc=%.3f  mIoU=%.3f | '
            'val   loss=%.4f  acc=%.3f  mIoU=%.3f',
            epoch + 1, num_epochs,
            train_metrics['loss'], train_metrics['pixel_acc'], train_metrics['mean_iou'],
            val_metrics['loss'], val_metrics['pixel_acc'], val_metrics['mean_iou'],
        )

        val_miou = val_metrics['mean_iou']
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_model(
                model,
                best_checkpoint,
                metadata={
                    'epoch': best_epoch,
                    'best_val_miou': best_val_miou,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics,
                },
            )
            logger.info('  ✓ Saved best checkpoint (val mIoU=%.4f)', best_val_miou)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(
                    'Early stopping: no improvement for %d epochs.', patience
                )
                break

    return {'best_val_miou': best_val_miou, 'best_epoch': best_epoch}


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    """Command-line entry point for training."""
    parser = argparse.ArgumentParser(
        description='Train UNetMobileNetV2 on nautical chart tiles.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--epochs',        type=int,   default=Config.EPOCHS,
                        help='Maximum training epochs.')
    parser.add_argument('--batch-size',    type=int,   default=Config.BATCH_SIZE,
                        help='Mini-batch size.')
    parser.add_argument('--lr',            type=float, default=Config.LEARNING_RATE,
                        help='Initial learning rate (Adam).')
    parser.add_argument('--device',        type=str,   default='cpu',
                        help="Training device: 'cpu' or 'cuda'.")
    parser.add_argument('--checkpoint-dir', type=Path,
                        default=Config.OUTPUT_BASE / 'checkpoints',
                        help='Directory for model checkpoints.')
    parser.add_argument('--tile-dir',      type=Path,  default=Config.OUTPUT_TILES,
                        help='Root tile directory (must contain train/ and val/).')
    parser.add_argument('--num-workers',   type=int,   default=0,
                        help='DataLoader workers (0 = main process, stable on CPU).')
    parser.add_argument('--max-batches',   type=int,   default=None,
                        help='Cap batches per epoch (useful for CPU smoke runs).')
    parser.add_argument('--patience',      type=int,   default=10,
                        help='Early-stopping patience (epochs without val mIoU gain).')
    parser.add_argument('--seed',          type=int,   default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Do not use pretrained MobileNetV2 encoder weights.')
    parser.add_argument('--num-classes',   type=int,   default=Config.NUM_CLASSES,
                        help='Number of segmentation classes.')
    parser.add_argument('--loss',          type=str,   default='combined',
                        choices=['combined', 'dice', 'ce'],
                        help='Loss function type.')

    args = parser.parse_args()

    # ── Reproducibility ────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Pretty-print settings ──────────────────────────────────────────────
    logger.info('=' * 60)
    logger.info('  Nautical Segmentation – Training (Sprint 3)')
    logger.info('=' * 60)
    logger.info('  device        : %s', args.device)
    logger.info('  epochs        : %d', args.epochs)
    logger.info('  batch-size    : %d', args.batch_size)
    logger.info('  lr            : %g', args.lr)
    logger.info('  loss          : %s', args.loss)
    logger.info('  num-classes   : %d', args.num_classes)
    logger.info('  patience      : %d', args.patience)
    logger.info('  max-batches   : %s', args.max_batches)
    logger.info('  num-workers   : %d', args.num_workers)
    logger.info('  pretrained    : %s', not args.no_pretrained)
    logger.info('  tile-dir      : %s', args.tile_dir)
    logger.info('  checkpoint-dir: %s', args.checkpoint_dir)
    logger.info('=' * 60)

    # ── Data ───────────────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent.parent / '02_prepare'))
    from dataloader import get_dataloaders

    train_loader, val_loader = get_dataloaders(
        tile_base=args.tile_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    logger.info('Train tiles: %d  |  Val tiles: %d',
                len(train_loader.dataset), len(val_loader.dataset))

    # ── Model ─────────────────────────────────────────────────────────────
    from model import create_model
    model = create_model(
        num_classes=args.num_classes,
        pretrained=not args.no_pretrained,
        device=args.device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Model params: %d total, %d trainable', total_params, trainable_params)

    # ── Loss & optimiser ──────────────────────────────────────────────────
    from losses import get_loss_function
    criterion = get_loss_function(loss_type=args.loss, num_classes=args.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── Train ─────────────────────────────────────────────────────────────
    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
        num_classes=args.num_classes,
        patience=args.patience,
        max_batches=args.max_batches,
    )

    logger.info('Training complete.  Best val mIoU=%.4f at epoch %d.',
                result['best_val_miou'], result['best_epoch'])


if __name__ == '__main__':
    main()
