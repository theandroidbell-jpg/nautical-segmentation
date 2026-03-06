"""
Tests for Sprint 3 components.

Tests cover (all run quickly on CPU):
- UNetMobileNetV2 forward-pass output shape
- DiceLoss, CombinedLoss, get_loss_function produce finite scalar outputs
- SegmentationAugmentation preserves shapes and mask dtype
- calculate_iou returns per-class IoU in [0, 1]
- A minimal end-to-end training step (synthetic data, no disk I/O)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure repo root and 03_train are importable
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / '03_train'))

from model import UNetMobileNetV2, create_model, save_model, load_model
from losses import DiceLoss, CombinedLoss, get_loss_function
from augment import (
    SegmentationAugmentation,
    random_horizontal_flip,
    random_rotation,
    adjust_brightness_contrast,
)
from train import calculate_iou, train_epoch, validate_epoch, train_model


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 3
BATCH = 2
H, W = 256, 256


def _random_batch(batch_size: int = BATCH, num_classes: int = NUM_CLASSES):
    """Return a random (images, masks) batch as CPU tensors."""
    images = torch.rand(batch_size, 3, H, W)
    masks = torch.randint(0, num_classes, (batch_size, H, W))
    return images, masks


def _tiny_loader(n_samples: int = 4, batch_size: int = 2, num_classes: int = NUM_CLASSES):
    """Create a DataLoader backed by synthetic tensors."""
    images = torch.rand(n_samples, 3, H, W)
    masks = torch.randint(0, num_classes, (n_samples, H, W))
    dataset = TensorDataset(images, masks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestUNetMobileNetV2:
    """Tests for the U-Net + MobileNetV2 model."""

    def test_forward_pass_shape(self):
        """Output logits have shape (B, num_classes, H, W)."""
        model = create_model(num_classes=NUM_CLASSES, pretrained=False, device='cpu')
        images, _ = _random_batch()
        with torch.no_grad():
            logits = model(images)
        assert logits.shape == (BATCH, NUM_CLASSES, H, W), (
            f"Expected {(BATCH, NUM_CLASSES, H, W)}, got {tuple(logits.shape)}"
        )

    def test_output_dtype(self):
        """Output is float32."""
        model = create_model(num_classes=NUM_CLASSES, pretrained=False, device='cpu')
        images, _ = _random_batch()
        with torch.no_grad():
            logits = model(images)
        assert logits.dtype == torch.float32

    def test_different_num_classes(self):
        """Model works for num_classes != 3."""
        model = create_model(num_classes=5, pretrained=False, device='cpu')
        images, _ = _random_batch()
        with torch.no_grad():
            logits = model(images)
        assert logits.shape[1] == 5

    def test_save_and_load(self, tmp_path: Path):
        """save_model / load_model round-trips produce identical outputs."""
        model = create_model(num_classes=NUM_CLASSES, pretrained=False, device='cpu')
        ckpt = tmp_path / 'ckpt.pth'
        save_model(model, ckpt, metadata={'epoch': 1, 'best_val_miou': 0.5})

        loaded = load_model(ckpt, num_classes=NUM_CLASSES, device='cpu')

        # Put both models in eval mode so BatchNorm behaves identically
        model.eval()
        images, _ = _random_batch(batch_size=1)
        with torch.no_grad():
            out_orig = model(images)
            out_loaded = loaded(images)

        assert torch.allclose(out_orig, out_loaded, atol=1e-5)

    def test_save_creates_parent_dirs(self, tmp_path: Path):
        """save_model creates missing parent directories."""
        model = create_model(num_classes=NUM_CLASSES, pretrained=False, device='cpu')
        deep_path = tmp_path / 'a' / 'b' / 'c' / 'model.pth'
        save_model(model, deep_path)
        assert deep_path.exists()


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestLossFunctions:
    """Tests for DiceLoss, CombinedLoss, and get_loss_function."""

    def test_dice_loss_finite(self):
        """DiceLoss returns a finite scalar."""
        images, masks = _random_batch()
        logits = torch.randn(BATCH, NUM_CLASSES, H, W)
        loss_fn = DiceLoss(num_classes=NUM_CLASSES)
        loss = loss_fn(logits, masks)
        assert loss.ndim == 0, 'Loss must be scalar'
        assert torch.isfinite(loss), f'Loss is not finite: {loss.item()}'

    def test_dice_loss_range(self):
        """DiceLoss is in [0, 1] for random inputs."""
        logits = torch.randn(BATCH, NUM_CLASSES, H, W)
        _, masks = _random_batch()
        loss_fn = DiceLoss(num_classes=NUM_CLASSES)
        loss = loss_fn(logits, masks).item()
        assert 0.0 <= loss <= 1.0 + 1e-6, f'Unexpected Dice loss: {loss}'

    def test_combined_loss_finite(self):
        """CombinedLoss returns a finite scalar."""
        logits = torch.randn(BATCH, NUM_CLASSES, H, W)
        _, masks = _random_batch()
        loss_fn = CombinedLoss(num_classes=NUM_CLASSES)
        loss = loss_fn(logits, masks)
        assert torch.isfinite(loss)

    def test_get_loss_function_combined(self):
        """get_loss_function('combined') returns CombinedLoss."""
        fn = get_loss_function('combined', num_classes=NUM_CLASSES)
        assert isinstance(fn, CombinedLoss)

    def test_get_loss_function_dice(self):
        """get_loss_function('dice') returns DiceLoss."""
        fn = get_loss_function('dice', num_classes=NUM_CLASSES)
        assert isinstance(fn, DiceLoss)

    def test_get_loss_function_ce(self):
        """get_loss_function('ce') returns CrossEntropyLoss."""
        fn = get_loss_function('ce', num_classes=NUM_CLASSES)
        assert isinstance(fn, nn.CrossEntropyLoss)

    def test_get_loss_function_invalid(self):
        """get_loss_function raises ValueError for unknown type."""
        with pytest.raises(ValueError):
            get_loss_function('unknown')

    def test_ce_loss_finite(self):
        """CrossEntropy loss from factory is finite."""
        fn = get_loss_function('ce', num_classes=NUM_CLASSES)
        logits = torch.randn(BATCH, NUM_CLASSES, H, W)
        _, masks = _random_batch()
        loss = fn(logits, masks)
        assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# Augmentation tests
# ---------------------------------------------------------------------------

class TestSegmentationAugmentation:
    """Tests for SegmentationAugmentation and helper functions."""

    def _make_tensors(self):
        image = torch.rand(3, H, W)
        mask = torch.randint(0, NUM_CLASSES, (H, W))
        return image, mask

    def test_shapes_preserved(self):
        """Augmented image and mask have the same shape as input."""
        aug = SegmentationAugmentation()
        image, mask = self._make_tensors()
        aug_image, aug_mask = aug(image, mask)
        assert aug_image.shape == image.shape
        assert aug_mask.shape == mask.shape

    def test_mask_dtype_preserved(self):
        """Augmentation does not change the mask dtype."""
        aug = SegmentationAugmentation()
        _, mask = self._make_tensors()
        _, aug_mask = aug(*self._make_tensors())
        assert aug_mask.dtype == torch.int64

    def test_image_values_in_range(self):
        """Image values stay within [0, 1] after photometric augmentation."""
        aug = SegmentationAugmentation(
            brightness_range=(0.5, 1.5),
            contrast_range=(0.8, 1.2),
        )
        image, mask = self._make_tensors()
        aug_image, _ = aug(image, mask)
        assert float(aug_image.min()) >= 0.0 - 1e-6
        assert float(aug_image.max()) <= 1.0 + 1e-6

    def test_no_augmentation_option(self):
        """Disabling all augmentations leaves tensors unchanged."""
        aug = SegmentationAugmentation(
            horizontal_flip=False,
            vertical_flip=False,
            rotation_degrees=0,
        )
        image, mask = self._make_tensors()
        aug_image, aug_mask = aug(image, mask)
        assert torch.equal(aug_image, image)
        assert torch.equal(aug_mask, mask)

    def test_numpy_random_horizontal_flip(self):
        """random_horizontal_flip preserves shape."""
        img = np.random.rand(3, H, W).astype(np.float32)
        msk = np.random.randint(0, NUM_CLASSES, (H, W)).astype(np.uint8)
        img2, msk2 = random_horizontal_flip(img, msk, p=1.0)
        assert img2.shape == img.shape
        assert msk2.shape == msk.shape

    def test_numpy_random_rotation(self):
        """random_rotation preserves shape (square input)."""
        img = np.random.rand(3, H, W).astype(np.float32)
        msk = np.random.randint(0, NUM_CLASSES, (H, W)).astype(np.uint8)
        img2, msk2 = random_rotation(img, msk, max_angle=90)
        assert img2.shape == img.shape
        assert msk2.shape == msk.shape

    def test_numpy_adjust_brightness_contrast(self):
        """adjust_brightness_contrast keeps output in [0, 1]."""
        img = np.random.rand(3, H, W).astype(np.float32)
        out = adjust_brightness_contrast(img)
        assert out.min() >= 0.0 - 1e-6
        assert out.max() <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------

class TestCalculateIoU:
    """Tests for the calculate_iou metric helper."""

    def test_perfect_prediction(self):
        """IoU is 1.0 when predictions match targets exactly."""
        targets = torch.randint(0, NUM_CLASSES, (BATCH, H, W))
        iou = calculate_iou(targets, targets, num_classes=NUM_CLASSES)
        for cls, score in iou.items():
            assert abs(score - 1.0) < 1e-6, f'Class {cls}: expected 1.0, got {score}'

    def test_iou_in_range(self):
        """IoU values are in [0, 1]."""
        preds = torch.randint(0, NUM_CLASSES, (BATCH, H, W))
        targets = torch.randint(0, NUM_CLASSES, (BATCH, H, W))
        iou = calculate_iou(preds, targets, num_classes=NUM_CLASSES)
        for cls, score in iou.items():
            assert 0.0 <= score <= 1.0, f'Class {cls}: IoU out of range: {score}'

    def test_returns_dict(self):
        """calculate_iou returns a dict."""
        preds = torch.zeros(BATCH, H, W, dtype=torch.long)
        targets = torch.zeros(BATCH, H, W, dtype=torch.long)
        result = calculate_iou(preds, targets, num_classes=NUM_CLASSES)
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# End-to-end training step
# ---------------------------------------------------------------------------

class TestEndToEndTraining:
    """Smoke tests for the training and validation loops."""

    def test_train_epoch_metrics(self):
        """train_epoch runs without error and returns expected metric keys."""
        model = create_model(num_classes=NUM_CLASSES, pretrained=False)
        loader = _tiny_loader()
        criterion = get_loss_function('combined', num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        metrics = train_epoch(
            model, loader, criterion, optimizer,
            device='cpu', epoch=0, num_classes=NUM_CLASSES,
            max_batches=2,
        )
        assert 'loss' in metrics
        assert 'pixel_acc' in metrics
        assert 'mean_iou' in metrics
        assert np.isfinite(metrics['loss'])
        assert 0.0 <= metrics['pixel_acc'] <= 1.0

    def test_validate_epoch_metrics(self):
        """validate_epoch runs without error and returns expected metric keys."""
        model = create_model(num_classes=NUM_CLASSES, pretrained=False)
        loader = _tiny_loader()
        criterion = get_loss_function('combined', num_classes=NUM_CLASSES)

        metrics = validate_epoch(
            model, loader, criterion,
            device='cpu', epoch=0, num_classes=NUM_CLASSES,
            max_batches=2,
        )
        assert 'loss' in metrics
        assert np.isfinite(metrics['loss'])

    def test_train_model_saves_checkpoint(self, tmp_path: Path):
        """train_model saves a best-model checkpoint after training."""
        model = create_model(num_classes=NUM_CLASSES, pretrained=False)
        train_loader = _tiny_loader()
        val_loader = _tiny_loader()
        criterion = get_loss_function('combined', num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=2,
            device='cpu',
            checkpoint_dir=tmp_path / 'ckpts',
            num_classes=NUM_CLASSES,
            patience=10,
            max_batches=2,
        )

        assert (tmp_path / 'ckpts' / 'best_model.pth').exists()
        assert 'best_val_miou' in result
        assert 'best_epoch' in result

    def test_train_model_early_stopping(self, tmp_path: Path):
        """train_model terminates early when patience is exceeded."""
        model = create_model(num_classes=NUM_CLASSES, pretrained=False)

        # Deterministic loader — mIoU should not improve after first epoch
        train_loader = _tiny_loader()
        val_loader = _tiny_loader()
        criterion = get_loss_function('combined', num_classes=NUM_CLASSES)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0)  # frozen

        result = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=20,
            device='cpu',
            checkpoint_dir=tmp_path / 'ckpts',
            num_classes=NUM_CLASSES,
            patience=2,
            max_batches=1,
        )

        # Should not have run all 20 epochs
        assert result['best_epoch'] <= 20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
