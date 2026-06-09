"""
U-Net Model Architecture

U-Net semantic segmentation model with MobileNetV2 encoder backbone.
Designed for multi-class segmentation using native classification codes.

The model accepts 4+ input channels:
  - Channels 0-2: RGB image
  - Channel 3:    Rasterized initial shapefile classification (class index)
  - Channels 4+:  Optional additional feature channels

Output: one logit per native classification code (17 classes by default,
corresponding to codes -1 and 0-20, with code -20 skipped).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import Config


class _InputAdapter(nn.Module):
    """Projects N-channel input to 3 channels expected by MobileNetV2.

    When in_channels == 3 this is an identity; otherwise a learned
    1×1 convolution maps the extra channels to the 3-channel space.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        if in_channels != 3:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(3),
                nn.ReLU(inplace=True),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class _DecoderBlock(nn.Module):
    """Single decoder block: upsample then double-conv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetMobileNetV2(nn.Module):
    """
    U-Net architecture with MobileNetV2 encoder.

    Accepts ``in_channels`` input channels (default 4: RGB + initial classification).
    When ``in_channels != 3`` an input adapter projects the extra channels to the
    3-channel space expected by MobileNetV2.

    Input:  (B, in_channels, 256, 256) float32
    Output: (B, num_classes, 256, 256) logits
    """

    # MobileNetV2 encoder stage output channels
    _ENC_CHANNELS = [16, 24, 32, 96, 1280]

    def __init__(
        self,
        num_classes: int = 17,
        in_channels: int = 4,
        pretrained: bool = True,
    ):
        """
        Initialize U-Net model.

        Args:
            num_classes: Number of output classes (default: 17)
            in_channels: Number of input channels (default: 4 = RGB + initial cls)
            pretrained: Use pretrained MobileNetV2 weights (default: True)
        """
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # ── Input adapter ─────────────────────────────────────────────
        self.input_adapter = _InputAdapter(in_channels)

        # ── Encoder (MobileNetV2) ──────────────────────────────────────────
        try:
            weights = (
                torchvision.models.MobileNet_V2_Weights.DEFAULT
                if pretrained
                else None
            )
            backbone = torchvision.models.mobilenet_v2(weights=weights)
        except AttributeError:
            # Older torchvision fallback
            backbone = torchvision.models.mobilenet_v2(pretrained=pretrained)

        feats = backbone.features  # nn.Sequential of 19 layers

        # Split into 5 stages that produce feature maps at successive strides.
        # For a 256×256 input the spatial sizes after each stage are:
        #   stage1 → 16 ch, 128×128  (stride ×2)
        #   stage2 → 24 ch,  64×64   (stride ×4)
        #   stage3 → 32 ch,  32×32   (stride ×8)
        #   stage4 → 96 ch,  16×16   (stride ×16)
        #   stage5 → 1280 ch,  8×8   (stride ×32)
        self.enc1 = nn.Sequential(*feats[0:2])    # 3  → 16
        self.enc2 = nn.Sequential(*feats[2:4])    # 16 → 24
        self.enc3 = nn.Sequential(*feats[4:7])    # 24 → 32
        self.enc4 = nn.Sequential(*feats[7:14])   # 32 → 96
        self.enc5 = nn.Sequential(*feats[14:19])  # 96 → 1280

        # ── Decoder ───────────────────────────────────────────────────────
        # Each block receives (upsampled previous output) cat (skip from encoder)
        self.dec5 = _DecoderBlock(1280, 96,  256)  # 8→16,   +e4(96)
        self.dec4 = _DecoderBlock(256,  32,  128)  # 16→32,  +e3(32)
        self.dec3 = _DecoderBlock(128,  24,   64)  # 32→64,  +e2(24)
        self.dec2 = _DecoderBlock(64,   16,   32)  # 64→128, +e1(16)
        self.dec1 = _DecoderBlock(32,    0,   16)  # 128→256 (no skip)

        # ── Segmentation head ────────────────────────────────────────
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Logit tensor of shape (B, num_classes, H, W)
        """
        x = self.input_adapter(x)  # → (B, 3, H, W)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d = self.dec5(e5, e4)
        d = self.dec4(d, e3)
        d = self.dec3(d, e2)
        d = self.dec2(d, e1)
        d = self.dec1(d)

        return self.head(d)


def create_model(
    num_classes: int = 17,
    in_channels: int = 4,
    pretrained: bool = True,
    device: str = 'cpu'
) -> UNetMobileNetV2:
    """
    Create and initialize a UNetMobileNetV2 model.

    Args:
        num_classes: Number of output classes (default: 17)
        in_channels: Number of input channels (default: 4 = RGB + initial cls)
        pretrained: Use pretrained MobileNetV2 encoder weights (default: True)
        device: Device string, e.g. 'cpu' or 'cuda' (default: 'cpu')

    Returns:
        Initialized UNetMobileNetV2 on *device*.
    """
    model = UNetMobileNetV2(
        num_classes=num_classes,
        in_channels=in_channels,
        pretrained=pretrained,
    )
    model = model.to(device)
    return model


def save_model(
    model: UNetMobileNetV2,
    checkpoint_path: Path,
    metadata: Optional[dict] = None,
) -> None:
    """
    Save model state_dict and optional metadata to a checkpoint file.

    Args:
        model: Model to save.
        checkpoint_path: Destination path (parent directory must exist).
        metadata: Optional dict with extra info (epoch, metrics, …).
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        'state_dict': model.state_dict(),
        'num_classes': model.num_classes,
        'in_channels': model.in_channels,
    }
    if metadata:
        payload.update(metadata)

    torch.save(payload, checkpoint_path)


def load_model(
    checkpoint_path: Path,
    num_classes: int = 17,
    in_channels: int = 4,
    device: str = 'cpu'
) -> UNetMobileNetV2:
    """
    Load a UNetMobileNetV2 from a saved checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file produced by save_model.
        num_classes: Number of output classes (overridden by checkpoint if present).
        in_channels: Number of input channels (overridden by checkpoint if present).
        device: Device to place the model on.

    Returns:
        UNetMobileNetV2 with loaded weights in eval mode.
    """
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device)

    n_cls = payload.get('num_classes', num_classes)
    n_ch = payload.get('in_channels', in_channels)
    model = UNetMobileNetV2(num_classes=n_cls, in_channels=n_ch, pretrained=False)
    model.load_state_dict(payload['state_dict'])
    model = model.to(device)
    model.eval()
    return model
