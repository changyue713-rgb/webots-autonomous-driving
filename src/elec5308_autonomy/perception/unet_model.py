"""
lane_detection/models.py
-------------------------
Segmentation model architectures.

UNet       – U-Net with a pretrained ResNet-18 encoder and skip connections.
             Ronneberger et al. (2015). U-Net: Convolutional Networks for
             Biomedical Image Segmentation. MICCAI.
             doi:10.1007/978-3-319-24574-4_28

SimpleUNet – Lightweight U-Net without a pretrained backbone.
             Suitable for CPU-only or quick-demo environments.

University of Sydney – ELEC5308 Intelligent Information Engineering Practice
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """
    Two consecutive  Conv2d(3×3) → BatchNorm2d → ReLU  blocks.

    This is the fundamental unit of both the encoder and decoder in the
    original U-Net architecture.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    One U-Net decoder step:
      Transposed Convolution (×2 upsampling)  →  concatenate skip  →  DoubleConv.

    The skip connection re-introduces encoder spatial detail that was lost
    during the downsampling path, which is the key innovation in U-Net.

    Parameters
    ----------
    in_ch   : channels arriving from the deeper decoder level.
    skip_ch : channels arriving from the matching encoder skip connection.
    out_ch  : output channels of this block.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        # Transposed convolution doubles spatial dimensions
        self.upsample = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=2, stride=2
        )
        # After concatenation we have out_ch (from upsample) + skip_ch
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)

        # Pad / crop to handle odd-dimension inputs gracefully
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)   # concat along channel axis
        return self.conv(x)


# ---------------------------------------------------------------------------
# U-Net with ResNet-18 encoder (primary model)
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    """
    U-Net with a pretrained ResNet-18 encoder.

    Architecture Overview
    ---------------------
    Encoder  (ResNet-18 backbone):
      Layer 0  → 64  ch, H/2  × W/2  (conv1 + bn1 + relu)
      Layer 1  → 64  ch, H/4  × W/4  (resnet layer1, after maxpool)
      Layer 2  → 128 ch, H/8  × W/8  (resnet layer2)
      Layer 3  → 256 ch, H/16 × W/16 (resnet layer3)
      Layer 4  → 512 ch, H/32 × W/32 (resnet layer4) ← bottleneck

    Decoder  (4 DecoderBlocks with skip connections):
      dec4: 512 → 256, H/32 → H/16  (skip from enc3: 256 ch)
      dec3: 256 → 128, H/16 → H/8   (skip from enc2: 128 ch)
      dec2: 128 → 64,  H/8  → H/4   (skip from enc1: 64 ch)
      dec1:  64 → 32,  H/4  → H/2   (skip from enc0: 64 ch)

    Head:
      TransposedConv2d(32→32, stride=2) → Conv2d(32→num_classes, 1×1)
      Output: (B, num_classes, H, W)

    Parameters
    ----------
    num_classes : number of output classes (e.g. 2 for binary lane/background).
    pretrained  : load ImageNet-pretrained weights for the ResNet-18 encoder.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.num_classes = num_classes

        # ── Encoder: ResNet-18 backbone ──────────────────────────────────
        weights  = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Re-use the backbone layers as encoder stages
        self.enc0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu
        )                                    # (B,  64, H/2,  W/2)
        self.pool = backbone.maxpool         # (B,  64, H/4,  W/4)
        self.enc1 = backbone.layer1          # (B,  64, H/4,  W/4)
        self.enc2 = backbone.layer2          # (B, 128, H/8,  W/8)
        self.enc3 = backbone.layer3          # (B, 256, H/16, W/16)
        self.enc4 = backbone.layer4          # (B, 512, H/32, W/32) ← bottleneck

        # ── Decoder: four blocks with skip connections ───────────────────
        self.dec4 = DecoderBlock(512, 256, 256)   # H/32 → H/16
        self.dec3 = DecoderBlock(256, 128, 128)   # H/16 → H/8
        self.dec2 = DecoderBlock(128,  64,  64)   # H/8  → H/4
        self.dec1 = DecoderBlock( 64,  64,  32)   # H/4  → H/2

        # ── Classification head ─────────────────────────────────────────
        # Final transposed conv: H/2 → H
        self.final_up = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.head     = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W) – normalised RGB image batch.

        Returns
        -------
        logits : (B, num_classes, H, W) – raw (unnormalised) class scores.
        """
        # ── Encoder ──────────────────────────────────────────────────────
        s0 = self.enc0(x)             # (B,  64, H/2,  W/2)
        s1 = self.enc1(self.pool(s0)) # (B,  64, H/4,  W/4)
        s2 = self.enc2(s1)            # (B, 128, H/8,  W/8)
        s3 = self.enc3(s2)            # (B, 256, H/16, W/16)
        s4 = self.enc4(s3)            # (B, 512, H/32, W/32) ← bottleneck

        # ── Decoder ──────────────────────────────────────────────────────
        d = self.dec4(s4, s3)         # (B, 256, H/16, W/16)
        d = self.dec3(d,  s2)         # (B, 128, H/8,  W/8)
        d = self.dec2(d,  s1)         # (B,  64, H/4,  W/4)
        d = self.dec1(d,  s0)         # (B,  32, H/2,  W/2)

        # ── Head ─────────────────────────────────────────────────────────
        d = self.final_up(d)          # (B,  32, H,    W)
        return self.head(d)           # (B, num_classes, H, W)

    def freeze_encoder(self):
        """Freeze all encoder (ResNet-18) parameters."""
        for module in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters (for fine-tuning)."""
        for module in [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]:
            for param in module.parameters():
                param.requires_grad = True

    def count_parameters(self) -> dict:
        """Return trainable and total parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ---------------------------------------------------------------------------
# Lightweight U-Net (no pretrained backbone)
# ---------------------------------------------------------------------------

class SimpleUNet(nn.Module):
    """
    Compact U-Net without an external pretrained backbone.

    Suitable for:
    * CPU-only environments (faster to forward-pass than the full UNet).
    * Quick conceptual demonstrations.
    * Ablation studies comparing trained vs. random-init encoders.

    Parameters
    ----------
    in_ch       : input channels (3 for RGB).
    num_classes : number of output classes.
    base_ch     : base channel width; each encoder stage doubles this.
                  Reduce for smaller memory footprint (e.g. base_ch=16).
    """

    def __init__(
        self,
        in_ch:       int = 3,
        num_classes: int = 2,
        base_ch:     int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        c = base_ch

        # ── Encoder ──────────────────────────────────────────────────────
        self.e1 = DoubleConv(in_ch, c)        # (B,  c, H,   W)
        self.e2 = DoubleConv(c,    c * 2)     # (B, 2c, H/2, W/2)
        self.e3 = DoubleConv(c*2,  c * 4)     # (B, 4c, H/4, W/4)
        self.e4 = DoubleConv(c*4,  c * 8)     # (B, 8c, H/8, W/8)

        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ───────────────────────────────────────────────────
        self.bottleneck = DoubleConv(c*8, c*16)  # (B, 16c, H/16, W/16)

        # ── Decoder ──────────────────────────────────────────────────────
        self.d4 = DecoderBlock(c*16, c*8, c*8)
        self.d3 = DecoderBlock(c*8,  c*4, c*4)
        self.d2 = DecoderBlock(c*4,  c*2, c*2)
        self.d1 = DecoderBlock(c*2,  c,   c)

        # ── Head ─────────────────────────────────────────────────────────
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))

        bn = self.bottleneck(self.pool(s4))

        d = self.d4(bn, s4)
        d = self.d3(d,  s3)
        d = self.d2(d,  s2)
        d = self.d1(d,  s1)

        return self.head(d)

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
