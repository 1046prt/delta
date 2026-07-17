"""
TinyCD: A (Not So) Deep Learning Model for Change Detection.

Based on: Codegoni, A., Lombardi, G., & Ferrari, A. "TinyCD: A (Not so) Deep
Learning Model for Change Detection". Neural Computing and Applications, 35(11),
8471-8486, 2023.

Simplified implementation for OSCD100 (3-channel RGB, 256x256).
Uses a lightweight CNN backbone instead of EfficientNet-b4 for CPU feasibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixingBlock(nn.Module):
    """Mix features from two temporal streams in space-time domain."""

    def __init__(self, channels: int):
        super().__init__()
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(channels),
            nn.PReLU(),
        )

    def forward(self, x, y):
        return self.mix(torch.cat([x, y], dim=1))


class MAMB(nn.Module):
    """Mix and Attention Mask Block.

    Generates spatial attention masks from mixed features using pixel-wise MLP.
    """

    def __init__(self, in_channels: int, mid_channels: int | None = None):
        super().__init__()
        mid_channels = mid_channels or in_channels // 2
        self.mask_gen = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(mid_channels),
            nn.PReLU(),
            nn.Conv2d(mid_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, mixed):
        return self.mask_gen(mixed)


class TinyCDBackbone(nn.Module):
    """Lightweight Siamese backbone with 4 levels."""

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # Level 1: 3 -> 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.PReLU(),
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.PReLU(),
        )
        # Level 2: 16 -> 32
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.PReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.PReLU(),
        )
        # Level 3: 32 -> 64
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.PReLU(),
        )
        # Level 4 (bottleneck): 64 -> 128
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.PReLU(),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool(x1))
        x3 = self.conv3(self.pool(x2))
        x4 = self.conv4(self.pool(x3))
        return x1, x2, x3, x4


class TinyCD(nn.Module):
    """TinyCD: Lightweight Siamese U-Net with MAMB attention.

    Architecture:
    - Siamese backbone extracts multi-scale features
    - MAMB generates spatial attention masks at each scale
    - Decoder uses attention masks as skip connections
    - Pixel-wise MLP classifier
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.backbone = TinyCDBackbone(in_channels)

        # MAMB blocks for each skip connection level
        # Input is concat of two streams: 16*2=32, 32*2=64, 64*2=128
        self.mamb1 = MAMB(32, 16)
        self.mamb2 = MAMB(64, 32)
        self.mamb3 = MAMB(128, 64)

        # Bottleneck mixing
        self.mix_bottleneck = MixingBlock(128)

        # Decoder
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, 3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.PReLU(),
        )

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.PReLU(),
        )

        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.PReLU(),
        )

        self.up1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(16 + 16, 16, 3, padding=1, bias=False),
            nn.InstanceNorm2d(16),
            nn.PReLU(),
        )

        # Pixel-wise MLP classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(16, 8, 1),
            nn.PReLU(),
            nn.Conv2d(8, out_channels, 1),
        )

    def forward(self, x):
        if x.dim() == 5:
            t1, t2 = x[:, 0], x[:, 1]
        else:
            c = x.shape[1] // 2
            t1, t2 = x[:, :c], x[:, c:]

        # Siamese backbone
        s1_1, s1_2, s1_3, s1_4 = self.backbone(t1)
        s2_1, s2_2, s2_3, s2_4 = self.backbone(t2)

        # MAMB: mix features and generate attention masks
        mix1 = torch.cat([s1_1, s2_1], dim=1)
        m1 = self.mamb1(mix1)
        att1 = s1_1 * m1 + s2_1 * (1 - m1)

        mix2 = torch.cat([s1_2, s2_2], dim=1)
        m2 = self.mamb2(mix2)
        att2 = s1_2 * m2 + s2_2 * (1 - m2)

        mix3 = torch.cat([s1_3, s2_3], dim=1)
        m3 = self.mamb3(mix3)
        att3 = s1_3 * m3 + s2_3 * (1 - m3)

        # Bottleneck mixing
        mix4 = self.mix_bottleneck(s1_4, s2_4)

        # Decoder
        x = self.up4(mix4)
        x = self.dec4(torch.cat([x, att3], dim=1))

        x = self.up3(x)
        x = self.dec3(torch.cat([x, att2], dim=1))

        x = self.up2(x)
        x = self.dec2(torch.cat([x, att1], dim=1))

        # Final upsample to full resolution
        x = self.up1(x)
        # Pad if needed
        if x.shape[2] != att1.shape[2] or x.shape[3] != att1.shape[3]:
            x = F.interpolate(x, size=att1.shape[2:], mode='bilinear', align_corners=True)
        x = self.dec1(torch.cat([x, att1], dim=1))

        return self.classifier(x)
