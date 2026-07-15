import torch
import torch.nn as nn

from .components import DoubleConv, DownBlock, UpBlock, OutConv


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 1,
        features: list[int] | None = None,
        use_attention: bool = False,
    ):
        super().__init__()
        features = features or [64, 128, 256, 512]
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(in_channels, features[0])

        self.down_blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down_blocks.append(DownBlock(features[i], features[i + 1]))

        self.up_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.up_blocks.append(UpBlock(features[i], features[i - 1], use_attention=use_attention))

        self.outc = OutConv(features[0], out_channels)

    def forward(self, x):
        skip_connections = []

        x = self.inc(x)
        skip_connections.append(x)

        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)

        skip_connections = skip_connections[:-1]
        skip_connections = list(reversed(skip_connections))

        for i, up in enumerate(self.up_blocks):
            x = up(x, skip_connections[i])

        return self.outc(x)


class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        features: list[int] | None = None,
    ):
        super().__init__()
        features = features or [64, 128, 256, 512]

        self.inc = DoubleConv(in_channels, features[0])

        self.down_blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down_blocks.append(DownBlock(features[i], features[i + 1]))

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(features[-1], features[-1] * 2),
            nn.ConvTranspose2d(features[-1] * 2, features[-1], kernel_size=2, stride=2),
        )

        self.up_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.up_blocks.append(UpBlock(features[i], features[i - 1]))

        self.outc = nn.Sequential(
            nn.Conv2d(features[0], in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        skip_connections = []

        x = self.inc(x)
        skip_connections.append(x)

        for down in self.down_blocks:
            x = down(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[:-1]
        skip_connections = list(reversed(skip_connections))

        for i, up in enumerate(self.up_blocks):
            x = up(x, skip_connections[i])

        return self.outc(x)
