import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import DoubleConv, DownBlock, UpBlock, OutConv


class SiameseEncoder(nn.Module):
    def __init__(self, in_channels: int, features: list[int]):
        super().__init__()
        self.inc = DoubleConv(in_channels, features[0])
        self.down_blocks = nn.ModuleList()
        for i in range(len(features) - 1):
            self.down_blocks.append(DownBlock(features[i], features[i + 1]))

    def forward(self, x):
        skips = []
        x = self.inc(x)
        skips.append(x)
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
        return x, skips


class DifferenceBlock(nn.Module):
    def __init__(self, use_abs: bool = True):
        super().__init__()
        self.use_abs = use_abs

    def forward(self, feat_t1, feat_t2):
        if self.use_abs:
            return torch.abs(feat_t1 - feat_t2)
        return feat_t1 - feat_t2


class SiameseUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        features: list[int] | None = None,
        use_attention: bool = False,
        use_abs_diff: bool = True,
    ):
        super().__init__()
        features = features or [64, 128, 256, 512]
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = SiameseEncoder(in_channels, features)
        self.diff_block = DifferenceBlock(use_abs=use_abs_diff)

        self.up_blocks = nn.ModuleList()
        for i in range(len(features) - 1, 0, -1):
            self.up_blocks.append(UpBlock(features[i], features[i - 1], use_attention=use_attention))

        self.outc = OutConv(features[0], out_channels)

    def forward(self, x):
        if x.dim() == 5:
            t1 = x[:, 0]
            t2 = x[:, 1]
        else:
            n_dates = 2
            c_per_date = x.shape[1] // n_dates
            t1 = x[:, :c_per_date]
            t2 = x[:, c_per_date:]

        bottleneck_t1, skips_t1 = self.encoder(t1)
        bottleneck_t2, skips_t2 = self.encoder(t2)

        diff_bottleneck = self.diff_block(bottleneck_t1, bottleneck_t2)
        diff_skips = [self.diff_block(s1, s2) for s1, s2 in zip(skips_t1[:-1], skips_t2[:-1])]

        diff_skips = list(reversed(diff_skips))

        x = diff_bottleneck
        for i, up in enumerate(self.up_blocks):
            x = up(x, diff_skips[i])

        return self.outc(x)
