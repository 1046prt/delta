"""
FC-Siam-diff: Fully Convolutional Siamese Network with Difference Skip Connections.

Based on: Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese
networks for change detection". In 2018 25th IEEE International Conference on
Image Processing (ICIP) (pp. 4063-4067). IEEE.

Reference: https://github.com/rcdaudt/fully_convolutional_change_detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ReplicationPad2d


class FCSiamDiff(nn.Module):
    """FC-Siam-diff: Siamese encoder with absolute difference skip connections."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels

        # Encoder level 1: in_channels -> 16
        self.conv11 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(16)
        self.do11 = nn.Dropout2d(p=0.2)
        self.conv12 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.do12 = nn.Dropout2d(p=0.2)

        # Encoder level 2: 16 -> 32
        self.conv21 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.do21 = nn.Dropout2d(p=0.2)
        self.conv22 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(32)
        self.do22 = nn.Dropout2d(p=0.2)

        # Encoder level 3: 32 -> 64
        self.conv31 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(64)
        self.do31 = nn.Dropout2d(p=0.2)
        self.conv32 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(64)
        self.do32 = nn.Dropout2d(p=0.2)
        self.conv33 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(64)
        self.do33 = nn.Dropout2d(p=0.2)

        # Encoder level 4: 64 -> 128
        self.conv41 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(128)
        self.do41 = nn.Dropout2d(p=0.2)
        self.conv42 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(128)
        self.do42 = nn.Dropout2d(p=0.2)
        self.conv43 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(128)
        self.do43 = nn.Dropout2d(p=0.2)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1,
                                           stride=2, output_padding=1)
        self.conv43d = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(128)
        self.do43d = nn.Dropout2d(p=0.2)
        self.conv42d = nn.ConvTranspose2d(128, 128, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(128)
        self.do42d = nn.Dropout2d(p=0.2)
        self.conv41d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(64)
        self.do41d = nn.Dropout2d(p=0.2)

        self.upconv3 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1,
                                           stride=2, output_padding=1)
        self.conv33d = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(64)
        self.do33d = nn.Dropout2d(p=0.2)
        self.conv32d = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(64)
        self.do32d = nn.Dropout2d(p=0.2)
        self.conv31d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(32)
        self.do31d = nn.Dropout2d(p=0.2)

        self.upconv2 = nn.ConvTranspose2d(32, 32, kernel_size=3, padding=1,
                                           stride=2, output_padding=1)
        self.conv22d = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(32)
        self.do22d = nn.Dropout2d(p=0.2)
        self.conv21d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(16)
        self.do21d = nn.Dropout2d(p=0.2)

        self.upconv1 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1,
                                           stride=2, output_padding=1)
        self.conv12d = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(16)
        self.do12d = nn.Dropout2d(p=0.2)
        self.conv11d = nn.ConvTranspose2d(16, out_channels, kernel_size=3, padding=1)

    def _encode(self, x):
        """Shared encoder path."""
        x11 = self.do11(F.relu(self.bn11(self.conv11(x))))
        x12 = self.do12(F.relu(self.bn12(self.conv12(x11))))
        x1p = F.max_pool2d(x12, kernel_size=2, stride=2)

        x21 = self.do21(F.relu(self.bn21(self.conv21(x1p))))
        x22 = self.do22(F.relu(self.bn22(self.conv22(x21))))
        x2p = F.max_pool2d(x22, kernel_size=2, stride=2)

        x31 = self.do31(F.relu(self.bn31(self.conv31(x2p))))
        x32 = self.do32(F.relu(self.bn32(self.conv32(x31))))
        x33 = self.do33(F.relu(self.bn33(self.conv33(x32))))
        x3p = F.max_pool2d(x33, kernel_size=2, stride=2)

        x41 = self.do41(F.relu(self.bn41(self.conv41(x3p))))
        x42 = self.do42(F.relu(self.bn42(self.conv42(x41))))
        x43 = self.do43(F.relu(self.bn43(self.conv43(x42))))

        return x12, x22, x33, x43

    def forward(self, x):
        """Forward pass. Input is stacked [T1, T2] as (B, C*2, H, W)."""
        if x.dim() == 5:
            t1, t2 = x[:, 0], x[:, 1]
        else:
            c = x.shape[1] // 2
            t1, t2 = x[:, :c], x[:, c:]

        # Shared encoder
        s1_1, s1_2, s1_3, s1_4 = self._encode(t1)
        s2_1, s2_2, s2_3, s2_4 = self._encode(t2)

        # Absolute difference at each level
        d1 = torch.abs(s1_1 - s2_1)  # 16 ch
        d2 = torch.abs(s1_2 - s2_2)  # 32 ch
        d3 = torch.abs(s1_3 - s2_3)  # 64 ch
        d4 = torch.abs(s1_4 - s2_4)  # 128 ch

        # Decoder with difference skip connections
        # Level 4: upsample d4(128) concat d4(128) = 256
        x4d = self.upconv4(d4)
        pad4 = ReplicationPad2d((0, d4.size(3) - x4d.size(3),
                                  0, d4.size(2) - x4d.size(2)))
        x4d = torch.cat((pad4(x4d), d4), 1)
        x4d = self.do43d(F.relu(self.bn43d(self.conv43d(x4d))))
        x4d = self.do42d(F.relu(self.bn42d(self.conv42d(x4d))))
        x4d = self.do41d(F.relu(self.bn41d(self.conv41d(x4d))))

        # Level 3: upsample(64) concat d3(64) = 128
        x3d = self.upconv3(x4d)
        pad3 = ReplicationPad2d((0, d3.size(3) - x3d.size(3),
                                  0, d3.size(2) - x3d.size(2)))
        x3d = torch.cat((pad3(x3d), d3), 1)
        x3d = self.do33d(F.relu(self.bn33d(self.conv33d(x3d))))
        x3d = self.do32d(F.relu(self.bn32d(self.conv32d(x3d))))
        x3d = self.do31d(F.relu(self.bn31d(self.conv31d(x3d))))

        # Level 2: upsample(32) concat d2(32) = 64
        x2d = self.upconv2(x3d)
        pad2 = ReplicationPad2d((0, d2.size(3) - x2d.size(3),
                                  0, d2.size(2) - x2d.size(2)))
        x2d = torch.cat((pad2(x2d), d2), 1)
        x2d = self.do22d(F.relu(self.bn22d(self.conv22d(x2d))))
        x2d = self.do21d(F.relu(self.bn21d(self.conv21d(x2d))))

        # Level 1: upsample(16) concat d1(16) = 32
        x1d = self.upconv1(x2d)
        pad1 = ReplicationPad2d((0, d1.size(3) - x1d.size(3),
                                  0, d1.size(2) - x1d.size(2)))
        x1d = torch.cat((pad1(x1d), d1), 1)
        x1d = self.do12d(F.relu(self.bn12d(self.conv12d(x1d))))
        out = self.conv11d(x1d)

        return out
