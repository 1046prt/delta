import sys
import os
import pytest
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.ndvi import compute_ndvi, compute_ndwi, compute_ndvi_torch, add_ndvi_channel_torch
from src.models.unet import UNet, Autoencoder
from src.models.siamese_unet import SiameseUNet
from src.models.fc_siam_diff import FCSiamDiff
from src.models.tinycd import TinyCD
from src.training.losses import CombinedLoss, DiceLoss, compute_pos_weight
from src.training.metrics import compute_iou, compute_f1, compute_precision, compute_recall


class TestNDVI:
    def test_compute_ndvi(self):
        nir = np.array([0.8, 0.6, 0.2, 0.0], dtype=np.float32)
        red = np.array([0.2, 0.3, 0.4, 0.0], dtype=np.float32)
        ndvi = compute_ndvi(nir, red)
        assert ndvi.shape == nir.shape
        assert np.all(ndvi >= -1.0) and np.all(ndvi <= 1.0)

    def test_ndvi_healthy_vegetation(self):
        nir = np.array([0.8], dtype=np.float32)
        red = np.array([0.1], dtype=np.float32)
        ndvi = compute_ndvi(nir, red)
        assert ndvi[0] > 0.6

    def test_ndvi_water(self):
        nir = np.array([0.1], dtype=np.float32)
        red = np.array([0.15], dtype=np.float32)
        ndvi = compute_ndvi(nir, red)
        assert ndvi[0] < 0

    def test_compute_ndwi(self):
        green = np.array([0.7], dtype=np.float32)
        nir = np.array([0.2], dtype=np.float32)
        ndwi = compute_ndwi(green, nir)
        assert ndwi[0] > 0

    def test_division_by_zero(self):
        nir = np.array([0.0], dtype=np.float32)
        red = np.array([0.0], dtype=np.float32)
        ndvi = compute_ndvi(nir, red)
        assert not np.isnan(ndvi[0])

    def test_ndvi_torch(self):
        nir = torch.tensor([0.8, 0.6, 0.2])
        red = torch.tensor([0.2, 0.3, 0.4])
        ndvi = compute_ndvi_torch(nir, red)
        assert ndvi.shape == nir.shape
        assert torch.all(ndvi >= -1.0) and torch.all(ndvi <= 1.0)

    def test_add_ndvi_channel_torch(self):
        image = torch.randn(4, 64, 64)
        result = add_ndvi_channel_torch(image, nir_band_idx=3, red_band_idx=0)
        assert result.shape == (5, 64, 64)


class TestModels:
    def test_unet_forward(self):
        model = UNet(in_channels=4, out_channels=1, features=[32, 64])
        x = torch.randn(2, 4, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_unet_with_attention(self):
        model = UNet(in_channels=4, out_channels=1, features=[32, 64], use_attention=True)
        x = torch.randn(2, 4, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_siamese_unet_5d(self):
        model = SiameseUNet(in_channels=4, out_channels=1, features=[32, 64])
        x = torch.randn(2, 2, 4, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_siamese_unet_4d(self):
        model = SiameseUNet(in_channels=4, out_channels=1, features=[32, 64])
        x = torch.randn(2, 8, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_autoencoder_forward(self):
        model = Autoencoder(in_channels=4, features=[32, 64])
        x = torch.randn(2, 4, 64, 64)
        out = model(x)
        assert out.shape == x.shape

    def test_autoencoder_output_range(self):
        model = Autoencoder(in_channels=4, features=[32, 64])
        x = torch.rand(2, 4, 64, 64)
        out = model(x)
        assert out.min() >= 0.0 and out.max() <= 1.0

    def test_fc_siam_diff_forward(self):
        model = FCSiamDiff(in_channels=3, out_channels=1)
        x = torch.randn(2, 6, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_fc_siam_diff_5d(self):
        model = FCSiamDiff(in_channels=3, out_channels=1)
        x = torch.randn(2, 2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_tinycd_forward(self):
        model = TinyCD(in_channels=3, out_channels=1)
        x = torch.randn(2, 6, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_tinycd_5d(self):
        model = TinyCD(in_channels=3, out_channels=1)
        x = torch.randn(2, 2, 3, 64, 64)
        out = model(x)
        assert out.shape == (2, 1, 64, 64)

    def test_fc_siam_diff_params(self):
        model = FCSiamDiff(in_channels=3, out_channels=1)
        params = sum(p.numel() for p in model.parameters())
        assert params < 5_000_000  # Should be ~2M

    def test_tinycd_params(self):
        model = TinyCD(in_channels=3, out_channels=1)
        params = sum(p.numel() for p in model.parameters())
        assert params < 5_000_000  # Should be ~1M


class TestLosses:
    def test_dice_loss(self):
        criterion = DiceLoss()
        pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = criterion(pred, target)
        assert loss.item() >= 0
        assert loss.item() <= 1

    def test_combined_loss(self):
        criterion = CombinedLoss(alpha=0.5)
        pred = torch.randn(2, 1, 64, 64)
        target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        loss = criterion(pred, target)
        assert loss.item() >= 0

    def test_combined_loss_perfect(self):
        criterion = CombinedLoss(alpha=0.5)
        target = torch.ones(2, 1, 64, 64)
        pred = torch.ones(2, 1, 64, 64) * 10
        loss = criterion(pred, target)
        assert loss.item() < 0.5

    def test_pos_weight(self):
        labels = torch.zeros(8, 1, 64, 64)
        labels[:, :, :5, :] = 1.0
        pw = compute_pos_weight(labels)
        assert pw.item() > 1.0


class TestMetrics:
    def test_iou_perfect(self):
        pred = torch.ones(4, 1, 64, 64) * 10
        target = torch.ones(4, 1, 64, 64)
        iou = compute_iou(pred, target, threshold=0.5)
        assert iou > 0.99

    def test_iou_zero(self):
        pred = torch.ones(4, 1, 64, 64) * -10
        target = torch.ones(4, 1, 64, 64)
        iou = compute_iou(pred, target, threshold=0.5)
        assert iou < 0.01

    def test_f1_perfect(self):
        pred = torch.ones(4, 1, 64, 64) * 10
        target = torch.ones(4, 1, 64, 64)
        f1 = compute_f1(pred, target, threshold=0.5)
        assert f1 > 0.99

    def test_precision(self):
        pred = torch.ones(4, 1, 64, 64) * 10
        target = torch.zeros(4, 1, 64, 64)
        prec = compute_precision(pred, target, threshold=0.5)
        assert prec < 0.01

    def test_recall(self):
        pred = torch.ones(4, 1, 64, 64) * -10
        target = torch.ones(4, 1, 64, 64)
        rec = compute_recall(pred, target, threshold=0.5)
        assert rec < 0.01
