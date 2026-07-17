import os
import sys
import tempfile
import torch
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.siamese_unet import SiameseUNet
from src.models.unet import UNet
from src.inference.predict import load_model, predict, predict_autoencoder


class TestLoadModel:
    def test_load_siamese_unet(self):
        model = SiameseUNet(6, 1, [64, 128], use_attention=False)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "model": {"type": "siamese_unet", "in_channels": 6, "features": [64, 128], "use_attention": False},
                    "data": {"normalize": False},
                },
            }, f.name)
            ckpt_path = f.name

        try:
            loaded_model, config, normalize = load_model(ckpt_path, torch.device("cpu"))
            assert isinstance(loaded_model, SiameseUNet)
            assert config["model"]["type"] == "siamese_unet"
            assert normalize is False
        finally:
            os.unlink(ckpt_path)

    def test_load_unet(self):
        model = UNet(6, 1, [64, 128])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "model": {"type": "unet", "in_channels": 3, "features": [64, 128]},
                    "data": {"normalize": True},
                },
            }, f.name)
            ckpt_path = f.name

        try:
            loaded_model, config, normalize = load_model(ckpt_path, torch.device("cpu"))
            assert isinstance(loaded_model, UNet)
            assert normalize is True
        finally:
            os.unlink(ckpt_path)

    def test_load_with_ema(self):
        model = SiameseUNet(6, 1, [64, 128])
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({
                "model_state_dict": model.state_dict(),
                "ema_state_dict": model.state_dict(),
                "config": {
                    "model": {"type": "siamese_unet", "in_channels": 6, "features": [64, 128], "use_attention": False},
                    "data": {"normalize": False},
                },
            }, f.name)
            ckpt_path = f.name

        try:
            loaded_model, _, _ = load_model(ckpt_path, torch.device("cpu"))
            assert isinstance(loaded_model, SiameseUNet)
        finally:
            os.unlink(ckpt_path)

    def test_unknown_model_type_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save({
                "model_state_dict": {},
                "config": {
                    "model": {"type": "unknown", "in_channels": 6, "features": [64]},
                    "data": {},
                },
            }, f.name)
            ckpt_path = f.name

        try:
            with pytest.raises(ValueError, match="Unknown model type"):
                load_model(ckpt_path, torch.device("cpu"))
        finally:
            os.unlink(ckpt_path)


class TestPredict:
    def test_predict_shape(self):
        model = SiameseUNet(3, 1, [32, 64], use_attention=False)
        model.eval()
        t1 = torch.randn(1, 3, 256, 256)
        t2 = torch.randn(1, 3, 256, 256)

        binary_mask, change_map = predict(model, t1, t2, torch.device("cpu"),
                                          threshold=0.5, patch_size=256, overlap=0)

        assert binary_mask.shape == (256, 256)
        assert change_map.shape == (256, 256)
        assert binary_mask.dtype == np.uint8
        assert np.isin(binary_mask, [0, 1]).all()

    def test_predict_with_overlap(self):
        model = SiameseUNet(3, 1, [32, 64], use_attention=False)
        model.eval()
        t1 = torch.randn(1, 3, 256, 256)
        t2 = torch.randn(1, 3, 256, 256)

        binary_mask, change_map = predict(model, t1, t2, torch.device("cpu"),
                                          threshold=0.5, patch_size=256, overlap=32)

        assert binary_mask.shape == (256, 256)
        assert change_map.shape == (256, 256)

    def test_predict_values_in_range(self):
        model = SiameseUNet(3, 1, [32, 64])
        model.eval()
        t1 = torch.randn(1, 3, 256, 256)
        t2 = torch.randn(1, 3, 256, 256)

        _, change_map = predict(model, t1, t2, torch.device("cpu"),
                                threshold=0.5, patch_size=256)

        assert change_map.min() >= 0.0
        assert change_map.max() <= 1.0

    def test_predict_threshold_zero_all_changed(self):
        model = SiameseUNet(3, 1, [32, 64])
        model.eval()
        t1 = torch.randn(1, 3, 256, 256)
        t2 = torch.randn(1, 3, 256, 256)

        binary_mask, _ = predict(model, t1, t2, torch.device("cpu"),
                                 threshold=0.0, patch_size=256)

        assert binary_mask.sum() == 256 * 256

    def test_predict_threshold_one_nothing_changed(self):
        model = SiameseUNet(3, 1, [32, 64])
        model.eval()
        t1 = torch.randn(1, 3, 256, 256)
        t2 = torch.randn(1, 3, 256, 256)

        binary_mask, _ = predict(model, t1, t2, torch.device("cpu"),
                                 threshold=1.0, patch_size=256)

        assert binary_mask.sum() == 0


class TestPredictAutoencoder:
    def test_autoencoder_predict_shape(self):
        from src.models.unet import Autoencoder
        model = Autoencoder(3, [32, 64])
        model.eval()
        t1 = torch.randn(1, 3, 256, 256)
        t2 = torch.randn(1, 3, 256, 256)

        binary_mask, change_map = predict_autoencoder(model, t1, t2, torch.device("cpu"))

        assert binary_mask.shape == (256, 256)
        assert change_map.shape == (256, 256)

    def test_autoencoder_non_divisible_size(self):
        from src.models.unet import Autoencoder
        model = Autoencoder(3, [32, 64])
        model.eval()
        t1 = torch.randn(1, 3, 200, 300)
        t2 = torch.randn(1, 3, 200, 300)

        binary_mask, change_map = predict_autoencoder(model, t1, t2, torch.device("cpu"))

        assert binary_mask.shape == (200, 300)
        assert change_map.shape == (200, 300)


class TestMetricsExtended:
    def test_oa_kappa_perfect(self):
        y_true = torch.zeros(10, 10)
        y_true[2:5, 2:5] = 1
        y_pred = y_true.clone()

        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        tn = ((y_pred == 0) & (y_true == 0)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()

        oa = (tp + tn) / (tp + fp + fn + tn)
        assert oa == 1.0

    def test_oa_kappa_random(self):
        y_true = torch.zeros(10, 10)
        y_true[2:5, 2:5] = 1
        y_pred = (torch.rand(10, 10) > 0.5).float()

        tp = ((y_pred == 1) & (y_true == 1)).sum().item()
        tn = ((y_pred == 0) & (y_true == 0)).sum().item()
        fp = ((y_pred == 1) & (y_true == 0)).sum().item()
        fn = ((y_pred == 0) & (y_true == 1)).sum().item()

        oa = (tp + tn) / (tp + fp + fn + tn)
        assert 0.0 <= oa <= 1.0
