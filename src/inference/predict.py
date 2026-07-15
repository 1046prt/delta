import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.unet import UNet
from src.models.siamese_unet import SiameseUNet
from src.models.unet import Autoencoder
from src.data.ndvi import add_ndvi_channel_torch


def load_model(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model_type = config["model"]["type"]
    in_ch = config["model"]["in_channels"]
    out_ch = config["model"].get("out_channels", 1)
    features = config["model"]["features"]
    use_attn = config["model"].get("use_attention", False)

    if model_type == "siamese_unet":
        model = SiameseUNet(in_ch, out_ch, features, use_attention=use_attn)
    elif model_type == "unet":
        model = UNet(in_ch, out_ch, features, use_attention=use_attn)
    elif model_type == "autoencoder":
        model = Autoencoder(in_ch, features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def preprocess_pair(
    t1_path: str,
    t2_path: str,
    bands: list[str] | None = None,
    use_ndvi: bool = True,
    patch_size: int = 256,
):
    try:
        import rasterio
        with rasterio.open(t1_path) as src:
            t1 = src.read().astype(np.float32)
        with rasterio.open(t2_path) as src:
            t2 = src.read().astype(np.float32)
    except Exception:
        t1 = np.array(Image.open(t1_path)).astype(np.float32).transpose(2, 0, 1)
        t2 = np.array(Image.open(t2_path)).astype(np.float32).transpose(2, 0, 1)

    h = min(t1.shape[1], t2.shape[1])
    w = min(t1.shape[2], t2.shape[2])
    h = (h // patch_size) * patch_size
    w = (w // patch_size) * patch_size
    t1 = t1[:, :h, :w]
    t2 = t2[:, :h, :w]

    t1_tensor = torch.from_numpy(t1).unsqueeze(0)
    t2_tensor = torch.from_numpy(t2).unsqueeze(0)

    if use_ndvi:
        t1_tensor = add_ndvi_channel_torch(t1_tensor.squeeze(0)).unsqueeze(0)
        t2_tensor = add_ndvi_channel_torch(t2_tensor.squeeze(0)).unsqueeze(0)

    return t1_tensor, t2_tensor, h, w


@torch.no_grad()
def predict(
    model,
    t1_tensor: torch.Tensor,
    t2_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
    patch_size: int = 256,
    overlap: int = 32,
) -> np.ndarray:
    model.eval()
    _, c, h, w = t1_tensor.shape

    change_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    stride = patch_size - overlap

    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            t1_patch = t1_tensor[:, :, y:y+patch_size, x:x+patch_size].to(device)
            t2_patch = t2_tensor[:, :, y:y+patch_size, x:x+patch_size].to(device)

            stacked = torch.cat([t1_patch, t2_patch], dim=1)
            pred = torch.sigmoid(model(stacked))

            change_map[y:y+patch_size, x:x+patch_size] += pred.squeeze().cpu().numpy()
            count_map[y:y+patch_size, x:x+patch_size] += 1.0

    count_map = np.maximum(count_map, 1.0)
    change_map /= count_map

    binary_mask = (change_map > threshold).astype(np.uint8)
    return binary_mask, change_map


@torch.no_grad()
def predict_autoencoder(
    model,
    t1_tensor: torch.Tensor,
    t2_tensor: torch.Tensor,
    device: torch.device,
    change_threshold: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    t1 = t1_tensor.to(device)
    t2 = t2_tensor.to(device)

    recon_t1 = model(t1)
    recon_t2 = model(t2)

    error_t1 = torch.mean((t1 - recon_t1) ** 2, dim=1, keepdim=True)
    error_t2 = torch.mean((t2 - recon_t2) ** 2, dim=1, keepdim=True)
    change_map = torch.abs(error_t1 - error_t2)

    change_np = change_map.squeeze().cpu().numpy()
    binary_mask = (change_np > change_threshold).astype(np.uint8)

    return binary_mask, change_np


def predict_from_paths(
    checkpoint_path: str,
    t1_path: str,
    t2_path: str,
    output_path: str | None = None,
    threshold: float = 0.5,
    device_str: str = "auto",
):
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    model, config = load_model(checkpoint_path, device)
    model_type = config["model"]["type"]
    use_ndvi = config["data"].get("use_ndvi", True)

    t1_tensor, t2_tensor, h, w = preprocess_pair(
        t1_path, t2_path,
        bands=config["data"]["bands"],
        use_ndvi=use_ndvi,
    )

    if model_type == "autoencoder":
        binary_mask, change_map = predict_autoencoder(
            model, t1_tensor, t2_tensor, device,
            change_threshold=config["inference"]["change_threshold"],
        )
    else:
        binary_mask, change_map = predict(
            model, t1_tensor, t2_tensor, device,
            threshold=threshold,
            patch_size=config["data"]["patch_size"],
            overlap=config["inference"].get("overlap", 32),
        )

    if output_path:
        change_img = (change_map * 255).astype(np.uint8)
        Image.fromarray(change_img).save(output_path)

    return binary_mask, change_map
