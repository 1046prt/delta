import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.unet import UNet
from src.models.siamese_unet import SiameseUNet
from src.models.unet import Autoencoder
from src.data.ndvi import add_ndvi_channel_torch


def load_model(checkpoint_path: str, device: torch.device):
    """Load a trained model from checkpoint with its config."""
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
        model = UNet(in_ch * 2, out_ch, features, use_attention=use_attn)
    elif model_type == "autoencoder":
        model = Autoencoder(in_ch, features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if "ema_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["ema_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    normalize = config.get("data", {}).get("normalize", False)
    return model, config, normalize


def preprocess_pair(
    t1_path: str,
    t2_path: str,
    bands: list[str] | None = None,
    use_ndvi: bool = True,
    patch_size: int = 256,
):
    """Load a pair of satellite images and prepare them as tensors for inference."""
    try:
        import rasterio
        with rasterio.open(t1_path) as src:
            t1 = src.read().astype(np.float32)
        with rasterio.open(t2_path) as src:
            t2 = src.read().astype(np.float32)
    except Exception:
        t1 = np.array(Image.open(t1_path)).astype(np.float32)
        t2 = np.array(Image.open(t2_path)).astype(np.float32)
        if t1.ndim == 2:
            t1 = np.stack([t1] * 3, axis=-1)
        elif t1.shape[-1] == 4:
            t1 = t1[:, :, :3]
        if t2.ndim == 2:
            t2 = np.stack([t2] * 3, axis=-1)
        elif t2.shape[-1] == 4:
            t2 = t2[:, :, :3]
        t1 = t1.transpose(2, 0, 1)
        t2 = t2.transpose(2, 0, 1)

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


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@torch.no_grad()
def predict(
    model,
    t1_tensor: torch.Tensor,
    t2_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
    patch_size: int = 256,
    overlap: int = 32,
    normalize: bool = False,
) -> np.ndarray:
    model.eval()
    _, c, h, w = t1_tensor.shape

    if normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        t1_tensor = (t1_tensor / 255.0 - mean) / std
        t2_tensor = (t2_tensor / 255.0 - mean) / std

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
    """Run change detection using reconstruction error from an autoencoder.

    Normalizes inputs to [0,1], reconstructs both dates, and computes
    the absolute difference of per-pixel MSE errors.
    """
    model.eval()

    _, c, orig_h, orig_w = t1_tensor.shape
    pad_h = (8 - orig_h % 8) % 8
    pad_w = (8 - orig_w % 8) % 8

    t1 = t1_tensor.to(device) / 255.0
    t2 = t2_tensor.to(device) / 255.0

    if pad_h > 0 or pad_w > 0:
        t1 = F.pad(t1, (0, pad_w, 0, pad_h))
        t2 = F.pad(t2, (0, pad_w, 0, pad_h))

    recon_t1 = model(t1)
    recon_t2 = model(t2)

    if pad_h > 0 or pad_w > 0:
        t1 = t1[:, :, :orig_h, :orig_w]
        t2 = t2[:, :, :orig_h, :orig_w]
        recon_t1 = recon_t1[:, :, :orig_h, :orig_w]
        recon_t2 = recon_t2[:, :, :orig_h, :orig_w]

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

    model, config, normalize = load_model(checkpoint_path, device)
    model_type = config["model"]["type"]
    use_ndvi = config["data"].get("use_ndvi", False)

    t1_tensor, t2_tensor, h, w = preprocess_pair(
        t1_path, t2_path,
        bands=config["data"].get("bands", None),
        use_ndvi=use_ndvi,
    )

    patch_size = config["data"].get("patch_size", 256)

    if model_type == "autoencoder":
        binary_mask, change_map = predict_autoencoder(
            model, t1_tensor, t2_tensor, device,
            change_threshold=config["inference"].get("change_threshold", 0.15),
        )
    else:
        binary_mask, change_map = predict(
            model, t1_tensor, t2_tensor, device,
            threshold=threshold,
            patch_size=patch_size,
            overlap=config["inference"].get("overlap", 32),
            normalize=normalize,
        )

    if output_path:
        change_img = (change_map * 255).astype(np.uint8)
        Image.fromarray(change_img).save(output_path)

    return binary_mask, change_map
