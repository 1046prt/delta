import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds

from src.models.siamese_unet import SiameseUNet

PATNA_AOI = [85.08, 25.57, 85.12, 25.62]
OUT_DIR = "outputs/patna"
os.makedirs(OUT_DIR, exist_ok=True)


def load_band(item, band_name, bbox, target_shape=(256, 256)):
    href = item.assets[band_name].href
    with rasterio.open(href) as src:
        window = from_bounds(bbox[0], bbox[1], bbox[2], bbox[3], src.transform)
        data = src.read(1, window=window).astype(np.float32)
    data = np.array(Image.fromarray(data).resize(target_shape, Image.BILINEAR))
    return data


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Connecting to Microsoft Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    print("Searching for T1 (2022) scenes...")
    search_t1 = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=PATNA_AOI,
        datetime="2022-01-01/2022-12-31",
        query={"eo:cloud_cover": {"lt": 10}},
        limit=10,
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
    )
    t1_items = list(search_t1.items())
    print(f"  Found {len(t1_items)} scenes")

    print("Searching for T2 (2024) scenes...")
    search_t2 = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=PATNA_AOI,
        datetime="2024-01-01/2024-12-31",
        query={"eo:cloud_cover": {"lt": 10}},
        limit=10,
        sortby=[{"field": "eo:cloud_cover", "direction": "asc"}],
    )
    t2_items = list(search_t2.items())
    print(f"  Found {len(t2_items)} scenes")

    if not t1_items or not t2_items:
        print("ERROR: No scenes found")
        return

    best_t1 = t1_items[0]
    best_t2 = t2_items[0]
    print(f"\nT1: {best_t1.datetime.strftime('%Y-%m-%d')} (cloud: {best_t1.properties['eo:cloud_cover']:.1f}%)")
    print(f"T2: {best_t2.datetime.strftime('%Y-%m-%d')} (cloud: {best_t2.properties['eo:cloud_cover']:.1f}%)")

    bands_config = ["B04", "B03", "B02"]
    print("\nLoading bands...")
    t1_bands = {}
    t2_bands = {}
    for band in bands_config:
        print(f"  {band}...")
        t1_bands[band] = load_band(best_t1, band, PATNA_AOI)
        t2_bands[band] = load_band(best_t2, band, PATNA_AOI)

    t1_rgb = np.stack([t1_bands["B04"], t1_bands["B03"], t1_bands["B02"]], axis=-1)
    t2_rgb = np.stack([t2_bands["B04"], t2_bands["B03"], t2_bands["B02"]], axis=-1)

    t1_rgb_vis = np.clip(t1_rgb / 3000, 0, 1)
    t2_rgb_vis = np.clip(t2_rgb / 3000, 0, 1)

    # --- Try multiple normalization strategies ---
    ckpt_path = "outputs/checkpoints/siamese_unet_rgb_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model_sup = SiameseUNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"].get("out_channels", 1),
        features=cfg["model"]["features"],
        use_attention=cfg["model"].get("use_attention", False),
    )
    model_sup.load_state_dict(ckpt["model_state_dict"])
    model_sup.to(device).eval()
    print("Model loaded")

    # OSCD data is uint8 [0,255]. Sentinel-2 is uint16 [0,10000].
    # Scale Sentinel-2 to match: clip to reasonable range, then scale to 0-255
    t1_scaled = np.clip(t1_rgb / 10000.0 * 255.0, 0, 255).astype(np.float32)
    t2_scaled = np.clip(t2_rgb / 10000.0 * 255.0, 0, 255).astype(np.float32)
    t1_255 = torch.from_numpy(t1_scaled.transpose(2, 0, 1)).unsqueeze(0)
    t2_255 = torch.from_numpy(t2_scaled.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        stacked = torch.cat([t1_255, t2_255], dim=1).to(device)
        pred = torch.sigmoid(model_sup(stacked)).squeeze().cpu().numpy()

    print(f"\nPrediction stats (Sentinel-2 scaled to 0-255):")
    print(f"  Min: {pred.min():.6f}, Max: {pred.max():.6f}, Mean: {pred.mean():.6f}")
    print(f"  Percentiles: 50%={np.percentile(pred, 50):.6f}, 90%={np.percentile(pred, 90):.6f}, 95%={np.percentile(pred, 95):.6f}, 99%={np.percentile(pred, 99):.6f}")

    # Try thresholds
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        mask = (pred > thr).astype(np.uint8)
        print(f"  threshold={thr:.1f}: {mask.sum()} px ({mask.mean()*100:.2f}%)")

    # Use a lower threshold since model predictions are compressed on out-of-domain data
    # The model was trained on OSCD (~0.5m Google Earth), not Sentinel-2 (10m)
    threshold = 0.22
    pred_binary = (pred > threshold).astype(np.uint8)
    pct = pred_binary.mean() * 100
    print(f"\nUsing threshold={threshold}: {pred_binary.sum()} px ({pct:.1f}%)")
    print(f"NOTE: Low confidence due to domain shift (OSCD -> Patna Sentinel-2)")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes[0, 0].imshow(t1_rgb_vis)
    axes[0, 0].set_title(f"Before ({best_t1.datetime.strftime('%Y-%m-%d')})")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(t2_rgb_vis)
    axes[0, 1].set_title(f"After ({best_t2.datetime.strftime('%Y-%m-%d')})")
    axes[0, 1].axis("off")
    axes[0, 2].imshow(pred, cmap="hot", vmin=0, vmax=1)
    axes[0, 2].set_title("Change Probability Map")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Raw Prediction")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(t2_rgb_vis, alpha=0.6)
    change_rgb = np.zeros((*pred_binary.shape, 3))
    change_rgb[pred_binary > 0] = [1, 0, 0]
    axes[1, 1].imshow(change_rgb, alpha=0.5)
    axes[1, 1].set_title(f"Change Overlay ({pct:.1f}%)")
    axes[1, 1].axis("off")

    hist = pred.flatten()
    axes[1, 2].hist(hist, bins=50, color="steelblue", edgecolor="white")
    axes[1, 2].axvline(x=threshold, color="red", linestyle="--", label=f"Threshold={threshold}")
    axes[1, 2].set_title("Prediction Distribution")
    axes[1, 2].set_xlabel("Change Probability")
    axes[1, 2].set_ylabel("Pixel Count")
    axes[1, 2].legend()

    plt.suptitle(f"Patna Change Detection — {best_t1.datetime.strftime('%Y-%m-%d')} to {best_t2.datetime.strftime('%Y-%m-%d')}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "patna_supervised_result.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved patna_supervised_result.png")

    pixel_size_m = 10
    hectare_per_pixel = (pixel_size_m ** 2) / 10000
    changed = pred_binary.sum()
    area = changed * hectare_per_pixel
    print(f"\nStatistics (threshold={threshold}):")
    print(f"  Changed pixels: {changed:,}")
    print(f"  Estimated area: {area:.2f} hectares")
    print(f"  Percentage: {pct:.2f}%")


if __name__ == "__main__":
    main()
