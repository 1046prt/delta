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

from src.inference.predict import load_model, predict

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

    ckpt_path = "outputs/checkpoints/siamese_unet_rgb_v3_best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "outputs/checkpoints/siamese_unet_rgb_v2_best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "outputs/checkpoints/siamese_unet_rgb_best.pt"

    print(f"\nLoading model from: {ckpt_path}")
    model, config, normalize = load_model(ckpt_path, device)
    print(f"Model loaded (normalize={normalize})")

    t1_scaled = np.clip(t1_rgb / 10000.0 * 255.0, 0, 255).astype(np.float32)
    t2_scaled = np.clip(t2_rgb / 10000.0 * 255.0, 0, 255).astype(np.float32)
    t1_tensor = torch.from_numpy(t1_scaled.transpose(2, 0, 1)).unsqueeze(0)
    t2_tensor = torch.from_numpy(t2_scaled.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        if normalize:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            t1_norm = (t1_tensor / 255.0 - mean) / std
            t2_norm = (t2_tensor / 255.0 - mean) / std
            stacked = torch.cat([t1_norm, t2_norm], dim=1).to(device)
        else:
            stacked = torch.cat([t1_tensor, t2_tensor], dim=1).to(device)

        pred = torch.sigmoid(model(stacked)).squeeze().cpu().numpy()

    print(f"\nPrediction stats:")
    print(f"  Min: {pred.min():.4f}, Max: {pred.max():.4f}, Mean: {pred.mean():.4f}")
    print(f"  Percentiles: 50%={np.percentile(pred, 50):.4f}, 90%={np.percentile(pred, 90):.4f}, 95%={np.percentile(pred, 95):.4f}, 99%={np.percentile(pred, 99):.4f}")

    for thr in [0.1, 0.2, 0.3, 0.4, 0.5]:
        mask = (pred > thr).astype(np.uint8)
        print(f"  threshold={thr:.1f}: {mask.sum()} px ({mask.mean()*100:.2f}%)")

    threshold = 0.3
    pred_binary = (pred > threshold).astype(np.uint8)
    pct = pred_binary.mean() * 100
    print(f"\nUsing threshold={threshold}: {pred_binary.sum()} px ({pct:.1f}%)")

    pixel_size_m = 10
    hectare_per_pixel = (pixel_size_m ** 2) / 10000
    changed = pred_binary.sum()
    area = changed * hectare_per_pixel

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(t1_rgb_vis)
    axes[0, 0].set_title(f"Before ({best_t1.datetime.strftime('%Y-%m-%d')})", fontsize=12)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(t2_rgb_vis)
    axes[0, 1].set_title(f"After ({best_t2.datetime.strftime('%Y-%m-%d')})", fontsize=12)
    axes[0, 1].axis("off")

    im = axes[0, 2].imshow(pred, cmap="hot", vmin=0, vmax=1)
    axes[0, 2].set_title("Change Probability", fontsize=12)
    axes[0, 2].axis("off")
    plt.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(pred_binary, cmap="gray")
    axes[1, 0].set_title(f"Binary Mask (t={threshold})", fontsize=12)
    axes[1, 0].axis("off")

    overlay = (t2_rgb_vis * 255).astype(np.uint8).copy()
    overlay[pred_binary > 0] = [255, 0, 0]
    blended = (t2_rgb_vis * 0.6 + (overlay / 255.0) * 0.4)
    blended = np.clip(blended, 0, 1)
    axes[1, 1].imshow(blended)
    axes[1, 1].set_title(f"Change Overlay ({pct:.1f}%)", fontsize=12)
    axes[1, 1].axis("off")

    axes[1, 2].hist(pred.flatten(), bins=50, color="steelblue", edgecolor="white")
    axes[1, 2].axvline(x=threshold, color="red", linestyle="--", label=f"Threshold={threshold}")
    axes[1, 2].set_title("Prediction Distribution", fontsize=12)
    axes[1, 2].set_xlabel("Change Probability")
    axes[1, 2].set_ylabel("Pixel Count")
    axes[1, 2].legend()

    plt.suptitle(
        f"Patna Change Detection: {best_t1.datetime.strftime('%Y-%m-%d')} to {best_t2.datetime.strftime('%Y-%m-%d')}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    result_path = os.path.join(OUT_DIR, "patna_result.png")
    plt.savefig(result_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {result_path}")

    np.save(os.path.join(OUT_DIR, "change_map.npy"), pred)
    Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, "change_map.png"))

    print(f"\n{'='*50}")
    print(f"PATNA CASE STUDY RESULTS")
    print(f"{'='*50}")
    print(f"  Location: Boring Road & Kurji, Patna")
    print(f"  T1: {best_t1.datetime.strftime('%Y-%m-%d')}")
    print(f"  T2: {best_t2.datetime.strftime('%Y-%m-%d')}")
    print(f"  Model: {os.path.basename(ckpt_path)}")
    print(f"  Threshold: {threshold}")
    print(f"  Changed pixels: {changed:,}")
    print(f"  Change area: {area:.2f} hectares")
    print(f"  Change percentage: {pct:.2f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
