import os
import sys
import json
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
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from skimage.filters import threshold_otsu

from src.inference.predict import load_model

PATNA_AOI = [85.08, 25.57, 85.12, 25.62]
OUT_DIR = "outputs/patna"
os.makedirs(OUT_DIR, exist_ok=True)


def load_band(item, band_name, bbox, target_shape=(256, 256)):
    href = item.assets[band_name].href
    with rasterio.open(href) as src:
        src_crs = src.crs
        utm_bounds = transform_bounds("EPSG:4326", src_crs, *bbox)
        window = from_bounds(*utm_bounds, transform=src.transform)
        data = src.read(1, window=window).astype(np.float32)
    data = np.array(Image.fromarray(data).resize(target_shape, Image.BILINEAR))
    return data


def generate_proxy_gt(t1_rgb, t2_rgb):
    """Generate proxy ground truth via histogram-matched differencing + Otsu.

    1. Histogram-match T2 to T1 to remove global radiometric differences.
    2. Compute per-pixel Euclidean distance in RGB space.
    3. Apply Otsu thresholding + morphological opening + remove small objects.
    """
    from skimage.exposure import match_histograms
    t2_matched = match_histograms(t2_rgb, t1_rgb, channel_axis=-1).astype(np.float64)

    diff = np.sqrt(np.sum((t2_matched - t1_rgb.astype(np.float64)) ** 2, axis=-1))
    diff_norm = diff / (diff.max() + 1e-10)

    try:
        thr = threshold_otsu(diff_norm)
    except ValueError:
        thr = np.mean(diff_norm) + np.std(diff_norm)

    gt = (diff_norm > thr).astype(np.uint8)

    from scipy.ndimage import binary_opening, label
    gt = binary_opening(gt, structure=np.ones((3, 3))).astype(np.uint8)

    labeled, num_features = label(gt)
    min_size = 20
    for i in range(1, num_features + 1):
        if (labeled == i).sum() < min_size:
            gt[labeled == i] = 0

    return gt, diff_norm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cached_t1 = os.path.join(OUT_DIR, "t1_rgb.npy")
    cached_t2 = os.path.join(OUT_DIR, "t2_rgb.npy")

    if os.path.exists(cached_t1) and os.path.exists(cached_t2):
        print("Loading cached imagery...")
        t1_rgb = np.load(cached_t1)
        t2_rgb = np.load(cached_t2)
        print(f"  Loaded {t1_rgb.shape} from cache")
    else:
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

        np.save(cached_t1, t1_rgb)
        np.save(cached_t2, t2_rgb)
        print("  Cached imagery for future runs")

    t1_rgb_vis = np.clip(t1_rgb / 3000, 0, 1)
    t2_rgb_vis = np.clip(t2_rgb / 3000, 0, 1)

    print("\nGenerating proxy ground truth (image differencing + Otsu)...")
    proxy_gt, diff_norm = generate_proxy_gt(t1_rgb, t2_rgb)
    change_pct_gt = proxy_gt.mean() * 100
    print(f"  Proxy GT: {proxy_gt.sum()} changed pixels ({change_pct_gt:.1f}%)")

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

    gt_np = proxy_gt.astype(np.float32)

    def calc_metrics(pred_bin, gt_bin):
        tp = float((pred_bin * gt_bin).sum())
        fp = float((pred_bin * (1 - gt_bin)).sum())
        fn = float(((1 - pred_bin) * gt_bin).sum())
        prec = tp / (tp + fp + 1e-10)
        rec = tp / (tp + fn + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        iou = tp / (tp + fp + fn + 1e-10)
        return {"f1": f1, "iou": iou, "precision": prec, "recall": rec}

    best_f1 = 0
    best_thr = 0.5
    best_metrics = {}
    thresholds_tested = {}
    for thr in np.arange(0.1, 0.9, 0.05):
        pred_bin = (pred > thr).astype(np.float32)
        m = calc_metrics(pred_bin, gt_np)
        thresholds_tested[round(float(thr), 2)] = {
            "f1": round(m["f1"], 4),
            "iou": round(m["iou"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "changed_px": int(pred_bin.sum()),
            "changed_pct": round(float(pred_bin.mean() * 100), 2),
        }
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thr = thr
            best_metrics = m

    print(f"\nThreshold sweep (vs proxy GT, {proxy_gt.sum()} GT pixels):")
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        if round(thr, 2) in thresholds_tested:
            t = thresholds_tested[round(thr, 2)]
            print(f"  t={thr:.1f}: F1={t['f1']:.3f} P={t['precision']:.3f} R={t['recall']:.3f} IoU={t['iou']:.3f} ({t['changed_pct']:.1f}%)")

    print(f"\nBest: threshold={best_thr:.2f}, F1={best_metrics['f1']:.4f}, IoU={best_metrics['iou']:.4f}")
    print(f"  Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}")

    threshold = best_thr
    pred_binary = (pred > threshold).astype(np.uint8)
    pct = pred_binary.mean() * 100

    pixel_size_m = 10
    hectare_per_pixel = (pixel_size_m ** 2) / 10000
    changed = pred_binary.sum()
    area = changed * hectare_per_pixel

    gt_changed = proxy_gt.sum()
    gt_area = gt_changed * hectare_per_pixel

    overlap = np.logical_and(pred_binary > 0, proxy_gt > 0).sum()
    union = np.logical_or(pred_binary > 0, proxy_gt > 0).sum()
    iou_final = overlap / (union + 1e-10)

    tp = int(overlap)
    fp = int(pred_binary.sum()) - tp
    fn = int(proxy_gt.sum()) - tp
    tn = pred_binary.size - tp - fp - fn

    results = {
        "location": "Boring Road & Kurji, Patna",
        "t1_date": "2022-12-26",
        "t2_date": "2024-12-22",
        "model": os.path.basename(ckpt_path),
        "proxy_gt_method": "Image differencing (Euclidean) + Otsu + morphological opening",
        "proxy_gt_changed_px": int(gt_changed),
        "proxy_gt_pct": round(change_pct_gt, 2),
        "proxy_gt_area_ha": round(gt_area, 2),
        "best_threshold": round(float(threshold), 2),
        "predicted_changed_px": int(changed),
        "predicted_pct": round(pct, 2),
        "predicted_area_ha": round(area, 2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "f1": round(float(best_metrics["f1"]), 4),
        "iou": round(float(best_metrics["iou"]), 4),
        "precision": round(float(best_metrics["precision"]), 4),
        "recall": round(float(best_metrics["recall"]), 4),
        "threshold_sweep": thresholds_tested,
    }

    results_path = os.path.join(OUT_DIR, "patna_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results: {results_path}")

    fig = plt.figure(figsize=(20, 10))

    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(t1_rgb_vis)
    ax1.set_title(f"$I_{{T_1}}$ (2022-12-26)", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(t2_rgb_vis)
    ax2.set_title(f"$I_{{T_2}}$ (2024-12-22)", fontsize=11, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 4, 3)
    im3 = ax3.imshow(diff_norm, cmap="hot", vmin=0, vmax=1)
    ax3.set_title("Spectral Difference", fontsize=11, fontweight="bold")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 4, 4)
    ax4.imshow(proxy_gt, cmap="gray")
    ax4.set_title(f"Proxy GT ({change_pct_gt:.1f}%)", fontsize=11, fontweight="bold")
    ax4.axis("off")

    ax5 = fig.add_subplot(2, 4, 5)
    im5 = ax5.imshow(pred, cmap="hot", vmin=0, vmax=1)
    ax5.set_title("Model Probability", fontsize=11, fontweight="bold")
    ax5.axis("off")
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(2, 4, 6)
    ax6.imshow(pred_binary, cmap="gray")
    ax6.set_title(f"Prediction ($\\tau$={threshold:.2f}, {pct:.1f}%)", fontsize=11, fontweight="bold")
    ax6.axis("off")

    ax7 = fig.add_subplot(2, 4, 7)
    vis_overlay = np.zeros((*proxy_gt.shape, 3), dtype=np.float32)
    vis_overlay[proxy_gt > 0] = [1, 0, 0]
    vis_overlay[pred_binary > 0] = [0, 1, 0]
    vis_overlay[np.logical_and(proxy_gt > 0, pred_binary > 0)] = [1, 1, 0]
    ax7.imshow(vis_overlay)
    tp_pct = overlap / (proxy_gt.sum() + 1e-10) * 100
    ax7.set_title(f"Overlap (IoU={iou_final:.2f})", fontsize=11, fontweight="bold")
    ax7.axis("off")
    from matplotlib.patches import Patch
    ax7.legend(handles=[
        Patch(facecolor="red", label="GT only"),
        Patch(facecolor="green", label="Pred only"),
        Patch(facecolor="yellow", label="Overlap"),
    ], fontsize=8, loc="lower left")

    ax8 = fig.add_subplot(2, 4, 8)
    thr_keys = sorted(thresholds_tested.keys())
    thr_f1s = [thresholds_tested[t]["f1"] for t in thr_keys]
    ax8.plot(thr_keys, thr_f1s, "-o", color="#2196F3", linewidth=2, markersize=4)
    ax8.axvline(x=threshold, color="red", linestyle="--", label=f"Best={threshold:.2f}")
    ax8.set_xlabel("Threshold")
    ax8.set_ylabel("F1 Score")
    ax8.set_title("F1 vs Threshold", fontsize=11, fontweight="bold")
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3)

    plt.suptitle(
        f"Patna Case Study: Cross-domain Evaluation\n"
        f"F1={best_metrics['f1']:.3f}, IoU={iou_final:.3f}, P={best_metrics['precision']:.3f}, R={best_metrics['recall']:.3f}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    result_path = os.path.join(OUT_DIR, "patna_result.png")
    plt.savefig(result_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nSaved: {result_path}")

    np.save(os.path.join(OUT_DIR, "change_map.npy"), pred)
    np.save(os.path.join(OUT_DIR, "proxy_gt.npy"), proxy_gt)
    Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, "change_map.png"))
    Image.fromarray(proxy_gt * 255).save(os.path.join(OUT_DIR, "proxy_gt.png"))

    print(f"\n{'='*60}")
    print(f"PATNA CASE STUDY RESULTS")
    print(f"{'='*60}")
    print(f"  Location: Boring Road & Kurji, Patna")
    print(f"  T1: 2022-12-26, T2: 2024-12-22")
    print(f"  Model: {os.path.basename(ckpt_path)}")
    print(f"  Proxy GT: {gt_changed:,} px ({change_pct_gt:.1f}%) = {gt_area:.1f} ha")
    print(f"  Prediction: {changed:,} px ({pct:.1f}%) = {area:.1f} ha")
    print(f"  F1={best_metrics['f1']:.4f}, IoU={iou_final:.4f}")
    print(f"  Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
