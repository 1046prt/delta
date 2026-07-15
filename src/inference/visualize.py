import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap


def plot_triptych(
    t1_image: np.ndarray,
    t2_image: np.ndarray,
    change_mask: np.ndarray,
    ground_truth: np.ndarray | None = None,
    title: str = "Change Detection",
    save_path: str | None = None,
    dpi: int = 150,
):
    if ground_truth is not None:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def to_rgb(img):
        if img.ndim == 3 and img.shape[0] <= 4:
            if img.shape[0] >= 3:
                rgb = img[:3].transpose(1, 2, 0)
                rgb = np.clip(rgb, 0, 1) if rgb.max() <= 1.0 else np.clip(rgb / 255.0, 0, 1)
                return rgb
        if img.ndim == 2:
            return img
        return img.transpose(1, 2, 0)

    axes[0].imshow(to_rgb(t1_image))
    axes[0].set_title("Before (T1)", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(to_rgb(t2_image))
    axes[1].set_title("After (T2)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(to_rgb(t2_image), alpha=0.6)
    change_overlay = np.zeros((*change_mask.shape, 3))
    change_overlay[change_mask > 0] = [1, 0, 0]
    axes[2].imshow(change_overlay, alpha=0.5)
    axes[2].set_title("Predicted Change", fontsize=12)
    axes[2].axis("off")

    red_patch = mpatches.Patch(color='red', label='Changed')
    axes[2].legend(handles=[red_patch], loc='lower right', fontsize=9)

    if ground_truth is not None:
        axes[3].imshow(to_rgb(t2_image), alpha=0.6)
        gt_overlay = np.zeros((*ground_truth.shape, 3))
        gt_overlay[ground_truth > 0] = [0, 1, 0]
        axes[3].imshow(gt_overlay, alpha=0.5)
        axes[3].set_title("Ground Truth", fontsize=12)
        axes[3].axis("off")
        green_patch = mpatches.Patch(color='green', label='True Change')
        axes[3].legend(handles=[green_patch], loc='lower right', fontsize=9)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_change_heatmap(
    change_map: np.ndarray,
    title: str = "Change Probability Heatmap",
    save_path: str | None = None,
    dpi: int = 150,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(change_map, cmap='hot', vmin=0, vmax=1)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_ndvi_comparison(
    ndvi_t1: np.ndarray,
    ndvi_t2: np.ndarray,
    title: str = "NDVI Comparison",
    save_path: str | None = None,
    dpi: int = 150,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = axes[0].imshow(ndvi_t1, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[0].set_title("NDVI - Before", fontsize=12)
    axes[0].axis("off")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(ndvi_t2, cmap='RdYlGn', vmin=-1, vmax=1)
    axes[1].set_title("NDVI - After", fontsize=12)
    axes[1].axis("off")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    ndvi_diff = ndvi_t2 - ndvi_t1
    im3 = axes[2].imshow(ndvi_diff, cmap='RdYlGn', vmin=-0.5, vmax=0.5)
    axes[2].set_title("NDVI Change", fontsize=12)
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_training_curves(
    history_path: str,
    save_path: str | None = None,
    dpi: int = 150,
):
    with open(history_path, "r") as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)
    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss", fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "val_metrics" in history and history["val_metrics"]:
        f1_vals = []
        iou_vals = []
        for entry in history["val_metrics"]:
            if isinstance(entry, dict):
                best_f1 = max(float(v.get("f1", 0)) for v in entry.values() if isinstance(v, dict))
                best_iou = max(float(v.get("iou", 0)) for v in entry.values() if isinstance(v, dict))
                f1_vals.append(best_f1)
                iou_vals.append(best_iou)

        if f1_vals:
            axes[1].plot(epochs[:len(f1_vals)], f1_vals, label="F1 Score", linewidth=2, color='tab:blue')
            axes[1].plot(epochs[:len(iou_vals)], iou_vals, label="IoU", linewidth=2, color='tab:orange')
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Score")
            axes[1].set_title("Validation Metrics", fontsize=12, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
    elif "val_f1" in history:
        axes[1].plot(epochs, history["val_f1"], label="F1 Score", linewidth=2, color='tab:blue')
        axes[1].plot(epochs, history["val_iou"], label="IoU", linewidth=2, color='tab:orange')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Validation Metrics", fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
