import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.inference.predict import load_model, predict

st.set_page_config(page_title="Delta — Satellite Change Detection", layout="wide")
st.title("Delta")
st.caption("Sentinel-2 urban change detection with Siamese U-Net")

st.sidebar.header("Configuration")

MODEL_DIR = "outputs/checkpoints"
MODEL_OPTIONS = {
    "Siamese U-Net v3 (Best, F1=0.56)": "siamese_unet_rgb_v3_best.pt",
    "Siamese U-Net v2": "siamese_unet_rgb_v2_best.pt",
    "UNet Baseline (F1=0.41)": "unet_baseline_best.pt",
}

selected_model_name = st.sidebar.selectbox("Model", list(MODEL_OPTIONS.keys()))
threshold = st.sidebar.slider("Threshold", 0.05, 0.9, 0.3, 0.05)
overlay_opacity = st.sidebar.slider("Overlay opacity", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("**v3 results:** F1=0.56, ROC AUC=0.84, PR AUC=0.56 on OSCD100 validation set.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Before (T1)")
    t1_file = st.file_uploader("Upload image", type=["tif", "tiff", "png", "jpg"], key="t1")

with col2:
    st.markdown("### After (T2)")
    t2_file = st.file_uploader("Upload image", type=["tif", "tiff", "png", "jpg"], key="t2")


def load_image(file) -> np.ndarray:
    img = np.array(Image.open(file)).astype(np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[-1] == 4:
        img = img[:, :, :3]
    return img


if t1_file and t2_file:
    t1_img = load_image(t1_file)
    t2_img = load_image(t2_file)

    col1.image(t1_img.astype(np.uint8), caption="Before", use_container_width=True)
    col2.image(t2_img.astype(np.uint8), caption="After", use_container_width=True)

    if st.button("Detect Changes", type="primary"):
        checkpoint_path = os.path.join(MODEL_DIR, MODEL_OPTIONS[selected_model_name])

        if not os.path.exists(checkpoint_path):
            st.error(f"Model not found: {checkpoint_path}")
            st.stop()

        with st.spinner("Running inference..."):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model, config, normalize = load_model(checkpoint_path, device)

            t1_tensor = torch.from_numpy(t1_img[:, :, :3].transpose(2, 0, 1)).unsqueeze(0).float()
            t2_tensor = torch.from_numpy(t2_img[:, :, :3].transpose(2, 0, 1)).unsqueeze(0).float()

            patch_size = config["data"].get("patch_size", 256)
            change_mask, change_map = predict(
                model, t1_tensor, t2_tensor, device,
                threshold=threshold,
                patch_size=patch_size,
                normalize=normalize,
            )

        total_pixels = change_mask.size
        changed_pixels = int(change_mask.sum())
        change_pct = (changed_pixels / total_pixels) * 100
        pixel_size_m = 10
        area_hectares = changed_pixels * (pixel_size_m ** 2) / 10000

        st.markdown("---")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Changed pixels", f"{changed_pixels:,}")
        r2.metric("Change %", f"{change_pct:.2f}%")
        r3.metric("Area", f"{area_hectares:.2f} ha")
        r4.metric("Threshold", f"{threshold:.2f}")

        st.markdown("---")

        vis_col1, vis_col2, vis_col3 = st.columns(3)

        with vis_col1:
            st.markdown("**Before**")
            st.image(t1_img.astype(np.uint8), use_container_width=True)

        with vis_col2:
            st.markdown("**After**")
            st.image(t2_img.astype(np.uint8), use_container_width=True)

        with vis_col3:
            st.markdown("**Detected Changes**")
            t2_norm = np.clip(t2_img / (t2_img.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
            change_rgb = t2_norm.copy()

            change_mask_resized = change_mask
            if change_mask.shape[:2] != change_rgb.shape[:2]:
                change_mask_resized = np.array(Image.fromarray(change_mask * 255).resize(
                    (change_rgb.shape[1], change_rgb.shape[0]), Image.NEAREST
                )) > 127

            change_rgb[change_mask_resized] = [255, 50, 50]
            blended = (t2_norm * (1 - overlay_opacity) + change_rgb * overlay_opacity).astype(np.uint8)
            st.image(blended, use_container_width=True)

        st.markdown("---")

        heat_col, hist_col = st.columns(2)

        with heat_col:
            st.markdown("**Probability heatmap**")
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            im = ax.imshow(change_map, cmap="hot", vmin=0, vmax=1)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            st.image(buf.getvalue(), use_container_width=True)

        with hist_col:
            st.markdown("**Score distribution**")
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            ax.hist(change_map.flatten(), bins=60, color="steelblue", edgecolor="white")
            ax.axvline(x=threshold, color="red", linestyle="--", linewidth=1.5, label=f"t={threshold}")
            ax.set_xlabel("Change probability")
            ax.set_ylabel("Pixel count")
            ax.legend()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
            plt.close(fig)
            st.image(buf.getvalue(), use_container_width=True)

        st.markdown("---")
        st.markdown("**Export results**")

        dl1, dl2, dl3 = st.columns(3)

        with dl1:
            change_img = Image.fromarray(change_mask * 255)
            buf = io.BytesIO()
            change_img.save(buf, format="PNG")
            st.download_button("Binary mask (PNG)", buf.getvalue(),
                               file_name="change_mask.png", mime="image/png")

        with dl2:
            prob_img = Image.fromarray((change_map * 255).astype(np.uint8))
            buf = io.BytesIO()
            prob_img.save(buf, format="PNG")
            st.download_button("Probability map (PNG)", buf.getvalue(),
                               file_name="change_map.png", mime="image/png")

        with dl3:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(t1_img.astype(np.uint8))
            axes[0].set_title("Before")
            axes[0].axis("off")
            axes[1].imshow(t2_img.astype(np.uint8))
            axes[1].set_title("After")
            axes[1].axis("off")
            axes[2].imshow(blended)
            axes[2].set_title("Changes")
            axes[2].axis("off")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            st.download_button("Comparison figure (PNG)", buf.getvalue(),
                               file_name="comparison.png", mime="image/png")

else:
    st.info("Upload a before/after satellite image pair to detect changes.")
    st.markdown("""
    ### Quick start
    1. Select a model from the sidebar
    2. Upload a **before** image (T1) — left upload box
    3. Upload an **after** image (T2) — right upload box
    4. Adjust the threshold slider
    5. Click **Detect Changes**

    ### Models
    | Model | F1 | ROC AUC | Notes |
    |-------|-----|---------|-------|
    | Siamese U-Net v3 | 0.56 | 0.84 | Best — uses ImageNet normalization, AdamW |
    | Siamese U-Net v2 | 0.40 | — | Baseline |

    ### Supported formats
    - GeoTIFF (`.tif`, `.tiff`)
    - RGB images (`.png`, `.jpg`)
    """)
