import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.inference.predict import load_model, predict, predict_autoencoder, preprocess_pair
from src.data.ndvi import compute_ndvi

st.set_page_config(page_title="Satellite Change Detection", layout="wide")
st.title("Satellite Change Detection")
st.subheader("Sentinel-2 Multi-Spectral Imagery | U-Net Architecture")

st.sidebar.header("Configuration")

MODEL_DIR = "outputs/checkpoints"
MODEL_OPTIONS = {
    "Siamese U-Net (Supervised)": "oscd_unet_rgb_ndvi_best.pt",
    "Self-Supervised Autoencoder": "oscd_self_supervised_best.pt",
}

selected_model_name = st.sidebar.selectbox("Select Model", list(MODEL_OPTIONS.keys()))
threshold = st.sidebar.slider("Change Threshold", 0.1, 0.9, 0.5, 0.05)
overlay_opacity = st.sidebar.slider("Overlay Opacity", 0.0, 1.0, 0.5, 0.05)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Before (T1)")
    t1_file = st.file_uploader("Upload T1 image", type=["tif", "tiff", "png", "jpg"], key="t1")

with col2:
    st.markdown("### After (T2)")
    t2_file = st.file_uploader("Upload T2 image", type=["tif", "tiff", "png", "jpg"], key="t2")

if t1_file and t2_file:
    t1_img = np.array(Image.open(t1_file)).astype(np.float32)
    if t1_img.ndim == 2:
        t1_img = np.stack([t1_img] * 3, axis=-1)
    elif t1_img.shape[-1] == 4:
        t1_img = t1_img[:, :, :3]

    t2_img = np.array(Image.open(t2_file)).astype(np.float32)
    if t2_img.ndim == 2:
        t2_img = np.stack([t2_img] * 3, axis=-1)
    elif t2_img.shape[-1] == 4:
        t2_img = t2_img[:, :, :3]

    col1.image(t1_img.astype(np.uint8), caption="Before image")
    col2.image(t2_img.astype(np.uint8), caption="After image")

    if st.button("Detect Changes", type="primary"):
        checkpoint_path = os.path.join(MODEL_DIR, MODEL_OPTIONS[selected_model_name])

        with st.spinner("Loading model..."):
            if os.path.exists(checkpoint_path):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model, config = load_model(checkpoint_path, device)
                use_ndvi = config["data"].get("use_ndvi", True)

                t1_tensor, t2_tensor, h, w = preprocess_pair(
                    t1_file.name, t2_file.name,
                    bands=config["data"]["bands"],
                    use_ndvi=use_ndvi,
                )

                if config["model"]["type"] == "autoencoder":
                    change_mask, change_map = predict_autoencoder(
                        model, t1_tensor, t2_tensor, device,
                        change_threshold=config["inference"]["change_threshold"],
                    )
                else:
                    change_mask, change_map = predict(
                        model, t1_tensor, t2_tensor, device,
                        threshold=threshold,
                        patch_size=config["data"]["patch_size"],
                    )
            else:
                st.warning("No trained model found. Please train a model first.")
                change_map = np.random.rand(min(t1_img.shape[0], 256), min(t1_img.shape[1], 256)).astype(np.float32)
                change_mask = (change_map > threshold).astype(np.uint8)

        st.markdown("---")
        st.markdown("### Results")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.markdown("**Before (T1)**")
            st.image(t1_img.astype(np.uint8), use_container_width=True)

        with result_col2:
            st.markdown("**After (T2)**")
            st.image(t2_img.astype(np.uint8), use_container_width=True)

        with result_col3:
            st.markdown("**Detected Changes**")
            overlay = (t2_img[:, :, :3] / t2_img[:, :, :3].max() * 255).astype(np.uint8) if t2_img.max() > 0 else t2_img.astype(np.uint8)
            change_rgb = overlay.copy()
            if change_mask.shape[:2] != change_rgb.shape[:2]:
                change_mask_resized = np.array(Image.fromarray(change_mask * 255).resize(
                    (change_rgb.shape[1], change_rgb.shape[0]), Image.NEAREST
                )) > 127
            else:
                change_mask_resized = change_mask > 0

            change_rgb[change_mask_resized] = [255, 0, 0]
            blended = (overlay * (1 - overlay_opacity) + change_rgb * overlay_opacity).astype(np.uint8)
            st.image(blended, use_container_width=True)

        st.markdown("---")
        stats_col1, stats_col2, stats_col3 = st.columns(3)

        total_pixels = change_mask.size
        changed_pixels = int(change_mask.sum())
        change_pct = (changed_pixels / total_pixels) * 100

        with stats_col1:
            st.metric("Total Pixels", f"{total_pixels:,}")
        with stats_col2:
            st.metric("Changed Pixels", f"{changed_pixels:,}")
        with stats_col3:
            st.metric("Change Area", f"{change_pct:.2f}%")

        st.markdown("---")
        st.markdown("### Change Probability Heatmap")
        st.image(change_map, caption="Change probability (red = high, blue = low)", use_container_width=True)

        if t1_img.shape[2] >= 4:
            st.markdown("---")
            st.markdown("### NDVI Comparison")
            nir_t1 = t1_img[:, :, 3]
            red_t1 = t1_img[:, :, 0]
            nir_t2 = t2_img[:, :, 3]
            red_t2 = t2_img[:, :, 0]

            ndvi_t1 = compute_ndvi(nir_t1, red_t1)
            ndvi_t2 = compute_ndvi(nir_t2, red_t2)

            ndvi_col1, ndvi_col2, ndvi_col3 = st.columns(3)
            with ndvi_col1:
                st.image(ndvi_t1, caption="NDVI Before", use_container_width=True, clamp=True)
            with ndvi_col2:
                st.image(ndvi_t2, caption="NDVI After", use_container_width=True, clamp=True)
            with ndvi_col3:
                ndvi_diff = ndvi_t2 - ndvi_t1
                st.image(ndvi_diff, caption="NDVI Change", use_container_width=True, clamp=True)

else:
    st.info("Upload before (T1) and after (T2) satellite images to detect changes.")
    st.markdown("""
    ### How to use
    1. Select a trained model from the sidebar
    2. Upload a "before" satellite image (T1)
    3. Upload an "after" satellite image (T2)
    4. Adjust the change detection threshold
    5. Click **Detect Changes**

    ### Models
    - **Siamese U-Net (Supervised)**: Trained on OSCD dataset with labeled change masks
    - **Self-Supervised Autoencoder**: Trained to reconstruct same-date images; detects anomalies on different-date pairs
    """)
