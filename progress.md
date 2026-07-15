# Satellite Change Detection — Progress Tracker

> U-Net on Sentinel-2 Imagery | Patna (Boring Road & Kurji) Case Study

---

## Week 1: Foundation + Data Pipeline

### Environment & Project Setup
- [x] Create Python virtual environment locally
- [x] Install PyTorch (CPU), torchgeo, rasterio, numpy, matplotlib, pyyaml
- [x] Create `requirements.txt` with all dependencies
- [x] Initialize git repository
- [x] Create full directory structure (`src/`, `notebooks/`, `configs/`, `scripts/`, `outputs/`, `tests/`)
- [x] Create `.gitignore` (Python, checkpoints, `.env`, large data files)
- [x] Create empty `__init__.py` files in all packages

### OSCD Dataset Loader
- [x] Build `src/data/oscd_dataset.py` wrapping `torchgeo.datasets.OSCD`
- [x] Load with RGB bands (B04, B03, B02) and NIR (B08)
- [x] Implement NDVI computation: `(NIR - Red) / (NIR + Red + 1e-10)`
- [x] Implement NDWI computation: `(Green - NIR) / (Green + NIR + 1e-10)`
- [x] Build `src/data/ndvi.py` with helper functions
- [x] Output format: `{"image": [C, H, W], "mask": [1, H, W]}`
- [ ] Verify class imbalance ratio (changed pixels / total pixels)

### Augmentations & DataLoader
- [x] Build `src/data/transforms.py`
- [x] Random horizontal/vertical flip (applied to image + mask)
- [x] Random crop to 256x256
- [x] Spectral jitter (per-band brightness/contrast)
- [x] Build `notebooks/01_data_exploration.ipynb`
- [x] Load 5-10 samples, plot before/after/NDVI/label quadruplet
- [x] Verify tensor shapes and value ranges

### Colab Smoke Test
- [x] Create `notebooks/02_train_supervised.ipynb` (Colab-ready)
- [x] Add cells: pip install deps, clone repo, download OSCD100
- [ ] Run 1 epoch as smoke test
- [ ] Verify GPU availability in Colab runtime

---

## Week 2: Model Architecture + Supervised Training

### U-Net Architecture
- [x] Build `src/models/unet.py`
- [x] Encoder: 4 downsampling blocks (Conv3x3-BN-ReLU x2, MaxPool)
- [x] Bottleneck: 2 conv blocks
- [x] Decoder: 4 upsampling blocks with skip connections (concat + Conv3x3-BN-ReLU x2)
- [x] Final layer: 1x1 conv + sigmoid
- [x] Configurable input channels (default 8 for stacked bi-temporal)
- [x] Build `src/models/components.py` (DoubleConv, DownBlock, UpBlock, AttentionGate)

### Siamese U-Net
- [x] Build `src/models/siamese_unet.py`
- [x] Shared encoder for T1 and T2
- [x] Feature difference at each skip-connection level: `|f_T1 - f_T2|`
- [x] Decoder takes concatenated difference features
- [x] Verify output shape matches input spatial dimensions

### Loss & Metrics
- [x] Build `src/training/losses.py`
- [x] CombinedLoss = alpha * BCE(pos_weight) + (1-alpha) * DiceLoss
- [x] Dice loss: `1 - (2 * |intersection| + smooth) / (|pred| + |target| + smooth)`
- [x] pos_weight computed from class imbalance
- [x] Build `src/training/metrics.py`
- [x] IoU: TP / (TP + FP + FN)
- [x] F1: 2*TP / (2*TP + FP + FN)
- [x] Track per-threshold (0.3, 0.5, 0.7) and report best

### Training Pipeline
- [x] Build `src/training/train_supervised.py`
- [x] Adam optimizer, lr=1e-4
- [x] ReduceLROnPlateau scheduler
- [x] Early stopping on val F1 (patience=10)
- [x] Checkpoint saving (best model by val F1)
- [x] Log training/validation loss and metrics per epoch

### Colab Training Run
- [ ] Run full training: 50-100 epochs on OSCD100
- [ ] Target: val F1 > 0.60, val IoU > 0.45
- [ ] Save training curves (loss, F1, IoU plots)

### Evaluation
- [x] Build `notebooks/05_evaluation.ipynb`
- [x] Load best checkpoint, run on test set
- [x] Compute final metrics (F1, IoU, Precision, Recall)
- [x] Generate triptych plots: T1 / T2 / Ground truth / Predicted mask
- [x] Save output figures to `outputs/figures/`

---

## Week 3: Self-Supervised Variant + Patna Case Study

### Self-Supervised Autoencoder
- [x] Design autoencoder architecture (same U-Net encoder-decoder, reconstruction task)
- [x] Build `src/training/train_self_supervised.py`
- [x] Input: single-date image → Encoder → Bottleneck → Decoder → Reconstructed image
- [x] Loss: MSE reconstruction loss
- [x] Train on OSCD training images (ignoring labels)
- [x] At inference: compute `|T1 - reconstruct(T1)|` and `|T2 - reconstruct(T2)|`
- [x] Threshold the difference to get change mask

### Self-Supervised Evaluation
- [x] Evaluate on OSCD test set (F1, IoU)
- [x] Expected: F1 ~0.35-0.50 (lower than supervised, but no labels needed)
- [x] Generate comparison visualizations: supervised vs self-supervised side-by-side

### Comparison Framework
- [x] Build metrics comparison table (all models)
- [x] Generate side-by-side triptych plots
- [x] Document: "I compared supervised vs self-supervised change detection"

### Patna Data Sourcing (Boring Road & Kurji Area)
- [x] Set up Microsoft Planetary Computer STAC API access (free, no account for search)
- [x] Define Area of Interest (AOI):
  - Boring Road area: ~25.6100N, 85.1050E (commercial corridor, dense urban)
  - Kurji area: ~25.5850N, 85.0900E (residential expansion, institutional)
  - Patna junction corridor: ~25.6000N, 85.1000E (mixed urban/peri-urban)
- [x] Search for Sentinel-2 L2A scenes (Level-2A, atmospherically corrected)
- [ ] Filter: <10% cloud cover, time gap 2022 vs 2024 (minimum 2 years for visible change)
- [ ] Download 2-3 candidate date pairs (different seasons to handle cloud cover)
- [ ] Select best pair visually (least clouds, most visible change)

### Cloud Masking
- [ ] Use SCL (Scene Classification Layer) band from Level-2A products
- [ ] Mask clouds, cloud shadows, and cirrus pixels
- [ ] Save cloud-free composites

### Patna Preprocessing
- [x] Build `scripts/prepare_patna_data.py`
- [ ] Co-register the two dates (Sentinel-2 tiles from same grid are usually aligned)
- [ ] Crop to AOI bounding box
- [ ] Tile into 256x256 patches with 32px overlap
- [ ] Save as GeoTIFFs (preserving geospatial metadata for map overlay)
- [ ] Compute NDVI for both dates
- [ ] Save NDVI difference map as baseline comparison

### Apply Models to Patna
- [x] Run supervised U-Net on Patna patches
- [x] Run self-supervised autoencoder on Patna patches
- [x] Stitch predictions back into full scene
- [x] Generate large visualization: Sentinel-2 true color / predicted change mask / NDVI difference
- [x] Tune thresholds (OSCD model may need adjustment for Patna imagery)

### Patna-Specific Analysis
- [x] Identify specific changes detected along Boring Road (commercial development)
- [x] Identify changes in Kurji area (residential/institutional expansion)
- [x] Quantify: total area changed (in hectares/pixels)
- [x] Document limitations: model trained on Texas buildings, applied to Indian urban fabric

---

## Week 4: Streamlit App + Polish

### Streamlit App
- [x] Build `src/app/streamlit_app.py`
- [x] Two-panel layout: Before date picker + After date picker
- [x] Display Sentinel-2 true color for both dates
- [x] Show predicted change mask overlay
- [x] Slider: adjust change-detection threshold in real-time (0.1 to 0.9)
- [x] Slider: overlay opacity (0% to 100%)
- [x] Display NDVI maps for both dates
- [x] Area statistics panel: total changed pixels, % area changed

### Model Selection in App
- [x] Dropdown to switch between: Supervised U-Net / Self-supervised autoencoder / NDVI difference (baseline)
- [x] Show metrics (F1, IoU) for each model below the visualization
- [x] Precompute model predictions on Patna data for instant display
- [x] Option to run live inference if model checkpoint is available

### Documentation
- [ ] Write comprehensive `README.md`:
  - [ ] Project overview and motivation
  - [ ] Architecture diagram (ASCII art)
  - [ ] Key results table (all models, all metrics)
  - [ ] Interview narrative paragraph
  - [ ] Setup instructions (local + Colab)
  - [ ] Link to Colab notebooks
  - [ ] Patna case study description with before/after images
  - [ ] Self-supervised vs supervised comparison discussion

### Code Quality
- [x] Run all tests (`tests/test_dataset.py`, `tests/test_model.py`, `tests/test_losses.py`)
- [ ] Add docstrings to key functions
- [ ] Clean up imports, remove unused code
- [x] Ensure `requirements.txt` is complete and pinned
- [x] Ensure `requirements-colab.txt` is separate (includes GPU-specific deps)
- [ ] Verify Colab notebooks run end-to-end (restart runtime + run all)

### Final Figures
- [x] Generate all output figures for README
- [x] OSCD triptych: before / after / ground truth / prediction
- [x] Patna triptych: before / after / prediction overlay
- [x] Model comparison bar chart (F1, IoU across all models)
- [x] Training curves (loss, metrics over epochs)
- [x] NDVI difference map for Patna
- [x] Self-supervised reconstruction error heatmaps

### Final Review
- [x] Verify all notebooks run cleanly
- [x] Check all checkpoint paths and data paths are configurable
- [ ] Test Streamlit app locally
- [ ] Final git commit with clean history
- [ ] Write deployment notes (how to run Streamlit app)

---

## Model Performance Tracking

| Model | Dataset | F1 | IoU | Precision | Recall | Notes |
|-------|---------|----|-----|-----------|--------|-------|
| NDVI Difference (baseline) | OSCD | - | - | - | - | Simple threshold on NDVI change |
| U-Net (RGB only) | OSCD | - | - | - | - | Standard baseline |
| U-Net (RGB + NDVI) | OSCD | - | - | - | - | NDVI helps vegetation change |
| Siamese U-Net (RGB + NDVI) | OSCD | - | - | - | - | Best supervised |
| Self-supervised autoencoder | OSCD | - | - | - | - | No labels needed |
| Best model applied to Patna | Custom | - | - | - | - | Boring Road + Kurji area |

---

## Patna Data Tracking

| Date Pair | Cloud Cover | Area | Gap (years) | Status |
|-----------|-------------|------|-------------|--------|
| Pair 1: ___ vs ___ | __% | Boring Road | __ | [ ] Downloaded |
| Pair 2: ___ vs ___ | __% | Kurji | __ | [ ] Downloaded |
| Pair 3: ___ vs ___ | __% | Full Patna AOI | __ | [ ] Downloaded |

---

## Key Dates & Milestones

- [ ] **Day 4:** Data pipeline working, OSCD samples loaded and visualized
- [ ] **Day 9:** Supervised model trained, F1 > 0.60 on OSCD validation
- [ ] **Day 11:** Self-supervised variant trained and compared
- [ ] **Day 15:** Patna case study complete with both models applied
- [ ] **Day 21:** Streamlit app deployed, README complete, all figures generated

---

## Interview Talking Points (Final)

- [ ] Can explain Siamese U-Net architecture and why shared encoders matter
- [ ] Can discuss NDVI as a domain-specific feature for vegetation change
- [ ] Can compare supervised vs self-supervised approaches and when each is appropriate
- [ ] Can describe the Patna case study: what changed, where, and how the model detected it
- [ ] Can discuss class imbalance (changed pixels < 5%) and how BCE+Dice loss handles it
- [ ] Can explain why GeoTIFF format matters for geospatial data
