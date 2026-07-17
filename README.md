# Delta

Satellite change detection for urban change in Patna, India using Sentinel-2 imagery and Siamese U-Net.

---

## What it does

Takes two satellite images (before/after) and produces a binary change mask showing where urban development or land-use change occurred. Trained on the OSCD100 dataset and applied to a real-world case study in Patna (Boring Road commercial corridor and Kurji residential area).

---

## Results

### OSCD100 Validation Set (20 samples)

| Model | F1 | ROC AUC | PR AUC | Precision | Recall |
|-------|-----|---------|--------|-----------|--------|
| **Siamese U-Net v3** | **0.56** | **0.84** | **0.56** | 0.46 | 0.72 |
| Siamese U-Net v2 | 0.40 | — | — | 0.52 | 0.33 |
| Siamese U-Net v1 | 0.50 | — | — | 0.82 | 0.36 |
| UNet Baseline | 0.41 | — | — | 0.47 | 0.36 |

**v3 improvements**: ImageNet normalization, AdamW optimizer, calibrated thresholds. Full threshold sweep and ROC/PR curves saved in `outputs/metrics/`.

### Patna Case Study

- **Location**: Boring Road (~25.61°N, 85.105°E) & Kurji (~25.585°N, 85.09°E), Patna, Bihar
- **Dates**: 2022-12-26 → 2024-12-22 (cloud-free, 0.0–0.9% cover)
- **Data**: Sentinel-2 L2A RGB bands (B04, B03, B02) via Microsoft Planetary Computer
- **Limitation**: Domain shift — model trained on OSCD applied to Patna Sentinel-2 (10m resolution)

---

## Quick start

```bash
git clone https://github.com/1046prt/delta.git
cd delta
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Train

```bash
# Best model (v3)
python src/training/train_supervised.py --config configs/train_supervised_v3.yaml

# Baseline
python src/training/train_supervised.py --config configs/train_supervised.yaml
```

### Run inference

```bash
python scripts/patna_case_study.py
```

### Streamlit app

```bash
streamlit run src/app/streamlit_app.py
```

---

## Architecture

```
Siamese U-Net (shared encoder):
  T1 → [Encoder] → |f_T1 - f_T2| → [Decoder] → Change Mask
  T2 → [Encoder]↗
```

- Encoder: 4-level downsampling (Conv3x3-BN-ReLU × 2 + MaxPool)
- Features: [64, 128, 256, 512] channels
- Attention gates in decoder
- ~7.8M parameters

---

## Project structure

```
delta/
├── configs/                      # Training configs (v2, v3, UNet baseline)
├── notebooks/                    # 5 Colab notebooks
├── scripts/
│   ├── evaluate.py               # Evaluation helpers
│   └── patna_case_study.py       # Planetary Computer inference
├── src/
│   ├── models/
│   │   ├── components.py
│   │   ├── unet.py               # UNet + Autoencoder
│   │   └── siamese_unet.py       # SiameseUNet
│   ├── training/
│   │   ├── losses.py             # CombinedLoss, FocalLoss, FocalDiceLoss, etc.
│   │   ├── metrics.py
│   │   ├── logging.py            # ExperimentLogger + optional wandb
│   │   └── train_supervised.py   # Main training script
│   ├── inference/
│   │   └── predict.py            # Inference pipeline
│   ├── data/
│   │   └── ndvi.py
│   └── app/
│       └── streamlit_app.py      # Demo app
├── tests/                        # 35 tests (all passing)
├── outputs/
│   ├── checkpoints/              # Model weights (.gitignored)
│   └── metrics/                  # Eval results (.gitignored)
└── requirements.txt
```

---

## Colab notebooks

| # | Notebook | Link |
|---|----------|------|
| 01 | Data Exploration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1046prt/delta/blob/main/notebooks/01_data_exploration.ipynb) |
| 02 | Supervised Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1046prt/delta/blob/main/notebooks/02_train_supervised.ipynb) |
| 03 | Self-Supervised Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1046prt/delta/blob/main/notebooks/03_train_self_supervised.ipynb) |
| 04 | Patna Case Study | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1046prt/delta/blob/main/notebooks/04_patna_case_study.ipynb) |
| 05 | Evaluation & Results | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1046prt/delta/blob/main/notebooks/05_evaluation.ipynb) |

---

## Limitations

- **Domain shift**: OSCD training data (~0.5m Google Earth) differs from Sentinel-2 (10m)
- **Small dataset**: 60 training samples
- **No cloud masking**: Patna analysis doesn't filter cloud-contaminated pixels
- **Single model**: No ensemble or test-time augmentation
