# Delta: Satellite Change Detection with U-Net

> Detecting urban change in Patna (Boring Road & Kurji), Bihar, India using Sentinel-2 imagery and Siamese U-Net architecture.

---

## Overview

This project implements **satellite change detection** using deep learning to identify urban expansion and land-use changes between two dates of Sentinel-2 multispectral imagery. Two approaches are compared:

1. **Supervised Siamese U-Net** — trained on labeled change masks from the OSCD100 dataset
2. **Self-Supervised Autoencoder** — learns to reconstruct input images; detects change via reconstruction error

Both models are applied to a **real-world case study** in Patna, India (Boring Road commercial corridor and Kurji residential area) using free Sentinel-2 data from Microsoft Planetary Computer.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SIAMESE U-NET                         │
│                                                         │
│  T1 Image ──→ ┌──────────┐                              │
│               │ Shared   │──→ |f_T1 - f_T2| ──→ Decoder │
│  T2 Image ──→ │ Encoder  │         │              │     │
│               └──────────┘    skip connections     │     │
│                                                     ↓     │
│                                              Change Mask  │
│                                              (1, 256, 256)│
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│               SELF-SUPERVISED AUTOENCODER                │
│                                                         │
│  T1 Image ──→ Encoder ──→ Bottleneck ──→ Decoder ──→ T1̂ │
│  T2 Image ──→ Encoder ──→ Bottleneck ──→ Decoder ──→ T2̂ │
│                                                         │
│  Change = |MSE(T1, T1̂) - MSE(T2, T2̂)| > threshold     │
└─────────────────────────────────────────────────────────┘
```

- **Encoder**: 4-level downsampling (Conv3x3-BN-ReLU × 2 + MaxPool)
- **Features**: [64, 128, 256, 512] channels
- **Attention gates** in decoder (supervised model)
- **Shared weights** in Siamese variant for both temporal inputs
- ~7.8M parameters (supervised), ~24M (autoencoder)

---

## Results

### OSCD100 Validation Set (20 samples)

| Model | F1 | IoU | Precision | Recall | Best Threshold |
|-------|------|-------|-----------|--------|----------------|
| **Siamese U-Net (Supervised)** | **0.5023** | **0.3354** | **0.8228** | 0.3615 | 0.7 |
| Self-Supervised Autoencoder | 0.0767 | 0.0405 | 0.0399 | 1.0000 | — |

- Supervised model trained for 12/50 epochs (early stopped at patience=10)
- Autoencoder converged on reconstruction (val loss 36.77) but reconstruction-error-based change detection is weak with limited training data

### Patna Case Study

- **Location**: Boring Road (~25.61°N, 85.105°E) & Kurji (~25.585°N, 85.09°E), Patna, Bihar
- **Dates**: 2022-12-26 → 2024-12-22 (cloud-free, 0.0–0.9% cover)
- **Data**: Sentinel-2 L2A RGB bands (B04, B03, B02) via Microsoft Planetary Computer
- **Detection**: Supervised model detects ~19.7% change in the AOI
- **Limitation**: Domain shift — model trained on OSCD (Texas, ~0.5m Google Earth) applied to Patna Sentinel-2 (10m)

---

## Project Structure

```
delta/
├── configs/
│   ├── train_supervised.yaml
│   └── train_self_supervised.yaml
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_train_supervised.ipynb
│   ├── 03_train_self_supervised.ipynb
│   ├── 04_patna_case_study.ipynb
│   └── 05_evaluation.ipynb
├── scripts/
│   ├── evaluate.py
│   └── patna_case_study.py
├── src/
│   ├── models/
│   │   ├── components.py      # DoubleConv, DownBlock, UpBlock, AttentionGate
│   │   ├── unet.py            # UNet + Autoencoder
│   │   └── siamese_unet.py    # SiameseUNet
│   ├── training/
│   │   ├── losses.py           # CombinedLoss (BCE + Dice)
│   │   ├── metrics.py          # F1, IoU, Precision, Recall
│   │   ├── train_supervised.py
│   │   └── train_self_supervised.py
│   ├── inference/
│   │   ├── predict.py          # Inference pipeline
│   │   └── visualize.py        # Visualization helpers
│   ├── data/
│   │   └── ndvi.py             # NDVI computation
│   └── app/
│       └── streamlit_app.py    # Web app
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_losses.py
│   └── ...
├── outputs/
│   ├── checkpoints/            # Model weights (.gitignored)
│   ├── metrics/                # Training logs (.gitignored)
│   └── patna/                  # Case study outputs
├── requirements.txt
└── README.md
```

---

## Setup

### Local (CPU)

```bash
git clone https://github.com/1046prt/delta.git
cd delta
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

### Train

```bash
# Supervised (Siamese U-Net)
python src/training/train_supervised.py --config configs/train_supervised.yaml

# Self-Supervised (Autoencoder)
python src/training/train_self_supervised.py --config configs/train_self_supervised.yaml
```

### Evaluate

```bash
python scripts/evaluate.py
```

### Patna Case Study

```bash
python scripts/patna_case_study.py
```

### Streamlit App

```bash
streamlit run src/app/streamlit_app.py
```

---

## Colab Notebooks

| Notebook | Description | Link |
|----------|-------------|------|
| 01 | Data Exploration | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1046prt/delta/blob/main/notebooks/01_data_exploration.ipynb) |
| 02 | Supervised Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1046prt/delta/blob/main/notebooks/02_train_supervised.ipynb) |
| 03 | Self-Supervised Training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1046prt/delta/blob/main/notebooks/03_train_self_supervised.ipynb) |
| 04 | Patna Case Study | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1046prt/delta/blob/main/notebooks/04_patna_case_study.ipynb) |
| 05 | Evaluation & Results | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1046prt/delta/blob/main/notebooks/05_evaluation.ipynb) |

---

## Key Technical Decisions

1. **Siamese architecture** with shared encoder ensures both dates are processed identically; feature difference `|f_T1 - f_T2|` captures temporal change
2. **Combined BCE + Dice loss** handles severe class imbalance (changed pixels < 5% of total)
3. **RGB-only input** (3 channels) chosen over RGB+NIR to reduce complexity; model has ~7.8M parameters
4. **Attention gates** in decoder focus on relevant spatial features
5. **Self-supervised autoencoder** as comparison baseline — no labels needed, detects any reconstruction anomaly
6. **OSCD100 dataset** (60 train / 20 val) from torchgeo for training; applied to Patna for generalization testing

---

## Interview Narrative

> "I built a satellite change detection system comparing supervised and self-supervised deep learning approaches. The supervised Siamese U-Net achieved F1 of 0.50 on the OSCD benchmark using a shared encoder with attention gates and combined BCE+Dice loss to handle class imbalance. The self-supervised autoencoder provides a label-free alternative — though with lower accuracy, it's valuable when annotated data is scarce. I applied both models to a real-world case study in Patna, India, detecting urban expansion along the Boring Road commercial corridor using free Sentinel-2 imagery from Microsoft Planetary Computer."

---

## Dataset

**OSCD100** (Onera Satellite Change Detection) — 100 pairs of registered multispectral satellite images from multiple cities worldwide. Contains binary change masks for urban development changes.

- 60 training / 20 validation / 20 test pairs
- 256×256 patches
- RGB bands (B04, B03, B02) at 10m resolution

---

## Limitations & Future Work

- **Domain shift**: OSCD training data (~0.5m Google Earth) differs significantly from Sentinel-2 (10m), limiting transfer to Patna
- **Small dataset**: Only 60 training samples constrains model capacity and generalization
- **No cloud masking**: Patna analysis doesn't filter cloud-contaminated pixels
- **Single model**: No ensemble or test-time augmentation

Future improvements: fine-tuning on Patna-specific data, incorporating NIR bands, cloud masking via SCL layer, and temporal compositing.
