# Reproducing Results

This document describes how to reproduce all results from scratch.

## Prerequisites

- Python 3.10+ (tested on 3.12)
- 16GB RAM recommended
- No GPU required (all training runs on CPU)

## Setup

```bash
git clone https://github.com/1046prt/delta.git
cd delta
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux/Mac
pip install -r requirements.txt
```

## Dataset

OSCD100 is auto-downloaded by torchgeo on first run:

```bash
python -c "from torchgeo.datasets import OSCD100; OSCD100(root='data/oscd100', split='train', bands=OSCD100.rgb_bands, download=True)"
```

## Training

### Siamese U-Net v3 (Best — F1=0.56)

```bash
python src/training/train_supervised.py --config configs/train_supervised_v3.yaml
```

- ~2 min/epoch on CPU, 48/80 epochs before early stop
- Total time: ~1.5 hours
- Checkpoint: `outputs/checkpoints/siamese_unet_rgb_v3_best.pt`

### Siamese U-Net v2 (Baseline)

```bash
python src/training/train_supervised.py --config configs/train_supervised.yaml
```

### UNet Baseline

```bash
python src/training/train_supervised.py --config configs/train_unet_baseline.yaml
```

## Evaluation

```bash
# Full evaluation with ROC/PR curves
python scripts/evaluate_full.py --checkpoint outputs/checkpoints/siamese_unet_rgb_v3_best.pt

# Paper results table + ONNX export + benchmarks
python scripts/paper_tables.py
```

## Visualizations

```bash
# Architecture diagram
python scripts/plot_architecture.py

# Qualitative comparison + failure analysis
python scripts/plot_qualitative.py
```

## Patna Case Study

```bash
python scripts/patna_case_study.py
```

Requires internet access (downloads Sentinel-2 data from Planetary Computer).

## Streamlit Demo

```bash
streamlit run src/app/streamlit_app.py
```

## Tests

```bash
python -m pytest tests/ -v
```

## Expected Results

| Model | F1 | ROC AUC | Params |
|-------|-----|---------|--------|
| Siamese U-Net v3 | 0.5607 | 0.8378 | 7.8M |
| Siamese U-Net v2 | 0.4004 | — | 7.8M |
| UNet Baseline | 0.4101 | — | 7.7M |
