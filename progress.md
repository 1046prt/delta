# Delta — Progress Log

## Training History

### Supervised v1 (siamese_unet_rgb_best.pt)
- 12/50 epochs, early stopped (patience=10)
- F1=0.5023 @ threshold=0.7
- Config: Adam, lr=0.001, CombinedLoss, no normalization

### Supervised v2 (siamese_unet_rgb_v2_best.pt)
- 52/80 epochs, early stopped (patience=10)
- F1=0.4004 @ threshold=0.75
- Config: Adam, lr=0.0005, FocalDiceLoss, cosine schedule

### Supervised v3 (siamese_unet_rgb_v3_best.pt) — BEST
- 48/80 epochs, early stopped (patience=10)
- **F1=0.5607 @ threshold=0.4**
- **ROC AUC=0.8378, PR AUC=0.5551**
- Config: ImageNet normalization, AdamW (weight_decay=0.0001), FocalDiceLoss, cosine schedule

### Self-Supervised Autoencoder (autoencoder_rgb_best.pt)
- 37/50 epochs
- F1=0.0767 (useless — predicts nothing)
- Fundamentally limited for change detection

### UNet Baseline (unet_baseline_best.pt)
- 20/20 epochs
- **F1=0.4101 @ threshold=0.75**
- Shows Siamese architecture (+37% F1) is worth the complexity

## Key Fixes Applied
- ImageNet normalization in training + inference (config-driven)
- Adam optimizer replaced by AdamW (weight_decay=0.0001)
- EMA added but disabled (hurt convergence on 60 samples)
- Evaluation double-sigmoid bug fixed
- Patna case study switched to Planetary Computer API (free)
- UNet baseline support added to training + predict.py
- Bug fix: UNet uses in_ch*2 internally (config stores per-date channels)

## Paper Readiness
- 35 tests passing (unit + inference)
- CI/CD pipeline (GitHub Actions)
- REPRODUCE.md (step-by-step)
- Dockerfile (reproducible environment)
- Architecture diagram (scripts/plot_architecture.py)
- Qualitative comparison figure (scripts/plot_qualitative.py)
- Failure case analysis (scripts/plot_qualitative.py)
- GradCAM visualization (scripts/plot_gradcam.py)
- Paper tables + benchmarks (scripts/paper_tables.py)
- Logging module with wandb support (src/training/logging.py)
- Missing metrics: OA, Kappa (scripts/paper_tables.py)
- LaTeX paper skeleton (paper/main.tex)
- Improved Streamlit app with export
- Polished README with results table
