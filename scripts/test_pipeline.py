import torch
import numpy as np
import sys, os
sys.path.insert(0, ".")
from src.inference.predict import predict, predict_autoencoder, load_model

device = torch.device("cpu")

print("=== Test 1: Load supervised model ===")
model, config = load_model("outputs/checkpoints/siamese_unet_rgb_best.pt", device)
print(f"  Model type: {config['model']['type']}")
print(f"  In channels: {config['model']['in_channels']}")

print("\n=== Test 2: Supervised predict on 256x256 ===")
dummy_t1 = torch.rand(1, 3, 256, 256) * 255.0
dummy_t2 = torch.rand(1, 3, 256, 256) * 255.0
try:
    mask, cmap = predict(model, dummy_t1, dummy_t2, device, threshold=0.5, patch_size=256)
    print(f"  mask: {mask.shape}, sum={mask.sum()}, dtype={mask.dtype}")
    print(f"  cmap: {cmap.shape}, min={cmap.min():.4f}, max={cmap.max():.4f}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n=== Test 3: Supervised predict on 512x512 ===")
dummy_t1l = torch.rand(1, 3, 512, 512) * 255.0
dummy_t2l = torch.rand(1, 3, 512, 512) * 255.0
try:
    mask_l, cmap_l = predict(model, dummy_t1l, dummy_t2l, device, threshold=0.5, patch_size=256, overlap=32)
    print(f"  mask: {mask_l.shape}, sum={mask_l.sum()}")
    print(f"  cmap: {cmap_l.shape}, min={cmap_l.min():.4f}, max={cmap_l.max():.4f}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n=== Test 4: Load autoencoder ===")
model_ae, config_ae = load_model("outputs/checkpoints/autoencoder_rgb_best.pt", device)
print(f"  Model type: {config_ae['model']['type']}")

print("\n=== Test 5: Autoencoder predict on 256x256 ===")
try:
    mask_ae, cmap_ae = predict_autoencoder(model_ae, dummy_t1, dummy_t2, device, change_threshold=0.15)
    print(f"  mask: {mask_ae.shape}, sum={mask_ae.sum()}")
    print(f"  cmap: {cmap_ae.shape}, min={cmap_ae.min():.6f}, max={cmap_ae.max():.6f}")
    print("  PASS")
except Exception as e:
    print(f"  FAIL: {e}")

print("\n=== All tests done ===")
