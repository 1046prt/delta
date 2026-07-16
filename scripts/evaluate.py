import os
import sys
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchgeo.datasets import OSCD100
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.siamese_unet import SiameseUNet
from src.models.unet import Autoencoder
from src.training.metrics import evaluate_all, compute_f1, compute_iou


class OSCDRGB(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"].float()
        mask = sample["mask"].float()
        if image.dim() == 4:
            t1, t2 = image[0], image[1]
        else:
            t1 = t2 = image
        return {"image": torch.stack([t1, t2], dim=0), "mask": mask, "t1": t1, "t2": t2}


def load_supervised(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SiameseUNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"].get("out_channels", 1),
        features=cfg["model"]["features"],
        use_attention=cfg["model"].get("use_attention", False),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def load_autoencoder(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = Autoencoder(in_channels=cfg["model"]["in_channels"], features=cfg["model"]["features"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = "data/oscd100"
    val_dataset = OSCD100(root=data_root, split="val", bands=OSCD100.rgb_bands, download=True)
    val_ds = OSCDRGB(val_dataset)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    print(f"Val samples: {len(val_ds)}")

    model_sup = load_supervised("outputs/checkpoints/siamese_unet_rgb_best.pt", device)
    print(f"Supervised params: {sum(p.numel() for p in model_sup.parameters()):,}")

    all_preds_sup = []
    all_masks = []
    all_t2 = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"]
            stacked = torch.cat([images[:, 0], images[:, 1]], dim=1)
            pred_sup = torch.sigmoid(model_sup(stacked))
            all_preds_sup.append(pred_sup.cpu())
            all_masks.append(masks)
            all_t2.append(batch["t2"])

    all_preds_sup = torch.cat(all_preds_sup)
    all_masks = torch.cat(all_masks)

    print(f"\nPredictions: {all_preds_sup.shape}, Masks: {all_masks.shape}")
    print(f"\n{'Threshold':<12} {'F1':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 50)

    best_f1 = 0
    best_thresh = 0.5
    results = {}
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        m = evaluate_all(all_preds_sup, all_masks, threshold=thresh)
        results[thresh] = m
        print(f"{thresh:<12.1f} {m['f1']:>8.4f} {m['iou']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = thresh

    print(f"\nBest: F1={best_f1:.4f} @ threshold={best_thresh}")

    best = results[best_thresh]
    print(f"\nBest metrics: F1={best['f1']:.4f}, IoU={best['iou']:.4f}, Precision={best['precision']:.4f}, Recall={best['recall']:.4f}")

    with open("outputs/metrics/evaluation_results.json", "w") as f:
        json.dump({
            "best_threshold": best_thresh,
            "best_metrics": best,
            "all_results": {str(k): v for k, v in results.items()},
        }, f, indent=2)
    print("\nResults saved to outputs/metrics/evaluation_results.json")


if __name__ == "__main__":
    main()
