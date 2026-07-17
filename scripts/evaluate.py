import os
import sys
import json
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchgeo.datasets import OSCD100
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.inference.predict import load_model
from src.training.metrics import evaluate_all


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


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


@torch.no_grad()
def get_logits(model, t1, t2, model_type, normalize=False):
    if normalize:
        t1 = (t1 / 255.0 - IMAGENET_MEAN.to(t1.device)) / IMAGENET_STD.to(t1.device)
        t2 = (t2 / 255.0 - IMAGENET_MEAN.to(t2.device)) / IMAGENET_STD.to(t2.device)

    if model_type == "autoencoder":
        recon_t1 = model(t1)
        recon_t2 = model(t2)
        err_t1 = torch.mean((t1 - recon_t1) ** 2, dim=1, keepdim=True)
        err_t2 = torch.mean((t2 - recon_t2) ** 2, dim=1, keepdim=True)
        return torch.abs(err_t1 - err_t2).squeeze(1)
    else:
        stacked = torch.cat([t1, t2], dim=1)
        return model(stacked).squeeze(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="outputs/checkpoints/siamese_unet_rgb_best.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, config, normalize = load_model(args.model, device)
    model_type = config["model"]["type"]
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_type} ({params:,} params)")

    data_root = "data/oscd100"
    val_dataset = OSCD100(root=data_root, split="val", bands=OSCD100.rgb_bands, download=True)
    val_ds = OSCDRGB(val_dataset)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    print(f"Val samples: {len(val_ds)}")

    all_logits = []
    all_masks = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            t1 = images[:, 0]
            t2 = images[:, 1]

            logits = get_logits(model, t1, t2, model_type, normalize=normalize)
            all_logits.append(logits.cpu())
            all_masks.append(masks.squeeze(1).cpu())

    all_logits = torch.cat(all_logits)
    all_masks = torch.cat(all_masks)

    print(f"\nLogits: {all_logits.shape}, Masks: {all_masks.shape}")
    print(f"\n{'Threshold':<12} {'F1':>8} {'IoU':>8} {'Precision':>10} {'Recall':>8}")
    print("-" * 50)

    best_f1 = 0
    best_thresh = 0.5
    results = {}
    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        m = evaluate_all(all_logits, all_masks, threshold=thresh)
        results[thresh] = m
        print(f"{thresh:<12.1f} {m['f1']:>8.4f} {m['iou']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f}")
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = thresh

    print(f"\nBest: F1={best_f1:.4f} @ threshold={best_thresh}")

    best = results[best_thresh]
    print(f"Best metrics: F1={best['f1']:.4f}, IoU={best['iou']:.4f}, Precision={best['precision']:.4f}, Recall={best['recall']:.4f}")

    out_path = args.model.replace("checkpoints", "metrics").replace(".pt", "_eval.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "model": model_type,
            "model_path": args.model,
            "params": params,
            "best_threshold": best_thresh,
            "best_metrics": best,
            "all_results": {str(k): v for k, v in results.items()},
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
