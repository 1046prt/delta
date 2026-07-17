import os, sys, json, time
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchgeo.datasets import OSCD100
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.inference.predict import load_model


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


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def get_logits(model, t1, t2, model_type, normalize=False):
    if normalize:
        t1 = (t1 / 255.0 - IMAGENET_MEAN.to(t1.device)) / IMAGENET_STD.to(t1.device)
        t2 = (t2 / 255.0 - IMAGENET_MEAN.to(t2.device)) / IMAGENET_STD.to(t2.device)
    if model_type == "autoencoder":
        err_t1 = torch.mean((t1 - model(t1)) ** 2, dim=1, keepdim=True)
        err_t2 = torch.mean((t2 - model(t2)) ** 2, dim=1, keepdim=True)
        return torch.abs(err_t1 - err_t2).squeeze(1)
    else:
        return model(torch.cat([t1, t2], dim=1)).squeeze(1)


def compute_metrics(logits_flat, masks_flat, threshold):
    pred = (torch.sigmoid(logits_flat) > threshold).float()
    tp = (pred * masks_flat).sum().item()
    fp = (pred * (1 - masks_flat)).sum().item()
    fn = ((1 - pred) * masks_flat).sum().item()
    tn = ((1 - pred) * (1 - masks_flat)).sum().item()
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    iou = tp / (tp + fp + fn + 1e-10)
    return {"f1": f1, "iou": iou, "precision": precision, "recall": recall,
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "gt_change": int(masks_flat.sum().item()),
            "pred_change": int(pred.sum().item())}


def main():
    device = torch.device("cpu")
    data_root = "data/oscd100"
    val_dataset = OSCD100(root=data_root, split="val", bands=OSCD100.rgb_bands, download=True)
    val_ds = OSCDRGB(val_dataset)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    print(f"Val samples: {len(val_ds)}")

    models_to_eval = {
        "SiameseUNet v3": "outputs/checkpoints/siamese_unet_rgb_v3_best.pt",
        "SiameseUNet v2": "outputs/checkpoints/siamese_unet_rgb_v2_best.pt",
        "UNet Baseline": "outputs/checkpoints/unet_baseline_best.pt",
        "TinyCD": "outputs/checkpoints/tinycd_best.pt",
        "FC-Siam-diff": "outputs/checkpoints/fc_siam_diff_best.pt",
        "Autoencoder": "outputs/checkpoints/autoencoder_rgb_best.pt",
    }

    all_results = {}

    for name, path in models_to_eval.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        model, config, normalize = load_model(path, device)
        model_type = config["model"]["type"]
        params = sum(p.numel() for p in model.parameters())
        print(f"  Model type: {model_type}, Params: {params:,}")

        all_logits = []
        all_masks = []
        all_probs = []
        sample_f1s = []

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                masks = batch["mask"]
                t1, t2 = images[:, 0], images[:, 1]
                logits = get_logits(model, t1, t2, model_type, normalize=normalize)
                probs = torch.sigmoid(logits).squeeze()
                mask_flat = masks.squeeze()
                all_logits.append(logits.cpu().squeeze())
                all_masks.append(mask_flat.cpu())
                all_probs.append(probs.cpu())
                sample_f1s.append((i, probs.cpu(), mask_flat.cpu()))

        all_logits = torch.cat(all_logits)
        all_masks = torch.cat(all_masks)
        all_probs = torch.cat(all_probs)

        # ROC AUC and PR AUC
        try:
            roc = roc_auc_score(all_masks.numpy().flatten(), all_probs.numpy().flatten())
            pr_auc = average_precision_score(all_masks.numpy().flatten(), all_probs.numpy().flatten())
        except:
            roc, pr_auc = 0.0, 0.0

        # Threshold sweep
        best_f1, best_thresh = 0, 0.5
        best_metrics = {}
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            m = compute_metrics(all_logits, all_masks, thresh)
            if m["f1"] > best_f1:
                best_f1, best_thresh = m["f1"], thresh
                best_metrics = m
            print(f"  t={thresh:.1f}  F1={m['f1']:.4f}  IoU={m['iou']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}")

        print(f"  BEST: F1={best_f1:.4f} @ t={best_thresh}  ROC_AUC={roc:.4f}  PR_AUC={pr_auc:.4f}")

        # Per-sample F1 at best threshold
        per_sample = []
        for idx, probs, mask in sample_f1s:
            pred = (probs > best_thresh).float()
            tp = (pred * mask).sum().item()
            fp = (pred * (1 - mask)).sum().item()
            fn = ((1 - pred) * mask).sum().item()
            p = tp / (tp + fp + 1e-10)
            r = tp / (tp + fn + 1e-10)
            f1 = 2 * p * r / (p + r + 1e-10)
            gt_change = int(mask.sum().item())
            pred_change = int(pred.sum().item())
            per_sample.append({
                "idx": idx, "f1": f1, "precision": p, "recall": r,
                "gt_change": gt_change, "pred_change": pred_change,
                "tp": int(tp), "fp": int(fp), "fn": int(fn)
            })

        # Sort by F1 for best/worst
        per_sample_sorted = sorted(per_sample, key=lambda x: x["f1"])

        print(f"\n  --- Best 5 samples ---")
        for s in per_sample_sorted[-5:]:
            print(f"    Sample {s['idx']}: F1={s['f1']:.4f}  GT={s['gt_change']}px  Pred={s['pred_change']}px")

        print(f"\n  --- Worst 5 samples ---")
        for s in per_sample_sorted[:5]:
            print(f"    Sample {s['idx']}: F1={s['f1']:.4f}  GT={s['gt_change']}px  Pred={s['pred_change']}px")

        all_results[name] = {
            "model_type": model_type,
            "params": params,
            "best_threshold": best_thresh,
            "best_f1": best_f1,
            "best_metrics": best_metrics,
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "per_sample": per_sample,
            "per_sample_worst": per_sample_sorted[:5],
            "per_sample_best": per_sample_sorted[-5:],
        }

        # Free memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # CPU inference benchmark
    print(f"\n{'='*60}")
    print("CPU Inference Benchmark")
    print(f"{'='*60}")

    bench_models = {
        "SiameseUNet v3": "outputs/checkpoints/siamese_unet_rgb_v3_best.pt",
        "UNet Baseline": "outputs/checkpoints/unet_baseline_best.pt",
        "TinyCD": "outputs/checkpoints/tinycd_best.pt",
        "FC-Siam-diff": "outputs/checkpoints/fc_siam_diff_best.pt",
        "Autoencoder": "outputs/checkpoints/autoencoder_rgb_best.pt",
    }

    dummy_t1 = torch.randn(1, 3, 256, 256)
    dummy_t2 = torch.randn(1, 3, 256, 256)

    for name, path in bench_models.items():
        model, config, normalize = load_model(path, device)
        model_type = config["model"]["type"]

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                if model_type == "autoencoder":
                    _ = torch.abs(
                        torch.mean((dummy_t1/255 - model(dummy_t1/255))**2, dim=1) -
                        torch.mean((dummy_t2/255 - model(dummy_t2/255))**2, dim=1)
                    )
                else:
                    _ = model(torch.cat([dummy_t1, dummy_t2], dim=1))

        # Benchmark
        times = []
        for _ in range(20):
            t1 = dummy_t1.clone()
            t2 = dummy_t2.clone()
            start = time.perf_counter()
            with torch.no_grad():
                if model_type == "autoencoder":
                    _ = torch.abs(
                        torch.mean((t1/255 - model(t1/255))**2, dim=1) -
                        torch.mean((t2/255 - model(t2/255))**2, dim=1)
                    )
                else:
                    _ = model(torch.cat([t1, t2], dim=1))
            end = time.perf_counter()
            times.append((end - start) * 1000)

        mean_ms = np.mean(times)
        std_ms = np.std(times)
        min_ms = np.min(times)
        max_ms = np.max(times)
        print(f"  {name:20s}: Mean={mean_ms:.1f}ms  Std={std_ms:.1f}ms  Min={min_ms:.1f}ms  Max={max_ms:.1f}ms")
        all_results[name]["benchmark"] = {
            "mean_ms": round(mean_ms, 1),
            "std_ms": round(std_ms, 1),
            "min_ms": round(min_ms, 1),
            "max_ms": round(max_ms, 1),
        }

        del model

    # Save all results
    os.makedirs("outputs/metrics", exist_ok=True)
    with open("outputs/metrics/paper_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll metrics saved to outputs/metrics/paper_metrics.json")


if __name__ == "__main__":
    main()
