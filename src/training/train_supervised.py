import os
import sys
import json
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.oscd_dataset import OSCDDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.unet import UNet
from src.models.siamese_unet import SiameseUNet
from src.training.losses import CombinedLoss, compute_pos_weight
from src.training.metrics import evaluate_all


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(config: dict) -> nn.Module:
    model_type = config["model"]["type"]
    in_ch = config["model"]["in_channels"]
    out_ch = config["model"]["out_channels"]
    features = config["model"]["features"]
    use_attn = config["model"].get("use_attention", False)

    if model_type == "siamese_unet":
        return SiameseUNet(in_ch, out_ch, features, use_attention=use_attn)
    elif model_type == "unet":
        return UNet(in_ch, out_ch, features, use_attention=use_attn)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def build_dataloaders(config: dict):
    data_cfg = config["data"]

    train_ds = OSCDDataset(
        root=data_cfg["root"], split="train",
        bands=data_cfg["bands"], patch_size=data_cfg["patch_size"],
        use_ndvi=data_cfg.get("use_ndvi", True),
        transforms=get_train_transforms(data_cfg["patch_size"]),
    )
    val_ds = OSCDDataset(
        root=data_cfg["root"], split="train",
        bands=data_cfg["bands"], patch_size=data_cfg["patch_size"],
        use_ndvi=data_cfg.get("use_ndvi", True),
        transforms=get_val_transforms(data_cfg["patch_size"]),
    )

    n_train = int(len(train_ds) * 0.85)
    n_val = len(train_ds) - n_train
    train_ds, val_ds = random_split(train_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=data_cfg["batch_size"],
        shuffle=True, num_workers=data_cfg["num_workers"],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg["batch_size"],
        shuffle=False, num_workers=data_cfg["num_workers"],
        pin_memory=True,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            preds = model(images)
            loss = criterion(preds, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, thresholds=[0.3, 0.5, 0.7]):
    model.eval()
    total_loss = 0.0
    all_metrics = {t: {"iou": 0, "f1": 0, "precision": 0, "recall": 0} for t in thresholds}
    n_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        preds = model(images)
        loss = criterion(preds, masks)
        total_loss += loss.item()

        for t in thresholds:
            metrics = evaluate_all(preds, masks, threshold=t)
            for k in all_metrics[t]:
                all_metrics[t][k] += metrics[k]

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    avg_metrics = {}
    for t in thresholds:
        avg_metrics[t] = {k: v / max(n_batches, 1) for k, v in all_metrics[t].items()}

    return avg_loss, avg_metrics


def train(config_path: str):
    config = load_config(config_path)
    exp_name = config["experiment"]["name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(config["experiment"]["seed"])
    np.random.seed(config["experiment"]["seed"])

    model = build_model(config).to(device)
    train_loader, val_loader = build_dataloaders(config)

    all_labels = []
    for batch in train_loader:
        all_labels.append(batch["mask"])
    all_labels = torch.cat(all_labels, dim=0)
    pos_weight = compute_pos_weight(all_labels).to(device)

    loss_cfg = config["loss"]
    criterion = CombinedLoss(
        alpha=loss_cfg["alpha"],
        pos_weight=pos_weight,
        smooth=loss_cfg.get("smooth", 1.0),
    )

    opt_cfg = config["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 0),
    )

    sched_cfg = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=sched_cfg["factor"],
        patience=sched_cfg["patience"],
        min_lr=sched_cfg["min_lr"],
    )

    train_cfg = config["training"]
    use_amp = train_cfg.get("mixed_precision", False) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    ckpt_dir = train_cfg["checkpoint_dir"]
    log_dir = train_cfg["log_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    best_f1 = 0.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_metrics": []}

    print(f"\nStarting training: {exp_name}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(train_cfg["max_epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_metrics = validate(model, val_loader, criterion, device, config["evaluation"]["thresholds"])

        best_threshold = config["evaluation"]["thresholds"][0]
        best_thresh_f1 = 0
        for t, metrics in val_metrics.items():
            if metrics["f1"] > best_thresh_f1:
                best_thresh_f1 = metrics["f1"]
                best_threshold = t

        scheduler.step(best_thresh_f1)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d}/{train_cfg['max_epochs']} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Best F1@{best_threshold}: {best_thresh_f1:.4f} | "
            f"IoU@{best_threshold}: {val_metrics[best_threshold]['iou']:.4f} | "
            f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_metrics"].append({str(t): m for t, m in val_metrics.items()})

        if best_thresh_f1 > best_f1:
            best_f1 = best_thresh_f1
            patience_counter = 0
            ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_f1": best_f1,
                "config": config,
            }, ckpt_path)
            print(f"  -> Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= train_cfg["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
            break

    history_path = os.path.join(log_dir, f"{exp_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_supervised.yaml")
    args = parser.parse_args()
    train(args.config)
