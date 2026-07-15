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
from src.models.unet import Autoencoder
from src.training.metrics import compute_iou, compute_f1


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


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

        t1 = images[:, 0]
        t2 = images[:, 1]

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            recon_t1 = model(t1)
            recon_t2 = model(t2)
            loss = criterion(recon_t1, t1) + criterion(recon_t2, t2)

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
def validate(model, loader, criterion, device, change_threshold=0.15):
    model.eval()
    total_loss = 0.0
    total_f1 = 0.0
    total_iou = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        t1 = images[:, 0]
        t2 = images[:, 1]

        recon_t1 = model(t1)
        recon_t2 = model(t2)
        loss = criterion(recon_t1, t1) + criterion(recon_t2, t2)
        total_loss += loss.item()

        error_t1 = torch.mean((t1 - recon_t1) ** 2, dim=1, keepdim=True)
        error_t2 = torch.mean((t2 - recon_t2) ** 2, dim=1, keepdim=True)
        change_map = torch.abs(error_t1 - error_t2)

        change_pred = (change_map > change_threshold).float()

        total_f1 += compute_f1(change_pred, masks)
        total_iou += compute_iou(change_pred, masks)
        n_batches += 1

    return (
        total_loss / max(n_batches, 1),
        total_f1 / max(n_batches, 1),
        total_iou / max(n_batches, 1),
    )


def train_self_supervised(config_path: str):
    config = load_config(config_path)
    exp_name = config["experiment"]["name"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(config["experiment"]["seed"])
    np.random.seed(config["experiment"]["seed"])

    model_cfg = config["model"]
    model = Autoencoder(
        in_channels=model_cfg["in_channels"],
        features=model_cfg["features"],
    ).to(device)

    train_loader, val_loader = build_dataloaders(config)

    criterion = nn.MSELoss()

    opt_cfg = config["optimizer"]
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt_cfg["lr"],
        weight_decay=opt_cfg.get("weight_decay", 0),
    )

    sched_cfg = config["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
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

    best_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_iou": []}

    change_threshold = config["inference"]["change_threshold"]

    print(f"\nStarting self-supervised training: {exp_name}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(train_cfg["max_epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp)
        val_loss, val_f1, val_iou = validate(model, val_loader, criterion, device, change_threshold)

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch+1:3d}/{train_cfg['max_epochs']} | "
            f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"Change F1: {val_f1:.4f} | IoU: {val_iou:.4f} | "
            f"LR: {lr:.2e} | Time: {elapsed:.1f}s"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_iou"].append(val_iou)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(ckpt_dir, f"{exp_name}_best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
                "config": config,
            }, ckpt_path)
            print(f"  -> Saved best model (Loss: {best_loss:.6f})")
        else:
            patience_counter += 1

        if patience_counter >= train_cfg["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    history_path = os.path.join(log_dir, f"{exp_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best reconstruction loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_self_supervised.yaml")
    args = parser.parse_args()
    train_self_supervised(args.config)
