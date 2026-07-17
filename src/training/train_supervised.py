import os
import sys
import json
import time
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from torchgeo.datasets import OSCD100
from src.models.siamese_unet import SiameseUNet
from src.models.unet import UNet
from src.training.losses import FocalDiceLoss, FocalTverskyLoss, BoundaryLoss, compute_pos_weight
from src.training.metrics import evaluate_all


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            self.shadow[name].data = self.decay * self.shadow[name].data + (1 - self.decay) * param.data

    def apply_shadow(self):
        self.backup = {name: param.clone() for name, param in self.model.named_parameters()}
        for name, param in self.model.named_parameters():
            param.data = self.shadow[name].data

    def restore(self):
        for name, param in self.model.named_parameters():
            param.data = self.backup[name].data
        del self.backup


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class OSCDDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, augment=False, normalize=False):
        self.dataset = dataset
        self.augment = augment
        self.normalize = normalize

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

        if self.augment:
            if torch.rand(1).item() > 0.5:
                t1 = torch.flip(t1, [-1])
                t2 = torch.flip(t2, [-1])
                mask = torch.flip(mask, [-1])
            if torch.rand(1).item() > 0.5:
                t1 = torch.flip(t1, [-2])
                t2 = torch.flip(t2, [-2])
                mask = torch.flip(mask, [-2])
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                t1 = torch.rot90(t1, k, [-2, -1]).contiguous()
                t2 = torch.rot90(t2, k, [-2, -1]).contiguous()
                mask = torch.rot90(mask, k, [-2, -1]).contiguous()
            if torch.rand(1).item() > 0.5:
                brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.4
                t1 = t1 * brightness
                t2 = t2 * brightness
                t1 = t1.clamp(0, 255)
                t2 = t2.clamp(0, 255)
            if torch.rand(1).item() > 0.5:
                noise_t1 = torch.randn_like(t1) * 5.0
                noise_t2 = torch.randn_like(t2) * 5.0
                t1 = (t1 + noise_t1).clamp(0, 255)
                t2 = (t2 + noise_t2).clamp(0, 255)
            if torch.rand(1).item() > 0.7:
                c = torch.randint(0, 3, (1,)).item()
                t1[c] = 0
                t2[c] = 0

        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
            t1 = (t1 / 255.0 - mean) / std
            t2 = (t2 / 255.0 - mean) / std

        image_stacked = torch.stack([t1, t2], dim=0)
        return {"image": image_stacked, "mask": mask}


def build_dataloaders(config: dict):
    data_cfg = config["data"]
    root = data_cfg["root"]
    normalize = data_cfg.get("normalize", False)

    train_dataset = OSCD100(root=root, split="train", bands=OSCD100.rgb_bands, download=True)
    val_dataset = OSCD100(root=root, split="val", bands=OSCD100.rgb_bands, download=True)

    train_ds = OSCDDatasetWrapper(train_dataset, augment=True, normalize=normalize)
    val_ds = OSCDDatasetWrapper(val_dataset, augment=False, normalize=normalize)

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


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, ema=None, model_type="siamese_unet"):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if model_type == "unet" and images.dim() == 5:
            b, n_dates, c, h, w = images.shape
            images = images.view(b, n_dates * c, h, w)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device.type, enabled=use_amp):
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

        if ema is not None:
            ema.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device, thresholds=None, model_type="siamese_unet"):
    if thresholds is None:
        thresholds = [round(x * 0.05, 2) for x in range(1, 19)]
    model.eval()
    total_loss = 0.0
    all_metrics = {t: {"iou": 0, "f1": 0, "precision": 0, "recall": 0} for t in thresholds}
    n_batches = 0

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        if model_type == "unet" and images.dim() == 5:
            b, n_dates, c, h, w = images.shape
            images = images.view(b, n_dates * c, h, w)

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

    in_ch = len(OSCD100.rgb_bands)
    model_type = config["model"].get("type", "siamese_unet")
    
    if model_type == "unet":
        model = UNet(
            in_channels=in_ch * 2, out_channels=1,
            features=config["model"]["features"],
            use_attention=config["model"].get("use_attention", False),
        ).to(device)
    else:
        model = SiameseUNet(
            in_channels=in_ch, out_channels=1,
            features=config["model"]["features"],
            use_attention=config["model"].get("use_attention", False),
        ).to(device)

    train_loader, val_loader = build_dataloaders(config)

    all_labels = []
    for batch in train_loader:
        all_labels.append(batch["mask"])
    all_labels = torch.cat(all_labels, dim=0)
    pos_weight = compute_pos_weight(all_labels).to(device)

    loss_cfg = config["loss"]
    loss_type = loss_cfg.get("type", "focal_dice")

    if loss_type == "focal_dice":
        criterion = FocalDiceLoss(
            focal_alpha=loss_cfg.get("focal_alpha", 0.75),
            focal_gamma=loss_cfg.get("focal_gamma", 2.0),
            alpha=loss_cfg.get("alpha", 0.6),
        )
    elif loss_type == "focal_tversky":
        criterion = FocalTverskyLoss(
            alpha=loss_cfg.get("alpha", 0.7),
            gamma=loss_cfg.get("gamma", 0.75),
        )
    elif loss_type == "boundary":
        criterion = BoundaryLoss()
    else:
        from src.training.losses import CombinedLoss
        criterion = CombinedLoss(
            alpha=loss_cfg["alpha"],
            pos_weight=pos_weight,
            smooth=loss_cfg.get("smooth", 1.0),
        )

    opt_cfg = config["optimizer"]
    use_adamw = opt_cfg.get("type", "adam") == "adamw"
    if use_adamw:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0.01),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg["lr"],
            weight_decay=opt_cfg.get("weight_decay", 0),
        )

    use_ema = config.get("ema", {}).get("enabled", False)
    ema_decay = config.get("ema", {}).get("decay", 0.999)
    ema = EMA(model, decay=ema_decay) if use_ema else None

    sched_cfg = config["scheduler"]
    sched_type = sched_cfg.get("type", "cosine")
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=sched_cfg.get("T_0", 10),
            T_mult=sched_cfg.get("T_mult", 2),
            eta_min=sched_cfg.get("min_lr", 1e-6),
        )
    else:
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
    print(f"Input channels: {in_ch} (RGB)")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}\n")

    for epoch in range(train_cfg["max_epochs"]):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, ema, model_type)
        
        if ema is not None:
            ema.apply_shadow()
        val_loss, val_metrics = validate(model, val_loader, criterion, device, model_type=model_type)
        if ema is not None:
            ema.restore()

        best_threshold = 0.05
        best_thresh_f1 = 0
        for t, metrics in val_metrics.items():
            if metrics["f1"] > best_thresh_f1:
                best_thresh_f1 = metrics["f1"]
                best_threshold = t

        if sched_type == "cosine":
            scheduler.step()
        else:
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
            save_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_f1": best_f1,
                "config": {
                    "model": {
                        "type": model_type,
                        "in_channels": in_ch,
                        "out_channels": 1,
                        "features": config["model"]["features"],
                        "use_attention": config["model"].get("use_attention", False),
                    },
                    "data": {"bands": list(OSCD100.rgb_bands), "use_ndvi": False, "normalize": config["data"].get("normalize", False)},
                    "inference": {"overlap": 32, "change_threshold": 0.5},
                },
            }
            if ema is not None:
                ema.apply_shadow()
                save_state["ema_state_dict"] = {k: v.clone() for k, v in model.state_dict().items()}
                ema.restore()
            torch.save(save_state, ckpt_path)
            print(f"  -> Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= train_cfg["early_stopping_patience"]:
            print(f"\nEarly stopping at epoch {epoch+1}")
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
