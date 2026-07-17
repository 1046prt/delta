import os
import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentLogger:
    name: str
    log_dir: str = "outputs/metrics"
    use_wandb: bool = False
    wandb_project: str = "delta-change-detection"

    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_iou": [],
            "val_precision": [],
            "val_recall": [],
            "lr": [],
            "epoch_time": [],
            "epoch": [],
        }
        self.start_time = time.time()
        self.wandb_run = None

        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.wandb_project,
                    name=self.name,
                    config={"experiment": self.name},
                )
            except ImportError:
                print("wandb not installed, skipping W&B logging")
                self.use_wandb = False

    def log_epoch(self, epoch, train_loss, val_loss, val_metrics, lr, epoch_time):
        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_f1"].append(val_metrics.get("f1", 0))
        self.history["val_iou"].append(val_metrics.get("iou", 0))
        self.history["val_precision"].append(val_metrics.get("precision", 0))
        self.history["val_recall"].append(val_metrics.get("recall", 0))
        self.history["lr"].append(lr)
        self.history["epoch_time"].append(epoch_time)

        log_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val/{k}": v for k, v in val_metrics.items()},
            "lr": lr,
            "epoch_time": epoch_time,
        }

        if self.use_wandb and self.wandb_run:
            try:
                import wandb
                wandb.log(log_data)
            except Exception:
                pass

    def save(self):
        path = os.path.join(self.log_dir, f"{self.name}_history.json")
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved history to: {path}")

        if self.use_wandb and self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass

    def summary(self):
        total_time = time.time() - self.start_time
        best_f1 = max(self.history["val_f1"]) if self.history["val_f1"] else 0
        best_epoch = self.history["epoch"][self.history["val_f1"].index(best_f1)] + 1 if best_f1 > 0 else 0

        print(f"\n{'='*50}")
        print(f"Experiment: {self.name}")
        print(f"Total time: {total_time/60:.1f} min")
        print(f"Best F1: {best_f1:.4f} @ epoch {best_epoch}")
        print(f"Epochs trained: {len(self.history['epoch'])}")
        print(f"{'='*50}")
