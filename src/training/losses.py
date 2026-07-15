import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)


class CombinedLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        pos_weight: torch.Tensor | None = None,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth

        bce_kwargs = {}
        if pos_weight is not None:
            bce_kwargs["pos_weight"] = pos_weight
        self.bce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(torch.sigmoid(pred), target)
        return self.alpha * bce_loss + (1 - self.alpha) * dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    pos = labels.sum()
    neg = labels.numel() - pos
    ratio = neg / (pos + 1e-10)
    return torch.clamp(ratio, max=50.0)
