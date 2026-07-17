import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FocalDiceLoss(nn.Module):
    def __init__(
        self,
        focal_alpha: float = 0.75,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0,
        alpha: float = 0.6,
    ):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice = DiceLoss(smooth=dice_smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fl = self.focal(pred, target)
        dl = self.dice(torch.sigmoid(pred), target)
        return self.alpha * fl + (1 - self.alpha) * dl


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.7, gamma: float = 0.75, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = torch.sigmoid(pred).view(-1)
        target_flat = target.view(-1)

        tp = (pred_flat * target_flat).sum()
        fp = ((1 - target_flat) * pred_flat).sum()
        fn = (target_flat * (1 - pred_flat)).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + (1 - self.alpha) * fn + self.smooth)
        return (1 - tversky) ** self.gamma


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _compute_boundary(mask: torch.Tensor) -> torch.Tensor:
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=mask.dtype, device=mask.device).view(1, 1, 3, 3)

        grad_x = F.conv2d(mask, kernel_x, padding=1)
        grad_y = F.conv2d(mask, kernel_y, padding=1)

        boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        boundary = (boundary > 0).float()
        return F.max_pool2d(boundary, 3, stride=1, padding=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_prob = torch.sigmoid(pred)
        target_b = self._compute_boundary(target.unsqueeze(1)).squeeze(1)
        pred_b = self._compute_boundary(pred_prob.unsqueeze(1)).squeeze(1)

        intersection = (pred_b * target_b).sum()
        union = pred_b.sum() + target_b.sum()
        return 1 - (2 * intersection + 1) / (union + 1)


def compute_pos_weight(labels: torch.Tensor) -> torch.Tensor:
    pos = labels.sum()
    neg = labels.numel() - pos
    ratio = neg / (pos + 1e-10)
    return torch.clamp(ratio, max=50.0)
