import torch
import numpy as np


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()

    intersection = (pred_binary * target_binary).sum().item()
    union = pred_binary.sum().item() + target_binary.sum().item() - intersection

    if union == 0:
        return 1.0
    return intersection / union


def compute_f1(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()

    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_precision(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()

    tp = (pred_binary * target_binary).sum().item()
    fp = (pred_binary * (1 - target_binary)).sum().item()

    return tp / (tp + fp + 1e-10)


def compute_recall(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()

    tp = (pred_binary * target_binary).sum().item()
    fn = ((1 - pred_binary) * target_binary).sum().item()

    return tp / (tp + fn + 1e-10)


def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_binary = (torch.sigmoid(pred) > threshold).float()
    target_binary = target.float()

    correct = (pred_binary == target_binary).float().sum().item()
    total = target_binary.numel()

    return correct / (total + 1e-10)


def evaluate_all(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    return {
        "iou": compute_iou(pred, target, threshold),
        "f1": compute_f1(pred, target, threshold),
        "precision": compute_precision(pred, target, threshold),
        "recall": compute_recall(pred, target, threshold),
        "pixel_accuracy": compute_pixel_accuracy(pred, target, threshold),
    }
