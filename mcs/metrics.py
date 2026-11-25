"""Pixel and object metrics."""
from __future__ import annotations

import numpy as np


def dice(pred: np.ndarray, true: np.ndarray) -> float:
    inter = np.logical_and(pred, true).sum()
    s = pred.sum() + true.sum()
    return 1.0 if s == 0 else float(2 * inter / s)


def iou(pred: np.ndarray, true: np.ndarray) -> float:
    union = np.logical_or(pred, true).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, true).sum() / union)


def count_error(pred_labels: np.ndarray, true_labels: np.ndarray) -> int:
    return int(len(np.unique(pred_labels)) - len(np.unique(true_labels)))
