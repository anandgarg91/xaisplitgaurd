"""
utils/metrics.py
────────────────
Evaluation metrics for XAI-SplitShield experiments.

Metrics implemented:
  - compute_asr()              Attack Success Rate
  - compute_clean_accuracy()   Clean task accuracy
  - compute_detection_metrics() Precision / Recall / F1 for batch detection
  - MetricsTracker             Running tracker across an epoch
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


# ── Standalone Metric Functions ───────────────────────────────────────────────

def compute_asr(
    client_model: torch.nn.Module,
    server_model: torch.nn.Module,
    test_loader,
    attack,
    target_class: int,
    device: torch.device,
    defense=None,
) -> float:
    """
    Computes Attack Success Rate (ASR):
    fraction of trigger-bearing test inputs classified as target_class.

    Args:
        client_model:  Client-side sub-network.
        server_model:  Server-side sub-network.
        test_loader:   DataLoader over CLEAN test data.
        attack:        Attack object with .inject_test(images) method.
        target_class:  Adversary's target class.
        device:        Torch device.
        defense:       Optional XAISplitShield (applied to smashed data).

    Returns:
        ASR in [0, 1].
    """
    client_model.eval()
    server_model.eval()

    total = 0
    success = 0

    for images, labels in test_loader:
        # Only evaluate on samples not already in target class
        non_target = labels != target_class
        images = images[non_target].to(device)
        if images.size(0) == 0:
            continue

        # Inject trigger
        triggered, _ = attack.inject_test(images)

        # Client forward — no grad needed
        with torch.no_grad():
            z = client_model(triggered)

        # Defense — SHAP attribution requires autograd
        if defense is not None and defense._warmup_done:
            dummy_labels = torch.zeros(z.size(0), dtype=torch.long, device=device)
            with torch.enable_grad():
                z, _ = defense.defend(z, dummy_labels)

        # Server forward — no grad needed
        with torch.no_grad():
            logits = server_model(z)
            preds  = logits.argmax(dim=-1)

        success += (preds == target_class).sum().item()
        total   += images.size(0)

    return success / total if total > 0 else 0.0


def compute_clean_accuracy(
    client_model: torch.nn.Module,
    server_model: torch.nn.Module,
    test_loader,
    device: torch.device,
    defense=None,
) -> float:
    """
    Computes clean task accuracy on unmodified test data.

    Returns:
        Accuracy in [0, 1].
    """
    client_model.eval()
    server_model.eval()

    total   = 0
    correct = 0

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Client forward — no grad needed
        with torch.no_grad():
            z = client_model(images)

        # Defense — SHAP attribution requires autograd, so we explicitly
        # enable gradients here. We still don't backprop into anything;
        # gradients are local to the SHAP/LRP attribution computations.
        if defense is not None and defense._warmup_done:
            with torch.enable_grad():
                z, _ = defense.defend(z, labels)

        # Server forward — no grad needed
        with torch.no_grad():
            logits = server_model(z)
            preds  = logits.argmax(dim=-1)

        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    return correct / total if total > 0 else 0.0


def compute_detection_metrics(
    predictions: List[bool],
    ground_truth: List[bool],
) -> Dict[str, float]:
    """
    Computes precision, recall, F1, and FPR for batch-level detection.

    Args:
        predictions:   List of bool — True if batch flagged as poisoned.
        ground_truth:  List of bool — True if batch is actually poisoned.

    Returns:
        Dict with keys: precision, recall, f1, fpr, tpr.
    """
    tp = sum(p and g for p, g in zip(predictions, ground_truth))
    fp = sum(p and not g for p, g in zip(predictions, ground_truth))
    fn = sum(not p and g for p, g in zip(predictions, ground_truth))
    tn = sum(not p and not g for p, g in zip(predictions, ground_truth))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr       = recall

    return {
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "fpr":       fpr,
        "tpr":       tpr,
    }


# ── Running Metrics Tracker ───────────────────────────────────────────────────

@dataclass
class MetricsTracker:
    """
    Accumulates per-batch metrics over an epoch and computes averages.

    Usage:
        tracker = MetricsTracker()
        for batch in loader:
            ...
            tracker.update(loss=loss.item(), acc=acc, aas=score, poisoned=flag)
        summary = tracker.summary()
    """
    _loss:     List[float] = field(default_factory=list)
    _acc:      List[float] = field(default_factory=list)
    _aas:      List[float] = field(default_factory=list)
    _poisoned: List[bool]  = field(default_factory=list)
    _gt:       List[bool]  = field(default_factory=list)

    def update(
        self,
        loss: float,
        acc: float,
        aas: float = 0.0,
        is_poisoned: bool = False,
        is_gt_poisoned: bool = False,
    ):
        self._loss.append(loss)
        self._acc.append(acc)
        self._aas.append(aas)
        self._poisoned.append(is_poisoned)
        self._gt.append(is_gt_poisoned)

    def summary(self) -> Dict[str, float]:
        det = compute_detection_metrics(self._poisoned, self._gt)
        return {
            "loss":      float(np.mean(self._loss)) if self._loss else 0.0,
            "acc":       float(np.mean(self._acc))  if self._acc else 0.0,
            "aas_mean":  float(np.mean(self._aas))  if self._aas else 0.0,
            "aas_max":   float(np.max(self._aas))   if self._aas else 0.0,
            **det,
        }

    def reset(self):
        self._loss.clear()
        self._acc.clear()
        self._aas.clear()
        self._poisoned.clear()
        self._gt.clear()
