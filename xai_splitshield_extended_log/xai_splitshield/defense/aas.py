"""
defense/aas.py
──────────────
Attribution Anomaly Score (AAS)
═════════════════════════════════
Quantifies deviation of the current batch's attribution distribution
from a dynamically maintained clean attribution baseline.

From the paper (§4.3):

    AAS(B_t) = KL(P_phi(B_t) || P_phi_clean) + lambda * ||mu_phi(B_t) - mu_phi_clean||_2

The KL term captures distributional deviation in attribution patterns.
The L2 term captures directional shift in mean attribution vector.
Both are characteristic of trigger-induced neuron activation changes.

Threshold calibration:
    During warm-up, AAS values from clean batches are collected. The
    detection threshold tau is set to the (1 - fpr)-th percentile of
    this empirical clean distribution, targeting a specified FPR.
"""

from __future__ import annotations
from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from defense.sda import SmashedDataAttribution


# ── Histogram Utilities ───────────────────────────────────────────────────────

def _to_histogram(phi_flat: Tensor, n_bins: int = 64) -> Tensor:
    """
    Converts a flat attribution tensor to a normalized histogram.
    Uses the range [min, max] of the tensor for binning.
    """
    phi_np = phi_flat.detach().cpu().float().numpy()
    hist, _ = np.histogram(phi_np, bins=n_bins, density=True)
    hist = hist + 1e-10       # Laplace smoothing to avoid log(0)
    hist = hist / hist.sum()
    return torch.tensor(hist, dtype=torch.float32)


def _kl_divergence(p: Tensor, q: Tensor) -> float:
    """KL(p || q) in nats."""
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    return float((p * (p / q).log()).sum().item())


# ── Clean Attribution Baseline ────────────────────────────────────────────────

class CleanAttributionBaseline:
    """
    Maintains an exponential moving average of the clean attribution
    histogram and mean vector, used as the reference in AAS computation.
    """

    def __init__(self, ema_decay: float = 0.95, n_bins: int = 64):
        self.ema_decay = ema_decay
        self.n_bins = n_bins
        self.mean_hist: Optional[Tensor] = None     # EMA of histogram
        self.mean_vec:  Optional[Tensor] = None     # EMA of mean attribution vector
        self.initialized = False

    def update(self, phi_batch: Tensor):
        """
        Updates the clean baseline with attribution from a clean batch.

        Args:
            phi_batch: Attribution tensor (B, ...) from SDA.
        """
        phi_flat = phi_batch.detach().flatten()
        hist = _to_histogram(phi_flat, self.n_bins)
        mean_vec = phi_batch.detach().mean(dim=0)

        if not self.initialized:
            self.mean_hist = hist
            self.mean_vec  = mean_vec
            self.initialized = True
        else:
            d = self.ema_decay
            self.mean_hist = d * self.mean_hist + (1 - d) * hist
            self.mean_vec  = d * self.mean_vec  + (1 - d) * mean_vec

    def get_hist(self) -> Tensor:
        assert self.initialized, "Baseline not yet initialized (call during warm-up)."
        return self.mean_hist

    def get_mean_vec(self) -> Tensor:
        assert self.initialized, "Baseline not yet initialized."
        return self.mean_vec


# ── AAS Computer ─────────────────────────────────────────────────────────────

class AttributionAnomalyScore:
    """
    Computes the Attribution Anomaly Score (AAS) for an incoming smashed-data batch.

    Args:
        sda:            SmashedDataAttribution module.
        ema_decay:      EMA decay for the clean attribution baseline.
        lam:            Weighting between KL and L2 terms (paper: 0.5).
        n_bins:         Histogram bins for KL estimation.
        fpr:            Target false positive rate for threshold calibration.
    """

    def __init__(
        self,
        sda: SmashedDataAttribution,
        ema_decay: float = 0.95,
        lam: float = 0.5,
        n_bins: int = 64,
        fpr: float = 0.01,
    ):
        self.sda = sda
        self.lam = lam
        self.n_bins = n_bins
        self.fpr = fpr

        self.baseline = CleanAttributionBaseline(ema_decay, n_bins)

        # Warm-up collection for threshold calibration
        self._warmup_scores: list[float] = []
        self.threshold: Optional[float] = None
        self.warmup_done = False

    # ── Warm-up Phase ─────────────────────────────────────────────────────────

    def warmup_update(self, z_batch: Tensor):
        """
        Called during warm-up with clean smashed-data batches.
        Updates the attribution baseline and collects AAS scores for
        threshold calibration.
        """
        phi = self.sda.attribute(z_batch)
        self.baseline.update(phi)
        self.sda.update_baseline(z_batch)

        if self.baseline.initialized:
            score = self._compute_score(phi)
            self._warmup_scores.append(score)

    def calibrate_threshold(self):
        """
        Sets the detection threshold tau at the (1 - fpr)-th percentile
        of the clean AAS distribution collected during warm-up.
        """
        if len(self._warmup_scores) == 0:
            raise RuntimeError("No warm-up scores collected. Call warmup_update() first.")
        percentile = (1 - self.fpr) * 100
        self.threshold = float(np.percentile(self._warmup_scores, percentile))
        self.warmup_done = True

    # ── Online Detection ──────────────────────────────────────────────────────

    def compute(self, z_batch: Tensor, update_baseline: bool = True) -> float:
        """
        Computes AAS for an incoming smashed-data batch.

        Args:
            z_batch:        Smashed data (B, C, H, W).
            update_baseline: If True, updates the EMA baseline (online learning).

        Returns:
            AAS score (float). Higher → more anomalous.
        """
        phi = self.sda.attribute(z_batch)
        score = self._compute_score(phi)

        if update_baseline:
            # Only update baseline with non-flagged batches to avoid drift
            if self.threshold is None or score <= self.threshold:
                self.baseline.update(phi)

        return score

    def is_poisoned(self, z_batch: Tensor, update_baseline: bool = True) -> tuple[bool, float]:
        """
        Returns (is_poisoned, aas_score) for the batch.
        Requires calibrate_threshold() to have been called.
        """
        assert self.warmup_done, "Call calibrate_threshold() before detection."
        score = self.compute(z_batch, update_baseline)
        return score > self.threshold, score

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute_score(self, phi: Tensor) -> float:
        """
        AAS = KL(P_phi_batch || P_phi_clean) + lambda * ||mu_phi_batch - mu_phi_clean||_2
        """
        if not self.baseline.initialized:
            return 0.0

        # KL divergence term
        phi_flat = phi.detach().flatten()
        batch_hist = _to_histogram(phi_flat, self.n_bins)
        clean_hist = self.baseline.get_hist()
        kl = _kl_divergence(batch_hist, clean_hist)

        # L2 mean-vector deviation term
        batch_mean = phi.detach().mean(dim=0)
        clean_mean = self.baseline.get_mean_vec().to(batch_mean.device)
        l2 = float((batch_mean - clean_mean).norm(p=2).item())

        return kl + self.lam * l2
