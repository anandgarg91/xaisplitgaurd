"""
defense/agns.py
───────────────
Attribution-Guided Neuron Suppression (AGNS)
═════════════════════════════════════════════
Upon detection of a potentially poisoned batch, AGNS identifies the top-k
neurons by attribution magnitude and applies stochastic activation masking
to neutralize the backdoor trigger signal.

From the paper (§4.4):
  - Top-k neurons in |phi(z)| are suppressed.
  - k is selected via bisection to satisfy a clean-accuracy preservation
    constraint: Clean_Acc(theta | AGNS_k) >= Clean_Acc(theta) - epsilon_acc.
  - Masking is applied to BOTH the forward smashed-data tensor and the
    backward gradient, preventing trigger-conditioned weight updates.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from defense.sda import SmashedDataAttribution


# ── AGNS Core ────────────────────────────────────────────────────────────────

class AGNS:
    """
    Attribution-Guided Neuron Suppression module.

    Args:
        sda:             SmashedDataAttribution instance.
        server_model:    Server-side nn.Module (used to evaluate accuracy constraint).
        epsilon_acc:     Maximum tolerated clean accuracy drop (default: 0.01 = 1%).
        p_mask:          Masking probability for high-confidence anomalous neurons.
        k_min:           Minimum number of neurons to suppress.
        k_max_fraction:  Maximum fraction of neurons eligible for suppression.
        device:          Torch device.
    """

    def __init__(
        self,
        sda: SmashedDataAttribution,
        server_model: nn.Module,
        epsilon_acc: float = 0.01,
        p_mask: float = 1.0,
        k_min: int = 1,
        k_max_fraction: float = 0.3,
        device: torch.device = torch.device("cpu"),
    ):
        self.sda = sda
        self.server_model = server_model
        self.epsilon_acc = epsilon_acc
        self.p_mask = p_mask
        self.k_min = k_min
        self.k_max_fraction = k_max_fraction
        self.device = device

        # Baseline clean accuracy estimated during warm-up
        self._clean_acc_baseline: Optional[float] = None

        # Clean validation buffer (label-free — only used for forward accuracy)
        self._clean_val_buffer: Optional[Tensor] = None
        self._clean_val_labels: Optional[Tensor] = None

    # ── Warm-up ───────────────────────────────────────────────────────────────

    def update_clean_val(self, z_batch: Tensor, labels: Tensor):
        """
        Stores a clean validation batch for accuracy-constraint evaluation.
        Only the most recent batch is kept (simple rolling buffer).
        """
        self._clean_val_buffer = z_batch.detach().to(self.device)
        self._clean_val_labels = labels.to(self.device)

    def calibrate_clean_accuracy(self):
        """
        Computes the baseline clean accuracy (without AGNS) on the stored
        validation buffer. Should be called at the end of warm-up.
        """
        if self._clean_val_buffer is None:
            raise RuntimeError("No clean validation data. Call update_clean_val() first.")
        self._clean_acc_baseline = self._eval_accuracy(self._clean_val_buffer,
                                                        self._clean_val_labels,
                                                        mask=None)

    # ── Suppression ───────────────────────────────────────────────────────────

    def suppress(self, z: Tensor, phi: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Applies attribution-guided masking to a (potentially poisoned) smashed batch.

        Args:
            z:    Smashed data batch (B, C, H, W).
            phi:  Pre-computed attribution tensor. If None, SDA is called internally.

        Returns:
            (z_suppressed, mask): Suppressed smashed data and the binary mask applied.
        """
        if phi is None:
            phi = self.sda.attribute_magnitude(z)

        d_total = z[0].numel()
        k_max = max(self.k_min, int(d_total * self.k_max_fraction))

        # Select k via bisection (accuracy constraint)
        k = self._bisect_k(z, phi, k_min=self.k_min, k_max=k_max)

        mask = self._build_mask(z, phi, k)
        z_suppressed = z * mask
        return z_suppressed, mask

    def suppress_gradient(self, grad: Tensor, mask: Tensor) -> Tensor:
        """
        Applies the same spatial mask to the backpropagated gradient,
        preventing trigger-conditioned weight updates in the server sub-network.
        """
        return grad * mask

    # ── Bisection for k ───────────────────────────────────────────────────────

    def _bisect_k(self, z: Tensor, phi: Tensor, k_min: int, k_max: int) -> int:
        """
        Binary search for the largest k such that suppressing the top-k
        neurons still satisfies the clean accuracy constraint.
        """
        if self._clean_acc_baseline is None:
            # Fallback: suppress a small fixed fraction
            return max(k_min, int(z[0].numel() * 0.05))

        acc_target = self._clean_acc_baseline - self.epsilon_acc

        lo, hi = k_min, k_max
        best_k = k_min

        while lo <= hi:
            mid = (lo + hi) // 2
            mask = self._build_mask(z, phi, mid)
            z_masked = z * mask

            if self._clean_val_buffer is not None:
                masked_val = self._clean_val_buffer * mask[:1].expand_as(self._clean_val_buffer)
                acc = self._eval_accuracy(masked_val, self._clean_val_labels, mask=None)
            else:
                acc = acc_target + 1.0  # No constraint data: allow suppression

            if acc >= acc_target:
                best_k = mid
                lo = mid + 1
            else:
                hi = mid - 1

        return best_k

    # ── Mask Builder ──────────────────────────────────────────────────────────

    def _build_mask(self, z: Tensor, phi: Tensor, k: int) -> Tensor:
        """
        Builds a binary mask that zeros the top-k neurons by attribution magnitude.
        Mask shape matches z: (B, C, H, W).
        """
        B = z.size(0)
        phi_flat = phi.view(B, -1)          # (B, D)
        d_total = phi_flat.size(1)
        k = min(k, d_total)

        # Top-k indices per sample
        _, top_idx = phi_flat.topk(k, dim=1, largest=True, sorted=False)

        mask_flat = torch.ones_like(phi_flat)

        # Apply stochastic masking: p_mask for top neurons
        rand = torch.rand_like(phi_flat)
        stochastic_suppress = (rand < self.p_mask).float()

        mask_flat.scatter_(1, top_idx, 1.0 - stochastic_suppress.gather(1, top_idx))

        mask = mask_flat.view_as(z)
        return mask.detach()

    # ── Accuracy Evaluation ───────────────────────────────────────────────────

    def _eval_accuracy(self, z: Tensor, labels: Tensor, mask: Optional[Tensor]) -> float:
        """Evaluates server-model accuracy on a smashed-data batch (label-free path)."""
        self.server_model.eval()
        with torch.no_grad():
            if mask is not None:
                z = z * mask
            logits = self.server_model(z)
            preds = logits.argmax(dim=-1)
            acc = (preds == labels).float().mean().item()
        return acc
