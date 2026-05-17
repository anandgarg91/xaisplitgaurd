"""
defense/awsa.py
───────────────
Attribution-Weighted Secure Aggregation (AWSA)
════════════════════════════════════════════════
In the multi-client SL setting, AWSA assigns each client a trust weight
inversely proportional to their cumulative Attribution Anomaly Score (AAS),
down-weighting poisoned gradient contributions without requiring knowledge
of which specific clients are malicious.

From the paper (§4.5):

    w_i(t) = exp(-beta * AAS_i(t)) / sum_j exp(-beta * AAS_j(t))

where AAS_i(t) is the EMA of client i's AAS over the past T rounds.

The aggregated gradient is:
    grad_agg = sum_i w_i(t) * grad_i

Proven Byzantine-resilient for f < N/3 malicious clients.
"""

from __future__ import annotations
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
import numpy as np


# ── Per-client AAS Tracker ────────────────────────────────────────────────────

class ClientAASTracker:
    """
    Tracks the rolling AAS history for a single client and computes its
    EMA trust score.
    """

    def __init__(self, window: int = 10, ema_decay: float = 0.9):
        self.window = window
        self.ema_decay = ema_decay
        self._history: deque = deque(maxlen=window)
        self._ema: float = 0.0
        self._initialized: bool = False

    def update(self, aas_score: float):
        self._history.append(aas_score)
        if not self._initialized:
            self._ema = aas_score
            self._initialized = True
        else:
            self._ema = self.ema_decay * self._ema + (1 - self.ema_decay) * aas_score

    @property
    def ema_aas(self) -> float:
        return self._ema if self._initialized else 0.0

    @property
    def mean_aas(self) -> float:
        return float(np.mean(list(self._history))) if self._history else 0.0


# ── AWSA Aggregator ───────────────────────────────────────────────────────────

class AWSA:
    """
    Attribution-Weighted Secure Aggregation for multi-client split learning.

    Args:
        num_clients:   Total number of clients N.
        beta:          Trust weight decay coefficient (paper: 2.0).
        window:        Look-back rounds for AAS EMA (paper: 10).
        ema_decay:     EMA decay for per-client AAS tracking.
        device:        Torch device.
    """

    def __init__(
        self,
        num_clients: int,
        beta: float = 2.0,
        window: int = 10,
        ema_decay: float = 0.9,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_clients = num_clients
        self.beta = beta
        self.device = device

        # Per-client AAS trackers
        self.trackers: Dict[int, ClientAASTracker] = {
            i: ClientAASTracker(window, ema_decay) for i in range(num_clients)
        }

    # ── AAS Update ────────────────────────────────────────────────────────────

    def update_client_aas(self, client_id: int, aas_score: float):
        """Updates the AAS tracker for client `client_id`."""
        assert client_id in self.trackers, f"Unknown client_id={client_id}"
        self.trackers[client_id].update(aas_score)

    # ── Trust Weight Computation ──────────────────────────────────────────────

    def compute_weights(self, active_clients: Optional[List[int]] = None) -> Dict[int, float]:
        """
        Computes softmax trust weights for all active clients.

        Args:
            active_clients: List of client IDs participating this round.
                            Defaults to all clients.

        Returns:
            Dict mapping client_id -> trust weight w_i(t).
        """
        if active_clients is None:
            active_clients = list(range(self.num_clients))

        aas_values = np.array([self.trackers[i].ema_aas for i in active_clients])

        # Softmax with negative beta * AAS (higher AAS → lower weight)
        log_weights = -self.beta * aas_values
        log_weights -= log_weights.max()   # Numerical stability
        weights = np.exp(log_weights)
        weights /= weights.sum()

        return {cid: float(w) for cid, w in zip(active_clients, weights)}

    # ── Gradient Aggregation ──────────────────────────────────────────────────

    def aggregate_gradients(
        self,
        client_gradients: Dict[int, Tensor],
        active_clients: Optional[List[int]] = None,
    ) -> Tensor:
        """
        Performs trust-weighted aggregation of client gradient tensors.

        Args:
            client_gradients: Dict mapping client_id -> gradient tensor.
            active_clients:   Subset of clients to aggregate (default: all).

        Returns:
            Aggregated gradient tensor.
        """
        if active_clients is None:
            active_clients = list(client_gradients.keys())

        weights = self.compute_weights(active_clients)
        agg = None

        for cid in active_clients:
            if cid not in client_gradients:
                continue
            grad = client_gradients[cid].to(self.device)
            w = weights[cid]
            if agg is None:
                agg = w * grad
            else:
                agg = agg + w * grad

        if agg is None:
            raise ValueError("No client gradients to aggregate.")

        return agg

    # ── Aggregate Server-side Parameter Updates ───────────────────────────────

    def aggregate_updates(
        self,
        client_updates: Dict[int, Dict[str, Tensor]],
        active_clients: Optional[List[int]] = None,
    ) -> Dict[str, Tensor]:
        """
        Trust-weighted aggregation of full server-model parameter updates
        (used in multi-round settings where server weights are re-initialized
        per client via parameter broadcasting).

        Args:
            client_updates: Dict mapping client_id -> {param_name: delta_tensor}.
            active_clients: Subset of clients.

        Returns:
            Aggregated parameter update dict.
        """
        if active_clients is None:
            active_clients = list(client_updates.keys())

        weights = self.compute_weights(active_clients)
        agg: Dict[str, Tensor] = {}

        for cid in active_clients:
            if cid not in client_updates:
                continue
            w = weights[cid]
            for name, delta in client_updates[cid].items():
                delta = delta.to(self.device)
                if name not in agg:
                    agg[name] = w * delta
                else:
                    agg[name] = agg[name] + w * delta

        return agg

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def get_trust_report(self) -> Dict[int, dict]:
        """Returns per-client trust diagnostics for logging/visualization."""
        weights = self.compute_weights()
        return {
            cid: {
                "ema_aas": self.trackers[cid].ema_aas,
                "mean_aas": self.trackers[cid].mean_aas,
                "trust_weight": weights[cid],
            }
            for cid in range(self.num_clients)
        }
