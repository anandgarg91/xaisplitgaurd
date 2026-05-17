"""
defense/xai_splitshield.py
───────────────────────────
XAISplitShield — Unified Defense Orchestrator
═══════════════════════════════════════════════
This module wires together all four defense components (SDA, AAS, AGNS, AWSA)
into a single interface used by the training loop in run_experiment.py.

Usage (single-client):
    shield = XAISplitShield(server_model, smashed_shape, device, cfg)
    shield.warmup(z_batch, labels)          # call for warmup_epochs
    shield.calibrate()                       # call once after warmup
    z_clean, report = shield.defend(z_batch, labels)

Usage (multi-client via AWSA):
    agg_grad = shield.aggregate(client_grads, client_aas_scores)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from defense.sda import SmashedDataAttribution
from defense.aas import AttributionAnomalyScore
from defense.agns import AGNS
from defense.awsa import AWSA


# ── Defense Report ────────────────────────────────────────────────────────────

@dataclass
class DefenseReport:
    """Per-batch defense output returned by XAISplitShield.defend()."""
    aas_score: float                      # Raw AAS value
    is_poisoned: bool                     # Detection decision
    num_suppressed: int                   # Neurons suppressed by AGNS
    suppression_mask: Optional[Tensor]    # Binary mask (B, C, H, W) or None
    phi: Optional[Tensor]                 # Attribution tensor (B, C, H, W)
    detection_threshold: float            # Current tau


# ── Main Orchestrator ─────────────────────────────────────────────────────────

class XAISplitShield:
    """
    Unified defense orchestrator combining SDA + AAS + AGNS (+ AWSA).

    Args:
        server_model:    Server-side nn.Module (used for attribution & AGNS).
        smashed_shape:   Shape of smashed data excluding batch dim.
        device:          Torch device.
        cfg:             Defense config dict (from base.yaml → defense section).
        num_clients:     Number of clients (1 = single-client SL).
    """

    def __init__(
        self,
        server_model: nn.Module,
        smashed_shape: tuple,
        device: torch.device,
        cfg: dict,
        num_clients: int = 1,
    ):
        self.server_model = server_model
        self.smashed_shape = smashed_shape
        self.device = device
        self.cfg = cfg
        self.num_clients = num_clients
        self._warmup_done = False

        # ── Instantiate SDA ──────────────────────────────────────────────────
        self.sda = SmashedDataAttribution(
            server_model=server_model,
            smashed_shape=smashed_shape,
            device=device,
            alpha=cfg.get("sda_alpha", 0.6),
            ema_decay=cfg.get("aas_ema_decay", 0.95),
            n_shap_samples=cfg.get("sda_n_samples", 20),
        )

        # ── Instantiate AAS ──────────────────────────────────────────────────
        self.aas_detector = AttributionAnomalyScore(
            sda=self.sda,
            ema_decay=cfg.get("aas_ema_decay", 0.95),
            lam=cfg.get("aas_lambda", 0.5),
            fpr=cfg.get("aas_fpr", 0.01),
        )

        # ── Instantiate AGNS ─────────────────────────────────────────────────
        self.agns = AGNS(
            sda=self.sda,
            server_model=server_model,
            epsilon_acc=cfg.get("agns_epsilon_acc", 0.01),
            p_mask=cfg.get("agns_p_mask", 1.0),
            device=device,
        )

        # ── Instantiate AWSA (multi-client) ──────────────────────────────────
        self.awsa: Optional[AWSA] = None
        if num_clients > 1:
            self.awsa = AWSA(
                num_clients=num_clients,
                beta=cfg.get("awsa_beta", 2.0),
                window=cfg.get("awsa_window", 10),
                device=device,
            )

    # ── Warm-up Phase ─────────────────────────────────────────────────────────

    def warmup(self, z_batch: Tensor, labels: Tensor):
        """
        Call this every batch during the warm-up phase with CLEAN smashed data.
        Updates SDA baseline, AAS clean distribution, and AGNS clean-val buffer.
        """
        z_batch = z_batch.to(self.device)
        labels = labels.to(self.device)
        self.aas_detector.warmup_update(z_batch)
        self.agns.update_clean_val(z_batch, labels)

    def calibrate(self):
        """
        Call once after warm-up completes.
        Sets AAS detection threshold and AGNS clean accuracy baseline.
        """
        self.aas_detector.calibrate_threshold()
        self.agns.calibrate_clean_accuracy()
        self._warmup_done = True

    # ── Online Defense ────────────────────────────────────────────────────────

    def defend(
        self,
        z_batch: Tensor,
        labels: Tensor,
        client_id: int = 0,
    ) -> Tuple[Tensor, DefenseReport]:
        """
        Main defense entry point. Called every training batch after warm-up.

        Pipeline:
            1. Compute attribution phi(Z) via SDA.
            2. Compute AAS and detect poisoning.
            3. If poisoned: suppress anomalous neurons via AGNS.
            4. Update AWSA trust tracker (multi-client).

        Args:
            z_batch:   Smashed data batch (B, C', H', W').
            labels:    Ground-truth labels (B,) — used only for AGNS accuracy.
            client_id: Client identifier (for AWSA tracking in multi-client).

        Returns:
            (z_out, report): Defended smashed data and defense report.
        """
        if not self._warmup_done:
            raise RuntimeError(
                "XAISplitShield.calibrate() must be called after warmup."
            )

        z_batch = z_batch.to(self.device)
        labels  = labels.to(self.device)

        # Step 1: Attribution
        phi = self.sda.attribute_magnitude(z_batch)

        # Step 2: Anomaly Detection
        is_poisoned, aas_score = self.aas_detector.is_poisoned(
            z_batch, update_baseline=True
        )

        # Step 3: Trigger Neutralization
        z_out = z_batch
        mask = None
        num_suppressed = 0
        if is_poisoned:
            z_out, mask = self.agns.suppress(z_batch, phi)
            num_suppressed = int((mask == 0).sum().item())

        # Step 4: AWSA trust update
        if self.awsa is not None:
            self.awsa.update_client_aas(client_id, aas_score)

        report = DefenseReport(
            aas_score=aas_score,
            is_poisoned=is_poisoned,
            num_suppressed=num_suppressed,
            suppression_mask=mask,
            phi=phi.detach() if phi is not None else None,
            detection_threshold=self.aas_detector.threshold,
        )
        return z_out, report

    # ── Multi-client Aggregation ──────────────────────────────────────────────

    def aggregate(
        self,
        client_gradients: Dict[int, Tensor],
        active_clients: Optional[List[int]] = None,
    ) -> Tensor:
        """
        AWSA-based trust-weighted gradient aggregation (multi-client only).

        Args:
            client_gradients: {client_id: gradient_tensor}
            active_clients:   Subset to aggregate (default: all).

        Returns:
            Aggregated gradient tensor.
        """
        if self.awsa is None:
            raise RuntimeError("AWSA requires num_clients > 1.")
        return self.awsa.aggregate_gradients(client_gradients, active_clients)

    def trust_report(self) -> Dict:
        """Returns per-client trust diagnostics from AWSA."""
        if self.awsa is None:
            return {}
        return self.awsa.get_trust_report()
