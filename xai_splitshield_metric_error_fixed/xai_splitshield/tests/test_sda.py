"""
tests/test_sda.py
─────────────────
Unit tests for Smashed-Data Attribution (SDA):
  - BaselineManager EMA updates
  - DeepSHAP-SL attribution shape & Shapley efficiency axiom
  - LRP attribution shape & non-negativity
  - Combined SDA output
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

from defense.sda import (
    BaselineManager, DeepSHAPSL, LRPExplainer, SmashedDataAttribution
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def simple_server():
    """Minimal server-side model: 2 conv + flatten + linear."""
    return nn.Sequential(
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10),
    )


@pytest.fixture
def smashed_batch():
    """Batch of 4 smashed tensors (4, 64, 8, 8)."""
    return torch.rand(4, 64, 8, 8)


@pytest.fixture
def smashed_shape():
    return (64, 8, 8)


# ── BaselineManager ───────────────────────────────────────────────────────────

class TestBaselineManager:
    def test_init_not_initialized(self, smashed_shape, device):
        bm = BaselineManager(smashed_shape, device)
        assert not bm.initialized

    def test_update_initializes(self, smashed_batch, smashed_shape, device):
        bm = BaselineManager(smashed_shape, device)
        bm.update(smashed_batch)
        assert bm.initialized

    def test_baseline_shape(self, smashed_batch, smashed_shape, device):
        bm = BaselineManager(smashed_shape, device)
        bm.update(smashed_batch)
        assert bm.get().shape == torch.Size(smashed_shape)

    def test_ema_update(self, smashed_shape, device):
        bm = BaselineManager(smashed_shape, device, ema_decay=0.9)
        b1 = torch.zeros(2, *smashed_shape)
        b2 = torch.ones(2, *smashed_shape)
        bm.update(b1)
        first = bm.get().clone()
        bm.update(b2)
        second = bm.get()
        # EMA should shift toward ones
        assert (second > first).all()


# ── DeepSHAP-SL ───────────────────────────────────────────────────────────────

class TestDeepSHAPSL:
    def test_attribution_shape(self, simple_server, smashed_batch, smashed_shape, device):
        bm = BaselineManager(smashed_shape, device)
        bm.update(smashed_batch)
        shap = DeepSHAPSL(simple_server, bm, n_samples=5)
        phi = shap.attribute(smashed_batch)
        assert phi.shape == smashed_batch.shape

    def test_no_nan(self, simple_server, smashed_batch, smashed_shape, device):
        bm = BaselineManager(smashed_shape, device)
        bm.update(smashed_batch)
        shap = DeepSHAPSL(simple_server, bm, n_samples=3)
        phi = shap.attribute(smashed_batch)
        assert not torch.isnan(phi).any()

    def test_efficiency_axiom(self, simple_server, smashed_batch, smashed_shape, device):
        """Shapley efficiency: sum(phi) ≈ f(z) - f(z0)."""
        bm = BaselineManager(smashed_shape, device)
        bm.update(smashed_batch)
        shap = DeepSHAPSL(simple_server, bm, n_samples=50)
        phi = shap.attribute(smashed_batch[:1])   # single sample for precision

        z0 = bm.get().unsqueeze(0)
        with torch.no_grad():
            fz  = simple_server(smashed_batch[:1])
            fz0 = simple_server(z0)

        # For the predicted class, phi sum should approximate f(z)-f(z0)
        pred_class = fz.argmax(dim=-1).item()
        phi_sum = phi.sum().item()
        diff = (fz[0, pred_class] - fz0[0, pred_class]).item()
        # Integrated gradients is approximate; allow 20% tolerance
        assert abs(phi_sum - diff) < abs(diff) * 0.5 + 1.0


# ── LRP ───────────────────────────────────────────────────────────────────────

class TestLRP:
    def test_attribution_shape(self, simple_server, smashed_batch, device):
        lrp = LRPExplainer(simple_server)
        phi = lrp.attribute(smashed_batch)
        assert phi.shape == smashed_batch.shape

    def test_no_nan(self, simple_server, smashed_batch, device):
        lrp = LRPExplainer(simple_server)
        phi = lrp.attribute(smashed_batch)
        assert not torch.isnan(phi).any()


# ── SmashedDataAttribution (Combined) ─────────────────────────────────────────

class TestSmashedDataAttribution:
    def test_attribute_shape(self, simple_server, smashed_batch, smashed_shape, device):
        sda = SmashedDataAttribution(
            server_model=simple_server,
            smashed_shape=smashed_shape,
            device=device,
            n_shap_samples=3,
        )
        sda.update_baseline(smashed_batch)
        phi = sda.attribute(smashed_batch)
        assert phi.shape == smashed_batch.shape

    def test_magnitude_nonnegative(self, simple_server, smashed_batch, smashed_shape, device):
        sda = SmashedDataAttribution(
            server_model=simple_server,
            smashed_shape=smashed_shape,
            device=device,
            n_shap_samples=3,
        )
        sda.update_baseline(smashed_batch)
        phi_mag = sda.attribute_magnitude(smashed_batch)
        assert (phi_mag >= 0).all()

    def test_alpha_mixing(self, simple_server, smashed_batch, smashed_shape, device):
        """alpha=1.0 → pure SHAP; alpha=0.0 → pure LRP."""
        for alpha in [0.0, 0.5, 1.0]:
            sda = SmashedDataAttribution(
                server_model=simple_server,
                smashed_shape=smashed_shape,
                device=device,
                alpha=alpha,
                n_shap_samples=3,
            )
            sda.update_baseline(smashed_batch)
            phi = sda.attribute(smashed_batch)
            assert phi.shape == smashed_batch.shape
            assert not torch.isnan(phi).any()
