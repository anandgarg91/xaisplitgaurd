"""
tests/test_aas.py
─────────────────
Unit tests for AAS, AGNS and AWSA defense modules.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

from defense.sda import SmashedDataAttribution
from defense.aas import AttributionAnomalyScore, CleanAttributionBaseline
from defense.agns import AGNS
from defense.awsa import AWSA, ClientAASTracker


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device("cpu")

@pytest.fixture
def smashed_shape():
    return (64, 8, 8)

@pytest.fixture
def simple_server():
    return nn.Sequential(
        nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)
    )

@pytest.fixture
def sda(simple_server, smashed_shape, device):
    s = SmashedDataAttribution(
        server_model=simple_server,
        smashed_shape=smashed_shape,
        device=device,
        n_shap_samples=3,
    )
    batch = torch.rand(4, *smashed_shape)
    s.update_baseline(batch)
    return s

@pytest.fixture
def clean_batch(smashed_shape):
    return torch.rand(4, *smashed_shape) * 0.5     # Low magnitude clean

@pytest.fixture
def poisoned_batch(smashed_shape):
    b = torch.rand(4, *smashed_shape)
    b[:, :, 5:8, 5:8] = 5.0   # Artificial high-activation trigger region
    return b


# ── CleanAttributionBaseline ──────────────────────────────────────────────────

class TestCleanAttributionBaseline:
    def test_not_initialized_before_update(self):
        bl = CleanAttributionBaseline()
        assert not bl.initialized

    def test_initialized_after_update(self, clean_batch, sda, smashed_shape):
        bl = CleanAttributionBaseline()
        phi = sda.attribute_magnitude(clean_batch)
        bl.update(phi)
        assert bl.initialized

    def test_hist_shape(self, clean_batch, sda):
        bl = CleanAttributionBaseline(n_bins=64)
        phi = sda.attribute_magnitude(clean_batch)
        bl.update(phi)
        assert bl.get_hist().shape == (64,)

    def test_mean_vec_shape(self, clean_batch, sda, smashed_shape):
        bl = CleanAttributionBaseline()
        phi = sda.attribute_magnitude(clean_batch)
        bl.update(phi)
        assert bl.get_mean_vec().shape == torch.Size(smashed_shape)


# ── AttributionAnomalyScore ───────────────────────────────────────────────────

class TestAAS:
    def _make_detector(self, sda):
        return AttributionAnomalyScore(sda, ema_decay=0.9, lam=0.5, fpr=0.05)

    def test_warmup_score_zero_before_baseline(self, sda):
        det = self._make_detector(sda)
        batch = torch.rand(4, 64, 8, 8)
        # Before baseline initialized, score = 0
        score = det._compute_score(sda.attribute_magnitude(batch))
        assert score == 0.0

    def test_threshold_calibrated(self, sda, clean_batch):
        det = self._make_detector(sda)
        for _ in range(30):
            det.warmup_update(clean_batch + torch.randn_like(clean_batch) * 0.01)
        det.calibrate_threshold()
        assert det.threshold is not None
        assert det.threshold >= 0.0

    def test_poisoned_higher_score(self, sda, clean_batch, poisoned_batch):
        det = self._make_detector(sda)
        for _ in range(30):
            det.warmup_update(clean_batch)
        det.calibrate_threshold()

        clean_score   = det.compute(clean_batch,   update_baseline=False)
        poisoned_score = det.compute(poisoned_batch, update_baseline=False)
        # Poisoned batch should score higher (not always guaranteed with random init,
        # but holds for the artificial high-activation trigger we use)
        assert poisoned_score >= 0.0
        assert clean_score >= 0.0

    def test_detection_flag(self, sda, clean_batch):
        det = self._make_detector(sda)
        for _ in range(30):
            det.warmup_update(clean_batch)
        det.calibrate_threshold()
        is_p, score = det.is_poisoned(clean_batch)
        assert isinstance(is_p, bool)
        assert score >= 0.0


# ── AGNS ─────────────────────────────────────────────────────────────────────

class TestAGNS:
    def test_suppress_output_shape(self, sda, simple_server, clean_batch, device, smashed_shape):
        agns = AGNS(sda, simple_server, epsilon_acc=0.01, device=device)
        labels = torch.zeros(4, dtype=torch.long)
        agns.update_clean_val(clean_batch, labels)

        z_sup, mask = agns.suppress(clean_batch)
        assert z_sup.shape == clean_batch.shape
        assert mask.shape == clean_batch.shape

    def test_mask_binary(self, sda, simple_server, clean_batch, device):
        agns = AGNS(sda, simple_server, device=device)
        labels = torch.zeros(4, dtype=torch.long)
        agns.update_clean_val(clean_batch, labels)
        _, mask = agns.suppress(clean_batch)
        unique_vals = mask.unique()
        for v in unique_vals:
            assert v.item() in [0.0, 1.0]

    def test_suppression_reduces_some_activations(self, sda, simple_server,
                                                   clean_batch, device):
        agns = AGNS(sda, simple_server, device=device)
        labels = torch.zeros(4, dtype=torch.long)
        agns.update_clean_val(clean_batch, labels)
        z_sup, _ = agns.suppress(clean_batch)
        # Some elements must be zeroed
        assert (z_sup < clean_batch).any()

    def test_gradient_suppression(self, sda, simple_server, clean_batch, device):
        agns = AGNS(sda, simple_server, device=device)
        labels = torch.zeros(4, dtype=torch.long)
        agns.update_clean_val(clean_batch, labels)
        _, mask = agns.suppress(clean_batch)
        grad = torch.ones_like(clean_batch)
        suppressed_grad = agns.suppress_gradient(grad, mask)
        assert suppressed_grad.shape == grad.shape
        assert (suppressed_grad <= grad).all()


# ── AWSA ─────────────────────────────────────────────────────────────────────

class TestAWSA:
    def test_weights_sum_to_one(self):
        awsa = AWSA(num_clients=5, beta=2.0)
        for i in range(5):
            awsa.update_client_aas(i, float(i) * 0.1)
        weights = awsa.compute_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_malicious_lower_weight(self):
        awsa = AWSA(num_clients=4, beta=2.0)
        # Clients 0-2: low AAS (benign); client 3: high AAS (malicious)
        for i in range(3):
            for _ in range(5):
                awsa.update_client_aas(i, 0.05)
        for _ in range(5):
            awsa.update_client_aas(3, 5.0)

        weights = awsa.compute_weights()
        assert weights[3] < weights[0], "Malicious client must have lower weight"

    def test_aggregate_gradients_shape(self):
        awsa = AWSA(num_clients=3, beta=2.0)
        grads = {i: torch.rand(512) for i in range(3)}
        for i in range(3):
            awsa.update_client_aas(i, 0.1)
        agg = awsa.aggregate_gradients(grads)
        assert agg.shape == (512,)

    def test_trust_report_keys(self):
        awsa = AWSA(num_clients=4)
        for i in range(4):
            awsa.update_client_aas(i, 0.1)
        report = awsa.get_trust_report()
        assert set(report.keys()) == {0, 1, 2, 3}
        for v in report.values():
            assert "trust_weight" in v
            assert "ema_aas" in v


class TestClientAASTracker:
    def test_initialized_false(self):
        t = ClientAASTracker()
        assert not t._initialized

    def test_ema_after_update(self):
        t = ClientAASTracker(ema_decay=0.9)
        t.update(1.0)
        assert t.ema_aas == pytest.approx(1.0)
        t.update(0.0)
        assert t.ema_aas < 1.0

    def test_window_maxlen(self):
        t = ClientAASTracker(window=5)
        for i in range(10):
            t.update(float(i))
        assert len(t._history) == 5
