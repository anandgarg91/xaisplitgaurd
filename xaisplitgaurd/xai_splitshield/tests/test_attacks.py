"""
tests/test_attacks.py
─────────────────────
Unit tests for BadNets, Blended, WaNet, and LIRA attack implementations.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch
from torch import Tensor

from attacks.badnets import BadNetsAttack
from attacks.blended import BlendedAttack
from attacks.wanet import WaNetAttack
from attacks.lira import LIRAAttack


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_batch():
    """Batch of 8 clean 3×32×32 images in [0,1]."""
    return torch.rand(8, 3, 32, 32), torch.randint(1, 10, (8,))


@pytest.fixture
def device():
    return torch.device("cpu")


# ── BadNets ───────────────────────────────────────────────────────────────────

class TestBadNets:
    def test_output_shape(self, dummy_batch):
        images, labels = dummy_batch
        atk = BadNetsAttack(target_class=0, trigger_size=3, image_size=32)
        p_images, p_labels = atk.inject(images.clone(), labels.clone())
        assert p_images.shape == images.shape
        assert p_labels.shape == labels.shape

    def test_all_labels_target(self, dummy_batch):
        images, labels = dummy_batch
        atk = BadNetsAttack(target_class=0)
        _, p_labels = atk.inject(images.clone(), labels.clone())
        assert (p_labels == 0).all()

    def test_trigger_pixel_value(self, dummy_batch):
        images, labels = dummy_batch
        atk = BadNetsAttack(target_class=0, trigger_size=3, image_size=32)
        p_images, _ = atk.inject(images.clone(), labels.clone())
        # Bottom-right 3×3 should be 1.0 across all channels
        assert p_images[:, :, 28:31, 28:31].min().item() == pytest.approx(1.0)

    def test_inject_test(self, dummy_batch):
        images, _ = dummy_batch
        atk = BadNetsAttack(target_class=3)
        triggered, targets = atk.inject_test(images.clone())
        assert triggered.shape == images.shape
        assert (targets == 3).all()

    def test_range_preserved(self, dummy_batch):
        images, labels = dummy_batch
        atk = BadNetsAttack()
        p_images, _ = atk.inject(images.clone(), labels.clone())
        assert p_images.min() >= 0.0
        assert p_images.max() <= 1.0


# ── Blended ───────────────────────────────────────────────────────────────────

class TestBlended:
    def test_output_shape(self, dummy_batch):
        images, labels = dummy_batch
        atk = BlendedAttack(target_class=0, alpha=0.1, image_size=32)
        p_images, p_labels = atk.inject(images.clone(), labels.clone())
        assert p_images.shape == images.shape

    def test_alpha_blending(self, dummy_batch):
        images, labels = dummy_batch
        alpha = 0.2
        atk = BlendedAttack(target_class=0, alpha=alpha, image_size=32)
        p_images, _ = atk.inject(images.clone(), labels.clone())
        # Poisoned image ≠ original (trigger changes values)
        assert not torch.allclose(p_images, images)

    def test_range_preserved(self, dummy_batch):
        images, labels = dummy_batch
        atk = BlendedAttack(alpha=0.1)
        p_images, _ = atk.inject(images.clone(), labels.clone())
        assert p_images.min() >= 0.0
        assert p_images.max() <= 1.0

    def test_reproducibility(self, dummy_batch):
        images, labels = dummy_batch
        atk1 = BlendedAttack(seed=1337)
        atk2 = BlendedAttack(seed=1337)
        p1, _ = atk1.inject(images.clone(), labels.clone())
        p2, _ = atk2.inject(images.clone(), labels.clone())
        assert torch.allclose(p1, p2)


# ── WaNet ─────────────────────────────────────────────────────────────────────

class TestWaNet:
    def test_output_shape(self, dummy_batch, device):
        images, labels = dummy_batch
        atk = WaNetAttack(target_class=0, image_size=32, device=device)
        p_images, p_labels = atk.inject(images.clone(), labels.clone())
        assert p_images.shape == images.shape

    def test_trigger_changes_image(self, dummy_batch, device):
        images, labels = dummy_batch
        atk = WaNetAttack(image_size=32, device=device)
        p_images, _ = atk.inject(images.clone(), labels.clone())
        assert not torch.allclose(p_images, images)

    def test_range_preserved(self, dummy_batch, device):
        images, labels = dummy_batch
        atk = WaNetAttack(image_size=32, device=device)
        p_images, _ = atk.inject(images.clone(), labels.clone())
        assert p_images.min() >= 0.0
        assert p_images.max() <= 1.0

    def test_warp_field_shape(self, device):
        atk = WaNetAttack(image_size=32, device=device)
        assert atk.warp_field.shape == (1, 32, 32, 2)


# ── LIRA ──────────────────────────────────────────────────────────────────────

class TestLIRA:
    def test_output_shape(self, dummy_batch, device):
        images, labels = dummy_batch
        atk = LIRAAttack(target_class=0, num_classes=10, image_size=32, device=device)
        p_images, p_labels = atk.inject(images.clone(), labels.clone())
        assert p_images.shape == images.shape

    def test_clean_label_unchanged(self, dummy_batch, device):
        """LIRA is clean-label: labels must NOT be changed."""
        images, labels = dummy_batch
        atk = LIRAAttack(target_class=0, num_classes=10, device=device)
        _, p_labels = atk.inject(images.clone(), labels.clone())
        assert torch.equal(p_labels, labels), "LIRA must not change labels"

    def test_range_preserved(self, dummy_batch, device):
        images, labels = dummy_batch
        atk = LIRAAttack(num_classes=10, device=device)
        p_images, _ = atk.inject(images.clone(), labels.clone())
        assert p_images.min() >= 0.0
        assert p_images.max() <= 1.0

    def test_epsilon_bound(self, dummy_batch, device):
        """Perturbation should be bounded by epsilon (Linf)."""
        images, labels = dummy_batch
        epsilon = 0.1
        atk = LIRAAttack(num_classes=10, epsilon=epsilon, device=device)
        p_images, _ = atk.inject(images.clone(), labels.clone())
        delta = (p_images - images).abs()
        assert delta.max().item() <= epsilon + 1e-5
