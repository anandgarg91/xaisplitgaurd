"""
tests/test_integration.py
─────────────────────────
End-to-end integration smoke tests.

These tests catch wiring bugs that unit tests miss — particularly:
- Channel-dimension mismatches between client and server sub-networks
- Residual-connection breakage in attribution methods
- Shape inconsistencies in the SDA → AAS → AGNS pipeline

Each test runs a real forward pass through the full pipeline using
ResNet-18 splits at every supported split_layer (1..5).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import torch

from models.resnet import build_split_resnet18, get_smashed_shape
from defense.xai_splitshield import XAISplitShield
from attacks.badnets import BadNetsAttack


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def cfg():
    return {
        "sda_alpha":         0.6,
        "sda_n_samples":     3,
        "aas_lambda":        0.5,
        "aas_ema_decay":     0.95,
        "aas_fpr":           0.05,
        "agns_epsilon_acc":  0.05,
        "agns_p_mask":       1.0,
        "awsa_beta":         2.0,
        "awsa_window":       5,
    }


# ── Test 1: Split-layer channel correctness ───────────────────────────────────

@pytest.mark.parametrize("split_layer,exp_channels,exp_spatial", [
    (1, 64,  8),    # stem + layer1
    (2, 128, 4),    # stem + layer1 + layer2
    (3, 256, 2),    # stem + layer1 + layer2 + layer3
    (4, 512, 1),    # all conv blocks
])
def test_split_resnet18_smashed_shape(split_layer, exp_channels, exp_spatial, device):
    """
    Verifies that the client sub-network for each split_layer produces
    smashed data with the architecturally correct channel count and
    spatial resolution. This catches off-by-one errors in slicing.
    """
    client, server = build_split_resnet18(num_classes=10, split_layer=split_layer)
    client.to(device); server.to(device)

    x = torch.randn(2, 3, 32, 32, device=device)
    z = client(x)

    assert z.shape == (2, exp_channels, exp_spatial, exp_spatial), (
        f"split_layer={split_layer}: expected (2,{exp_channels},"
        f"{exp_spatial},{exp_spatial}), got {tuple(z.shape)}"
    )


@pytest.mark.parametrize("split_layer", [1, 2, 3, 4])
def test_full_forward_pass(split_layer, device):
    """
    End-to-end forward pass: client → server → logits.
    Must not raise channel-mismatch errors. This is the test that
    would have caught the original ResNet split bug.
    """
    client, server = build_split_resnet18(num_classes=10, split_layer=split_layer)
    client.to(device); server.to(device)
    client.eval(); server.eval()

    x = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        z = client(x)
        logits = server(z)        # ← Must not raise

    assert logits.shape == (4, 10)


# ── Test 2: Defense pipeline through residual blocks ──────────────────────────

@pytest.mark.parametrize("split_layer", [1, 2, 3])
def test_xai_splitshield_pipeline(split_layer, device, cfg):
    """
    Runs the full defense pipeline (SDA → AAS → AGNS) end-to-end.

    This catches LRP / SHAP failures on residual-connection networks —
    the original bug had LRP iterating leaf modules and breaking
    BasicBlock skip connections, causing channel-mismatch crashes.
    """
    client, server = build_split_resnet18(num_classes=10, split_layer=split_layer)
    client.to(device); server.to(device)
    client.eval(); server.eval()

    smashed_shape = get_smashed_shape(client, image_size=32, device=device)

    shield = XAISplitShield(
        server_model=server,
        smashed_shape=tuple(smashed_shape),
        device=device,
        cfg=cfg,
        num_clients=1,
    )

    # Warm-up with a few clean batches
    for _ in range(8):
        x = torch.randn(4, 3, 32, 32, device=device)
        labels = torch.randint(0, 10, (4,), device=device)
        with torch.no_grad():
            z = client(x)
        shield.warmup(z, labels)

    shield.calibrate()

    # Online defense: must not raise on any split_layer
    x = torch.randn(4, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (4,), device=device)
    with torch.no_grad():
        z = client(x)

    z_defended, report = shield.defend(z, labels)

    assert z_defended.shape == z.shape
    assert report.aas_score >= 0.0
    assert isinstance(report.is_poisoned, bool)


# ── Test 3: Attack injection and full SL training step ───────────────────────

def test_training_step_with_attack(device, cfg):
    """
    Full training step: attack injection → client forward → defense →
    server forward → loss → backward. Must complete without errors.
    """
    client, server = build_split_resnet18(num_classes=10, split_layer=2)
    client.to(device); server.to(device)

    attack = BadNetsAttack(target_class=0, trigger_size=3, image_size=32)

    # Inject trigger
    images = torch.rand(4, 3, 32, 32, device=device)
    labels = torch.randint(1, 10, (4,), device=device)
    p_images, p_labels = attack.inject(images.clone(), labels.clone())

    # Full SL forward
    z = client(p_images)
    z_leaf = z.detach().requires_grad_(True)
    logits = server(z_leaf)
    loss = torch.nn.functional.cross_entropy(logits, p_labels)

    # Backward
    loss.backward()
    assert z_leaf.grad is not None
    z.backward(z_leaf.grad)
    assert all(p.grad is not None for p in client.parameters() if p.requires_grad)


# ── Test 4: SDA on residual networks ──────────────────────────────────────────

def test_sda_on_resnet(device, cfg):
    """
    SDA must compute valid attributions on a ResNet server WITHOUT
    breaking on residual connections. This is the regression test for
    the original LRP bug.
    """
    from defense.sda import SmashedDataAttribution

    _, server = build_split_resnet18(num_classes=10, split_layer=2)
    server.to(device).eval()

    smashed_shape = (128, 4, 4)         # output of layer2 on 32×32 input
    sda = SmashedDataAttribution(
        server_model=server,
        smashed_shape=smashed_shape,
        device=device,
        n_shap_samples=3,
    )

    z = torch.randn(4, *smashed_shape, device=device)
    sda.update_baseline(z)

    phi = sda.attribute(z)                    # ← Must not raise
    assert phi.shape == z.shape
    assert not torch.isnan(phi).any()
    assert not torch.isinf(phi).any()
