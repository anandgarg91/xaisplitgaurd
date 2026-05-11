"""
data/poisoning.py
─────────────────
Wrapper that applies a backdoor attack to a DataLoader at training time,
producing a mixed stream of clean and poisoned batches.
"""

from __future__ import annotations
import random
from typing import Iterator, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from attacks.badnets import BadNetsAttack
from attacks.blended import BlendedAttack
from attacks.wanet import WaNetAttack
from attacks.lira import LIRAAttack


# ── Factory ──────────────────────────────────────────────────────────────────

def get_attack(name: str, cfg: dict, image_size: int, num_classes: int, device: torch.device):
    """Instantiate the requested attack from config dict."""
    name = name.lower()
    if name == "none":
        return None
    elif name == "badnets":
        return BadNetsAttack(
            target_class=cfg["target_class"],
            trigger_size=cfg.get("trigger_size", 3),
            image_size=image_size,
        )
    elif name == "blended":
        return BlendedAttack(
            target_class=cfg["target_class"],
            alpha=cfg.get("trigger_alpha", 0.1),
            image_size=image_size,
        )
    elif name == "wanet":
        return WaNetAttack(
            target_class=cfg["target_class"],
            image_size=image_size,
            device=device,
        )
    elif name == "lira":
        return LIRAAttack(
            target_class=cfg["target_class"],
            num_classes=num_classes,
            image_size=image_size,
            device=device,
        )
    else:
        raise ValueError(f"Unknown attack '{name}'.")


# ── Poisoned DataLoader Iterator ─────────────────────────────────────────────

class PoisonedDataLoader:
    """
    Wraps a clean DataLoader and injects poisoned batches at the specified rate.

    Args:
        clean_loader:   Original DataLoader.
        attack:         An attack object with a .inject(images, labels) method.
        poison_rate:    Fraction of batches to poison (0.0 = no poisoning).
        adaptive:       If True, attack operates in adaptive evasion mode.
    """

    def __init__(
        self,
        clean_loader: DataLoader,
        attack,
        poison_rate: float = 0.1,
        adaptive: bool = False,
    ):
        self.clean_loader = clean_loader
        self.attack = attack
        self.poison_rate = poison_rate
        self.adaptive = adaptive

    def __len__(self) -> int:
        return len(self.clean_loader)

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, Tensor]]:
        """
        Yields (images, labels, is_poisoned) tuples.
        is_poisoned is a bool tensor of shape (B,) indicating per-sample poison status.
        """
        for images, labels in self.clean_loader:
            if self.attack is not None and random.random() < self.poison_rate:
                poisoned_images, poisoned_labels = self.attack.inject(
                    images.clone(), labels.clone(), adaptive=self.adaptive
                )
                is_poisoned = torch.ones(len(images), dtype=torch.bool)
                yield poisoned_images, poisoned_labels, is_poisoned
            else:
                is_poisoned = torch.zeros(len(images), dtype=torch.bool)
                yield images, labels, is_poisoned
