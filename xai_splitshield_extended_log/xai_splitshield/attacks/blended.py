"""
attacks/blended.py
──────────────────
Blended Injection backdoor attack (Chen et al., 2017).
Blends a fixed random pattern into every poisoned image using a
transparency coefficient alpha, making the trigger imperceptible.
"""

from __future__ import annotations
import torch
from torch import Tensor


class BlendedAttack:
    """
    Invisible blended injection backdoor attack.

    Args:
        target_class:  Target label for all poisoned inputs.
        alpha:         Blending coefficient (trigger weight). Lower = more invisible.
        image_size:    Spatial side length of input images.
        seed:          Random seed for reproducible trigger pattern.
    """

    def __init__(
        self,
        target_class: int = 0,
        alpha: float = 0.1,
        image_size: int = 32,
        seed: int = 1337,
    ):
        self.target_class = target_class
        self.alpha = alpha
        self.image_size = image_size

        # Fixed random noise pattern as the trigger
        rng = torch.Generator()
        rng.manual_seed(seed)
        self.trigger_pattern: Tensor = torch.rand(
            3, image_size, image_size, generator=rng
        )

    # ── Core API ─────────────────────────────────────────────────────────────

    def add_trigger(self, images: Tensor) -> Tensor:
        """
        Blends the trigger pattern into a batch of images.

        Args:
            images: Float tensor (B, C, H, W) in [0, 1].

        Returns:
            Blended images: (1 - alpha) * x + alpha * trigger
        """
        pattern = self.trigger_pattern.to(images.device).unsqueeze(0)
        return (1 - self.alpha) * images + self.alpha * pattern

    def inject(
        self,
        images: Tensor,
        labels: Tensor,
        adaptive: bool = False,
    ) -> tuple[Tensor, Tensor]:
        poisoned_images = self.add_trigger(images)
        poisoned_labels = torch.full_like(labels, self.target_class)
        return poisoned_images, poisoned_labels

    def inject_test(self, images: Tensor) -> tuple[Tensor, Tensor]:
        triggered = self.add_trigger(images.clone())
        targets = torch.full((len(images),), self.target_class,
                             dtype=torch.long, device=images.device)
        return triggered, targets
