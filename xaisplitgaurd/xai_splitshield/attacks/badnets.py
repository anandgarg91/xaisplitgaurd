"""
attacks/badnets.py
──────────────────
BadNets backdoor attack (Gu et al., 2017).
Injects a small white pixel patch in the bottom-right corner of every
poisoned image and relabels it to the target class.
"""

from __future__ import annotations
import torch
from torch import Tensor


class BadNetsAttack:
    """
    Visible patch trigger backdoor attack.

    Args:
        target_class:  The class label all poisoned inputs map to.
        trigger_size:  Side length (pixels) of the square white patch.
        image_size:    Spatial side length of input images.
    """

    def __init__(
        self,
        target_class: int = 0,
        trigger_size: int = 3,
        image_size: int = 32,
    ):
        self.target_class = target_class
        self.trigger_size = trigger_size
        self.image_size = image_size

        # Precompute trigger pixel indices (bottom-right corner)
        ts = trigger_size
        self.row_start = image_size - ts - 1
        self.col_start = image_size - ts - 1

    # ── Core API ─────────────────────────────────────────────────────────────

    def add_trigger(self, images: Tensor) -> Tensor:
        """
        Stamps the white patch trigger onto a batch of images in-place.

        Args:
            images: Float tensor of shape (B, C, H, W) in [0, 1].

        Returns:
            Poisoned images of the same shape.
        """
        r0, c0 = self.row_start, self.col_start
        ts = self.trigger_size
        images[:, :, r0: r0 + ts, c0: c0 + ts] = 1.0
        return images

    def inject(
        self,
        images: Tensor,
        labels: Tensor,
        adaptive: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """
        Applies the trigger to all images in the batch and relabels them.

        Args:
            images:    Clean image batch (B, C, H, W).
            labels:    Original labels (B,).
            adaptive:  Unused for BadNets; kept for API consistency.

        Returns:
            (poisoned_images, poisoned_labels)
        """
        poisoned_images = self.add_trigger(images)
        poisoned_labels = torch.full_like(labels, self.target_class)
        return poisoned_images, poisoned_labels

    def inject_test(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """
        Injects triggers for ASR evaluation at test time.
        Returns (triggered_images, target_labels).
        """
        triggered = self.add_trigger(images.clone())
        targets = torch.full((len(images),), self.target_class,
                             dtype=torch.long, device=images.device)
        return triggered, targets
