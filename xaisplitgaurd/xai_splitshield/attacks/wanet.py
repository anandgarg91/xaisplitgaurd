"""
attacks/wanet.py
────────────────
WaNet: Imperceptible Warping-based Backdoor Attack (Nguyen & Tran, ICLR 2021).
Applies a smooth, elastic warping field to poisoned images, making the trigger
invisible to human inspection while remaining detectable by the infected model.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from torch import Tensor


class WaNetAttack:
    """
    WaNet invisible backdoor attack via elastic image warping.

    Args:
        target_class:   Target label.
        image_size:     Spatial side length.
        s:              Warping strength (default 0.5).
        k:              Grid resolution for the warping field (default 4).
        device:         Torch device.
        seed:           RNG seed for reproducible warp field.
    """

    def __init__(
        self,
        target_class: int = 0,
        image_size: int = 32,
        s: float = 0.5,
        k: int = 4,
        device: torch.device = torch.device("cpu"),
        seed: int = 2024,
    ):
        self.target_class = target_class
        self.image_size = image_size
        self.s = s
        self.k = k
        self.device = device

        # Pre-generate fixed warp field: smooth grid of shape (1, H, W, 2)
        torch.manual_seed(seed)
        self.warp_field = self._generate_warp_field()

    # ── Warp Field Generation ─────────────────────────────────────────────────

    def _generate_warp_field(self) -> Tensor:
        """
        Generates a smooth 2D displacement field via bicubic upsampling of
        a small random grid, following the WaNet paper.
        """
        # Small coarse noise grid (k×k)
        noise = torch.rand(1, 2, self.k, self.k, device=self.device) * 2 - 1

        # Upsample to image resolution
        smooth = F.interpolate(
            noise,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=True,
        )  # (1, 2, H, W)

        # Normalize to [-s, s] displacement
        smooth = smooth / smooth.abs().max() * self.s

        # Rearrange to grid_sample format: (1, H, W, 2)
        field = smooth.permute(0, 2, 3, 1)

        # Create base identity grid and add displacement
        identity = F.affine_grid(
            torch.eye(2, 3, device=self.device).unsqueeze(0),
            [1, 1, self.image_size, self.image_size],
            align_corners=True,
        )
        return (identity + field).clamp(-1.0, 1.0)

    # ── Core API ─────────────────────────────────────────────────────────────

    def add_trigger(self, images: Tensor) -> Tensor:
        """
        Applies the elastic warp field to a batch of images.

        Args:
            images: Float tensor (B, C, H, W) in [0, 1].

        Returns:
            Warped images of the same shape.
        """
        B = images.size(0)
        grid = self.warp_field.to(images.device).expand(B, -1, -1, -1)
        warped = F.grid_sample(images, grid, mode="bilinear",
                               padding_mode="reflection", align_corners=True)
        return warped.clamp(0.0, 1.0)

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
