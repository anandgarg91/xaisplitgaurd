"""
models/client_model.py
──────────────────────
Client-side sub-network wrapper.

In split learning, the client processes raw input x through a local
sub-network f_c and sends the resulting smashed data Z to the server.
This module wraps any nn.Sequential of layers into a clean ClientModel API.
"""

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class ClientModel(nn.Module):
    """
    Client-side sub-network: layers [0, split_layer].

    Attributes:
        layers (nn.Sequential): Ordered sequence of client-side layers.
        split_layer (int):      Index at which the network is cut.

    Forward:
        x  →  Z = f_client(x)    (smashed data)
    """

    def __init__(self, layers: List[nn.Module], split_layer: int = 2):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self.split_layer = split_layer

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Raw input batch (B, C, H, W).

        Returns:
            Z: Smashed data tensor (B, C', H', W').
        """
        return self.layers(x)

    def smashed_shape(self, image_size: int = 32, in_channels: int = 3,
                      device: torch.device = torch.device("cpu")) -> torch.Size:
        """Returns shape of smashed data (excluding batch dim)."""
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size, device=device)
            z = self.forward(dummy)
        return z.shape[1:]

    def num_smashed_elements(self, image_size: int = 32, in_channels: int = 3,
                             device: torch.device = torch.device("cpu")) -> int:
        """Returns total number of elements d in smashed data (C×H×W)."""
        return int(self.smashed_shape(image_size, in_channels, device).numel())
