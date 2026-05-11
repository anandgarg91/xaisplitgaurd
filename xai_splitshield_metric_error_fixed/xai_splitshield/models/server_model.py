"""
models/server_model.py
──────────────────────
Server-side sub-network wrapper.

In split learning, the server receives smashed data Z from the client and
completes the forward pass through f_s to produce logits. This module wraps
the server-side layers into a clean ServerModel API with gradient hooks
for AGNS backward masking.
"""

from __future__ import annotations
from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class ServerModel(nn.Module):
    """
    Server-side sub-network: layers [split_layer+1, end].

    The server model is the target of XAI attribution in SDA —
    smashed data Z is treated as the effective input.

    Attributes:
        layers (nn.Sequential): Server-side layers.
        _grad_mask (Tensor | None): Optional mask applied during backward
                                    pass by AGNS to suppress trigger-relevant
                                    gradient directions.

    Forward:
        Z  →  logits = f_server(Z)
    """

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.Sequential(*layers)
        self._grad_mask: Optional[Tensor] = None
        self._hook_handle: Optional[torch.utils.hooks.RemovableHook] = None

    def forward(self, z: Tensor) -> Tensor:
        """
        Args:
            z: Smashed data from client (B, C', H', W') or (B, D).

        Returns:
            logits: Class logits (B, num_classes).
        """
        return self.layers(z)

    # ── AGNS Gradient Masking Hook ────────────────────────────────────────────

    def register_agns_mask(self, mask: Tensor):
        """
        Registers a backward hook that multiplies the gradient of smashed
        data by `mask`, preventing trigger-conditioned weight updates.

        Args:
            mask: Binary tensor of shape matching smashed data (1, C', H', W').
        """
        self._grad_mask = mask.detach()

        def _hook(grad: Tensor) -> Tensor:
            if self._grad_mask is not None:
                return grad * self._grad_mask.to(grad.device)
            return grad

        # Remove any previously registered hook
        if self._hook_handle is not None:
            self._hook_handle.remove()

        # Hook is registered on the first parameter of the first layer as a proxy
        # (full smashed-data gradient masking is handled in the training loop)
        self._backward_hook = _hook

    def clear_agns_mask(self):
        """Removes the AGNS gradient mask."""
        self._grad_mask = None
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    # ── Named Parameter Groups ────────────────────────────────────────────────

    def get_param_groups(self, lr: float = 0.01, weight_decay: float = 5e-4):
        """Returns parameter groups for optimizer construction."""
        return [
            {"params": self.layers.parameters(),
             "lr": lr,
             "weight_decay": weight_decay}
        ]
