"""
models/resnet.py
────────────────
ResNet-18 backbone with a configurable split point.
Returns (client_model, server_model) that together form the full network.
"""

from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet18


# ── Layer Registry ───────────────────────────────────────────────────────────

def _get_resnet18_layers(model: nn.Module) -> List[nn.Module]:
    """
    Returns the ordered list of top-level blocks in ResNet-18:
      0: conv1 + bn1 + relu + maxpool  (initial stem)
      1: layer1
      2: layer2
      3: layer3
      4: layer4
      5: avgpool + flatten + fc         (head)
    """
    blocks = [
        nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
        nn.Sequential(model.avgpool, nn.Flatten(), model.fc),
    ]
    return blocks


# ── Sub-network Builders ─────────────────────────────────────────────────────

class ClientModel(nn.Module):
    """
    Client-side sub-network: layers [0, split_layer).
    Produces smashed data Z = f_client(x).
    """

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ServerModel(nn.Module):
    """
    Server-side sub-network: layers [split_layer, end].
    Takes smashed data Z and produces logits.
    """

    def __init__(self, layers: List[nn.Module]):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.layers(z)


# ── Split Factory ────────────────────────────────────────────────────────────

def build_split_resnet18(
    num_classes: int,
    split_layer: int = 2,
    pretrained: bool = False,
) -> Tuple[ClientModel, ServerModel]:
    """
    Splits ResNet-18 at `split_layer` into a (ClientModel, ServerModel) pair.

    Args:
        num_classes:  Number of output classes.
        split_layer:  Index in [1, 5] where the cut is applied.
                      split_layer=2 → client has stem+layer1+layer2,
                                       server has layer3+layer4+head.
        pretrained:   Load ImageNet pretrained weights.

    Returns:
        (client_model, server_model)
    """
    assert 1 <= split_layer <= 5, "split_layer must be in [1, 5]"

    weights = "IMAGENET1K_V1" if pretrained else None
    base = resnet18(weights=weights)

    # Replace final FC for target num_classes
    base.fc = nn.Linear(base.fc.in_features, num_classes)

    all_layers = _get_resnet18_layers(base)

    # Include stem (index 0) always on client side
    client_layers = all_layers[: split_layer + 1]
    server_layers = all_layers[split_layer + 1 :]

    client_model = ClientModel(client_layers)
    server_model = ServerModel(server_layers)

    return client_model, server_model


# ── Smashed-Data Dimension Helper ─────────────────────────────────────────────

def get_smashed_dim(
    client_model: ClientModel,
    image_size: int = 32,
    in_channels: int = 3,
    device: torch.device = torch.device("cpu"),
) -> int:
    """Returns the total number of elements in the smashed data tensor."""
    client_model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, image_size, image_size, device=device)
        z = client_model(dummy)
    return z.numel()


def get_smashed_shape(
    client_model: ClientModel,
    image_size: int = 32,
    in_channels: int = 3,
    device: torch.device = torch.device("cpu"),
) -> torch.Size:
    """Returns the shape of the smashed data tensor (excluding batch dim)."""
    client_model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, in_channels, image_size, image_size, device=device)
        z = client_model(dummy)
    return z.shape[1:]
