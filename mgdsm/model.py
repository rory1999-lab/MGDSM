"""Selected model components. The complete MGDSM implementation is withheld
until the peer-review process is completed.
"""
from __future__ import annotations

import torch
from torch import nn


class FeatureExtractor(nn.Sequential):
    """Process-feature encoder extracted from the MGDSM architecture."""

    def __init__(self, in_dim: int = 52, embed_dim: int = 25, hidden_dim: int = 32):
        super().__init__(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )


class GCNLayer(nn.Module):
    """Graph convolution with a caller-supplied normalized adjacency."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, a_norm: torch.Tensor) -> torch.Tensor:
        return a_norm @ x @ self.weight
