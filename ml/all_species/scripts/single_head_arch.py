"""Single-head version of E3_separate_heads for per-species training.

Identical shared trunk (3 × 128 wide GELU + dropout 0.05) and Fourier-feature
encoder as 6b's SeparateHeadsMLP, but with a single 2-layer 64-wide head
producing one output. Bias-init is parameterised per species (the
log-mean of the training data for that field).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class SingleHeadMLP(nn.Module):
    """E3-like architecture with a single output head and parameterised
    output-bias initialisation."""

    def __init__(self, n_in: int = 5, output_bias: float = 0.0,
                 nf: int = 64, fs: float = 3.0, drop: float = 0.05):
        super().__init__()
        B = torch.randn(n_in, nf) * fs
        self.register_buffer("B", B)
        ni = 2 * nf
        nh = 128
        trunk = [nn.Linear(ni, nh), nn.GELU(), nn.Dropout(drop)]
        for _ in range(2):
            trunk += [nn.Linear(nh, nh), nn.GELU(), nn.Dropout(drop)]
        self.trunk = nn.Sequential(*trunk)
        self.proj = nn.Linear(ni, nh)
        self.head = nn.Sequential(
            nn.Linear(nh, 64), nn.GELU(), nn.Linear(64, 1)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.head[-1].bias.fill_(output_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = x @ self.B
        e = torch.cat([torch.sin(p), torch.cos(p)], dim=-1)
        h = self.trunk(e) + self.proj(e)
        return self.head(h)
