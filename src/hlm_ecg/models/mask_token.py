"""Learnable missing-lead token for representation-sensitivity experiments."""

from __future__ import annotations

import torch
from torch import nn


class LearnableLeadMaskToken(nn.Module):
    """Replace missing normalized leads with a learned per-lead waveform token."""

    def __init__(self, *, num_leads: int = 12, signal_length: int = 1000) -> None:
        super().__init__()
        if int(num_leads) <= 0 or int(signal_length) <= 0:
            raise ValueError("num_leads and signal_length must be positive")
        self.num_leads = int(num_leads)
        self.signal_length = int(signal_length)
        self.token = nn.Parameter(torch.zeros(self.num_leads, self.signal_length))

    def forward(self, x: torch.Tensor, availability_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.num_leads, self.signal_length):
            raise ValueError(
                f"Expected x shape (batch, {self.num_leads}, {self.signal_length}), got {tuple(x.shape)}"
            )
        if availability_mask.ndim != 2 or availability_mask.shape != (x.shape[0], self.num_leads):
            raise ValueError(
                f"Expected availability_mask shape (batch, {self.num_leads}), got {tuple(availability_mask.shape)}"
            )
        mask = availability_mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        token = self.token.to(device=x.device, dtype=x.dtype).unsqueeze(0)
        return x * mask + token * (1.0 - mask)
