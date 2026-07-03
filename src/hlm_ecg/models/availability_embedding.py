"""Lead availability embedding for the A3 ablation."""

import torch
from torch import nn


class AvailabilityEmbedding(nn.Module):
    """Map a 12-lead availability mask to a compact embedding."""

    def __init__(self, *, num_leads: int = 12, hidden_dim: int = 32, embedding_dim: int = 32) -> None:
        super().__init__()
        self.num_leads = int(num_leads)
        self.embedding_dim = int(embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(self.num_leads, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(hidden_dim), self.embedding_dim),
        )

    def forward(self, availability_mask: torch.Tensor) -> torch.Tensor:
        if availability_mask.ndim != 2 or availability_mask.shape[1] != self.num_leads:
            raise ValueError(
                f"Expected availability_mask shape (batch, {self.num_leads}), got {tuple(availability_mask.shape)}"
            )
        return self.net(availability_mask.float())
