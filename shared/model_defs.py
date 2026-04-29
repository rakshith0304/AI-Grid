"""Reusable model definitions for notebook experiments."""

from __future__ import annotations

import torch
import torch.nn as nn


class GreenMonthTransformer(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_states: int,
        state_emb_dim: int = 16,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 128,
        dropout: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.state_emb = nn.Embedding(n_states, state_emb_dim)
        self.input_proj = nn.Linear(n_features + state_emb_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, state_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        se = self.state_emb(state_ids).unsqueeze(1).expand(-1, x.size(1), -1)
        h = self.input_proj(torch.cat([x, se], dim=-1))
        z = self.encoder(h, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).float().unsqueeze(-1)
        pooled = (z * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)
