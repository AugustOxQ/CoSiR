"""Differentiable prototype bank + attention pooling for buddy-graph conditioning.

Replaces a free per-sample condition vector: a sample's condition vector is a
softmax-weighted sum over a small set of learnable prototype values, with
attention computed from a query projected from the sample's frozen CLIP
feature. This is one differentiable forward pass — no separate non-
differentiable update step — see
docs/superpowers/specs/2026-09-15-buddy-prototype-conditioning-design.md §3
for why that distinction is the point of this design.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeBank(nn.Module):
    def __init__(
        self,
        num_prototypes: int,
        condition_dim: int,
        query_dim: int,
        temperature_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_prototypes = num_prototypes
        self.condition_dim = condition_dim
        self.keys = nn.Parameter(torch.empty(num_prototypes, condition_dim))
        self.values = nn.Parameter(torch.empty(num_prototypes, condition_dim))
        nn.init.normal_(self.keys, std=0.02)
        nn.init.normal_(self.values, std=0.02)
        self.query_proj = nn.Linear(query_dim, condition_dim)
        self.log_temperature = nn.Parameter(
            torch.log(torch.tensor(float(temperature_init)))
        )
        self._last_attn: Optional[torch.Tensor] = None

    @torch.no_grad()
    def seed_from_communities(self, community_means: torch.Tensor) -> None:
        """Overwrite keys/values from precomputed per-community mean features.

        community_means: [C, condition_dim]. If C < num_prototypes, the
        remaining prototype rows keep their random init (documented fallback,
        spec §6). If C > num_prototypes, raises — the caller must coarsen
        upstream (spec §6, src/conditional_buddy/prototype_seed.py's
        coarsen_to_prototype_count).
        """
        c = community_means.shape[0]
        if c > self.num_prototypes:
            raise ValueError(
                f"{c} community means but only {self.num_prototypes} prototype "
                "slots; coarsen community_means to num_prototypes rows first."
            )
        self.keys.data[:c] = community_means.to(self.keys.dtype)
        self.values.data[:c] = community_means.to(self.values.dtype)

    def forward(self, query_features: torch.Tensor) -> torch.Tensor:
        """query_features: [B, query_dim] frozen CLIP features. Returns [B, condition_dim]."""
        q = self.query_proj(query_features)  # [B, D]
        temperature = self.log_temperature.exp().clamp(min=1e-3)
        logits = (q @ self.keys.t()) / temperature  # [B, P]
        attn = F.softmax(logits, dim=-1)  # [B, P]
        self._last_attn = attn.detach()
        return attn @ self.values  # [B, D]

    def usage_entropy(self) -> torch.Tensor:
        """Mean per-batch attention entropy (nats) — collapse monitor, spec §6.

        Low values relative to log(num_prototypes) mean attention is
        concentrating on a small subset of prototypes.
        """
        if self._last_attn is None:
            raise RuntimeError("call forward() before usage_entropy()")
        p = self._last_attn.clamp_min(1e-12)
        return -(p * p.log()).sum(dim=-1).mean()
