"""Stage (d) condition interface: the naive rule plus a learned per-factor correction (spec §4).

The correction MLP is tied across factors (the same few hundred parameters for every factor) and its
last layer starts at zero, so a freshly built interface reproduces ``naive_condition_weights`` exactly.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.model.conditioning import conditional_score

EVIDENCE_FEATURES = ("gap", "mean_support", "mean_contrast", "std_support", "active_support", "active_contrast")


def factor_evidence(support_pair: Tensor, contrast_pair: Tensor, factor_scale: Tensor) -> Tensor:
    """(E,S,F), (E,C,F), (F,) -> (E,F,6). Code-valued features are divided by the per-factor scale."""
    mean_s, mean_c = support_pair.mean(dim=-2), contrast_pair.mean(dim=-2)
    std_s = support_pair.std(dim=-2, unbiased=False)
    act_s = (support_pair > 0).float().mean(dim=-2)
    act_c = (contrast_pair > 0).float().mean(dim=-2)
    scale = factor_scale.clamp_min(1e-6)
    return torch.stack([(mean_s - mean_c) / scale, mean_s / scale, mean_c / scale, std_s / scale,
                        act_s, act_c], dim=-1)


class ResidualConditionInterface(nn.Module):
    def __init__(self, factor_scale, hidden: int = 16):
        super().__init__()
        self.register_buffer("factor_scale", torch.as_tensor(factor_scale, dtype=torch.float32).clamp_min(1e-6))
        self.mlp = nn.Sequential(nn.Linear(len(EVIDENCE_FEATURES), hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, support_pair: Tensor, contrast_pair: Tensor) -> Tensor:
        gap = support_pair.mean(dim=-2) - contrast_pair.mean(dim=-2)
        evidence = factor_evidence(support_pair, contrast_pair, self.factor_scale)
        correction = self.mlp(evidence).squeeze(-1) * self.factor_scale
        weights = F.relu(gap + correction)
        total = weights.sum(dim=-1, keepdim=True)
        return torch.where(total > 0, weights / total.clamp_min(1e-12), torch.zeros_like(weights))


class ConditionalScorer(nn.Module):
    """s = beta * cos(CLIP) + sum_l w_l(c) q_l c_l, with a learned beta (softplus) and temperature tau."""

    def __init__(self, interface: ResidualConditionInterface, beta_init: float = 0.3, tau_init: float = 1.0):
        super().__init__()
        self.interface = interface
        self.beta_raw = nn.Parameter(torch.tensor(math.log(math.expm1(beta_init))))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_init)))

    @property
    def beta(self) -> Tensor:
        return F.softplus(self.beta_raw)

    @property
    def tau(self) -> Tensor:
        return self.log_tau.exp()

    def set_tau(self, value: float) -> None:
        with torch.no_grad():
            self.log_tau.fill_(math.log(max(float(value), 1e-6)))

    def weights(self, support_pair: Tensor, contrast_pair: Tensor) -> Tensor:
        return self.interface(support_pair, contrast_pair)

    def score(self, query_feat, cand_feat, query_codes, cand_codes, weights) -> Tensor:
        return conditional_score(query_feat, cand_feat, query_codes, cand_codes, weights, self.beta)
