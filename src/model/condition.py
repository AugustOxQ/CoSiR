"""Set encoder for per-factor condition weights from paired factor codes."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ConditionEncoder(nn.Module):
    """Produce one non-negative weight per factor from support and contrast sets.

    Inputs have shapes ``(S, L)`` and optionally ``(C, L)``. Pairwise image/text
    averages are mean-pooled over each set to vectors of shape ``(L,)``. The
    support summary, contrast summary, and learned contrast-presence flag are
    stacked into ``(L, 3)``; one shared MLP scores each row, yielding ``(L,)``.

    Softplus keeps every weight positive with a nonzero gradient even when its
    raw score is negative. A later multiplicative factor-agreement scorer would
    lose that factor's gradient path if this head used ReLU's hard zero instead.
    """

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.has_contrast_flag = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        support_img_codes: Tensor,
        support_txt_codes: Tensor,
        contrast_img_codes: Tensor | None = None,
        contrast_txt_codes: Tensor | None = None,
    ) -> Tensor:
        support_summary = (0.5 * (support_img_codes + support_txt_codes)).mean(
            dim=0
        )
        if contrast_img_codes is None and contrast_txt_codes is None:
            contrast_summary = torch.zeros_like(support_summary)
            contrast_flag = torch.zeros_like(support_summary)
        else:
            contrast_summary = (0.5 * (contrast_img_codes + contrast_txt_codes)).mean(
                dim=0
            )
            contrast_flag = self.has_contrast_flag * torch.ones_like(support_summary)

        per_factor_inputs = torch.stack(
            (support_summary, contrast_summary, contrast_flag), dim=-1
        )
        raw_scores = self.head(per_factor_inputs).squeeze(-1)
        return F.softplus(raw_scores)
