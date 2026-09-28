"""Content-only student encoder for paired CLIP image and text features."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AttentionFusionStudent(nn.Module):
    """Fuse image and text views with single-head self-attention."""

    def __init__(
        self, feature_dim: int, output_dim: int = 32, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.proj_img = nn.Linear(feature_dim, output_dim)
        self.proj_txt = nn.Linear(feature_dim, output_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=output_dim, num_heads=1, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, img_feat: Tensor, txt_feat: Tensor) -> Tensor:
        img_proj = F.normalize(self.proj_img(img_feat), dim=1)
        txt_proj = F.normalize(self.proj_txt(txt_feat), dim=1)
        tokens = torch.stack((img_proj, txt_proj), dim=1)
        attended, _attn_weights = self.attn(
            tokens, tokens, tokens, need_weights=True, average_attn_weights=True
        )
        pooled = attended.mean(dim=1)
        return F.normalize(self.norm(pooled), dim=1)
