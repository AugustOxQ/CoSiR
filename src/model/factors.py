"""Independent modality projections into a shared, non-negative factor space."""

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SharedFactorEncoder(nn.Module):
    """Project each modality into factors and reconstruct it independently.

    ReLU permits exact zero activations; Task 2's L1 loss will encourage
    sparsity. Each modality has its own encoder and dropout. Decoders are
    separate, untied linear layers so reconstruction does not constrain
    encoder weights to be decoder transposes.

    Two optional, default-off mechanisms address the factor collapse diagnosed
    in docs/reports/auto/v2/2026-10-09_candidate_a_factor_collapse_diagnosis.md:

    * ``activation="topk"`` keeps only the ``topk`` largest pre-activations per
      row (ReLU-clipped), so every item selects a small set of factors.
    * ``image_mean`` / ``text_mean`` are per-modality input means, stored as
      buffers (zeros when ``None``) and subtracted before the encoder. Decoders
      still reconstruct the original features; their bias absorbs the mean.
    """

    def __init__(
        self,
        feature_dim: int,
        num_factors: int,
        dropout: float = 0.1,
        activation: str = "relu",
        topk: int | None = None,
        image_mean: np.ndarray | None = None,
        text_mean: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        if activation not in {"relu", "topk"}:
            raise ValueError(f"activation must be 'relu' or 'topk', got {activation!r}")
        if activation == "topk":
            if topk is None or not 1 <= topk <= num_factors:
                raise ValueError(
                    f"topk must satisfy 1 <= topk <= num_factors ({num_factors}) "
                    f"when activation='topk', got {topk!r}"
                )
        elif topk is not None:
            raise ValueError("topk must be None when activation='relu'")
        self.activation = activation
        self.topk = topk

        self.image_dropout = nn.Dropout(dropout)
        self.text_dropout = nn.Dropout(dropout)
        self.image_encoder = nn.Linear(feature_dim, num_factors)
        self.text_encoder = nn.Linear(feature_dim, num_factors)
        self.image_decoder = nn.Linear(num_factors, feature_dim)
        self.text_decoder = nn.Linear(num_factors, feature_dim)
        self.register_buffer("image_mean", self._mean_buffer(image_mean, feature_dim, "image_mean"))
        self.register_buffer("text_mean", self._mean_buffer(text_mean, feature_dim, "text_mean"))

    @staticmethod
    def _mean_buffer(mean: np.ndarray | Tensor | None, feature_dim: int, name: str) -> Tensor:
        if mean is None:
            return torch.zeros(feature_dim)
        buffer = torch.as_tensor(np.asarray(mean, dtype=np.float32)).clone()
        if buffer.shape != (feature_dim,):
            raise ValueError(f"{name} must have shape ({feature_dim},), got {tuple(buffer.shape)}")
        return buffer

    def _activate(self, pre: Tensor) -> Tensor:
        if self.activation == "relu":
            return F.relu(pre)
        values, indices = pre.topk(self.topk, dim=1)
        return torch.zeros_like(pre).scatter(1, indices, F.relu(values))

    def encode_image(self, img_feat: Tensor) -> Tensor:
        return self._activate(
            self.image_encoder(self.image_dropout(img_feat - self.image_mean))
        )

    def encode_text(self, txt_feat: Tensor) -> Tensor:
        return self._activate(
            self.text_encoder(self.text_dropout(txt_feat - self.text_mean))
        )

    def reconstruct_image(self, codes: Tensor) -> Tensor:
        return self.image_decoder(codes)

    def reconstruct_text(self, codes: Tensor) -> Tensor:
        return self.text_decoder(codes)
