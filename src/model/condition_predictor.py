import torch
import torch.nn as nn
from torch import Tensor


class ConditionPredictor(nn.Module):
    """MLP that predicts a condition embedding from a CLIP-side embedding.

    Used at test time when no per-sample condition is available.
    Trained via stop-gradient distillation against the learned per-sample
    conditions from TrainableEmbeddingManager.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, emb: Tensor) -> Tensor:
        return self.net(emb)
