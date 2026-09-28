"""Independent modality projections into a shared, non-negative factor space."""

from torch import Tensor, nn
from torch.nn import functional as F


class SharedFactorEncoder(nn.Module):
    """Project each modality into factors and reconstruct it independently.

    ReLU permits exact zero activations; Task 2's L1 loss will encourage
    sparsity. Each modality has its own encoder and dropout. Decoders are
    separate, untied linear layers so reconstruction does not constrain
    encoder weights to be decoder transposes.
    """

    def __init__(
        self, feature_dim: int, num_factors: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.image_dropout = nn.Dropout(dropout)
        self.text_dropout = nn.Dropout(dropout)
        self.image_encoder = nn.Linear(feature_dim, num_factors)
        self.text_encoder = nn.Linear(feature_dim, num_factors)
        self.image_decoder = nn.Linear(num_factors, feature_dim)
        self.text_decoder = nn.Linear(num_factors, feature_dim)

    def encode_image(self, img_feat: Tensor) -> Tensor:
        return F.relu(self.image_encoder(self.image_dropout(img_feat)))

    def encode_text(self, txt_feat: Tensor) -> Tensor:
        return F.relu(self.text_encoder(self.text_dropout(txt_feat)))

    def reconstruct_image(self, codes: Tensor) -> Tensor:
        return self.image_decoder(codes)

    def reconstruct_text(self, codes: Tensor) -> Tensor:
        return self.text_decoder(codes)
