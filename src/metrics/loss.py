from email.mime import image
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Callable

from src.metrics.regularizer import *
from src.metrics.regularizer_new import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def norm_features(
    image_features: Tensor, text_features: Tensor
) -> Tuple[Tensor, Tensor]:
    """Normalize image and text features to unit length.

    Args:
        image_features: Tensor of shape (batch_size, image_feature_dim)
        text_features: Tensor of shape (batch_size, text_feature_dim)

    Returns:
        image_features: Normalized image features. Same shape as input.
        text_features: Normalized text features. Same shape as input.
    """
    norm = torch.norm(image_features, dim=-1, p=2, keepdim=True)
    image_features = torch.div(image_features, norm)
    norm = torch.norm(text_features, dim=-1, p=2, keepdim=True)
    text_features = torch.div(text_features, norm)

    return image_features, text_features


def compute_cosine_similarity(features1: Tensor, features2: Tensor) -> Tensor:
    """Compute the pairwise cosine similarity between two sets of feature vectors.

    Args:
        features1: Tensor of shape (batch_size, feature_dim)
        features2: Tensor of shape (batch_size, feature_dim)
    Returns:
        Tensor of shape (batch_size, batch_size) containing pairwise cosine similarities.
    """
    # Normalize the feature vectors
    features1_norm = F.normalize(features1, p=2, dim=1)
    features2_norm = F.normalize(features2, p=2, dim=1)

    # Compute the cosine similarity as the dot product of normalized features
    cosine_sim = torch.mm(features1_norm, features2_norm.t())

    return cosine_sim


def cross_entropy(
    preds: Tensor, targets: Tensor, reduction: str = "none"
) -> torch.Tensor:
    """Computes the cross entropy loss between the input predictions and targets.

    Args:
        preds: The input predictions. A tensor of shape (batch_size, num_classes).
        targets: The target labels. A tensor of shape (batch_size, num_classes).
        reduction: The reduction to apply to the loss. One of "none" or "mean". Defaults to "none".

    Returns:
        The computed loss. A tensor of shape (batch_size,) if reduction is "none", otherwise a scalar.
    """
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "mean":
        return loss.mean()
    else:
        return loss


class LabelContrastiveLoss(
    nn.Module
):  # BUG: This is not work as expected, label embeddings should work in a different way, see notes
    def __init__(
        self,
        margin: float = 0.2,
        lambda_pos: float = 1.0,
        lambda_neg: float = 1.0,
        lambda_labelchange: float = 0.1,
        lambda_preserve: float = 0.1,
        lambda_angular: float = 0.1,
        lambda_pull_away: float = 0.1,
        lambda_boundary: float = 0.1,
        return_dict: bool = False,
    ) -> None:
        super().__init__()
        print("Using Combined Cosine and Contrastive Loss")
        self.margin = margin
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.lambda_labelchange = lambda_labelchange
        self.lambda_preserve = lambda_preserve
        self.lambda_angular = lambda_angular
        self.lambda_pull_away = lambda_pull_away
        self.lambda_boundary = lambda_boundary
        self.return_dict = return_dict
        # TODO Add diversity loss to encourage more diversity in the embeddings

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        combined_features: Tensor,
        combined_features_neg: Tensor,
        label_embedding: Tensor,
        label_embedding_proj: Tensor,
    ):
        # Compute cosine similarity
        cos_pos = F.cosine_similarity(
            combined_features, image_features, dim=-1
        )  # Positive contrast
        cos_orig = F.cosine_similarity(
            text_features, image_features, dim=-1
        )  # Original contrast
        cos_neg = F.cosine_similarity(
            combined_features_neg, image_features, dim=-1
        )  # Negative cvidiaontrast

        loss_improve = torch.clamp(
            cos_orig + self.margin - cos_pos, min=0
        ).mean()  # Let combined features be closer to image features
        loss_neg = torch.clamp(
            cos_pos - cos_neg + self.margin, min=0
        ).mean()  # Let combined features be further from neg

        label_change_loss = label_change_regularizer(
            combined_features,
            text_features,
            label_embedding_proj,
            alpha=self.lambda_labelchange,
        )

        text_preserve_loss = text_preserve_regularizer(
            combined_features, text_features, alpha=self.lambda_preserve
        )

        boundary_loss = boundary_penalty(
            label_embedding, radius=5.0, alpha=self.lambda_boundary
        )

        angular_loss = angular_consistency_loss(
            label_embedding_proj,
            text_features,
            combined_features,
            alpha=self.lambda_angular,
        )

        pull_away_loss = pull_away_diversity_loss(
            label_embedding_proj, alpha=self.lambda_pull_away
        )

        total_loss = (
            self.lambda_pos * loss_improve
            + self.lambda_neg * loss_neg
            + angular_loss
            + pull_away_loss
            + label_change_loss
            + text_preserve_loss
            + boundary_loss
        )

        # Before first k epoch, we not contrast, but rather fully improve over original
        early_loss = loss_improve + boundary_loss  # + 0.5 * loss_neg

        loss_dict = {
            "loss_improve": loss_improve,
            "loss_neg": loss_neg,
            "loss_label_change": label_change_loss,
            "loss_preserve": text_preserve_loss,
            "loss_angular": angular_loss,
            "loss_pull_away": pull_away_loss,
            "loss_boundary": boundary_loss,
            "total_loss": total_loss,
            "early_loss": early_loss,
        }

        if self.return_dict:
            return loss_dict
        else:
            return total_loss


class LabelContrastiveLoss_enhance(nn.Module):
    def __init__(
        self,
        margin: float = 0.2,
        lambda_pos: float = 1.0,
        lambda_neg: float = 1.0,
        lambda_labelchange: float = 0.1,
        lambda_preserve: float = 0.1,
        lambda_angular: float = 0.1,
        lambda_pull_away: float = 0.1,
        lambda_boundary: float = 0.1,
        return_dict: bool = False,
    ) -> None:
        super().__init__()
        print("Using Combined Cosine and Contrastive Loss")
        self.margin = margin
        self.lambda_pos = lambda_pos
        self.lambda_neg = lambda_neg
        self.lambda_labelchange = lambda_labelchange
        self.lambda_preserve = lambda_preserve
        self.lambda_angular = lambda_angular
        self.lambda_pull_away = lambda_pull_away
        self.lambda_boundary = lambda_boundary
        self.return_dict = return_dict

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        combined_features: Tensor,
        combined_features_neg: Tensor,
        label_embedding: Tensor,
        label_embedding_proj: Tensor,
        model,
    ):
        # Compute cosine similarity
        cos_pos = F.cosine_similarity(
            combined_features, image_features, dim=-1
        )  # Positive contrast
        cos_orig = F.cosine_similarity(
            text_features, image_features, dim=-1
        )  # Original contrast
        cos_neg = F.cosine_similarity(
            combined_features_neg, image_features, dim=-1
        )  # Negative cvidiaontrast

        loss_improve = torch.clamp(
            cos_orig + self.margin - cos_pos, min=0
        ).mean()  # Let combined features be closer to image features
        loss_neg = torch.clamp(
            cos_pos - cos_neg + self.margin, min=0
        ).mean()  # Let combined features be further from neg

        # Additional regularizers
        laplacian_loss = manifold_smoothness_loss_sparse(
            label_embedding, text_features, combined_features, model=model, alpha=0.15
        )

        angular_loss = angular_gradient_consistency_loss(
            label_embedding, text_features, combined_features, model=model, alpha=0.1
        )

        rotation_loss = rotation_semantic_orthogonality_loss(
            label_embedding, text_features, combined_features, model=model, alpha=0.1
        )

        radius_loss = radius_monotonicity_loss(
            label_embedding, text_features, model, alpha=0.15
        )

        boundary_loss = boundary_penalty(
            label_embedding, radius=5.0, alpha=self.lambda_boundary
        )

        total_loss = (
            self.lambda_pos * loss_improve
            + self.lambda_neg * loss_neg
            + laplacian_loss
            + angular_loss
            + rotation_loss
            + boundary_loss
            + radius_loss
        )

        loss_dict = {
            "loss_improve": loss_improve,
            "loss_neg": loss_neg,
            "loss_laplacian": laplacian_loss,
            "loss_angular": angular_loss,
            "loss_rotation": rotation_loss,
            "loss_boundary": boundary_loss,
            "total_loss": total_loss,
        }

        if self.return_dict:
            return loss_dict
        else:
            return total_loss


def main(): ...


if __name__ == "__main__":
    main()
