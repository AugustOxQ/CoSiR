from email.mime import image
from typing import Any, Dict, Optional, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Callable, List
from numpy import round

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


def imix_loss(
    text_emb: Tensor,
    image_emb: Tensor,
    conditions: Tensor,
    model: nn.Module,
    alpha: float = 1.0,
    lambda_imix: float = 0.1,
) -> Tensor:
    B = text_emb.shape[0]
    device = text_emb.device

    temperature = 0.07

    beta = torch.distributions.Beta(alpha, alpha)
    lam = beta.sample([B]).to(device)
    lam = torch.max(lam, 1.0 - lam)
    perm = torch.randperm(B, device=device)

    lam_exp = lam.view(-1, 1)
    text_mixed = lam_exp * text_emb + (1.0 - lam_exp) * text_emb[perm]

    cond_mixed = lam_exp * conditions + (1.0 - lam_exp) * conditions[perm]
    cond_mixed = F.normalize(cond_mixed, dim=-1)

    combined_mixed = model.combine(text_mixed, None, cond_mixed)

    image_norm = F.normalize(image_emb, dim=-1)
    logits = combined_mixed @ image_norm.T / temperature

    target_A = torch.arange(B, device=device)
    target_B = perm

    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = lam * criterion(logits, target_A) + (1.0 - lam) * criterion(logits, target_B)

    return lambda_imix * loss.mean()


class LabelContrastiveLoss_enhance(nn.Module):
    def __init__(
        self,
        margin: float = 0.2,
        lambda_contrastive: float = 1.0,
        lambda_laplacian: float = 0.1,
        lambda_collapse: float = 0.0,
        lambda_boundary: float = 0.0,
        lambda_mixup: float = 0.0,  # imix loss weight
        mixup_alpha: float = 1.0,
        lambda_delta: float = 0.0,  # delta norm penalty weight
        return_dict: bool = False,
    ) -> None:
        super().__init__()
        print("Using Polar axis regularization loss")
        self.margin = margin
        self.lambda_pos = lambda_contrastive
        self.lambda_laplacian = lambda_laplacian
        self.lambda_collapse = lambda_collapse
        self.lambda_boundary = lambda_boundary
        self.lambda_mixup = lambda_mixup
        self.mixup_alpha = mixup_alpha
        self.lambda_delta = lambda_delta
        self.temperature = 0.07
        self.return_dict = return_dict

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        combined_features: Tensor,
        combined_features_neg: Optional[Tensor],
        label_embedding: Tensor,  # type: ignore
        model: nn.Module,
        delta: Optional[Tensor] = None,
    ):
        # Compute pairwise cosine similarity matrix [N, N] for InfoNCE loss

        batch_size = combined_features.shape[0]

        cos_pos = compute_cosine_similarity(
            combined_features, image_features
        )  # [N, N] pairwise similarities

        loss_improve = (
            (
                F.cross_entropy(
                    cos_pos / self.temperature,
                    torch.arange(batch_size, device=cos_pos.device),
                )
                + F.cross_entropy(
                    cos_pos.T / self.temperature,
                    torch.arange(batch_size, device=cos_pos.device),
                )
            )
            / 2
            if self.lambda_pos > 0
            else 0.0
        )

        # Secondary Loss: This is how the condition space is ensured to be smooth
        laplacian_loss = (
            manifold_smoothness_loss_sparse(
                label_embedding,
                text_features,
                combined_features,
                k=10,
                model=model,
                alpha=1.0,
            )
            if self.lambda_laplacian > 0
            else 0.0
        )

        collapse_loss = (
            -F.normalize(label_embedding, dim=-1).var(dim=0).mean()
            if self.lambda_collapse > 0
            else 0.0
        )

        # Regularizer Loss: this prevents the condition space from being too large
        boundary_loss = (
            boundary_penalty(
                label_embedding,
                radius=10.0,
                alpha=1.0,
            )
            if self.lambda_boundary > 0
            else 0.0
        )

        mixup_loss = (
            imix_loss(
                text_features,
                image_features,
                label_embedding,
                model,
                alpha=self.mixup_alpha,
                lambda_imix=1.0,
            )
            if self.lambda_mixup > 0
            else 0.0
        )

        delta_loss = (
            delta.norm(dim=-1).mean()
            if self.lambda_delta > 0 and delta is not None
            else 0.0
        )

        total_loss = (
            self.lambda_pos * loss_improve
            + self.lambda_laplacian * laplacian_loss
            + self.lambda_collapse * collapse_loss
            + self.lambda_boundary * boundary_loss
            + self.lambda_mixup * mixup_loss
            + self.lambda_delta * delta_loss
        )

        with torch.no_grad():
            diag_sim = cos_pos.diag().mean()
            off_diag_sim = (cos_pos.sum() - cos_pos.diag().sum()) / (
                batch_size * (batch_size - 1)
            )

            diag_sim_gap = diag_sim - off_diag_sim
            off_diag_sim_gap = off_diag_sim - diag_sim
            total_sim_gap = diag_sim - off_diag_sim

        loss_dict = {
            "loss_improve": loss_improve,
            "loss_laplacian": laplacian_loss,
            "loss_boundary": boundary_loss,
            "loss_mixup": mixup_loss,
            "loss_delta": delta_loss,
            "diag_sim_gap": diag_sim_gap,
            "off_diag_sim_gap": off_diag_sim_gap,
            "total_sim_gap": total_sim_gap,
            "total_loss": total_loss,
        }

        if self.return_dict:
            return loss_dict
        else:
            return total_loss


class PrototypeLoss(nn.Module):
    """Pulls conditions toward K-1 learned prototypes on the unit circle.

    Prototype 0 is fixed at [0, 0] (null / no-modulation cluster).
    The remaining K-1 prototypes are learnable unit-circle directions.
    """

    def __init__(
        self,
        K: int = 16,
        temp_start: float = 1.0,
        temp_end: float = 0.1,
        total_epochs: int = 500,
        lambda_attraction: float = 1.0,
        lambda_entropy: float = 0.1,
        lambda_repulsion: float = 0.1,
    ):
        super().__init__()
        # prototype 0: fixed null (no modulation)
        self.null_proto: Tensor
        self.register_buffer("null_proto", torch.zeros(1, 2))

        # K-1 learnable prototypes, evenly spaced on unit circle
        angles = torch.linspace(0, 2 * torch.pi, K)[:-1]  # K-1 angles
        protos = torch.stack([angles.cos(), angles.sin()], dim=-1)
        self.prototypes = nn.Parameter(protos)  # [K-1, 2]

        self.K = K
        self.temp_start = temp_start
        self.temp_end = temp_end
        self.total_epochs = total_epochs
        self.lambda_attraction = lambda_attraction
        self.lambda_entropy = lambda_entropy
        self.lambda_repulsion = lambda_repulsion

    def get_temperature(self, epoch: int) -> float:
        alpha = min(epoch / self.total_epochs, 1.0)
        return self.temp_start + alpha * (self.temp_end - self.temp_start)

    def forward(self, conditions: Tensor, epoch: int) -> Tuple[Tensor, Tensor]:
        # [K, 2]: null prototype stays at [0,0], rest normalized onto unit circle
        learned = F.normalize(self.prototypes, dim=-1)  # [K-1, 2]
        proto = torch.cat([self.null_proto, learned], dim=0)  # [K, 2]

        cond = F.normalize(conditions, dim=-1)  # [B, 2]
        tau = self.get_temperature(epoch)

        dists = torch.cdist(cond, proto)  # [B, K]
        assign = F.gumbel_softmax(-dists, tau=tau, hard=True)  # [B, K]
        usage = assign.mean(0)  # [K]

        # Attraction: pull condition toward its assigned prototype
        assigned_proto = assign @ proto  # [B, 2]
        # conditions assigned to null prototype have assigned_proto=[0,0],
        # cosine_similarity=0 → attraction term = 1 (maximum), so they are
        # not artificially pulled anywhere — the null cluster is a free sink
        null_mask = (assign[:, 0] == 0)  # [B] — not assigned to null
        if null_mask.any():
            cond_active = cond[null_mask]
            proto_active = assigned_proto[null_mask]
            attraction = (1 - F.cosine_similarity(cond_active, proto_active, dim=-1)).mean()
        else:
            attraction = torch.tensor(0.0, device=conditions.device)

        # Entropy: encourage uniform prototype usage (prevent collapse)
        entropy = -(usage * (usage + 1e-8).log()).sum()

        # Repulsion: push learned prototypes apart (ignore null prototype)
        K_learned = learned.shape[0]
        proto_sim = learned @ learned.T  # [K-1, K-1]
        mask = ~torch.eye(K_learned, dtype=torch.bool, device=learned.device)
        repulsion = proto_sim[mask].mean()

        loss = self.lambda_attraction * attraction - self.lambda_entropy * entropy + self.lambda_repulsion * repulsion
        return loss, usage


def main(): ...


if __name__ == "__main__":
    main()
