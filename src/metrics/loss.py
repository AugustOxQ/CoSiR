"""Loss functions for CoSiR contrastive training."""

from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.metrics.regularizer import boundary_penalty, manifold_smoothness_loss_sparse


def compute_cosine_similarity(features1: Tensor, features2: Tensor) -> Tensor:
    """Return pairwise cosine similarity matrix [N, N]."""
    features1_norm = F.normalize(features1, p=2, dim=1)
    features2_norm = F.normalize(features2, p=2, dim=1)
    return torch.mm(features1_norm, features2_norm.t())


def imix_loss(
    text_emb: Tensor,
    image_emb: Tensor,
    conditions: Tensor,
    model: nn.Module,
    alpha: float = 1.0,
    lambda_imix: float = 0.1,
    full_features: Optional[Tensor] = None,
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

    full_mixed = None
    if full_features is not None:
        full_mixed = lam.view(-1, 1, 1) * full_features + (1.0 - lam).view(-1, 1, 1) * full_features[perm]

    combined_mixed = model.combine(text_mixed, full_mixed, cond_mixed)

    image_norm = F.normalize(image_emb, dim=-1)
    logits = combined_mixed @ image_norm.T / temperature

    target_A = torch.arange(B, device=device)
    target_B = perm

    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = lam * criterion(logits, target_A) + (1.0 - lam) * criterion(logits, target_B)

    return lambda_imix * loss.mean()


def predictor_consistency_loss(
    pred_cond: Tensor,
    label_embeddings: Tensor,
    stopgrad: bool = True,
) -> Tensor:
    """Cosine-distance consistency loss between the condition predictor's output and the
    per-sample condition table.

    stopgrad=True (default): only the predictor receives gradient from this term -- today's
    one-way distillation (predictor learns to reproduce the table).
    stopgrad=False: gradient also flows into label_embeddings, pulling the table toward what
    the predictor -- a bounded-capacity function of the frozen input feature -- can represent
    (Experiment 11.3, docs/superpowers/specs/2026-08-04-buddy-publication-plan-design.md).
    """
    target = label_embeddings.detach() if stopgrad else label_embeddings
    return (1 - F.cosine_similarity(pred_cond, target, dim=-1)).mean()


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
        lambda_gate: float = 0.0,   # gate entropy maximization weight
        lambda_gate_logit: float = 0.0,  # L2 penalty on raw gate logit (prevents sigmoid saturation)
        lambda_preserve: float = 0.0,  # input preservation weight
        preserve_tau: float = 0.3,  # max allowed deviation from input (in L2 of unit vectors)
        return_dict: bool = False,
        lambda_predictor: float = 0.0,
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
        self.lambda_gate = lambda_gate
        self.lambda_gate_logit = lambda_gate_logit
        self.lambda_preserve = lambda_preserve
        self.preserve_tau = preserve_tau
        self.temperature = 0.07
        self.return_dict = return_dict
        self.lambda_predictor = lambda_predictor

    def _symmetric_laplacian_loss(
        self,
        conditions: Tensor,
        features: Tensor,
        full_features: Tensor,
        combined_features: Tensor,
        model: nn.Module,
    ) -> Tensor:
        """Laplacian term for one conditioned modality, preserving its full sequence."""
        batch_size = len(conditions)
        dist_matrix = torch.cdist(conditions, conditions)
        dist_matrix = dist_matrix.masked_fill(
            torch.eye(batch_size, device=dist_matrix.device, dtype=torch.bool), float("inf")
        )
        _, neighbor_indices = torch.topk(dist_matrix, min(10, batch_size - 1), largest=False, dim=1)
        random_neighbor_idx = torch.randint(0, neighbor_indices.shape[1], (batch_size,), device=conditions.device)
        selected_neighbors = neighbor_indices[torch.arange(batch_size, device=conditions.device), random_neighbor_idx]
        neighbor_combined = model.combine(features, full_features, conditions[selected_neighbors])
        delta_current = combined_features - F.normalize(features, p=2, dim=1)
        delta_neighbor = neighbor_combined - F.normalize(features, p=2, dim=1)
        smoothness = F.cosine_similarity(delta_current, delta_neighbor, dim=-1)
        distances = dist_matrix.gather(1, selected_neighbors.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=10.0)
        return ((1 - smoothness) * torch.exp(-distances + 1e-8)).mean()

    def forward_symmetric(self, outputs: Mapping[str, object], model: nn.Module):
        """Compute Option-A loss from the named ``symmetric_shared`` model output.

        The contrastive matrix couples the two conditioned modalities.  Terms tied to
        a modality are averaged; terms tied to the one shared condition table are
        evaluated exactly once.
        """
        img_features = outputs["img_emb"]
        txt_features = outputs["txt_emb"]
        img_full = outputs["img_full"]
        txt_full = outputs["txt_full"]
        img_combined = outputs["img_comb_emb"]
        txt_combined = outputs["txt_comb_emb"]
        label_embedding = outputs["lbl_emb"]
        diagnostics = outputs["combiner_diagnostics"]
        assert isinstance(img_features, Tensor) and isinstance(txt_features, Tensor)
        assert isinstance(img_full, Tensor) and isinstance(txt_full, Tensor)
        assert isinstance(img_combined, Tensor) and isinstance(txt_combined, Tensor)
        assert isinstance(label_embedding, Tensor) and isinstance(diagnostics, Mapping)
        img_diagnostics = diagnostics["img"]
        txt_diagnostics = diagnostics["txt"]
        assert isinstance(img_diagnostics, Mapping) and isinstance(txt_diagnostics, Mapping)

        batch_size = img_combined.shape[0]
        cos_pos = compute_cosine_similarity(img_combined, txt_combined)
        targets = torch.arange(batch_size, device=cos_pos.device)
        loss_improve = (
            (F.cross_entropy(cos_pos / self.temperature, targets) + F.cross_entropy(cos_pos.T / self.temperature, targets)) / 2
            if self.lambda_pos > 0 else 0.0
        )

        laplacian_loss = (
            (self._symmetric_laplacian_loss(label_embedding, img_features, img_full, img_combined, model)
             + self._symmetric_laplacian_loss(label_embedding, txt_features, txt_full, txt_combined, model)) / 2
            if self.lambda_laplacian > 0 else 0.0
        )
        collapse_loss = -F.normalize(label_embedding, dim=-1).var(dim=0).mean() if self.lambda_collapse > 0 else 0.0
        boundary_loss = boundary_penalty(label_embedding, radius=10.0, alpha=1.0) if self.lambda_boundary > 0 else 0.0
        mixup_loss = (
            (imix_loss(txt_features, img_features, label_embedding, model, alpha=self.mixup_alpha, lambda_imix=1.0, full_features=txt_full)
             + imix_loss(img_features, txt_features, label_embedding, model, alpha=self.mixup_alpha, lambda_imix=1.0, full_features=img_full)) / 2
            if self.lambda_mixup > 0 else 0.0
        )

        def _mean_pair(name: str, fn):
            img_value, txt_value = img_diagnostics.get(name), txt_diagnostics.get(name)
            return (fn(img_value) + fn(txt_value)) / 2 if img_value is not None and txt_value is not None else 0.0

        delta_loss = _mean_pair("delta", lambda value: value.norm(dim=-1).mean()) if self.lambda_delta > 0 else 0.0
        gate_entropy_loss = _mean_pair(
            "gate",
            lambda value: -(value * torch.log(value + 1e-8) + (1 - value) * torch.log(1 - value + 1e-8)).mean(),
        ) if self.lambda_gate > 0 else 0.0
        gate_logit_loss = _mean_pair("gate_logit", lambda value: value.pow(2).mean()) if self.lambda_gate_logit > 0 else 0.0
        preserve_loss = (
            (F.relu((img_combined - F.normalize(img_features, dim=-1)).norm(dim=-1) - self.preserve_tau).pow(2).mean()
             + F.relu((txt_combined - F.normalize(txt_features, dim=-1)).norm(dim=-1) - self.preserve_tau).pow(2).mean()) / 2
            if self.lambda_preserve > 0 else 0.0
        )
        img_predicted, txt_predicted = outputs.get("img_predicted_condition"), outputs.get("txt_predicted_condition")
        predictor_loss = (
            (predictor_consistency_loss(img_predicted, label_embedding) + predictor_consistency_loss(txt_predicted, label_embedding)) / 2
            if self.lambda_predictor > 0 and isinstance(img_predicted, Tensor) and isinstance(txt_predicted, Tensor) else 0.0
        )

        total_loss = (
            self.lambda_pos * loss_improve + self.lambda_laplacian * laplacian_loss
            + self.lambda_collapse * collapse_loss + self.lambda_boundary * boundary_loss
            + self.lambda_mixup * mixup_loss + self.lambda_delta * delta_loss
            - self.lambda_gate * gate_entropy_loss + self.lambda_preserve * preserve_loss
            + self.lambda_gate_logit * gate_logit_loss + self.lambda_predictor * predictor_loss
        )
        with torch.no_grad():
            diag_sim = cos_pos.diag().mean()
            off_diag_sim = (cos_pos.sum() - cos_pos.diag().sum()) / (batch_size * (batch_size - 1))
        return {
            "loss_improve": loss_improve, "loss_laplacian": laplacian_loss,
            "loss_collapse": collapse_loss, "loss_boundary": boundary_loss, "loss_mixup": mixup_loss,
            "loss_delta": delta_loss, "loss_gate_entropy": gate_entropy_loss,
            "loss_gate_logit": gate_logit_loss, "loss_preserve": preserve_loss,
            "loss_predictor": predictor_loss, "diag_sim_gap": diag_sim - off_diag_sim,
            "off_diag_sim_gap": off_diag_sim - diag_sim, "total_sim_gap": diag_sim - off_diag_sim,
            "total_loss": total_loss,
        }

    def forward(
        self,
        image_features: Tensor,
        text_features: Tensor,
        combined_features: Tensor,
        combined_features_neg: Optional[Tensor],
        label_embedding: Tensor,  # type: ignore
        model: nn.Module,
        delta: Optional[Tensor] = None,
        scalar: Optional[Tensor] = None,
        gate_logit: Optional[Tensor] = None,
    ):
        batch_size = combined_features.shape[0]

        cos_pos = compute_cosine_similarity(combined_features, image_features)  # [N, N]

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

        # Secondary Loss: ensures the condition space is smooth
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

        # Regularizer: prevents the condition space from expanding without bound
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

        # Maximize gate entropy: penalize scalar near 0 or 1 → encourages adaptive gating
        gate_entropy_loss = (
            -(scalar * torch.log(scalar + 1e-8) + (1 - scalar) * torch.log(1 - scalar + 1e-8)).mean()
            if self.lambda_gate > 0 and scalar is not None
            else 0.0
        )

        # Preserve: penalise combined output that deviates far from the combine-side input.
        # Reference is text when combine_side='txt', image when combine_side='img'.
        # Both tensors must be unit-normalised for tau to be meaningful (tau=0.3 ≈ ~17°).
        preserve_loss = (
            F.relu(
                (
                    combined_features
                    - F.normalize(
                        text_features if getattr(model, "combine_side", "txt") == "txt" else image_features,
                        dim=-1,
                    )
                ).norm(dim=-1)
                - self.preserve_tau
            ).pow(2).mean()
            if self.lambda_preserve > 0
            else 0.0
        )

        # L2 on the raw gate logit (pre-sigmoid).  Gradient = 2*logit, which grows
        # with logit magnitude — provides a counter-force that doesn't vanish through
        # the sigmoid, preventing the gate logit from saturating at ±∞.
        gate_logit_loss = (
            gate_logit.pow(2).mean()
            if self.lambda_gate_logit > 0 and gate_logit is not None
            else 0.0
        )

        total_loss = (
            self.lambda_pos * loss_improve
            + self.lambda_laplacian * laplacian_loss
            + self.lambda_collapse * collapse_loss
            + self.lambda_boundary * boundary_loss
            + self.lambda_mixup * mixup_loss
            + self.lambda_delta * delta_loss
            - self.lambda_gate * gate_entropy_loss  # subtract to maximise entropy
            + self.lambda_preserve * preserve_loss
            + self.lambda_gate_logit * gate_logit_loss
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
            "loss_gate_entropy": gate_entropy_loss,
            "loss_gate_logit": gate_logit_loss,
            "loss_preserve": preserve_loss,
            "diag_sim_gap": diag_sim_gap,
            "off_diag_sim_gap": off_diag_sim_gap,
            "total_sim_gap": total_sim_gap,
            "total_loss": total_loss,
        }

        if self.return_dict:
            return loss_dict
        else:
            return total_loss
