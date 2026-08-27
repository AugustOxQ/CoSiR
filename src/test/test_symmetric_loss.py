"""Focused contract tests for the symmetric shared-conditioning loss API."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from src.metrics.loss import LabelContrastiveLoss_enhance


def test_symmetric_loss_averages_paired_terms_and_uses_both_conditioned_embeddings():
    """Catches a one-sided loss branch or pairing either conditioned side with a raw side."""
    criterion = LabelContrastiveLoss_enhance(
        lambda_contrastive=1.0,
        lambda_laplacian=0.0,
        lambda_delta=1.0,
        lambda_gate=1.0,
        lambda_gate_logit=1.0,
        lambda_preserve=1.0,
        lambda_predictor=1.0,
        preserve_tau=0.3,
        return_dict=True,
    )
    img_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    txt_emb = img_emb.clone()
    img_comb_emb = img_emb.clone()
    txt_comb_emb = F.normalize(torch.tensor([[1.0, 1.0], [1.0, -1.0]]), dim=-1)
    lbl_emb = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    outputs = {
        "img_emb": img_emb,
        "txt_emb": txt_emb,
        "img_full": img_emb.unsqueeze(1),
        "txt_full": txt_emb.unsqueeze(1),
        "img_comb_emb": img_comb_emb,
        "txt_comb_emb": txt_comb_emb,
        "lbl_emb": lbl_emb,
        "img_predicted_condition": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "txt_predicted_condition": torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
        "combiner_diagnostics": {
            "img": {
                "delta": torch.tensor([[3.0, 4.0], [0.0, 0.0]]),
                "gate": torch.tensor([[0.1], [0.2]]),
                "gate_logit": torch.tensor([[1.0], [2.0]]),
            },
            "txt": {
                "delta": torch.tensor([[0.0, 0.0], [6.0, 8.0]]),
                "gate": torch.tensor([[0.4], [0.5]]),
                "gate_logit": torch.tensor([[3.0], [4.0]]),
            },
        },
    }

    result = criterion.forward_symmetric(outputs, model=SimpleNamespace(combine_side="txt"))

    logits = img_comb_emb @ txt_comb_emb.T / criterion.temperature
    expected_contrastive = (F.cross_entropy(logits, torch.tensor([0, 1])) + F.cross_entropy(logits.T, torch.tensor([0, 1]))) / 2
    expected_delta = torch.tensor((2.5 + 5.0) / 2)
    expected_gate_entropy = -(
        torch.tensor([0.1, 0.2]).mul(torch.log(torch.tensor([0.1, 0.2]))).add(
            (1 - torch.tensor([0.1, 0.2])) * torch.log(1 - torch.tensor([0.1, 0.2]))
        ).mean()
        + torch.tensor([0.4, 0.5]).mul(torch.log(torch.tensor([0.4, 0.5]))).add(
            (1 - torch.tensor([0.4, 0.5])) * torch.log(1 - torch.tensor([0.4, 0.5]))
        ).mean()
    ) / 2
    expected_gate_logit = torch.tensor((2.5 + 12.5) / 2)
    expected_preserve = (
        F.relu((img_comb_emb - img_emb).norm(dim=-1) - 0.3).pow(2).mean()
        + F.relu((txt_comb_emb - txt_emb).norm(dim=-1) - 0.3).pow(2).mean()
    ) / 2
    expected_predictor = (
        (1 - F.cosine_similarity(outputs["img_predicted_condition"], lbl_emb, dim=-1)).mean()
        + (1 - F.cosine_similarity(outputs["txt_predicted_condition"], lbl_emb, dim=-1)).mean()
    ) / 2

    torch.testing.assert_close(result["loss_improve"], expected_contrastive)
    torch.testing.assert_close(result["loss_delta"], expected_delta)
    torch.testing.assert_close(result["loss_gate_entropy"], expected_gate_entropy)
    torch.testing.assert_close(result["loss_gate_logit"], expected_gate_logit)
    torch.testing.assert_close(result["loss_preserve"], expected_preserve)
    torch.testing.assert_close(result["loss_predictor"], expected_predictor)


def test_symmetric_loss_applies_shared_table_regularizers_once():
    """Catches accidentally summing collapse or boundary penalties once per modality."""
    criterion = LabelContrastiveLoss_enhance(
        lambda_contrastive=0.0,
        lambda_laplacian=0.0,
        lambda_collapse=1.0,
        lambda_boundary=1.0,
        return_dict=True,
    )
    emb = torch.eye(2)
    labels = torch.tensor([[12.0, 0.0], [0.0, 10.0]])
    outputs = {
        "img_emb": emb,
        "txt_emb": emb,
        "img_full": emb.unsqueeze(1),
        "txt_full": emb.unsqueeze(1),
        "img_comb_emb": emb,
        "txt_comb_emb": emb,
        "lbl_emb": labels,
        "combiner_diagnostics": {"img": {}, "txt": {}},
    }

    result = criterion.forward_symmetric(outputs, model=SimpleNamespace(combine_side="txt"))

    # One table: boundary = ((12 - 10)^2 + 0) / 2 = 2, collapse = -0.5.
    torch.testing.assert_close(result["loss_boundary"], torch.tensor(2.0))
    torch.testing.assert_close(result["loss_collapse"], torch.tensor(-0.5))
    torch.testing.assert_close(result["total_loss"], torch.tensor(1.5))
