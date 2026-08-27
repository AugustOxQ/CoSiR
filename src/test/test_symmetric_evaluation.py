"""Focused CPU checks for symmetric shared-conditioning test evaluation."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.config import EvaluationConfig
from src.eval.metrics import OracleMetrics
from src.eval.pipeline import TestEvaluator as EvaluationUnderTest
from src.model.cosirmodel import CoSiRModel


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(projection_dim=3)
        self.vision_model = nn.Identity()
        self.text_model = nn.Identity()


class _ImageSideHeldAtInitialization(nn.Module):
    """Identity on marked image rows; additive conditioning on text rows."""

    def forward(
        self,
        features,
        _full_features,
        conditions,
        return_delta=False,
        return_scalar=False,
    ):
        if torch.all(features[:, 0] > 5):
            combined = features
        else:
            combined = features + conditions
        return F.normalize(combined, dim=-1)


@pytest.fixture
def cosir_with_image_side_at_initialization(monkeypatch):
    monkeypatch.setattr(
        "src.model.cosirmodel.get_backbone", lambda *_args, **_kwargs: _Backbone()
    )
    model = CoSiRModel(
        d_model=6,
        nhead=1,
        num_layers=1,
        label_dim=3,
        dropout=0.0,
        combine_side="txt",
        conditioning_mode="symmetric_shared",
    ).eval()
    model.combiner = _ImageSideHeldAtInitialization()
    return model


def _without_prefix(metrics, prefix):
    return {key.removeprefix(f"{prefix}/"): value for key, value in metrics.items()}


class _AdditiveEvaluationModel(nn.Module):
    def __init__(
        self,
        predictor_matrix=None,
        conditioning_mode="symmetric_shared",
        combine_side="txt",
    ) -> None:
        super().__init__()
        self.conditioning_mode = conditioning_mode
        self.combine_side = combine_side
        self.condition_predictor = nn.Identity()
        self.predictor_matrix = predictor_matrix

    @staticmethod
    def combine(features, _full_features, conditions):
        return F.normalize(features + conditions, dim=-1)

    @staticmethod
    def project_other(features):
        return features

    def predict_condition(self, features):
        if self.predictor_matrix is None:
            return torch.zeros_like(features)
        return features @ self.predictor_matrix


def _oracle_fixture():
    image_embeddings = torch.tensor(
        [
            [0.39229682, -0.22356401, -0.31950027],
            [-1.20503712, 1.04446352, -0.63322771],
            [0.57310677, 0.54094744, -0.39190584],
        ]
    )
    text_embeddings = torch.tensor(
        [
            [-1.04267883, 1.31861734, 0.74763900],
            [-1.32648063, -1.24129713, -0.10280493],
            [-0.94976312, 0.61810493, -0.23847501],
        ]
    )
    representatives = torch.tensor(
        [
            [0.03676357, -0.73684454, -0.08870159],
            [-2.34788489, 0.63870591, -2.22269726],
        ]
    )
    return image_embeddings, text_embeddings, representatives


def _one_to_one_mappings(size=3):
    return torch.arange(size), torch.arange(size).unsqueeze(1)


def test_coupled_oracle_reduces_to_legacy_when_image_side_is_held_at_initialization(
    cosir_with_image_side_at_initialization,
):
    """Catches a coupled oracle inconsistent with the one-sided limiting case."""
    config = EvaluationConfig(
        device="cpu", k_vals=[1, 2], batch_size=2, cpu_offload=True, print_metrics=False
    )
    oracle = OracleMetrics(config)
    image_embeddings = torch.tensor(
        [[10.0, 1.0, 0.0], [10.0, 0.0, 1.0], [10.0, -1.0, 0.0]]
    )
    text_embeddings = torch.tensor(
        [[0.8, 0.1, 0.0], [0.7, -0.2, 0.4], [0.9, -0.1, -0.3]]
    )
    representatives = torch.tensor(
        [[0.0, 0.0, 0.0], [0.2, -0.4, 0.3]]
    )
    text_to_image = torch.arange(3)
    image_to_text = torch.arange(3).unsqueeze(1)

    asymmetric_model = copy.deepcopy(cosir_with_image_side_at_initialization)
    asymmetric_model.conditioning_mode = "asymmetric"
    legacy = oracle.compute_oracle_recall_average(
        asymmetric_model,
        representatives,
        image_embeddings,
        text_embeddings,
        torch.empty(3, 0, 3),
        text_to_image,
        image_to_text,
        prefix="legacy",
        aggregation="max",
    )
    coupled = oracle.compute_symmetric_coupled_oracle_recall(
        cosir_with_image_side_at_initialization,
        representatives,
        image_embeddings,
        text_embeddings,
        text_to_image,
        image_to_text,
        prefix="coupled",
        aggregation="max",
    )

    assert _without_prefix(coupled, "coupled") == _without_prefix(legacy, "legacy")


@pytest.mark.parametrize(
    ("aggregation", "expected_t2i_r1", "expected_i2t_r1"),
    [("max", 0.0, 33.3), ("mean", 0.0, 0.0)],
)
def test_coupled_oracle_conditions_both_galleries_with_the_same_representative(
    aggregation, expected_t2i_r1, expected_i2t_r1
):
    """Catches one-sided conditioning or independently paired candidates."""
    image_embeddings, text_embeddings, representatives = _oracle_fixture()
    text_to_image, image_to_text = _one_to_one_mappings()
    oracle = OracleMetrics(
        EvaluationConfig(
            device="cpu", k_vals=[1], batch_size=2, cpu_offload=True, print_metrics=False
        )
    )

    metrics = oracle.compute_symmetric_coupled_oracle_recall(
        _AdditiveEvaluationModel(),
        representatives,
        image_embeddings,
        text_embeddings,
        text_to_image,
        image_to_text,
        prefix="coupled",
        aggregation=aggregation,
    )

    assert metrics["coupled/t2i_R1"] == expected_t2i_r1
    assert metrics["coupled/i2t_R1"] == expected_i2t_r1


@pytest.mark.parametrize("aggregation", ["max", "mean"])
def test_independent_two_sided_oracle_uses_the_cross_product_only_as_a_diagnostic(
    aggregation,
):
    """Catches an independent diagnostic silently reduced to coupled candidates."""
    image_embeddings, text_embeddings, representatives = _oracle_fixture()
    text_to_image, image_to_text = _one_to_one_mappings()
    oracle = OracleMetrics(
        EvaluationConfig(
            device="cpu", k_vals=[1], batch_size=2, cpu_offload=True, print_metrics=False
        )
    )

    metrics = oracle.compute_symmetric_independent_oracle_recall(
        _AdditiveEvaluationModel(),
        representatives,
        image_embeddings,
        text_embeddings,
        text_to_image,
        image_to_text,
        prefix="independent_oracle_diagnostic",
        aggregation=aggregation,
    )

    assert metrics["independent_oracle_diagnostic/t2i_R1"] == 33.3
    assert metrics["independent_oracle_diagnostic/i2t_R1"] == 33.3


def test_two_sided_predictor_conditions_each_gallery_from_its_own_prediction():
    """Catches reuse of the legacy predictor path that conditions only one side."""
    image_embeddings = torch.tensor(
        [
            [2.00647402, 1.95346582, 0.15171342],
            [-0.42691326, -0.50587970, -0.77228338],
            [2.99065113, 0.43607843, 1.20578814],
        ]
    )
    text_embeddings = torch.tensor(
        [
            [0.47915298, -0.46109703, -0.00611631],
            [0.29209486, -0.76939911, -0.96333611],
            [-0.32882884, 0.22596647, 0.01268283],
        ]
    )
    predictor_matrix = torch.tensor(
        [
            [0.58013844, 0.69863945, -0.30218017],
            [1.40187597, -1.23711264, -1.12981200],
            [-2.05117488, 0.88607919, -0.02707753],
        ]
    )
    text_to_image, image_to_text = _one_to_one_mappings()
    oracle = OracleMetrics(
        EvaluationConfig(
            device="cpu", k_vals=[1], batch_size=2, cpu_offload=True, print_metrics=False
        )
    )

    metrics = oracle.compute_symmetric_predictor_recall(
        _AdditiveEvaluationModel(predictor_matrix),
        image_embeddings,
        text_embeddings,
        text_to_image,
        image_to_text,
        prefix="two_sided_predictor",
    )

    assert metrics["two_sided_predictor/t2i_R1"] == 0.0
    assert metrics["two_sided_predictor/i2t_R1"] == 0.0


def _evaluator_with_embeddings(model, image_embeddings, text_embeddings):
    text_to_image, image_to_text = _one_to_one_mappings(len(image_embeddings))
    evaluator = EvaluationUnderTest(
        EvaluationConfig(
            device="cpu", k_vals=[1], batch_size=2, cpu_offload=True, print_metrics=False
        )
    )
    evaluator._get_or_extract_embeddings = lambda *_args: (
        image_embeddings,
        text_embeddings,
        torch.empty(len(text_embeddings), 0, text_embeddings.shape[-1]),
        [f"text-{index}" for index in range(len(text_embeddings))],
        text_to_image,
        image_to_text,
    )
    return evaluator


def _metric_groups(metrics):
    return {key.split("/", 1)[0] for key in metrics}


def test_symmetric_pipeline_exposes_three_explicit_tiers_and_raw_baseline():
    image_embeddings, text_embeddings, representatives = _oracle_fixture()
    evaluator = _evaluator_with_embeddings(
        _AdditiveEvaluationModel(torch.eye(3)), image_embeddings, text_embeddings
    )

    result = evaluator.evaluate(
        _AdditiveEvaluationModel(torch.eye(3)),
        processor=None,
        dataloader=object(),
        label_embeddings=representatives,
        use_oracle=False,
    )

    assert _metric_groups(result.metrics) == {
        "epoch",
        "test_coupled_oracle",
        "test_coupled_oracle_diff",
        "test_independent_oracle_diagnostic",
        "test_raw",
        "test_two_sided_predictor",
        "test_two_sided_predictor_diff",
    }
    assert not any(key.startswith("test_oracle/") for key in result.metrics)
    assert not any(key.startswith("test_pre_original/") for key in result.metrics)


@pytest.mark.parametrize(
    ("use_oracle", "expected_groups"),
    [
        (
            False,
            {
                "test_oracle",
                "test_oracle_img",
                "test_oracle_imgtxt",
                "test_raw",
                "test_diff",
                "test_pre_original",
                "test_pre_diff",
                "epoch",
            },
        ),
        (
            True,
            {
                "test_oracle",
                "test_raw",
                "test_diff",
                "test_pre_original",
                "test_pre_diff",
                "epoch",
            },
        ),
    ],
)
def test_asymmetric_pipeline_retains_legacy_metric_groups(use_oracle, expected_groups):
    image_embeddings, text_embeddings, representatives = _oracle_fixture()
    model = _AdditiveEvaluationModel(
        torch.eye(3), conditioning_mode="asymmetric", combine_side="img"
    )
    evaluator = _evaluator_with_embeddings(model, image_embeddings, text_embeddings)

    result = evaluator.evaluate(
        model,
        processor=None,
        dataloader=object(),
        label_embeddings=representatives,
        use_oracle=use_oracle,
    )

    assert _metric_groups(result.metrics) == expected_groups
