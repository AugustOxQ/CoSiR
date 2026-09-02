"""Identity-initialization and wiring checks for Phase 3 fusion combiners.

Usage:
    conda activate CoSiR
    python src/test/20260902_combiner_architecture/test_identity_init.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import torch
import torch.nn.functional as F

from src.model.combiner import (
    CombinerFiLMResidual,
    CombinerLowRankAdapter,
    CombinerResidualControl,
)
from src.model.cosirmodel import CoSiRModel


COMBINERS = (
    CombinerResidualControl,
    CombinerLowRankAdapter,
    CombinerFiLMResidual,
)


def _assert_identity_init_and_terminal_gradients(combiner_cls, feature_dim: int, label_dim: int) -> None:
    torch.manual_seed(feature_dim + label_dim)
    general_features = torch.randn(4, feature_dim)
    label_features = torch.randn(4, label_dim)
    combiner = combiner_cls(clip_feature_dim=feature_dim, label_dim=label_dim, dropout=0.0)

    output = combiner(general_features, None, label_features)
    torch.testing.assert_close(output, F.normalize(general_features, dim=-1), atol=1e-5, rtol=0)

    output.sum().backward()
    # Exact identity initialization necessarily masks gradients to the MLP layers
    # before a zero-initialized terminal projection. Verify those terminal
    # parameters instead: their nonzero gradients guarantee the branch can leave
    # identity on the first optimizer step.
    for name, parameter in combiner.named_parameters():
        if name in combiner.identity_init_parameters:
            assert parameter.grad is not None, f"{combiner_cls.__name__}.{name} has no gradient"
            assert torch.count_nonzero(parameter.grad) > 0, f"{combiner_cls.__name__}.{name} has an all-zero gradient"


def test_identity_initialization() -> None:
    for combiner_cls in COMBINERS:
        for feature_dim in (512, 768):
            for label_dim in (2, 16):
                _assert_identity_init_and_terminal_gradients(combiner_cls, feature_dim, label_dim)


def test_cosir_model_wiring() -> None:
    for combiner_type in ("residual_control", "lowrank", "film"):
        model = CoSiRModel(
            backbone_model="openai/clip-vit-base-patch32",
            backbone_trainable=False,
            label_dim=2,
            combiner_type=combiner_type,
        )
        for combine_side in ("img", "txt"):
            model.combine_side = combine_side
            output = model.combine(torch.randn(2, model.feature_dim), None, torch.randn(2, 2))
            assert output.shape == (2, model.feature_dim)


if __name__ == "__main__":
    test_identity_initialization()
    test_cosir_model_wiring()
    print("Phase 3 combiner identity-init checks passed.")
