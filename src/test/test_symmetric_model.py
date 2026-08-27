from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.model.cosirmodel import CoSiRModel


class _Encoder(nn.Module):
    def __init__(self, offset: float) -> None:
        super().__init__()
        self.offset = offset

    def forward(self, values):
        return SimpleNamespace(
            pooler_output=values + self.offset,
            last_hidden_state=values.unsqueeze(1).repeat(1, 2, 1) + self.offset,
        )


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(projection_dim=4)
        self.vision_model = _Encoder(offset=1.0)
        self.text_model = _Encoder(offset=2.0)


@pytest.fixture
def model_factory(monkeypatch):
    monkeypatch.setattr(
        "src.model.cosirmodel.get_backbone", lambda *_args, **_kwargs: _Backbone()
    )

    def _make(**kwargs):
        return CoSiRModel(d_model=8, nhead=2, num_layers=1, label_dim=3, dropout=0.0, **kwargs).eval()

    return _make


def _inputs():
    return (
        {"values": torch.arange(8, dtype=torch.float32).reshape(2, 4)},
        {"values": torch.arange(8, 16, dtype=torch.float32).reshape(2, 4)},
        torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
    )


def test_symmetric_shared_forward_returns_paired_embeddings_diagnostics_and_predictions(model_factory):
    model = model_factory(conditioning_mode="symmetric_shared")
    image_calls = []
    predictor_calls = []
    model.combiner.register_forward_hook(lambda *_args: image_calls.append(None))
    model.condition_predictor.register_forward_hook(lambda *_args: predictor_calls.append(None))

    output = model(*_inputs())

    assert set(output) == {
        "img_emb",
        "txt_emb",
        "img_full",
        "txt_full",
        "lbl_emb",
        "img_comb_emb",
        "txt_comb_emb",
        "img_predicted_condition",
        "txt_predicted_condition",
        "combiner_diagnostics",
    }
    assert output["img_comb_emb"].shape == (2, 4)
    assert output["txt_comb_emb"].shape == (2, 4)
    assert output["img_predicted_condition"].shape == (2, 3)
    assert output["txt_predicted_condition"].shape == (2, 3)
    assert set(output["combiner_diagnostics"]) == {"img", "txt"}
    for diagnostics in output["combiner_diagnostics"].values():
        assert set(diagnostics) == {"delta", "gate", "gate_logit"}
        assert diagnostics["delta"].shape == (2, 4)
        assert diagnostics["gate"].shape == (2, 1)
        assert diagnostics["gate_logit"].shape == (2, 1)
    assert len(image_calls) == 2
    assert len(predictor_calls) == 2


def test_default_asymmetric_mode_preserves_legacy_forward_tuple(model_factory):
    torch.manual_seed(7)
    default_model = model_factory(combine_side="img")
    torch.manual_seed(7)
    explicit_model = model_factory(combine_side="img", conditioning_mode="asymmetric")
    inputs = _inputs()

    default_output = default_model(*inputs)
    explicit_output = explicit_model(*inputs)

    assert default_model.conditioning_mode == "asymmetric"
    assert isinstance(default_output, tuple)
    assert len(default_output) == 5
    for default_value, explicit_value in zip(default_output, explicit_output):
        torch.testing.assert_close(default_value, explicit_value, rtol=0, atol=0)


def test_conditioning_mode_is_validated(model_factory):
    with pytest.raises(ValueError, match="conditioning_mode"):
        model_factory(conditioning_mode="separate_combiners")
