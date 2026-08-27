"""Focused CPU integration checks for symmetric shared-conditioning training."""

import copy
import importlib
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.metrics.loss import LabelContrastiveLoss_enhance
from src.eval.pipeline import TrainEvaluator


training = importlib.import_module("src.hook.train_cosir")


class _TinyCombiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.condition_projection = nn.Linear(2, 3, bias=False)

    def forward(
        self,
        features,
        _full_features,
        conditions,
        return_delta=False,
        return_scalar=False,
    ):
        delta = self.condition_projection(conditions)
        combined = F.normalize(features + delta, dim=-1)
        if return_delta and return_scalar:
            gate_logit = delta.mean(dim=-1, keepdim=True)
            return combined, delta, torch.sigmoid(gate_logit), gate_logit
        return combined


class _TinySymmetricModel(nn.Module):
    conditioning_mode = "symmetric_shared"

    def __init__(self) -> None:
        super().__init__()
        self.feature_dim = 3
        self.combine_side = "img"
        self.combiner = _TinyCombiner()
        self.condition_predictor = nn.Linear(3, 2, bias=False)
        self.other_proj = nn.Linear(3, 3)
        self.label_encoder = nn.Identity()

    def combine(self, features, full_features, conditions, **kwargs):
        return self.combiner(features, full_features, conditions, **kwargs)

    def combine_symmetric(self, img, img_full, txt, txt_full, conditions):
        def _one(features, full_features):
            combined, delta, gate, gate_logit = self.combine(
                features,
                full_features,
                conditions,
                return_delta=True,
                return_scalar=True,
            )
            return combined, {"delta": delta, "gate": gate, "gate_logit": gate_logit}

        img_combined, img_diag = _one(img, img_full)
        txt_combined, txt_diag = _one(txt, txt_full)
        return {
            "img_comb_emb": img_combined,
            "txt_comb_emb": txt_combined,
            "combiner_diagnostics": {"img": img_diag, "txt": txt_diag},
        }

    def predict_symmetric_conditions(self, img, txt):
        return {
            "img_predicted_condition": self.condition_predictor(img),
            "txt_predicted_condition": self.condition_predictor(txt),
        }


def test_symmetric_cpu_step_keeps_frozen_conditions_fixed_and_updates_trained_conditions():
    """Catches a detached shared table or an update that ignores the freeze policy."""
    compute_loss = getattr(training, "_compute_symmetric_batch_loss", None)
    assert compute_loss is not None, "symmetric training integration is missing"

    torch.manual_seed(19)
    base_model = _TinySymmetricModel()
    initial_conditions = torch.tensor(
        [[0.20, -0.10], [-0.30, 0.40], [0.15, 0.25]], dtype=torch.float32
    )
    img = torch.tensor(
        [[1.0, 0.1, 0.0], [0.0, 1.0, 0.2], [0.1, 0.0, 1.0]], dtype=torch.float32
    )
    txt = torch.tensor(
        [[0.8, 0.2, 0.1], [0.1, 0.9, 0.0], [0.0, 0.2, 0.9]], dtype=torch.float32
    )
    criterion = LabelContrastiveLoss_enhance(
        lambda_contrastive=1.0,
        lambda_laplacian=0.0,
        lambda_predictor=1.0,
        return_dict=True,
    )

    def _step(train_conditions: bool):
        model = copy.deepcopy(base_model)
        conditions = nn.Parameter(initial_conditions.clone(), requires_grad=train_conditions)
        optimizer = torch.optim.SGD([*model.parameters(), conditions], lr=0.2)
        before = conditions.detach().clone()

        outputs, losses = compute_loss(
            model,
            criterion,
            img,
            txt,
            img.unsqueeze(1),
            txt.unsqueeze(1),
            conditions,
        )
        optimizer.zero_grad()
        losses["total_loss"].backward()
        condition_grad = None if conditions.grad is None else conditions.grad.detach().clone()
        optimizer.step()
        return before, conditions.detach().clone(), condition_grad, outputs, losses

    frozen_before, frozen_after, frozen_grad, _, frozen_losses = _step(False)
    trained_before, trained_after, trained_grad, outputs, trained_losses = _step(True)

    torch.testing.assert_close(frozen_after, frozen_before, rtol=0, atol=0)
    assert frozen_grad is None
    assert trained_grad is not None and trained_grad.norm().item() > 0
    assert not torch.equal(trained_after, trained_before)
    assert set(outputs["combiner_diagnostics"]) == {"img", "txt"}
    assert torch.isfinite(frozen_losses["total_loss"])
    assert torch.isfinite(trained_losses["total_loss"])


def test_symmetric_optimizer_contains_each_shared_parameter_once_and_omits_other_projection():
    """Catches duplicated tied weights or optimizing the unused asymmetric projection."""
    model = _TinySymmetricModel()
    conditions = nn.Parameter(torch.zeros(3, 2))
    manager = SimpleNamespace(embeddings=conditions)
    cfg = SimpleNamespace(
        optimizer=SimpleNamespace(lr=1e-3, lr_label=1e-2, weight_decay=0.05),
        scheduler=SimpleNamespace(type="CosineAnnealingLR", T_max=1, eta_min=0.0),
        train=SimpleNamespace(epochs=2),
    )

    optimizer, _ = training._build_optimizer_and_scheduler(cfg, model, manager)

    parameter_ids = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    other_projection_ids = {id(parameter) for parameter in model.other_proj.parameters()}
    assert len(parameter_ids) == len(set(parameter_ids))
    assert id(conditions) in parameter_ids
    assert other_projection_ids.isdisjoint(parameter_ids)
    assert {id(parameter) for parameter in model.combiner.parameters()} <= set(parameter_ids)
    assert {id(parameter) for parameter in model.condition_predictor.parameters()} <= set(parameter_ids)


def test_asymmetric_optimizer_keeps_legacy_other_projection_group():
    model = _TinySymmetricModel()
    model.conditioning_mode = "asymmetric"
    conditions = nn.Parameter(torch.zeros(3, 2))
    cfg = SimpleNamespace(
        optimizer=SimpleNamespace(lr=1e-3, lr_label=1e-2, weight_decay=0.05),
        scheduler=SimpleNamespace(type="CosineAnnealingLR", T_max=1, eta_min=0.0),
        train=SimpleNamespace(epochs=2),
    )

    optimizer, _ = training._build_optimizer_and_scheduler(
        cfg, model, SimpleNamespace(embeddings=conditions)
    )
    parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert len(optimizer.param_groups) == 4
    assert {id(parameter) for parameter in model.other_proj.parameters()} <= parameter_ids


class _FeatureManager:
    def __init__(self, data):
        self.data = data

    def get_features_by_chunk(self, _batch_id):
        return self.data


class _EmbeddingManager:
    def __init__(self, conditions):
        self.conditions = conditions

    def get_embeddings(self, sample_ids):
        return self.conditions[sample_ids]


class _AdditiveSymmetricModel(nn.Module):
    conditioning_mode = "symmetric_shared"
    combine_side = "txt"

    @staticmethod
    def combine(features, _full_features, conditions, epoch=None):
        del epoch
        return features + conditions


class _AdditiveAsymmetricModel(_AdditiveSymmetricModel):
    conditioning_mode = "asymmetric"


def test_train_evaluator_scores_two_conditioned_modalities_in_symmetric_mode():
    """Catches routing symmetric train evaluation through the one-sided legacy score."""
    img = torch.tensor(
        [
            [-1.6052763, 0.2324857, 2.2398701],
            [0.8472938, 1.2006443, -0.4015503],
            [-1.4260197, 0.9039317, 0.8557156],
        ]
    )
    txt = torch.tensor(
        [
            [0.6888809, 0.8849857, 1.7706430],
            [-0.0809429, 0.0512639, -0.8687565],
            [-0.2757461, -1.2717220, 0.8816379],
        ]
    )
    conditions = torch.tensor(
        [
            [-0.6639689, 0.2638975, 0.1610463],
            [1.1758317, -0.3615134, 0.4681450],
            [-0.7289384, -0.9131688, -0.0513311],
        ]
    )
    feature_manager = _FeatureManager(
        {"img_features": img, "txt_features": txt, "sample_ids": torch.arange(3)}
    )

    result = TrainEvaluator().evaluate(
        _AdditiveSymmetricModel(),
        feature_manager,
        _EmbeddingManager(conditions),
        dataloader=[None],
        device="cpu",
        max_batches=1,
    )

    assert result.metrics["val/mean_rank_comb"] == 1.0
    assert TrainEvaluator._mean_rank(img, txt + conditions) == 4 / 3


def test_train_evaluator_keeps_legacy_one_sided_asymmetric_score():
    img = torch.tensor(
        [
            [-1.6052763, 0.2324857, 2.2398701],
            [0.8472938, 1.2006443, -0.4015503],
            [-1.4260197, 0.9039317, 0.8557156],
        ]
    )
    txt = torch.tensor(
        [
            [0.6888809, 0.8849857, 1.7706430],
            [-0.0809429, 0.0512639, -0.8687565],
            [-0.2757461, -1.2717220, 0.8816379],
        ]
    )
    conditions = torch.tensor(
        [
            [-0.6639689, 0.2638975, 0.1610463],
            [1.1758317, -0.3615134, 0.4681450],
            [-0.7289384, -0.9131688, -0.0513311],
        ]
    )
    feature_manager = _FeatureManager(
        {"img_features": img, "txt_features": txt, "sample_ids": torch.arange(3)}
    )

    result = TrainEvaluator().evaluate(
        _AdditiveAsymmetricModel(),
        feature_manager,
        _EmbeddingManager(conditions),
        dataloader=[None],
        device="cpu",
        max_batches=1,
    )

    assert result.metrics["val/mean_rank_comb"] == 4 / 3


class _RecordingExperiment:
    def __init__(self, directory):
        self.directory = directory
        self.saved = []

    def save_artifact(self, **kwargs):
        self.saved.append(kwargs)


class _PersistedEmbeddingManager:
    sample_ids = [0, 1, 2]

    @staticmethod
    def _copy_to(_path):
        return None


def _symmetric_cfg():
    return SimpleNamespace(
        data=SimpleNamespace(dataset_type="synthetic"),
        model=SimpleNamespace(
            conditioning_mode="symmetric_shared",
            combine_side="img",
            embedding_dim=2,
            hidden_dim=4,
            num_layers=1,
            dropout=0.0,
        ),
    )


def test_symmetric_snapshot_and_checkpoint_persist_mode_without_unused_projection(tmp_path):
    """Catches ambiguous artifacts that reload Option A as a one-sided checkpoint."""
    cfg = _symmetric_cfg()
    model = _TinySymmetricModel()
    experiment = _RecordingExperiment(tmp_path)
    test_set = SimpleNamespace(
        image_path=str(tmp_path),
        annotations=[{"image": "0.jpg"}, {"image": "1.jpg"}],
        captions_per_image=1,
    )
    img = torch.eye(2, 3)
    txt = torch.eye(2, 3)
    maps = torch.tensor([[0], [1]])

    training._save_condition_viz_snapshot(
        cfg,
        0,
        experiment,
        model,
        img,
        txt,
        ["zero", "one"],
        maps,
        torch.tensor([0, 1]),
        test_set,
        torch.zeros(3, 2),
        [0, 1, 2],
        torch.zeros(2, 2),
        [],
    )
    snapshot = torch.load(
        tmp_path / "condition_viz" / "epoch_0000.pt", map_location="cpu", weights_only=False
    )

    training._save_final_artifacts(
        model, _PersistedEmbeddingManager(), experiment, cfg
    )
    checkpoint = next(item["data"] for item in experiment.saved if item["name"] == "phase_1_model")

    for artifact in (snapshot, checkpoint):
        assert artifact["conditioning_mode"] == "symmetric_shared"
        assert "other_proj_state_dict" not in artifact
        assert "other_proj_config" not in artifact


def test_asymmetric_checkpoint_payload_keeps_legacy_schema():
    model = _TinySymmetricModel()
    model.conditioning_mode = "asymmetric"

    payload = training._model_persistence_payload(_symmetric_cfg(), model)

    assert "conditioning_mode" not in payload
    assert "other_proj_state_dict" in payload
    assert "other_proj_config" in payload
