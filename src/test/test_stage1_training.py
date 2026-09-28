"""End-to-end checks for Stage 1 teacher-graph training."""

import re

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from src.model.student import AttentionFusionStudent
from src.train.stage1 import Stage1Config, train_stage1


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def paired_features() -> tuple[np.ndarray, np.ndarray, csr_matrix]:
    """Twenty-four distinct latent pairs with matching teacher edges."""
    rng = np.random.default_rng(123)
    centers = rng.standard_normal((24, 12)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    latent = np.repeat(centers, 2, axis=0)
    img = latent + 0.18 * rng.standard_normal(latent.shape).astype(np.float32)
    txt = latent + 0.18 * rng.standard_normal(latent.shape).astype(np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    left = np.arange(0, 48, 2)
    right = left + 1
    graph = csr_matrix(
        (np.ones(48, dtype=np.float32),
         (np.concatenate((left, right)), np.concatenate((right, left)))),
        shape=(48, 48),
    )
    return img, txt, graph


def test_training_loss_decreases_over_short_run(paired_features, capsys):
    img, txt, graph = paired_features

    train_stage1(img, txt, graph, Stage1Config(epochs=35, batch_size=12), device="cpu")

    losses = [float(value) for value in re.findall(r"epoch=\d+ loss=([\d.]+)", capsys.readouterr().out)]
    assert len(losses) == 35
    assert np.mean(losses[-5:]) < np.mean(losses[:5]) - 0.02


def test_seed_42_repeats_final_embeddings_exactly_on_cpu(paired_features):
    img, txt, graph = paired_features
    config = Stage1Config(epochs=5, batch_size=12, seed=42)

    _, first = train_stage1(img, txt, graph, config, device="cpu")
    _, second = train_stage1(img, txt, graph, config, device="cpu")

    assert np.array_equal(first, second)


def test_zero_edge_graph_raises_clear_value_error(paired_features):
    img, txt, _graph = paired_features
    empty_graph = csr_matrix((len(img), len(img)))

    with pytest.raises(ValueError, match="no.*edges|zero.*edges"):
        train_stage1(img, txt, empty_graph, Stage1Config(epochs=1), device="cpu")


def test_returns_full_dataset_unit_norm_embeddings(paired_features):
    img, txt, graph = paired_features

    model, embeddings = train_stage1(
        img, txt, graph, Stage1Config(epochs=2, batch_size=8), device="cpu"
    )

    assert isinstance(model, AttentionFusionStudent)
    assert embeddings.shape == (len(img), model.norm.normalized_shape[0])
    np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6, rtol=0)
    assert not model.training


def test_training_embeds_only_unique_sampled_nodes(paired_features, monkeypatch):
    img, txt, _graph = paired_features
    graph = csr_matrix(
        (np.ones(2, dtype=np.float32), ([0, 1], [1, 0])), shape=(len(img), len(img))
    )
    batch_sizes = []
    original_forward = AttentionFusionStudent.forward

    def recording_forward(self, img_batch, txt_batch):
        batch_sizes.append(len(img_batch))
        return original_forward(self, img_batch, txt_batch)

    monkeypatch.setattr(AttentionFusionStudent, "forward", recording_forward)

    train_stage1(img, txt, graph, Stage1Config(epochs=1, batch_size=8), device="cpu")

    assert batch_sizes == [2, len(img)]
