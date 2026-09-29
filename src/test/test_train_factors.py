"""End-to-end checks for edge-sampled factor discovery."""

import re

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from src.model.factors import SharedFactorEncoder
from src.train.train_factors import FactorTrainingConfig, train_factors


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def paired_features() -> tuple[np.ndarray, np.ndarray, csr_matrix]:
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
    train_factors(
        img, txt, graph, FactorTrainingConfig(num_factors=8, epochs=50, batch_size=12),
        device="cpu",
    )
    losses = [float(value) for value in re.findall(
        r"factor epoch=\d+ loss=([\d.]+)", capsys.readouterr().out
    )]
    assert len(losses) == 50
    assert np.isfinite(losses).all()
    assert np.mean(losses[-5:]) < np.mean(losses[:5]) - 0.02


def test_seed_42_repeats_final_codes_exactly_on_cpu(paired_features):
    img, txt, graph = paired_features
    config = FactorTrainingConfig(num_factors=8, epochs=5, batch_size=12, seed=42)
    _, first_img, first_txt = train_factors(img, txt, graph, config, device="cpu")
    _, second_img, second_txt = train_factors(img, txt, graph, config, device="cpu")
    assert np.array_equal(first_img, second_img)
    assert np.array_equal(first_txt, second_txt)


def test_returns_full_nonnegative_codes_and_eval_model(paired_features):
    img, txt, graph = paired_features
    model, img_codes, txt_codes = train_factors(
        img, txt, graph, FactorTrainingConfig(num_factors=8, epochs=2, batch_size=8),
        device="cpu",
    )
    assert isinstance(model, SharedFactorEncoder)
    assert img_codes.shape == txt_codes.shape == (len(img), 8)
    assert np.isfinite(img_codes).all() and np.isfinite(txt_codes).all()
    assert (img_codes >= 0).all() and (txt_codes >= 0).all()
    assert not model.training


def test_zero_edge_graph_raises_clear_value_error(paired_features):
    img, txt, _ = paired_features
    with pytest.raises(ValueError, match="no.*edges|zero.*edges"):
        train_factors(
            img, txt, csr_matrix((len(img), len(img))),
            FactorTrainingConfig(epochs=1), device="cpu",
        )


def test_training_uses_unique_edge_nodes_with_positive_and_negative_pairs(
    paired_features, monkeypatch
):
    img, txt, _ = paired_features
    # Three disjoint edges; all are sampled when batch_size exceeds edge count.
    left = np.array([0, 2, 4])
    right = left + 1
    graph = csr_matrix(
        (np.ones(6, dtype=np.float32),
         (np.concatenate((left, right)), np.concatenate((right, left)))),
        shape=(len(img), len(img)),
    )
    batch_sizes = []
    original = SharedFactorEncoder.encode_image

    def recording_encode(self, features):
        batch_sizes.append(len(features))
        return original(self, features)

    monkeypatch.setattr(SharedFactorEncoder, "encode_image", recording_encode)
    train_factors(
        img, txt, graph, FactorTrainingConfig(num_factors=8, epochs=1, batch_size=8),
        device="cpu",
    )
    assert batch_sizes[0] == 6
    assert batch_sizes[-1] == len(img)


def test_usage_balance_weight_changes_learned_factor_mass(paired_features):
    img, txt, graph = paired_features
    base = dict(num_factors=8, epochs=80, batch_size=24, seed=42)
    _, img_zero, txt_zero = train_factors(
        img, txt, graph, FactorTrainingConfig(**base, lambda_usage_balance=0.0), device="cpu"
    )
    _, img_strong, txt_strong = train_factors(
        img, txt, graph, FactorTrainingConfig(**base, lambda_usage_balance=5.0), device="cpu"
    )

    def top_two_share(image, text):
        mass = 0.5 * (image.mean(axis=0) + text.mean(axis=0))
        return np.sort(mass)[-2:].sum() / mass.sum()

    assert top_two_share(img_strong, txt_strong) < top_two_share(img_zero, txt_zero) - 0.05


def _ring_fixture() -> tuple[np.ndarray, np.ndarray, csr_matrix]:
    """Tiny CPU fixture shared by the golden and checkpoint tests: N=64, D=16, ring graph."""
    n, d = 64, 16
    rng = np.random.default_rng(2026)
    img = rng.standard_normal((n, d)).astype(np.float32)
    txt = rng.standard_normal((n, d)).astype(np.float32)
    nodes = np.arange(n)
    nxt = (nodes + 1) % n
    graph = csr_matrix(
        (np.ones(2 * n, dtype=np.float32),
         (np.concatenate((nodes, nxt)), np.concatenate((nxt, nodes)))),
        shape=(n, n),
    )
    return img, txt, graph


def test_default_config_codes_match_pre_change_golden_values():
    """Pins the default-config training computation; values were captured from the code before
    the default-off collapse-fix mechanisms were added (docs/reports/2026-10-09_*diagnosis.md)."""
    img, txt, graph = _ring_fixture()
    config = FactorTrainingConfig(num_factors=8, epochs=5, batch_size=16)
    _, img_codes, _ = train_factors(img, txt, graph, config, device="cpu")
    golden = np.array(
        [
            [0.755346, 0.277184, 0.554156, 0.21297, 0.0, 0.0, 0.0, 0.123],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.333562, 0.856291, 1.006007, 0.019826, 0.0, 0.0, 0.160164, 0.0],
            [1.442109, 0.805504, 0.730896, 0.29193, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert np.array_equal(np.round(img_codes[:4], 6), golden)


def test_checkpoint_round_trip_gives_identical_codes(tmp_path):
    from src.train.train_factors import load_factor_checkpoint, save_factor_checkpoint

    # tiny centered TopK + InfoNCE run, then save/load, then re-encode
    img, txt, graph = _ring_fixture()
    config = FactorTrainingConfig(num_factors=8, epochs=3, batch_size=16, agreement="infonce",
                                  activation="topk", topk=3, center_inputs=True, lambda_decorrelation=1.0)
    model, img_codes, _ = train_factors(img, txt, graph, config, device="cpu", group_ids=np.arange(64) // 2)
    save_factor_checkpoint(model, config, tmp_path / "f.pt")
    loaded, loaded_config = load_factor_checkpoint(tmp_path / "f.pt")
    with torch.no_grad():
        again = loaded.encode_image(torch.as_tensor(img, dtype=torch.float32)).numpy()
    assert loaded_config == config and np.array_equal(again, img_codes)


def test_group_ids_length_mismatch_and_unknown_agreement_raise():
    img, txt, graph = _ring_fixture()
    with pytest.raises(ValueError):
        train_factors(img, txt, graph, FactorTrainingConfig(num_factors=8, epochs=1, batch_size=16,
                                                            agreement="bogus"), device="cpu")
    with pytest.raises(ValueError):
        train_factors(img, txt, graph, FactorTrainingConfig(num_factors=8, epochs=1, batch_size=16,
                                                            agreement="infonce"),
                      device="cpu", group_ids=np.arange(63))


def test_each_fix_mechanism_is_wired_into_training():
    img, txt, graph = _ring_fixture()
    base = dict(num_factors=8, epochs=5, batch_size=16)
    _, default_codes, _ = train_factors(img, txt, graph, FactorTrainingConfig(**base), device="cpu")
    for overrides in (dict(agreement="infonce"), dict(lambda_decorrelation=1.0),
                      dict(activation="topk", topk=2), dict(center_inputs=True)):
        _, codes, _ = train_factors(img, txt, graph, FactorTrainingConfig(**base, **overrides),
                                    device="cpu")
        assert not np.array_equal(codes, default_codes), overrides


def test_center_inputs_stores_feature_means_in_model_buffers():
    img, txt, graph = _ring_fixture()
    off, _, _ = train_factors(img, txt, graph, FactorTrainingConfig(num_factors=8, epochs=1,
                                                                    batch_size=16), device="cpu")
    on, _, _ = train_factors(img, txt, graph, FactorTrainingConfig(num_factors=8, epochs=1,
                                                                   batch_size=16, center_inputs=True),
                             device="cpu")
    assert not off.image_mean.any() and not off.text_mean.any()
    assert np.allclose(on.image_mean.numpy(), img.mean(axis=0))
    assert np.allclose(on.text_mean.numpy(), txt.mean(axis=0))


def test_group_ids_length_is_validated_for_any_agreement():
    img, txt, graph = _ring_fixture()
    with pytest.raises(ValueError, match="group_ids"):
        train_factors(img, txt, graph, FactorTrainingConfig(num_factors=8, epochs=1, batch_size=16),
                      device="cpu", group_ids=np.arange(65))
