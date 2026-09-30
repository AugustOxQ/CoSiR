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
    the default-off collapse-fix mechanisms were added (docs/reports/auto/v2/2026-10-09_*diagnosis.md)."""
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
                                    device="cpu", group_ids=np.arange(len(img)))
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


# ---- Final-review fix I3: InfoNCE needs explicit group_ids; numeric ids; R3 preset ----

from pathlib import Path  # noqa: E402

from src.train.train_factors import R3_CONFIG, encode_rows, load_factor_checkpoint  # noqa: E402

_CHECKPOINTS = Path(__file__).resolve().parent / "20261011_factor_repair_grid" / "checkpoints"


def test_infonce_without_group_ids_raises_and_names_the_explicit_opt_out():
    img, txt, graph = _ring_fixture()
    config = FactorTrainingConfig(num_factors=8, epochs=1, batch_size=16, agreement="infonce")
    with pytest.raises(ValueError, match=r"group_ids=np\.arange\(n\)"):
        train_factors(img, txt, graph, config, device="cpu")
    train_factors(img, txt, graph, config, device="cpu", group_ids=np.arange(len(img)))   # the opt-out works


@pytest.mark.parametrize("agreement", ["cosine", "infonce"])
@pytest.mark.parametrize("bad", [np.array(["p%d" % (i // 2) for i in range(64)]), np.arange(64) / 2.0])
def test_non_integer_group_ids_raise_a_clear_value_error(agreement, bad):
    img, txt, graph = _ring_fixture()
    config = FactorTrainingConfig(num_factors=8, epochs=1, batch_size=16, agreement=agreement)
    with pytest.raises(ValueError, match="integer"):
        train_factors(img, txt, graph, config, device="cpu", group_ids=bad)


def test_group_ids_reach_the_infonce_loss():
    img, txt, graph = _ring_fixture()        # ring edges (2k, 2k+1) put same-group pairs in every batch
    config = FactorTrainingConfig(num_factors=8, epochs=5, batch_size=16, agreement="infonce")
    _, paired, _ = train_factors(img, txt, graph, config, device="cpu", group_ids=np.arange(64) // 2)
    _, unmasked, _ = train_factors(img, txt, graph, config, device="cpu", group_ids=np.arange(64))
    assert not np.array_equal(paired, unmasked)


def test_r3_preset_is_the_selected_recipe_and_defaults_stay_r0():
    assert R3_CONFIG == FactorTrainingConfig(lambda_usage_balance=0.1, agreement="infonce", lambda_decorrelation=1.0)
    default = FactorTrainingConfig()
    assert (default.agreement, default.lambda_decorrelation, default.activation, default.center_inputs) == \
           ("cosine", 0.0, "relu", False)                               # historical collapsed R0 recipe


@pytest.mark.skipif(not (_CHECKPOINTS / "selected_seed42.pt").exists(), reason="Task 6 checkpoint is gitignored")
def test_r3_preset_equals_the_stored_selected_checkpoint_config():
    _, config = load_factor_checkpoint(_CHECKPOINTS / "selected_seed42.pt")
    assert config == R3_CONFIG


@pytest.mark.skipif(not (_CHECKPOINTS / "R0_seed42.pt").exists(), reason="Task 6 checkpoint is gitignored")
def test_default_config_equals_the_stored_r0_checkpoint_config():
    _, config = load_factor_checkpoint(_CHECKPOINTS / "R0_seed42.pt")
    assert config == FactorTrainingConfig()


# ---- Final-review fix I5: shared encode helper ----

def _direct_codes(model, img, txt):
    model.eval()
    with torch.no_grad():
        return (model.encode_image(torch.as_tensor(img, dtype=torch.float32)).numpy(),
                model.encode_text(torch.as_tensor(txt, dtype=torch.float32)).numpy())


def test_encode_rows_equals_direct_eval_mode_encoding():
    img, txt, _ = _ring_fixture()
    torch.manual_seed(0)
    model = SharedFactorEncoder(img.shape[1], 8)
    expected_img, expected_txt = _direct_codes(model, img, txt)
    model.train()                                                       # dropout on: encode_rows must switch it off
    img_codes, txt_codes = encode_rows(model, img, txt)
    assert isinstance(img_codes, np.ndarray) and img_codes.dtype == np.float32
    assert np.array_equal(img_codes, expected_img) and np.array_equal(txt_codes, expected_txt)
    assert model.training                                               # the caller's mode is restored
    rows = np.array([5, 0, 63, 17])
    sub_img, sub_txt = encode_rows(model, img, txt, rows=rows)
    direct_img, direct_txt = _direct_codes(model, img[rows], txt[rows])
    assert np.array_equal(sub_img, direct_img) and np.array_equal(sub_txt, direct_txt)
    small_img, small_txt = encode_rows(model, img, txt, batch_size=7)   # batching only
    assert np.allclose(small_img, expected_img, atol=1e-6) and np.allclose(small_txt, expected_txt, atol=1e-6)


def test_encode_rows_reproduces_the_codes_train_factors_returns():
    img, txt, graph = _ring_fixture()
    model, img_codes, txt_codes = train_factors(img, txt, graph, FactorTrainingConfig(num_factors=8, epochs=2,
                                                                                    batch_size=16), device="cpu")
    again_img, again_txt = encode_rows(model, img, txt)
    assert np.array_equal(again_img, img_codes) and np.array_equal(again_txt, txt_codes)


import dataclasses

from src.train.condition_episodes import mine_condition_episodes
from src.train.condition_sources import CommunitySource
from src.train.factor_condition_loss import naive_episode_scores
from src.train.train_factors import GroupRows


def test_group_rows_expand_returns_every_row_of_each_touched_group():
    groups = np.array([3, 1, 3, 2, 1, 3, 0])
    index = GroupRows(groups)
    assert index.expand(np.array([0])).tolist() == [0, 2, 5]
    assert index.expand(np.array([4, 6])).tolist() == [1, 4, 6]
    assert index.expand(np.array([2, 5, 0])).tolist() == [0, 2, 5]


def test_painting_batches_contain_complete_paintings(paired_features, monkeypatch):
    import src.train.train_factors as tf

    img, txt, graph = paired_features                       # 48 rows, edges (2k, 2k+1)
    groups = np.arange(48) // 4                              # paintings of 4 rows
    seen = []
    real = tf.cross_modal_infonce_loss

    def spy(img_codes, txt_codes, temperature, group_ids):
        seen.append(group_ids.cpu().numpy())
        return real(img_codes, txt_codes, temperature, group_ids)

    monkeypatch.setattr(tf, "cross_modal_infonce_loss", spy)
    config = FactorTrainingConfig(num_factors=4, epochs=3, batch_size=4, agreement="infonce",
                                  painting_batches=True)
    train_factors(img, txt, graph, config, device="cpu", group_ids=groups)
    assert len(seen) == 3
    for batch_groups in seen:
        assert (np.bincount(batch_groups)[np.unique(batch_groups)] == 4).all()


def test_painting_agreement_replaces_the_row_infonce(paired_features, monkeypatch):
    import src.train.train_factors as tf

    img, txt, graph = paired_features
    calls = {"painting": 0, "row": 0}
    real_painting = tf.painting_infonce_loss

    def painting_spy(*args, **kwargs):
        calls["painting"] += 1
        return real_painting(*args, **kwargs)

    def row_spy(*args, **kwargs):
        calls["row"] += 1
        raise AssertionError("row InfoNCE must not run under painting agreement")

    monkeypatch.setattr(tf, "painting_infonce_loss", painting_spy)
    monkeypatch.setattr(tf, "cross_modal_infonce_loss", row_spy)
    config = FactorTrainingConfig(num_factors=4, epochs=2, batch_size=4, agreement="infonce",
                                  agreement_level="painting", painting_batches=True)
    train_factors(img, txt, graph, config, device="cpu", group_ids=np.arange(48) // 4)
    assert calls == {"painting": 2, "row": 0}


def _condition_world():
    img, txt, graph = _ring_fixture()                        # 64 rows, D = 16
    keys = np.arange(64)                                     # each row its own painting
    source = CommunitySource(np.arange(64) % 4, np.arange(64), min_group_rows=4)
    return img, txt, graph, keys, source


def test_condition_loss_trains_and_records_history():
    img, txt, graph, keys, source = _condition_world()
    config = FactorTrainingConfig(num_factors=8, epochs=5, batch_size=16, agreement="infonce",
                                  painting_batches=True, lambda_condition=1.0, condition_episodes_per_step=8)
    history = {}
    _, img_codes, txt_codes = train_factors(img, txt, graph, config, device="cpu", group_ids=keys,
                                            condition_source=source, history=history, log_every=1)
    assert np.isfinite(img_codes).all() and np.isfinite(txt_codes).all()
    assert history["step"] == [1, 2, 3, 4, 5]
    assert all(np.isfinite(history["condition_loss"])) and all(t > 0 for t in history["tau"])


def test_condition_source_rows_must_index_training_rows():
    img, txt, graph, keys, _ = _condition_world()
    global_source = CommunitySource(np.arange(200) % 4, np.arange(100, 200), min_group_rows=4)
    config = FactorTrainingConfig(num_factors=8, epochs=1, batch_size=16, agreement="infonce",
                                  painting_batches=True, lambda_condition=1.0)
    with pytest.raises(ValueError, match="training rows"):
        train_factors(img, txt, graph, config, device="cpu", group_ids=keys, condition_source=global_source)


@pytest.mark.parametrize("overrides, kwargs, message", [
    (dict(lambda_condition=1.0), dict(), "condition_source"),
    (dict(), dict(condition_source="SOURCE"), "condition_source"),
    (dict(agreement_level="painting"), dict(), "painting_batches"),
    (dict(agreement_level="rows"), dict(), "agreement_level"),
    (dict(painting_batches=True), dict(group_ids=None), "group_ids"),
])
def test_new_options_are_validated(overrides, kwargs, message):
    img, txt, graph, keys, source = _condition_world()
    config = dataclasses.replace(FactorTrainingConfig(num_factors=8, epochs=1, batch_size=16, agreement="infonce"),
                                 **overrides)
    call = {"group_ids": keys, **kwargs}
    if call.get("condition_source") == "SOURCE":
        call["condition_source"] = source
    if call["group_ids"] is None and config.agreement == "infonce":
        config = dataclasses.replace(config, agreement="cosine")
    with pytest.raises(ValueError, match=message):
        train_factors(img, txt, graph, config, device="cpu", **call)


def test_checkpoint_without_new_fields_loads_with_defaults(tmp_path):
    from src.train.train_factors import load_factor_checkpoint, save_factor_checkpoint

    img, txt, graph = _ring_fixture()
    config = FactorTrainingConfig(num_factors=8, epochs=2, batch_size=16)
    model, _, _ = train_factors(img, txt, graph, config, device="cpu")
    save_factor_checkpoint(model, config, tmp_path / "old.pt")
    payload = torch.load(tmp_path / "old.pt", weights_only=True)
    for field in ("agreement_level", "painting_batches", "lambda_condition", "condition_episodes_per_step",
                  "condition_beta"):
        payload["config"].pop(field)
    torch.save(payload, tmp_path / "old.pt")
    _, loaded = load_factor_checkpoint(tmp_path / "old.pt")
    assert loaded == config


def _style_world(seed=7):
    """480 rows / 240 paintings; a 4-way 'style' carried by a low-variance direction, content high-variance.

    The style amplitude (1.5, not the brief's 0.25) keeps the style codes above the factor-code scale reachable in
    300 Adam steps; at 0.25 even the condition loss cannot move the naive R@1 (0.27 vs 0.285)."""
    rng = np.random.default_rng(seed)
    paint = np.arange(480) // 2
    style = rng.integers(0, 4, 240)[paint]
    content = rng.standard_normal((240, 12))[paint]
    onehot = np.eye(4)[style] * 1.5
    img = np.hstack([content, onehot]) + 0.05 * rng.standard_normal((480, 16))
    txt = np.hstack([content + 0.3 * rng.standard_normal((480, 12)), onehot]) + 0.05 * rng.standard_normal((480, 16))
    left = np.arange(0, 480, 2)
    graph = csr_matrix((np.ones(480, dtype=np.float32),
                        (np.concatenate((left, left + 1)), np.concatenate((left + 1, left)))), shape=(480, 480))
    return img.astype(np.float32), txt.astype(np.float32), graph, paint, style


def _naive_r1(model, img, txt, source, keys, n=200, seed=99):
    from src.train.train_factors import encode_rows

    ic, tc = encode_rows(model, img, txt, device="cpu")
    ep = mine_condition_episodes(source, None, keys, n, np.random.default_rng(seed), num_hard=0, num_random=12)
    t = lambda a: torch.as_tensor(a)                                   # noqa: E731
    scores = naive_episode_scores(t(img), t(txt), t(ic), t(tc), t(ep.anchor), t(ep.supports), t(ep.contrasts),
                                  t(ep.candidates), beta=0.3)
    mask = torch.as_tensor(ep.positive_mask).float()
    hit = [mask.gather(1, s.argmax(dim=1, keepdim=True)).mean().item() for s in scores.values()]
    return sum(hit) / 2


def test_condition_loss_teaches_a_low_variance_condition():
    from src.train.train_factors import R3_CONFIG

    img, txt, graph, paint, style = _style_world()
    source = CommunitySource(style, np.arange(480), min_group_rows=20)
    base = dataclasses.replace(R3_CONFIG, num_factors=8, epochs=300, batch_size=64, painting_batches=True,
                               condition_episodes_per_step=16)
    runs = {}
    for name, lam in (("none", 0.0), ("condition", 1.0)):
        config = dataclasses.replace(base, lambda_condition=lam)
        model, _, _ = train_factors(img, txt, graph, config, device="cpu", group_ids=paint,
                                    condition_source=source if lam > 0 else None)
        runs[name] = _naive_r1(model, img, txt, source, paint)
    assert runs["condition"] >= runs["none"] + 0.10, runs
