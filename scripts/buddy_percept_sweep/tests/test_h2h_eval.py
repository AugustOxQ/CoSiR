import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from scripts.buddy_percept_sweep import h2h_eval
from scripts.buddy_percept_sweep.h2h_eval import _auc_for, eval_labels, stage1_metrics, stage2_metrics
from scripts.buddy_percept_sweep.h2h_store import H2HStore
from scripts.buddy_percept_sweep.h2h_trial import Stage2Config
from scripts.buddy_percept_sweep.h2h_types import EVAL_TRANSFER_K


def test_eval_labels_follow_nearest_train_topic():
    rng = np.random.default_rng(0)
    a = rng.normal([20, 0], 0.1, size=(50, 2)); b = rng.normal([0, 20], 0.1, size=(50, 2))
    train = np.vstack([a, b]).astype(np.float32); labels = np.array([0] * 50 + [1] * 50)
    held = np.vstack([rng.normal([20, 0], 0.1, size=(10, 2)), rng.normal([0, 20], 0.1, size=(10, 2))]).astype(np.float32)
    out = eval_labels(train, labels, held, np.arange(20))
    assert list(out) == [0] * 10 + [1] * 10


def test_eval_labels_uses_fixed_k_and_only_the_subset(monkeypatch):
    seen = {}

    def spy(train_embeddings, train_labels, query_embeddings, k):
        seen["k"] = k
        seen["query"] = query_embeddings.copy()
        return np.zeros(len(query_embeddings), dtype=np.int64)

    monkeypatch.setattr(h2h_eval, "assign_to_train_communities", spy)
    held = np.arange(12, dtype=np.float32).reshape(6, 2)
    out = eval_labels(np.zeros((30, 2), dtype=np.float32), np.zeros(30, dtype=np.int64), held, np.array([1, 4]))
    assert seen["k"] == EVAL_TRANSFER_K == 20
    assert np.array_equal(seen["query"], held[[1, 4]]) and len(out) == 2


# ---------------------------------------------------------------- _auc_for (Review Focus 2)

def test_auc_for_skips_topic_absent_from_labels_and_stays_finite():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 10 + [1] * 10)          # topic 2 has no evaluation-label members
    scores = rng.random((20, 3))
    scores[:10, 0] += 1.0                            # topic 0 separable
    macro, skipped = _auc_for(scores, labels, 3)
    assert skipped == 1
    assert math.isfinite(macro) and 0.0 <= macro <= 1.0


def test_auc_for_all_topics_skipped_gives_nan_not_crash():
    macro, skipped = _auc_for(np.random.default_rng(0).random((5, 2)), np.zeros(5, dtype=np.int64), 2)
    assert skipped == 2 and math.isnan(macro)


# ---------------------------------------------------------------- stage1_metrics

def _stub_pilot(seen):
    """Independent re-clustering on a stub graph: nodes connected iff they
    share the sign of their first coordinate (two cliques)."""
    def build_single_modality_graph(name, nodes, pipeline, affect_pilot, device, expected_nodes):
        seen.append((name, pipeline, expected_nodes, device))
        side = np.asarray(nodes)[:, 0] > 0
        dense = (side[:, None] == side[None, :]).astype(float)
        np.fill_diagonal(dense, 0.0)
        return csr_matrix(dense)

    return SimpleNamespace(pipeline="train-pipeline", heldout_pipeline="heldout-pipeline", affect_pilot="affect",
                           single_modality=SimpleNamespace(build_single_modality_graph=build_single_modality_graph))


def _two_blob_data(rng):
    train = np.vstack([rng.normal([5, 0], 0.1, size=(30, 2)), rng.normal([-5, 0], 0.1, size=(30, 2))]).astype(np.float32)
    train_labels = np.array([0] * 30 + [1] * 30)
    # held-out rows alternate blob; subset picks the first 6 of each blob
    held = np.vstack([rng.normal([5, 0], 0.1, size=(8, 2)), rng.normal([-5, 0], 0.1, size=(8, 2))]).astype(np.float32)
    emotion = np.array(["awe"] * 8 + ["fear"] * 8, dtype=object)
    genre = np.array(["x"] * 8 + ["y"] * 8, dtype=object)
    subset = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])
    return train, train_labels, held, emotion, genre, subset


def test_stage1_metrics_keys_values_and_pilot_call():
    rng = np.random.default_rng(0)
    train, train_labels, held, emotion, genre, subset = _two_blob_data(rng)
    seen = []
    out = stage1_metrics(train, train_labels, held, subset, emotion, genre, seed=3,
                         pilot=_stub_pilot(seen), device="cpu")
    assert set(out) == {"transfer_emo", "transfer_genre", "ind_emo", "ind_genre", "ind_k"}
    assert out["transfer_emo"] == pytest.approx(1.0) and out["transfer_genre"] == pytest.approx(1.0)
    assert out["ind_emo"] == pytest.approx(1.0) and out["ind_genre"] == pytest.approx(1.0)
    assert out["ind_k"] == 2 and isinstance(out["ind_k"], int)
    assert seen == [("heldout-independent", "heldout-pipeline", 12, "cpu")]


def test_stage1_metrics_native_keys_only_with_native_subset():
    rng = np.random.default_rng(0)
    train, train_labels, held, emotion, genre, subset = _two_blob_data(rng)
    native = np.array([0] * 6 + [1] * 6)
    out = stage1_metrics(train, train_labels, held, subset, emotion, genre, seed=3,
                         pilot=_stub_pilot([]), device="cpu", native_subset=native)
    assert out["native_emo"] == pytest.approx(1.0) and out["native_genre"] == pytest.approx(1.0)


def test_stage1_metrics_reuses_given_transfer_labels(monkeypatch):
    rng = np.random.default_rng(0)
    train, train_labels, held, emotion, genre, subset = _two_blob_data(rng)
    monkeypatch.setattr(h2h_eval, "eval_labels", lambda *a, **k: pytest.fail("recomputed transfer labels"))
    given = np.array([0] * 6 + [1] * 6)
    out = stage1_metrics(train, train_labels, held, subset, emotion, genre, seed=3,
                         pilot=_stub_pilot([]), device="cpu", transfer_labels=given)
    assert out["transfer_emo"] == pytest.approx(1.0)


# ---------------------------------------------------------------- stage2_metrics

def _tiny_store(n_train=40, n_heldout=20, d=8, n_topics=4, seed=0):
    rng = np.random.default_rng(seed)
    train_labels = np.arange(n_train) % n_topics
    heldout_labels = np.arange(n_heldout) % n_topics
    protos = rng.normal(size=(n_topics, d))
    train_patches = protos[train_labels][:, None, :] + 0.5 * rng.normal(size=(n_train, 50, d))
    heldout_patches = protos[heldout_labels][:, None, :] + 0.5 * rng.normal(size=(n_heldout, 50, d))
    empty = np.zeros(0)
    store = H2HStore(
        train_paintings=empty, heldout_paintings=np.empty(n_heldout, dtype=object),
        train_img=empty, train_txt=empty, heldout_img=empty, heldout_txt=empty,
        train_content_raw=empty, heldout_content_raw=empty, train_affect28=empty, heldout_affect28=empty,
        train_percept_h=empty, heldout_percept_h=empty,
        train_emotion=empty, heldout_emotion=empty, train_genre=empty, heldout_genre=empty,
        train_patches=torch.as_tensor(train_patches, dtype=torch.float32),
        heldout_patches=torch.as_tensor(heldout_patches, dtype=torch.float32),
    )
    train_emb = (protos[train_labels][:, :3] + 0.1 * rng.normal(size=(n_train, 3))).astype(np.float32)
    return store, train_emb, train_labels, heldout_labels


def test_stage2_metrics_one_mapper_scored_on_every_label_set(monkeypatch):
    store, train_emb, train_labels, heldout_labels = _tiny_store()
    built = []
    real_mapper = h2h_eval.ParameterizedAttentionPoolingMapper

    def spy_mapper(**kwargs):
        built.append(kwargs)
        return real_mapper(**kwargs)

    monkeypatch.setattr(h2h_eval, "ParameterizedAttentionPoolingMapper", spy_mapper)
    subset = np.array([0, 3, 5, 6, 9, 10, 12, 15])     # every topic twice
    sets ={"primary": heldout_labels[subset], "native": (heldout_labels[subset] + 1) % 4}
    out = stage2_metrics(Stage2Config(mapper_epochs=30), store, train_emb, train_labels, subset, sets,
                         seed=0, device="cpu")
    assert set(out) == {"auc_primary", "skipped_primary", "auc_native", "skipped_native"}
    assert len(built) == 1
    assert built[0]["n_topics"] == 4 and built[0]["d_model"] == 8
    assert all(math.isfinite(out[k]) for k in ("auc_primary", "auc_native"))
    assert out["skipped_primary"] == 0 and isinstance(out["skipped_primary"], int)
    assert out["auc_primary"] > 0.8 > out["auc_native"]   # the mapper learned the primary topics


def test_stage2_metrics_is_deterministic_per_seed():
    store, train_emb, train_labels, heldout_labels = _tiny_store()
    subset = np.arange(20)
    cfg = Stage2Config(mapper_epochs=10, target_cutoff=0.3, train_target_k=5, class_balanced_loss=True)
    first = stage2_metrics(cfg, store, train_emb, train_labels, subset, {"primary": heldout_labels}, 7, "cpu")
    torch.rand(10); np.random.rand(10)
    second = stage2_metrics(cfg, store, train_emb, train_labels, subset, {"primary": heldout_labels}, 7, "cpu")
    assert first == second


def test_stage2_metrics_cutoff_targets_use_train_target_k(monkeypatch):
    store, train_emb, train_labels, heldout_labels = _tiny_store()
    seen = {}
    real_votes, real_build = h2h_eval.cosine_vote_fractions, h2h_eval.build_targets

    def spy_votes(train_embeddings, labels, query_embeddings, n_topics, k):
        seen["k"], seen["n_votes"] = k, n_topics
        return real_votes(train_embeddings, labels, query_embeddings, n_topics, k)

    def spy_build(fractions, n_topics, cutoff):
        seen["cutoff"] = cutoff
        return real_build(fractions, n_topics, cutoff)

    monkeypatch.setattr(h2h_eval, "cosine_vote_fractions", spy_votes)
    monkeypatch.setattr(h2h_eval, "build_targets", spy_build)
    stage2_metrics(Stage2Config(mapper_epochs=2, target_cutoff=0.3, train_target_k=7), store, train_emb,
                   train_labels, np.arange(20), {"primary": heldout_labels}, 0, "cpu")
    assert seen == {"k": 7, "n_votes": 4, "cutoff": 0.3}


def test_stage2_metrics_width_is_number_of_train_topics():
    """Review Focus 4: target width = distinct train labels actually used."""
    store, train_emb, _, heldout_labels = _tiny_store()
    labels3 = np.arange(40) % 3
    out = stage2_metrics(Stage2Config(mapper_epochs=2), store, train_emb, labels3, np.arange(20),
                         {"primary": heldout_labels % 3}, 0, "cpu")
    assert out["skipped_primary"] == 0


def test_stage2_metrics_rejects_non_contiguous_train_labels():
    store, train_emb, _, heldout_labels = _tiny_store()
    gappy = np.where(np.arange(40) % 2 == 0, 0, 2)
    with pytest.raises(ValueError):
        stage2_metrics(Stage2Config(mapper_epochs=1), store, train_emb, gappy, np.arange(20),
                       {"primary": heldout_labels % 2}, 0, "cpu")
