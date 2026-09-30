import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from scripts.buddy_percept_sweep import h2h_topics
from scripts.buddy_percept_sweep.h2h_topics import build_topic_graph, leiden_on_graph, target_k_partition


def _cliques(n_cliques, size):
    rows, cols = [], []
    for c in range(n_cliques):
        base = c * size
        for i in range(size):
            for j in range(size):
                if i != j:
                    rows.append(base + i); cols.append(base + j)
        rows.append(base); cols.append(((c + 1) % n_cliques) * size)   # ring link keeps it connected
        cols.append(base); rows.append(((c + 1) % n_cliques) * size)
    n = n_cliques * size
    return csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def test_leiden_finds_cliques():
    labels = leiden_on_graph(_cliques(6, 10), resolution=1.0, seed=0)
    assert len(np.unique(labels)) == 6


def test_target_k_hits_band():
    emb = np.random.default_rng(0).normal(size=(60, 4)).astype(np.float32)
    labels, info = target_k_partition(emb, _cliques(6, 10), k_target=6, tolerance=0,
                                      merge_threshold=0.0, seed=0)
    assert info["hit"] and len(np.unique(labels)) == 6


def test_target_k_reports_miss_instead_of_raising():
    emb = np.random.default_rng(0).normal(size=(60, 4)).astype(np.float32)
    labels, info = target_k_partition(emb, _cliques(6, 10), k_target=40, tolerance=0,
                                      merge_threshold=0.0, seed=0, max_steps=4)
    assert info["hit"] is False and len(labels) == 60


# ---------------------------------------------------------------- leiden_on_graph

def test_leiden_labels_are_contiguous_int64_and_one_per_clique():
    labels = leiden_on_graph(_cliques(6, 10), resolution=1.0, seed=3)
    assert labels.dtype == np.int64 and labels.shape == (60,)
    assert sorted(np.unique(labels).tolist()) == list(range(6))
    for c in range(6):
        assert len(np.unique(labels[c * 10:(c + 1) * 10])) == 1


@pytest.mark.parametrize("seed", [0, 42])
def test_leiden_at_resolution_one_equals_pilot_detect_communities(seed):
    """Spec §4: pilot_repaired + resolution 1.0 reproduces the pilots'
    modularity Leiden (`detect_communities`) on the same graph and seed."""
    from src.conditional_buddy.buddy_graph import mutual_knn
    from src.conditional_buddy.prototype_seed import detect_communities

    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(10, 16))
    emb = (centers[rng.integers(0, 10, 400)] + 0.6 * rng.normal(size=(400, 16))).astype(np.float32)
    graph = mutual_knn(emb, K=15, device="cpu")
    ours = leiden_on_graph(graph, resolution=1.0, seed=seed)
    theirs = detect_communities(graph, seed=seed)
    assert np.array_equal(ours, theirs)


def test_leiden_is_deterministic_per_seed():
    rng = np.random.default_rng(1)
    dense = (rng.random((80, 80)) < 0.08).astype(float)
    dense = np.triu(dense, 1); dense = dense + dense.T
    graph = csr_matrix(dense)
    assert np.array_equal(leiden_on_graph(graph, 0.7, 5), leiden_on_graph(graph, 0.7, 5))


# ---------------------------------------------------------------- target_k_partition

def test_target_k_bisects_log_resolution_and_returns_closest_on_miss(monkeypatch):
    """Fake Leiden with K = floor(10 * r): start at r=1, then geometric
    midpoints of the live bracket; the miss returns the closest-K result."""
    seen = []

    def fake_leiden(graph, resolution, seed):
        seen.append(resolution)
        k = max(1, int(10 * resolution))
        return np.arange(100) % k

    monkeypatch.setattr(h2h_topics, "leiden_on_graph", fake_leiden)
    emb = np.zeros((100, 2), dtype=np.float32)
    labels, info = target_k_partition(emb, None, k_target=40, tolerance=0, merge_threshold=0.0,
                                      seed=0, max_steps=3, lo=0.02, hi=20.0)
    r2 = math.exp((math.log(1.0) + math.log(20.0)) / 2)          # K=10 < 40 -> lo = 1
    r3 = math.exp((math.log(1.0) + math.log(r2)) / 2)            # K=44 > 40 -> hi = r2
    assert seen == pytest.approx([1.0, r2, r3])
    assert info["hit"] is False and info["steps"] == 3
    assert info["resolution"] == pytest.approx(r2)                # K=44 is closest to 40 (vs 10, 21)
    assert info["k_raw"] == 44 and info["k_after_merge"] == 44
    assert len(np.unique(labels)) == 44


def test_target_k_stops_at_first_hit_within_tolerance(monkeypatch):
    seen = []

    def fake_leiden(graph, resolution, seed):
        seen.append(resolution)
        return np.arange(100) % max(1, int(10 * resolution))

    monkeypatch.setattr(h2h_topics, "leiden_on_graph", fake_leiden)
    labels, info = target_k_partition(np.zeros((100, 2), dtype=np.float32), None, k_target=12,
                                      tolerance=2, merge_threshold=0.0, seed=0)
    assert seen == [1.0] and info["hit"] is True and info["steps"] == 1
    assert info["k_raw"] == 10 and info["k_after_merge"] == 10 and info["resolution"] == 1.0


def test_target_k_counts_topics_after_merge_with_given_threshold(monkeypatch):
    calls = []
    real_merge = h2h_topics.merge_small_communities

    def spy_merge(embedding, labels, min_fraction):
        calls.append(min_fraction)
        return real_merge(embedding, labels, min_fraction)

    def fake_leiden(graph, resolution, seed):
        # 5 large communities (19 each) + 5 singletons -> merge 0.05 leaves 5.
        return np.concatenate([np.repeat(np.arange(5), 19), np.arange(5, 10)])

    monkeypatch.setattr(h2h_topics, "merge_small_communities", spy_merge)
    monkeypatch.setattr(h2h_topics, "leiden_on_graph", fake_leiden)
    emb = np.random.default_rng(0).normal(size=(100, 3)).astype(np.float32)
    labels, info = target_k_partition(emb, None, k_target=5, tolerance=0, merge_threshold=0.05, seed=0)
    assert calls == [0.05]
    assert info["hit"] and info["k_raw"] == 10 and info["k_after_merge"] == 5
    assert sorted(np.unique(labels).tolist()) == list(range(5))


def test_target_k_passes_seed_to_every_leiden_call(monkeypatch):
    seeds = []

    def fake_leiden(graph, resolution, seed):
        seeds.append(seed)
        return np.zeros(10, dtype=np.int64)

    monkeypatch.setattr(h2h_topics, "leiden_on_graph", fake_leiden)
    target_k_partition(np.zeros((10, 2), dtype=np.float32), None, k_target=5, tolerance=0,
                       merge_threshold=0.0, seed=17, max_steps=4)
    assert seeds == [17, 17, 17, 17]


# ---------------------------------------------------------------- build_topic_graph

def test_build_topic_graph_mknn_is_symmetric_binary():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(50, 6)).astype(np.float32)
    graph = build_topic_graph(emb, "mknn", pilot=None, device="cpu", k_neighbors=5)
    assert isinstance(graph, csr_matrix) and graph.shape == (50, 50)
    assert (graph != graph.T).nnz == 0
    assert set(np.unique(graph.data).tolist()) == {1.0}


def test_build_topic_graph_pilot_repaired_calls_pilot_builder():
    seen = []
    marker = csr_matrix((4, 4))

    def build_single_modality_graph(name, nodes, pipeline, affect_pilot, device, expected_nodes):
        seen.append((name, nodes, pipeline, affect_pilot, device, expected_nodes))
        return marker

    pilot = SimpleNamespace(pipeline="train-pipeline", heldout_pipeline="heldout-pipeline",
                            affect_pilot="affect",
                            single_modality=SimpleNamespace(build_single_modality_graph=build_single_modality_graph))
    emb = np.ones((4, 3), dtype=np.float32)
    assert build_topic_graph(emb, "pilot_repaired", pilot, "cuda:1") is marker
    (name, nodes, pipeline, affect, device, expected), = seen
    assert (name, pipeline, affect, device, expected) == ("train-topics", "train-pipeline", "affect", "cuda:1", 4)
    assert nodes is emb


def test_build_topic_graph_rejects_unknown_kind():
    with pytest.raises(ValueError):
        build_topic_graph(np.zeros((4, 2), dtype=np.float32), "knn", None, "cpu")
