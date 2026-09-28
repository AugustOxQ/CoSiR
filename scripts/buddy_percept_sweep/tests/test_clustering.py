import numpy as np
from scipy.sparse import csr_matrix

from scripts.buddy_percept_sweep import clustering
from scripts.buddy_percept_sweep.clustering import leiden_partition, merge_small_communities


def test_leiden_partition_returns_gap_free_labels():
    rng = np.random.default_rng(0)
    # Two well-separated synthetic blobs.
    cluster_a = rng.normal(loc=0.0, scale=0.05, size=(30, 8))
    cluster_b = rng.normal(loc=5.0, scale=0.05, size=(30, 8))
    embedding = np.concatenate([cluster_a, cluster_b]).astype(np.float32)
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    labels = leiden_partition(embedding, resolution=1.0, k_neighbors=10)
    assert labels.shape == (60,)
    unique = sorted(set(labels.tolist()))
    assert unique == list(range(len(unique)))  # 0-indexed, no gaps


def test_degenerate_low_resolution_does_not_crash():
    rng = np.random.default_rng(0)
    embedding = rng.normal(size=(40, 8)).astype(np.float32)
    embedding /= np.linalg.norm(embedding, axis=1, keepdims=True)
    labels = leiden_partition(embedding, resolution=0.001, k_neighbors=10)
    assert labels.shape == (40,)
    assert len(set(labels.tolist())) >= 1


def test_leiden_graph_has_one_edge_per_mutual_knn_pair(monkeypatch):
    adjacency = csr_matrix(np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ]))
    monkeypatch.setattr(clustering, "mutual_knn", lambda *args, **kwargs: adjacency)

    graphs = []
    original_find_partition = clustering.leidenalg.find_partition

    def capture_graph(graph, *args, **kwargs):
        graphs.append(graph)
        return original_find_partition(graph, *args, **kwargs)

    monkeypatch.setattr(clustering.leidenalg, "find_partition", capture_graph)
    leiden_partition(np.zeros((4, 2), dtype=np.float32), resolution=1.0)

    assert len(graphs) == 1
    assert graphs[0].ecount() == 3
    assert set(graphs[0].get_edgelist()) == {(0, 1), (1, 2), (2, 3)}


def test_merge_small_communities_no_op_at_zero_threshold():
    labels = np.array([0, 0, 0, 1, 1, 2])
    embedding = np.random.default_rng(0).normal(size=(6, 4)).astype(np.float32)
    merged, label_map = merge_small_communities(embedding, labels, min_fraction=0.0)
    assert np.array_equal(merged, labels)
    assert label_map == {0: 0, 1: 1, 2: 2}


def test_merge_small_communities_merges_below_threshold():
    # Community 2 has 1/6 members (~16.7%), below a 20% threshold.
    labels = np.array([0, 0, 0, 1, 1, 2])
    # Explicit directions make centroid cosine similarity unambiguous.
    embedding = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [1.0, -0.1, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.1, 0.9, 0.0, 0.0],
        [0.95, 0.05, 0.0, 0.0],  # community 2, close to community 0
    ], dtype=np.float32)
    merged, label_map = merge_small_communities(embedding, labels, min_fraction=0.20)
    assert len(set(merged.tolist())) == 2  # community 2 absorbed
    assert label_map[2] == label_map[0]  # merged into its nearest (community 0)
