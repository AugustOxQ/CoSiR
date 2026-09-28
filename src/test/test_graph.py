"""Behavior tests for the content teacher graph wrapper."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.model.graph import GraphConfig, build_content_graph


@pytest.fixture
def feature_pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    img = rng.standard_normal((32, 12)).astype(np.float32)
    txt = rng.standard_normal((32, 10)).astype(np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    return img, txt


def test_content_graph_is_symmetric_binary_adjacency(feature_pair):
    img, txt = feature_pair

    graph = build_content_graph(img, txt, GraphConfig(k=2))

    assert isinstance(graph, csr_matrix)
    assert graph.shape == (32, 32)
    assert (graph != graph.T).nnz == 0
    assert np.all(graph.data == 1)


def test_content_graph_connects_all_nodes(feature_pair):
    img, txt = feature_pair

    # At k=1 this fixed fixture has isolated nodes and separate components.
    graph = build_content_graph(img, txt, GraphConfig(k=1))

    assert np.all(np.diff(graph.indptr) >= 1)
    assert connected_components(graph, directed=False, return_labels=False) == 1


def test_content_graph_is_repeatable_with_seed_42(feature_pair):
    img, txt = feature_pair
    config = GraphConfig(k=2, seed=42)

    first = build_content_graph(img, txt, config)
    second = build_content_graph(img, txt, config)

    assert (first != second).nnz == 0


def test_content_graph_rejects_unsupported_min_degree(feature_pair):
    img, txt = feature_pair

    with pytest.raises(ValueError, match="min_degree=1"):
        build_content_graph(img, txt, GraphConfig(k=2, min_degree=2))
