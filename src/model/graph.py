"""Typed entry point to the existing content buddy graph construction."""

from dataclasses import dataclass

import numpy as np
import torch
from scipy.sparse import csr_matrix

from src.conditional_buddy import buddy_graph


@dataclass
class GraphConfig:
    """Settings inherited from the existing buddy graph pipeline.

    ``k`` is the per-modality mutual-KNN size (``K`` in ``buddy_graph``).
    ``alpha`` weights image and text similarity when connecting components.
    ``min_degree=1`` reflects ``buddy_graph.ensure_min_degree``'s fixed
    guarantee; that function does not support another degree threshold.
    ``seed`` matches the wider buddy pipeline's configuration, but these
    graph-construction functions do not consume a random seed.
    """

    k: int = 30
    alpha: float = 0.5
    min_degree: int = 1
    seed: int = 42


def build_content_graph(
    img_features: np.ndarray,
    txt_features: np.ndarray,
    config: GraphConfig,
) -> csr_matrix:
    """Return the connected, symmetric binary content graph for paired features.

    Feature rows are copied and L2-normalized because ``ensure_connected``
    expects normalized inputs. Graph topology is delegated to ``buddy_graph``.
    """
    if config.min_degree != 1:
        raise ValueError("buddy_graph.ensure_min_degree supports only min_degree=1")

    img_normalized = np.array(img_features, dtype=np.float32, order="C", copy=True)
    txt_normalized = np.array(txt_features, dtype=np.float32, order="C", copy=True)
    img_normalized /= np.linalg.norm(img_normalized, axis=1, keepdims=True) + 1e-12
    txt_normalized /= np.linalg.norm(txt_normalized, axis=1, keepdims=True) + 1e-12

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = device == "cuda"
    img_graph = buddy_graph.mutual_knn(
        img_normalized, config.k, device=device, use_half=use_half
    )
    txt_graph = buddy_graph.mutual_knn(
        txt_normalized, config.k, device=device, use_half=use_half
    )
    graph = buddy_graph.union_graph(img_graph, txt_graph)
    graph, _ = buddy_graph.ensure_min_degree(
        graph, img_normalized, txt_normalized, device=device, use_half=use_half
    )
    graph, _ = buddy_graph.ensure_connected(
        graph,
        img_normalized,
        txt_normalized,
        alpha=config.alpha,
        device=device,
        use_half=use_half,
    )
    return graph
