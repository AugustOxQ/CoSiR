"""Per-agent-process cache of the fixed, hyperparameter-independent inputs
(dedup CLIP features, GoEmotions affect embeddings, patch features) plus a
PCA-fit cache keyed by `content_pca_dim`, since that one input IS swept
(spec §7 risk: "content_pca_dim as an extra invalidates the fit-once PCA
cache").
"""
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA


@dataclass
class RawInputs:
    train_content_raw: np.ndarray
    heldout_content_raw: np.ndarray
    train_affect: np.ndarray
    heldout_affect: np.ndarray
    train_emotion: list
    heldout_emotion: list
    train_genre: np.ndarray
    heldout_genre: np.ndarray
    train_patches: torch.Tensor
    heldout_patches: torch.Tensor


@dataclass
class FixedInputs:
    train_content: np.ndarray
    train_affect: np.ndarray
    heldout_content: np.ndarray
    heldout_affect: np.ndarray
    train_emotion: list
    heldout_emotion: list
    train_genre: np.ndarray
    heldout_genre: np.ndarray
    train_patches: torch.Tensor
    heldout_patches: torch.Tensor
    content_pca_dim: int


class FixedInputCache:
    """Lives for the lifetime of one wandb agent process."""

    def __init__(self) -> None:
        self._raw: Optional[RawInputs] = None
        self._fitted_by_dim: dict[int, FixedInputs] = {}

    def get(self, content_pca_dim: int, raw_loader: Callable[[], RawInputs]) -> FixedInputs:
        if self._raw is None:
            self._raw = raw_loader()
        if content_pca_dim not in self._fitted_by_dim:
            pca = PCA(n_components=content_pca_dim, random_state=42)
            train_content = pca.fit_transform(self._raw.train_content_raw).astype(np.float32)
            heldout_content = pca.transform(self._raw.heldout_content_raw).astype(np.float32)
            self._fitted_by_dim[content_pca_dim] = FixedInputs(
                train_content=train_content,
                train_affect=self._raw.train_affect,
                heldout_content=heldout_content,
                heldout_affect=self._raw.heldout_affect,
                train_emotion=self._raw.train_emotion,
                heldout_emotion=self._raw.heldout_emotion,
                train_genre=self._raw.train_genre,
                heldout_genre=self._raw.heldout_genre,
                train_patches=self._raw.train_patches,
                heldout_patches=self._raw.heldout_patches,
                content_pca_dim=content_pca_dim,
            )
        return self._fitted_by_dim[content_pca_dim]
