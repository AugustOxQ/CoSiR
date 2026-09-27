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
        self._pca_dim: Optional[int] = None
        self._fitted: Optional[FixedInputs] = None

    def get(self, content_pca_dim: int, raw_loader: Callable[[], RawInputs]) -> FixedInputs:
        if self._raw is None:
            self._raw = raw_loader()
        if self._fitted is None or self._pca_dim != content_pca_dim:
            fit_dim = min(content_pca_dim, *self._raw.train_content_raw.shape)
            pca = PCA(n_components=fit_dim, random_state=42)
            train_content = pca.fit_transform(self._raw.train_content_raw).astype(np.float32)
            heldout_content = pca.transform(self._raw.heldout_content_raw).astype(np.float32)
            if fit_dim < content_pca_dim:
                padding = (0, content_pca_dim - fit_dim)
                train_content = np.pad(train_content, ((0, 0), padding))
                heldout_content = np.pad(heldout_content, ((0, 0), padding))
            self._fitted = FixedInputs(
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
            self._pca_dim = content_pca_dim
        return self._fitted
