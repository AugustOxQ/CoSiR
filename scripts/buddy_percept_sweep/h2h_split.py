"""Deterministic val/test halving of the held-out paintings, stratified by
(majority emotion, genre present)."""
import hashlib
import math

import numpy as np

from scripts.buddy_percept_sweep.h2h_types import H2HSplit


def make_split(heldout_emotion: np.ndarray, heldout_genre: np.ndarray, seed: int = 0) -> H2HSplit:
    emotion = np.asarray(heldout_emotion)
    has_genre = np.asarray(heldout_genre) != ""
    if len(emotion) != len(has_genre):
        raise ValueError("emotion and genre must have the same length")
    rng = np.random.default_rng(seed)
    strata = sorted({(str(e), bool(g)) for e, g in zip(emotion, has_genre)})
    val_parts, test_parts = [], []
    for label, present in strata:
        idx = np.flatnonzero((emotion.astype(str) == label) & (has_genre == present))
        perm = rng.permutation(idx)
        n_val = math.ceil(len(perm) / 2)
        val_parts.append(perm[:n_val])
        test_parts.append(perm[n_val:])
    val = np.sort(np.concatenate(val_parts)).astype(np.int64)
    test = np.sort(np.concatenate(test_parts)).astype(np.int64)
    digest = hashlib.sha1(val.tobytes() + test.tobytes()).hexdigest()
    return H2HSplit(val_idx=val, test_idx=test, digest=digest)
