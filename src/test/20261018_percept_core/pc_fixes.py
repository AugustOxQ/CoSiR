"""Pair rows, Eq. 2 fusion, fit-only maps, center selection and caption unions."""
import re

import numpy as np

FIX_TESTS = 30


def unit_tests_pass(line):
    """Spec section 6 (b): the pytest summary of test_pc_fixes.py says exactly FIX_TESTS passed, nothing else."""
    return isinstance(line, str) and re.fullmatch(
        rf'{FIX_TESTS} passed(?:, \d+ warnings?)? in \d+(?:\.\d+)?s(?: \([\d:]+\))?', line.strip()) is not None


def pair_rows(pid, painting_partition, partition='fit'):
    """Caption indices in frozen order; pid indexes the per-painting labels (CM-5)."""
    assert pid.ndim == painting_partition.ndim == 1 and len(pid) > 0
    assert np.issubdtype(pid.dtype, np.integer)
    assert (pid >= 0).all() and (pid < len(painting_partition)).all()
    assert partition in ('fit', 'dev', 'eval')
    assert np.isin(painting_partition, ('fit', 'dev', 'eval')).all()
    return np.flatnonzero(painting_partition[pid] == partition).astype(np.int64)


def fuse_eq2(image, text, emotion):
    """Normalize I/T/E, then apply Eq. 2 row-wise without normalizing h (D11)."""
    assert image.shape == text.shape == emotion.shape
    normalized = []
    for x in (image, text, emotion):
        assert x.ndim == 2 and min(x.shape) > 0 and x.dtype == np.float32
        assert np.isfinite(x).all()
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        assert (norm > 0).all()
        normalized.append(x / norm)
    image, text, emotion = normalized
    content = (image + text) / 2
    return (2 * content * np.linalg.norm(emotion, axis=1, keepdims=True) +
            emotion * np.linalg.norm(content, axis=1, keepdims=True)) / 3


def pca_map(emotion_fit, n_components=512):
    """Return fit mean (D,) and full-SVD map (D,k); apply as (E-mean) @ map."""
    assert emotion_fit.ndim == 2 and emotion_fit.dtype == np.float32
    assert np.isfinite(emotion_fit).all()
    assert 0 < n_components <= min(emotion_fit.shape)
    mean = emotion_fit.mean(axis=0, dtype=np.float64).astype(np.float32)
    _, _, vt = np.linalg.svd(emotion_fit - mean, full_matrices=False)
    vt = vt[:n_components]
    # Fixed signs (largest-magnitude loading positive), so Eq. 2 does not depend on the LAPACK build.
    vt = vt * np.sign(vt[np.arange(len(vt)), np.abs(vt).argmax(axis=1)])[:, None]
    return mean, vt.T.copy()


def orthogonal_map(input_dim, output_dim, seed=0):
    """Return (output_dim,input_dim) QR map; normalized E maps as E @ map.T."""
    assert 0 < input_dim <= output_dim
    matrix = np.random.default_rng(seed).normal(size=(output_dim, input_dim))
    q, r = np.linalg.qr(matrix, mode='reduced')
    # Unique QR (positive diagonal of R), so the map does not depend on the LAPACK build.
    return (q * np.sign(np.diag(r))).astype(np.float32)


def largest_gap(center_norms):
    """Sorted norms, first largest-gap index, its size and the mean gap (D9)."""
    assert center_norms.ndim == 1 and len(center_norms) >= 2
    assert np.isfinite(center_norms).all() and (center_norms >= 0).all()
    ordered = np.sort(center_norms.astype(np.float64))
    gaps = np.diff(ordered)
    index = np.argmax(gaps)
    return ordered, index, gaps[index], gaps.mean()


def kept_centers(center_norms, gamma=2):
    """Algorithm 1 mask in original center order; tied gaps use the first (D9)."""
    assert np.isfinite(gamma) and gamma > 0
    ordered, index, gap, mean_gap = largest_gap(center_norms)
    if gap > gamma * mean_gap:
        threshold = (ordered[index] + ordered[index + 1]) / 2
        return center_norms <= threshold
    return np.ones(center_norms.shape, dtype=bool)


def top_used(fit_counts, k=40):
    """A3 mask in original center order; equal counts favor the lower index."""
    assert fit_counts.ndim == 1 and np.issubdtype(fit_counts.dtype, np.integer)
    assert isinstance(k, int) and 0 < k <= len(fit_counts)
    assert (fit_counts >= 0).all()
    order = np.lexsort((-np.arange(len(fit_counts)), fit_counts))
    kept = np.zeros(len(fit_counts), dtype=bool)
    kept[order[-k:]] = True
    return kept


def multi_hot(pid, topic, painting_ids, n_topics):
    """Caption-topic union in requested painting order, using frozen pid indices."""
    assert pid.ndim == topic.ndim == painting_ids.ndim == 1
    assert len(pid) == len(topic) and len(pid) > 0 and len(painting_ids) > 0
    for x in (pid, topic, painting_ids):
        assert np.issubdtype(x.dtype, np.integer) and (x >= 0).all()
    assert n_topics > 0 and (topic < n_topics).all()
    assert len(np.unique(painting_ids)) == len(painting_ids)
    lookup = np.full(max(pid.max(), painting_ids.max()) + 1, -1, dtype=np.int64)
    lookup[painting_ids] = np.arange(len(painting_ids))
    local = lookup[pid]
    selected = local >= 0
    assert (np.bincount(local[selected], minlength=len(painting_ids)) > 0).all(), \
        'painting without captions in partition'
    target = np.zeros((len(painting_ids), n_topics), dtype=np.float32)
    target[local[selected], topic[selected]] = 1
    return target
