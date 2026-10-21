"""Metric helpers for the r10 D2 ablation (spec D18, section 3.6). Pure numpy / scikit-learn, no file I/O.

probe            5-fold stratified multinomial logistic regression; balanced accuracy over pooled out-of-fold predictions.
linear_cka       linear CKA (Kornblith et al. 2019), feature Gram form, float64.
knn_overlap      mean top-k cosine neighbour overlap between two views of the same rows (blockwise); knn_chance.
size_entropy     entropy of cluster sizes divided by log k.
effective_rank   Roy and Vetterli (2007) effective rank of the column-centred matrix.
principal_angles cosines of the principal angles between top-r PCA subspaces; subspace_chance.
"""
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


def _centred(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=0, keepdims=True)


# ---------------------------------------------------------------------------------------------------- probe
def probe(x, y, folds=5, seed=0, C=1.0, max_iter=1000):
    """Linear probe of labels y on features x (D18).

    Rows with y < 0 are dropped. Stratified `folds`-fold split (shuffle, random_state=seed); per fold a StandardScaler
    and a multinomial LogisticRegression(C, max_iter) fitted on the training folds only. Convergence warnings are
    counted (over all folds), returned, and re-emitted once as a single ConvergenceWarning; they are not silenced.

    Returns a dict: balanced_accuracy (pooled out-of-fold), per_class_recall (array aligned with `classes`),
    classes (sorted labels kept), majority_rate (largest class share), chance (1 / n_classes), n, n_classes,
    convergence_warnings (int).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    keep = y >= 0
    x, y = x[keep], y[keep]
    classes, yi = np.unique(y, return_inverse=True)
    nc = len(classes)
    pred = np.full(len(yi), -1, dtype=np.int64)
    n_conv = 0
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(x, yi):
        sc = StandardScaler().fit(x[tr])
        clf = LogisticRegression(C=C, max_iter=max_iter)  # lbfgs, multinomial for >2 classes
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            clf.fit(sc.transform(x[tr]), yi[tr])
        for w in caught:
            if issubclass(w.category, ConvergenceWarning):
                n_conv += 1
            else:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        pred[te] = clf.predict(sc.transform(x[te]))
    if n_conv:
        warnings.warn(f'probe: {n_conv} of {folds} folds did not converge (max_iter={max_iter})', ConvergenceWarning)
    counts = np.bincount(yi, minlength=nc)
    recall = np.bincount(yi[pred == yi], minlength=nc) / counts
    return dict(balanced_accuracy=float(recall.mean()), per_class_recall=recall, classes=classes,
                majority_rate=float(counts.max() / counts.sum()), chance=1.0 / nc, n=int(len(yi)),
                n_classes=int(nc), convergence_warnings=int(n_conv))


# ---------------------------------------------------------------------------------------------------- CKA
def linear_cka(x, y):
    """Linear CKA of x (n, dx) and y (n, dy) on column-centred features: ||Yc'Xc||_F^2 / (||Xc'Xc||_F ||Yc'Yc||_F)."""
    xc, yc = _centred(x), _centred(y)
    if xc.shape[0] != yc.shape[0]:
        raise ValueError('x and y need the same number of rows')
    num = np.linalg.norm(yc.T @ xc, 'fro') ** 2
    den = np.linalg.norm(xc.T @ xc, 'fro') * np.linalg.norm(yc.T @ yc, 'fro')
    return float(num / den)


# ---------------------------------------------------------------------------------------------------- kNN overlap
def _unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _topk_cosine(u, rows, k):
    """Indices (len(rows), k) of the top-k cosine neighbours of u[rows] among all rows of u, self excluded."""
    s = u[rows] @ u.T
    s[np.arange(len(rows)), rows] = -np.inf
    return np.argpartition(-s, k - 1, axis=1)[:, :k]


def knn_overlap(x, y, k=20, block=500):
    """Mean over rows of |top-k cosine neighbours in x and in y| / k, self excluded. Blockwise over rows."""
    n = len(x)
    if len(y) != n:
        raise ValueError('x and y need the same number of rows')
    if not 1 <= k <= n - 1:
        raise ValueError('need 1 <= k <= n - 1')
    ux, uy = _unit(x), _unit(y)
    total = 0
    for a in range(0, n, block):
        rows = np.arange(a, min(a + block, n))
        nx, ny = _topk_cosine(ux, rows, k), _topk_cosine(uy, rows, k)
        total += int((nx[:, :, None] == ny[:, None, :]).sum())
    return total / (n * k)


def knn_chance(n, k):
    """Expected overlap of two unrelated neighbour sets: k / (n - 1)."""
    return k / (n - 1)


# ---------------------------------------------------------------------------------------------------- partitions, rank
def size_entropy(partition, k):
    """Entropy of the cluster sizes (labels 0..k-1) divided by log k: 1 for equal sizes, 0 for one cluster."""
    p = np.bincount(np.asarray(partition, dtype=np.int64), minlength=k).astype(np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(k))


def effective_rank(x):
    """Roy and Vetterli (2007): exp of the entropy of the singular values (divided by their sum) of the centred x."""
    s = np.linalg.svd(_centred(x), compute_uv=False)
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


# ---------------------------------------------------------------------------------------------------- subspaces
def _top_subspace(x, r):
    # right singular vectors of the centred matrix: orthonormal (d, r) basis of the top-r PCA subspace
    _, _, vt = np.linalg.svd(_centred(x), full_matrices=False)
    return vt[:r].T


def principal_angles(a, b, r):
    """Cosines (descending, length r) of the principal angles between the top-r PCA subspaces of a and b (centred),
    and their mean squared cosine. a and b share the feature dimension."""
    if np.shape(a)[1] != np.shape(b)[1]:
        raise ValueError('a and b need the same number of columns')
    cos = np.linalg.svd(_top_subspace(a, r).T @ _top_subspace(b, r), compute_uv=False)
    cos = np.clip(cos, 0.0, 1.0)
    return cos, float((cos ** 2).mean())


def subspace_chance(r, d):
    """Expected mean squared cosine of two unrelated r-dimensional subspaces of R^d: r / d."""
    return r / d
