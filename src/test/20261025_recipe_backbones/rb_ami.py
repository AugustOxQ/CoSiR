"""Part 1 partitions and AMI (rule §3.1): fixed k-means partitions of a recipe's P export, the score A and the draws.

The statistic is 12-22's (`pav_score.partitions`, `ab_score`): MiniBatchKMeans k = 20 at random states 0, 1, 2
(n_init 3, batch 4,096, other arguments default) and `adjusted_mutual_info_score` with default arguments. Differences
this rule sets: all of P is clustered before any label is masked, the float32 cast of the saved float16 export is
used as it is (unit norm checked to 1e-3, not re-normalized), and the draws are the paintings only (no seeds).
"""
import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score as ami
from threadpoolctl import threadpool_limits

K = 20
STATES = (0, 1, 2)
N_INIT = 3
BATCH_SIZE = 4096
THREADS = 8
NORM_TOL = 1e-3
N_DRAWS = 2000


def versions():
    return dict(sklearn=sklearn.__version__, numpy=np.__version__, scipy=scipy.__version__,
                threadpoolctl=threadpoolctl.__version__)


def unit_float32(x):
    """The float32 cast of a saved float16 (or float32) export; finite and unit norm within 1e-3; not re-normalized."""
    x = np.asarray(x)
    assert x.dtype in (np.float16, np.float32) and x.ndim == 2, (x.dtype, x.shape)
    x = x.astype(np.float32)
    assert np.isfinite(x).all(), 'nonfinite rows'
    assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < NORM_TOL, 'rows not unit norm'
    return x


def partitions(x):
    """int64 [3, n]: k-means labels of all rows of `x`, one row per random state 0, 1, 2."""
    x = unit_float32(x)
    with threadpool_limits(limits=THREADS):
        parts = [MiniBatchKMeans(n_clusters=K, random_state=s, n_init=N_INIT, batch_size=BATCH_SIZE).fit_predict(x)
                 for s in STATES]
    return np.stack(parts).astype(np.int64)


def score(labels, parts, idx):
    """A: the mean over the three partitions of AMI(labels[idx], parts[j][idx]); `idx` indexes P (repeats allowed).

    `labels` is the aspect's label vector over all of P (-1 masked), `parts` int [3, n]; every indexed row is labelled.
    """
    labels, parts, idx = np.asarray(labels), np.asarray(parts), np.asarray(idx)
    assert labels.ndim == 1 and parts.shape == (len(STATES), len(labels)), (labels.shape, parts.shape)
    assert np.issubdtype(idx.dtype, np.integer) and idx.ndim == 1 and len(idx) > 0
    assert idx.min() >= 0 and idx.max() < len(labels)
    y = labels[idx]
    assert (y >= 0).all(), 'index vector holds a masked row'
    return float(np.mean([ami(y, parts[j][idx]) for j in range(len(STATES))]))


def draws(n_labelled, n_draws=N_DRAWS):
    """int32 [n_draws, n_labelled]: positions in L_a of the bootstrap rows, 12-22's draws.

    One `default_rng(0)` per aspect; draw b is `rng.integers(0, n, size=n)`; the caller indexes L_a with it.
    """
    assert n_labelled > 0 and n_draws > 0
    rng = np.random.default_rng(0)
    out = np.empty((n_draws, n_labelled), dtype=np.int32)
    for b in range(n_draws):
        out[b] = rng.integers(0, n_labelled, size=n_labelled)
    return out
