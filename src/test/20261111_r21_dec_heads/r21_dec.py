"""r21: the pure math of a DEC loss on c*'s unit head outputs (ticket 01).

No file reads and no training loop. DEC follows our PercepT build
(`20261018_percept_core/pc_sept.py`, imported, never edited): Student-t soft
assignments (alpha = 1) of unit head outputs z [n, d] to centres [k, d], a
target P from a full pass over the fit paintings, KL(P || q), and PercepT's
balance term. Tensors in and out are torch; k-means works on numpy.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261018_percept_core'))
import pc_sept  # noqa: E402

LAMBDA_BALANCE = pc_sept.LAMBDA_BALANCE  # 1000, PercepT's ratio to the DEC term
BALANCE_CLAMP = 1e-8  # pc_sept's clamp on the mean assignment before the log
FORMS = ('plain', 'start', 'teacher')  # 'off' never reaches this module
ROW_SUM_ATOL = 1e-4
UNIT_ATOL = 1e-3  # head outputs and teacher rows are unit length to this
SOFT_CHUNK = 4096  # rows per chunk of a no-grad soft_assign (the [n, k, d] broadcast)


def _rows(name, a, d=None):
    """Assert `a` is a finite 2-d tensor (with d columns if given)."""
    assert torch.is_tensor(a) and a.ndim == 2, f'{name}: need a 2-d tensor'
    assert torch.isfinite(a).all(), f'{name}: not finite'
    if d is not None:
        assert a.shape[1] == d, f'{name}: {a.shape[1]} columns, need {d}'


def _dist(name, a):
    _rows(name, a)
    assert a.shape[1] >= 2, f'{name}: k = {a.shape[1]}, need at least 2'
    assert ((a.sum(1) - 1).abs() <= ROW_SUM_ATOL).all(), f'{name}: rows do not sum to 1'


def _unit(name, a):
    """Assert the rows of a tensor or array have length 1 (to UNIT_ATOL)."""
    norms = torch.linalg.vector_norm(a, dim=1) if torch.is_tensor(a) else np.linalg.norm(a, axis=1)
    assert ((norms - 1) ** 2 <= UNIT_ATOL ** 2).all(), f'{name}: rows are not unit length'


def _np(a):
    """A numpy view of an array, tensor (any device) or list."""
    return a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)


def soft_assign(z, centres, k=None, chunk=SOFT_CHUNK):
    """Student-t soft assignments q [n, k] (alpha = 1), pc_sept's, on z's device.

    z is unit length. pc_sept's function broadcasts to [n, k, d] (17.7 GiB for a
    full pass of 42,969 x 27 x 2048), so without grad the rows go through it in
    chunks of `chunk` and are concatenated: the same rows, bit for bit. With
    grad (a training batch) the one call runs. `k`, if given, is asserted to be
    the centres' count.
    """
    _rows('z', z)
    _unit('z', z)
    _rows('centres', centres, d=z.shape[1])
    assert centres.shape[0] >= 2, 'k must be at least 2'
    assert k is None or centres.shape[0] == k, f'{centres.shape[0]} centres, k = {k}'
    if torch.is_grad_enabled():
        return pc_sept.soft_assignments(z, centres)
    return torch.cat([pc_sept.soft_assignments(z[i:i + chunk], centres)
                      for i in range(0, z.shape[0], chunk)])


def target(q_full):
    """Target P [n, k], detached, from q of a full pass over all fit paintings.

    The column frequency q.sum(0) is taken over every row passed, so the caller
    passes the whole fit set, never a batch.
    """
    _dist('q_full', q_full)
    return pc_sept.target_distribution(q_full)


def dec_kl(p, q):
    """KL(P || q), `F.kl_div(q.log(), p, 'batchmean')` as pc_sept's loop."""
    _dist('p', p)
    _dist('q', q)
    assert not p.requires_grad, 'the target carries no gradient'
    assert p.shape == q.shape
    return F.kl_div(q.log(), p, reduction='batchmean', log_target=False)


def pull_kl(r, q):
    """KL(r || q) in the same form; r is q0 (start pull) or t (teacher pull)."""
    return dec_kl(r, q)


def balance_kl(q):
    """PercepT's KL(uniform || mean q), summed over clusters, mean q clamped."""
    _dist('q', q)
    mean_q = q.mean(dim=0)
    uniform = torch.full_like(mean_q, 1.0 / mean_q.shape[0])
    return F.kl_div(mean_q.clamp_min(BALANCE_CLAMP).log(), uniform, reduction='sum')


def dec_block(q, p, r, settings):
    """lam * (KL(P||q) + [KL(r||q) if a pull] + [1000 * KL(u||mean q) if balance]).

    `settings` is the arm's dict (form, lam, balance). The pull is present for
    the forms `start` and `teacher` and then `r` is required; for `plain` `r`
    must be None. The balance mean is over the rows of `q` passed.
    Returns (total tensor, parts): the unweighted terms as floats, keys
    `kl_p`, and `pull` / `balance` when present.
    """
    form = settings['form']
    assert form in FORMS, f'form {form!r}'
    assert (r is not None) == (form != 'plain'), f'pull target r does not match form {form!r}'
    lam = float(settings['lam'])
    assert lam >= 0.0 and np.isfinite(lam)
    assert isinstance(settings['balance'], bool)
    assert q.shape[1] >= 2 and p.shape == q.shape and (r is None or r.shape == q.shape)
    kl_p = dec_kl(p, q)
    inner = kl_p
    parts = {'kl_p': kl_p.item()}
    if r is not None:
        pull = pull_kl(r, q)
        inner = inner + pull
        parts['pull'] = pull.item()
    if settings['balance']:
        bal = balance_kl(q)
        inner = inner + LAMBDA_BALANCE * bal
        parts['balance'] = bal.item()
    return lam * inner, parts


def kmeans_centres(z, k, seed):
    """PercepT's centre start: KMeans(k, n_init=10, random_state=seed) on z.

    Returns float32 centres [k, d] as a CPU torch tensor (move them to the
    device yourself).
    """
    _rows('z', z)
    assert z.shape[0] >= k
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(z.detach().cpu().numpy())
    return torch.from_numpy(km.cluster_centers_.astype(np.float32))


def teacher_clusters(x_unit, k):
    """D12: KMeans(k, n_init=10, random_state=0) on the unit instructed image.

    `x_unit` is a numpy array [n, d]. Returns (centres float32 [k, d],
    labels int64 [n]) as numpy.
    """
    x = np.asarray(x_unit)
    assert x.ndim == 2 and np.isfinite(x).all() and x.shape[0] >= k >= 2
    _unit('x_unit', x)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(x)
    return km.cluster_centers_.astype(np.float32), km.labels_.astype(np.int64)


def teacher_soft(x_unit, centres):
    """t [n, k]: Student-t (alpha = 1) soft assignments in the teacher's space.

    Returns a CPU torch tensor without grad, computed in chunks (see soft_assign).
    """
    x = torch.as_tensor(np.asarray(x_unit), dtype=torch.float32)
    c = torch.as_tensor(np.asarray(centres), dtype=torch.float32)
    with torch.no_grad():
        return soft_assign(x, c)


def cluster_means(z, labels, k):
    """Mean of z per teacher cluster, float32 [k, d] on z's device (the teacher form's start centres).

    `labels` is a numpy array or a tensor on any device.
    """
    _rows('z', z)
    labels = _np(labels).astype(np.int64)
    assert labels.shape == (z.shape[0],) and labels.min() >= 0 and labels.max() < k
    counts = np.bincount(labels, minlength=k)
    assert (counts > 0).all(), f'empty teacher cluster: {np.flatnonzero(counts == 0)}'
    idx = torch.from_numpy(labels).to(z.device)
    sums = torch.zeros(k, z.shape[1], dtype=torch.float64, device=z.device)
    sums.index_add_(0, idx, z.double())
    return (sums / torch.from_numpy(counts).to(z.device).double()[:, None]).float()


def changed_fraction(prev_argmax, cur_argmax):
    """Fraction of paintings whose cluster differs between two passes (S1's input)."""
    prev = _np(prev_argmax)
    cur = _np(cur_argmax)
    assert prev.shape == cur.shape and prev.ndim == 1 and prev.size > 0
    return float((prev != cur).mean())


def cluster_shares(argmax, k):
    """(largest share, smallest share) of the k clusters; the smallest is 0 if one is empty."""
    a = _np(argmax)
    assert a.ndim == 1 and a.size > 0 and a.min() >= 0 and a.max() < k
    shares = np.bincount(a, minlength=k) / a.size
    return float(shares.max()), float(shares.min())
