"""Independent stage-1c numerics; no round scoring imports. CPU is the default.

Bootstrap draw transcripts and fast hypergeometric EMI adapted from the permitted
r17/r23 independent C13 implementations; read-out selection from the r11 spec.
"""
import os
import time
from functools import lru_cache

ORIGINAL_CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
import r29rd_sc1 as rd
import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import gammaln
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

DEVICE = 'cpu'
TEXT_MENU = ('r0', 'center', 'center_k10')


def set_device(device):
    global DEVICE
    if device == 'cuda':
        if ORIGINAL_CUDA_VISIBLE_DEVICES is None:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ORIGINAL_CUDA_VISIBLE_DEVICES
        if not torch.cuda.is_available():
            raise RuntimeError('Controller requested cuda but CUDA is unavailable')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    DEVICE = device


def unit(x):
    return rd.unit(x)


def tower(x, state, side, batch_size=1024):
    weights = {k: v.to(DEVICE) for k, v in state.items() if k.startswith(side + '.')}
    out = []
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            z = torch.as_tensor(x[start:start + batch_size], device=DEVICE)
            z = F.gelu(F.linear(z, weights[f'{side}.trunk.0.weight'], weights[f'{side}.trunk.0.bias']))
            z = F.linear(z, weights[f'{side}.output.weight'], weights[f'{side}.output.bias'])
            out.append(F.normalize(z, dim=1).cpu().numpy())
    value = np.concatenate(out)
    assert np.isfinite(value).all()
    return value


def means(x):
    return x.astype(np.float64).mean(axis=0).astype(np.float32)


def order_scores(scores, own=None):
    if own is not None:
        scores[torch.arange(len(scores), device=scores.device), torch.as_tensor(own, device=scores.device)] = -torch.inf
    order = torch.argsort(scores, dim=1, descending=True, stable=True)
    if own is not None:
        own = torch.as_tensor(own, device=scores.device)
        order = order[order != own[:, None]].reshape(len(scores), -1)
    return order


def ap_torch(order, qlabels, glabels, cast32=False):
    q = torch.as_tensor(qlabels, device=order.device)
    g = torch.as_tensor(glabels, device=order.device)[order]
    rel = (g == q[:, None]) & (q[:, None] >= 0)
    rank = (g >= 0).cumsum(1).clamp_min(1)
    nrel = rel.sum(1)
    ap = ((rel.cumsum(1).double() / rank) * rel).sum(1) / nrel.clamp_min(1)
    ap[(q < 0) | (nrel == 0)] = float('nan')
    return (ap.float() if cast32 else ap).cpu().numpy()


def rank_ap(q, g, qlabels, glabels, queries=None, own=False, ks=(0,), batch_size=128, cast32=False):
    queries = np.arange(len(q)) if queries is None else np.asarray(queries)
    gt = torch.as_tensor(g, device=DEVICE)
    result = {k: [] for k in ks}
    with torch.inference_mode():
        for start in range(0, len(queries), batch_size):
            chosen = queries[start:start + batch_size]
            qt = torch.as_tensor(q[chosen], device=DEVICE)
            order = order_scores(qt @ gt.T, chosen if own else None)
            for k in ks:
                ranking = order
                if k:
                    expanded = F.normalize(qt + gt[order[:, :k]].sum(1), dim=1)
                    ranking = order_scores(expanded @ gt.T, chosen if own else None)
                result[k].append(ap_torch(ranking, qlabels[chosen], glabels, cast32))
    return {k: np.concatenate(v) for k, v in result.items()}


def text_ap(t, i, labels, direction, menu=TEXT_MENU, fitted=None, limit=0, batch_size=128):
    qs, gs = dict(t2i=('t', 'i'), i2t=('i', 't'), t2t=('t', 't'))[direction]
    vectors = dict(t=t, i=i)
    fitted = fitted or {side: means(x) for side, x in vectors.items()}
    queries = rd.selected_queries(len(vectors[qs]), limit)
    result = {}
    if 'r0' in menu:
        result['r0'] = rank_ap(vectors[qs], vectors[gs], labels[qs], labels[gs], queries,
                               qs == gs, (0,), batch_size, True)[0]
    centered = [r for r in menu if r != 'r0']
    if centered:
        q, g = unit(vectors[qs] - fitted[qs]), unit(vectors[gs] - fitted[gs])
        values = rank_ap(q, g, labels[qs], labels[gs], queries, qs == gs,
                         tuple(0 if r == 'center' else 10 for r in centered), batch_size, True)
        result.update({r: values[0 if r == 'center' else 10] for r in centered})
    return result, queries


def menu(width):
    sizes = sorted({min(width, d) for d in (64, 256, 1024)}, reverse=True)
    return [f'{w}+k{k}' for k in (0, 1, 3, 5, 10) for w in ['none', 'center', *[f'whiten{d}' for d in sizes]]]


def parse_readout(name):
    w, k = name.split('+k')
    return w, int(k)


def mapper(fit, w):
    if w == 'none':
        return None, None
    mu = means(fit)
    if w == 'center':
        return mu, None
    # Covariance and eigenvectors in float64; descending principal components.
    centered = fit.astype(np.float64) - fit.astype(np.float64).mean(0)
    cov = centered.T @ centered / (len(fit) - 1)
    from scipy.linalg import eigh
    size = min(int(w.removeprefix('whiten')), cov.shape[0])
    eigenvalues, eigenvectors = eigh(cov, subset_by_index=(cov.shape[0] - size, cov.shape[0] - 1), driver='evr')
    eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
    epsilon = 1e-6 * max(float(eigenvalues[0]), 1e-12)
    transform = (eigenvectors / np.sqrt(np.maximum(eigenvalues, 0) + epsilon)).astype(np.float32)
    return mu, transform


def apply_map(x, mapping):
    mu, transform = mapping
    if mu is None:
        return unit(x)
    z = x - mu
    return unit(z if transform is None else z @ transform)


def image_ap(val, x, labels, readout, limit=0, batch_size=128):
    w, k = parse_readout(readout)
    z = apply_map(unit(x), mapper(unit(val), w))
    labelled = np.flatnonzero(labels >= 0)
    z, y = z[labelled], labels[labelled]
    queries = rd.selected_queries(len(z), limit)
    ap = rank_ap(z, z, y, y, queries, True, (k,), batch_size, False)[k]
    return ap.astype(np.float64), labelled[queries]


def crossfit(x, labels, readouts=None, limit=0, batch_size=128):
    x = unit(x)
    readouts = menu(x.shape[1]) if readouts is None else readouts
    needed_w = list(dict.fromkeys(parse_readout(r)[0] for r in readouts))
    halves = {r: [] for r in readouts}
    for parity in (0, 1):
        fit, test = x[np.arange(len(x)) % 2 != parity], x[np.arange(len(x)) % 2 == parity]
        y = labels[np.arange(len(x)) % 2 == parity]
        for w in needed_w:
            z = apply_map(test, mapper(fit, w))
            valid = y >= 0
            z, yy = z[valid], y[valid]
            queries = rd.selected_queries(len(z), limit)
            ks = tuple(parse_readout(r)[1] for r in readouts if parse_readout(r)[0] == w)
            aps = rank_ap(z, z, yy, yy, queries, True, ks, batch_size)
            for k, ap in aps.items():
                halves[f'{w}+k{k}'].append(float(np.nanmean(ap.astype(np.float64))))
    return {r: float(np.mean(v)) for r, v in halves.items()}


def select_image(rows, ranges):
    base = rows['none+k0']
    assert all(np.isfinite(ranges[a]) and ranges[a] > 0 for a in rd.ASPECTS)
    scores = {r: float(np.mean([(v[a] - base[a]) / ranges[a] for a in rd.ASPECTS])) for r, v in rows.items()}
    kept = {r: all(v[a] >= base[a] - ranges[a] - 1e-12 for a in rd.ASPECTS) for r, v in rows.items()}
    best = max(scores[r] for r in rows if kept[r])
    tied = [r for r in rows if kept[r] and scores[r] >= best - .001 - 1e-12]
    def priority(r):
        w, k = parse_readout(r)
        return (r != 'none+k0', k > 0, 0 if w == 'none' else 1 if w == 'center' else 2,
                k, -int(w.removeprefix('whiten')) if w.startswith('whiten') else 0)
    return min(tied, key=priority), scores, kept


def partitions(x, k):
    x = np.asarray(x, dtype=np.float32)
    with threadpool_limits(8):
        return np.stack([MiniBatchKMeans(n_clusters=k, n_init=3, batch_size=4096,
                                         random_state=s).fit_predict(x) for s in range(3)]).astype(np.int32)


@lru_cache(maxsize=32)
def log_factorials(n):
    return gammaln(np.arange(n + 1, dtype=np.float64) + 1)


def fast_ami(C):
    C = C[C.sum(1) > 0][:, C.sum(0) > 0]
    n = int(C.sum())
    if not n or C.shape == (1, 1):
        return 1.
    a, b = C.sum(1), C.sum(0)
    i, j = np.nonzero(C)
    values = C[i, j].astype(np.float64)
    mi = max(0., float(np.sum(values / n * (np.log(values) + np.log(n) - np.log(a[i]) - np.log(b[j])))))
    ha, hb = -np.sum(a / n * np.log(a / n)), -np.sum(b / n * np.log(b / n))
    au, ac = np.unique(a, return_counts=True)
    bu, bc = np.unique(b, return_counts=True)
    g, emi = log_factorials(n), 0.
    for ai, multiplicity in zip(au, ac):
        ks = np.arange(1, min(int(ai), int(bu.max())) + 1, dtype=np.int64)[None, :]
        bj = bu[:, None]
        valid = (ks <= bj) & (ks >= np.maximum(1, ai + bj - n))
        lp = (g[ai] + g[n-ai] + g[bj] + g[n-bj] - g[n] - g[ks] - g[ai-ks]
              - g[np.maximum(bj-ks, 0)] - g[np.maximum(n-ai-bj+ks, 0)])
        term = np.exp(np.where(valid, lp, -np.inf)) * (ks / n) * (np.log(n) + np.log(ks) - np.log(ai) - np.log(bj))
        emi += multiplicity * float(np.sum(term * bc[:, None]))
    den = (ha + hb) / 2 - emi
    eps = np.finfo(np.float64).eps
    den = min(den, -eps) if den < 0 else max(den, eps)
    return float((mi - emi) / den)


def part_value(parts, y, weights=None, seeds=None):
    parts = np.asarray(parts)
    p = parts[None] if parts.ndim == 2 else parts
    seeds = np.arange(len(p)) if seeds is None else seeds
    good = y >= 0
    yy = y[good]
    w = np.ones(good.sum(), dtype=np.int64) if weights is None else weights[good]
    ny = int(yy.max()) + 1
    out = {}
    for seed in np.unique(seeds):
        vals = []
        for partition in p[seed]:
            pp = partition[good]
            nk = int(pp.max()) + 1
            C = np.bincount(yy * nk + pp, weights=w, minlength=ny*nk).reshape(ny, nk).astype(np.int64)
            vals.append(fast_ami(C))
        out[seed] = float(np.mean(vals))
    return float(np.mean([out[s] for s in seeds]))


_BOOT_CONTEXT = None

def _ami_worker(task):
    candidate, comparator, y = _BOOT_CONTEXT
    rows, seeds = task
    w = np.bincount(rows, minlength=len(y))
    return part_value(candidate, y, w, seeds) - part_value(comparator, y, w)


def _one_thread(candidate, comparator, y):
    global _BOOT_CONTEXT
    _BOOT_CONTEXT = candidate, comparator, y
    torch.set_num_threads(1)
    threadpool_limits(1)


def ami_interval(candidate, comparator, y, draws=2000, workers=8):
    import multiprocessing as mp
    global _BOOT_CONTEXT
    cv, bv = part_value(candidate, y), part_value(comparator, y)
    rng = np.random.default_rng(0)
    tasks = [(rng.integers(0, len(y), size=len(y)), rng.choice(5, size=5, replace=True)) for _ in range(draws)]
    _BOOT_CONTEXT = candidate, comparator, y
    if workers > 1:
        # Never fork after CUDA or multithreaded Torch work in a full pipeline.
        context = 'spawn'
        with mp.get_context(context).Pool(workers, initializer=_one_thread,
                                        initargs=(candidate, comparator, y)) as pool:
            boot = np.asarray(pool.map(_ami_worker, tasks, chunksize=4))
    else:
        boot = np.asarray([_ami_worker(task) for task in tasks])
    _BOOT_CONTEXT = None
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(value=cv, comparator_value=bv, d=cv-bv, lo=float(lo), hi=float(hi))


def retrieval_interval(candidate, comparator, pid=None, draws=10000):
    candidate, comparator = np.asarray(candidate, dtype=np.float64), np.asarray(comparator, dtype=np.float64)
    assert candidate.shape == (5, len(comparator))
    mask = np.isfinite(comparator)
    assert np.array_equal(np.isfinite(candidate), np.broadcast_to(mask, candidate.shape))
    c, b = candidate[:, mask], comparator[mask]
    if pid is None:
        units = np.arange(mask.sum())
    else:
        # Sorted painting positions are the evaluation order.
        _, units = np.unique(np.asarray(pid)[mask], return_inverse=True)
    n = int(units.max()) + 1
    counts = np.bincount(units, minlength=n).astype(np.float64)
    sums = np.stack([np.bincount(units, weights=row, minlength=n) for row in c])
    baseline = np.bincount(units, weights=b, minlength=n)
    rng, boot = np.random.default_rng(0), np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 128):
        size = min(128, draws-start)
        weights = np.empty((size, n), dtype=np.float64)
        selected = np.empty((size, 5), dtype=np.int64)
        for j in range(size):
            weights[j] = np.bincount(rng.integers(0, n, size=n), minlength=n)
            selected[j] = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
        den = weights @ counts
        cm = weights @ sums.T
        bm = weights @ baseline
        boot[start:start+size] = np.take_along_axis(cm, selected, axis=1).mean(1) / den - bm / den
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(value=float(c.mean()), comparator_value=float(b.mean()), d=float(c.mean()-b.mean()),
                lo=float(lo), hi=float(hi), n_queries=int(mask.sum()), n_units=n)


def outcome(lo, r):
    return 'WIN' if lo > 0 else 'LEVEL' if lo > -r else 'BELOW'


def ladder(cells):
    def level(p, q):
        return 3 if p == 15 and q >= 12 else 2 if p == 15 else 1 if p >= 12 else 0
    p = sum(c['id'].endswith('/p1') and c['outcome'] == 'WIN' for c in cells)
    q = sum(c['id'].endswith('/q1') and c['outcome'] in ('WIN', 'LEVEL') for c in cells)
    unknown_p = sum(c['id'].endswith('/p1') and c['outcome'] == 'INCONCLUSIVE' for c in cells)
    unknown_q = sum(c['id'].endswith('/q1') and c['outcome'] == 'INCONCLUSIVE' for c in cells)
    result = dict(wins_p1=p, wins_or_level_q1=q, level=level(p, q))
    if unknown_p or unknown_q:
        result.update(inconclusive_above=level(p+unknown_p, q+unknown_q) > result['level'])
    return result


def crossfit_multi(x, labels, readouts=None, limit=0, batch_size=128):
    """Share label-free fits across aspects, especially the 6144-wide untold row."""
    x = unit(x)
    readouts = menu(x.shape[1]) if readouts is None else readouts
    halves = {a: {r: [] for r in readouts} for a in labels}
    needed_w = list(dict.fromkeys(parse_readout(r)[0] for r in readouts))
    for parity in (0, 1):
        test_mask = np.arange(len(x)) % 2 == parity
        fit, test = x[~test_mask], x[test_mask]
        spectral = None
        for w in needed_w:
            if w.startswith('whiten'):
                if spectral is None:
                    maximum = max(int(z.removeprefix('whiten')) for z in needed_w if z.startswith('whiten'))
                    spectral = mapper(fit, f'whiten{maximum}')
                mapping = spectral[0], spectral[1][:, :int(w.removeprefix('whiten'))]
            else:
                mapping = mapper(fit, w)
            mapped = apply_map(test, mapping)
            ks = tuple(parse_readout(r)[1] for r in readouts if parse_readout(r)[0] == w)
            for aspect, original_y in labels.items():
                y = original_y[test_mask]
                valid = y >= 0
                z, y = mapped[valid], y[valid]
                queries = rd.selected_queries(len(z), limit)
                aps = rank_ap(z, z, y, y, queries, True, ks, batch_size)
                for k, ap in aps.items():
                    halves[aspect][f'{w}+k{k}'].append(float(np.nanmean(ap.astype(np.float64))))
    return {a: {r: float(np.mean(v)) for r, v in rows.items()} for a, rows in halves.items()}
