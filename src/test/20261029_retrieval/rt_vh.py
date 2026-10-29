"""rt_vh (r11 ticket 08, constitution C13): the retrieval round's numbers re-derived from the rule's text alone.

Written from the spec's rule (docs/superpowers/specs/2026-10-29-retrieval-design.md: 3.4, 3.5, 6) and the data, without
any rt_ module (this file imports none). The initial implementation was written without reading the main code;
the C13 fix follows the written Numeric convention in 20261029_retrieval_log.md. It recomputes, from the stores:

  stage0a   c*'s validation mAP per seed and aspect under plain cosine, the ranges r_a, the emotion nDCG range
  stage1    SC1-SC6: every comparison (d, interval, outcome, r), each SC's pass, the GO
  stage2    SC7-SC9 (and c*'s switch cells): the same, the switch GO
  stage3    emotion nDCG, c* against the plain and instructed images, the reading (holds / qualified)
  stage4    every arm's cross-fit score per read-out, the guard, the pick
  stage5    SC10-SC15 with stage 4's picks (read from <outputs>/stage4/picks.json): the same as stage 1

and compares them with the run's file under --outputs, values within 1e-6, same outcomes and picks. Each mode
writes rederive_<stage>.json beside the checked file and exits 1 on any disagreement (exit 2: input missing).

  python rt_vh.py stage1 [--outputs DIR] [--inputs DIR] [--smoke N] [--workers 8]

Numeric convention: see 20261029_retrieval_log.md. Input rows normalize once in float32; CRL cosines and their
normalization use float64 before one float32 cast. Every W (including none) and its normalization use float64,
then one float32 cast. QE and similarities use float64. Ranking: stable descending sort, ties keep gallery row
order. Frozen r10 pieces used read-only: rs_vh's three label functions (and its ANNOTATIONS and GENRE_CSVS), and
ct2_data.split2('full') for the sorted validation names.
"""
import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

# ---- constants and paths (this file's own) --------------------------------------------------------------------
MAIN = Path('/project/CoSiR')                                   # frozen, gitignored inputs live only here
RS_DIR = MAIN / 'src/test/20261027_recipe_sweep'                # r10's round, imported read-only
OUTPUTS_DEFAULT = Path('/data/SSD2/cosir_r11/20261029_retrieval/outputs')
HELD_LIST = MAIN / 'src/test/20261020_claim_test/outputs/held/held_paintings.json'
PERCEPT_HELD = str(MAIN / 'src/test/20261020_claim_test/outputs/held/percept/core__k40/P3__s{seed}.npz')          # 'prob'
PERCEPT_VAL = 'percept_val/P3__s{seed}.npz'      # under --inputs; 'prob'; spec amendment A1: the held mapper's weights
R10_OUT = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full')
CSTAR_HELD = str(R10_OUT / 'held/k11=40/{aspect}__s{seed}.npz')                                                # 'img'
CSTAR_VAL = str(R10_OUT / 'train/k11=40/R6-5/{file}__s{seed}.npz')                                             # 'img'
CSTAR_VAL_FILE = {'emotion': 'emotion__Gres', 'style': 'style__Fres05', 'genre': 'genre__R0o'}
STORE = Path('/data/SSD2/pre_extract/artelingo_full')
PLAIN_STORES = {'plain': 'qwen3vl_emb_2b', 'clip': 'clip_l14', 'siglip2': 'siglip2_so400m'}    # index.json images[i].painting
PROMPTS = STORE / 'qwen3vl_emb_2b_aspect_prompts'
INSTRUCTED = {'emotion': 'emotion_focus', 'style': 'style_word', 'genre': 'genre_word'}
INSTRUCTED_SPLIT = {'held': '{}_held', 'val': '{}_s2img'}                                          # ids.npz painting_names

ASPECTS = ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2, 3, 4)
ARMS = ('cstar', 'plain', 'instructed', 'clip', 'siglip2', 'percept', 'crl')
SEEDED = {'cstar', 'percept'}
HEADED = {'cstar', 'instructed', 'crl'}                         # the arm has one embedding per aspect name (the head)
N = {'held': 12281, 'val': 6152}
N_GENRE = {'held': 9949, 'val': 4960}
DRAWS = 10000
KS = (0, 1, 3, 5, 10)
WHITEN_SIZES = (64, 256, 1024)
TIE = 0.001
TOL = 1e-6
ROW = 256                                                       # query rows per block (mAP)
ROW_G = 64                                                      # query rows per block (nDCG, a 9-wide gain)
CHECK_HELD_STAGES = ('stage1', 'stage2', 'stage3', 'stage5')
STAGE_DIR = {'stage0a': 'stage0a', 'stage1': 'stage1', 'stage2': 'stage2', 'stage3': 'stage3', 'stage4': 'stage4',
             'stage5': 'stage5'}
RUN_FILE = {'stage0a': 'ranges.json', 'stage1': 'verdict.json', 'stage2': 'verdict.json', 'stage3': 'verdict.json',
            'stage4': 'scores.json', 'stage5': 'verdict.json'}


class Missing(Exception):
    """A run file or an input of the run is not there (exit 2, not a disagreement)."""


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


# ---- multiprocessing: fork, one BLAS thread per worker ---------------------------------------------------------
_ST = {}


def _init_worker():
    threadpool_limits(limits=1)


def pmap(fn, tasks, workers):
    """Ordered map of fn over tasks in forked workers; fn reads the module-level _ST set before the fork."""
    tasks = list(tasks)
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    with mp.get_context('fork').Pool(min(workers, len(tasks)), initializer=_init_worker) as pool:
        return pool.map(fn, tasks, chunksize=1)


# ---- ranking, AP and nDCG (rule 3.4) ---------------------------------------------------------------------------

def _order(G, Qv, qidx):
    """Gallery order for each query row of Qv (the gallery G holds the query at row qidx): cosine descending, stable
    (ties keep gallery row order), the query itself removed. G, Qv float64 with unit rows. int64 [B, m-1]."""
    S = Qv @ G.T
    S[np.arange(len(qidx)), qidx] = -np.inf
    order = np.argsort(-S, axis=1, kind='stable')
    assert (order[:, -1] == qidx).all(), 'the query must sort last (non-finite similarity?)'
    return order[:, :-1]


def _ap_rel(rel):
    """AP of each row of a 0/1 relevance matrix over the full ranking; nan when the row has no relevant item."""
    cum = np.cumsum(rel, axis=1)
    nrel = cum[:, -1]
    ranks = np.arange(1, rel.shape[1] + 1, dtype=np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        ap = (rel * cum / ranks).sum(axis=1) / nrel
    ap[nrel == 0] = np.nan
    return ap


def ap_core(G, lab, qidx, ks=(0,)):
    """AP per query (rows qidx of the gallery G) for each query-expansion size k. G float64 unit rows, lab int labels
    (all >= 0). k = 0: plain cosine. k > 0: the query is replaced by the normalized mean of itself and its top-k
    gallery neighbours (the query excluded), then the gallery is ranked again (query excluded). dict k -> float64."""
    out = {k: [] for k in ks}
    for i in range(0, len(qidx), ROW):
        qi = np.asarray(qidx[i:i + ROW])
        Q = G[qi]
        order = _order(G, Q, qi)
        for k in ks:
            if k == 0:
                o = order
            else:
                v = Q + G[order[:, :k]].sum(axis=1)
                norm = np.linalg.norm(v, axis=1, keepdims=True)
                assert (norm > 0).all()
                o = _order(G, v / norm, qi)
            rel = (lab[o] == lab[qi][:, None]).astype(np.float64)
            out[k].append(_ap_rel(rel))
    return {k: np.concatenate(v) for k, v in out.items()}


def ndcg_core(G, P, qidx):
    """nDCG per query (rows qidx; every row of G is a gallery painting, the query excluded). Gain of x for q:
    sum_c min(P_q,c, P_x,c); DCG = sum gain_rank / log2(rank + 1) over the full ranking; nDCG = DCG / the DCG of the
    gallery sorted by gain. P float64 [m, 9]."""
    m = len(G)
    disc = 1.0 / np.log2(np.arange(1, m, dtype=np.float64) + 1)
    out = []
    for i in range(0, len(qidx), ROW_G):
        qi = np.asarray(qidx[i:i + ROW_G])
        order = _order(G, G[qi], qi)
        gain = np.minimum(P[qi][:, None, :], P[None, :, :]).sum(axis=-1)
        mask = np.ones(gain.shape, dtype=bool)
        mask[np.arange(len(qi)), qi] = False
        ideal = -np.sort(-gain[mask].reshape(len(qi), m - 1), axis=1)
        dcg = np.take_along_axis(gain, order, axis=1) @ disc
        idcg = ideal @ disc
        assert (idcg > 0).all(), 'a query with no overlapping gallery painting'
        out.append(dcg / idcg)
    return np.concatenate(out)


def _blk_ap(qidx):
    return ap_core(_ST['G'], _ST['l'], qidx, _ST['ks'])


def _blk_ndcg(qidx):
    return ndcg_core(_ST['G'], _ST['P'], qidx)


def ap_all(X32, lab, workers, ks=(0,)):
    """AP per query over the rows with lab >= 0 (queries in row order, gallery the same rows); nan = dropped
    (no relevant gallery painting). Returns dict k -> float64 [n_labelled]."""
    sel = np.flatnonzero(lab >= 0)
    G = np.ascontiguousarray(X32[sel]).astype(np.float64)
    assert np.isfinite(G).all()
    _ST.update(G=G, l=lab[sel], ks=tuple(ks))
    step = max(ROW, -(-len(sel) // (workers * 4) // ROW) * ROW)
    res = pmap(_blk_ap, [np.arange(i, min(i + step, len(sel))) for i in range(0, len(sel), step)], workers)
    return {k: np.concatenate([r[k] for r in res]) for k in ks}


def ndcg_all(X32, P, workers):
    G = np.ascontiguousarray(X32).astype(np.float64)
    assert np.isfinite(G).all()
    _ST.update(G=G, P=P)
    step = max(ROW_G, -(-len(G) // (workers * 4) // ROW_G) * ROW_G)
    res = pmap(_blk_ndcg, [np.arange(i, min(i + step, len(G))) for i in range(0, len(G), step)], workers)
    return np.concatenate(res)


# ---- read-outs (rule 6, selection rule) ------------------------------------------------------------------------
# A read-out is (w, size, k): w in none / center / whiten; size = whitening dimensions (0 otherwise); k = expansion.
R0 = ('none', 0, 0)


def ro_name(ro):
    w, size, k = ro
    return f'{w}{size or ""}+k{k}'


def ro_parse(name, width):
    m = re.fullmatch(r'(none|center|centering|whiten)(\d*)\+k(\d+)', name)
    assert m, f'unreadable read-out name {name!r}'
    w = 'center' if m.group(1) == 'centering' else m.group(1)
    size = 0
    if w == 'whiten':
        size = min(int(m.group(2)), width) if m.group(2) else width
    else:
        assert not m.group(2), name
    return (w, size, int(m.group(3)))


def menu(width):
    """The arm's read-outs: W in none, center, whiten 64/256/1024 (a size at or above the width becomes the full
    width, repeats dropped) crossed with k in 0, 1, 3, 5, 10."""
    ws = [('none', 0), ('center', 0)]
    for s in WHITEN_SIZES:
        w = ('whiten', min(s, width))
        if w not in ws:
            ws.append(w)
    return [(w, s, k) for (w, s) in ws for k in KS]


def tie_key(ro):
    """Lexicographic tie order: R0 first, no expansion before expansion, none < center < whiten, smaller k, larger
    whitening size."""
    w, size, k = ro
    return (ro != R0, k > 0, ('none', 'center', 'whiten').index(w), k, -size)


def l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def fit_stats(Xfit32):
    """Mean and covariance eigen-decomposition of the fit set (every painting, no labels), float64.
    Returns mu, eigenvalues descending, eigenvectors as columns (descending)."""
    X = Xfit32.astype(np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    lam, U = np.linalg.eigh(Xc.T @ Xc / (len(X) - 1))
    return mu, lam[::-1], U[:, ::-1]


def apply_w(w, size, stats, X32):
    """The read-out's W stage followed by L2 normalization; float32 [n, d'] (rows normalized)."""
    if w == 'none':
        return l2(X32.astype(np.float64)).astype(np.float32)
    mu, lam, U = stats
    Xc = X32.astype(np.float64) - mu
    if w == 'center':
        return l2(Xc).astype(np.float32)
    eps = 1e-6 * lam[0]
    z = (Xc @ U[:, :size]) / np.sqrt(lam[:size] + eps)
    return l2(z).astype(np.float32)


# ---- statistics (rule 3.4) --------------------------------------------------------------------------------------

def paired(cand, comp, draws):
    """cand, comp: float64 [nq] (unseeded) or [5, nq] (seeded, rows = seeds 0..4). Returns d (all queries, all
    seeds) and the 95% bootstrap interval; rng = default_rng(0); per draw I_b = rng.integers(0, n, size=n), then
    (when either side is seeded) S_b = rng.choice([0..4], size=5, replace=True), shared; d_b = mean over I_b (and
    S_b) of cand - comp."""
    seeded = cand.ndim == 2 or comp.ndim == 2
    c = cand if cand.ndim == 2 else cand[None, :]
    k = comp if comp.ndim == 2 else comp[None, :]
    diff = c - k                                                  # [5, n] or [1, n]; broadcasts when one side is seeded
    if seeded:
        diff = np.broadcast_to(diff, (5, diff.shape[1]))
    n = diff.shape[1]
    rng = np.random.default_rng(0)
    d = np.empty(draws)
    for b in range(draws):
        idx = rng.integers(0, n, size=n)
        if seeded:
            s = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
            w = np.bincount(s, minlength=5) / 5.0
            d[b] = (w @ diff)[idx].mean()
        else:
            d[b] = diff[0][idx].mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(diff.mean()), [float(lo), float(hi)]


def outcome(lower, r):
    if lower > 0:
        return 'WIN'
    return 'LEVEL' if lower > -r else 'BELOW'


# ---- data -------------------------------------------------------------------------------------------------------

class Data:
    """Names, labels and embeddings of the held list and the sorted validation names (first `smoke` rows of each
    when smoke is set). Embeddings come back float32 with rows normalized (norm clipped at 1e-12)."""

    def __init__(self, inputs, smoke=0):
        self.inputs = Path(inputs)                                # derived inputs: percept_val/, crl/
        self.smoke = smoke
        for p in (str(RS_DIR),):
            if p not in sys.path:
                sys.path.insert(0, p)
        import rs_vh                                              # r10's frozen label functions (read-only)
        import ct2_data                                           # r10's frozen splits
        self.rs = rs_vh
        held = json.loads(HELD_LIST.read_text())['names']
        assert len(held) == N['held'] and held == sorted(held)
        s = ct2_data.split2('full')
        val = [str(x) for x in s['all_painting_ids'][s['dev']]]
        assert len(val) == N['val'] and val == sorted(val)
        self.names = {'held': held, 'val': val}
        if smoke:
            self.names = {k: v[:smoke] for k, v in self.names.items()}
        rows = json.loads(rs_vh.ANNOTATIONS.read_text())
        self.labels, self.dist = {}, {}
        codes = {e: i for i, e in enumerate(sorted(rs_vh.EMOTIONS))}
        for sp, nm in self.names.items():
            self.labels[sp] = dict(emotion=rs_vh.emotion_labels(nm, rows), style=rs_vh.style_labels(nm, rows),
                                   genre=rs_vh.genre_labels(nm, rows, rs_vh.GENRE_CSVS))
            counts = {n: np.zeros(len(codes)) for n in nm}
            for r in rows:
                if r['painting'] in counts:
                    counts[r['painting']][codes[r['emotion']]] += 1
            self.dist[sp] = np.stack([counts[n] / counts[n].sum() for n in nm])     # caption-emotion distributions
            if not smoke:
                assert int((self.labels[sp]['genre'] >= 0).sum()) == N_GENRE[sp]
        self._cache = {}
        self._index = {}

    # -- stores --
    def _store_rows(self, store, sp):
        if store not in self._index:
            imgs = json.loads((STORE / store / 'index.json').read_text())['images']
            self._index[store] = {r['painting']: i for i, r in enumerate(imgs)}
        idx = self._index[store]
        return np.asarray([idx[n] for n in self.names[sp]], dtype=np.int64)

    def _prompt_rows(self, folder, sp):
        with np.load(PROMPTS / folder / 'ids.npz', allow_pickle=False) as z:
            ids = z['painting_names'].tolist()
        idx = {n: i for i, n in enumerate(ids)}
        assert len(idx) == len(ids)
        return np.asarray([idx[n] for n in self.names[sp]], dtype=np.int64)

    def crl_text(self, aspect):
        path = self.inputs / 'crl' / f'{aspect}_text.npy'
        if not path.exists():
            raise Missing(f'CRL text embeddings not found: {path}')
        t = np.load(path)
        assert t.dtype == np.float32 and t.ndim == 2 and t.shape[1] == 768, (path, t.shape, t.dtype)
        assert np.abs(np.linalg.norm(t, axis=1) - 1).max() < 1e-4, 'CRL text rows are not normalized'
        return t

    def emb(self, arm, sp, head=None, seed=None):
        """float32 [n, d] normalized rows. head = the aspect name for headed arms (cstar, instructed, crl)."""
        key = (arm, sp, head, seed)
        if key in self._cache:
            return self._cache[key]
        assert (head is not None) == (arm in HEADED), key
        assert (seed is not None) == (arm in SEEDED), key
        n = len(self.names[sp])
        if arm == 'cstar':
            path = (CSTAR_HELD.format(aspect=head, seed=seed) if sp == 'held'
                    else CSTAR_VAL.format(file=CSTAR_VAL_FILE[head], seed=seed))
            with np.load(path, allow_pickle=False) as z:
                x = z['img'][:n]
            assert x.shape == (n, 2048) and x.dtype == np.float16, (path, x.shape, x.dtype)
        elif arm in PLAIN_STORES:
            store = PLAIN_STORES[arm]
            img = np.load(STORE / store / 'img.npy', mmap_mode='r')
            x = np.asarray(img[self._store_rows(store, sp)])
            assert x.dtype == np.float16 and x.shape[0] == n, (store, x.shape, x.dtype)
        elif arm == 'instructed':
            folder = INSTRUCTED_SPLIT[sp].format(INSTRUCTED[head])
            img = np.load(PROMPTS / folder / 'img.npy', mmap_mode='r')
            x = np.asarray(img[self._prompt_rows(folder, sp)])
            assert x.dtype == np.float16 and x.shape == (n, 2048), (folder, x.shape, x.dtype)
        elif arm == 'percept':
            path = PERCEPT_HELD.format(seed=seed) if sp == 'held' else str(self.inputs / PERCEPT_VAL.format(seed=seed))
            if not Path(path).exists():
                raise Missing(f'PercepT outputs not found: {path}')
            with np.load(path, allow_pickle=False) as z:
                x = z['prob'][:n]
            assert x.shape == (n, 40) and x.dtype == np.float32, (path, x.shape, x.dtype)
        elif arm == 'crl':
            clip = self.emb('clip', sp)                           # normalized CLIP L/14 image rows
            v = clip.astype(np.float64) @ self.crl_text(head).astype(np.float64).T
            x = l2(v).astype(np.float32)                          # one cast, after cosine-vector normalization
            assert np.isfinite(x).all()
            self._cache[key] = x
            return x
        else:
            raise AssertionError(arm)
        x = x.astype(np.float32)
        x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        assert np.isfinite(x).all()
        self._cache[key] = x
        return x

    def width(self, arm, head=None):
        return self.emb(arm, 'val', head, 0 if arm in SEEDED else None).shape[1]


def arm_heads(arm):
    return ASPECTS if arm in HEADED else (None,)


def arm_seeds(arm):
    return SEEDS if arm in SEEDED else (None,)


def describe_loading(data, arms):
    for arm in arms:
        for sp in ('held', 'val'):
            try:
                x = data.emb(arm, sp, ASPECTS[0] if arm in HEADED else None, 0 if arm in SEEDED else None)
                log(f'  loaded {arm:10s} {sp:5s} {x.shape} {x.dtype} norm[{np.linalg.norm(x, axis=1).min():.4f},'
                    f'{np.linalg.norm(x, axis=1).max():.4f}]')
            except Missing as e:
                log(f'  {arm} {sp}: {e}')


# ---- checking ---------------------------------------------------------------------------------------------------

class Check:
    def __init__(self, tol=TOL):
        self.tol = tol
        self.items = []
        self.bad = []

    def num(self, name, mine, theirs):
        ok = (mine is not None and theirs is not None and np.isfinite(mine) and np.isfinite(theirs)
              and abs(float(mine) - float(theirs)) <= self.tol)
        self._add(name, mine, theirs, ok)

    def eq(self, name, mine, theirs):
        self._add(name, mine, theirs, mine == theirs)

    def _add(self, name, mine, theirs, ok):
        self.items.append(dict(name=name, mine=mine, theirs=theirs, ok=bool(ok)))
        if not ok:
            self.bad.append(self.items[-1])

    def problem(self, name, text):
        self._add(name, text, None, False)


def jnum(x):
    if isinstance(x, dict):
        return {str(k): jnum(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jnum(v) for v in x]
    if isinstance(x, np.ndarray):
        return jnum(x.tolist())
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def load_run_file(outputs, stage, fname=None):
    path = Path(outputs) / STAGE_DIR[stage] / (fname or RUN_FILE[stage])
    if not path.exists():
        raise Missing(f'run file not found: {path}')
    return path, json.loads(path.read_text())


def finish(outputs, stage, check, mine, args, extra=None):
    out = Path(outputs) / STAGE_DIR[stage] / f'rederive_{stage}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(stage=stage, smoke=args.smoke, draws=args.draws, tol=check.tol, agree=not check.bad,
               n_checked=len(check.items), disagreements=check.bad, mine=jnum(mine))
    if extra:
        rec.update(jnum(extra))
    out.write_text(json.dumps(jnum(rec), indent=1))
    log(f'{stage}: {len(check.items)} values compared, {len(check.bad)} disagreements; wrote {out}')
    for b in check.bad[:20]:
        log('  DISAGREE', b)
    return 1 if check.bad else 0


# ---- the run: per-query values, ranges --------------------------------------------------------------------------

class Run:
    def __init__(self, data, args):
        self.d, self.args, self.workers = data, args, args.workers
        self.pq = {}
        self._ranges = None
        self.dropped = {}

    def dropped_text(self):
        return {f'{sp}.{a}': n for (sp, a), n in self.dropped.items()}

    def perq(self, sp, arm, head, lab_aspect, ro=None, picks_fit=None):
        """AP per query (non-dropped, row order) of the arm's embeddings scored with lab_aspect labels: float64 [nq]
        (unseeded) or [5, nq]. ro: a read-out (w, size, k) fit on all validation paintings (stage 5)."""
        key = (sp, arm, head, lab_aspect, ro)
        if key in self.pq:
            return self.pq[key]
        lab = self.d.labels[sp][lab_aspect]
        rows = []
        for seed in arm_seeds(arm):
            x = self.d.emb(arm, sp, head, seed)
            k = 0
            if ro is not None:
                w, size, k = ro
                stats = None
                if w != 'none':
                    xv = self.d.emb(arm, 'val', head, seed)
                    stats = fit_stats(xv) if w == 'whiten' else (xv.astype(np.float64).mean(axis=0), None, None)
                x = apply_w(w, size, stats, x)
            ap = ap_all(x, lab, self.workers, ks=(k,))[k]
            rows.append(ap)
        ap = np.stack(rows)
        keep = ~np.isnan(ap[0])
        assert (keep == ~np.isnan(ap)).all(), 'dropped queries differ between seeds'
        self.dropped[(sp, lab_aspect)] = int((~keep).sum())
        out = ap[:, keep] if arm in SEEDED else ap[0][keep]
        self.pq[key] = out
        return out

    def perq_ndcg(self, sp, arm, head):
        key = ('ndcg', sp, arm, head)
        if key in self.pq:
            return self.pq[key]
        rows = [ndcg_all(self.d.emb(arm, sp, head, seed), self.d.dist[sp], self.workers) for seed in arm_seeds(arm)]
        out = np.stack(rows) if arm in SEEDED else rows[0]
        self.pq[key] = out
        return out

    def ranges(self):
        """r_a: max - min over seeds of c*'s validation mAP (R0); emotion nDCG likewise."""
        if self._ranges is None:
            res = dict(map={}, ndcg={})
            for a in ASPECTS:
                per = self.perq('val', 'cstar', a, a).mean(axis=1)
                res['map'][a] = dict(per_seed=per.tolist(), r=float(per.max() - per.min()))
            per = self.perq_ndcg('val', 'cstar', 'emotion').mean(axis=1)
            res['ndcg']['emotion'] = dict(per_seed=per.tolist(), r=float(per.max() - per.min()))
            self._ranges = res
        return self._ranges

    def r(self, a, metric='map'):
        return self.ranges()[metric][a]['r']


# ---- comparisons and SCs ---------------------------------------------------------------------------------------

def compare(run, sp, cand, comp, aspect, metric='map', ro_c=None, ro_k=None):
    """cand, comp = (arm, head). Returns the comparison record (d, interval, r, outcome)."""
    if metric == 'map':
        c = run.perq(sp, cand[0], cand[1], aspect, ro_c)
        k = run.perq(sp, comp[0], comp[1], aspect, ro_k)
    else:
        c = run.perq_ndcg(sp, cand[0], cand[1])
        k = run.perq_ndcg(sp, comp[0], comp[1])
    d, iv = paired(c, k, run.args.draws)
    r = run.r(aspect, metric)
    return dict(candidate=list(cand), comparator=list(comp), aspect=aspect, d=d, interval=iv, r=r,
                outcome=outcome(iv[0], r), n_queries=int(c.shape[-1]))


def ckey(c):
    return (tuple(c['candidate']), tuple(c['comparator']), c['aspect'])


def check_sc(chk, sid, mine, theirs):
    """mine: {'comparisons': [...], 'pass': bool}; theirs: the file's entry for the same SC id."""
    if theirs is None:
        chk.problem(f'{sid}: missing in file', None)
        return
    by = {}
    for c in theirs['comparisons']:
        by[(tuple(c['candidate']), tuple(c['comparator']), c['aspect'])] = c
    chk.eq(f'{sid}.n_comparisons', len(mine['comparisons']), len(theirs['comparisons']))
    for c in mine['comparisons']:
        nm = f'{sid}.{c["candidate"]}>{c["comparator"]}.{c["aspect"]}'
        t = by.get(ckey(c))
        if t is None:
            chk.problem(nm + ': comparison not in file', None)
            continue
        chk.num(nm + '.d', c['d'], t['d'])
        chk.num(nm + '.lo', c['interval'][0], t['interval'][0])
        chk.num(nm + '.hi', c['interval'][1], t['interval'][1])
        chk.num(nm + '.r', c['r'], t['r'])
        chk.eq(nm + '.outcome', c['outcome'], t['outcome'])
    chk.eq(f'{sid}.pass', mine['pass'], bool(theirs['pass']))


def as_go(v):
    if isinstance(v, str):
        s = v.strip().upper().replace('_', '-')
        assert s in ('GO', 'NO-GO', 'TRUE', 'FALSE'), v
        return s in ('GO', 'TRUE')
    return bool(v)


def passes(kind, comps):
    if kind == 'win':
        return all(c['outcome'] == 'WIN' for c in comps)
    return all(c['outcome'] in ('WIN', 'LEVEL') for c in comps)


def table_sc(run, first_id, picks=None):
    """SCs of a main table: first_id, first_id+1, first_id+2 against plain (WIN), the next three against
    instructed (LEVEL or WIN). picks: {arm: read-out} (stage 5) or None (R0)."""
    ro = (lambda arm: picks[arm]) if picks else (lambda arm: None)
    sc = {}
    for j, (opp, kind) in enumerate((('plain', 'win'), ('instructed', 'level'))):
        for i, a in enumerate(ASPECTS):
            cand = ('cstar', a)
            comp = ('plain', None) if opp == 'plain' else ('instructed', a)
            c = compare(run, 'held', cand, comp, a, ro_c=ro('cstar'), ro_k=ro(opp))
            sc[f'SC{first_id + 3 * j + i}'] = dict(comparisons=[c], **{'pass': passes(kind, [c])})
    return sc


def check_table(run, chk, stage, first_id, picks, vfile):
    sc = table_sc(run, first_id, picks)
    for sid, m in sc.items():
        check_sc(chk, sid, m, vfile['sc'].get(sid))
    chk.eq('sc ids', sorted(sc), sorted(vfile['sc']))
    go = all(m['pass'] for m in sc.values())
    chk.eq('go', go, as_go(vfile['go']))
    return sc, go


def map_block(run, sp, arms_aspects, metric='map'):
    """{arm: {aspect: {mean, per_seed}}} for the arms the SCs use."""
    out = {}
    for arm, head, a in arms_aspects:
        per = run.perq(sp, arm, head, a) if metric == 'map' else run.perq_ndcg(sp, arm, head)
        out.setdefault(arm, {})[a] = dict(mean=float(per.mean()),
                                          per_seed=per.mean(axis=1).tolist() if per.ndim == 2 else None)
    return out


def check_means(chk, label, mine, theirs):
    """Compare the arms and aspects present in both of mine {arm: {aspect: {mean, per_seed}}} and theirs."""
    for arm, by_a in mine.items():
        for a, m in by_a.items():
            t = (theirs.get(arm) or {}).get(a)
            if t is None:
                chk.problem(f'{label}.{arm}.{a}: not in file', None)
                continue
            chk.num(f'{label}.{arm}.{a}.mean', m['mean'], t['mean'])
            if m['per_seed'] is not None and t.get('per_seed') is not None:
                for s, (x, y) in enumerate(zip(m['per_seed'], t['per_seed'])):
                    chk.num(f'{label}.{arm}.{a}.s{s}', x, y)


# ---- stages ----------------------------------------------------------------------------------------------------

def stage0a(run, args):
    path, f = load_run_file(args.outputs, 'stage0a')
    chk = Check(args.tol)
    mine = run.ranges()
    for a in ASPECTS:
        for s in range(5):
            chk.num(f'map.{a}.s{s}', mine['map'][a]['per_seed'][s], f['map'][a]['per_seed'][s])
        chk.num(f'map.{a}.r', mine['map'][a]['r'], f['map'][a]['r'])
    for s in range(5):
        chk.num(f'ndcg.emotion.s{s}', mine['ndcg']['emotion']['per_seed'][s], f['ndcg']['emotion']['per_seed'][s])
    chk.num('ndcg.emotion.r', mine['ndcg']['emotion']['r'], f['ndcg']['emotion']['r'])
    return finish(args.outputs, 'stage0a', chk, mine, args, dict(checked_file=str(path), dropped=run.dropped_text()))


def stage1(run, args):
    path, f = load_run_file(args.outputs, 'stage1')
    chk = Check(args.tol)
    sc, go = check_table(run, chk, 'stage1', 1, None, f)
    mp_ = map_block(run, 'held', [(arm, head, a) for a in ASPECTS for arm, head in
                                  (('cstar', a), ('plain', None), ('instructed', a))])
    if 'map' in f:
        check_means(chk, 'map', mp_, f['map'])
    mine = dict(sc=sc, go=go, map=mp_, ranges=run.ranges())
    return finish(args.outputs, 'stage1', chk, mine, args, dict(checked_file=str(path), dropped=run.dropped_text()))


def stage2(run, args):
    path, f = load_run_file(args.outputs, 'stage2')
    chk = Check(args.tol)
    sc = {}
    for i, a in enumerate(ASPECTS):                               # SC7, SC8, SC9: column a
        comps = [compare(run, 'held', ('cstar', a), ('cstar', b), a) for b in ASPECTS if b != a]
        sc[f'SC{7 + i}'] = {'comparisons': comps, 'pass': passes('win', comps)}
    for sid, m in sc.items():
        check_sc(chk, sid, m, f['sc'].get(sid))
    chk.eq('sc ids', sorted(sc), sorted(f['sc']))
    go = all(m['pass'] for m in sc.values())
    chk.eq('switch_go', go, as_go(f['switch_go']))
    # c*'s switch cells (stage2/switch.json): mean over seeds of the mAP of head h scored with a-labels
    cells = {h: {a: float(run.perq('held', 'cstar', h, a).mean()) for a in ASPECTS} for h in ASPECTS}
    sw = Path(args.outputs) / 'stage2' / 'switch.json'
    if sw.exists():
        t = json.loads(sw.read_text()).get('cstar')
        if t is None:
            chk.problem('switch.json: no cstar entry', None)
        else:
            for h in ASPECTS:
                for a in ASPECTS:
                    chk.num(f'switch.cstar.{h}.{a}', cells[h][a], t[h][a])
    else:
        log(f'note: {sw} not found; c* switch cells not compared')
    mine = dict(sc=sc, switch_go=go, switch_cstar=cells)
    return finish(args.outputs, 'stage2', chk, mine, args, dict(checked_file=str(path), dropped=run.dropped_text()))


def stage3(run, args):
    path, f = load_run_file(args.outputs, 'stage3')
    chk = Check(args.tol)
    comps = {'ND-plain': compare(run, 'held', ('cstar', 'emotion'), ('plain', None), 'emotion', 'ndcg'),
             'ND-instructed': compare(run, 'held', ('cstar', 'emotion'), ('instructed', 'emotion'), 'emotion', 'ndcg')}
    for sid, c in comps.items():
        t = f['sc'].get(sid)
        if t is None:
            chk.problem(f'{sid}: missing in file', None)
            continue
        by = {ckey(x): x for x in t['comparisons']}
        x = by.get(ckey(c))
        if x is None:
            chk.problem(f'{sid}: comparison not in file', None)
            continue
        for key in ('d', 'r'):
            chk.num(f'{sid}.{key}', c[key], x[key])
        chk.num(f'{sid}.lo', c['interval'][0], x['interval'][0])
        chk.num(f'{sid}.hi', c['interval'][1], x['interval'][1])
        chk.eq(f'{sid}.outcome', c['outcome'], x['outcome'])
    # the reading: SC1's and SC4's outcomes (emotion, mAP, R0) re-derived here
    sc1 = compare(run, 'held', ('cstar', 'emotion'), ('plain', None), 'emotion')
    sc4 = compare(run, 'held', ('cstar', 'emotion'), ('instructed', 'emotion'), 'emotion')
    reading = ('holds' if (comps['ND-plain']['outcome'], comps['ND-instructed']['outcome']) ==
               (sc1['outcome'], sc4['outcome']) else 'qualified')
    chk.eq('reading', reading, f['reading'])
    nd = map_block(run, 'held', [('cstar', 'emotion', 'emotion'), ('plain', None, 'emotion'),
                                 ('instructed', 'emotion', 'emotion')], 'ndcg')
    if 'ndcg' in f:
        check_means(chk, 'ndcg', nd, f['ndcg'])
    mine = dict(sc=comps, reading=reading, sc1_outcome=sc1['outcome'], sc4_outcome=sc4['outcome'], ndcg=nd,
                ranges=run.ranges())
    return finish(args.outputs, 'stage3', chk, mine, args, dict(checked_file=str(path)))


def _s4_task(t):
    """One (arm, aspect, seed, half): fit each W on the other half's paintings, score this half (queries and
    gallery inside it). Returns {(w, size, k): mAP}."""
    arm, a, seed, half = t
    data = _ST['data']
    head = a if arm in HEADED else None
    X = data.emb(arm, 'val', head, seed)
    lab_all = data.labels['val'][a]
    pos = np.arange(len(X))
    fit = pos % 2 != half                                         # fit on the other half
    sc = ~fit
    Xs, lab = X[sc], lab_all[sc]
    sel = np.flatnonzero(lab >= 0)
    width = X.shape[1]
    out = {}
    stats = None
    for w, size in dict.fromkeys((r[0], r[1]) for r in menu(width)):
        if w != 'none' and stats is None:
            stats = fit_stats(X[fit])
        Z = apply_w(w, size, stats, Xs)
        G = np.ascontiguousarray(Z[sel]).astype(np.float64)
        res = ap_core(G, lab[sel], np.arange(len(sel)), KS)
        for k in KS:
            out[(w, size, k)] = float(np.nanmean(res[k]))
    return t, out


def stage4(run, args):
    d = run.d
    path, f = load_run_file(args.outputs, 'stage4', 'scores.json')
    ppath, fp = load_run_file(args.outputs, 'stage4', 'picks.json')
    chk = Check(args.tol)
    ranges = run.ranges()
    # every arm's validation embeddings are loaded before the fork
    tasks = []
    for arm in ARMS:
        if arm == 'crl':
            for a in ASPECTS:
                d.crl_text(a)
        for a in ASPECTS:
            for seed in arm_seeds(arm):
                d.emb(arm, 'val', a if arm in HEADED else None, seed)
                for half in (0, 1):
                    tasks.append((arm, a, seed, half))
    log(f'stage4: {len(tasks)} tasks')
    _ST['data'] = d
    res = dict(pmap(_s4_task, tasks, args.workers))
    mine, picks = {}, {}
    for arm in ARMS:
        widths = {d.width(arm, a if arm in HEADED else None) for a in ASPECTS}
        assert len(widths) == 1, f'{arm}: widths differ between aspects {widths}'
        width = widths.pop()
        mp_ = {}
        for ro in menu(width):
            mp_[ro] = {a: float(np.mean([np.mean([res[(arm, a, s, h)][ro] for h in (0, 1)]) for s in arm_seeds(arm)]))
                       for a in ASPECTS}
        base = mp_[R0]
        table = {}
        for ro, m in mp_.items():
            score = sum((m[a] - base[a]) / ranges['map'][a]['r'] for a in ASPECTS) / 3
            kept = all(m[a] - base[a] >= -ranges['map'][a]['r'] for a in ASPECTS)
            table[ro] = dict(map=m, score=score, kept=kept)
        best = max(t['score'] for t in table.values() if t['kept'])
        pool = [ro for ro, t in table.items() if t['kept'] and best - t['score'] <= TIE]
        pick = min(pool, key=tie_key)
        mine[arm] = {ro_name(ro): t for ro, t in table.items()}
        picks[arm] = ro_name(pick)
        # compare with the file
        theirs = f.get(arm)
        if theirs is None:
            chk.problem(f'{arm}: not in scores.json', None)
            continue
        tp = {ro_parse(k, width): v for k, v in theirs.items()}
        chk.eq(f'{arm}.readouts', sorted(map(ro_name, table)), sorted(map(ro_name, tp)))
        for ro, t in table.items():
            x = tp.get(ro)
            if x is None:
                continue
            for a in ASPECTS:
                chk.num(f'{arm}.{ro_name(ro)}.map.{a}', t['map'][a], x['map'][a])
            chk.num(f'{arm}.{ro_name(ro)}.score', t['score'], x['score'])
            chk.eq(f'{arm}.{ro_name(ro)}.kept', t['kept'], bool(x['kept']))
        theirs_pick = fp.get(arm)
        chk.eq(f'{arm}.pick', pick, ro_parse(theirs_pick, width) if theirs_pick else None)
    chk.eq('arms', sorted(picks), sorted(fp))
    return finish(args.outputs, 'stage4', chk, dict(scores=mine, picks=picks, ranges=ranges), args,
                  dict(checked_file=str(path), picks_file=str(ppath)))


def stage5(run, args):
    d = run.d
    path, f = load_run_file(args.outputs, 'stage5')
    ppath, fp = load_run_file(args.outputs, 'stage4', 'picks.json')
    chk = Check(args.tol)
    picks = {arm: ro_parse(fp[arm], d.width(arm, 'emotion' if arm in HEADED else None)) for arm in
             ('cstar', 'plain', 'instructed')}
    mine_picks = Path(args.outputs) / 'stage4' / 'rederive_stage4.json'
    if mine_picks.exists():                                       # this check's own stage 4 picks must be the run's
        mp_ = json.loads(mine_picks.read_text())
        if mp_.get('smoke') == args.smoke:
            for arm, name in mp_['mine']['picks'].items():
                if arm in picks:
                    chk.eq(f'stage4 pick {arm} (rederive_stage4.json vs picks.json)', name, fp[arm])
    sc, go = check_table(run, chk, 'stage5', 10, picks, f)
    mine = dict(sc=sc, go=go, picks={a: ro_name(r) for a, r in picks.items()}, ranges=run.ranges())
    return finish(args.outputs, 'stage5', chk, mine, args, dict(checked_file=str(path), picks_file=str(ppath)))


STAGES = dict(stage0a=stage0a, stage1=stage1, stage2=stage2, stage3=stage3, stage4=stage4, stage5=stage5)
LOAD_ARMS = dict(stage0a=('cstar',), stage1=('cstar', 'plain', 'instructed'), stage2=('cstar',),
                 stage3=('cstar', 'plain', 'instructed'), stage4=ARMS, stage5=('cstar', 'plain', 'instructed'))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('mode', choices=sorted(STAGES))
    ap.add_argument('--outputs', type=Path, default=OUTPUTS_DEFAULT)
    ap.add_argument('--inputs', type=Path, default=OUTPUTS_DEFAULT,
                    help='folder of the derived inputs (percept_val/, crl/); the real one also in smoke runs')
    ap.add_argument('--smoke', type=int, default=0, metavar='N')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--tol', type=float, default=TOL)
    args = ap.parse_args(argv)
    args.draws = 200 if args.smoke else DRAWS
    log(f'{args.mode}: outputs {args.outputs}, smoke {args.smoke}, draws {args.draws}, workers {args.workers}')
    try:
        data = Data(args.inputs, args.smoke)
    except FileNotFoundError as e:
        log(f'STOP (nothing compared): {e}')
        return 2
    log('  names', {k: len(v) for k, v in data.names.items()}, 'labelled',
        {sp: {a: int((data.labels[sp][a] >= 0).sum()) for a in ASPECTS} for sp in data.names})
    describe_loading(data, LOAD_ARMS[args.mode])
    try:
        run_file = RUN_FILE[args.mode]
        load_run_file(args.outputs, args.mode, run_file)          # stop before any heavy work when it is missing
        if args.mode == 'stage4':
            load_run_file(args.outputs, 'stage4', 'picks.json')
        if args.mode == 'stage5':
            load_run_file(args.outputs, 'stage4', 'picks.json')
        return STAGES[args.mode](Run(data, args), args)
    except (Missing, FileNotFoundError) as e:
        log(f'STOP (nothing compared): {e}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
