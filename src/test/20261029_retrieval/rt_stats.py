"""r11 retrieval: statistics, outcomes and the SC table (spec 2026-10-29-retrieval-design.md, 3.4 and 6).

Pure functions, no file reading.
"""
from typing import NamedTuple

import numpy as np


def outcome(lower, r):
    """r10's T-WIN / T-LEVEL / T-BELOW on the interval's lower bound; r is the seed range of the aspect."""
    if lower > 0:
        return 'WIN'
    if lower > -r:
        return 'LEVEL'
    return 'BELOW'


def seed_range(values5):
    """r_a: max minus min over seeds 0 to 4 of c*'s validation score (mAP, or nDCG for stage 3) under R0."""
    v = np.asarray(values5, dtype=np.float64)
    assert v.shape == (5,), v.shape
    return float(v.max() - v.min())


def chance(labels):
    """Sum over classes of p_c squared, with p_c the class shares among the labelled rows (label >= 0)."""
    labels = np.asarray(labels)
    lab = labels[labels >= 0]
    assert len(lab), 'no labelled row'
    p = np.bincount(lab) / len(lab)
    return float(np.sum(p ** 2))


_CHUNK = 250          # draws held in memory at once (a [chunk, n] float64 count matrix)


def _check_side(x, name):
    assert isinstance(x, np.ndarray) and x.dtype == np.float64, f'{name}: float64 array needed'
    assert x.ndim == 1 or (x.ndim == 2 and x.shape[0] == 5), f'{name}: shape {x.shape}, need [n] or [5, n]'
    return x


def paired(cand, comp, draws=10000):
    """Joint bootstrap of candidate minus comparator over queries (and seeds).

    Each side is float64[n] (unseeded) or float64[5, n] (seeded, row s = seed s). One default_rng(0) per call; per
    draw, in this order: I = rng.integers(0, n, size=n); then, only when either side is seeded,
    S = rng.choice([0..4], size=5, replace=True), shared by both sides. d_b = (mean over S of the mean over I of the
    candidate) - (the same for the comparator); an unseeded side is its mean over I. The point value d is the mean
    over all queries and all five seeds. Returns {'d': float, 'interval': [lo, hi] (2.5 and 97.5 percentiles),
    'boot': float64[draws]}.
    """
    cand, comp = _check_side(cand, 'cand'), _check_side(comp, 'comp')
    n = cand.shape[-1]
    assert comp.shape[-1] == n and n > 0, f'query counts differ: {cand.shape} vs {comp.shape}'
    seeded = cand.ndim == 2 or comp.ndim == 2
    # d_b is linear in the counts of I: mean over I of x = counts @ x / n. Stack both sides as columns:
    # candidate rows then comparator rows (an unseeded side is one column).
    cols = np.vstack([np.atleast_2d(cand), np.atleast_2d(comp)])           # [kc + km, n]
    kc = 1 if cand.ndim == 1 else 5
    rng = np.random.default_rng(0)
    boot = np.empty(draws, dtype=np.float64)
    for lo in range(0, draws, _CHUNK):
        m = min(_CHUNK, draws - lo)
        counts = np.empty((m, n), dtype=np.float64)
        sel = np.empty((m, 5), dtype=np.int64) if seeded else None
        for b in range(m):                       # the draw order is the plain loop's: I, then S
            counts[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)
            if seeded:
                sel[b] = rng.choice([0, 1, 2, 3, 4], size=5, replace=True)
        means = counts @ cols.T / n                                         # [m, kc + km]
        mc, mm = means[:, :kc], means[:, kc:]
        if cand.ndim == 2:
            mc = np.take_along_axis(mc, sel, axis=1).mean(axis=1)
        else:
            mc = mc[:, 0]
        if comp.ndim == 2:
            mm = np.take_along_axis(mm, sel, axis=1).mean(axis=1)
        else:
            mm = mm[:, 0]
        boot[lo:lo + m] = mc - mm
    d = float(cand.mean() - comp.mean())
    return dict(d=d, interval=np.percentile(boot, [2.5, 97.5]).tolist(), boot=boot)


# ---- the SC table (spec section 6) ------------------------------------------------------------------------------

ASPECTS = ('emotion', 'style', 'genre')
OUTCOMES = ('WIN', 'LEVEL', 'BELOW')
BARS = {'WIN': {'WIN'}, 'LEVEL_OR_WIN': {'WIN', 'LEVEL'}}


class Comparison(NamedTuple):
    """candidate, comparator: (arm, head); head is the aspect name whose head (c*) or instruction (instructed Qwen)
    ranks the queries, None for an aspect-free arm (plain Qwen). Labels are the SC's aspect. bar: a key of BARS."""
    candidate: tuple
    comparator: tuple
    bar: str


class Sc(NamedTuple):
    """One success criterion: scored on `aspect`'s labels in `stage`, every comparison needing its bar."""
    id: str
    stage: int
    aspect: str
    readout: str                 # 'R0' (plain cosine) or 'pick' (each arm's stage 4 read-out)
    metric: str
    comparisons: tuple


def _build():
    table = {}
    for a, aspect in enumerate(ASPECTS):
        table[f'SC{1 + a}'] = Sc(f'SC{1 + a}', 1, aspect, 'R0', 'map',
                                 (Comparison(('cstar', aspect), ('plain', None), 'WIN'),))
        table[f'SC{4 + a}'] = Sc(f'SC{4 + a}', 1, aspect, 'R0', 'map',
                                 (Comparison(('cstar', aspect), ('instructed', aspect), 'LEVEL_OR_WIN'),))
        table[f'SC{7 + a}'] = Sc(f'SC{7 + a}', 2, aspect, 'R0', 'map',
                                 tuple(Comparison(('cstar', aspect), ('cstar', h), 'WIN')
                                       for h in ASPECTS if h != aspect))
    for i in range(1, 7):
        base = table[f'SC{i}']
        table[f'SC{9 + i}'] = base._replace(id=f'SC{9 + i}', stage=5, readout='pick')
    return {k: table[k] for k in sorted(table, key=lambda s: int(s[2:]))}


SC = _build()
TABLE_IDS = {1: tuple(f'SC{i}' for i in range(1, 7)), 5: tuple(f'SC{i}' for i in range(10, 16))}
SWITCH_IDS = ('SC7', 'SC8', 'SC9')


def sc_pass(sc_id, outcomes):
    """True when every comparison of the SC meets its bar. outcomes: one outcome per comparison, in SC[...].comparisons
    order (a bare string is accepted for an SC with one comparison)."""
    sc = SC[sc_id]
    if isinstance(outcomes, str):
        outcomes = [outcomes]
    outcomes = list(outcomes)
    assert len(outcomes) == len(sc.comparisons), f'{sc_id} needs {len(sc.comparisons)} outcomes, got {outcomes}'
    assert all(o in OUTCOMES for o in outcomes), outcomes
    return all(o in BARS[c.bar] for o, c in zip(outcomes, sc.comparisons))


def _go(outcomes, ids):
    assert set(outcomes) == set(ids), f'need exactly {sorted(ids)}, got {sorted(outcomes)}'
    return all([sc_pass(i, outcomes[i]) for i in ids])


def table_go(outcomes, stage):
    """GO for a table: all six of its criteria pass. stage 1 (SC1 to SC6) or 5 (SC10 to SC15);
    outcomes: {sc_id: outcomes for sc_pass}, exactly those six."""
    assert stage in TABLE_IDS, stage
    return _go(outcomes, TABLE_IDS[stage])


def switch_go(outcomes):
    """Switch GO: SC7, SC8 and SC9 all pass; outcomes: {sc_id: [outcome against each other head]}, exactly those."""
    return _go(outcomes, SWITCH_IDS)


def stage3_reading(nd_plain, nd_instructed, sc1, sc4):
    """'holds' when the emotion nDCG outcomes (c* against plain; against instructed) equal SC1's and SC4's,
    else 'qualified'. Changes no outcome."""
    assert all(o in OUTCOMES for o in (nd_plain, nd_instructed, sc1, sc4))
    return 'holds' if (nd_plain, nd_instructed) == (sc1, sc4) else 'qualified'
