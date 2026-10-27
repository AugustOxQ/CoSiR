"""Independent re-derivation of the development side (ticket 12): rule §4.2 V1c, §4.3 V3c, §4.5 (C13).

Written from DECISION_RULE.md and the raw inputs alone: P's names, the labels, the exports {aspect}__P.npz of each
run, and (part b only) the score files' A values. It never imports or reads the round's scorers (rs_sweep, rs_stage3,
rs_ablation). Every threshold, range, tolerance, grid rule, tie rule and draw order below is written from the rule,
with the section in a comment. From rs_sheet it takes only configuration names (name, parse, config, stage1) and the
seeds; from rs_paths the paths and the frozen roots.

Modes (each writes rederive_<check>.json beside the file it checks and exits 1 on any disagreement):
  partitions --machine M      V1c part (a): every partition and A_a of stages 1 and 2 (CPU only; runs on the node)
  sweep --machine M           V1c part (b): gains, M, guard, knob gains, grid, pool, TOP from the A values
  stage3                      V3c: partitions, A-bar, e-bar, M-bar, the joint bootstrap, qualifiers, c*
  ablation --final NAME       the development ablation's point values e_a (rule §4.5)
  self-check                  the same code on 12-27's frozen stage A and stage B files
"""
import argparse
import hashlib
import io
import itertools
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

import rs_paths as paths
import rs_sheet as sheet

ROOT = paths.ROUND.parents[2]                 # the checkout root, for src.data (labels)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- constants, each from the rule -------------------------------------------------------------------------------
ASPECTS = ('emotion', 'style', 'genre')
R_A = dict(emotion=0.0046, style=0.0082, genre=0.0128)    # §3: the seed ranges r_a
TOL = 1e-6                                                # §4.2 V1c, §4.3 V3c, §4.5: "within 1e-6"
GUARD_FRACTION = 0.5                                      # §3.2: gain_a >= -r_a / 2
N_P, WIDTH, N_DRAWS = 29232, 2048, 2000                   # §3: 2,000 draws; P has 29,232 paintings
K, STATES, N_INIT, BATCH = 20, (0, 1, 2), 3, 4096         # EG §3.3 / RB §3.1: k-means on the float32 export
NORM_TOL = 1e-3                                           # RB §3.1: |norm - 1| < 1e-3, not re-normalised
MAX_GRID_KNOBS, MAX_LEVELS, TOP_N = 4, 2, 3               # §4.2: the 4 largest g_k; G's value + at most 2; TOP of 3
SWEEP_SEED, DEV_SEEDS = sheet.SWEEP_SEED, tuple(sheet.DEV_SEEDS)     # seeds: 0; 1, 2, 3 (§2.2, §2.3)
PERCENTILES = (2.5, 97.5)                                 # §3: numpy.percentile(., [2.5, 97.5])
KNOBS = tuple(f'k{i}' for i in range(1, 14))
G_NAME, G_REPEAT = 'G', 'G-repeat'                        # G0 (reference) and G0' (node's repeat run, excluded from the pool)
REMOVALS = tuple(sheet.REMOVALS)                          # the eight removals of spec §3.2 (names only)
G_ARM = paths.G_ARM                                       # 12-27's name for G: GS-0.02


class ExportProblem(AssertionError):
    """A saved export fails V1b (dtype, shape, finite, unit norm)."""


# ---- stages 1 and 2 (rule §3.2, §4.2): pure functions of the A values -----------------------------------------------

G_CFG = sheet.config()


def changed(cfg):
    """The knobs set away from G's value, in sheet order."""
    return tuple(k for k in KNOBS if cfg[k] != G_CFG[k])


def _sheet_values():
    """{knob: its three values in the sheet's listed order}, read off the stage 1 run list."""
    out = {}
    for cfg in sheet.stage1():
        (k,) = changed(cfg)
        out.setdefault(k, []).append(cfg[k])
    return out


VALUES = _sheet_values()
STAGE1_NAMES = [sheet.name(c) for c in sheet.stage1()]


def knob_numbers(cfg):
    """§4.2 TOP tie-break 2: the sorted tuple of the changed knobs' numbers."""
    return tuple(sorted(int(k[1:]) for k in changed(cfg)))


def positions(cfg):
    """§4.2 TOP tie-break 3: the positions of the changed knobs' values in the sheet, in knob order."""
    return tuple(VALUES[k].index(cfg[k]) for k in changed(cfg))


def top_key(name, m):
    """§4.2 TOP: the largest M; ties to fewer changed knobs, then the sorted knob numbers (lexicographic), then the
    positions of their values in the sheet."""
    cfg = sheet.parse(name)
    return (-m, len(changed(cfg)), knob_numbers(cfg), positions(cfg))


def gains(A, A0):
    """§3.2: gain_a(c) = A_a(c) - A_a(G0)."""
    return {a: A[a] - A0[a] for a in ASPECTS}


def mean_gain(g):
    """§3.2: M(c) = (1/3) sum_a gain_a / r_a."""
    return sum(g[a] / R_A[a] for a in ASPECTS) / 3


def guard_ok(g):
    """§3.2: gain_a >= -r_a / 2 for every aspect."""
    return all(g[a] >= -R_A[a] * GUARD_FRACTION for a in ASPECTS)


def scored_rows(runs):
    """One row per run; `runs` = {name: dict(status, A)}. Counting = passed V1b (status ok) and the guard."""
    assert runs[G_NAME]['status'] == 'ok', 'G0 failed: nothing to compare with'
    A0, rows = runs[G_NAME]['A'], {}
    for name, rec in runs.items():
        if rec['status'] != 'ok':
            rows[name] = dict(name=name, status='failed', A=None, gain=None, M=None, guard=None, counting=False)
            continue
        g = gains(rec['A'], A0)
        ok = guard_ok(g)
        rows[name] = dict(name=name, status='ok', A=dict(rec['A']), gain=g, M=mean_gain(g), guard=ok, counting=ok)
    return rows


def derive_stage1(runs):
    """Rule §4.2 stage 1: knob gains g_k (largest M over the knob's counting values, none when no value counts), the
    grid knobs (g_k > 0; the 4 largest, ties to the sheet's order), GRID-n or SWEEP-NONE, each grid knob's levels,
    the grid's points (all-G = G0, one changed knob = a stage 1 run reused, the rest new)."""
    missing = [n for n in [G_NAME] + STAGE1_NAMES if n not in runs]
    assert not missing, f'stage 1 runs missing: {missing}'
    rows = scored_rows(runs)
    knob_gains = {}
    for k in KNOBS:
        ms = [rows[sheet.name(sheet.config({k: v}))]['M'] for v in VALUES[k]
              if rows[sheet.name(sheet.config({k: v}))]['counting']]
        knob_gains[k] = max(ms) if ms else None
    positive = [k for k in KNOBS if knob_gains[k] is not None and knob_gains[k] > 0]
    chosen = sorted(positive, key=lambda k: (-knob_gains[k], int(k[1:])))[:MAX_GRID_KNOBS]
    grid_knobs = [k for k in KNOBS if k in chosen]
    levels = {k: level_values(k, rows) for k in grid_knobs}
    points = grid_points(grid_knobs, levels)
    return dict(rows=rows, knob_gains=knob_gains, grid_knobs=grid_knobs,
                outcome=f'GRID-{len(grid_knobs)}' if grid_knobs else 'SWEEP-NONE', levels=levels,
                grid=dict(points=points, new=[p['name'] for p in points if p['kind'] == 'new'],
                          reused=[p['name'] for p in points if p['kind'] == 'reused']))


def level_values(k, rows):
    """§4.2 stage 2: G's value and the knob's counting values with M > 0, at most two (the largest M; ties to the
    listed order of values)."""
    cands = []
    for pos, v in enumerate(VALUES[k]):
        row = rows[sheet.name(sheet.config({k: v}))]
        if row['counting'] and row['M'] > 0:
            cands.append((-row['M'], pos, v))
    return [G_CFG[k]] + [v for _, _, v in sorted(cands)[:MAX_LEVELS]]


def grid_points(grid_knobs, levels):
    """§4.2: every combination of the grid knobs' levels, all other knobs at G's value (no grid knob, no grid)."""
    points = []
    if not grid_knobs:
        return points
    for combo in itertools.product(*[levels[k] for k in grid_knobs]):
        cfg = sheet.config({k: v for k, v in zip(grid_knobs, combo) if v != G_CFG[k]})
        n = len(changed(cfg))
        points.append(dict(name=sheet.name(cfg), kind='g0' if n == 0 else 'reused' if n == 1 else 'new'))
    return points


def derive_stage2(runs):
    """Rule §4.2 stage 2: the new points' rows, the pool (every counting run of stages 1 and 2 with M > 0, G0 and G0'
    excluded) and TOP (the 3 largest M, ties by `top_key`), TOP-NONE when the pool is empty."""
    s1 = derive_stage1(runs)
    assert s1['grid_knobs'], 'SWEEP-NONE: stage 2 does not run'
    new = s1['grid']['new']
    missing = [n for n in new if n not in runs]
    assert not missing, f'stage 2 runs missing: {missing}'
    rows = scored_rows(runs)
    pool = [n for n in STAGE1_NAMES + new if rows[n]['counting'] and rows[n]['M'] > 0]
    top = sorted(pool, key=lambda n: top_key(n, rows[n]['M']))[:TOP_N]
    return dict(stage1=s1, rows={n: rows[n] for n in new}, pool=pool, top=top,
                outcome='TOP' if pool else 'TOP-NONE')


# ---- stage 3 (rule §3.3, §4.3): the joint bootstrap, qualifiers, c* -------------------------------------------------

def mean_ami(labels, parts, idx):
    """A_a(v, I) for one seed: the mean over the seed's three partitions of AMI(labels[I], partition[I]) (sklearn
    defaults). `idx` carries only labelled rows."""
    assert len(idx) and (labels[idx] >= 0).all() and parts.shape == (len(STATES), len(labels))
    return float(np.mean([adjusted_mutual_info_score(labels[idx], p[idx]) for p in parts]))


def draw_stream(n_rows, n_draws, seeds=DEV_SEEDS):
    """§3.3: the draws of one configuration, from its own `default_rng(0)`; for each b, in this order: I_b =
    `rng.integers(0, n_rows, size=n_rows)` (rows of P in its order), then S_b = `rng.choice(seeds, size=3,
    replace=True)`."""
    rng = np.random.default_rng(0)
    for _ in range(n_draws):
        yield rng.integers(0, n_rows, size=n_rows), rng.choice(list(seeds), size=3, replace=True)


def aspect_stream(labels_a, n_draws, seeds=DEV_SEEDS):
    """EG §3.3 "Fresh seeds" (the per-aspect mode of §3.3's scorer check): its own `default_rng(0)` per aspect; I_b =
    L_a[`rng.integers(0, len(L_a), size=len(L_a))`], then S_b."""
    rng = np.random.default_rng(0)
    L = np.flatnonzero(labels_a >= 0)
    for _ in range(n_draws):
        yield L[rng.integers(0, len(L), size=len(L))], rng.choice(list(seeds), size=3, replace=True)


_INPUT = None


def _draw_chunk(task):
    """Scores of draws start..stop-1: X[b, arm, aspect, seed index] (nan for a seed S_b does not hold) and S_b per
    aspect. The stream is regenerated from the start so that any chunking gives the same draws."""
    labels, arms, mode, seeds = _INPUT
    start, stop = task
    names = list(arms)
    n_rows = len(labels[ASPECTS[0]])
    if mode == 'joint':
        streams = {a: None for a in ASPECTS}
        shared = draw_stream(n_rows, stop, seeds)
    else:
        streams = {a: aspect_stream(labels[a], stop, seeds) for a in ASPECTS}
    X = np.full((stop - start, len(names), len(ASPECTS), len(seeds)), np.nan)
    S = np.zeros((stop - start, len(ASPECTS), 3), np.int64)
    with threadpool_limits(limits=1):
        for b in range(stop):
            if mode == 'joint':
                I, Sb = next(shared)
                draw = {a: (I[labels[a][I] >= 0], Sb) for a in ASPECTS}           # unlabelled rows dropped, repeats kept
            else:
                draw = {a: next(streams[a]) for a in ASPECTS}
            if b < start:
                continue
            for ai, a in enumerate(ASPECTS):
                idx, Sb = draw[a]
                S[b - start, ai] = Sb
                for s in sorted(set(int(x) for x in Sb)):
                    for ni, name in enumerate(names):
                        X[b - start, ni, ai, seeds.index(s)] = mean_ami(labels[a], arms[name][a][s], idx)
    return X, S


def bootstrap_scores(labels, arms, n_draws, workers, mode='joint', seeds=DEV_SEEDS):
    """X[b, arm, aspect, seed], S[b, aspect, 3] over all draws. `arms[name][aspect][seed]` = int64 [3, n]."""
    global _INPUT
    assert mode in ('joint', 'aspect') and 1 <= workers <= 8 and n_draws > 0
    _INPUT = labels, arms, mode, tuple(seeds)
    edges = np.linspace(0, n_draws, min(workers, n_draws) + 1, dtype=int)
    tasks = list(zip(edges[:-1], edges[1:]))
    if len(tasks) == 1:
        chunks = [_draw_chunk(tasks[0])]
    else:
        with multiprocessing.get_context('fork').Pool(len(tasks)) as pool:
            chunks = pool.map(_draw_chunk, tasks)
    return np.concatenate([c[0] for c in chunks]), np.concatenate([c[1] for c in chunks])


def point_values(labels, arms, seeds=DEV_SEEDS):
    """A-bar_a(v) = the mean over the seeds of A_a(v, L_a, s); also the per-seed values."""
    per_seed, abar = {}, {}
    for name, by_aspect in arms.items():
        per_seed[name], abar[name] = {}, {}
        for a in ASPECTS:
            idx = np.flatnonzero(labels[a] >= 0)
            per_seed[name][a] = {s: mean_ami(labels[a], by_aspect[a][s], idx) for s in seeds}
            abar[name][a] = float(np.mean(list(per_seed[name][a].values())))
    return abar, per_seed


def effects(labels, ref, arms, n_draws=N_DRAWS, workers=8, mode='joint', seeds=DEV_SEEDS):
    """§3.3 for every arm but `ref`: A-bar_a, e-bar_a = A-bar_a(c) - A-bar_a(ref), M-bar = (1/3) sum_a e-bar_a / r_a,
    and the intervals from the draws: e_a,b = the mean over S_b of A_a(c, I_b, s) - A_a(ref, I_b, s); M_b = (1/3) sum_a
    e_a,b / r_a (joint mode only); 95% by numpy.percentile(., [2.5, 97.5])."""
    assert ref in arms
    abar, per_seed = point_values(labels, arms, seeds)
    names = list(arms)
    out = {ref: dict(Abar=abar[ref], A=per_seed[ref])}
    if n_draws:
        X, S = bootstrap_scores(labels, arms, n_draws, workers, mode, seeds)
        counts = (S[..., None] == np.array(seeds)).sum(2)                           # [b, aspect, seed]
    for c in names:
        if c == ref:
            continue
        e = {a: dict(e=abar[c][a] - abar[ref][a], lo=None, hi=None) for a in ASPECTS}
        m = dict(M=mean_gain({a: e[a]['e'] for a in ASPECTS}), lo=None, hi=None)
        if n_draws:
            diff = np.where(counts > 0, X[:, names.index(c)] - X[:, names.index(ref)], 0.0)
            e_b = (counts * diff).sum(-1) / 3                                       # [b, aspect]
            for ai, a in enumerate(ASPECTS):
                e[a]['lo'], e[a]['hi'] = map(float, np.percentile(e_b[:, ai], PERCENTILES))
            m_b = (e_b / np.array([R_A[a] for a in ASPECTS])).sum(1) / 3
            m['lo'], m['hi'] = map(float, np.percentile(m_b, PERCENTILES))
        out[c] = dict(Abar=abar[c], A=per_seed[c], e=e, M=m)
    return out


def qualifies(e_lo, m_lo):
    """§4.3 Qualifies: for every aspect the lower bound of e-bar_a's interval > -r_a, and M-bar's lower bound > 0."""
    return all(e_lo[a] > -R_A[a] for a in ASPECTS) and m_lo > 0


def choose_pick(order, stats):
    """§4.3 PICK: c* = the qualifier with the largest unrounded M-bar, ties to TOP's order; KEEP-G when none qualifies."""
    qualifiers = [n for n in order if stats[n]['status'] == 'ok' and stats[n]['qualifies']]
    if not qualifiers:
        return dict(pick=None, outcome='KEEP-G', qualifiers=[])
    best = max(stats[n]['M']['M'] for n in qualifiers)
    return dict(pick=next(n for n in qualifiers if stats[n]['M']['M'] == best), outcome='PICK',
                qualifiers=qualifiers)


# ---- raw inputs: hashes, P, labels, exports, partitions (rule §3.2; EG §3.3; RB §3.1) -----------------------------

THREADS = 8                                               # RB §3.1: fixed, independent of the environment


def sha(data):
    return hashlib.sha256(data).hexdigest()


def array_sha(array):
    return sha(np.ascontiguousarray(array).tobytes())


def versions():
    return dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                threadpoolctl=threadpoolctl.__version__)


def write_json(path, value):
    """Atomic: write beside, then rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    with os.fdopen(fd, 'w') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def code_state():
    """(git HEAD, dirty): dirty = `git status --porcelain -- '*.py'` run in the round folder is not empty."""
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=here, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain', '--', '*.py'], cwd=here, text=True).strip())
    except (OSError, subprocess.CalledProcessError):      # the node's copy of the code may have no .git: say so
        return 'no-git', None
    return head, dirty


class Recorder:
    """Reads every input through itself and keeps its SHA-256 (the report lists them all)."""

    def __init__(self):
        self.hashes = {}

    def bytes(self, path):
        data = Path(path).read_bytes()
        self.hashes[str(path)] = sha(data)
        return data

    def json(self, path, keys=()):
        data = json.loads(self.bytes(path))
        assert isinstance(data, dict) and set(keys) <= data.keys(), f'{path}: missing keys {keys}'
        return data

    def npz(self, path):
        with np.load(io.BytesIO(self.bytes(path)), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}


class Context:
    """P's names (sorted, hash checked), the labels per aspect (int64 over all of P, -1 = none), P's size and width."""

    def __init__(self, names, labels, n_rows=N_P, width=WIDTH, p_sha=None):
        assert len(names) == n_rows and list(names) == sorted(set(names)), 'P count/order/unique'
        assert set(labels) == set(ASPECTS)
        for y in labels.values():
            assert y.shape == (n_rows,) and y.dtype == np.int64 and (y >= -1).all() and (y >= 0).any()
        self.names, self.labels, self.n_rows, self.width = list(names), labels, n_rows, width
        self.p_sha = sha('\n'.join(names).encode('utf-8')) if p_sha is None else p_sha

    def label_report(self):
        return dict(label_counts={a: int((y >= 0).sum()) for a, y in self.labels.items()},
                    label_sha256={a: array_sha(y) for a, y in self.labels.items()})


def labels_from_annotations(names, annotations, genre_ids):
    """Copied from rb_vp1.labels_from_annotations (12-24's re-derivation; RB §3.1 "Labels"). Emotion: plurality of the
    painting's caption emotions, ties to the smallest code (codes = sorted names over the annotations, "something
    else" included). Style: the first caption's, codes = sorted style names. Genre: `genre_ids` (-1 = none)."""
    emotion_names = sorted({r['emotion'] for r in annotations})
    style_names = sorted({r['art_style'] for r in annotations})
    emotion_code = {n: i for i, n in enumerate(emotion_names)}
    style_code = {n: i for i, n in enumerate(style_names)}
    wanted = set(names)
    counts, first_style = {}, {}
    for row in annotations:
        p = row['painting']
        if p not in wanted:
            continue
        if p not in counts:
            counts[p] = np.zeros(len(emotion_names), np.int64)
            first_style[p] = style_code[row['art_style']]
        counts[p][emotion_code[row['emotion']]] += 1
    assert wanted <= set(counts), f'{len(wanted - set(counts))} names without annotations'
    genre_ids = np.asarray(genre_ids)
    assert genre_ids.shape == (len(names),)
    return dict(emotion=np.array([int(np.argmax(counts[p])) for p in names], np.int64),
                style=np.array([first_style[p] for p in names], np.int64),
                genre=genre_ids.astype(np.int64))


def load_context(rec):
    """P from 12-24's P.json (hash equal to 12-25's), annotations and the WikiArt genre CSVs for the labels."""
    p = rec.json(paths.RB_OUT / 'part1' / 'P.json', ('names', 'sha256'))
    names = p['names']
    assert sha('\n'.join(names).encode('utf-8')) == p['sha256'] == paths.P_SHA256, 'P hash'
    from src.data.artelingo import ANNOTATIONS_PATH
    from src.data.wikiart_genre import GENRE_CSVS, load_wikiart_genre
    annotations = json.loads(rec.bytes(ANNOTATIONS_PATH))
    for path in GENRE_CSVS:
        rec.bytes(path)
    labels = labels_from_annotations(names, annotations, load_wikiart_genre(np.asarray(names)))
    return Context(names, labels, p_sha=p['sha256'])


def read_export(rec, path, ctx):
    """V1b on one export: float16, [n_rows, width], finite, |norm - 1| < 1e-3 (the saved rows, not re-normalised).
    Returns the float32 cast and the SHA-256 of the saved array."""
    z = rec.npz(path)
    assert 'img' in z and set(z) <= {'img', 'cap'}, f'{path}: keys {sorted(z)}'
    saved = z['img']
    if saved.dtype != np.float16 or saved.shape != (ctx.n_rows, ctx.width):
        raise ExportProblem(f'{path}: dtype/shape {saved.dtype} {saved.shape}')
    x = saved.astype(np.float32)
    if not np.isfinite(x).all():
        raise ExportProblem(f'{path}: non-finite rows')
    err = float(np.abs(np.linalg.norm(x.astype(np.float64), axis=1) - 1).max())
    if not err < NORM_TOL:
        raise ExportProblem(f'{path}: unit norm error {err:.3g}')
    return x, array_sha(saved)


def partitions(x, threads=THREADS):
    """int64 [3, n]: MiniBatchKMeans k = 20, random states 0, 1, 2, n_init 3, batch 4,096 (all else sklearn's
    defaults) on all of P, on the float32 cast, under threadpoolctl."""
    out = []
    with threadpool_limits(limits=threads):
        for state in STATES:
            km = MiniBatchKMeans(n_clusters=K, random_state=state, n_init=N_INIT, batch_size=BATCH)
            out.append(km.fit_predict(x).astype(np.int64))
    parts = np.stack(out)
    assert parts.shape == (len(STATES), len(x)) and ((parts >= 0) & (parts < K)).all()
    return parts


def derive_arm(rec, ctx, folders, threads=THREADS):
    """Partitions of one configuration's runs. `folders` = {seed: run folder}. Failed when a run.json says failed
    or an export fails V1b (the V1d/V3d reading: the run that is left after its one rerun)."""
    parts, img, problems, run_sha = {a: {} for a in ASPECTS}, {}, [], {}
    x_sha = {seed: {} for seed in folders}
    for seed, folder in folders.items():
        run = rec.json(Path(folder) / 'run.json', ('status',))
        run_sha[seed] = rec.hashes[str(Path(folder) / 'run.json')]
        assert run['status'] in ('ok', 'failed'), f'{folder}: status'
        if 'p_sha256' in run:
            assert run['p_sha256'] == ctx.p_sha, f'{folder}: names another P'
        if run['status'] == 'failed':
            problems.append(f's{seed}: run failed ({run.get("cause")})')
            continue
        for a in ASPECTS:
            try:
                x, img[f'{a}/s{seed}'] = read_export(rec, Path(folder) / f'{a}__P.npz', ctx)
                x_sha[seed][a] = img[f'{a}/s{seed}']
            except ExportProblem as error:
                problems.append(str(error))
                continue
            parts[a][seed] = partitions(x, threads)
    return dict(status='failed' if problems else 'ok', cause='; '.join(problems) or None, parts=parts,
                img_sha256=img, run_sha256=run_sha, img_by_seed=x_sha)


def a_value(ctx, aspect, parts):
    """A_a(c, L_a): the mean over the three states of AMI against the aspect's labels, over its labelled rows."""
    return mean_ami(ctx.labels[aspect], parts, np.flatnonzero(ctx.labels[aspect] >= 0))


# ---- comparison: items with value, expected, diff -------------------------------------------------------------------

MISSING = '<missing>'


def add_item(items, key, value, expected, same=None):
    """One comparison. Numbers: |diff| <= TOL. Everything else: equal. `same` overrides the pass test (partitions)."""
    if isinstance(value, bool) or isinstance(expected, bool) or not all(
            isinstance(v, (int, float)) for v in (value, expected)):
        ok = (value == expected) if same is None else same
        diff = None
    else:
        diff = abs(value - expected)
        ok = bool(np.isfinite(diff) and diff <= TOL) if same is None else same
    items[key] = dict(value=value, expected=expected, diff=diff, **{'pass': bool(ok)})


def compare(items, path, ours, theirs):
    """Walk `ours` (dicts by key, lists by position) against `theirs`; every leaf becomes an item. Keys of `theirs`
    that `ours` lacks are not compared. A 'cause' is prose: both nonempty strings or both none."""
    if isinstance(ours, dict):
        theirs = theirs if isinstance(theirs, dict) else {}
        for k, v in ours.items():
            compare(items, f'{path}/{k}', v, theirs.get(k, MISSING))
    elif isinstance(ours, (list, tuple)) and isinstance(theirs, (list, tuple)) and len(ours) == len(theirs):
        for i, (a, b) in enumerate(zip(ours, theirs)):
            compare(items, f'{path}/{i}', a, b)
    elif path.endswith('/cause'):
        add_item(items, path, ours, theirs,
                 same=(ours is None and theirs is None) or all(isinstance(v, str) and v.strip() for v in (ours, theirs)))
    else:
        add_item(items, path, ours if not isinstance(ours, (list, tuple)) else list(ours),
                 theirs if not isinstance(theirs, tuple) else list(theirs))


def sets_equal(items, key, ours, theirs):
    """Order-free comparison (the rule fixes no order): sorted lists of names (a dict counts as its keys)."""
    add_item(items, key, sorted(ours), sorted(theirs) if isinstance(theirs, (list, tuple, dict)) else theirs)


def report(check, items, rec, ctx, **extra):
    head, dirty = code_state()
    head_ctx = dict(p_sha256=ctx.p_sha, **ctx.label_report()) if ctx is not None else {}
    return dict(check=check, **{'pass': bool(items) and all(i['pass'] for i in items.values())}, items=items,
                input_sha256=dict(rec.hashes), **head_ctx, versions=versions(),
                threads=THREADS, rule_commit=paths.RULE_COMMIT, code_commit=head, code_dirty=dirty,
                code_sha256=sha(Path(__file__).read_bytes()), **extra)


# ---- part (a): partitions and A of stages 1 and 2 (rule §4.2 V1c) ---------------------------------------------------

def score_dir(outputs, machine):
    return Path(outputs) / 'scores' / machine


def graph_gate_failure(rec, outputs, gate, variant=None):
    """§2.2 / §2.5: a failed graph reproduction stops only the runs that need those graphs."""
    path = Path(outputs) / 'gates' / f'{gate}.json'
    if not path.exists():
        return None
    data = rec.json(path, ('pass',))
    item = data if variant is None else data.get('items', {}).get(variant)
    if item is None:
        return None
    assert isinstance(item.get('pass'), bool), f'{path}: gate pass is not boolean'
    return f'{gate}/{variant or "sample"}: graph reproduction failed' if item['pass'] is False else None


def gated_sweep_runs(rec, outputs, runs):
    """§2.2: k11's three stage 1 values do not run and are not counting when its check fails."""
    cause = graph_gate_failure(rec, outputs, 'k11_sample')
    if cause:
        runs = dict(runs)
        runs.update({n: dict(status='failed', A=None, cause=cause) for n in STAGE1_NAMES if n.startswith('k11=')})
    return runs


def stage_run_folders(outputs, frozen_g=None, skipped=None):
    """{name: seed-0 run folder} of this machine's stage 1 and 2 runs: the sample-scale run folders under train/ with
    a seed 0 and no removal. G0 is train/G/s0, or, when the machine has none (the fallback), 12-27's frozen
    GS-0.02 seed 0 run."""
    out = {}
    for d in sorted((Path(outputs) / 'train').glob('*')):
        if not d.is_dir():
            continue
        try:
            removal = d.name != G_REPEAT and sheet.parse(d.name)['remove']
        except (AssertionError, ValueError, KeyError):
            if skipped is not None:
                skipped.append(d.name)
            continue
        if removal or not (d / 's0' / 'run.json').exists():
            continue
        out[d.name] = d / 's0'
    if G_NAME not in out:
        out[G_NAME] = Path(paths.RG_TRAIN / G_ARM / 's0' if frozen_g is None else frozen_g)
    return out


def part_a(outputs, machine, ctx, rec=None, only=None, threads=THREADS, frozen_g=None, dest=None):
    """V1c (a): recompute every partition and A_a from the exports; compare with the scorer's partitions and score
    files: equal partitions, A within 1e-6, same status. Writes scores/<machine>/rederive_partitions.json."""
    rec = rec or Recorder()
    sdir = score_dir(outputs, machine)
    skipped = []
    folders = stage_run_folders(outputs, frozen_g, skipped)
    names = sorted(folders if only is None else only)
    blocked = gated_sweep_runs(rec, outputs, {})
    names = sorted(set(names) | (set(blocked) if only is None else set(blocked) & set(only)))
    items, runs = {}, {}
    for name in names:
        if name in blocked:
            runs[name] = blocked[name]
            theirs_path = sdir / f'{name}__s{SWEEP_SEED}.json'
            if theirs_path.exists():
                theirs = rec.json(theirs_path, ('status', 'A'))
                compare(items, f'gate/{name}', dict(status='failed', A=None), theirs)
            continue
        arm = derive_arm(rec, ctx, {SWEEP_SEED: folders[name]}, threads)
        A = None if arm['status'] == 'failed' else {
            a: a_value(ctx, a, arm['parts'][a][SWEEP_SEED]) for a in ASPECTS}
        runs[name] = dict(status=arm['status'], cause=arm['cause'], A=A, source=str(folders[name]),
                          img_sha256=arm['img_sha256'],
                          partitions_sha256={a: array_sha(arm['parts'][a][SWEEP_SEED]) for a in ASPECTS
                                             if SWEEP_SEED in arm['parts'][a]})
        theirs_path = sdir / f'{name}__s{SWEEP_SEED}.json'
        theirs = rec.json(theirs_path) if theirs_path.exists() else None
        if theirs is None:
            add_item(items, f'score_file/{name}', 'missing', 'present', same=False)
            continue
        add_item(items, f'status/{name}', runs[name]['status'], theirs['status'])
        if theirs['status'] == 'ok' and A is not None:
            for a in ASPECTS:
                add_item(items, f'A/{name}/{a}', A[a], theirs['A'][a])
                pfile = sdir / 'partitions' / name / f's{SWEEP_SEED}' / f'{a}__partitions.npz'
                saved = rec.npz(pfile)['labels'] if pfile.exists() else None
                same = saved is not None and saved.dtype == np.int64 and np.array_equal(saved, arm['parts'][a][SWEEP_SEED])
                add_item(items, f'partitions/{name}/{a}', array_sha(arm['parts'][a][SWEEP_SEED]),
                         MISSING if saved is None else array_sha(saved), same=same)
        elif (theirs['status'] == 'ok') != (A is not None):
            add_item(items, f'A/{name}', A, theirs.get('A'), same=False)
    if only is None:
        extra = sorted(p.name[:-len(f'__s{SWEEP_SEED}.json')] for p in sdir.glob(f'*__s{SWEEP_SEED}.json')
                       if p.name[:-len(f'__s{SWEEP_SEED}.json')] not in set(folders) | set(blocked))
        for name in extra:
            add_item(items, f'score_file/{name}', 'no run folder', 'run folder', same=False)
    rep = report('V1c partitions', items, rec, ctx, machine=machine, n_runs=len(runs), runs=runs, skipped_folders=skipped)
    write_json(dest or sdir / 'rederive_partitions.json', rep)
    return rep


# ---- part (b): the decisions of stages 1 and 2 from the A values (rule §4.2 V1c) ---------------------------------

def guard_bool(guard):
    """The guard as one bool (a file may keep it per aspect)."""
    return all(guard.values()) if isinstance(guard, dict) else guard


def compare_rows(items, path, mine_rows, theirs_rows):
    """Rows by name: status, A, gain, M, guard, counting. `theirs_rows` = the file's list of row dicts."""
    if not isinstance(theirs_rows, list):
        add_item(items, path, 'rows', theirs_rows, same=False)
        return
    by_name = {r['name']: r for r in theirs_rows}
    for name, row in mine_rows.items():
        r = by_name.get(name)
        if r is None:
            add_item(items, f'{path}/{name}', 'row', MISSING, same=False)
            continue
        compare(items, f'{path}/{name}', {k: row[k] for k in ('status', 'A', 'gain', 'M', 'counting')}, r)
        add_item(items, f'{path}/{name}/guard', row['guard'], None if row['guard'] is None else guard_bool(
            r.get('guard', MISSING)))


def compare_stage1(items, mine, theirs):
    """mine = derive_stage1(scorer's A values); theirs = sweep/<machine>/stage1.json. Lists the rule fixes no order
    for (grid knobs, levels, grid points) are compared as sets."""
    compare(items, 'stage1/ranges', R_A, theirs.get('ranges', MISSING))
    compare(items, 'stage1/reference/A', mine['rows'][G_NAME]['A'], theirs.get('reference', {}).get('A', MISSING))
    compare_rows(items, 'stage1/runs', {n: mine['rows'][n] for n in STAGE1_NAMES}, theirs.get('runs', MISSING))   # G is `reference`
    sets_equal(items, 'stage1/failed', [n for n in STAGE1_NAMES if mine['rows'][n]['status'] == 'failed'],
               [f['name'] for f in theirs['failed']] if 'failed' in theirs else MISSING)
    compare(items, 'stage1/knob_gains', mine['knob_gains'], theirs.get('knob_gains', MISSING))
    sets_equal(items, 'stage1/grid_knobs', mine['grid_knobs'], theirs.get('grid_knobs', MISSING))
    add_item(items, 'stage1/outcome', mine['outcome'], theirs.get('outcome', MISSING))
    levels = theirs.get('levels', {})
    for k, vs in mine['levels'].items():
        add_item(items, f'stage1/levels/{k}', sorted(vs), sorted(levels[k]) if k in levels else MISSING)
    add_item(items, 'stage1/levels/knobs', sorted(mine['levels']), sorted(levels))
    grid = theirs.get('grid', {})
    kinds = {p['name']: p['kind'] for p in grid.get('points', [])}
    compare(items, 'stage1/grid/kinds', {p['name']: p['kind'] for p in mine['grid']['points']}, kinds)
    add_item(items, 'stage1/grid/n_points', len(mine['grid']['points']), len(kinds))
    for key in ('new', 'reused'):
        sets_equal(items, f'stage1/grid/{key}', mine['grid'][key], grid.get(key, MISSING))


def compare_stage2(items, mine, theirs):
    """mine = derive_stage2(scorer's A values); theirs = sweep/<machine>/stage2.json. TOP is ordered; the pool is a
    set; the outcome is TOP-NONE or not (the file's word for the other case is not fixed by the rule)."""
    compare_rows(items, 'stage2/new_runs', mine['rows'], theirs.get('new_runs', MISSING))
    sets_equal(items, 'stage2/failed', [n for n, r in mine['rows'].items() if r['status'] == 'failed'],
               [f['name'] for f in theirs['failed']] if 'failed' in theirs else MISSING)
    sets_equal(items, 'stage2/pool', mine['pool'], theirs.get('pool', MISSING))
    top = [r['name'] if isinstance(r, dict) else r for r in theirs['top']] if 'top' in theirs else MISSING
    add_item(items, 'stage2/top', mine['top'], top)
    add_item(items, 'stage2/outcome_is_top_none', mine['outcome'] == 'TOP-NONE',
             theirs.get('outcome') == 'TOP-NONE')


def runs_from(records):
    """{name: dict(status, A)} from score files or from rederive_partitions.json's runs."""
    return {n: dict(status=r['status'], A=r['A']) for n, r in records.items()}


def scorer_runs(rec, sdir):
    """Every seed 0 score file of the machine: {name: dict(status, A)}."""
    suffix = f'__s{SWEEP_SEED}.json'
    return runs_from({p.name[:-len(suffix)]: rec.json(p, ('status', 'A')) for p in sorted(sdir.glob('*' + suffix))})


def sweep(outputs, machine):
    """V1c (b): derive stages 1 and 2 from the scorer's A values and, separately, from my A values of part (a); the
    two derivations must give the same decisions, and the first must equal sweep/<machine>/stage{1,2}.json."""
    rec, sdir = Recorder(), score_dir(outputs, machine)
    part = rec.json(sdir / 'rederive_partitions.json', ('pass', 'runs'))
    assert part['pass'] is True, 'part (a) did not pass or did not run: fix it first (rule §4.2 V1c)'
    theirs_runs = gated_sweep_runs(rec, outputs, scorer_runs(rec, sdir))
    mine_runs = gated_sweep_runs(rec, outputs, runs_from(part['runs']))
    reports = []
    for stage, derive, compare_fn in ((1, derive_stage1, compare_stage1), (2, derive_stage2, compare_stage2)):
        path = Path(outputs) / 'sweep' / machine / f'stage{stage}.json'
        if not path.exists():
            continue
        try:
            d_t, d_m = derive(theirs_runs), derive(mine_runs)
        except AssertionError as error:
            raise AssertionError(f'stage {stage}: {error} (rerun `partitions` after the runs it needs exist)') from error
        d1_t, d1_m = (d_t, d_m) if stage == 1 else (d_t['stage1'], d_m['stage1'])
        items = {}
        sets_equal(items, 'both_sets/stage1/grid_knobs', d1_m['grid_knobs'], d1_t['grid_knobs'])
        add_item(items, 'both_sets/stage1/outcome', d1_m['outcome'], d1_t['outcome'])
        sets_equal(items, 'both_sets/stage1/grid_new', d1_m['grid']['new'], d1_t['grid']['new'])
        for name, row in d1_m['rows'].items():
            if name in d1_t['rows']:
                compare(items, f'both_sets/rows/{name}', {k: row[k] for k in ('status', 'counting')},
                        d1_t['rows'][name])
        if stage == 2:
            add_item(items, 'both_sets/stage2/top', d_m['top'], d_t['top'])
            sets_equal(items, 'both_sets/stage2/pool', d_m['pool'], d_t['pool'])
        theirs = rec.json(path)
        compare_fn(items, d_t, theirs)
        rep = report(f'V1c stage {stage}', items, rec, None, machine=machine,
                     mine=dict(outcome=d1_m['outcome'], grid_knobs=d1_m['grid_knobs'], new=d1_m['grid']['new'],
                               **({} if stage == 1 else dict(pool=d_m['pool'], top=d_m['top']))),
                     scorer_sha256={n: rec.hashes[str(sdir / f'{n}__s{SWEEP_SEED}.json')] for n in theirs_runs
                                    if str(sdir / f'{n}__s{SWEEP_SEED}.json') in rec.hashes})
        write_json(path.parent / f'rederive_stage{stage}.json', rep)
        reports.append(rep)
    assert reports, f'no stage1.json under {Path(outputs) / "sweep" / machine}'
    return reports


# ---- stage 3 (V3c) and the development ablation (rule §4.3, §4.5): files to numbers ----------------------------------

def cache_file(outputs, stage, name, seed, aspect):
    """Where the scorer keeps one run's partitions: <outputs>/<stage>/cache/<name>/s<seed>/{aspect}__partitions.npz
    (stage = 'stage3' or 'ablation'; ticket 06's and 10's layout)."""
    return Path(outputs) / stage / 'cache' / name / f's{seed}' / f'{aspect}__partitions.npz'


def compare_cache(items, rec, outputs, stage, name, arm, required, item_name=None):
    """The scorer's saved partitions of an ok arm against mine, every seed and aspect. A file that is not there is
    a difference when `required`, and not looked for otherwise."""
    for s in DEV_SEEDS:
        for a in ASPECTS:
            path = cache_file(outputs, stage, name, s, a)
            if required or path.exists():
                compare_partition_file(items, rec, path, f'partitions/{item_name or name}/s{s}/{a}', arm['parts'][a][s])


def compare_partition_file(items, rec, path, key, mine):
    """Equal partitions: the saved [3, n] int64 array equals mine, element for element."""
    saved = rec.npz(path)['labels'] if Path(path).exists() else None
    same = saved is not None and saved.dtype == np.int64 and np.array_equal(saved, mine)
    add_item(items, key, array_sha(mine), MISSING if saved is None else array_sha(saved), same=same)


def dev_arms(rec, ctx, arm_folders, threads):
    """derive_arm for each {name: {seed: folder}}; a failed arm is dropped from the bootstrap."""
    arms = {name: derive_arm(rec, ctx, folders, threads) for name, folders in arm_folders.items()}
    assert arms[next(iter(arms))]['status'] == 'ok', 'the reference arm failed V1b: nothing to compare with'
    return arms


def arm_stats(arms, ctx, ref, n_draws, workers):
    """Joint-bootstrap statistics of every ok arm against `ref` (all ok arms share the draws)."""
    ok = {n: a['parts'] for n, a in arms.items() if a['status'] == 'ok'}
    eff = effects(ctx.labels, ref, ok, n_draws, workers)
    out = {}
    for name, arm in arms.items():
        if name == ref:
            continue
        if arm['status'] != 'ok':
            out[name] = dict(status='failed', cause=arm['cause'], qualifies=False)
            continue
        c = eff[name]
        out[name] = dict(status='ok', cause=None, Abar=c['Abar'], A=c['A'], e=c['e'], M=c['M'],
                         qualifies=bool(n_draws and qualifies({a: c['e'][a]['lo'] for a in ASPECTS}, c['M']['lo'])),
                         img_sha256=arms[name]['img_sha256'])
    return out, eff[ref]


def stage3_core(outputs, top, ctx, rec, n_draws=N_DRAWS, workers=8, threads=THREADS, g_folders=None,
                g_partitions=None):
    """V3c: the partitions of TOP's configurations at seeds 1 to 3 and of G's frozen runs, A-bar, e-bar, M-bar, the
    joint bootstrap, the qualifiers and c*. G's runs: 12-27's GS-0.02 seeds 1 to 3 (`g_folders`); their partitions
    are compared with 12-27's stage B partitions (`g_partitions`) when given. TOP's partitions are compared with
    <outputs>/stage3/cache/<name>/s<seed>/ (required for every ok configuration)."""
    g_folders = g_folders or {s: paths.RG_TRAIN / G_ARM / f's{s}' for s in DEV_SEEDS}
    folders = {G_NAME: g_folders}
    folders.update({n: {s: paths.run_dir(outputs, n, s) for s in DEV_SEEDS} for n in top})
    arms = dev_arms(rec, ctx, folders, threads)
    items = {}
    if g_partitions is not None:
        for s in DEV_SEEDS:
            for a in ASPECTS:
                compare_partition_file(items, rec, Path(g_partitions) / f's{s}' / f'{a}__partitions.npz',
                                       f'partitions/G/s{s}/{a}', arms[G_NAME]['parts'][a][s])
    compare_cache(items, rec, outputs, 'stage3', G_ARM, arms[G_NAME], False, item_name='G_cache')     # if the scorer cached G's too
    for name in top:
        if arms[name]['status'] == 'ok':
            compare_cache(items, rec, outputs, 'stage3', name, arms[name], True)
    stats, ref = arm_stats(arms, ctx, G_NAME, n_draws, workers)
    chosen = choose_pick(list(top), stats)
    pick_config = None if chosen['pick'] is None else json.loads(json.dumps(sheet.parse(chosen['pick'])))
    inputs = dict(runs={n: {str(seed): dict(run_sha256=arms[n]['run_sha256'][seed],
                                          img_sha256=arms[n]['img_by_seed'][seed] or None)
                            for seed in DEV_SEEDS} for n in top})
    return dict(top=list(top), G=dict(Abar=ref['Abar'], A=ref['A']), configs=stats, **chosen, pick_config=pick_config,
                inputs=inputs,
                failed=[n for n in top if stats[n]['status'] == 'failed'],
                not_qualifying=[n for n in top if stats[n]['status'] == 'ok' and not stats[n]['qualifies']],
                n_draws=n_draws, items=items, items_pass=bool(items) and all(i['pass'] for i in items.values()))


def ablation_core(outputs, final, ctx, rec, n_draws=0, workers=8, threads=THREADS, g_folders=None, g_partitions=None,
                  removals=REMOVALS):
    """The development ablation's e_a(x) = A-bar_a(x) - A-bar_a(F) at seeds 1 to 3 (rule §3.5, §4.5); intervals only
    when n_draws > 0. F = `final`: G (12-27's frozen runs) or a TOP run of this round. The scorer's partitions
    (<outputs>/ablation/cache/<arm>/s<seed>/) are compared for every ok arm, and for F's when they exist."""
    cfg = sheet.parse(final)
    assert not cfg['remove'] and set(removals) <= set(REMOVALS), (final, removals)
    base = {k: cfg[k] for k in changed(cfg)}
    names = {x: sheet.name(sheet.config(base, remove=(x,))) for x in removals}
    g_folders = g_folders or {s: paths.RG_TRAIN / G_ARM / f's{s}' for s in DEV_SEEDS}
    folders = {final: g_folders if final == G_NAME else {s: paths.run_dir(outputs, final, s) for s in DEV_SEEDS}}
    blocked = {}
    variants = dict(content_correction='uncorrected', instructed_graphs='default_view')  # §2.5
    for x, n in names.items():
        cause = graph_gate_failure(rec, outputs, 'ablation_graphs', variants[x]) if x in variants else None
        if cause:
            blocked[n] = dict(status='failed', cause=cause)
        else:
            assert all((paths.run_dir(outputs, n, s) / 'run.json').exists() for s in DEV_SEEDS), \
                f'ablation arm without runs: {n}'
            folders[n] = {s: paths.run_dir(outputs, n, s) for s in DEV_SEEDS}
    arms = dev_arms(rec, ctx, folders, threads)
    arms.update(blocked)
    items = {}
    if final == G_NAME and g_partitions is not None:
        for seed in DEV_SEEDS:
            for a in ASPECTS:
                compare_partition_file(items, rec, Path(g_partitions) / f's{seed}' / f'{a}__partitions.npz',
                                       f'partitions/G/s{seed}/{a}', arms[final]['parts'][a][seed])
    compare_cache(items, rec, outputs, 'ablation', G_ARM if final == G_NAME else final, arms[final], False,
                  item_name='G_cache' if final == G_NAME else final)
    if final != G_NAME:
        compare_cache(items, rec, outputs, 'stage3', final, arms[final], False, item_name=f'{final}_stage3_cache')
    for name in names.values():
        if arms[name]['status'] == 'ok':
            compare_cache(items, rec, outputs, 'ablation', name, arms[name], True)
    stats, ref = arm_stats(arms, ctx, final, n_draws, workers)
    seed_inputs = lambda arm: {str(s): dict(run_sha256=arm['run_sha256'][s], img_sha256=arm['img_by_seed'][s])   # noqa: E731
                               for s in DEV_SEEDS}
    inputs = dict(final_runs=seed_inputs(arms[final]),
                  arm_runs={x: seed_inputs(arms[n]) for x, n in names.items() if arms[n]['status'] == 'ok'})
    return dict(final=final, F=dict(Abar=ref['Abar'], A=ref['A']), arms=stats, removal_arms=names, n_draws=n_draws,
                inputs=inputs, items=items)


# ---- the verdict files: one place each to adapt (stage3/verdict.json, ablation/dev.json) ------------------------------

def norm_config(rec, name=None):
    """A stage 3 result of one configuration, read into the keys mine uses. Layout of ticket 06 (examples written by
    its tests): rec = {status, cause, effects, M, conditions, qualifies}; effects[a] = {e, lo, hi}, with Abar (a number,
    or {config: value, G: value}) and A ({config: {seed: value}}) beside, as in 12-27's stage B file; a flat `Abar`
    {aspect: value} or `e` in rec is read too. M = {M, lo, hi}. `conditions` is not compared (its keys are not fixed)."""
    if not isinstance(rec, dict):
        return MISSING
    eff = rec.get('effects', rec.get('e'))
    e, abar, a_seed, abar_ref = MISSING, rec.get('Abar', MISSING), MISSING, MISSING
    if isinstance(eff, dict):
        e = {a: (dict(v) if not isinstance(v, dict) else {k: v[k] for k in ('e', 'lo', 'hi') if k in v})
             for a, v in eff.items()}
        for a, v in eff.items():
            if isinstance(v, dict) and 'Abar' in v:
                abar = {} if abar == MISSING else abar
                abar[a] = v['Abar'].get(name, MISSING) if isinstance(v['Abar'], dict) else v['Abar']
                if isinstance(v['Abar'], dict):
                    abar_ref = {} if abar_ref == MISSING else abar_ref
                    abar_ref[a] = v['Abar'].get(G_NAME, v['Abar'].get(G_ARM, MISSING))
            if isinstance(v, dict) and isinstance(v.get('A'), dict) and name in v['A']:
                a_seed = {} if a_seed == MISSING else a_seed
                a_seed[a] = v['A'][name]
    m = rec.get('M', rec.get('Mbar', MISSING))
    return dict(status=rec.get('status', MISSING), cause=rec.get('cause'), Abar=abar, e=e, A=a_seed, Abar_ref=abar_ref,
                M=m if isinstance(m, dict) or m == MISSING else dict(M=m), qualifies=rec.get('qualifies', MISSING))


def _drop_none(value):
    """Intervals mine did not compute (n_draws 0) are not compared."""
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    return value


def seed_keys(per_seed):
    """{seed: value} -> {"seed": value}, the JSON form."""
    return {str(s): v for s, v in per_seed.items()}


def compare_config(items, path, name, mine, theirs):
    """One stage 3 configuration: status, cause, qualifies and, for an ok one, Abar, e, M with the intervals mine
    computed; per-seed A when the file carries it."""
    t = norm_config(theirs, name)
    if t == MISSING:
        add_item(items, path, 'record', MISSING, same=False)
        return
    add_item(items, f'{path}/status', mine['status'], t['status'])
    add_item(items, f'{path}/qualifies', mine['qualifies'], t['qualifies'])
    if mine['status'] != 'ok':
        compare(items, f'{path}/cause', mine['cause'], t['cause'])
        for key in ('effects', 'M'):
            add_item(items, f'{path}/{key}', None, theirs.get(key, MISSING))
        return
    for key in ('Abar', 'e', 'M'):
        compare(items, f'{path}/{key}', _drop_none(mine[key]), t[key])
    if t['A'] != MISSING:
        compare(items, f'{path}/A', {a: seed_keys(v) for a, v in mine['A'].items()}, t['A'])


def compare_stage3(items, mine, verdict, stage2=None):
    """mine = stage3_core's record; verdict = stage3/verdict.json (ticket 06's layout; `written` is not compared).
    Compared: ranges, seeds, draws, TOP (ordered), outcome, pick, pick_config, qualifiers / not_qualifying / failed (as
    sets), every configuration's numbers, G's Abar and per-seed A, and the run/export hashes.
    `lead` and `conditions` are reported only."""
    compare(items, 'stage3/ranges', R_A, verdict.get('ranges', MISSING))
    add_item(items, 'stage3/seeds', list(DEV_SEEDS), verdict.get('seeds', MISSING))
    add_item(items, 'stage3/draw_mode', 'joint', verdict.get('draw_mode', MISSING))     # §3.3: the joint bootstrap
    add_item(items, 'stage3/n_draws', mine['n_draws'], verdict.get('n_draws', MISSING))
    add_item(items, 'stage3/reference/name', G_NAME, verdict.get('reference', {}).get('name', MISSING))
    add_item(items, 'stage3/top', mine['top'], verdict.get('top', MISSING))
    add_item(items, 'stage3/outcome', mine['outcome'], verdict.get('outcome', MISSING))
    add_item(items, 'stage3/pick', mine['pick'], verdict.get('pick', MISSING))
    add_item(items, 'stage3/pick_config', mine['pick_config'], verdict.get('pick_config', MISSING))
    for key in ('qualifiers', 'not_qualifying', 'failed'):
        sets_equal(items, f'stage3/{key}', mine[key], verdict.get(key, MISSING))
    recs = verdict.get('results', {})
    sets_equal(items, 'stage3/results/names', mine['configs'], recs)
    for name, c in mine['configs'].items():
        compare_config(items, f'stage3/{name}', name, c, recs.get(name))
        ref = norm_config(recs.get(name), name)
        if ref != MISSING and ref['Abar_ref'] != MISSING:
            compare(items, f'stage3/{name}/G_Abar', mine['G']['Abar'], ref['Abar_ref'])
        if c['status'] == 'ok':
            expected_g = {a: seed_keys(v) for a, v in mine['G']['A'].items()}
            theirs_g = {a: recs.get(name, {}).get('effects', {}).get(a, {}).get('A', {}).get(G_ARM, MISSING)
                        for a in ASPECTS}
            compare(items, f'stage3/{name}/G_A', expected_g, theirs_g)
    compare(items, 'stage3/inputs/runs', mine.get('inputs', {}).get('runs', {}),
            verdict.get('inputs', {}).get('runs', MISSING))
    if stage2 is not None:
        add_item(items, 'stage3/inputs/stage2_sha256', stage2['sha256'], verdict.get('inputs', {}).get('stage2_sha256', MISSING))
        add_item(items, 'stage3/stage2_top', mine['top'], stage2['top'])


def compare_dev_arm(items, path, name, arm, final, theirs):
    """One removal of ablation/dev.json: results[x] = {arm, status, cause, aspects, M}; aspects[a] = {effect, lo, hi,
    arm_ami, final_ami, arm_seed_ami, final_seed_ami}; M = {M, lo, hi}. A failed arm has aspects and M null."""
    if not isinstance(theirs, dict):
        add_item(items, path, 'record', MISSING, same=False)
        return
    add_item(items, f'{path}/arm', name, theirs.get('arm', MISSING))
    add_item(items, f'{path}/status', arm['status'], theirs.get('status', MISSING))
    if arm['status'] != 'ok':
        compare(items, f'{path}/cause', arm['cause'], theirs.get('cause'))
        add_item(items, f'{path}/aspects', None, theirs.get('aspects', MISSING))
        add_item(items, f'{path}/M', None, theirs.get('M', MISSING))
        return
    aspects = {a: _drop_none(dict(effect=arm['e'][a]['e'], lo=arm['e'][a]['lo'], hi=arm['e'][a]['hi'],
                                  arm_ami=arm['Abar'][a], final_ami=final['Abar'][a],
                                  arm_seed_ami=seed_keys(arm['A'][a]), final_seed_ami=seed_keys(final['A'][a])))
               for a in ASPECTS}
    compare(items, f'{path}/aspects', aspects, theirs.get('aspects', MISSING))
    compare(items, f'{path}/M', _drop_none(arm['M']), theirs.get('M', MISSING))


def compare_ablation(items, mine, verdict):
    """mine = ablation_core's record; verdict = ablation/dev.json (ticket 10's layout; `written`, `host`, the code
    fields and `draws` (prose) are not compared). The removals compared are the file's `removals`."""
    final = verdict.get('final', {})
    add_item(items, 'ablation/final/name', mine['final'], final.get('name', MISSING) if isinstance(final, dict) else MISSING)
    add_item(items, 'ablation/final_key', G_ARM if mine['final'] == G_NAME else mine['final'], verdict.get('final_key', MISSING))
    compare(items, 'ablation/ranges', R_A, verdict.get('ranges', MISSING))
    add_item(items, 'ablation/seeds', list(DEV_SEEDS), verdict.get('seeds', MISSING))
    add_item(items, 'ablation/draw_mode', 'joint', verdict.get('draw_mode', MISSING))   # §3.5: §3.3's joint bootstrap
    if mine['n_draws']:
        add_item(items, 'ablation/n_draws', mine['n_draws'], verdict.get('n_draws', MISSING))
    sets_equal(items, 'ablation/removals', mine['removal_arms'], verdict.get('removals', MISSING))
    recs = verdict.get('results', {})
    sets_equal(items, 'ablation/results/names', mine['removal_arms'], recs)
    for x, name in mine['removal_arms'].items():
        compare_dev_arm(items, f'ablation/{x}', name, mine['arms'][name], mine['F'], recs.get(x))
    sets_equal(items, 'ablation/failed', [f'{x}:{n}' for x, n in mine['removal_arms'].items()
                                          if mine['arms'][n]['status'] == 'failed'],
               [f'{f["removal"]}:{f["arm"]}' for f in verdict['failed']] if 'failed' in verdict else MISSING)
    compare(items, 'ablation/inputs', mine['inputs'], {k: verdict.get('inputs', {}).get(k, MISSING) for k in mine['inputs']})


def stage3(outputs, sweep_machine, ctx=None, rec=None, g_partitions='frozen', **kw):
    """V3c: TOP from my own stage 2 derivation (my A values of part (a) on the machine of stages 1 and 2), then
    `stage3_core`, then the comparison with stage3/verdict.json. Writes stage3/rederive_stage3.json."""
    rec = rec or Recorder()
    ctx = ctx or load_context(rec)
    assert kw.get('n_draws', N_DRAWS) > 0, 'the qualifiers need the intervals: --draws must be positive'
    part = rec.json(score_dir(outputs, sweep_machine) / 'rederive_partitions.json', ('pass', 'runs'))
    assert part['pass'] is True, 'part (a) did not pass or did not run: fix it first (rule §4.2 V1c)'
    d2 = derive_stage2(gated_sweep_runs(rec, outputs, runs_from(part['runs'])))
    assert d2['top'], 'TOP-NONE: stage 3 does not run'
    s2_path = Path(outputs) / 'sweep' / sweep_machine / 'stage2.json'
    s2 = rec.json(s2_path, ('top',))
    stage2 = dict(sha256=rec.hashes[str(s2_path)], top=[r['name'] if isinstance(r, dict) else r for r in s2['top']])
    if g_partitions == 'frozen':
        g_partitions = paths.RG_STAGE_B / 'partitions' / G_ARM
    mine = stage3_core(outputs, d2['top'], ctx, rec, g_partitions=g_partitions, **kw)
    items = mine.pop('items')
    del mine['items_pass']
    path = Path(outputs) / 'stage3' / 'verdict.json'
    verdict = rec.json(path)
    add_item(items, 'stage3/stage2_machine', sweep_machine, verdict.get('stage2_machine', MISSING))
    compare_stage3(items, mine, verdict, stage2)
    rep = report('V3c stage 3', items, rec, ctx, mine=mine, sweep_machine=sweep_machine)
    write_json(path.parent / 'rederive_stage3.json', rep)
    return rep


def development_final(outputs, rec):
    """§4.2, §4.3, §2.5: F is the stage 3 pick, else G only after every sweep has ended without TOP."""
    root = Path(outputs)
    path = root / 'stage3/verdict.json'
    if path.exists():
        verdict = rec.json(path, ('outcome', 'pick'))
        assert verdict['outcome'] in ('PICK', 'KEEP-G'), f'{path}: unknown outcome'
        return verdict
    ended = []
    for first in sorted((root / 'sweep').glob('*/stage1.json')):
        s1 = rec.json(first, ('outcome',))
        second = first.with_name('stage2.json')
        s2 = rec.json(second, ('outcome',)) if second.exists() else None
        outcome = s1['outcome'] if s1['outcome'] == 'SWEEP-NONE' else s2['outcome'] if s2 else None
        assert outcome in ('SWEEP-NONE', 'TOP-NONE'), f'{first.parent.name}: missing stage 3 verdict after {outcome}'
        ended.append(outcome)
    assert ended, 'missing stage 3 verdict and no completed machine sweep'
    return dict(outcome=None, pick=None, terminal_outcomes=sorted(set(ended)))


def ablation(outputs, final, ctx=None, rec=None, **kw):
    """The development ablation: ablation_core against ablation/dev.json, for the removals the file lists (all eight
    unless it is a smoke); `final` must be the stage 3 pick, or G after KEEP-G / SWEEP-NONE / TOP-NONE. `n_draws` -1 takes the file's.
    Writes ablation/rederive_dev.json."""
    rec = rec or Recorder()
    ctx = ctx or load_context(rec)
    path = Path(outputs) / 'ablation' / 'dev.json'
    theirs = rec.json(path, ('removals', 'smoke', 'n_draws'))
    assert set(theirs['removals']) <= set(REMOVALS) and (theirs['smoke'] or set(theirs['removals']) == set(REMOVALS)), \
        f'removals {theirs["removals"]}: a real run covers the rule\'s eight'
    if kw.get('n_draws') == -1:
        kw['n_draws'] = theirs['n_draws']
    s3 = development_final(outputs, rec)
    mine = ablation_core(outputs, final, ctx, rec, removals=theirs['removals'], **kw)
    items = mine.pop('items')
    add_item(items, 'ablation/final_is_stage3_final', final, s3['pick'] if s3['outcome'] == 'PICK' else G_NAME)
    final_meta = theirs.get('final', {})
    if 'terminal_outcomes' in s3:
        outcome = final_meta.get('outcome', MISSING)
        add_item(items, 'ablation/final/outcome', s3['terminal_outcomes'], outcome,
                 same=outcome in s3['terminal_outcomes'])
        add_item(items, 'ablation/final/pick', None, final_meta.get('pick', MISSING))
    else:
        add_item(items, 'ablation/final/outcome_and_pick', [s3['outcome'], s3['pick']],
                 [final_meta.get('outcome', MISSING), final_meta.get('pick', MISSING)])
    add_item(items, 'ablation/p_sha256', ctx.p_sha, theirs.get('p_sha256', MISSING))
    compare_ablation(items, mine, theirs)
    rep = report('development ablation', items, rec, ctx, mine=mine)
    write_json(path.parent / 'rederive_dev.json', rep)
    return rep


# ---- self-check on 12-27's frozen stage A and stage B files ---------------------------------------------------------

def frozen_folder(arm, seed):
    """12-27's run folder of an arm, else 12-25's (GR-8 and R6-5 live there)."""
    for root in (paths.RG_TRAIN, paths.EG_OUT / 'stage_iii' / 'train'):
        if (root / arm / f's{seed}' / 'run.json').exists():
            return root / arm / f's{seed}'
    raise AssertionError(f'no frozen run folder for {arm} s{seed}')


def frozen_export(arm, seed, aspect):
    if arm == 'R6-5' and seed == 0:                 # 12-24's reference run (RB part 1), named by seed
        return paths.RB_OUT / 'part1' / 'train' / arm / f'{aspect}__s0__P.npz'
    return frozen_folder(arm, seed) / f'{aspect}__P.npz'


def frozen_parts(rec, ctx, arm, seed, threads):
    out = {}
    for a in ASPECTS:
        x, _ = read_export(rec, frozen_export(arm, seed, a), ctx)
        out[a] = partitions(x, threads)
    return out


def self_check(outputs, ctx=None, rec=None, all_arms=False, n_draws=0, workers=8, threads=THREADS):
    """The same code on 12-27's frozen files. Stage A (seed 0): every partition equals RG_STAGE_A/partitions and every
    A_a equals RG_STAGE_A/verdict.json's. Stage B (seeds 1 to 3, GS-0.02 against R6-5): partitions equal
    RG_STAGE_B/partitions, A-bar and e-bar equal RG_STAGE_B/verdict.json's, and with n_draws > 0 the intervals do too
    (per-aspect draws for all three aspects; the joint draws for emotion and style, rule §3.3)."""
    rec = rec or Recorder()
    ctx = ctx or load_context(rec)
    va = rec.json(paths.RG_STAGE_A / 'verdict.json', ('results', 'base_A'))
    vb = rec.json(paths.RG_STAGE_B / 'verdict.json', ('results', 'reference'))
    ref = vb['reference']
    arms_a = ([ref] + sorted(va['results'])) if all_arms else [ref, G_ARM]
    arms_b = sorted(vb['results']) if all_arms else [G_ARM]
    items = {}
    for arm in arms_a:
        parts = frozen_parts(rec, ctx, arm, 0, threads)
        for a in ASPECTS:
            expected = va['base_A'][a] if arm == ref else va['results'][arm]['effects'][a]['A']
            add_item(items, f'stage_a/{arm}/A/{a}', a_value(ctx, a, parts[a]), expected)
            compare_partition_file(items, rec, paths.RG_STAGE_A / 'partitions' / arm / 's0' / f'{a}__partitions.npz',
                                   f'stage_a/{arm}/partitions/{a}', parts[a])
    arms = {}
    for arm in [ref] + arms_b:
        arms[arm] = {a: {} for a in ASPECTS}
        for s in DEV_SEEDS:
            for a, p in frozen_parts(rec, ctx, arm, s, threads).items():
                arms[arm][a][s] = p
                compare_partition_file(items, rec, paths.RG_STAGE_B / 'partitions' / arm / f's{s}' / f'{a}__partitions.npz',
                                       f'stage_b/{arm}/s{s}/partitions/{a}', p)
    modes = [('aspect', ASPECTS), ('joint', ('emotion', 'style'))] if n_draws else [('points', ASPECTS)]
    for mode, aspects in modes:
        eff = effects(ctx.labels, ref, arms, n_draws if mode != 'points' else 0, workers,
                      mode='aspect' if mode == 'points' else mode)
        for arm in arms_b:
            for a in aspects:
                exp = vb['results'][arm]['effects'][a]
                add_item(items, f'stage_b/{mode}/{arm}/e/{a}/e', eff[arm]['e'][a]['e'], exp['e'])
                if mode == 'points':
                    for name in (arm, ref):
                        add_item(items, f'stage_b/{arm}/Abar/{a}/{name}', eff[name]['Abar'][a], exp['Abar'][name])
                        for s in DEV_SEEDS:
                            add_item(items, f'stage_b/{arm}/A/{a}/{name}/s{s}', eff[name]['A'][a][s], exp['A'][name][str(s)])
                else:
                    for bound in ('lo', 'hi'):
                        add_item(items, f'stage_b/{mode}/{arm}/e/{a}/{bound}', eff[arm]['e'][a][bound], exp[bound])
    rep = report('self-check', items, rec, ctx, n_draws=n_draws, arms_a=arms_a, arms_b=arms_b)
    write_json(Path(outputs) / 'rederive_selfcheck.json', rep)
    return rep


# ---- command line -------------------------------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='mode', required=True)
    for mode in ('partitions', 'sweep', 'stage3', 'ablation', 'self-check'):
        p = sub.add_parser(mode)
        p.add_argument('--outputs', type=Path, default=Path('/tmp/claude-0/r10-rs-12') if mode == 'self-check'
                       else paths.OUTPUTS)
        if mode in ('partitions', 'sweep'):
            p.add_argument('--machine', choices=('local', 'node402'), required=True)
        if mode == 'partitions':
            p.add_argument('--runs', nargs='+', help='only these runs (a partial check: the unscored files are not listed)')
            p.add_argument('--threads', type=int, default=THREADS)
        if mode == 'stage3':
            p.add_argument('--sweep-machine', choices=('local', 'node402'), required=True,
                           help='the machine whose A values chose TOP')
        if mode == 'ablation':
            p.add_argument('--final', required=True, help="the final model F: 'G' or the stage 3 pick's name")
        if mode in ('stage3', 'ablation', 'self-check'):
            p.add_argument('--workers', type=int, choices=range(1, 9), default=8)
        if mode in ('stage3', 'ablation', 'self-check'):
            p.add_argument('--draws', type=int, default={'stage3': N_DRAWS, 'ablation': 0, 'self-check': 0}[mode],
                           help="bootstrap draws (0 = point values only; ablation: -1 = the dev.json's own count)")
        if mode == 'self-check':
            p.add_argument('--all-arms', action='store_true', help="every arm of 12-27's stage A and B (slower)")
    args = parser.parse_args(argv)
    paths.check_outputs(args.outputs)
    paths.check_imports()
    if args.mode == 'partitions':
        ctx = load_context(Recorder())
        reports = [part_a(args.outputs, args.machine, ctx, only=args.runs, threads=args.threads)]
    elif args.mode == 'sweep':
        reports = sweep(args.outputs, args.machine)
    elif args.mode == 'stage3':
        reports = [stage3(args.outputs, args.sweep_machine, n_draws=args.draws, workers=args.workers)]
    elif args.mode == 'ablation':
        reports = [ablation(args.outputs, args.final, n_draws=args.draws, workers=args.workers)]
    else:
        reports = [self_check(args.outputs, all_arms=args.all_arms, n_draws=args.draws, workers=args.workers)]
    for rep in reports:
        bad = [k for k, i in rep['items'].items() if not i['pass']]
        print(f'{rep["check"]}: pass={rep["pass"]}; {len(rep["items"])} items, {len(bad)} differ; first: {bad[:5]}')
    return 0 if all(r['pass'] for r in reports) else 1


if __name__ == '__main__':
    sys.exit(main())
