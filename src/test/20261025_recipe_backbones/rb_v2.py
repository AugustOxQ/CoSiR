"""V2 (rule §4.2, §4.3; ticket 15): the part 2 statistic and the ArtELingo-28 d re-derived from the rules' text alone.

Written without the round's scorers: the CT rule §3 and the RB rule §3.2, §3.3 are the only formulas. Given the held
list, the annotations, the saved outputs and the CT's R-COL records, it recomputes per candidate the labels, the
partitions, the eligibility, every entry's A_e(I0), d and the interval, then compares them with
`part2/verdict_stage{3,4,5}.json` and the saved partitions and writes `part2/rederive_v2_stage{3,4,5}.json`.

CLI (from the round folder):
  python rb_v2.py --stage 3|4|5 [--candidates K2 K3 ...] [--scale K2=full ...] [--outputs DIR] [--smoke] [--workers 8]
  python rb_v2.py --ct-check 1|2     # the acceptance run: the CT's own verdict_stage{1,2}.json from its saved outputs
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

import rb_paths

TOL = 1e-6
SEEDS = (0, 1, 2, 3, 4)
DRAWS = 2000
EMOTIONS = ('amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness', 'something else')
ANNOTATIONS = Path('/data/PDD/artelingo/artelingo_train.json')
A28_PARQUET = rb_paths.PRE / 'artelingo28/read_set_a28.parquet'
A28_CSV = Path('/data/PDD/artelingo/ArtELingo/semeval-artelingo28/artelingo28_public_noworkerid_split.csv')
PAPER_VERSION = 'clip_roberta'          # the Stage 1 version of B2 to B5 in the R-COL records
# the CT's comparators: folder under held/percept, arm P, R-COL cut (None: always eligible)
B_ARMS = {'B1': ('core__k40', 'P3', None), 'B2': ('paper__strict', 'P1', 'strict'),
          'B3': ('paper__strict', 'P4', 'strict'), 'B4': ('paper__k40', 'P1', 'k40'),
          'B5': ('paper__k40', 'P4', 'k40'), 'B6': ('core__k40', 'P2', None)}
MATCHED = {'K2': 'M2', 'K3': 'M3', 'K5': 'M5'}
SCALE_OF_STAGE = {1: 'sample', 2: 'full', 3: 'sample', 4: 'full'}   # stages 1, 2: the CT's own
THREADS = 1


def sha256_names(names):
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


# ---- labels (rule §3.2, §3.3) -----------------------------------------------------------------------------

def _plurality(counts):
    """(code, is_tie): the largest count, ties to the smallest code."""
    top = counts.max()
    return int(np.argmax(counts)), int((counts == top).sum()) > 1


def held_emotion_labels(names, annotations):
    """Plurality of each painting's caption emotion codes (codes = sorted names, 'something else' included)."""
    names = list(names)
    assert names == sorted(names), 'names must be in sorted() order'
    assert {r['emotion'] for r in annotations} == set(EMOTIONS), 'emotion names differ from the nine'
    code = {e: i for i, e in enumerate(EMOTIONS)}
    counts = {n: np.zeros(len(EMOTIONS), dtype=np.int64) for n in names}
    for r in annotations:
        if r['painting'] in counts:
            counts[r['painting']][code[r['emotion']]] += 1
    assert all(c.sum() > 0 for c in counts.values()), 'a painting without a caption'
    return np.asarray([_plurality(counts[n])[0] for n in names], dtype=np.int64)


def a28_emotion_labels(parquet, csv):
    """(names, labels, info): read-set rows joined by row_id (= CSV data-row position) to the CSV strings."""
    rows = csv.iloc[parquet['row_id'].to_numpy()]
    assert (rows['painting'].to_numpy() == parquet['painting'].to_numpy()).all(), 'row_id joins another painting'
    emotion = rows['emotion'].replace({'other': 'something else'}).to_numpy()
    assert set(emotion) <= set(EMOTIONS), set(emotion) - set(EMOTIONS)
    names = sorted(parquet['painting'].unique())
    code = {e: i for i, e in enumerate(EMOTIONS)}
    counts = {n: np.zeros(len(EMOTIONS), dtype=np.int64) for n in names}
    for p, e in zip(parquet['painting'].to_numpy(), emotion):
        counts[p][code[e]] += 1
    picked = [_plurality(counts[n]) for n in names]
    ties = sum(t for _, t in picked)
    info = dict(paintings=len(names), ties=ties, unique=len(names) - ties)
    return names, np.asarray([c for c, _ in picked], dtype=np.int64), info


def load_a28_tables():
    return pd.read_parquet(A28_PARQUET), pd.read_csv(A28_CSV)


def label_counts(labels):
    return {str(k): int(v) for k, v in enumerate(np.bincount(labels, minlength=len(EMOTIONS)))}


# ---- R-COL eligibility (CT rule §4) -----------------------------------------------------------------------

def _collapsed(run, cut):
    if cut == 'strict':
        return run['kept_count'] < 10 or run['kept_count'] > 90 or run['share'] > 0.5
    return run['share'] > 0.5


def rcol_eligible_seeds(runs, cut, nonfinite=(), version=PAPER_VERSION):
    """Seeds of one paper settings cut that are not collapsed; seeds with non-finite held outputs count as collapsed."""
    chosen = [r for r in runs if r['version'] == version and r['cut'] == cut]
    seeds = [r['seed'] for r in chosen]
    assert len(set(seeds)) == len(seeds), 'two R-COL runs for one seed'
    return sorted(r['seed'] for r in chosen if not _collapsed(r, cut) and r['seed'] not in set(nonfinite))


def rcol_notes(rcol):
    """Disagreements of the record with the rule's text: a run's `collapsed` flag, an entry's seed list."""
    notes = []
    for r in rcol['runs']:
        if r['version'] == PAPER_VERSION and r.get('collapsed') is not None \
                and bool(r['collapsed']) != bool(_collapsed(r, r['cut'])):
            notes.append(f"run {r['version']} s{r['seed']} {r['cut']}: record collapsed={r['collapsed']}, rule says "
                         f"{_collapsed(r, r['cut'])}")
    for arm, (_, _, cut) in B_ARMS.items():
        if cut is not None:
            want = rcol_eligible_seeds(rcol['runs'], cut)
            if rcol['entries'][arm]['seeds'] != want:
                notes.append(f"{arm}: record seeds {rcol['entries'][arm]['seeds']}, rule says {want}")
    return notes


# ---- partitions (CT rule §3) ------------------------------------------------------------------------------

def own_partition(prob):
    """Argmax over the K probabilities, lowest index on a tie: int64 [1, n]."""
    return np.argmax(np.asarray(prob), axis=1).astype(np.int64)[None, :]


def l2_probabilities(prob):
    """The K sigmoid probabilities in float32 divided by their float32 row norm."""
    p = np.asarray(prob, dtype=np.float32)
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    assert (norm > 0).all(), 'a zero probability row'
    return p / norm


def r20_partition(x):
    """MiniBatchKMeans k = 20, random states 0, 1, 2, n_init 3, batch 4096, on float32: int64 [3, n].

    The input is cast to float32 and its unit norm checked (|norm - 1| < 1e-3), not repaired."""
    x = np.asarray(x).astype(np.float32)
    assert np.isfinite(x).all()
    assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < 1e-3, 'rows are not unit norm'
    out = []
    with threadpool_limits(limits=THREADS):
        for state in (0, 1, 2):
            km = MiniBatchKMeans(n_clusters=20, random_state=state, n_init=3, batch_size=4096)
            out.append(km.fit(x).labels_)
    return np.stack(out).astype(np.int64)


@dataclass
class Arm:
    """name and its fixed partitions: parts[seed] = {'r20': int [3, n], 'own': int [1, n]} ('own' for entries only)."""
    name: str
    parts: dict


# ---- the draws and the statistic (CT rule §3, RB rule §3.2) -----------------------------------------------

def draw_plan(n, arm_seeds, draws):
    """Per draw I_b = rng.integers(0, n, size=n), then S_b = rng.choice(seeds, size=len(seeds), replace=True) for each
    arm with eligible seeds in the dict's order; arms without seeds draw nothing."""
    rng = np.random.default_rng(0)
    idx = np.empty((draws, n), dtype=np.int64)
    seeds = []
    for b in range(draws):
        idx[b] = rng.integers(0, n, size=n)
        drawn = {}
        for name, pool in arm_seeds.items():
            if len(pool):
                drawn[name] = rng.choice(pool, size=len(pool), replace=True)
        seeds.append(drawn)
    return idx, seeds


_STATE = {}


def _ami_mean(labels, partitions, I):
    lab = labels[I]
    return float(np.mean([adjusted_mutual_info_score(lab, p[I]) for p in partitions]))


def _boot_chunk(span):
    """d_b for draws lo..hi: the maximum over entries is taken again in each draw."""
    st = _STATE
    out = []
    with threadpool_limits(limits=1):
        for b in range(*span):
            I, S = st['idx'][b], st['seeds'][b]
            cache = {}

            def value(name, seed, reading):
                key = (name, int(seed), reading)
                if key not in cache:
                    cache[key] = _ami_mean(st['labels'], st['parts'][name][int(seed)][reading], I)
                return cache[key]

            a_cand = float(np.mean([value(st['cand'], s, 'r20') for s in S[st['cand']]]))
            best = max(float(np.mean([value(e, s, r) for s in S[e]])) for e in st['entries'] for r in ('r20', 'own'))
            out.append(a_cand - best)
    return out


def rederive(labels, cand, entries, draws=DRAWS, workers=8):
    """A, A_e(I0), d and the 95% interval of the paired bootstrap, from fixed partitions.

    cand: Arm (r20 only); entries: Arms in the rule's order, each with r20 and own; an entry without seeds is left out."""
    labels = np.asarray(labels, dtype=np.int64)
    assert labels.min() >= 0, 'masked labels are not handled'
    n = len(labels)
    arms = [cand] + list(entries)
    arm_seeds = {a.name: sorted(a.parts) for a in arms}
    assert arm_seeds[cand.name], 'the candidate has no seeds'
    live = [a.name for a in entries if arm_seeds[a.name]]
    parts = {a.name: {s: {r: np.asarray(p) for r, p in a.parts[s].items()} for s in a.parts} for a in arms}
    for a in arms:
        for s in a.parts:
            assert parts[a.name][s]['r20'].shape == (3, n), (a.name, s)
            if a.name != cand.name:
                assert parts[a.name][s]['own'].shape == (1, n), (a.name, s)

    I0 = np.arange(n)
    per_seed = {a.name: {str(s): {r: [float(adjusted_mutual_info_score(labels, p)) for p in parts[a.name][s][r]]
                                  for r in (('r20',) if a.name == cand.name else ('r20', 'own'))}
                         for s in sorted(a.parts)} for a in arms if arm_seeds[a.name]}

    def mean_over(name, reading):
        return float(np.mean([np.mean(per_seed[name][str(s)][reading]) for s in arm_seeds[name]]))

    a_cand = mean_over(cand.name, 'r20')
    a_e = {f'{e}-{r}': mean_over(e, r) for e in live for r in ('r20', 'own')}
    d = a_cand - max(a_e.values())

    idx, seeds = draw_plan(n, arm_seeds, draws)
    _STATE.clear()
    _STATE.update(labels=labels, parts=parts, idx=idx, seeds=seeds, cand=cand.name, entries=live)
    try:
        step = max(1, -(-draws // (max(workers, 1) * 4)))
        spans = [(lo, min(lo + step, draws)) for lo in range(0, draws, step)]
        if workers <= 1:
            pieces = [_boot_chunk(sp) for sp in spans]
        else:
            with mp.get_context('fork').Pool(workers) as pool:
                pieces = pool.map(_boot_chunk, spans)
    finally:
        _STATE.clear()
    d_b = [float(v) for piece in pieces for v in piece]
    assert len(d_b) == draws
    lo, hi = np.percentile(d_b, [2.5, 97.5])
    return dict(A_cand=a_cand, A_e=a_e, per_seed=per_seed, d=float(d), interval=[float(lo), float(hi)],
                bootstrap_d=d_b, eligible_seeds={a.name: arm_seeds[a.name] for a in arms})


# ---- reading the saved outputs ----------------------------------------------------------------------------

@dataclass
class Roots:
    """Where things are. `ct[scale]` is the CT output root of that scale; `rcol[scale]` its R-COL record."""
    outputs: Path
    ct: dict
    held_sha: str
    rcol: dict
    annotations: object = ANNOTATIONS     # a path or the list of annotation rows
    a28: tuple = None                     # (parquet, csv) tables; None: the real ones
    a28_paintings: int = 746

    @classmethod
    def default(cls, outputs=None):
        return cls(outputs=Path(outputs or rb_paths.OUTPUTS), ct={'sample': rb_paths.CT_OUT, 'full': rb_paths.CT_FULL},
                   held_sha=rb_paths.HELD_SHA256,
                   rcol={'sample': rb_paths.CT_OUT / 'percept/rcol_stage1.json',
                         'full': rb_paths.CT_FULL / 'percept/rcol.json'})

    def annotation_rows(self):
        if isinstance(self.annotations, (str, Path)):
            return json.loads(Path(self.annotations).read_text())
        return self.annotations


def _sha_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _partition_task(task):
    """Load one output, check its sidecar, and return its fixed partitions (run in a worker)."""
    path = Path(task['path'])
    side = json.loads(path.with_suffix('.json').read_text())
    arr = np.load(path)[task['key']]
    assert arr.shape[0] == task['n'], (path, arr.shape)
    if task['side'] == 'ct':
        assert side['held_sha256'] == task['list_sha'] and side['rows'] == task['n'], path
    else:
        want = dict(list_sha256=task['list_sha'], rows=task['n'], width=arr.shape[1], dtype=str(arr.dtype),
                    scale=task['scale'], arm=task['arm'], seed=task['seed'], aspect=task['aspect'])
        for k, v in want.items():
            assert side[k] == v, (path, k, side[k], v)
        assert side['sha256'] == _sha_array(arr), (path, 'sha256 of the saved array')
    if task['kind'] == 'student':
        assert arr.dtype == np.float16 and arr.shape[1] == task['width'], path
        assert np.isfinite(arr).all(), path
        return dict(finite=True, parts={'r20': r20_partition(arr)})
    assert arr.dtype == np.float32 and arr.ndim == 2, path
    if not np.isfinite(arr).all():
        assert task['droppable'], f'non-finite rows in {path}'
        return dict(finite=False, parts=None)
    return dict(finite=True, parts={'r20': r20_partition(l2_probabilities(arr)), 'own': own_partition(arr)})


def arm_plans(cand, mode, scale, roots):
    """The arms of one candidate in the rule's order: cand, B1..B6 (or B1, B6 on ArtELingo-28), the matched arm."""
    ct = Path(roots.ct[scale])
    new_dir = Path(roots.outputs) / 'part2' / scale / mode
    plans = []

    def plan(name, kind, rel, key, side, droppable=False, cut=None, aspect='percept', width=None, folder=None):
        plans.append(dict(name=name, kind=kind, rel=rel, key=key, side=side, droppable=droppable, cut=cut,
                          aspect=aspect, width=width, folder=folder))

    if cand == 'K1' and mode == 'held':
        plan('K1', 'student', 'd2/emotion__s{s}', 'img', 'ct', aspect='emotion', width=2048, folder='K1')
    else:
        plan(cand, 'student', f'{cand}/emotion__s{{s}}', 'img', 'new', aspect='emotion', width=2048, folder=cand)
    for b, (folder, p, cut) in B_ARMS.items():
        if mode == 'held':
            plan(b, 'prob', f'percept/{folder}/{p}__s{{s}}', 'prob', 'ct', droppable=cut is not None, cut=cut,
                 folder=b)
        elif b in ('B1', 'B6'):
            plan(b, 'prob', f'{b}/{p}__s{{s}}', 'prob', 'new', folder=b)
    if cand in MATCHED:
        plan(MATCHED[cand], 'prob', f'{MATCHED[cand]}/P3__s{{s}}', 'prob', 'new', folder=MATCHED[cand])
    for p in plans:
        base = (ct / 'held') if p['side'] == 'ct' else new_dir
        p['tasks'] = {}
        p['base'] = base
    return plans


def saved_partition_paths(roots, scale, mode, plan, seed):
    """Where the saved partitions of (arm, seed) may be: the new round's folder and the CT's own (CT arms)."""
    tag = scale if mode == 'held' else f'a28_{scale}'
    name = 'emotion' if plan['kind'] == 'student' else 'percept'
    found = {}
    for reading in ('r20', 'own'):
        f = Path(roots.outputs) / 'part2/partitions' / tag / plan['name'] / f'{name}__s{seed}__{reading}.npz'
        if f.exists():
            found[(reading, 'new')] = (f, 'labels')
    if mode == 'held' and plan['side'] == 'ct':
        f = Path(roots.ct[scale]) / 'held/partitions' / f"{plan['folder']}__s{seed}.npz"
        if f.exists():
            for reading in ('r20', 'own'):
                if reading in np.load(f).files:
                    found[(reading, 'ct')] = (f, reading)
    return found


def run_candidate(cand, stage, roots, draws=DRAWS, workers=8, scale=None, seeds=SEEDS, drop=()):
    """Re-derive one candidate (or model, stage 5): labels, partitions, eligibility, A_e, d, the interval.
    Stages 1, 2 are the CT's own (K1 with its arm list: drop=('B6',))."""
    mode = 'a28' if stage == 5 else 'held'
    scale = scale or SCALE_OF_STAGE.get(stage)
    assert scale in ('sample', 'full'), scale
    ct = Path(roots.ct[scale])
    if mode == 'held':
        held = json.loads((ct / 'held/held_paintings.json').read_text())
        names = held['names']
        assert names == sorted(names) and sha256_names(names) == held['sha256'] == roots.held_sha, 'held list'
        labels = held_emotion_labels(names, roots.annotation_rows())
        info = None
    else:
        parquet, csv = roots.a28 or load_a28_tables()
        names, labels, info = a28_emotion_labels(parquet, csv)
        assert len(names) == roots.a28_paintings, len(names)
    n, list_sha = len(names), sha256_names(names)
    assert len(labels) == n

    plans = [p for p in arm_plans(cand, mode, scale, roots) if p['name'] not in drop]
    rcol = json.loads(Path(roots.rcol[scale]).read_text()) if mode == 'held' else None
    notes = rcol_notes(rcol) if rcol else []
    tasks = []
    for p in plans:
        use = list(seeds)
        if p['cut'] is not None:
            use = [s for s in use if s in rcol_eligible_seeds(rcol['runs'], p['cut'])]
        for s in use:
            t = dict(path=str(p['base'] / (p['rel'].format(s=s) + '.npz')), key=p['key'], side=p['side'],
                     kind=p['kind'], droppable=p['droppable'], n=n, list_sha=list_sha, scale=scale, arm=p['folder'],
                     seed=s, aspect=p['aspect'], width=p['width'])
            tasks.append((p['name'], s, t))
    if workers <= 1:
        outs = [_partition_task(t) for _, _, t in tasks]
    else:
        with mp.get_context('fork').Pool(workers) as pool:
            outs = pool.map(_partition_task, [t for _, _, t in tasks], chunksize=1)
    parts = {p['name']: {} for p in plans}
    dropped = {p['name']: [] for p in plans}
    for (name, s, _), out in zip(tasks, outs):
        if out['finite']:
            parts[name][s] = out['parts']
        else:
            dropped[name].append(s)

    check = dict(checked=0, unchecked=0, mismatches=[])
    for p in plans:
        for s, pt in parts[p['name']].items():
            saved = saved_partition_paths(roots, scale, mode, p, s)
            for reading, arr in pt.items():
                files = [(src, f, k) for (r, src), (f, k) in saved.items() if r == reading]
                if not files:
                    check['unchecked'] += 1
                for src, f, k in files:
                    check['checked'] += 1
                    got = np.load(f)[k]
                    if got.shape != arr.shape or not np.array_equal(got, arr):
                        check['mismatches'].append(f"{p['name']} s{s} {reading}: differs from {f}")

    arms = [Arm(p['name'], parts[p['name']]) for p in plans]
    res = rederive(labels, arms[0], arms[1:], draws=draws, workers=workers)
    res.update(candidate=cand, stage=stage, scale=scale, n=n, list_sha256=list_sha, label_counts=label_counts(labels),
               label_info=info, eligibility_notes=notes, nonfinite_dropped={k: v for k, v in dropped.items() if v},
               partitions=check, draws=draws, seeds=list(seeds), _partitions=parts)
    return res


def rederive_ct(stage, workers=8):
    """The acceptance run: the CT's K1 and B1 to B5 (no B6, no matched arm) on its saved outputs at one scale."""
    return run_candidate('K1', stage, Roots.default(), workers=workers, drop=('B6',))


# ---- comparison with a verdict file -----------------------------------------------------------------------

def _first(block, keys, what):
    for k in keys:
        if k in block:
            return block[k]
    raise AssertionError(f'verdict block has none of {keys} ({what})')


def read_block(path, name):
    """The candidate's (or model's) block of a verdict file, normalised to the fields that are compared."""
    verdict = json.loads(Path(path).read_text())
    found = [v[name] for k in ('candidates', 'models', 'blocks') if isinstance(verdict.get(k), dict)
             for v in [verdict[k]] if name in v]
    if name in verdict and isinstance(verdict[name], dict):
        found.append(verdict[name])
    assert len(found) == 1, f'{path}: expected one block for {name}, found {len(found)}'
    b = found[0]
    elig = b.get('eligibility', b)
    return dict(d=_first(b, ('d', 'd_A28'), 'd'), A_cand=_first(b, (f'A_{name}', 'A_cand', 'A_model', 'A_K'), 'A'),
                A_e=_first(b, ('A_e',), 'A_e'), interval=_first(b, ('interval',), 'interval'),
                eligible_seeds=_first(elig, ('eligible_seeds',), 'eligible seeds'),
                bootstrap_d=b.get('bootstrap_d'))


def compare(result, block, tol=TOL):
    """Equal partitions and eligibility; d, each A_e, both bounds within tol (rule §4.2 V2)."""
    bad, diffs = [], []

    def num(label, mine, theirs):
        if theirs is None:
            bad.append(f'{label}: missing in the verdict')
            return
        diff = abs(float(mine) - float(theirs))
        diffs.append(diff)
        if not diff <= tol:
            bad.append(f'{label}: re-derived {mine!r}, verdict {theirs!r} (diff {diff:.3g})')

    num('d', result['d'], block['d'])
    num('A_cand', result['A_cand'], block['A_cand'])
    for k in sorted(set(result['A_e']) | set(block['A_e'])):
        num(f'A_e[{k}]', result['A_e'].get(k, float('nan')), block['A_e'].get(k))
    num('interval lower', result['interval'][0], block['interval'][0])
    num('interval upper', result['interval'][1], block['interval'][1])
    for arm in sorted(set(result['eligible_seeds']) | set(block['eligible_seeds'])):
        mine, theirs = result['eligible_seeds'].get(arm), block['eligible_seeds'].get(arm)
        if mine is None or theirs is None or [int(s) for s in mine] != [int(s) for s in theirs]:
            bad.append(f'eligible seeds {arm}: re-derived {mine}, verdict {theirs}')
    bad += [f'R-COL record: {m}' for m in result.get('eligibility_notes', [])]
    bad += [f'partitions: {m}' for m in result.get('partitions', {}).get('mismatches', [])]
    out = dict(agree=not bad, mismatches=bad, max_abs_diff=max(diffs) if diffs else None, tolerance=tol)
    if block.get('bootstrap_d') is not None and result.get('bootstrap_d') is not None \
            and len(block['bootstrap_d']) == len(result['bootstrap_d']):
        out['bootstrap_max_abs_diff'] = float(np.max(np.abs(np.asarray(block['bootstrap_d'])
                                                            - np.asarray(result['bootstrap_d']))))
    return out


# ---- stage runs and the CLI -------------------------------------------------------------------------------

def check_smoke_outputs(path):
    assert str(Path(path).resolve()).startswith('/tmp/claude-0/'), f'smoke needs an outputs path under /tmp/claude-0/: {path}'


def _blocks_in(path):
    verdict = json.loads(Path(path).read_text())
    names = set()
    for k in ('candidates', 'models', 'blocks'):
        if isinstance(verdict.get(k), dict):
            names |= set(verdict[k])
    names |= {k for k, v in verdict.items() if isinstance(v, dict) and len(k) == 2 and k[0] == 'K' and k[1].isdigit()}
    return sorted(names)


def run_stage(stage, roots, candidates=None, draws=DRAWS, workers=8, scales=None, smoke=False):
    """Re-derive every candidate of a stage, compare with part2/verdict_stage{stage}.json, write the re-derivation."""
    out_dir = Path(roots.outputs) / 'part2'
    verdict = out_dir / f'verdict_stage{stage}.json'
    assert verdict.exists(), f'{verdict} does not exist: V2 runs after the verdict file'
    candidates = list(candidates or _blocks_in(verdict))
    assert candidates, 'no candidate'
    if smoke:
        check_smoke_outputs(roots.outputs)
        draws = 20
    seeds = (0,) if smoke else SEEDS
    scales = dict(scales or {})
    written = {}
    stage4 = out_dir / 'verdict_stage4.json'
    for cand in candidates:
        scale = scales.get(cand)
        if stage == 5 and scale is None:
            scale = 'full' if cand == 'K1' or (stage4.exists() and cand in _blocks_in(stage4)) else 'sample'
        t0 = time.time()
        res = run_candidate(cand, stage, roots, draws=draws, workers=workers, scale=scale, seeds=seeds)
        res.pop('_partitions')
        res['comparison'] = compare(res, read_block(verdict, cand))
        res['seconds'] = round(time.time() - t0, 1)
        written[cand] = res
    agree = all(r['comparison']['agree'] for r in written.values())
    doc = dict(schema=1, stage=stage, smoke=smoke, agree=agree, tolerance=TOL, rule=str(rb_paths.RULE),
               verdict_file=str(verdict), versions=dict(sklearn=sklearn.__version__, numpy=np.__version__),
               code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), candidates=written)
    target = out_dir / f'rederive_v2_stage{stage}.json'
    target.write_text(json.dumps(doc))
    return dict(agree=agree, file=str(target), candidates={c: r['comparison']['mismatches'] for c, r in written.items()})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', type=int, choices=(3, 4, 5))
    ap.add_argument('--ct-check', type=int, choices=(1, 2))
    ap.add_argument('--candidates', nargs='*')
    ap.add_argument('--scale', nargs='*', default=[], help='stage 5 only: K2=full K3=sample ...')
    ap.add_argument('--outputs', default=str(rb_paths.OUTPUTS))
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args(argv)
    assert bool(args.stage) != bool(args.ct_check), 'give --stage or --ct-check'
    if args.ct_check:
        res = rederive_ct(args.ct_check, workers=args.workers)
        verdict = json.loads((rb_paths.MAIN_TEST / f'20261020_claim_test/verdict_stage{args.ct_check}.json').read_text())
        dif = max(abs(res['d'] - verdict['d']), *(abs(res['A_e'][k] - v) for k, v in verdict['A_e'].items()),
                  *(abs(a - b) for a, b in zip(res['interval'], verdict['interval'])))
        print(f"CT stage {args.ct_check}: d {res['d']:.6f} vs {verdict['d']:.6f}; max diff {dif:.2e}; partitions "
              f"checked {res['partitions']['checked']}, mismatches {len(res['partitions']['mismatches'])}")
        return 0 if dif < TOL else 1
    roots = Roots.default(args.outputs)
    summary = run_stage(args.stage, roots, candidates=args.candidates, workers=args.workers, smoke=args.smoke,
                        scales=dict(s.split('=') for s in args.scale))
    print(json.dumps(summary, indent=1))
    return 0 if (summary['agree'] or args.smoke) else 1


if __name__ == '__main__':
    sys.exit(main())
