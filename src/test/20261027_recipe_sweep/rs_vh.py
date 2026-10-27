"""rs_vh (ticket 11, rule C13): the held side of the r10 recipe sweep re-derived from the rule's text alone.

Written without this round's scorers (rs_stage0, rs_stage4, rs_ablation are not read). Given the rule, the RB and CT
rules, the held list H, the captions, the saved held outputs and the frozen eligibility records, it recomputes labels,
partitions, every A, d, both bounds and the outcomes, then compares with the verdict file. Low-level pieces (partitions,
draw order, labels, eligibility) are adapted from 12-24's `rb_v2.py` and 12-27's `rg_vc2.py`, each marked below.

  python rs_vh.py stage0                          # V0b: <outputs>/stage0/verdict.json, both scales
  python rs_vh.py stage4 --candidate NAME         # V4b: <outputs>/stage4/verdict.json
  python rs_vh.py ablation --final NAME           # rule 4.5: <outputs>/ablation/held.json
  python rs_vh.py selfcheck [--draws N]           # the three recomputations on frozen data (K4, G)

Each mode writes rederive_<check>.json beside the checked file and exits 1 on any disagreement (values within 1e-6).
"""
import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

import rs_paths
import rs_sheet

# ---- constants, each from the rule text -----------------------------------------------------------------------
TOL = 1e-6                                   # V0b, V4b, rule 4.5: values within 1e-6
DRAWS = 2000                                 # rule 3: every bootstrap has 2,000 draws
RANGES = dict(emotion=0.0046, style=0.0082, genre=0.0128)    # rule 3: r_a
ASPECTS = ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2, 3, 4)                      # rule 2.1, 2.4: five model seeds (both scales)
CLAIM_BAR = 0.01                             # rule 4.1: GO and MM need d >= 0.01
HELD_SHA256 = 'a0b1139c6056fd858ddbd2ee1dd4bafd1453e590818c6aa59883364c0288a876'   # CT rule 3, RG rule 3.2
N_HELD, WIDTH = 12281, 2048                  # rule 3.4, spec 3.3
N_GENRE_LABELLED = 9949                      # RG rule 3.2
K, STATES, N_INIT, BATCH = 20, (0, 1, 2), 3, 4096        # RB rule 3.1: MiniBatchKMeans partitions
UNIT_TOL = 1e-3                              # unit norm checked, |norm - 1| < 1e-3, not repaired
READINGS = ('r20', 'own')                    # CT rule 3: an entry's two readings
EMOTIONS = ('amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness',
            'something else')                # CT rule 3: nine names, codes sorted by name
# rb_v2.B_ARMS (CT rule 2, RB rule 2): comparator -> (folder under held/percept, arm file, R-COL cut or None)
B_ARMS = {'B1': ('core__k40', 'P3', None), 'B2': ('paper__strict', 'P1', 'strict'),
          'B3': ('paper__strict', 'P4', 'strict'), 'B4': ('paper__k40', 'P1', 'k40'),
          'B5': ('paper__k40', 'P4', 'k40'), 'B6': ('core__k40', 'P2', None)}
PAPER_VERSION = 'clip_roberta'               # rb_v2: the version of B2 to B5 in the R-COL records
REMOVALS = rs_sheet.REMOVALS                 # spec 3.2 table; names only
PROMPTS = dict(emotion='emotion_focus', style='style_word', genre='genre_word')   # ct_diag: T's stores, '<key>_held'

# ---- where the inputs are (frozen, read only) -----------------------------------------------------------------
HELD_LIST = rs_paths.CT_OUT / 'held/held_paintings.json'
ANNOTATIONS = Path('/data/PDD/artelingo/artelingo_train.json')
GENRE_CSVS = (Path('/data/SSD/wikiart_genre/genre_train.csv'), Path('/data/SSD/wikiart_genre/genre_val.csv'))
PROMPT_ROOT = rs_paths.PRE / 'artelingo_full/qwen3vl_emb_2b_aspect_prompts'
CT_ROOT = {'sample': rs_paths.CT_OUT, 'full': rs_paths.CT_FULL}
RCOL = {'sample': rs_paths.CT_OUT / 'percept/rcol_stage1.json', 'full': rs_paths.CT_FULL / 'percept/rcol.json'}
RB_K4_DIRS = {s: rs_paths.RB_PART2 / s / 'held/K4' for s in ('sample', 'full')}
RB_K4_PARTITIONS = {s: rs_paths.RB_PART2 / 'partitions' / s / 'K4' for s in ('sample', 'full')}
RB_T_PARTITIONS = rs_paths.RB_PART2 / 'partitions/full/T'
SELF_OUT = Path('/tmp/claude-0/r10-rs-11')


# ---- small helpers --------------------------------------------------------------------------------------------

def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def names_sha(names):
    return hashlib.sha256('\n'.join(names).encode('utf-8')).hexdigest()


def write_json(path, obj):
    """Atomic: write beside, then replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.part')
    tmp.write_text(json.dumps(obj, indent=1, default=_plain, allow_nan=False) + '\n')
    os.replace(tmp, path)


def _plain(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x))


def item(name, value, expected, tol=TOL):
    """One comparison: numbers within tol, anything else equal. `diff` is None for non-numbers that differ."""
    numeric = all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
                  for v in (value, expected))
    if numeric:
        diff = abs(float(value) - float(expected))
        ok = bool(diff <= tol)
    else:
        ok = bool(value == expected) if not isinstance(value, np.ndarray) else False
        diff = 0.0 if ok else None
    return dict(name=name, value=value, expected=expected, diff=diff, ok=ok)


def exit_code(items):
    return 0 if items and all(i['ok'] for i in items) else 1


def finish(check, checked_path, items, inputs, extra):
    """Write rederive_<check>.json beside the file it checks; return the exit code."""
    checked_path = Path(checked_path)
    doc = dict(schema=1, check=check, tolerance=TOL, rule_commit=rs_paths.RULE_COMMIT, checked_file=str(checked_path),
               checked_file_sha256=file_sha(checked_path) if checked_path.is_file() else None,
               versions=dict(numpy=np.__version__, sklearn=sklearn.__version__),
               code_sha256=file_sha(__file__), items=items, inputs_sha256=inputs, **extra)
    doc['pass'] = exit_code(items) == 0
    write_json(checked_path.parent / f'rederive_{check}.json', doc)
    return exit_code(items)


# ---- the outcomes (rule 4.1, 4.4) -----------------------------------------------------------------------------

def claim_suffix(d, lower):
    """GO d >= 0.01 and lower > 0; MM d >= 0.01 and lower <= 0; IN 0 < d < 0.01; NO d <= 0."""
    if d >= CLAIM_BAR:
        return 'GO' if lower > 0 else 'MM'
    return 'IN' if d > 0 else 'NO'


def stage0_outcome(scale, d, lower):
    return f"G-{'ES' if scale == 'sample' else 'EF'}-{claim_suffix(d, lower)}"


def emotion_margin_outcome(d, lower):
    return f'PE-{claim_suffix(d, lower)}'


def t_outcome(aspect, lower):
    """T-WIN lower bound > 0; T-LEVEL not T-WIN and lower bound > -r_a; T-BELOW otherwise."""
    if lower > 0:
        return 'T-WIN'
    return 'T-LEVEL' if lower > -RANGES[aspect] else 'T-BELOW'


def licensed_e0(sample_outcome, full_outcome):
    """E0: G-EF-GO, and the sample scale has an outcome other than G-ES-NO."""
    return full_outcome == 'G-EF-GO' and sample_outcome is not None and sample_outcome != 'G-ES-NO'


# ---- labels (CT rule 3, RB rule 3.1 and 3.2, RG rule 3.2) -----------------------------------------------------

def emotion_labels(names, rows):
    """Plurality of each painting's caption emotion codes, ties to the smallest code (adapted from rb_v2)."""
    codes = {e: i for i, e in enumerate(sorted(EMOTIONS))}
    assert {r['emotion'] for r in rows} == set(codes), 'emotion names differ from the nine'
    counts = {n: np.zeros(len(codes), dtype=np.int64) for n in names}
    for r in rows:
        if r['painting'] in counts:
            counts[r['painting']][codes[r['emotion']]] += 1
    assert all(c.sum() > 0 for c in counts.values()), 'a painting without a caption'
    return np.asarray([int(np.argmax(counts[n])) for n in names], dtype=np.int64)    # argmax: first = smallest code


def style_labels(names, rows):
    """The first caption's art style, codes by sorted name; constant per painting (asserted)."""
    codes = {s: i for i, s in enumerate(sorted({r['art_style'] for r in rows}))}
    first = {}
    for r in rows:
        assert first.setdefault(r['painting'], r['art_style']) == r['art_style'], 'style must be constant per painting'
    return np.asarray([codes[first[n]] for n in names], dtype=np.int64)


def genre_labels(names, rows, csvs):
    """The first caption's painting, joined to the WikiArt tables; missing or conflicting entries are -1
    (adapted from rg_vc2.genre_labels)."""
    codes = {}
    for path in csvs:
        with Path(path).open() as handle:
            for relative, raw in csv.reader(handle):
                name, code = Path(relative).stem, int(raw)
                assert 0 <= code < 10, (path, code)
                if name in codes and codes[name] != code:
                    codes[name] = -1
                else:
                    codes.setdefault(name, code)
    first = {}
    for r in rows:
        first.setdefault(r['painting'], codes.get(r['painting'], -1))
    assert all(n in first for n in names), 'held painting without captions'
    return np.asarray([first[n] for n in names], dtype=np.int64)


class Ctx:
    """H, its hash and the three label vectors; `sources` collects the SHA-256 of every file read."""

    def __init__(self, held=HELD_LIST, annotations=ANNOTATIONS, genre_csvs=GENRE_CSVS):
        self.sources = {}
        value = json.loads(Path(held).read_text())
        assert set(value) == {'names', 'sha256'}
        self.names = value['names']
        assert len(self.names) == len(set(self.names)) == N_HELD and self.names == sorted(self.names)
        self.list_sha = names_sha(self.names)
        assert self.list_sha == value['sha256'] == HELD_SHA256, 'held list hash'
        rows = json.loads(Path(annotations).read_text())
        self.labels = dict(emotion=emotion_labels(self.names, rows), style=style_labels(self.names, rows),
                           genre=genre_labels(self.names, rows, genre_csvs))
        assert (self.labels['emotion'] >= 0).all() and (self.labels['style'] >= 0).all()
        assert len(set(self.labels['style'].tolist())) <= 27
        assert int((self.labels['genre'] >= 0).sum()) == N_GENRE_LABELLED, 'labelled genre count'
        for path in (held, annotations, *genre_csvs):
            self.record(path)

    def record(self, path):
        self.sources[str(path)] = file_sha(path)
        return path


# ---- partitions (RB rule 3.1, CT rule 3) -----------------------------------------------------------------------

def r20_partition(x):
    """MiniBatchKMeans k = 20, states 0, 1, 2, n_init 3, batch 4096, on float32; unit norm checked, not repaired
    (adapted from rb_v2.r20_partition): int64 [3, n]."""
    x = np.asarray(x).astype(np.float32)
    assert np.isfinite(x).all()
    assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < UNIT_TOL, 'rows are not unit norm'
    out = []
    with threadpool_limits(limits=1):
        for state in STATES:
            km = MiniBatchKMeans(n_clusters=K, random_state=state, n_init=N_INIT, batch_size=BATCH)
            out.append(km.fit(x).labels_)
    return np.stack(out).astype(np.int64)


def own_partition(prob):
    """Argmax over the K probabilities, lowest index on a tie: int64 [1, n]."""
    return np.argmax(np.asarray(prob), axis=1).astype(np.int64)[None, :]


def l2_probabilities(prob):
    """The sigmoid probabilities in float32 divided by their float32 row norm (CT rule 3)."""
    p = np.asarray(prob, dtype=np.float32)
    norm = np.linalg.norm(p, axis=1, keepdims=True)
    assert (norm > 0).all(), 'a zero probability row'
    return p / norm


def v0c_problems(x, n=N_HELD, width=WIDTH):
    """V0c: n finite rows of unit norm, float16, `width` wide. Returns the list of problems."""
    problems = []
    if x.shape != (n, width):
        return [f'shape {x.shape}, expected {(n, width)}']
    if x.dtype != np.float16:
        problems.append(f'dtype {x.dtype}')
    if not np.isfinite(x).all():
        return problems + ['non-finite rows']
    if np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1).max() >= UNIT_TOL:
        problems.append('rows are not unit norm')
    return problems


def check_sidecar(side, x, aspect, seed, n, list_sha):
    """The sidecar of a saved output set names the same rows, list, aspect, seed and array hash."""
    want = dict(rows=n, list_sha256=list_sha, aspect=aspect, seed=seed, sha256=array_sha(x))
    for key, value in want.items():
        assert side[key] == value, (key, side[key], value)


def _partition_task(t):
    """Load one saved output (in a worker), check its sidecar, return its fixed partitions."""
    path = Path(t['path'])
    side = json.loads(path.with_suffix('.json').read_text())
    with np.load(path, allow_pickle=False) as z:
        assert t['key'] in z.files, (path, z.files)
        x = z[t['key']]
    assert x.shape[0] == N_HELD, (path, x.shape)
    if t['kind'] == 'student':
        check_sidecar(side, x, t['aspect'], t['seed'], N_HELD, HELD_SHA256)
        problems = v0c_problems(x)
        return dict(problems=problems, parts=None if problems else dict(r20=r20_partition(x)))
    assert side['held_sha256'] == HELD_SHA256 and side['rows'] == N_HELD, path
    assert x.dtype == np.float32 and x.ndim == 2, path
    assert side['array_sha256s']['prob'] == array_sha(x), (path, 'sha256 of the saved array')
    if not np.isfinite(x).all():
        assert t['droppable'], f'non-finite rows in {path}'
        return dict(problems=['non-finite rows'], parts=None)
    return dict(problems=[], parts=dict(r20=r20_partition(l2_probabilities(x)), own=own_partition(x)))


def _run_tasks(tasks, workers):
    assert 1 <= workers <= 8
    if workers == 1:
        return [_partition_task(t) for t in tasks]
    with mp.get_context('fork').Pool(workers) as pool:
        return pool.map(_partition_task, tasks, chunksize=1)


_CACHE = {}


def student_parts(ctx, cdir, aspect, workers):
    """The five seeds' r20 partitions of one aspect's saved output set: (parts, problems by seed)."""
    key = (str(cdir), aspect)
    if key not in _CACHE:
        tasks = []
        for s in SEEDS:
            path = Path(cdir) / f'{aspect}__s{s}.npz'
            ctx.record(path), ctx.record(path.with_suffix('.json'))
            tasks.append(dict(path=str(path), key='img', kind='student', aspect=aspect, seed=s))
        outs = _run_tasks(tasks, workers)
        _CACHE[key] = ({s: o['parts'] for s, o in zip(SEEDS, outs) if o['parts']},
                       {s: o['problems'] for s, o in zip(SEEDS, outs) if o['problems']})
    return _CACHE[key]


# ---- the comparators B1 to B6 and their eligibility (CT rule 4 R-COL, RB rule 2) --------------------------------

def _collapsed(run, cut):
    if cut == 'strict':
        return run['kept_count'] < 10 or run['kept_count'] > 90 or run['share'] > 0.5
    return run['share'] > 0.5


def rcol_eligible_seeds(runs, cut, nonfinite=(), version=PAPER_VERSION):
    """Seeds of one settings cut that are not collapsed; seeds with non-finite outputs count as collapsed."""
    chosen = [r for r in runs if r['version'] == version and r['cut'] == cut]
    assert len({r['seed'] for r in chosen}) == len(chosen), 'two R-COL runs for one seed'
    return sorted(r['seed'] for r in chosen if not _collapsed(r, cut) and r['seed'] not in set(nonfinite))


def rcol_notes(rcol):
    """Where the record disagrees with the rule's text: an entry's seed list."""
    notes = []
    for arm, (_, _, cut) in B_ARMS.items():
        if cut is not None and rcol['entries'][arm]['seeds'] != rcol_eligible_seeds(rcol['runs'], cut):
            notes.append(f"{arm}: record seeds {rcol['entries'][arm]['seeds']}, rule says "
                         f"{rcol_eligible_seeds(rcol['runs'], cut)}")
    return notes


def _same_partitions(mine, path, key, label, report):
    """Compare with a saved partition file when it exists; count the others as unchecked."""
    if path is None or not Path(path).is_file():
        report['unchecked'] += 1
        return
    with np.load(path, allow_pickle=False) as z:
        theirs = z[key]
    report['checked'] += 1
    if theirs.shape != mine.shape or not np.array_equal(theirs, mine):
        report['mismatches'].append(f'{label}: differs from {path}')


def entry_parts(ctx, scale, workers):
    """B1 to B6 at one scale: partitions of the eligible seeds, the eligible seed lists, the notes."""
    if ('entries', scale) in _CACHE:
        return _CACHE[('entries', scale)]
    root = Path(CT_ROOT[scale]) / 'held'
    rcol = json.loads(ctx.record(RCOL[scale]).read_text())
    tasks, owners = [], []
    for name, (folder, arm, cut) in B_ARMS.items():
        use = SEEDS if cut is None else [s for s in SEEDS if s in rcol_eligible_seeds(rcol['runs'], cut)]
        for s in use:
            path = root / 'percept' / folder / f'{arm}__s{s}.npz'
            ctx.record(path), ctx.record(path.with_suffix('.json'))
            tasks.append(dict(path=str(path), key='prob', kind='prob', droppable=cut is not None))
            owners.append((name, s))
    outs = _run_tasks(tasks, workers)
    parts = {name: {} for name in B_ARMS}
    dropped = {}
    for (name, s), o in zip(owners, outs):
        if o['parts']:
            parts[name][s] = o['parts']
        else:
            dropped.setdefault(name, []).append(s)
    saved = dict(checked=0, unchecked=0, mismatches=[])
    for name in B_ARMS:
        for s, pt in parts[name].items():
            path = root / 'partitions' / f'{name}__s{s}.npz'
            for reading in READINGS:
                _same_partitions(pt[reading], path, reading, f'{name} s{s} {reading}', saved)
    _CACHE[('entries', scale)] = dict(parts=parts, eligible={n: sorted(p) for n, p in parts.items()},
                                      dropped=dropped, notes=rcol_notes(rcol), saved=saved)
    return _CACHE[('entries', scale)]


# ---- the draws and the bootstrap (RB rule 3.2, RG rule 3.2, rule 3.5) -----------------------------------------

def draw_plan(n, pools, draws):
    """Per draw I_b = rng.integers(0, n, size=n), then for each pool with seeds, in the dict's order,
    S_b = rng.choice(pool, size=len(pool), replace=True). One default_rng(0) per plan (adapted from rb_v2.draw_plan)."""
    rng = np.random.default_rng(0)
    idx = np.empty((draws, n), dtype=np.int64)
    seeds = []
    for b in range(draws):
        idx[b] = rng.integers(0, n, size=n)
        seeds.append({name: rng.choice(pool, size=len(pool), replace=True) for name, pool in pools.items() if len(pool)})
    return idx, seeds


def _values(lab, parts, rows, cache):
    """val(arm, seed, reading): the mean over the partitions' AMIs on `rows` (labels masked before: lab = labels[rows])."""
    def val(arm, seed, reading):
        key = (arm, int(seed), reading)
        if key not in cache:
            cache[key] = float(np.mean([adjusted_mutual_info_score(lab, p[rows]) for p in parts[arm][int(seed)][reading]]))
        return cache[key]
    return val


_STATE = {}


def _draw_chunk(span):
    st = _STATE
    out = []
    with threadpool_limits(limits=1):
        for b in range(*span):
            idx = st['idx'][b]
            rows = idx[st['labels'][idx] >= 0]              # draw over all rows, drop the unlabelled inside the AMI
            val = _values(st['labels'][rows], st['parts'], rows, {})
            out.append(st['stat'](val, st['seeds'][b]))
    return out


def bootstrap(labels, parts, pools, stat, draws, workers):
    """stat(val, S) -> list of floats, once per draw; returns an array [draws, k]."""
    idx, seeds = draw_plan(len(labels), pools, draws)
    _STATE.clear()
    _STATE.update(labels=labels, parts=parts, idx=idx, seeds=seeds, stat=stat)
    try:
        step = max(1, -(-draws // (max(workers, 1) * 4)))
        spans = [(lo, min(lo + step, draws)) for lo in range(0, draws, step)]
        if workers <= 1:
            pieces = [_draw_chunk(sp) for sp in spans]
        else:
            with mp.get_context('fork').Pool(workers) as pool:
                pieces = pool.map(_draw_chunk, spans)
    finally:
        _STATE.clear()
    return np.asarray([v for piece in pieces for v in piece], dtype=np.float64).reshape(draws, -1)


def point_table(labels, parts):
    """I0 = every labelled row: {arm: {seed: {reading: [AMI of each partition]}}}."""
    rows = np.flatnonzero(labels >= 0)
    lab = labels[rows]
    return {arm: {s: {r: [float(adjusted_mutual_info_score(lab, p[rows])) for p in ps[r]] for r in ps}
                  for s, ps in seeds.items()} for arm, seeds in parts.items()}


def _point_val(table):
    return lambda arm, seed, reading: float(np.mean(table[arm][int(seed)][reading]))


def _table_json(table, arms):
    return {arm: {str(s): table[arm][s] for s in sorted(table[arm])} for arm in arms}


def _interval(boot):
    return np.percentile(boot, [2.5, 97.5]).tolist() if len(boot) else None


def _mean_over(val, arm, seeds, reading='r20'):
    return float(np.mean([val(arm, s, reading) for s in seeds]))


def entries_statistic(labels, cand, entries, draws=DRAWS, workers=8, cand_name='G'):
    """RB rule 3.2: d = A_cand - the maximum over the eligible entries' two readings; the maximum is taken again in
    each draw; the draws are I_b, then S_b for the candidate, then for each entry with seeds in the dict's order.
    cand: {seed: {'r20'}}; entries: {name: {seed: {'r20', 'own'}}} (an entry without seeds is left out)."""
    assert cand_name not in entries and cand
    labels = np.asarray(labels)
    parts = {cand_name: cand, **{e: p for e, p in entries.items() if p}}
    live = [e for e in entries if entries[e]]
    pools = {name: sorted(p) for name, p in parts.items()}
    table = point_table(labels, parts)

    def stat(val, S):
        best = max(_mean_over(val, e, S[e], r) for e in live for r in READINGS)
        return [_mean_over(val, cand_name, S[cand_name]) - best]

    point = _point_val(table)
    a_e = {f'{e}-{r}': _mean_over(point, e, pools[e], r) for e in live for r in READINGS}
    a_cand = _mean_over(point, cand_name, pools[cand_name])
    boot = bootstrap(labels, parts, pools, stat, draws, workers)[:, 0] if draws else np.empty(0)
    return dict(A_cand=a_cand, A_e=a_e, d=a_cand - max(a_e.values()), interval=_interval(boot),
                bootstrap_d=boot.tolist(), per_seed=_table_json(table, parts),
                eligible_seeds={cand_name: pools[cand_name], **{e: sorted(entries[e]) for e in entries}})


def fixed_statistic(labels, cand, fixed, draws=DRAWS, workers=8, cand_name='G'):
    """RG rule 3.2: d_T = A_cand(I, S) - A_T(I); T's partitions are fixed, no seed; draws: I_b, then S_b for cand."""
    labels = np.asarray(labels)
    parts = {cand_name: cand, 'T': fixed}
    pools = {cand_name: sorted(cand)}
    table = point_table(labels, parts)

    def stat(val, S):
        return [_mean_over(val, cand_name, S[cand_name]) - val('T', 0, 'r20')]

    point = _point_val(table)
    a_cand, a_fixed = _mean_over(point, cand_name, pools[cand_name]), point('T', 0, 'r20')
    boot = bootstrap(labels, parts, pools, stat, draws, workers)[:, 0] if draws else np.empty(0)
    return dict(A_cand=a_cand, A_fixed=a_fixed, d=a_cand - a_fixed, interval=_interval(boot),
                bootstrap_d=boot.tolist(), per_seed=_table_json(table, parts))


def paired_statistic(labels, xs, final, draws=DRAWS, workers=8):
    """Rule 3.5: for each arm x, A_x(I, S) - A_F(I, S) with ONE S_b shared by every arm; draws: I_b, then S_b.
    xs: {name: {seed: {'r20'}}}; final: the same for F. Returns {name: result}."""
    labels = np.asarray(labels)
    seeds = sorted(final)
    assert all(sorted(p) == seeds for p in xs.values()), 'every arm needs the same seeds'
    parts = {**xs, '_F': final}
    names = list(xs)
    table = point_table(labels, parts)

    def stat(val, S):
        f = _mean_over(val, '_F', S['S'])
        return [_mean_over(val, x, S['S']) - f for x in names]

    point = _point_val(table)
    a_f = _mean_over(point, '_F', seeds)
    boot = bootstrap(labels, parts, {'S': seeds}, stat, draws, workers) if draws else np.empty((0, len(names)))
    out = {}
    for j, x in enumerate(names):
        a_x = _mean_over(point, x, seeds)
        out[x] = dict(A_arm=a_x, A_final=a_f, d=a_x - a_f, interval=_interval(boot[:, j]),
                      bootstrap_d=boot[:, j].tolist())
    return out


# ---- stage 0 (rule 3.1, 4.1) ----------------------------------------------------------------------------------

def stage0_scale(ctx, scale, cand_dir, cand_name='G', draws=DRAWS, workers=8, cand_saved=None):
    """One scale: the emotion statistic of the candidate's saved held outputs against B1 to B6."""
    cand, problems = student_parts(ctx, cand_dir, 'emotion', workers)
    if problems:                                            # V0d: a seed that fails V0c leaves the scale with no outcome
        return dict(candidate=cand_name, scale=scale, claim=None, v_failure={str(s): p for s, p in problems.items()})
    entries = entry_parts(ctx, scale, workers)
    res = entries_statistic(ctx.labels['emotion'], cand, entries['parts'], draws, workers, cand_name)
    saved = dict(checked=entries['saved']['checked'], unchecked=entries['saved']['unchecked'],
                 mismatches=list(entries['saved']['mismatches']))
    for s, pt in cand.items():
        _same_partitions(pt['r20'], None if cand_saved is None else Path(cand_saved) / f'emotion__s{s}__r20.npz',
                         'labels', f'{cand_name} s{s} r20', saved)
    lower = res['interval'][0] if draws else None
    res.update(candidate=cand_name, scale=scale, v_failure=None, partitions=saved, eligibility_notes=entries['notes'],
               nonfinite_dropped=entries['dropped'], n_resamples=draws, held_sha256=HELD_SHA256,
               claim=claim_suffix(res['d'], lower) if draws else None)
    return res


def held_dir(outputs, name):
    """G's full held outputs are 12-27's; every other run's are in this round's full tree (spec: shared interfaces)."""
    if name == 'G':
        return rs_paths.RG_FULL / 'held' / rs_paths.G_ARM
    return rs_paths.full_root(outputs) / 'held' / name


def stage0(ctx, outputs, draws=DRAWS, workers=8):
    """Both scales: G's sample seeds are this round's held inference, G's full seeds are 12-27's."""
    dirs = {'sample': Path(outputs) / 'stage0/sample/held/G', 'full': held_dir(outputs, 'G')}
    blocks = {}
    for scale, cdir in dirs.items():
        saved = Path(outputs) / 'stage0/partitions' / scale / 'G'
        b = stage0_scale(ctx, scale, cdir, 'G', draws, workers, saved if saved.is_dir() else None)
        b['outcome_id'] = None if b['claim'] is None else stage0_outcome(scale, b['d'], b['interval'][0])
        blocks[scale] = b
    return dict(blocks=blocks, n_resamples=draws, held_sha256=HELD_SHA256,
                E0=licensed_e0(blocks['sample']['outcome_id'], blocks['full']['outcome_id']),
                outcomes={s: b['outcome_id'] for s, b in blocks.items()})


# ---- stage 4 (rule 3.4, 4.4) ----------------------------------------------------------------------------------

def t_partitions(ctx, aspect):
    """T_a: r20 partitions of the instructed Qwen image of the aspect on H, recomputed from its store and compared
    with 12-24's frozen file (adapted from rg_vc2.derive)."""
    if ('T', aspect) in _CACHE:
        return _CACHE[('T', aspect)]
    store = PROMPT_ROOT / f'{PROMPTS[aspect]}_held'
    meta = json.loads((store / 'meta.json').read_text())
    assert meta['held_sha256'] == HELD_SHA256 and meta['shape'] == [N_HELD, WIDTH] and meta['dtype'] == 'float16'
    with np.load(store / 'ids.npz', allow_pickle=False) as z:
        assert z.files == ['painting_names'] and z['painting_names'].tolist() == ctx.names
    x = np.load(store / 'img.npy', allow_pickle=False)
    assert array_sha(x) == meta['img_sha256'], 'T store hash'
    assert not v0c_problems(x), 'T store rows'
    x = x.astype(np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)      # ct_diag.target_images: float32, row-normalized
    frozen = RB_T_PARTITIONS / f'{aspect}__s0__r20.npz'
    for p in (store / 'meta.json', store / 'ids.npz', store / 'img.npy', frozen):
        ctx.record(p)
    mine = r20_partition(x)
    with np.load(frozen, allow_pickle=False) as z:
        equal = bool(np.array_equal(z['labels'], mine))
    _CACHE[('T', aspect)] = (mine, equal)
    return _CACHE[('T', aspect)]


def stage4(ctx, outputs, candidate, draws=DRAWS, workers=8, margin=True):
    """d_T,a per aspect against T_a (RG rule 3.2 with c* in G's place) and the emotion margin over B1 to B6."""
    cdir = held_dir(outputs, candidate)
    aspects = {}
    for a in ASPECTS:
        cand, problems = student_parts(ctx, cdir, a, workers)
        t, equal = t_partitions(ctx, a)
        res = dict(T_equals_frozen=equal)
        if problems:                                        # V4d: a seed that fails V0c leaves no outcome
            res.update(outcome_id=None, v_failure={str(s): p for s, p in problems.items()})
        else:
            res.update(fixed_statistic(ctx.labels[a], cand, {0: dict(r20=t)}, draws, workers, candidate))
            res.update(outcome_id=t_outcome(a, res['interval'][0]) if draws else None,
                       size_in_seed_ranges=res['d'] / RANGES[a], labelled_rows=int((ctx.labels[a] >= 0).sum()))
        aspects[a] = res
    out = dict(candidate=candidate, aspects=aspects, n_resamples=draws, held_sha256=HELD_SHA256)
    if margin:
        b = stage0_scale(ctx, 'full', cdir, candidate, draws, workers)
        b['outcome_id'] = None if b['claim'] is None else emotion_margin_outcome(b['d'], b['interval'][0])
        out['emotion_margin'] = b
    return out


# ---- the held ablation (rule 3.5, 4.5) ------------------------------------------------------------------------

def removal_names(final):
    """Run names of F's eight removal arms: F's configuration with one removal added (rs_sheet.name)."""
    cfg = rs_sheet.parse(final)
    return {r: rs_sheet.name(dict(cfg, remove=(r,))) for r in REMOVALS}


def ablation(ctx, outputs, final, draws=DRAWS, workers=8):
    """Delta_a(x) = A_x(I0, S0) - A_F(I0, S0) with its interval, per removal and aspect; absolute AMIs beside."""
    names = removal_names(final)
    f_parts, arms, status = {}, {}, {}
    for a in ASPECTS:
        f_parts[a], problems = student_parts(ctx, held_dir(outputs, final), a, workers)
        assert not problems and len(f_parts[a]) == len(SEEDS), f'F ({final}) {a}: {problems}'
    for x, name in names.items():
        cdir = held_dir(outputs, name)
        if not all((cdir / f'{a}__s{s}.npz').is_file() for a in ASPECTS for s in SEEDS):
            status[x] = dict(name=name, status='missing')       # an arm that failed has no outputs: no effect
            continue
        got = {a: student_parts(ctx, cdir, a, workers) for a in ASPECTS}
        bad = {a: p for a, (_, p) in got.items() if p}
        status[x] = dict(name=name, status='failed V0c' if bad else 'ok', problems=bad)
        if not bad:
            arms[x] = {a: got[a][0] for a in ASPECTS}
    effects = {x: {} for x in arms}
    for a in ASPECTS:
        res = paired_statistic(ctx.labels[a], {x: arms[x][a] for x in arms}, f_parts[a], draws, workers) if arms else {}
        for x, r in res.items():
            effects[x][a] = r
    return dict(final=final, removals=status, effects=effects, n_resamples=draws, held_sha256=HELD_SHA256,
                A_final={a: effects[next(iter(effects))][a]['A_final'] for a in ASPECTS} if effects else None)


# ---- comparison with a verdict file ---------------------------------------------------------------------------

def _first(d, keys):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return None


def _flatten(x, prefix=''):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from _flatten(v, f'{prefix}.{k}')
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            yield from _flatten(v, f'{prefix}.{i}')
    else:
        yield prefix, x


def max_abs_difference(mine, theirs):
    """The largest absolute difference over the numbers both nested structures hold; None when they differ in shape."""
    a, b = dict(_flatten(mine)), dict(_flatten(theirs))
    if set(a) != set(b):
        return None
    return max((abs(float(a[k]) - float(b[k])) for k in a), default=0.0)


def compare_block(prefix, mine, theirs, outcome=None):
    """One stage-0-style block (K4's layout) against a verdict block; `outcome` maps an outcome id to what is compared."""
    outcome = outcome or (lambda s: s)
    if mine.get('v_failure'):
        return [item(f'{prefix}.outcome_id', None, theirs.get('outcome_id'))]
    items = [item(f'{prefix}.d', mine['d'], theirs.get('d')),
             item(f'{prefix}.A_cand', mine['A_cand'], _first(theirs, ('A_cand', 'A_G', 'A_c')))]
    their_ae = theirs.get('A_e') or {}
    for k in sorted(set(mine['A_e']) | set(their_ae)):
        items.append(item(f'{prefix}.A_e[{k}]', mine['A_e'].get(k), their_ae.get(k)))
    for j, side in enumerate(('lower', 'upper')):
        items.append(item(f'{prefix}.interval_{side}', mine['interval'][j], (theirs.get('interval') or [None, None])[j]))
    their_seeds = (theirs.get('eligibility') or theirs).get('eligible_seeds') or {}
    mine_seeds = {k: v for k, v in mine['eligible_seeds'].items()}
    for arm in sorted(set(mine_seeds) | set(their_seeds)):
        mine_list = mine_seeds.get(arm)
        their_list = their_seeds.get(arm)
        items.append(item(f'{prefix}.eligible_seeds[{arm}]', mine_list,
                          None if their_list is None else [int(s) for s in their_list]))
    if theirs.get('bootstrap_d') is not None:
        same_len = len(theirs['bootstrap_d']) == len(mine['bootstrap_d'])
        diff = float(np.max(np.abs(np.asarray(theirs['bootstrap_d']) - np.asarray(mine['bootstrap_d'])))) if same_len else None
        items.append(item(f'{prefix}.bootstrap_d_max_abs_diff', 'different length' if diff is None else diff, 0.0))
    if theirs.get('per_seed') is not None:
        mine_ps = {k: v for k, v in mine['per_seed'].items() if k in theirs['per_seed']}
        diff = max_abs_difference(mine_ps, {k: theirs['per_seed'][k] for k in mine_ps})
        items.append(item(f'{prefix}.per_seed_max_abs_diff', 'different keys' if diff is None else diff, 0.0))
    items.append(item(f'{prefix}.outcome_id', outcome(mine['outcome_id']), outcome(theirs.get('outcome_id'))))
    return items


def compare_stage0(mine, verdict):
    """The re-derived stage 0 against <outputs>/stage0/verdict.json (ticket 05's keys: blocks, outcomes, E0, ...)."""
    items = [item('held_sha256', mine['held_sha256'], verdict.get('held_sha256')),
             item('n_resamples', mine['n_resamples'], verdict.get('n_resamples'))]
    for scale in ('sample', 'full'):
        items += compare_block(scale, mine['blocks'][scale], verdict['blocks'][scale])
        items.append(item(f'{scale}.outcomes', mine['blocks'][scale]['outcome_id'], verdict['outcomes'].get(scale)))
    items.append(item('E0', mine['E0'], verdict.get('E0')))
    return items


# Where stage 4's and the ablation's numbers sit in the verdict files is not fixed yet (build 9): each item is looked
# up under these alternatives, in order. Edit here once ticket 09 and 10 fix their keys.
STAGE4_ASPECT_PATHS = (('aspects', '{a}'), ('d_T', '{a}'))
STAGE4_MARGIN_PATHS = (('emotion_margin',), ('pe',), ('blocks', 'emotion_margin'))
ABLATION_PATHS = (('removals', '{x}', '{a}'), ('removals', '{x}', 'aspects', '{a}'), ('effects', '{x}', '{a}'),
                  ('arms', '{x}', '{a}'))


def lookup(doc, paths, **fmt):
    """The first of the alternative key paths that exists in doc, else None."""
    for path in paths:
        node = doc
        for key in path:
            key = key.format(**fmt)
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if node is not None:
            return node
    return None


def compare_t_block(prefix, mine, theirs, aspect):
    """d_T,a against a verdict block (RG's verdict.json d_T, RG's reported.json margin T, or ticket 09's)."""
    if theirs is None:
        return [item(f'{prefix}.present', 'present', 'missing from the verdict')]
    if mine.get('v_failure'):
        return [item(f'{prefix}.outcome_id', None, _first(theirs, ('outcome_id', 'outcome')))]
    a_t = _first(theirs, ('A_T', 'A_fixed')) if _first(theirs, ('A_T', 'A_fixed')) is not None else \
        (theirs.get('A_e') or {}).get('T-r20')
    items = [item(f'{prefix}.d_T', mine['d'], _first(theirs, ('d', 'd_T'))),
             item(f'{prefix}.A_cand', mine['A_cand'], _first(theirs, ('A_cand', 'A_G', 'A_c'))),
             item(f'{prefix}.A_T', mine['A_fixed'], a_t)]
    if mine['interval'] is None:                            # point values only (draws = 0)
        return items
    for j, side in enumerate(('lower', 'upper')):
        items.append(item(f'{prefix}.interval_{side}', mine['interval'][j], (theirs.get('interval') or [None, None])[j]))
    if theirs.get('bootstrap_d') is not None:
        same = len(theirs['bootstrap_d']) == len(mine['bootstrap_d'])
        diff = float(np.max(np.abs(np.asarray(theirs['bootstrap_d']) - np.asarray(mine['bootstrap_d'])))) if same else 'different length'
        items.append(item(f'{prefix}.bootstrap_d_max_abs_diff', diff, 0.0))
    their_id = _first(theirs, ('outcome_id', 'outcome'))
    if their_id is not None:
        items.append(item(f'{prefix}.outcome_id', mine['outcome_id'], their_id))
    return items


def compare_stage4(mine, verdict):
    """The re-derived stage 4 against <outputs>/stage4/verdict.json."""
    items = [item('held_sha256', mine['held_sha256'], verdict.get('held_sha256'))]
    for a in ASPECTS:
        items.append(item(f'{a}.T_partitions_equal_frozen', mine['aspects'][a]['T_equals_frozen'], True))
        items += compare_t_block(a, mine['aspects'][a], lookup(verdict, STAGE4_ASPECT_PATHS, a=a), a)
    margin = lookup(verdict, STAGE4_MARGIN_PATHS)
    if margin is None:
        items.append(item('emotion_margin', 'present', 'missing from the verdict'))
    else:
        items += compare_block('emotion_margin', mine['emotion_margin'], margin)
    return items


def compare_ablation(mine, verdict):
    """The re-derived held ablation (every Delta_a and its interval, absolute AMIs) against <outputs>/ablation/held.json."""
    items = [item('held_sha256', mine['held_sha256'], verdict.get('held_sha256'))]
    for x, effect in mine['effects'].items():
        for a, r in effect.items():
            theirs = lookup(verdict, ABLATION_PATHS, x=x, a=a)
            p = f'{x}.{a}'
            if theirs is None:
                items.append(item(f'{p}.present', 'present', 'missing from the verdict'))
                continue
            items += [item(f'{p}.delta', r['d'], _first(theirs, ('delta', 'd', 'd_x'))),
                      item(f'{p}.A_arm', r['A_arm'], _first(theirs, ('A_arm', 'A_x'))),
                      item(f'{p}.A_final', r['A_final'], _first(theirs, ('A_final', 'A_F')))]
            for j, side in enumerate(('lower', 'upper')):
                items.append(item(f'{p}.interval_{side}', r['interval'][j],
                                  (theirs.get('interval') or [None, None])[j]))
    return items


# ---- the self-check on frozen data (ticket 11, item 2) -----------------------------------------------------------

def selfcheck(ctx, draws=DRAWS, workers=8):
    """Stage 0 on 12-24's K4 outputs against the K4 blocks of RB's verdict_stage3/4; stage 4 on G's held outputs
    against 12-27's verdict.json and reported.json; the paired statistic's point value against 12-27's d_R."""
    items, extra = [], {}
    for scale, name in (('sample', 'verdict_stage3.json'), ('full', 'verdict_stage4.json')):
        path = rs_paths.RB_PART2 / name
        ctx.record(path)
        frozen = json.loads(path.read_text())['blocks']['K4']
        b = stage0_scale(ctx, scale, RB_K4_DIRS[scale], 'K4', draws, workers, RB_K4_PARTITIONS[scale])
        b['outcome_id'] = stage0_outcome(scale, b['d'], b['interval'][0]) if draws else None
        suffix = lambda s: None if s is None else s.rsplit('-', 1)[-1]          # S1-GO / S2-GO / G-ES-GO: compare the tail
        extra[f'K4_{scale}'] = b
        if draws:
            items += compare_block(f'K4.{scale}', b, frozen, outcome=suffix)
        items += [item(f'K4.{scale}.partitions_mismatches', len(b['partitions']['mismatches']), 0),
                  item(f'K4.{scale}.partitions_checked_gt_0', b['partitions']['checked'] > 0, True),
                  item(f'K4.{scale}.R-COL notes', len(b['eligibility_notes']), 0)]
        if not draws:
            items += [item(f'K4.{scale}.d', b['d'], frozen['d']), item(f'K4.{scale}.A_cand', b['A_cand'], frozen['A_cand'])]
    verdict_c, reported_c = rs_paths.RG_STAGE_C / 'verdict.json', rs_paths.RG_STAGE_C / 'reported.json'
    ctx.record(verdict_c), ctx.record(reported_c)
    rg_verdict, rg_reported = json.loads(verdict_c.read_text()), json.loads(reported_c.read_text())
    res = stage4(ctx, rs_paths.OUTPUTS, 'G', draws, workers, margin=False)
    extra['G_stage4'] = res
    frozen_t = {'genre': rg_verdict['d_T'], **{a: rg_reported['G']['aspects'][a]['margins']['T'] for a in ('emotion', 'style')}}
    expected_outcome = dict(emotion='T-WIN', style='T-LEVEL', genre='T-LEVEL')      # rule 4.4: "at the time of writing"
    for a in ASPECTS:
        items.append(item(f'G.{a}.T_partitions_equal_frozen', res['aspects'][a]['T_equals_frozen'], True))
        theirs = {k: v for k, v in frozen_t[a].items() if k != 'outcome_id'}
        items += compare_t_block(f'G.{a}', res['aspects'][a], theirs, a)
        if draws:
            items.append(item(f'G.{a}.outcome_id (rule 4.4 text)', res['aspects'][a]['outcome_id'], expected_outcome[a]))
    # the paired statistic (held ablation) with G as F and K4 as x: its point value is minus RG's d_R on genre
    g = student_parts(ctx, held_dir(rs_paths.OUTPUTS, 'G'), 'genre', workers)[0]
    k4 = student_parts(ctx, RB_K4_DIRS['full'], 'genre', workers)[0]
    pair = paired_statistic(ctx.labels['genre'], {'K4': k4}, g, draws=0, workers=1)['K4']
    items += [item('paired.genre.delta = -d_R', pair['d'], -rg_verdict['d_R']['d']),
              item('paired.genre.A_arm = A_R', pair['A_arm'], rg_verdict['d_R']['A_R']),
              item('paired.genre.A_final = A_G', pair['A_final'], rg_verdict['d_R']['A_G'])]
    extra['paired'] = pair
    return items, extra


# ---- command line ---------------------------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='mode', required=True)

    def add(name, outputs=rs_paths.OUTPUTS):
        s = sub.add_parser(name)
        s.add_argument('--outputs', type=Path, default=outputs)
        s.add_argument('--workers', type=int, default=8, choices=range(1, 9))
        return s

    add('stage0')
    add('stage4').add_argument('--candidate', required=True)
    add('ablation').add_argument('--final', required=True)
    add('selfcheck', SELF_OUT).add_argument('--draws', type=int, default=DRAWS)
    return p


def verdict_file(mode, outputs):
    return Path(outputs) / {'stage0': 'stage0/verdict.json', 'stage4': 'stage4/verdict.json',
                            'ablation': 'ablation/held.json'}[mode]


def run(args):
    rs_paths.check_imports()
    ctx = Ctx()
    t0 = time.time()
    if args.mode == 'selfcheck':
        items, extra = selfcheck(ctx, args.draws, args.workers)
        checked = Path(args.outputs) / 'selfcheck.json'
        code = finish('selfcheck', checked, items, ctx.sources, dict(draws=args.draws, mine=extra,
                                                                      seconds=round(time.time() - t0, 1)))
    else:
        outputs = rs_paths.check_outputs(args.outputs)
        checked = ctx.record(verdict_file(args.mode, outputs))           # V runs after the verdict file exists
        verdict = json.loads(checked.read_text())
        if args.mode == 'stage0':
            mine = stage0(ctx, outputs, DRAWS, args.workers)
            items = compare_stage0(mine, verdict)
        elif args.mode == 'stage4':
            mine = stage4(ctx, outputs, args.candidate, DRAWS, args.workers)
            items = compare_stage4(mine, verdict)
        else:
            mine = ablation(ctx, outputs, args.final, DRAWS, args.workers)
            items = compare_ablation(mine, verdict)
        code = finish(args.mode, checked, items, ctx.sources, dict(mine=mine, seconds=round(time.time() - t0, 1)))
    bad = [i for i in items if not i['ok']]
    worst = max((i['diff'] for i in items if isinstance(i['diff'], float)), default=0.0)
    print(f"{args.mode}: {len(items)} items, {len(bad)} disagree, largest numeric difference {worst:.3g}")
    for i in bad[:20]:
        print('  DISAGREE', i['name'], 'mine', i['value'], 'expected', i['expected'], 'diff', i['diff'])
    return code


def main(argv=None):
    return run(build_parser().parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
