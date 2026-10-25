"""Part 1 pick (rule §3.1 and §4.1): fixed partitions of each recipe's P exports, effects against R0-5, the
2,000-draw intervals, gain, guard, pass, g and the outcome.

Reads `part1/P.json` and `part1/train/<recipe>/<aspect>__s0__P.npz` (+ `__s0.json`), writes
`part1/partitions/<recipe>/<aspect>.npz` (key `labels`, int64 [3, 29232], with a sibling `.json`) and `part1/pick.json`.
Decisions use unrounded values; only the pick's `round(g, 2)` rounds. Reported values have no rounding here.
"""
import argparse
import hashlib
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import rb_paths  # noqa: F401  (sys.path)
import rb_ami
import rb_data
import rb_rows

ASPECTS = rb_paths.RB_ASPECTS                                   # ('emotion', 'style', 'genre')
SEED_RANGES = {'emotion': 0.0046, 'style': 0.0082, 'genre': 0.0128}   # the user's values (rule §3.1)
REFERENCE = rb_rows.REFERENCE
CANDIDATES = rb_rows.TIE_ORDER
N_P = 29232
WIDTH = 2048
SEED = 0
_CODE = ('rb_ami.py', 'rb_pick.py', 'rb_rows.py')


# ------------------------------------------------------------------ the rule's conditions
def gain(lo, e, aspect):
    """Gain on an aspect ("at least small"): the interval's lower bound > 0 and e >= the aspect's seed range."""
    return bool(lo > 0 and e >= SEED_RANGES[aspect])


def guard(e_by_aspect):
    """e >= -r on all three aspects (point estimates)."""
    return bool(all(e_by_aspect[a] >= -SEED_RANGES[a] for a in ASPECTS))


def mean_gain(e_by_aspect):
    """g = (e_emotion / r_emotion + e_style / r_style + e_genre / r_genre) / 3, losses included."""
    return (e_by_aspect['emotion'] / SEED_RANGES['emotion'] + e_by_aspect['style'] / SEED_RANGES['style']
            + e_by_aspect['genre'] / SEED_RANGES['genre']) / 3


def evaluate(effects):
    """{vp2_ok, guard, pass, g, flag_g_le_0} of one candidate from {aspect: {e, lo, hi}}."""
    e = {a: effects[a]['e'] for a in ASPECTS}
    has_guard = guard(e)
    has_gain = any(gain(effects[a]['lo'], effects[a]['e'], a) for a in ASPECTS)
    g = mean_gain(e)
    return {'vp2_ok': True, 'vp2_problems': [], 'guard': has_guard, 'pass': bool(has_gain and has_guard),
            'g': g, 'flag_g_le_0': bool(g <= 0)}


def pick(g_of_passing):
    """The passing candidate with the largest `round(g, 2)`; on equal values the first in R0-3, R6-5, R6-3, R7-5, R7-3."""
    assert set(g_of_passing) <= set(CANDIDATES)
    best = None
    for name in CANDIDATES:
        if name in g_of_passing and (best is None or round(g_of_passing[name], 2) > round(g_of_passing[best], 2)):
            best = name
    return best


def decide(effects, vp2_problems=None):
    """({candidate: block}, outcome, recipe). `effects[c]` is None only for a candidate that failed VP2 (never passes)."""
    vp2_problems = vp2_problems or {}
    assert set(effects) == set(CANDIDATES), sorted(effects)
    assert set(vp2_problems) <= set(CANDIDATES)
    blocks = {}
    for name in CANDIDATES:
        if name in vp2_problems:
            assert vp2_problems[name] and effects[name] is None, f'{name}: VP2 failure and effects disagree'
            blocks[name] = {'vp2_ok': False, 'vp2_problems': list(vp2_problems[name]), 'guard': None,
                            'pass': False, 'g': None, 'flag_g_le_0': None}
        else:
            assert effects[name] is not None, f'{name}: no effects and no VP2 problem'
            blocks[name] = evaluate(effects[name])
    chosen = pick({n: b['g'] for n, b in blocks.items() if b['pass']})
    if chosen is None:
        return blocks, 'P-KEEP', REFERENCE
    return blocks, f'P-{chosen}', chosen


def triggers(outcome, recipe):
    """Rule §4.1 "Triggers": K4 exists unless P-KEEP; K5 (and M5) only under P-R7-5 or P-R7-3."""
    return {'K4': bool(outcome != 'P-KEEP'), 'K5': bool(recipe in ('R7-5', 'R7-3'))}


# ------------------------------------------------------------------ the bootstrap
_BOOT = {}


def _set_state(state):
    _BOOT.update(state)


def _init_worker(state):
    _set_state(state)
    threadpool_limits(limits=1)        # one thread per worker process; the parent's limits are left alone


def _draw_scores(b):
    """A(c, I_b) of every recipe for draw b; I_b = L_a[draws[b]] (the same draw serves every recipe and partition)."""
    idx = _BOOT['labelled'][_BOOT['draws'][b]]
    return np.array([rb_ami.score(_BOOT['labels'], p, idx) for p in _BOOT['parts']])


def bootstrap_scores(labels, parts, n_draws=rb_ami.N_DRAWS, workers=8):
    """{recipe: float64 [n_draws]}: A(c, I_b) for b = 1 to n_draws, one `default_rng(0)` per aspect (rule §3.1).

    Parallel over draws in `workers` processes; each draw is computed alone, so the result does not depend on `workers`.
    """
    assert 1 <= workers <= 8 and n_draws > 0
    labels = np.asarray(labels)
    names = list(parts)
    assert labels.ndim == 1 and all(np.shape(p) == (3, len(labels)) for p in parts.values()), 'partition shapes'
    labelled = np.flatnonzero(labels >= 0)
    state = dict(labels=labels, labelled=labelled, parts=[parts[n] for n in names],
                 draws=rb_ami.draws(len(labelled), n_draws))
    if workers == 1 or n_draws < 2:
        _set_state(state)
        rows = [_draw_scores(b) for b in range(n_draws)]
    else:
        with multiprocessing.get_context('fork').Pool(workers, initializer=_init_worker, initargs=(state,)) as pool:
            rows = pool.map(_draw_scores, range(n_draws), chunksize=max(1, n_draws // (workers * 8)))
    scores = np.stack(rows)
    assert scores.shape == (n_draws, len(names)) and np.isfinite(scores).all()
    return {n: scores[:, i] for i, n in enumerate(names)}


def aspect_effects(labels, parts, n_draws=rb_ami.N_DRAWS, workers=8):
    """{recipe: {A, e, lo, hi}} for one aspect: point estimates on all of L_a, interval from `n_draws` shared draws."""
    assert REFERENCE in parts
    labels = np.asarray(labels)
    labelled = np.flatnonzero(labels >= 0)
    point = {n: rb_ami.score(labels, p, labelled) for n, p in parts.items()}
    boot = bootstrap_scores(labels, parts, n_draws, workers)
    out = {}
    for n in parts:
        lo, hi = np.percentile(boot[n] - boot[REFERENCE], [2.5, 97.5])      # default linear method
        out[n] = dict(A=point[n], e=point[n] - point[REFERENCE], lo=float(lo), hi=float(hi))
    return out


# ------------------------------------------------------------------ files
def array_sha256(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def code_sha256(files=_CODE):
    return {f: hashlib.sha256((rb_paths.ROUND / f).read_bytes()).hexdigest() for f in files}


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.json')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def export_path(outputs, recipe, aspect):
    return Path(outputs) / 'part1' / 'train' / recipe / f'{aspect}__s{SEED}__P.npz'


def vp2_problem(x, n_rows=N_P):
    """VP2 on one export's image array: None, or the first problem (rows, dtype, finite, unit norm)."""
    if x.dtype != np.float16 or x.shape != (n_rows, WIDTH):
        return f'shape/dtype {x.shape} {x.dtype}, expected ({n_rows}, {WIDTH}) float16'
    if not np.isfinite(x).all():
        return 'nonfinite rows'
    err = float(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1).max())
    if not err < rb_ami.NORM_TOL:
        return f'norm error {err} (limit {rb_ami.NORM_TOL})'
    return None


def load_export(outputs, recipe, aspect, p_sha):
    """The image array of a recipe's P export, after checking the trainer's record: P sha256, names, array hashes."""
    path = export_path(outputs, recipe, aspect)
    record = json.loads(path.with_name(f'{aspect}__s{SEED}.json').read_text())
    assert record['recipe'] == recipe and record['aspect'] == aspect and record['seed'] == SEED, (recipe, aspect)
    assert record['p_exports'] is True, f'{recipe}/{aspect}: the record lists no P exports'
    assert record['p_sha256'] == p_sha, f'{recipe}/{aspect}: P sha256 of the export differs from P.json'
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == ['cap', 'img'], (path, z.files)
        arrays = {k: z[k] for k in z.files}
    listed = record['files'][path.name]
    for key, x in arrays.items():
        want = listed[key]
        assert list(x.shape) == want['shape'] and str(x.dtype) == want['dtype'], f'{path}: {key} shape/dtype'
        assert array_sha256(x) == want['sha256'], f'{path}: {key} sha256 differs from the trainer record'
    return arrays['img']


def _identity(recipe, aspect, p_sha, source_sha):
    versions = rb_ami.versions()
    return dict(recipe=recipe, aspect=aspect, p_sha256=p_sha, img_sha256=source_sha,
                sklearn=versions['sklearn'], numpy=versions['numpy'], rb_ami_sha256=code_sha256(('rb_ami.py',))['rb_ami.py'])


def fixed_partitions(outputs, recipe, aspect, img, p_sha):
    """int64 [3, n]: the recipe's fixed partitions of one aspect, computed once and saved as
    `part1/partitions/<recipe>/<aspect>.npz` (key `labels`) with a sibling `.json`; a saved set is reused only if
    its inputs (export array, P, versions, rb_ami.py) are the same, else the run stops."""
    path = Path(outputs) / 'part1' / 'partitions' / recipe / f'{aspect}.npz'
    meta_path = path.with_suffix('.json')
    identity = _identity(recipe, aspect, p_sha, array_sha256(img))
    if path.exists() or meta_path.exists():
        assert path.exists() and meta_path.exists(), f'{path}: partial partition file'
        meta = json.loads(meta_path.read_text())
        assert meta['identity'] == identity, f'{path}: saved partitions came from other inputs; move them aside'
        with np.load(path, allow_pickle=False) as z:
            assert z.files == ['labels']
            labels = z['labels']
        want = meta['arrays']['labels']
        assert list(labels.shape) == want['shape'] and str(labels.dtype) == want['dtype']
        assert array_sha256(labels) == want['sha256'], f'{path}: sha256 differs from its record'
    else:
        labels = rb_ami.partitions(img)
        path.parent.mkdir(parents=True, exist_ok=True)
        pending = path.with_name(path.stem + '.pending.npz')
        np.savez(pending, labels=labels)
        pending.replace(path)
        _atomic_json(meta_path, dict(
            identity=identity, arrays={'labels': dict(shape=list(labels.shape), dtype=str(labels.dtype),
                                                      sha256=array_sha256(labels))},
            code_sha256=code_sha256(), source=dict(p_sha256=p_sha, img_sha256=identity['img_sha256']),
            versions=rb_ami.versions(), threads=rb_ami.THREADS, k=rb_ami.K, states=list(rb_ami.STATES)))
    assert labels.shape == (3, len(img)) and labels.dtype == np.int64
    return labels


# ------------------------------------------------------------------ the pick
def run(outputs, labels, *, expect_names=None, n_draws=rb_ami.N_DRAWS, workers=8, smoke=False, n_rows=N_P,
        vp2_failed=None):
    """Partitions, effects and the outcome of part 1; writes `part1/pick.json` and returns it.

    `labels` = {aspect: int64 [n_rows]} in P order (-1 masked): `rb_data.p_labels()`. `expect_names` (the real P)
    is checked against `part1/P.json`. `vp2_failed` = {recipe: cause} for candidates that still fail VP2 after their
    forced rerun (rule VP3): excluded like an in-file VP2 failure, exports not loaded (they may not exist). R0-5 in
    it stops part 1.
    """
    vp2_failed = dict(vp2_failed or {})
    assert REFERENCE not in vp2_failed, f'{REFERENCE} fails VP2, part 1 stops (rule VP3): {vp2_failed[REFERENCE]}'
    assert set(vp2_failed) <= set(CANDIDATES), f'not candidates: {sorted(set(vp2_failed) - set(CANDIDATES))}'
    assert all(isinstance(c, str) and c for c in vp2_failed.values()), 'every VP2 failure needs a cause'
    outputs = Path(outputs)
    names = rb_data.read_p_json(outputs, expect=expect_names)
    assert len(names) == n_rows, (len(names), n_rows)
    p_sha = rb_data.names_sha256(names)
    assert set(labels) == set(ASPECTS)
    for a in ASPECTS:
        assert labels[a].dtype == np.int64 and labels[a].shape == (n_rows,) and labels[a].min() >= -1
        assert (labels[a] >= 0).any(), a
    problems, images = {n: [c] for n, c in vp2_failed.items()}, {}
    for name in rb_rows.ORDER:
        if name in vp2_failed:
            continue
        for aspect in ASPECTS:
            img = load_export(outputs, name, aspect, p_sha)
            problem = vp2_problem(img, n_rows)
            if problem:
                problems.setdefault(name, []).append(f'{aspect}: {problem}')
            elif name not in problems:
                images[name, aspect] = img
    assert REFERENCE not in problems, f'{REFERENCE} fails VP2, part 1 stops (rule VP3): {problems[REFERENCE]}'
    scored = [n for n in rb_rows.ORDER if n not in problems]
    parts = {a: {n: fixed_partitions(outputs, n, a, images[n, a], p_sha) for n in scored} for a in ASPECTS}
    per_aspect = {a: aspect_effects(labels[a], parts[a], n_draws, workers) for a in ASPECTS}
    recipes = {}
    for n in rb_rows.ORDER:
        recipes[n] = None if n in problems else {
            a: dict(per_aspect[a][n], gain=gain(per_aspect[a][n]['lo'], per_aspect[a][n]['e'], a)) for a in ASPECTS}
    effects = {n: recipes[n] for n in CANDIDATES}
    candidates, outcome, recipe = decide(effects, {n: v for n, v in problems.items() if n in CANDIDATES})
    pick_json = dict(
        p_sha256=p_sha, n_rows=n_rows, smoke=bool(smoke), n_draws=n_draws, versions=rb_ami.versions(),
        threads=rb_ami.THREADS, seed_ranges=dict(SEED_RANGES), reference=REFERENCE, tie_order=list(CANDIDATES),
        label_counts={a: int((labels[a] >= 0).sum()) for a in ASPECTS},
        label_sha256={a: array_sha256(labels[a]) for a in ASPECTS},      # int64 bytes in P order
        vp2_failed=vp2_failed, triggers=triggers(outcome, recipe),
        recipes=recipes, candidates=candidates, outcome=outcome, recipe=recipe,
        flag_g_le_0=bool(recipe != REFERENCE and candidates[recipe]['flag_g_le_0']),
        sources={n: {a: array_sha256(images[n, a]) for a in ASPECTS} for n in scored}, code_sha256=code_sha256())
    pick_json = json.loads(json.dumps(pick_json, allow_nan=False))
    path = outputs / 'part1' / 'pick.json'
    if path.exists():
        assert json.loads(path.read_text()) == pick_json, f'{path} exists with other results; move it aside'
    else:
        _atomic_json(path, pick_json)
    return pick_json


def parse_vp2_failed(items):
    """['RECIPE=CAUSE', ...] -> {recipe: cause}; the cause is everything after the first '='."""
    out = {}
    for item in items:
        recipe, sep, cause = item.partition('=')
        assert sep and recipe and cause, f'--vp2-failed wants RECIPE=CAUSE, got {item!r}'
        assert recipe not in out, f'--vp2-failed twice for {recipe}'
        out[recipe] = cause
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='Part 1: fixed partitions, effects against R0-5 and the pick.')
    parser.add_argument('--outputs', type=Path, default=rb_paths.OUTPUTS)
    parser.add_argument('--smoke', action='store_true', help='20 bootstrap draws; outputs under /tmp/claude-0/')
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    parser.add_argument('--vp2-failed', action='append', default=[], metavar='RECIPE=CAUSE',
                        help='a candidate still failing VP2 after its forced rerun (repeatable); not scored, never passes')
    args = parser.parse_args(argv)
    vp2_failed = parse_vp2_failed(args.vp2_failed)
    if args.smoke:
        assert str(args.outputs.resolve()).startswith('/tmp/claude-0/'), 'smoke needs --outputs under /tmp/claude-0/'
    rb_paths.check_imports()
    names = rb_data.p_names()
    assert rb_data.names_sha256(names) == rb_data.p_sha256()
    pick_json = run(args.outputs, rb_data.p_labels(), expect_names=names, n_draws=20 if args.smoke else rb_ami.N_DRAWS,
                    workers=args.workers, smoke=args.smoke, vp2_failed=vp2_failed)
    print(f"{args.outputs / 'part1' / 'pick.json'}: outcome {pick_json['outcome']}, recipe {pick_json['recipe']}"
          f"{' (g <= 0: flagged)' if pick_json['flag_g_le_0'] else ''}", flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
