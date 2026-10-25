"""Stage iii scorer, seed 0 (rule §3.3 and §4.3): each variant's P exports against R6-5, the guard, the closest method
and the extras, the winner and the III outcome, and the VG1 scorer half.

Modes (`python eg_score3.py MODE`):
  first       the five first variants (`eg_variants.FIRST`): guard, closest method, w*, the extras
              -> stage_iii/score_first.json
  all         the five and the two extras of score_first.json: winner, III outcome, fresh-seeds trigger, kept weights
              -> stage_iii/score_all.json (VG4: any other variant set is refused)
  check vg1   the scorer gives pick.json's R6-5 values and T_g within 1e-6 -> stage_iii/vg1_scorer.json

Statistic: the RB rule §3.1 with R6-5 in R0-5's place (`rb_ami` partitions and A, `rb_pick.bootstrap_scores`
draws). Decisions use unrounded values. Caches (fixed partitions, per-variant bootstrap scores) go under
`<outputs>/stage_iii/partitions/<variant>/s0/`; the frozen 12-24 outputs are only read.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

import eg_paths
import eg_variants as V
import rb_ami
import rb_data
import rb_pick

ASPECTS = eg_paths.ASPECTS
BASE = V.BASE
SEED = 0
RANGES = V.RANGES
PROMISING = V.T_G - V.RANGES['genre']             # T_g - r_genre, unrounded (0.4861162...)
TOL = 1e-6                                        # VG1


# ------------------------------------------------------------------ the rule's conditions (pure)
def is_guarded(e_emotion, e_style):
    """Guard (rule §4.3): e_emotion >= -r_emotion and e_style >= -r_style, point estimates."""
    return bool(e_emotion >= -RANGES['emotion'] and e_style >= -RANGES['style'])


def condition_record(effects):
    """A scored variant: `effects` = {aspect: {A, e, lo, hi}} -> the record the conditions read."""
    return dict(status='ok', cause=None, effects=effects, A_genre=effects['genre']['A'], e_genre=effects['genre']['e'],
                guarded=is_guarded(effects['emotion']['e'], effects['style']['e']))


def failed_record(cause):
    """A failed run (rule §4.3 "Failed runs"): no A_genre, not guarded, counts as run for its ladder."""
    assert cause, 'a failed run needs its cause'
    return dict(status='failed', cause=str(cause), effects=None, A_genre=None, e_genre=None, guarded=False)


def _best_in_order(results, names):
    """The name with the largest A_genre (unrounded); on equal values the first in `names`."""
    best = None
    for n in names:
        if best is None or results[n]['A_genre'] > results[best]['A_genre']:
            best = n
    return best


def closest_variant(first):
    """The first variant whose method is the closest: the guarded one with the largest A_genre among the five; if
    none is guarded, the largest A_genre among the five; ties in the order of `FIRST`. None if every run failed."""
    assert set(first) == set(V.FIRST), sorted(first)
    ran = [v for v in V.FIRST if first[v]['A_genre'] is not None]
    pool = [v for v in ran if first[v]['guarded']] or ran
    return _best_in_order(first, pool) if pool else None


def extras_of(variant, guarded, run=V.FIRST):
    """The two extra variants for the closest method (names, ladder order). `variant` is w*'s variant, `guarded` its
    guard; `run` = the variants run so far (failed ones count). w* the largest rung run and guarded: the next two
    rungs above. Otherwise the nearest rung not yet run below w* and the nearest above; a ladder end gives none."""
    method, w = V.parse(variant)
    ladder = V.LADDERS[method]
    ran = {V.parse(v)[1] for v in run if V.parse(v)[0] == method}
    assert w in ran, f'{variant} is not among the runs'
    i = ladder.index(w)
    if guarded and w == max(ran):
        rungs = list(ladder[i + 1:i + 3])
    else:
        below = [r for r in reversed(ladder[:i]) if r not in ran][:1]
        above = [r for r in ladder[i + 1:] if r not in ran][:1]
        rungs = below + above
    return [V.name(method, r) for r in rungs]


def decide_first(first):
    """Closest method, w* and the extras from the five first variants' records."""
    v = closest_variant(first)
    if v is None:
        return dict(closest=None, method=None, rung=None, guarded_pool=False, closest_guarded=None, extras=[])
    method, rung = V.parse(v)
    return dict(closest=v, method=method, rung=rung, guarded_pool=any(first[x]['guarded'] for x in V.FIRST),
                closest_guarded=bool(first[v]['guarded']), extras=extras_of(v, first[v]['guarded']))


def pick_winner(results, extras):
    """v*: among the five and the extras, the guarded variant with the largest A_genre (unrounded); ties in the
    order FIRST then `extras` as given (ladder order). None if no variant is guarded."""
    order = list(V.FIRST) + list(extras)
    assert set(results) == set(order), sorted(results)
    pool = [v for v in order if results[v]['status'] == 'ok' and results[v]['guarded']]
    return _best_in_order(results, pool) if pool else None


def outcome(winner, a_genre):
    """III-GO: a winner with A_genre >= T_g; III-PROM: T_g - r_genre <= A_genre < T_g; III-NONE otherwise."""
    if winner is None:
        return 'III-NONE'
    if a_genre >= V.T_G:
        return 'III-GO'
    return 'III-PROM' if a_genre >= PROMISING else 'III-NONE'


def fresh_trigger(winner, e_genre):
    """Fresh seeds 1 to 3 run when a winner exists and its point gain e_genre >= r_genre."""
    return bool(winner is not None and e_genre >= RANGES['genre'])


def split_weights(results):
    """(kept, rest): the variants with A_genre >= T_g - r_genre keep their weights; the rest (failed runs too) go on
    the pending deletions list for the wrap-up. Names in the order of `results`."""
    kept = [v for v, r in results.items() if r['A_genre'] is not None and r['A_genre'] >= PROMISING]
    return kept, [v for v in results if v not in kept]


def check_vg4(variants, extras):
    """VG4: the scored variant set is FIRST plus the two extras of score_first.json, nothing else."""
    variants, extras = list(variants), list(extras)
    assert len(set(extras)) == len(extras) <= 2 and not set(extras) & set(V.FIRST), extras
    for e in extras:
        V.parse(e)
    assert sorted(variants) == sorted(list(V.FIRST) + extras), \
        f'VG4: variant set {sorted(variants)} is not FIRST plus the extras {extras}'


# ------------------------------------------------------------------ effects against R6-5
class _NoCache:
    def get(self, name):
        return None

    def put(self, name, boot):
        pass


def aspect_effects(labels, parts, reference, n_draws=rb_ami.N_DRAWS, workers=8, cache=None):
    """{name: {A, e, lo, hi}} for one aspect with `reference` in R0-5's place (`rb_pick.aspect_effects` otherwise):
    A on all of L_a, e = A - A(reference), the interval from the shared draws. `cache.get/put(name)` hold per-arm
    bootstrap scores (each draw is scored per arm alone, so an arm's scores do not depend on the other arms)."""
    assert reference in parts
    cache = cache or _NoCache()
    labels = np.asarray(labels)
    labelled = np.flatnonzero(labels >= 0)
    point = {n: rb_ami.score(labels, p, labelled) for n, p in parts.items()}
    boot = {n: cache.get(n) for n in parts}
    missing = {n: parts[n] for n in parts if boot[n] is None}
    if missing:
        for n, b in rb_pick.bootstrap_scores(labels, missing, n_draws, workers).items():
            cache.put(n, b)
            boot[n] = b
    assert all(b.shape == (n_draws,) and np.isfinite(b).all() for b in boot.values())
    out = {}
    for n in parts:
        lo, hi = np.percentile(boot[n] - boot[reference], [2.5, 97.5])        # default linear method
        out[n] = dict(A=float(point[n]), e=float(point[n] - point[reference]), lo=float(lo), hi=float(hi))
    return out


# ------------------------------------------------------------------ files
def _json_dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.json')
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    pending.replace(path)


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_state():
    """git HEAD of this checkout, whether tracked files differ, and the scorer's own hash."""
    def git(*a):
        return subprocess.run(['git', '-C', str(eg_paths.ROUND), *a], capture_output=True, text=True, check=True).stdout
    return dict(code_commit=git('rev-parse', 'HEAD').strip(), code_dirty=bool(git('status', '--porcelain', '-uno').strip()),
                code_sha256=_file_sha256(Path(__file__)))


def stage_dir(outputs):
    return Path(outputs) / 'stage_iii'


def cache_dir(outputs, variant):
    return stage_dir(outputs) / 'partitions' / variant / f's{SEED}'


def _write_npz_with_record(path, key, array, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.stem + '.pending.npz')
    np.savez(pending, **{key: array})
    pending.replace(path)
    _json_dump(path.with_suffix('.json'), dict(record, array=dict(shape=list(array.shape), dtype=str(array.dtype),
                                                                   sha256=rb_pick.array_sha256(array))))


def _read_npz_with_record(path, key, identity):
    """The cached array if `path` exists with the same identity and its recorded hash; None if absent."""
    meta_path = path.with_suffix('.json')
    if not (path.exists() or meta_path.exists()):
        return None
    assert path.exists() and meta_path.exists(), f'{path}: partial cache file'
    meta = json.loads(meta_path.read_text())
    assert meta['identity'] == identity, f'{path}: cached from other inputs; move it aside'
    with np.load(path, allow_pickle=False) as z:
        assert z.files == [key]
        x = z[key]
    want = meta['array']
    assert list(x.shape) == want['shape'] and str(x.dtype) == want['dtype'] and rb_pick.array_sha256(x) == want['sha256'], \
        f'{path}: differs from its record'
    return x


def _identity(variant, aspect, p_sha, img_sha, **extra):
    versions = rb_ami.versions()
    return dict(variant=variant, aspect=aspect, seed=SEED, p_sha256=p_sha, img_sha256=img_sha,
                sklearn=versions['sklearn'], numpy=versions['numpy'],
                code_sha256=rb_pick.code_sha256(('rb_ami.py', 'rb_pick.py')), **extra)


def fixed_partitions(outputs, variant, aspect, img, p_sha):
    """int64 [3, n]: the variant's fixed partitions of one aspect (RB rule §3.1), computed once and cached."""
    path = cache_dir(outputs, variant) / f'{aspect}__partitions.npz'
    identity = _identity(variant, aspect, p_sha, rb_pick.array_sha256(img))
    parts = _read_npz_with_record(path, 'labels', identity)
    if parts is None:
        parts = rb_ami.partitions(img)
        _write_npz_with_record(path, 'labels', parts, dict(identity=identity, versions=rb_ami.versions(),
                                                           k=rb_ami.K, states=list(rb_ami.STATES)))
    assert parts.shape == (3, len(img)) and parts.dtype == np.int64
    return parts


class DiskBoots:
    """Per-arm bootstrap scores of one aspect, cached under the variant's folder."""

    def __init__(self, outputs, aspect, n_draws, p_sha, label_sha, img_sha):
        self.outputs, self.aspect, self.n_draws = outputs, aspect, n_draws
        self.p_sha, self.label_sha, self.img_sha = p_sha, label_sha, img_sha

    def _where(self, name):
        path = cache_dir(self.outputs, name) / f'{self.aspect}__boot.npz'
        return path, _identity(name, self.aspect, self.p_sha, self.img_sha[name], label_sha256=self.label_sha,
                               n_draws=self.n_draws)

    def get(self, name):
        path, identity = self._where(name)
        return _read_npz_with_record(path, 'boot', identity)

    def put(self, name, boot):
        path, identity = self._where(name)
        _write_npz_with_record(path, 'boot', np.asarray(boot, dtype=np.float64), dict(identity=identity))


# ------------------------------------------------------------------ inputs
def load_context():
    """P's names (hash = 12-24's), the labels (hashes = pick.json's) and pick.json; the frozen files are only read."""
    names = rb_data.p_names()
    p_sha = rb_data.names_sha256(names)
    assert p_sha == eg_paths.P_SHA256, 'P differs from 12-24\'s'
    assert rb_data.read_p_json(eg_paths.RB_OUT, expect=names) == names
    pick = json.loads((eg_paths.RB_PART1 / 'pick.json').read_text())
    assert pick['p_sha256'] == p_sha and pick['n_rows'] == rb_data.N_P
    labels = rb_data.p_labels()
    for a in ASPECTS:
        assert rb_pick.array_sha256(labels[a]) == pick['label_sha256'][a], f'{a}: labels differ from pick.json\'s'
        assert int((labels[a] >= 0).sum()) == pick['label_counts'][a]
    return dict(names=names, p_sha=p_sha, labels=labels, pick=pick)


def base_images(p_sha):
    """R6-5's seed-0 P exports (12-24, frozen), after the trainer record's checks and VG3."""
    out = {}
    for a in ASPECTS:
        img = rb_pick.load_export(eg_paths.RB_OUT, BASE, a, p_sha)
        problem = rb_pick.vp2_problem(img)
        assert problem is None, f'{BASE}/{a}: {problem}'
        out[a] = img
    return out


def read_run(outputs, variant, p_sha, declared_failed=None):
    """{status, cause, best_epoch, run_sha256, img {aspect: array}} of one seed-0 variant run. A run.json with
    status 'failed' has no exports read. A variant named in `declared_failed` (rule §4.3: VG3 still failing after its
    rerun) has its exports read and must fail VG3 on at least one aspect; the problems are kept as `vg3_problem`
    beside the declared cause, and the record is a failed run."""
    declared_failed = declared_failed or {}
    folder = eg_paths.iii_run(outputs, variant, SEED)
    run_path = folder / 'run.json'
    assert run_path.exists(), f'{variant}: {run_path} missing (the run has not finished)'
    run = json.loads(run_path.read_text())
    assert run['variant'] == variant and run['seed'] == SEED, (variant, run.get('variant'), run.get('seed'))
    assert run['switches'] == V.switches(variant), f'{variant}: run.json switches {run["switches"]}'
    assert run['status'] in ('ok', 'failed'), (variant, run['status'])
    assert p_sha == eg_paths.P_SHA256, 'P differs from 12-24\'s'
    assert run.get('p_sha256') == p_sha, f'{variant}: run.json p_sha256 {run.get("p_sha256")!r} is not the P of 12-24'
    rec = dict(status=run['status'], cause=run.get('cause'), best_epoch=run.get('best_epoch'),
               run_sha256=_file_sha256(run_path), img=None)
    if run['status'] == 'failed':
        assert run.get('cause'), f'{variant}: failed run without a cause'
        return rec
    img, problems = {}, {}
    for a in ASPECTS:
        with np.load(folder / f'{a}__P.npz', allow_pickle=False) as z:
            assert 'img' in z.files, (variant, a, z.files)
            img[a] = z['img']
        problem = rb_pick.vp2_problem(img[a])                     # VG3: 29,232 finite unit-norm rows
        if problem is not None:
            problems[a] = problem
    if variant in declared_failed:
        assert problems, f'VG3 {variant}: --failed declared it, but its exports pass VG3 on every aspect (rule §4.3)'
        return dict(rec, status='failed', cause=declared_failed[variant], vg3_problem=problems)
    assert not problems, f'VG3 {variant}: {problems} (rerun it, or declare the failure with --failed)'
    return dict(rec, img=img)


def check_variant_dirs(outputs, allowed):
    """VG4: no seed-0 run folder of a variant outside `allowed` (the base's folder is the fresh seeds' and VG1's)."""
    root = stage_dir(outputs) / 'train'
    found = sorted(p.parent.name for p in root.glob(f'*/s{SEED}')) if root.exists() else []
    extra = [v for v in found if v not in set(allowed) | {BASE}]
    assert not extra, f'VG4: run folders of variants outside FIRST and the extras: {extra}'


# ------------------------------------------------------------------ scoring
def score_variants(outputs, variants, n_draws, workers, declared_failed=None, ctx=None):
    """{'base': {aspect: A}, 'results': {variant: record}, 'identity': {...}} for `variants` against R6-5."""
    declared_failed = declared_failed or {}
    assert set(declared_failed) <= set(variants), \
        f'--failed names {sorted(set(declared_failed) - set(variants))}, which are not among the scored {list(variants)}'
    ctx = ctx or load_context()
    p_sha, labels = ctx['p_sha'], ctx['labels']
    base = base_images(p_sha)
    runs = {v: read_run(outputs, v, p_sha, declared_failed) for v in variants}
    ok = [v for v in variants if runs[v]['status'] == 'ok']
    label_sha = {a: rb_pick.array_sha256(labels[a]) for a in ASPECTS}
    img_sha = {BASE: {a: rb_pick.array_sha256(base[a]) for a in ASPECTS}}
    img_sha.update({v: {a: rb_pick.array_sha256(runs[v]['img'][a]) for a in ASPECTS} for v in ok})
    identity = dict(p_sha256=p_sha, label_sha256=label_sha, n_draws=n_draws, variants=list(variants),
                    img_sha256=img_sha, run_sha256={v: runs[v]['run_sha256'] for v in variants},
                    failed={v: runs[v]['cause'] for v in variants if runs[v]['status'] == 'failed'})
    effects = {v: {} for v in ok}
    base_A = {}
    for a in ASPECTS:
        imgs = {BASE: base[a], **{v: runs[v]['img'][a] for v in ok}}
        parts = {n: fixed_partitions(outputs, n, a, imgs[n], p_sha) for n in imgs}
        cache = DiskBoots(outputs, a, n_draws, p_sha, label_sha[a], {n: img_sha[n][a] for n in imgs})
        eff = aspect_effects(labels[a], parts, BASE, n_draws, workers, cache)
        base_A[a] = eff[BASE]['A']
        for v in ok:
            effects[v][a] = eff[v]
        print(f'{a}: scored {len(imgs)} arms', flush=True)
    results = {}
    for v in variants:
        if runs[v]['status'] == 'failed':
            rec = dict(failed_record(runs[v]['cause']), **({'vg3_problem': runs[v]['vg3_problem']}
                                                           if 'vg3_problem' in runs[v] else {}))
        else:
            rec = condition_record(effects[v])
        results[v] = dict(rec, best_epoch=runs[v]['best_epoch'])
    return dict(base=base_A, results=results, identity=identity)


def _header(ctx, n_draws, smoke):
    return dict(smoke=bool(smoke), n_draws=n_draws, reference=BASE, seed=SEED, t_g=V.T_G, promising=PROMISING,
                ranges=dict(RANGES), versions=rb_ami.versions(), label_counts={a: int((ctx['labels'][a] >= 0).sum()) for a in ASPECTS},
                **code_state())


def _write_or_skip(path, record):
    """Write `record`; an existing file with the same identity is kept (never overwritten), another one stops."""
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text())
        assert old['identity'] == record['identity'], f'{path} exists with other inputs; move it aside'
        print(f'{path}: exists with the same inputs, kept', flush=True)
        return old
    _json_dump(path, record)
    return record


def run_first(outputs, n_draws, workers, smoke=False, declared_failed=None):
    path = stage_dir(outputs) / 'score_first.json'
    check_variant_dirs(outputs, V.FIRST + tuple(json.loads(path.read_text())['extras'] if path.exists() else ()))
    ctx = load_context()
    scored = score_variants(outputs, list(V.FIRST), n_draws, workers, declared_failed, ctx)
    d = decide_first(scored['results'])
    record = dict(stage='first', **_header(ctx, n_draws, smoke), base_A=scored['base'], variants=list(V.FIRST),
                  results=scored['results'], **d, identity=scored['identity'])
    return _write_or_skip(path, json.loads(json.dumps(record, allow_nan=False)))


def run_all(outputs, n_draws, workers, smoke=False, declared_failed=None, variants=None):
    first_path = stage_dir(outputs) / 'score_first.json'
    assert first_path.exists(), f'{first_path} missing: run `first` before `all`'
    first = json.loads(first_path.read_text())
    assert first['stage'] == 'first' and first['n_draws'] == n_draws and first['smoke'] == bool(smoke)
    extras = first['extras']
    scored_set = list(V.FIRST) + list(extras)
    if variants is not None:
        check_vg4(variants, extras)
    check_vg4(scored_set, extras)
    check_variant_dirs(outputs, scored_set)
    ctx = load_context()
    assert first['identity']['p_sha256'] == ctx['p_sha']
    scored = score_variants(outputs, scored_set, n_draws, workers, declared_failed, ctx)
    results = scored['results']
    again = decide_first({v: results[v] for v in V.FIRST})
    assert again['extras'] == extras, f'extras differ from score_first.json: {again["extras"]} vs {extras}'
    for v in V.FIRST:                                   # the five were scored by `first` with the same code
        old = first['results'][v]
        assert old['status'] == results[v]['status'], v
        if old['A_genre'] is not None:
            assert abs(old['A_genre'] - results[v]['A_genre']) < 1e-9, f'{v}: A_genre differs from score_first.json'
    winner = pick_winner(results, extras)
    a_win = results[winner]['A_genre'] if winner else None
    e_win = results[winner]['e_genre'] if winner else None
    kept, rest = split_weights(results)
    record = dict(stage='all', **_header(ctx, n_draws, smoke), base_A=scored['base'], variants=scored_set, extras=extras,
                  results=results, winner=winner, A_genre_winner=a_win, e_genre_winner=e_win,
                  outcome=outcome(winner, a_win), fresh_seeds=fresh_trigger(winner, e_win),
                  kept_weights=kept, pending_deletions=rest, identity=scored['identity'])
    return _write_or_skip(stage_dir(outputs) / 'score_all.json', json.loads(json.dumps(record, allow_nan=False)))


# ------------------------------------------------------------------ VG1, scorer half
def run_check_vg1(outputs):
    """The scorer's R6-5 A values equal pick.json's and its T_g value equals `eg_variants.T_G` within 1e-6."""
    import rb_part1_report
    path = stage_dir(outputs) / 'vg1_scorer.json'
    ctx = load_context()
    labels, p_sha, pick = ctx['labels'], ctx['p_sha'], ctx['pick']
    base = base_images(p_sha)
    items, same_parts = {}, {}
    for a in ASPECTS:
        parts = fixed_partitions(outputs, BASE, a, base[a], p_sha)
        value = rb_ami.score(labels[a], parts, np.flatnonzero(labels[a] >= 0))
        want = pick['recipes'][BASE][a]['A']
        items[f'{BASE}_{a}'] = dict(value=float(value), expected=want, diff=float(abs(value - want)),
                                    **{'pass': bool(abs(value - want) <= TOL)})
        with np.load(eg_paths.RB_PART1 / 'partitions' / BASE / f'{a}.npz', allow_pickle=False) as z:
            same_parts[a] = bool(np.array_equal(z['labels'], parts))
        print(f'{a}: A {value:.10f} (pick.json {want:.10f})', flush=True)
    reported = json.loads((eg_paths.RB_PART1 / 'reported.json').read_text())['references_p']['genre_word']['genre']
    assert reported == V.T_G, 'eg_variants.T_G differs from reported.json'
    value = rb_part1_report.kmeans_ami(rb_data.p_instructed_images('genre_word'), labels['genre'])
    items['T_g'] = dict(value=float(value), expected=V.T_G, diff=float(abs(value - V.T_G)),
                        **{'pass': bool(abs(value - V.T_G) <= TOL)})
    print(f'T_g {value:.10f} (eg_variants.T_G {V.T_G:.10f})', flush=True)
    record = dict(stage='vg1_scorer', **{'pass': all(i['pass'] for i in items.values())}, tol=TOL, items=items,
                  partitions_equal_to_12_24=same_parts, p_sha256=p_sha, versions=rb_ami.versions(), **code_state())
    record = json.loads(json.dumps(record, allow_nan=False))
    if path.exists():
        old = json.loads(path.read_text())
        assert old['pass'] == record['pass'] and all(abs(old['items'][k]['value'] - v['value']) < 1e-9
                                                      for k, v in items.items()), f'{path} exists with other values'
        print(f'{path}: exists with the same values, kept', flush=True)
    else:
        _json_dump(path, record)
    return record


# ------------------------------------------------------------------ CLI
def parse_failed(items, scored=None):
    """['VARIANT=CAUSE', ...] -> {variant: cause}: a variant still failing VG3 after its rerun. With `scored` (the
    variants this mode scores) a variant outside it is refused."""
    out = {}
    for item in items:
        variant, sep, cause = item.partition('=')
        assert sep and cause, f'--failed wants VARIANT=CAUSE, got {item!r}'
        V.parse(variant)
        assert scored is None or variant in scored, f'--failed {variant}: not among the scored variants {list(scored)}'
        assert variant not in out, f'--failed twice for {variant}'
        out[variant] = cause
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description='Stage iii scorer, seed 0.')
    parser.add_argument('mode', choices=['first', 'all', 'check'])
    parser.add_argument('what', nargs='?', choices=['vg1'], help='for `check`')
    parser.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
    parser.add_argument('--smoke', action='store_true', help='20 bootstrap draws; outputs under /tmp/claude-0/')
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    parser.add_argument('--failed', action='append', default=[], metavar='VARIANT=CAUSE',
                        help='a variant still failing VG3 after its forced rerun (repeatable)')
    parser.add_argument('--variants', nargs='*', help='`all`: the variant set to check against VG4 (default: derived)')
    args = parser.parse_args(argv)
    assert (args.mode == 'check') == (args.what == 'vg1'), 'usage: check vg1 | first | all'
    if args.smoke:
        assert eg_paths.smoke_ok(args.outputs), 'smoke needs --outputs under /tmp/claude-0/'
    eg_paths.check_imports()
    n_draws = 20 if args.smoke else rb_ami.N_DRAWS
    scored = list(V.FIRST)
    if args.mode == 'all':
        first_path = stage_dir(args.outputs) / 'score_first.json'
        scored += json.loads(first_path.read_text())['extras'] if first_path.exists() else []
    failed = parse_failed(args.failed, scored if args.mode != 'check' else None)
    assert not (args.mode == 'check' and failed), '--failed does not apply to `check vg1`'
    if args.mode == 'check':
        rec = run_check_vg1(args.outputs)
        print(f"{stage_dir(args.outputs) / 'vg1_scorer.json'}: VG1 scorer half {'PASS' if rec['pass'] else 'FAIL'}; "
              + ', '.join(f"{k} diff {v['diff']:.2e}" for k, v in rec['items'].items()))
    elif args.mode == 'first':
        rec = run_first(args.outputs, n_draws, args.workers, args.smoke, failed)
        print(f"{stage_dir(args.outputs) / 'score_first.json'}: closest {rec['closest']} "
              f"({'guarded' if rec['closest_guarded'] else 'not guarded'}), extras {' '.join(rec['extras']) or 'none'}")
    else:
        rec = run_all(args.outputs, n_draws, args.workers, args.smoke, failed, args.variants)
        print(f"{stage_dir(args.outputs) / 'score_all.json'}: winner {rec['winner']}, {rec['outcome']}, "
              f"A_genre {rec['A_genre_winner']}, fresh seeds {'yes' if rec['fresh_seeds'] else 'no'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
