"""Stage iii scorer, seed 0 (rule §3.3 and §4.3): each variant's P exports against R6-5, the guard, the closest method
and the extras, the winner and the III outcome, and the VG1 scorer half.

Modes (`python eg_score3.py MODE`):
  first       the five first variants (`eg_variants.FIRST`): guard, closest method, w*, the extras
              -> stage_iii/score_first.json
  all         the five and the two extras of score_first.json: winner, III outcome, fresh-seeds trigger, kept weights
              -> stage_iii/score_all.json (VG4: any other variant set is refused)
  fresh       the winner and R6-5 on seeds 1 to 3: the C outcome (rule §3.3, §4.3), after `all` (trigger) and ticket 11's
              fresh_parity.json -> stage_iii/score_fresh.json; --failed VARIANT=CAUSE for a seed whose exports still fail VG3
  report      every variant's reported readings (intervals on P, old 5,000, Leiden, best epochs, GA adaptor, references,
              the fresh result) -> stage_iii/reported.{json,md}, after `all` (and `fresh` when triggered)
  check vg1   the scorer gives pick.json's R6-5 values and T_g within 1e-6 -> stage_iii/vg1_scorer.json

Statistic: the RB rule §3.1 with R6-5 in R0-5's place (`rb_ami` partitions and A, `rb_pick.bootstrap_scores`
draws). Decisions use unrounded values. Caches (fixed partitions, per-variant bootstrap scores) go under
`<outputs>/stage_iii/partitions/<variant>/s0/`; the frozen 12-24 outputs are only read.
"""
import argparse
import hashlib
import json
import multiprocessing
import subprocess
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

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


def cache_dir(outputs, variant, seed=SEED):
    return stage_dir(outputs) / 'partitions' / variant / f's{seed}'


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


def _identity(variant, aspect, p_sha, img_sha, seed=SEED, **extra):
    versions = rb_ami.versions()
    return dict(variant=variant, aspect=aspect, seed=seed, p_sha256=p_sha, img_sha256=img_sha,
                sklearn=versions['sklearn'], numpy=versions['numpy'],
                code_sha256=rb_pick.code_sha256(('rb_ami.py', 'rb_pick.py')), **extra)


def fixed_partitions(outputs, variant, aspect, img, p_sha, seed=SEED):
    """int64 [3, n]: the variant's fixed partitions of one aspect (RB rule §3.1), computed once and cached."""
    path = cache_dir(outputs, variant, seed) / f'{aspect}__partitions.npz'
    identity = _identity(variant, aspect, p_sha, rb_pick.array_sha256(img), seed)
    parts = _read_npz_with_record(path, 'labels', identity)
    if parts is None:
        parts = rb_ami.partitions(img)
        _write_npz_with_record(path, 'labels', parts, dict(identity=identity, versions=rb_ami.versions(),
                                                           k=rb_ami.K, states=list(rb_ami.STATES)))
    assert parts.shape == (3, len(img)) and parts.dtype == np.int64
    return parts


class DiskBoots:
    """Per-arm bootstrap scores of one aspect at one seed, cached under the variant's folder. `extra` goes into the
    identity (the fresh seeds' draws differ from seed 0's, so they say so)."""

    def __init__(self, outputs, aspect, n_draws, p_sha, label_sha, img_sha, seed=SEED, **extra):
        self.outputs, self.aspect, self.n_draws, self.seed = outputs, aspect, n_draws, seed
        self.p_sha, self.label_sha, self.img_sha, self.extra = p_sha, label_sha, img_sha, extra

    def _where(self, name):
        path = cache_dir(self.outputs, name, self.seed) / f'{self.aspect}__boot.npz'
        return path, _identity(name, self.aspect, self.p_sha, self.img_sha[name], self.seed,
                               label_sha256=self.label_sha, n_draws=self.n_draws, **self.extra)

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


def read_run(outputs, variant, p_sha, declared_failed=None, seed=SEED):
    """{status, cause, best_epoch, run_sha256, img {aspect: array}} of one variant run (seed 0, or a fresh seed). A run.json with
    status 'failed' has no exports read. A variant named in `declared_failed` (rule §4.3: VG3 still failing after its
    rerun) has its exports read and must fail VG3 on at least one aspect; the problems are kept as `vg3_problem`
    beside the declared cause, and the record is a failed run."""
    declared_failed = declared_failed or {}
    folder = eg_paths.iii_run(outputs, variant, seed)
    run_path = folder / 'run.json'
    assert run_path.exists(), f'{variant}: {run_path} missing (the run has not finished)'
    run = json.loads(run_path.read_text())
    assert run['variant'] == variant and run['seed'] == seed, (variant, run.get('variant'), run.get('seed'))
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


# ------------------------------------------------------------------ fresh seeds (rule §3.3, §4.3), pure parts
FRESH = V.FRESH_SEEDS                                  # (1, 2, 3)
FRESH_TAG = 'fresh-v1: I_b then S_b from one default_rng(0) per aspect (rule §3.3)'


def fresh_draws(n_labelled, n_draws, seeds=FRESH):
    """(idx int32 [n_draws, n_labelled], S int64 [n_draws, len(seeds)]): one `default_rng(0)` per aspect; for b in
    order, I_b = L_a[idx[b]] (`integers(0, n, size=n)`), then S_b = `choice(seeds, size=3, replace=True)`, which both
    arms share. S_b sits between the I_b in the stream, so these I_b are not seed 0's (`rb_ami.draws`)."""
    assert n_labelled > 0 and n_draws > 0
    rng = np.random.default_rng(0)
    idx = np.empty((n_draws, n_labelled), dtype=np.int32)
    chosen = np.empty((n_draws, len(seeds)), dtype=np.int64)
    for b in range(n_draws):
        idx[b] = rng.integers(0, n_labelled, size=n_labelled)
        chosen[b] = rng.choice(list(seeds), size=len(seeds), replace=True)
    return idx, chosen


def paired_gains(boot_v, boot_r, S):
    """float64 [n_draws]: e_b = the mean over S_b of A(v*, I_b, s) - A(R6-5, I_b, s). `boot_*` = {seed: [n_draws]},
    S int [n_draws, 3] (seeds drawn, repeats allowed, the same for both arms)."""
    seeds = sorted(boot_v)
    assert seeds == sorted(boot_r) and set(np.unique(S)) <= set(seeds)
    diff = {s: np.asarray(boot_v[s], np.float64) - np.asarray(boot_r[s], np.float64) for s in seeds}
    out = np.empty(len(S))
    for b in range(len(S)):
        out[b] = np.mean([diff[s][b] for s in S[b]])
    return out


_FRESH_STATE = {}


def _init_fresh(state):
    _FRESH_STATE.update(state)
    threadpool_limits(limits=1)


def _fresh_draw_scores(b):
    idx = _FRESH_STATE['labelled'][_FRESH_STATE['draws'][b]]
    return np.array([rb_ami.score(_FRESH_STATE['labels'], p, idx) for p in _FRESH_STATE['parts']])


def fresh_boot_scores(labels, parts, draws, workers):
    """{name: float64 [n_draws]}: A(arm, I_b) for the given draws (`fresh_draws`' idx), each draw scored per arm alone."""
    labels = np.asarray(labels)
    names = list(parts)
    labelled = np.flatnonzero(labels >= 0)
    state = dict(labels=labels, labelled=labelled, parts=[parts[n] for n in names], draws=draws)
    if workers == 1 or len(draws) < 2:
        _FRESH_STATE.update(state)
        rows = [_fresh_draw_scores(b) for b in range(len(draws))]
    else:
        with multiprocessing.get_context('fork').Pool(workers, initializer=_init_fresh, initargs=(state,)) as pool:
            rows = pool.map(_fresh_draw_scores, range(len(draws)), chunksize=max(1, len(draws) // (workers * 8)))
    scores = np.stack(rows)
    assert scores.shape == (len(draws), len(names)) and np.isfinite(scores).all()
    return {n: scores[:, i] for i, n in enumerate(names)}


def fresh_aspect(labels, parts, arm, ref, seeds, n_draws, workers=8, caches=None):
    """One aspect's fresh-seed result for `arm` (v*) against `ref` (R6-5). `parts` = {name: {seed: int [3, n]}}.
    {A {name: {seed: A(name, L_a, s)}}, Abar {name: mean over seeds}, e = Abar(arm) - Abar(ref), lo, hi: the 95%
    interval of the paired e_b}. `caches` = {seed: object with get(name)/put(name, boot)} for the per-arm scores."""
    labels = np.asarray(labels)
    L = np.flatnonzero(labels >= 0)
    caches = caches or {}
    names = (arm, ref)
    assert set(parts) == set(names) and all(sorted(parts[n]) == sorted(seeds) for n in names)
    A = {n: {s: float(rb_ami.score(labels, parts[n][s], L)) for s in seeds} for n in names}
    Abar = {n: float(np.mean([A[n][s] for s in seeds])) for n in names}
    draws, S = fresh_draws(len(L), n_draws, seeds)
    boot = {n: {} for n in names}
    for s in seeds:
        cache = caches.get(s, _NoCache())
        got = {n: cache.get(n) for n in names}
        missing = {n: parts[n][s] for n in names if got[n] is None}
        if missing:
            for n, b in fresh_boot_scores(labels, missing, draws, workers).items():
                cache.put(n, b)
                got[n] = b
        for n in names:
            assert got[n].shape == (n_draws,) and np.isfinite(got[n]).all()
            boot[n][s] = got[n]
    lo, hi = np.percentile(paired_gains(boot[arm], boot[ref], S), [2.5, 97.5])
    return dict(A=A, Abar=Abar, e=float(Abar[arm] - Abar[ref]), lo=float(lo), hi=float(hi))


def c_outcome(effects):
    """C-HOLDS / C-QUAL / C-FAILS (rule §4.3) from `effects` = {aspect: {e, lo, ...}}, the fresh-seed means.
    C-HOLDS: e_genre >= r_genre with lo_genre > 0, and the guards on the means (e_emotion >= -r_emotion, e_style >=
    -r_style). C-QUAL: e_genre > 0 and both guards, not C-HOLDS. C-FAILS otherwise."""
    guards = effects['emotion']['e'] >= -RANGES['emotion'] and effects['style']['e'] >= -RANGES['style']
    g = effects['genre']
    if guards and g['e'] >= RANGES['genre'] and g['lo'] > 0:
        return 'C-HOLDS'
    return 'C-QUAL' if guards and g['e'] > 0 else 'C-FAILS'


def genre_vs_targets(abar_genre):
    """Ā_genre(v*) against T_g and T_g - r_genre (unrounded)."""
    return dict(A_bar=float(abar_genre), t_g=V.T_G, promising=PROMISING, diff_t_g=float(abar_genre - V.T_G),
                diff_promising=float(abar_genre - PROMISING), reaches_t_g=bool(abar_genre >= V.T_G),
                reaches_promising=bool(abar_genre >= PROMISING))


def reading_both(seed0_outcome, c):
    """The "reading both" sentence (rule §4.3, D34): the seed-0 outcome stands; the C outcome says whether it repeats."""
    assert seed0_outcome in ('III-GO', 'III-PROM', 'III-NONE') and c in ('C-HOLDS', 'C-QUAL', 'C-FAILS', None)
    if c is None:
        return f'Seed-0 outcome {seed0_outcome} stands; no C outcome, so nothing says whether it repeats on fresh seeds.'
    if seed0_outcome == 'III-NONE':
        return (f'Seed-0 outcome III-NONE stands; the winner\'s C outcome is {c}, reported, and makes it neither GO nor '
                'promising. R6-5 stays the recipe.')
    if c == 'C-FAILS':
        return f'Seed-0 outcome {seed0_outcome} stands, and it is not repeated on fresh seeds (C-FAILS).'
    how = 'C-HOLDS' if c == 'C-HOLDS' else 'C-QUAL (qualified: positive gain, guards hold, but short of C-HOLDS)'
    return f'Seed-0 outcome {seed0_outcome} stands, and the fresh seeds repeat it: {how}.'


# ------------------------------------------------------------------ fresh seeds, files
def check_fresh_dirs(outputs, winner):
    """VG4: the fresh seeds' run folders belong to the winner and R6-5 only."""
    root = stage_dir(outputs) / 'train'
    found = sorted({p.parent.name for s in FRESH for p in root.glob(f'*/s{s}')}) if root.exists() else []
    extra = [v for v in found if v not in {winner, BASE}]
    assert not extra, f'VG4: fresh-seed run folders of variants other than the winner {winner} and {BASE}: {extra}'


def read_fresh_run(outputs, variant, seed, p_sha, declared_failed):
    """`read_run` for a fresh seed. A variant in `declared_failed` is declared failed only on the seeds whose exports
    fail VG3 (its other seeds are read as they are)."""
    try:
        return read_run(outputs, variant, p_sha, None, seed)
    except AssertionError as err:
        if variant in declared_failed and str(err).startswith('VG3'):
            return read_run(outputs, variant, p_sha, declared_failed, seed)
        raise


def parity_cause(parity):
    """Why the fresh-seed parity (ticket 11's `fresh_parity.json`) fails: one clause per seed that does not pass."""
    parts = []
    for s in sorted(parity['seeds']):
        rec = parity['seeds'][s]
        if rec['pass']:
            continue
        if rec['run_status'] != 'ok':
            parts.append(f"seed {s}: R6-5 run {rec['run_status']} ({rec['run_cause']})")
        else:
            bad = [a for a, equal in rec['weights_equal'].items() if not equal]
            parts.append(f"seed {s}: R6-5 weights differ from K4's in {', '.join(bad) or 'an unlisted way'}")
    return '; '.join(parts)


def read_parity(outputs):
    path = stage_dir(outputs) / 'fresh_parity.json'
    assert path.exists(), f'{path} missing: ticket 11\'s `parity --seeds 1 2 3` has not run'
    parity = json.loads(path.read_text())
    assert sorted(parity['seeds']) == [str(s) for s in FRESH], sorted(parity['seeds'])
    assert all(isinstance(r['pass'], bool) for r in parity['seeds'].values()) and isinstance(parity['pass'], bool)
    assert parity['pass'] == all(r['pass'] for r in parity['seeds'].values()), 'fresh_parity.json: pass disagrees with its seeds'
    return parity, _file_sha256(path)


def run_fresh(outputs, n_draws, workers, smoke=False, declared_failed=None, ctx=None):
    """C outcome of the winner on fresh seeds 1 to 3 (rule §3.3, §4.3) -> stage_iii/score_fresh.json.

    Needs score_all.json with a winner and the trigger, and fresh_parity.json. A parity failure or a failed fresh run
    writes the record without a C outcome and with its cause."""
    declared_failed = declared_failed or {}
    all_path = stage_dir(outputs) / 'score_all.json'
    assert all_path.exists(), f'{all_path} missing: run `all` before `fresh`'
    sa = json.loads(all_path.read_text())
    assert sa['stage'] == 'all' and sa['n_draws'] == n_draws and sa['smoke'] == bool(smoke), \
        'score_all.json was made with other settings (stage, n_draws, smoke)'
    winner = sa['winner']
    assert winner is not None and sa['fresh_seeds'] is True, 'fresh seeds are not triggered (no winner, or e_genre < r_genre)'
    V.parse(winner)
    assert set(declared_failed) <= {winner, BASE}, f'--failed {sorted(declared_failed)}: only {winner} and {BASE} run on fresh seeds'
    ctx = ctx or load_context()
    p_sha, labels = ctx['p_sha'], ctx['labels']
    assert sa['identity']['p_sha256'] == p_sha
    parity, parity_sha = read_parity(outputs)
    check_fresh_dirs(outputs, winner)
    label_sha = {a: rb_pick.array_sha256(labels[a]) for a in ASPECTS}
    identity = dict(p_sha256=p_sha, label_sha256=label_sha, n_draws=n_draws, winner=winner, seeds=list(FRESH),
                    score_all_sha256=_file_sha256(all_path), fresh_parity_sha256=parity_sha,
                    failed=dict(declared_failed))
    seed0 = sa['outcome']
    base = dict(stage='fresh', **_header(ctx, n_draws, smoke), winner=winner, seeds=list(FRESH), seed0_outcome=seed0,
                A_genre_seed0=sa['A_genre_winner'], parity=dict(parity, sha256=parity_sha))
    path = stage_dir(outputs) / 'score_fresh.json'

    def no_outcome(cause, runs=None):
        record = dict(base, c_outcome=None, no_outcome_cause=cause, effects=None, genre_vs_targets=None, runs=runs,
                      reading=reading_both(seed0, None), identity=identity)
        return _write_or_skip(path, json.loads(json.dumps(record, allow_nan=False)))

    if not parity['pass']:
        return no_outcome(f'fresh-seed parity with K4 fails ({parity_cause(parity)}); the cause is to be traced')
    runs = {n: {s: read_fresh_run(outputs, n, s, p_sha, declared_failed) for s in FRESH} for n in (winner, BASE)}
    summary = {n: {str(s): dict(status=r['status'], cause=r['cause'], best_epoch=r['best_epoch'],
                                run_sha256=r['run_sha256']) for s, r in by.items()} for n, by in runs.items()}
    identity['run_sha256'] = {n: {str(s): r['run_sha256'] for s, r in by.items()} for n, by in runs.items()}
    failed = [f'{n} seed {s}: {r["cause"]}' for n, by in runs.items() for s, r in by.items() if r['status'] == 'failed']
    if failed:
        return no_outcome('a fresh-seed run failed (' + '; '.join(failed) + ')', summary)
    img_sha = {n: {s: {a: rb_pick.array_sha256(runs[n][s]['img'][a]) for a in ASPECTS} for s in FRESH} for n in runs}
    identity['img_sha256'] = {n: {str(s): v for s, v in by.items()} for n, by in img_sha.items()}
    effects = {}
    for a in ASPECTS:
        parts = {n: {s: fixed_partitions(outputs, n, a, runs[n][s]['img'][a], p_sha, s) for s in FRESH} for n in runs}
        caches = {s: DiskBoots(outputs, a, n_draws, p_sha, label_sha[a], {n: img_sha[n][s][a] for n in runs}, seed=s,
                               draws=FRESH_TAG) for s in FRESH}
        effects[a] = fresh_aspect(labels[a], parts, winner, BASE, FRESH, n_draws, workers, caches)
        print(f'{a}: fresh seeds scored', flush=True)
    c = c_outcome(effects)
    abar = effects['genre']['Abar'][winner]
    record = dict(base, c_outcome=c, no_outcome_cause=None, effects=effects, genre_vs_targets=genre_vs_targets(abar),
                  runs=summary, reading=reading_both(seed0, c), identity=identity)
    return _write_or_skip(path, json.loads(json.dumps(record, allow_nan=False)))


# ------------------------------------------------------------------ reported readings (rule §4.3 last line; D30)
PROBE_GENRE = 0.504009                                 # the ceiling probe's ridge AMI for genre on P (probe.json), reported only
N_OLD = 5000


def adaptor_partitions(x):
    """`rb_ami.partitions` for the GA adaptor's output: it predicts standardized targets, so its rows are not unit
    norm. float32, finite; the same k-means (k 20, states 0 to 2, n_init 3, batch 4096)."""
    from sklearn.cluster import MiniBatchKMeans
    from threadpoolctl import threadpool_limits as limits
    x = np.asarray(x)
    assert x.dtype in (np.float16, np.float32) and x.ndim == 2
    x = x.astype(np.float32)
    assert np.isfinite(x).all(), 'nonfinite adaptor rows'
    with limits(limits=rb_ami.THREADS):
        parts = [MiniBatchKMeans(n_clusters=rb_ami.K, random_state=s, n_init=rb_ami.N_INIT,
                                 batch_size=rb_ami.BATCH_SIZE).fit_predict(x) for s in rb_ami.STATES]
    return np.stack(parts).astype(np.int64)


def cached_reading(outputs, variant, aspect, kind, identity, compute):
    """A reading cached beside the variant's partitions (`<aspect>__<kind>.json`, identity checked): Leiden is slow."""
    path = cache_dir(outputs, variant) / f'{aspect}__{kind}.json'
    identity = dict(identity, kind=kind, variant=variant, aspect=aspect, versions=rb_ami.versions())
    if path.exists():
        old = json.loads(path.read_text())
        assert old['identity'] == identity, f'{path}: cached from other inputs; move it aside'
        return old['value']
    value = json.loads(json.dumps(compute(), allow_nan=False))
    _json_dump(path, dict(identity=identity, value=value))
    return value


def old5000_export(outputs, variant, aspect, p_sha):
    """The old-5,000 image export of a seed-0 run (R6-5: 12-24's frozen one, after the trainer record's checks)."""
    import rb_part1_report
    if variant == BASE:
        record = rb_part1_report.record_of(eg_paths.RB_OUT, BASE, aspect)
        return rb_part1_report.load_export(eg_paths.RB_OUT, BASE, aspect, 'old5000', '', record, N_OLD, p_sha)['img']
    with np.load(eg_paths.iii_run(outputs, variant, SEED) / f'{aspect}__old5000.npz', allow_pickle=False) as z:
        assert z.files == ['img'], (variant, aspect, z.files)
        x = z['img']
    assert x.shape == (N_OLD, rb_pick.WIDTH) and x.dtype == np.float16, (variant, aspect, x.shape, x.dtype)
    return x


def base_epochs():
    import rb_part1_report
    out = {}
    for a in ASPECTS:
        r = rb_part1_report.record_of(eg_paths.RB_OUT, BASE, a)
        out[a] = r['best_epoch']
    return out


def report_variant(outputs, variant, entry, ctx, old_labels, imgs, leiden_fn, identity):
    """The readings of one variant: Leiden on P and AMI on the old 5,000 per aspect, and for GA the adaptor's AMI."""
    import rb_part1_report
    p_sha, labels = ctx['p_sha'], ctx['labels']
    entry = dict(entry, leiden={}, old5000={})
    for a in ASPECTS:
        ident = dict(identity, img_sha256=rb_pick.array_sha256(imgs[a]), label_sha256=rb_pick.array_sha256(labels[a]))
        entry['leiden'][a] = cached_reading(outputs, variant, a, 'leiden', ident, lambda: leiden_fn(imgs[a], labels[a]))
        old = old5000_export(outputs, variant, a, p_sha)
        ident = dict(identity, img_sha256=rb_pick.array_sha256(old), label_sha256=rb_pick.array_sha256(old_labels[a]))

        def old_ami():
            try:
                return rb_part1_report.kmeans_ami(old, old_labels[a])
            except AssertionError as err:                    # VG3 covers P only; an old-5,000 problem is reported
                return f'not scored: {err}'
        entry['old5000'][a] = cached_reading(outputs, variant, a, 'old5000', ident, old_ami)
    method = V.parse(variant)[0]
    if method == 'GA' and entry['status'] == 'ok':
        with np.load(eg_paths.iii_run(outputs, variant, SEED) / 'genre__P_adaptor.npz', allow_pickle=False) as z:
            assert z.files == ['adaptor'], z.files
            ad = z['adaptor']
        assert ad.shape == (rb_pick.N_P, rb_pick.WIDTH), ad.shape
        y = labels['genre']
        ident = dict(identity, adaptor_sha256=rb_pick.array_sha256(ad), label_sha256=rb_pick.array_sha256(y))
        value = cached_reading(outputs, variant, 'genre', 'adaptor', ident,
                               lambda: rb_ami.score(y, adaptor_partitions(ad), np.flatnonzero(y >= 0)))
        head = entry['p']['genre']['A']
        entry['adaptor'] = dict(A=value, head_A=head, adaptor_minus_head=float(value - head))
    return entry


def run_report(outputs, smoke=False, ctx=None, old_labels=None, leiden_fn=None):
    """stage_iii/reported.{json,md}: every variant's readings (decide nothing). After score_all.json, and
    score_fresh.json when the fresh seeds were triggered."""
    stage = stage_dir(outputs)
    all_path = stage / 'score_all.json'
    assert all_path.exists(), f'{all_path} missing: the reported readings come after `all`'
    sa = json.loads(all_path.read_text())
    assert sa['stage'] == 'all' and sa['smoke'] == bool(smoke), 'score_all.json is a smoke file (or not): --smoke must match'
    fresh = None
    if sa['fresh_seeds']:
        fresh_path = stage / 'score_fresh.json'
        assert fresh_path.exists(), f'{fresh_path} missing: the fresh seeds were triggered; run `fresh` first'
        fresh = json.loads(fresh_path.read_text())
        assert fresh['stage'] == 'fresh' and fresh['winner'] == sa['winner'] and fresh['smoke'] == bool(smoke)
        assert fresh['identity']['score_all_sha256'] == _file_sha256(all_path), 'score_fresh.json is for another score_all.json'
    ctx = ctx or load_context()
    p_sha, labels = ctx['p_sha'], ctx['labels']
    assert sa['identity']['p_sha256'] == p_sha
    import rb_data as data
    old_labels = old_labels if old_labels is not None else data.old5000_labels()
    if leiden_fn is None:
        import rb_part1_report
        leiden_fn = rb_part1_report.leiden_ami
    frozen = json.loads((eg_paths.RB_PART1 / 'reported.json').read_text())
    for a in ASPECTS:
        assert int((old_labels[a] >= 0).sum()) == frozen['label_counts_old5000'][a], f'{a}: old-5,000 labels differ from 12-24\'s'
    base = base_images(p_sha)
    results = sa['results']
    variants = [BASE] + list(sa['variants'])
    entries = {}
    for v in variants:
        if v == BASE:
            entry = dict(status='ok', cause=None, best_epoch=base_epochs(),
                         p={a: dict(A=sa['base_A'][a], e=0.0, lo=0.0, hi=0.0) for a in ASPECTS})
            imgs = base
        else:
            r = results[v]
            entry = dict(status=r['status'], cause=r['cause'], best_epoch=r['best_epoch'],
                         p=r['effects'] if r['status'] == 'ok' else None)
            if r['status'] == 'failed':
                entries[v] = dict(entry, leiden=None, old5000=None)
                continue
            run = read_run(outputs, v, p_sha)
            assert run['run_sha256'] == sa['identity']['run_sha256'][v], f'{v}: run.json changed since score_all.json'
            imgs = run['img']
            for a in ASPECTS:
                assert rb_pick.array_sha256(imgs[a]) == sa['identity']['img_sha256'][v][a], f'{v}/{a}: export changed since score_all.json'
        entries[v] = report_variant(outputs, v, entry, ctx, old_labels, imgs, leiden_fn, dict(p_sha256=p_sha))
        print(f'{v}: read', flush=True)
    probe = json.loads((eg_paths.RB_PART1 / 'probe.json').read_text())['aspects']['genre']['ami']['P']['ridge']
    assert abs(probe - PROBE_GENRE) < 1e-6, probe
    refs = frozen['references_p']
    assert refs['genre_word']['genre'] == V.T_G
    check = dict(leiden_p={a: abs(entries[BASE]['leiden'][a]['mean'] - frozen['p_leiden'][BASE][a]['mean']) for a in ASPECTS},
                 old5000={a: abs(entries[BASE]['old5000'][a] - frozen['old5000'][BASE][a]) for a in ASPECTS})
    record = dict(stage='reported', smoke=bool(smoke), reference=BASE, seed=SEED, p_sha256=p_sha, variants=variants,
                  winner=sa['winner'], seed0_outcome=sa['outcome'], A_genre_winner=sa['A_genre_winner'],
                  fresh_seeds_triggered=bool(sa['fresh_seeds']), entries=entries,
                  references=dict(t_g=V.T_G, promising=PROMISING, probe_genre_ridge_p=float(probe),
                                  instructed_genre_image_p=dict(refs['genre_word']), default_image_p=dict(refs['default'])),
                  base_vs_12_24=check, fresh=None if fresh is None else {k: fresh[k] for k in (
                      'c_outcome', 'no_outcome_cause', 'effects', 'genre_vs_targets', 'reading', 'runs')},
                  leiden=dict(graph='union 20-NN cosine', resolution=1.0, seeds=[0, 1, 2]),
                  label_counts_p={a: int((labels[a] >= 0).sum()) for a in ASPECTS},
                  label_counts_old5000={a: int((old_labels[a] >= 0).sum()) for a in ASPECTS}, versions=rb_ami.versions(),
                  **code_state(), identity=dict(p_sha256=p_sha, score_all_sha256=_file_sha256(all_path),
                                                score_fresh_sha256=_file_sha256(stage / 'score_fresh.json') if fresh else None))
    record = _write_or_skip(stage / 'reported.json', json.loads(json.dumps(record, allow_nan=False)))
    md_path = stage / 'reported.md'
    text = report_markdown(record) + '\n'
    if md_path.exists():
        assert md_path.read_text() == text, f'{md_path} differs from the record; move it aside'
    else:
        md_path.write_text(text)
    return record


def _f(x, d=4):
    return x if isinstance(x, str) else f'{x:.{d}f}'


def report_markdown(r):
    asp = list(ASPECTS)
    ent = r['entries']
    head = '| variant | ' + ' | '.join(asp) + ' |\n|---|' + '---|' * len(asp)
    lines = ['# Stage iii: reported readings (decide nothing)', '',
             f"P sha256 {r['p_sha256']}{'; SMOKE RUN' if r['smoke'] else ''}. Seed 0 unless stated; reference {r['reference']}. "
             f"Seed-0 outcome: {r['seed0_outcome']}; winner {r['winner']} (A_genre {r['A_genre_winner']}). "
             f"Labelled on P: {r['label_counts_p']}; on the old 5,000: {r['label_counts_old5000']}.", '',
             '## Three aspects on P: A (e against R6-5 [95% interval])', '', head]
    for v in r['variants']:
        e = ent[v]
        if e['status'] != 'ok':
            lines.append(f"| {v} | failed run: {e['cause']} |  |  |")
        elif v == r['reference']:
            lines.append(f'| {v} | ' + ' | '.join(_f(e['p'][a]['A']) for a in asp) + ' |')
        else:
            lines.append(f'| {v} | ' + ' | '.join(
                f"{_f(e['p'][a]['A'])} ({e['p'][a]['e']:+.4f} [{e['p'][a]['lo']:+.4f}, {e['p'][a]['hi']:+.4f}])" for a in asp) + ' |')
    lines += ['', '## The old 5,000: A per aspect (k-means, as part 1)', '', head]
    lines += [f'| {v} | ' + ' | '.join(_f(ent[v]['old5000'][a]) for a in asp) + ' |' for v in r['variants'] if ent[v]['status'] == 'ok']
    lines += ['', '## Leiden AMI on P (as the RB rule\'s part 1: union 20-NN cosine graph, resolution 1, mean of seeds 0 to 2)', '', head]
    lines += [f'| {v} | ' + ' | '.join(_f(ent[v]['leiden'][a]['mean']) for a in asp) + ' |' for v in r['variants'] if ent[v]['status'] == 'ok']
    lines += ['', '## Best epochs', '', head]
    lines += [f'| {v} | ' + ' | '.join(str(ent[v]['best_epoch'][a]) if ent[v]['best_epoch'] else '-' for a in asp) + ' |'
              for v in r['variants']]
    ga = [v for v in r['variants'] if 'adaptor' in ent[v]]
    if ga:
        lines += ['', '## GA: the adaptor\'s own output against the head (genre AMI on P)', '',
                  '| variant | adaptor output | head | adaptor - head |', '|---|---|---|---|']
        lines += [f"| {v} | {_f(ent[v]['adaptor']['A'])} | {_f(ent[v]['adaptor']['head_A'])} | {ent[v]['adaptor']['adaptor_minus_head']:+.4f} |" for v in ga]
    ref = r['references']
    lines += ['', '## References (reported, decide nothing)', '',
              f"T_g (instructed Qwen genre image, genre AMI on P) {ref['t_g']:.7f}; T_g - r_genre {ref['promising']:.7f}; "
              f"ceiling probe (ridge, genre, P) {ref['probe_genre_ridge_p']:.6f}.", '', head,
              '| instructed genre image | ' + ' | '.join(_f(ref['instructed_genre_image_p'][a]) for a in asp) + ' |',
              '| default image | ' + ' | '.join(_f(ref['default_image_p'][a]) for a in asp) + ' |']
    if r['fresh'] is not None:
        f = r['fresh']
        lines += ['', f"## Fresh seeds 1 to 3: {r['winner']} against R6-5 (C outcome {f['c_outcome']})", '']
        if f['c_outcome'] is None:
            lines += [f"No C outcome: {f['no_outcome_cause']}"]
        else:
            lines += ['| aspect | Abar winner | Abar R6-5 | e (3-seed means) | 95% interval |', '|---|---|---|---|---|']
            for a in asp:
                x = f['effects'][a]
                lines.append(f"| {a} | {_f(x['Abar'][r['winner']])} | {_f(x['Abar']['R6-5'])} | {x['e']:+.4f} | [{x['lo']:+.4f}, {x['hi']:+.4f}] |")
            g = f['genre_vs_targets']
            lines += ['', f"Abar_genre {g['A_bar']:.7f}: against T_g {g['diff_t_g']:+.7f} ({'reaches' if g['reaches_t_g'] else 'below'}), "
                          f"against T_g - r_genre {g['diff_promising']:+.7f} ({'reaches' if g['reaches_promising'] else 'below'})."]
        lines += ['', f['reading']]
    lines += ['', f"Base R6-5 here against 12-24's reported.json (max abs difference): {r['base_vs_12_24']}."]
    return '\n'.join(lines)


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
    parser.add_argument('mode', choices=['first', 'all', 'fresh', 'report', 'check'])
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
    if args.mode == 'fresh':
        all_path = stage_dir(args.outputs) / 'score_all.json'
        assert all_path.exists(), f'{all_path} missing: run `all` before `fresh`'
        scored = [json.loads(all_path.read_text())['winner'], BASE]
    assert args.mode != 'report' or not args.failed, '--failed does not apply to `report` (score_all.json holds the failed runs)'
    failed = parse_failed(args.failed, scored if args.mode in ('first', 'all', 'fresh') else None)
    assert not (args.mode == 'check' and failed), '--failed does not apply to `check vg1`'
    if args.mode == 'check':
        rec = run_check_vg1(args.outputs)
        print(f"{stage_dir(args.outputs) / 'vg1_scorer.json'}: VG1 scorer half {'PASS' if rec['pass'] else 'FAIL'}; "
              + ', '.join(f"{k} diff {v['diff']:.2e}" for k, v in rec['items'].items()))
    elif args.mode == 'first':
        rec = run_first(args.outputs, n_draws, args.workers, args.smoke, failed)
        print(f"{stage_dir(args.outputs) / 'score_first.json'}: closest {rec['closest']} "
              f"({'guarded' if rec['closest_guarded'] else 'not guarded'}), extras {' '.join(rec['extras']) or 'none'}")
    elif args.mode == 'fresh':
        rec = run_fresh(args.outputs, n_draws, args.workers, args.smoke, failed)
        print(f"{stage_dir(args.outputs) / 'score_fresh.json'}: {rec['winner']} on seeds 1 to 3: "
              f"{rec['c_outcome'] or 'no C outcome (' + rec['no_outcome_cause'] + ')'}; {rec['reading']}")
    elif args.mode == 'report':
        rec = run_report(args.outputs, args.smoke)
        print(f"{stage_dir(args.outputs) / 'reported.json'}: {len(rec['variants'])} variants, winner {rec['winner']}, "
              f"fresh {'read' if rec['fresh'] else 'not triggered'}")
    else:
        rec = run_all(args.outputs, n_draws, args.workers, args.smoke, failed, args.variants)
        print(f"{stage_dir(args.outputs) / 'score_all.json'}: winner {rec['winner']}, {rec['outcome']}, "
              f"A_genre {rec['A_genre_winner']}, fresh seeds {'yes' if rec['fresh_seeds'] else 'no'}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
