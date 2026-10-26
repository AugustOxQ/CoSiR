"""Independent VA2/VB2 re-derivation from RG §3.1, §4.1–2 and EG §3.3.

Scorer records and partition caches are opened only in the final comparison.
Only eg_vg2's independent statistical/data helpers are reused; no scorer imports.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

import rg_paths as paths
import rg_variants as variants
from eg_vg2 import mean_ami, score_effects
import scipy
import sklearn
import threadpoolctl

BASE = 'R6-5'
ASPECTS = paths.ASPECTS
RANGES = variants.RANGES
ROOT = Path(__file__).resolve().parents[3]


class ExportProblem(AssertionError):
    """Saved image rows fail VA3/VB3."""


def array_sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def versions():
    return dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
                threadpoolctl=threadpoolctl.__version__)


def stage_a_record(run, effects):
    good = run['status'] == 'ok'
    return dict(status=run['status'], cause=run.get('cause'), best_epoch=run['best_epoch'],
                effects=effects, A_genre=effects['genre']['A'] if good else None,
                e_genre=effects['genre']['e'] if good else None,
                guarded=good and all(effects[a]['e'] >= -RANGES[a] for a in ('emotion', 'style')))


def climb_decision(records):
    rungs = [v for v in variants.LADDER if v in records]
    assert rungs[:3] == list(variants.LADDER[:3])
    assert rungs == list(variants.LADDER[:len(rungs)]), 'ladder has a hole'
    newest = rungs[-1]
    current = records[newest]
    rivals = {v: records[v]['A_genre'] for v in rungs[:-1] if records[v]['guarded']}
    rising = bool(current['guarded'] and all(current['A_genre'] > a for a in rivals.values()))
    next_arm = variants.LADDER[len(rungs)] if rising and len(rungs) < len(variants.LADDER) else None
    reason = ('failed' if current['status'] == 'failed' else 'unguarded' if not current['guarded'] else
              'cap' if rising and next_arm is None else 'rising' if rising else 'not_strictly_above')
    return dict(rungs_run=rungs, newest=newest, next=next_arm, still_rising=rising,
                reason=reason, rivals=rivals)


def target_level(a):
    return 'target' if a is not None and a >= variants.T_G else 'promising' if (
        a is not None and a >= variants.T_G-RANGES['genre']) else 'neither'


def choose_a(records, eq_pass):
    order = [v for v in variants.A_TIE_ORDER if v in records]
    pool = [v for v in order if records[v]['guarded'] and (v != 'GQ-8' or eq_pass)]
    assert pool, 'GR-8 must provide a guarded winner'
    winner = max(pool, key=lambda v: records[v]['A_genre'])
    kept = [v for v in order if records[v]['A_genre'] is not None and
            records[v]['A_genre'] >= variants.T_G-RANGES['genre']]
    return dict(winner=winner, A_genre_winner=records[winner]['A_genre'],
                outcome='A-SAME' if winner == 'GR-8' else 'A-NEW',
                levels={v: target_level(records[v]['A_genre']) for v in order},
                kept_weights=kept, pending_deletions=[v for v in order if v not in kept],
                stage_b_new=list(variants.B_NEW) + ([] if winner == 'GR-8' else [winner]))


def confirmation(effects):
    guarded = all(effects[a]['e'] >= -RANGES[a] for a in ('emotion', 'style'))
    genre = effects['genre']
    if guarded and genre['e'] >= RANGES['genre'] and genre['lo'] > 0:
        return 'C-HOLDS'
    return 'C-QUAL' if guarded and genre['e'] > 0 else 'C-FAILS'


def choose_b(records, v_star_a):
    order = list(variants.B_TIE_ORDER) + ([] if v_star_a == 'GR-8' else [v_star_a])
    confirmed = [v for v in order if records[v]['confirmed']]
    held = max(confirmed, key=lambda v: records[v]['Abar_genre']) if confirmed else None
    within = [v for v in confirmed if v != held and
              records[held]['Abar_genre'] - records[v]['Abar_genre'] <= RANGES['genre']]
    return dict(confirmed=confirmed, v_held=held, outcome='B-PICK' if held else 'B-NONE',
                within_one_range=within,
                v_held_switches=variants.switches(held) if held else None)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def differences(ours, theirs, path=''):
    """Compare numeric leaves within 1e-6; decisions, types and keys exactly."""
    missing = '<missing>'
    if isinstance(ours, dict) and isinstance(theirs, dict):
        return [d for k in sorted(set(ours) | set(theirs)) for d in
                differences(ours.get(k, missing), theirs.get(k, missing), f'{path}/{k}')]
    if isinstance(ours, list) and isinstance(theirs, list) and len(ours) == len(theirs):
        return [d for i, (a, b) in enumerate(zip(ours, theirs)) for d in differences(a, b, f'{path}/{i}')]
    if isinstance(ours, bool) or isinstance(theirs, bool):
        equal = type(ours) is type(theirs) and ours == theirs
    elif isinstance(ours, (float, int)) and isinstance(theirs, (float, int)):
        equal = np.isfinite(ours) and np.isfinite(theirs) and abs(ours-theirs) <= 1e-6
    else:
        equal = ours == theirs
    return [] if equal else [dict(path=path, ours=ours, theirs=theirs)]


class Inputs:
    """Raw export reader. The scorer's records cannot select any input here."""

    def __init__(self, outputs, rb, frozen_root, n_rows, width, p_sha):
        self.outputs, self.rb = Path(outputs), Path(rb)
        self.frozen_root = None if frozen_root is None else Path(frozen_root)
        self.n_rows, self.width, self.p_sha = n_rows, width, p_sha
        self.hashes, self.img_hashes, self.parts, self.runs = {}, {}, {}, {}
        self.failed = {}

    def bytes(self, path):
        data = Path(path).read_bytes()
        self.hashes[str(path)] = sha(data)
        return data

    def json(self, path, keys=()):
        data = json.loads(self.bytes(path))
        assert isinstance(data, dict) and set(keys) <= data.keys(), f'{path}: missing keys {keys}'
        # Reject non-finite JSON recursively, without depending on scorer code.
        json.dumps(data, allow_nan=False)
        return data

    def folder(self, arm, seed):
        if arm == BASE and seed == 0:
            return self.rb/'train'/BASE
        if arm == 'GR-8' or arm == BASE:
            return self.frozen_root/arm/f's{seed}' if self.frozen_root else paths.frozen_run(arm, seed)
        return paths.run_dir(self.outputs, arm, seed)

    def run_record(self, arm, seed):
        key = arm, seed
        if key not in self.runs:
            path = self.folder(arm, seed)/'run.json'
            record = self.json(path, ('variant', 'seed', 'switches', 'status', 'best_epoch', 'p_sha256'))
            assert record['variant'] == arm and record['seed'] == seed, f'{path}: identity'
            assert record['p_sha256'] == self.p_sha, f'{path}: P hash'
            frozen = arm in (BASE, 'GR-8')
            expected = variants.eg_form(variants.switches(arm)) if frozen else variants.switches(arm)
            assert record['switches'] == expected, f'{path}: switches'
            assert record['status'] in ('ok', 'failed'), f'{path}: status'
            epochs = record['best_epoch']
            assert isinstance(epochs, dict) and set(epochs) <= set(ASPECTS)
            assert all(v is None or type(v) is int and v >= 0 for v in epochs.values())
            if record['status'] == 'failed':
                assert isinstance(record.get('cause'), str) and record['cause'], f'{path}: failure cause'
                assert record.get('failure_kind') in ('nonfinite_loss', 'vg3', 'va3', 'vb3'), f'{path}: failure_kind'
                if record['failure_kind'] != 'nonfinite_loss':
                    assert record.get('rerun_of'), f'{path}: final export failure requires rerun'
            else:
                assert set(epochs) == set(ASPECTS) and record.get('cause') is None
            assert isinstance(record.get('code_commit'), str) and record['code_commit'], f'{path}: commit'
            self.runs[key] = record
        return self.runs[key]

    def scoring_run(self, arm, seed):
        run = self.run_record(arm, seed)
        # A trainer may finish successfully yet fail the export check after its
        # one allowed rerun. Establish that failure from the arrays, not a score.
        if run.get('rerun_of') and (run['status'] == 'ok' or run.get('failure_kind') != 'nonfinite_loss'):
            problems = []
            for aspect in ASPECTS:
                try:
                    self.export_float32(arm, seed, aspect, allow_failed=True)
                except ExportProblem as error:
                    problems.append(str(error))
            if problems:
                run = dict(run, status='failed', cause='VA3/VB3 after rerun: '+'; '.join(problems))
            else:
                assert run['status'] == 'ok', f'{arm}/s{seed}: claimed export failure passes'
        if run['status'] == 'failed':
            self.failed[arm, seed] = run['cause']
        return run

    def export_float32(self, arm, seed, aspect, allow_failed=False):
        key = arm, seed, aspect
        folder = self.folder(arm, seed)
        base0 = arm == BASE and seed == 0
        if not base0:
            assert allow_failed or self.run_record(arm, seed)['status'] == 'ok'
        path = folder/(f'{aspect}__s0__P.npz' if base0 else f'{aspect}__P.npz')
        self.bytes(path)
        with np.load(path, allow_pickle=False) as z:
            assert 'img' in z.files and set(z.files) <= {'img', 'cap'}, f'{path}: keys'
            saved = z['img']
            if saved.dtype != np.float16 or saved.shape != (self.n_rows, self.width):
                raise ExportProblem(f'{path}: dtype/shape')
            if 'cap' in z.files:
                assert z['cap'].shape == saved.shape and np.isfinite(z['cap']).all(), f'{path}: captions'
        self.img_hashes[key] = array_sha(saved)
        x = saved.astype(np.float32)
        if not np.isfinite(x).all():
            raise ExportProblem(f'{path}: nonfinite export')
        if not (np.abs(np.linalg.norm(x, axis=1)-1) < 1e-3).all():
            raise ExportProblem(f'{path}: unit norm')
        side_path = folder/(f'{aspect}__s0.json' if base0 else f'{aspect}.json')
        if base0 or side_path.exists():
            side = self.json(side_path, ('aspect',))
            assert side['aspect'] == aspect, f'{path}: aspect'
            if base0:
                assert (side['recipe'], side['seed'], side['p_sha256']) == (BASE, 0, self.p_sha)
            else:
                assert side['best_epoch'] == self.runs[arm, seed]['best_epoch'][aspect]
                assert side['arrays'][path.name]['img'] == self.img_hashes[key], f'{path}: sidecar image hash'
        return x

    def partitions(self, arm, seed, aspect):
        key = arm, seed, aspect
        if key in self.parts:
            return self.parts[key]
        x = self.export_float32(arm, seed, aspect)
        with threadpool_limits(limits=8):
            parts = np.stack([MiniBatchKMeans(n_clusters=20, random_state=s, n_init=3,
                                             batch_size=4096).fit_predict(x) for s in (0, 1, 2)]).astype(np.int64)
        assert parts.shape == (3, self.n_rows) and ((parts >= 0) & (parts < 20)).all()
        self.parts[key] = parts
        return parts


def load_labels(inputs, annotations, genre_ids):
    p = inputs.json(inputs.rb/'P.json', ('names', 'sha256'))
    names = p['names']
    assert len(names) == inputs.n_rows and names == sorted(set(names)), 'P count/order/unique'
    assert sha('\n'.join(names).encode()) == p['sha256'] == inputs.p_sha, 'P hash'
    if annotations is None:
        from src.data.artelingo import ANNOTATIONS_PATH
        annotations = json.loads(inputs.bytes(ANNOTATIONS_PATH))
    assert annotations and all({'painting', 'emotion', 'art_style'} <= r.keys() for r in annotations)
    if genre_ids is None:
        from src.data.wikiart_genre import GENRE_CSVS, load_wikiart_genre
        for path in GENRE_CSVS:
            inputs.bytes(path)
        genre_ids = load_wikiart_genre(names)
    # The label helper used by eg_vg2; no scorer dependency.
    from rb_vp1 import labels_from_annotations
    labels = labels_from_annotations(names, annotations, genre_ids)
    for y in labels.values():
        assert y.shape == (inputs.n_rows,) and y.dtype == np.int64 and (y >= -1).all() and (y >= 0).any()
    return labels


def seed0_scores(inputs, labels, order, draws, workers, *, intervals=True):
    runs = {v: inputs.scoring_run(v, 0) for v in order}
    effects = {v: {} for v in order if runs[v]['status'] == 'ok'}
    base = {}
    for a in ASPECTS:
        parts = {v: {0: inputs.partitions(v, 0, a)} for v in [BASE]+list(effects)}
        if intervals:
            scores = score_effects(labels[a], parts, draws, workers)
        else:
            idx = np.flatnonzero(labels[a] >= 0)
            values = {v: mean_ami(labels[a], p[0], idx) for v, p in parts.items()}
            scores = {v: dict(A=value, e=value-values[BASE], lo=None, hi=None)
                      for v, value in values.items()}
        base[a] = scores[BASE]['A']
        for v in effects:
            effects[v][a] = scores[v]
    return base, {v: stage_a_record(runs[v], effects.get(v)) for v in order}


def derive_a(inputs, labels, phase, arm, draws, workers):
    if phase == 'rung' and arm is None:
        available = [v for v in variants.LADDER[2:] if (inputs.folder(v, 0)/'run.json').exists()]
        arm = available[-1] if available else 'GR-32'
    order = list(variants.LADDER[:3])
    base, records = seed0_scores(inputs, labels, order, draws, workers, intervals=phase != 'rung')
    decisions = []
    while True:
        decision = climb_decision(records)
        decisions.append(decision)
        if phase == 'rung' and (arm is None or decision['newest'] == arm):
            break
        if decision['next'] is None:
            assert phase != 'rung', f'{arm}: rung would not run under the rule'
            break
        next_arm = decision['next']
        _, scored = seed0_scores(inputs, labels, [next_arm], draws, workers, intervals=phase != 'rung')
        records.update(scored)
        order.append(next_arm)
    if phase == 'rung':
        return dict(decision, rungs={v: {k: records[v][k] for k in
                    ('status', 'cause', 'A_genre', 'e_genre', 'guarded')} |
                    {f'e_{a}': records[v]['effects'][a]['e'] if records[v]['effects'] else None
                     for a in ('emotion', 'style')} for v in order}, decisions=decisions)
    _, others = seed0_scores(inputs, labels, list(variants.A_OTHER), draws, workers)
    records.update(others)
    order.extend(variants.A_OTHER)
    eq = inputs.json(inputs.outputs/'regression/eq.json', ('pass',))
    assert type(eq['pass']) is bool
    return dict(base_A=base, arms=order, climb_rungs=decisions[-1]['rungs_run'],
                still_rising_at_cap=decisions[-1]['still_rising'] and decisions[-1]['newest'] == 'GR-256',
                results=records, eq_pass=eq['pass'], decisions=decisions, **choose_a(records, eq['pass']))


def run(outputs, phase, *, smoke=False, workers=8, arm=None, rb_part1=None,
        frozen_root=None, n_rows=paths.N_P, width=paths.WIDTH, expected_p_sha=paths.P_SHA256,
        annotations=None, genre_ids=None, fresh_reference=None):
    """Recompute from raw inputs first. Explicit input arguments support CPU fixtures."""
    assert phase in ('rung', 'a', 'b') and 1 <= workers <= 8
    assert arm is None or arm in variants.LADDER[2:]
    draws = 20 if smoke else 2000
    inputs = Inputs(outputs, paths.RB_PART1 if rb_part1 is None else rb_part1,
                    frozen_root, n_rows, width, expected_p_sha)
    bad, ours, labels = [], None, {}
    complete = False
    try:
        import eg_vg2
        import rb_vp1
        for source in (paths.RULE, Path(variants.__file__), Path(paths.__file__),
                       Path(eg_vg2.__file__), Path(rb_vp1.__file__),
                       paths.ROUND.parent/'20261025_encoders_genre/DECISION_RULE.md',
                       paths.ROUND.parent/'20261025_recipe_backbones/DECISION_RULE.md'):
            inputs.bytes(source)
        labels = load_labels(inputs, annotations, genre_ids)
        seed0 = derive_a(inputs, labels, 'a' if phase == 'b' else phase, arm, draws, workers)
        ours = derive_b(inputs, labels, seed0, draws, workers) if phase == 'b' else seed0
        # All computations precede the first scorer record read.
        compare_a(inputs, labels, seed0, 'a' if phase == 'b' else phase, draws, bad)
        if phase == 'b':
            compare_b(inputs, labels, ours, seed0, draws, bad,
                      paths.EG_III/'score_fresh.json' if fresh_reference is None else fresh_reference)
        complete = True
    except (AssertionError, OSError, KeyError, ValueError, TypeError) as error:
        bad.append(dict(path=str(getattr(error, 'filename', None) or 'inputs'),
                        ours='valid complete raw inputs and comparison record',
                        theirs=f'{type(error).__name__}: {error}'))
    report = dict(phase=phase, agree=complete and not bad, complete=complete, differences=bad,
                  recomputed=ours, smoke=smoke, n_draws=draws, input_hashes=inputs.hashes,
                  intervals_recomputed=phase != 'rung', versions=versions(),
                  label_counts={a: int((y >= 0).sum()) for a, y in labels.items()},
                  label_sha256={a: array_sha(y) for a, y in labels.items()},
                  img_sha256={f'{v}/s{s}/{a}': h for (v, s, a), h in inputs.img_hashes.items()},
                  code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  code_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)),
                  code_sha256=sha(Path(__file__).read_bytes()))
    dest = Path(outputs)/('stage_b/rederive_vb2.json' if phase == 'b' else f'stage_a/rederive_va2_{phase}.json')
    write_json(dest, report)
    return report


def check_fields(ours, theirs, bad, path):
    assert isinstance(theirs, dict), f'{path}: expected object'
    bad.extend(differences(ours, {k: theirs.get(k, '<missing>') for k in ours}, path))


def identity(inputs, labels, order, draws, fresh=False):
    seeds = variants.FRESH_SEEDS if fresh else (0,)
    images, runs, failed = {}, {}, {}
    for v in [BASE]+list(order):
        by_seed = {str(s): {a: inputs.img_hashes[v, s, a] for a in ASPECTS
                          if (v, s, a) in inputs.parts} for s in seeds}
        by_seed = {s: a for s, a in by_seed.items() if a}
        if by_seed:
            images[v] = by_seed if fresh else by_seed['0']
        if v == BASE and not fresh:
            continue
        digests = {}
        for s in seeds:
            if (v, s) not in inputs.runs:
                continue
            digests[str(s)] = inputs.hashes[str(inputs.folder(v, s)/'run.json')]
            if (v, s) in inputs.failed:
                failed[f'{v}/s{s}' if fresh else v] = inputs.failed[v, s]
        if digests:
            runs[v] = digests if fresh else digests['0']
    return dict(p_sha256=inputs.p_sha, label_sha256={a: array_sha(y) for a, y in labels.items()},
                n_draws=draws, arms=order, img_sha256=images, run_sha256=runs, failed=failed)


def compare_parts(inputs, stage, bad):
    for (v, s, a), ours in inputs.parts.items():
        if (stage == 'b') == (s == 0):
            continue
        path = inputs.outputs/f'stage_{stage}/partitions'/v/f's{s}'/f'{a}__partitions.npz'
        prefix = f'partitions/{stage}/{v}/s{s}/{a}'
        if not path.exists():
            bad.append(dict(path=prefix, ours='independent partition cache', theirs='<missing>'))
            continue
        inputs.bytes(path)
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == {'labels'}, f'{path}: keys'
            actual = z['labels']
        if actual.dtype != np.int64 or actual.shape != ours.shape or not np.array_equal(actual, ours):
            bad.append(dict(path=prefix, ours=dict(sha256=array_sha(ours), shape=list(ours.shape)),
                            theirs=dict(sha256=array_sha(actual), shape=list(actual.shape), dtype=str(actual.dtype))))


def compare_decision(expected, actual, records, bad, prefix):
    check_fields({k: expected[k] for k in ('rungs_run', 'newest', 'next', 'still_rising', 'rivals')},
                 actual, bad, prefix)
    # Reasons are descriptive prose; the guard/newest/rivals/next encode their rule meaning.
    if not isinstance(actual.get('reason'), str) or not actual['reason']:
        bad.append(dict(path=prefix+'/reason', ours='nonempty explanation', theirs=actual.get('reason')))
    rungs = {}
    for v in expected['rungs_run']:
        r = records[v]
        rungs[v] = {k: r[k] for k in ('status', 'cause', 'A_genre', 'e_genre', 'guarded')}
        rungs[v].update({f'e_{a}': r['effects'][a]['e'] if r['effects'] else None
                         for a in ('emotion', 'style')})
    bad.extend(differences(rungs, actual.get('rungs', '<missing>'), prefix+'/rungs'))


def compare_a(inputs, labels, ours, phase, draws, bad):
    pick_path = inputs.rb/'pick.json'
    if pick_path.exists():
        pick = inputs.json(pick_path, ('p_sha256', 'n_rows', 'label_sha256'))
        check_fields(dict(p_sha256=inputs.p_sha, n_rows=inputs.n_rows,
                          label_sha256={a: array_sha(y) for a, y in labels.items()}),
                     pick, bad, 'frozen_labels')
    climb = inputs.json(inputs.outputs/'stage_a/climb.json', ('decisions',))
    actual_decisions = climb['decisions']
    assert isinstance(actual_decisions, list) and actual_decisions, 'climb: nonempty decisions'
    check_fields(dict(smoke=draws == 20, t_g=variants.T_G, promising=variants.T_G-RANGES['genre'],
                      ranges=RANGES, p_sha256=inputs.p_sha), climb, bad, 'climb')
    if phase == 'rung':
        # Rung records carry the same rule quantities but no full effects.
        records = {v: dict(r, effects={a: dict(e=r[f'e_{a}']) for a in ('emotion', 'style')}
                          if r['status'] == 'ok' else None) for v, r in ours['rungs'].items()}
        compare_decision(ours, actual_decisions[-1], records, bad, 'climb/last')
        check_fields(identity(inputs, labels, ours['rungs_run'], draws),
                     actual_decisions[-1].get('identity', {}), bad, 'climb/last/identity')
    else:
        if len(ours['decisions']) != len(actual_decisions):
            bad.extend(differences(len(ours['decisions']), len(actual_decisions), 'climb/decision_count'))
        for i, (d, saved) in enumerate(zip(ours['decisions'], actual_decisions)):
            compare_decision(d, saved, ours['results'], bad, f'climb/decisions/{i}')
            check_fields(identity(inputs, labels, d['rungs_run'], draws),
                         saved.get('identity', {}), bad, f'climb/decisions/{i}/identity')
        path = inputs.outputs/'stage_a/verdict.json'
        theirs = inputs.json(path, ('results', 'identity'))
        expected = {k: v for k, v in ours.items() if k != 'decisions'}
        expected.update(smoke=draws == 20, n_draws=draws, reference=BASE, seed=0,
                        t_g=variants.T_G, promising=variants.T_G-RANGES['genre'], ranges=RANGES,
                        label_counts={a: int((y >= 0).sum()) for a, y in labels.items()})
        expected['climb_reason'] = actual_decisions[-1].get('reason')
        check_fields(expected, theirs, bad, 'stage_a/verdict')
        check_fields(identity(inputs, labels, ours['arms'], draws), theirs['identity'], bad, 'stage_a/identity')
    compare_parts(inputs, 'a', bad)


def derive_b(inputs, labels, seed0, draws, workers):
    winner = seed0['winner']
    order = list(variants.B_NEW)+['GR-8']+([] if winner == 'GR-8' else [winner])
    runs = {v: {str(s): inputs.scoring_run(v, s) for s in variants.FRESH_SEEDS} for v in [BASE]+order}
    usable = [v for v in order if all(r['status'] == 'ok' for r in runs[v].values()) and
              all(r['status'] == 'ok' for r in runs[BASE].values())]
    effects = {v: {} for v in usable}
    for a in ASPECTS:
        parts = {v: {s: inputs.partitions(v, s, a) for s in variants.FRESH_SEEDS} for v in [BASE]+usable}
        scored = score_effects(labels[a], parts, draws, workers, fresh=True)
        for v in usable:
            effects[v][a] = scored[v]
    records = {}
    for v in order:
        eff = effects.get(v)
        failures = [r['cause'] for r in list(runs[v].values())+list(runs[BASE].values()) if r['status'] == 'failed']
        c = confirmation(eff) if eff else None
        records[v] = dict(status='ok' if eff else 'failed', cause='; '.join(failures) if failures else None,
                          effects=eff, Abar_genre=eff['genre']['Abar'][v] if eff else None,
                          e_genre=eff['genre']['e'] if eff else None, c_outcome=c, confirmed=c == 'C-HOLDS')
    return dict(a_outcome=seed0['outcome'], v_star_a=winner, stage_b_arms=order, results=records,
                levels={v: records[v]['c_outcome'] for v in order}, runs=runs,
                **choose_b(records, winner))


def compare_effects(ours, actual, bad, prefix, fresh=False):
    if ours is None:
        bad.extend(differences(None, actual, prefix))
        return
    assert isinstance(actual, dict), f'{prefix}: effects object'
    bad.extend(differences(sorted(ours), sorted(actual), prefix+'/aspects'))
    for a, effect in ours.items():
        # Abar/e/bounds are mandatory. Some writers also retain each seed's A.
        expected = {k: v for k, v in effect.items() if not fresh or k != 'A' or k in actual.get(a, {})}
        check_fields(expected, actual.get(a, {}), bad, f'{prefix}/{a}')


def compare_b(inputs, labels, ours, seed0, draws, bad, fresh_reference):
    ref = inputs.json(fresh_reference, ('effects', 'identity'))
    actual_hashes = {v: {str(s): {a: inputs.img_hashes[v, s, a] for a in ASPECTS}
                        for s in variants.FRESH_SEEDS} for v in ('GR-8', BASE)}
    hash_diffs = differences(actual_hashes, ref['identity'].get('img_sha256'), 'VB1/frozen_hashes')
    bad.extend(hash_diffs)
    before = len(bad)
    compare_effects(ours['results']['GR-8']['effects'], ref['effects'], bad, 'VB1/frozen_effects', fresh=True)
    # A smoke checks all point estimates and hashes; its 20-draw intervals cannot
    # reproduce the frozen 2000-draw bounds. Full VB2 checks all bounds.
    bounds_bad = bad[before:]
    if draws == 20:
        retained = [d for d in bounds_bad if not d['path'].endswith(('/lo', '/hi'))]
        bad[before:] = retained
    vb1_ok = not hash_diffs and len(bad) == before
    vb1 = inputs.json(inputs.outputs/'stage_b/vb1.json', ('pass', 'items', 'hashes_checked', 'bounds_checked'))
    check_fields({'pass': vb1_ok, 'hashes_checked': not hash_diffs, 'bounds_checked': draws != 20,
                  'reference_sha256': inputs.hashes[str(fresh_reference)], 'p_sha256': inputs.p_sha},
                 vb1, bad, 'VB1')
    check_fields({'img_sha256': actual_hashes}, vb1.get('identity', {}), bad, 'VB1/identity')
    assert isinstance(vb1['items'], dict) and vb1['items'], 'VB1: nonempty comparison items'
    for name, item in vb1['items'].items():
        assert isinstance(item, dict) and {'value', 'expected', 'diff', 'pass'} <= item.keys(), f'VB1/{name}: keys'
        diff = abs(item['value']-item['expected'])
        check_fields({'diff': diff, 'pass': diff <= variants.TOL},
                     dict(item, diff=abs(item['diff'])), bad, f'VB1/items/{name}')
    ours['vb1_pass'] = vb1_ok
    theirs = inputs.json(inputs.outputs/'stage_b/verdict.json', ('results', 'identity'))
    keys = ('a_outcome', 'v_star_a', 'stage_b_arms', 'levels', 'v_held', 'outcome', 'v_held_switches')
    header = dict(smoke=draws == 20, n_draws=draws, reference=BASE, seeds=list(variants.FRESH_SEEDS),
                  t_g=variants.T_G, promising=variants.T_G-RANGES['genre'], ranges=RANGES,
                  label_counts={a: int((y >= 0).sum()) for a, y in labels.items()}, vb1_pass=vb1_ok)
    check_fields(dict(header, **{k: ours[k] for k in keys}), theirs, bad, 'stage_b/verdict')
    for k in ('confirmed', 'within_one_range'):
        bad.extend(differences(sorted(ours[k]), sorted(theirs.get(k, [])), 'stage_b/'+k))
    bad.extend(differences(sorted(ours['results']), sorted(theirs['results']), 'stage_b/results/arms'))
    for v, r in ours['results'].items():
        saved = theirs['results'].get(v, {})
        check_fields({k: r[k] for k in ('status', 'Abar_genre', 'e_genre', 'c_outcome', 'confirmed')},
                     saved, bad, f'stage_b/results/{v}')
        # Failure prose is not prescribed, but a failed arm must explain its cause.
        if (r['cause'] is None and saved.get('cause') is not None) or (
                r['cause'] is not None and not (isinstance(saved.get('cause'), str) and saved['cause'])):
            bad.extend(differences(r['cause'], saved.get('cause'), f'stage_b/results/{v}/cause'))
        compare_effects(r['effects'], saved.get('effects'), bad, f'stage_b/results/{v}/effects', fresh=True)
    check_fields(identity(inputs, labels, ours['stage_b_arms'], draws, fresh=True),
                 theirs['identity'], bad, 'stage_b/identity')
    compare_parts(inputs, 'b', bad)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('rung', 'a', 'b'), required=True)
    parser.add_argument('--outputs', type=Path, default=paths.OUTPUTS)
    parser.add_argument('--arm', choices=variants.LADDER[2:], help='newest rung to check; default: newest raw run')
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--smoke', action='store_true', help='20 draws; writes only under /tmp/claude-0/')
    args = parser.parse_args(argv)
    if args.arm and args.phase != 'rung':
        parser.error('--arm applies only to --phase rung')
    if args.smoke and not paths.smoke_ok(args.outputs):
        parser.error('--smoke refuses writes outside /tmp/claude-0/')
    paths.check_imports()
    report = run(args.outputs, args.phase, arm=args.arm, smoke=args.smoke, workers=args.workers)
    print(f'VA2/VB2 phase={args.phase} agree={report["agree"]}; '
          f'{len(report["differences"])} differences; {report["n_draws"]} draws; complete={report["complete"]}')
    return 0 if report['agree'] else 1


if __name__ == '__main__':
    sys.exit(main())
