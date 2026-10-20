"""Stage 2a regression check and the stage 2 V1 gate (spec 3.5 row 2a, rule V1).

`python ct2_stage2a.py --device cuda` trains D2, core and paper settings at seed 0 in sample mode into
outputs/stage2/s2a/, compares them with stage 1's seed 0 artifacts in outputs/, and writes stage2a.json:

  regression/d2/<aspect>                       the D2 NPZ hash
  regression/stage1/<settings>/<version>       the five legacy arrays of each stage 1 run
  regression/stage2/<settings>/<cut>/<arm>     every mapper's eval_prob (strict P1 seed 0 is absent in
                                               stage 1 and skipped, as stage 1 did)
  regression/rcol/<version>                    the paper R-COL judgments of seed 0
  replay/d2/<aspect>, replay/percept/...       inference on the 5,000 names equals the saved array bit for bit
  training/code                                every step trained under the code identity checked now

A hash difference passes only with a named cause (--causes JSON) and all three image-side AMIs within 1e-4
(`compare_hashes`); a stage 1 difference checks every dependent mapper. A changed recipe fails outright.
The pass binds the ct2 and ct code, the reference folders, the host, the device and the artifact bytes. Nothing
is written outside outputs/stage2/s2a/ (and its chunked sibling) and the report; no held data is read.

One extra, NON-GATING result is reported under the report's own `chunked` key and never changes PASS or FAIL:
core stage 1 seed 0 with the chunked DEC (dec_chunk_rows=1024), trained under outputs/stage2/s2a_chunked/,
compared with stage 1's arrays (hash match or not; if not, AMIs of q_paint_eval and topic counts).

require_pass(path, device) is the V1 gate every stage 2 step that trains or reads held data calls.
"""
import argparse
import datetime
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

import ct_d2
import ct_score
import ct_stage0
from ct_stage0 import (ARMS, ASPECTS, GRAPHS, VERSIONS, array_sha256, compare_hashes, compare_replay,
                       file_sha256, verify_dependency)

HERE = Path(__file__).resolve().parent
REFERENCE = ct_score.OUTPUTS            # stage 1's outputs in the main checkout
S2A = HERE / 'outputs/stage2/s2a'
REPORT = HERE / 'stage2a.json'
ROWS = ct_d2.ROWS
LEGACY = ('center_norms', 'fit_counts', 'kept', 'topic', 'q_paint_eval')
VERSION_OF = dict(P1='clip_roberta', P2='clip_roberta', P3='qwen_qwen', P4='qwen_qwen')
MAPPERS = (tuple(('core', 'k40', arm) for arm in ARMS) +
           (('paper', 'k40', 'P1'), ('paper', 'k40', 'P4'), ('paper', 'strict', 'P4')))
STAGE1 = tuple((settings, version) for settings in ('core', 'paper') for version in VERSIONS)
TRAINING_STEPS = ('d2', 'core', 'paper')
TRAINING_CODE = 'stage2a_training_code.json'
TRAINING_FOLDERS = dict(d2='d2', core='percept/core', paper='percept/paper')
D2_GRAPH_KEYS = ('graph_identity', 'graph_sha256', 'graph_archives')   # ct2_graphs saves one row per aspect
SHARE_TOLERANCE = 1e-4          # stage 1's topic disagreement tolerance (0.01% of pairs)
SAMPLE_FIT_PAIRS = 67327
DEC_CHUNK_ROWS = 1024


def mapper_key(settings, cut, arm):
    return f'{settings}/{cut}/{arm}'


def required_items():
    return ({'training/code'} |
            {f'regression/d2/{a}' for a in ASPECTS} |
            {f'regression/stage1/{s}/{v}' for s, v in STAGE1} |
            {f'regression/stage2/{mapper_key(*m)}' for m in MAPPERS} |
            {f'regression/rcol/{v}' for v in VERSIONS} |
            {f'replay/d2/{a}' for a in ASPECTS} |
            {f'replay/percept/{mapper_key(*m)}' for m in MAPPERS})


# --- comparisons ------------------------------------------------------------------------------------

def _fingerprint(x):
    return f'{array_sha256(x)}|{x.dtype}|{x.shape}'


def compare_arrays(actual, expected, actual_ami=None, expected_ami=None, cause=''):
    """Exact bytes, dtype and shape of every array, or the named-cause AMI escape of compare_hashes."""
    return compare_hashes({k: _fingerprint(v) for k, v in actual.items()},
                          {k: _fingerprint(v) for k, v in expected.items()}, actual_ami, expected_ami, cause)


def compare_judgments(actual, expected):
    """Collapse judgments of the same (version, seed, cut) runs: count and flag equal, share within 1e-4."""
    def index(runs):
        keys = [(r['version'], r['seed'], r['cut']) for r in runs]
        return dict(zip(keys, runs)) if len(keys) == len(set(keys)) else None
    a, e = index(actual), index(expected)
    passed = bool(e) and a is not None and set(a) == set(e)
    differences = {}
    if passed:
        for key in e:
            same = (a[key]['kept_count'] == e[key]['kept_count'] and a[key]['collapsed'] == e[key]['collapsed']
                    and abs(a[key]['share'] - e[key]['share']) <= SHARE_TOLERANCE)
            if not same:
                differences['/'.join(map(str, key))] = dict(actual={k: a[key][k] for k in ('kept_count', 'share', 'collapsed')},
                                                            expected={k: e[key][k] for k in ('kept_count', 'share', 'collapsed')})
    passed = passed and not differences
    return dict(status='PASS' if passed else 'FAIL', differences=differences, runs=len(expected))


# --- code identity ----------------------------------------------------------------------------------

def ct2_identity():
    return {f'{HERE.name}/{p.name}': file_sha256(p) for p in sorted(HERE.glob('ct2_*.py'))}


def reference_identity():
    """Code stage 0's identity omits but the ct2 data path imports (facts Q7)."""
    root = ct_stage0.HERE.parents[2]
    paths = list((ct_stage0.HERE.parent / '20261018_purified_buddy_pilot').glob('pbp_*.py'))
    paths += [root / 'src' / name for name in ('data/artelingo_splits.py', 'data/splits.py',
                                               'data/wikiart_genre.py', 'utils/feature_manager.py')]
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(paths)}


def code_identity():
    """The ct code identity of stage 0 plus every ct2 module and the extra reference code."""
    return {**ct_stage0.code_identity(), **ct2_identity(), **reference_identity()}


def training_host():
    return ct_stage0.training_host()


# --- where artifacts may be written -----------------------------------------------------------------

def chunked_root(root):
    root = Path(root)
    return root.with_name(root.name + '_chunked')


def check_root(root):
    """outputs/stage2/s2a (or its chunked sibling), or a scratch folder outside every outputs/ tree."""
    root = Path(root).resolve()
    if any(root.is_relative_to(allowed.resolve()) for allowed in (S2A, chunked_root(S2A))):
        return root
    for outputs in (HERE / 'outputs', REFERENCE):
        assert not root.is_relative_to(outputs.resolve()), f'{root}: stage 2a may write only under {S2A}, not outputs/'
    return root


# --- training ---------------------------------------------------------------------------------------

def read_training_code(root):
    path = Path(root) / TRAINING_CODE
    return json.loads(path.read_text()) if path.exists() else {}


def write_training_code(root, step, identity):
    assert step in TRAINING_STEPS
    path = Path(root) / TRAINING_CODE
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(dict(read_training_code(root), **{step: identity}), indent=2, sort_keys=True,
                                  allow_nan=False) + '\n')
    pending.replace(path)


def training_code_item(root):
    """Every step must have started under the code identity being checked now."""
    recorded, current = read_training_code(root), code_identity()
    stale = [step for step in TRAINING_STEPS if recorded.get(step) != current]
    return dict(status='FAIL' if stale else 'PASS', stale_steps=stale)


def prepare_training(root, step):
    """Artifacts of a step recorded under other code are renamed into stale/<stamp>/, never deleted."""
    from zoneinfo import ZoneInfo
    root, current = Path(root), code_identity()
    if read_training_code(root).get(step) == current:
        return []
    folder = root / TRAINING_FOLDERS[step]
    moved = []
    if folder.exists():
        stamp = datetime.datetime.now(ZoneInfo('Europe/Amsterdam')).strftime('%Y%m%d_%H%M%S')
        target = root / 'stale' / stamp / TRAINING_FOLDERS[step]
        assert not target.exists(), f'stale target exists: {target}'
        target.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(target)
        moved.append(target)
    write_training_code(root, step, current)
    return moved


def _no_held_outputs(root):
    assert not any((root / 'held').rglob('*.npz')), 'V4: stage 2a retraining after held output exists'


def train(root=S2A, device='cuda'):
    """Seed 0 in sample mode: D2, then core and paper stage 1 and stage 2. Gating runs use the mode's default DEC."""
    import ct2_d2
    import ct2_percept
    root = check_root(root)
    _no_held_outputs(root)
    prepare_training(root, 'd2')
    ct2_d2.train('sample', [0], device, root)
    for settings, cuts in (('core', ('k40',)), ('paper', ('strict', 'k40'))):
        prepare_training(root, settings)
        for version in VERSIONS:
            ct2_percept.run_stage1('sample', settings, version, 0, root=root, device=device)
        for cut in cuts:
            for arm in (ARMS if settings == 'core' else ('P1', 'P4')):
                ct2_percept.run_stage2('sample', settings, cut, arm, 0, root=root, device=device)


def train_chunked(root, device='cuda'):
    """Non-gating: core stage 1 seed 0 with the chunked DEC, in its own root."""
    import ct2_percept
    root = check_root(root)
    _no_held_outputs(root)
    prepare_training(root, 'core')
    for version in VERSIONS:
        ct2_percept.run_stage1('sample', 'core', version, 0, root=root, device=device, dec_chunk_rows=DEC_CHUNK_ROWS)


# --- seams the check reaches the GPU or the stores through (replaced in tests) ----------------------

@lru_cache(None)
def context():
    """The 5,000 evaluation names (frozen sample, train side) and their development labels."""
    import ct_held
    import ct2_data
    from pbp_graphs import label_vectors
    s = ct2_data.split2('sample')
    names = s['all_painting_ids'][s['subset_pid']].tolist()
    assert len(names) == len(set(names)) == 5000
    labels = ct_held.painting_labels(names)
    old = label_vectors(s, s['subset_pid'])
    assert all(np.array_equal(labels[a], old[a]) for a in ASPECTS), 'development label mismatch'
    return names, labels


def verify_percept(path, root):
    import ct2_percept
    return ct2_percept.verify_output(path, root=root)


def infer_d2(root, aspect, names, device):
    import ct2_d2
    ct2_d2._weights_info('sample', root, aspect, 0)  # mode, names and code checks; ct2_d2.infer refuses sample mode
    return ct_d2.infer(aspect, 0, names, device=device, outdir=root)


def infer_mapper(root, settings, cut, arm, names, device):
    import ct2_percept
    return ct2_percept.infer_stage2('sample', settings, cut, arm, 0, names, root=root, device=device)


def read_rcol_runs(path):
    import ct2_verdict
    return ct2_verdict.read_stage1(path, SAMPLE_FIT_PAIRS)


def score_means(kind, x, labels):
    values = ct_score.score_dev(x, labels, kind=kind)
    return {a: float(np.mean(v)) for a, v in values.items()}


# --- paths ------------------------------------------------------------------------------------------

def d2_path(tree, aspect):
    return Path(tree) / 'd2/train/D2' / f'{aspect}__{GRAPHS[aspect]}__s0.npz'


def stage1_path(tree, settings, version):
    return Path(tree) / 'percept' / settings / 'stage1' / f'{version}__s0.npz'


def stage2_path(tree, settings, cut, arm):
    return Path(tree) / 'percept' / settings / 'stage2' / cut / f'{arm}__s0.npz'


def legacy_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        assert set(LEGACY) <= set(z.files), f'{path}: missing {set(LEGACY) - set(z.files)}'
        arrays = {k: z[k] for k in LEGACY}
    assert all(np.isfinite(x).all() for x in arrays.values())
    return arrays


# --- chunked, non-gating ----------------------------------------------------------------------------

def chunked_item(root, reference, version):
    """Core stage 1 seed 0 with the chunked DEC against stage 1: hash match, or AMIs and topic counts."""
    path, old = stage1_path(root, 'core', version), stage1_path(reference, 'core', version)
    result = dict(gating=False)
    if not path.exists():
        return dict(result, status='NOT RUN')
    try:
        actual, expected = legacy_arrays(path), legacy_arrays(old)
        matched = compare_arrays(actual, expected)['status'] == 'PASS'
        result.update(status='MATCH' if matched else 'DIFFERENT', hash_match=matched,
                      differing_arrays=[k for k in LEGACY if _fingerprint(actual[k]) != _fingerprint(expected[k])])
        if not matched:
            labels = context()[1]
            ami = {name: score_means('percept', arrays['q_paint_eval'], labels)
                   for name, arrays in (('actual', actual), ('expected', expected))}
            result.update(actual_ami=ami['actual'], expected_ami=ami['expected'],
                          ami_differences={a: abs(ami['actual'][a] - ami['expected'][a]) for a in ASPECTS},
                          topics={name: dict(kept=int(arrays['kept'].sum()), used=int(len(np.unique(arrays['topic']))))
                                  for name, arrays in (('actual', actual), ('expected', expected))})
    except Exception as error:  # non-gating: whatever goes wrong is a note, never a missing report
        result.update(status='ERROR', error=f'{type(error).__name__}: {error}')
    return result


# --- the report and the gate ------------------------------------------------------------------------

def write_report(path, items, device, sources, root, chunked=None):
    """Persist every item including failures; never imply a partial pass."""
    from zoneinfo import ZoneInfo
    missing = sorted(required_items() - set(items))
    failed = sorted(k for k, v in items.items() if v['status'] != 'PASS')
    report = dict(schema=1, stage='2a', smoke=False, status='FAIL' if missing or failed else 'PASS',
                  device=str(device), training_host=training_host(), root=str(Path(root).resolve()),
                  items=items, missing=missing, failed=failed, sources=sources, chunked=chunked,
                  code=code_identity(), written=datetime.datetime.now(ZoneInfo('Europe/Amsterdam')).isoformat())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    pending.replace(path)
    return report


def require_pass(path=HERE / 'stage2a.json', device='cuda'):
    """V1: fail closed on a partial, stale, foreign-host, foreign-device or edited-code report."""
    report = json.loads(Path(path).read_text())
    assert report['schema'] == 1 and report['stage'] == '2a' and report['smoke'] is False
    assert report['status'] == 'PASS' and not report['missing'] and not report['failed'], 'Stage 2a did not pass'
    assert set(report['items']) == required_items(), 'incomplete Stage 2a'
    assert all(item['status'] == 'PASS' for item in report['items'].values()), 'failed Stage 2a item'
    assert report['root'] == str(S2A.resolve()), 'Stage 2a root is not outputs/stage2/s2a'
    assert report['training_host'] == training_host(), 'Stage 2a training host differs'
    assert report['device'] == str(device), 'Stage 2a device differs from training device'
    assert report['code'] == code_identity(), 'Stage 2a code changed'
    for source, sha256 in report['sources'].items():
        assert file_sha256(source) == sha256, f'Stage 2a source changed: {source}'
    return report


# --- the check --------------------------------------------------------------------------------------

def check(root=S2A, destination=REPORT, *, device='cuda', causes=None, reference=REFERENCE, only=None,
          chunked=True, chunked_error=None):
    """Compare the sample-size seed 0 artifacts under root with stage 1's under reference.

    causes maps item IDs to explanations established by the controller. only (tests) limits the run to
    item IDs starting with one of the given prefixes; the report then lists the others as missing.
    chunked_error is the message of a failed non-gating chunked training; it is reported and gates nothing.
    All inputs are development data; no held list or held feature is read.
    """
    from threadpoolctl import threadpool_limits
    root, reference = check_root(root), Path(reference).resolve()
    assert not root.is_relative_to(reference), 'stage 2a root inside the stage 1 outputs'
    causes = causes or {}
    items, sources = {}, {}

    def remember(path):
        path = Path(path).resolve()
        sources[str(path)] = file_sha256(path)
        return path

    def read_json(path):
        return json.loads(remember(path).read_text())

    def recipe_diff(config, reference_config, exempt=('device',)):
        return sorted(k for k, v in reference_config.items() if k not in exempt and config.get(k) != v)

    def current_checks(config, reference_config, label, exempt=('device',)):
        assert config.get('dec_chunk_rows') is None, \
            f'{label} trained with chunked DEC (dec_chunk_rows={config["dec_chunk_rows"]})'
        changed = recipe_diff(config, reference_config, exempt)
        assert not changed, f'{label} recipe changed: {changed}'
        assert config['device'] == str(device), f'{label} training device mismatch'
        assert config['mode'] == 'sample', f'{label} mode mismatch'
        assert config['seed'] == 0 and not config['smoke']

    @lru_cache(None)
    def d2(aspect, current):
        tree = root if current else reference
        path = d2_path(tree, aspect)
        meta = read_json(path.with_suffix('.json'))
        assert file_sha256(remember(path)) == meta['output_sha256'], 'D2 output hash mismatch'
        c = meta['config']
        if current:
            current_checks(c, d2(aspect, False)[1]['config'], 'D2', ('device',) + D2_GRAPH_KEYS)
            weight = Path(tree) / 'd2/weights' / f'{aspect}__s0.pt'
            wm = read_json(weight.with_suffix('.json'))
            assert file_sha256(remember(weight)) == wm['weights_sha256']
            assert wm['output_sha256'] == meta['output_sha256'] and wm['config'] == c
        assert (c['aspect'], c['seed'], c['graph_row']) == (aspect, 0, GRAPHS[aspect])
        assert c['row'] == ROWS['D2'] and not c['smoke']
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == {'img', 'cap'}
            arrays = {k: z[k] for k in z.files}
        for x in arrays.values():
            assert x.shape == (5000, 2048) and x.dtype == np.float16 and np.isfinite(x).all()
            assert np.all(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1) < 1e-3)
        return arrays, meta

    @lru_cache(None)
    def percept(settings, cut, name, current):
        """name is a version (stage 1; cut ignored) or an arm (stage 2)."""
        tree = root if current else reference
        stage = 1 if name in VERSIONS else 2
        path = stage1_path(tree, settings, name) if stage == 1 else stage2_path(tree, settings, cut, name)
        meta = read_json(path.with_suffix('.json'))
        remember(path)
        if current:
            old = percept(settings, cut, name, False)[1]
            current_checks(meta['config'], old['config'], f'{settings}/{cut}/{name}')
            assert meta['sample_sha256'] == old['sample_sha256'], 'sample changed'
            verify_percept(path, root)
            remember(path.with_suffix('.pt'))
            if stage == 2:
                source = percept(settings, 'k40', VERSION_OF[name], True)[1]
                verify_dependency(meta, source, file_sha256(stage1_path(root, settings, VERSION_OF[name])))
        else:
            assert meta['config']['seed'] == 0 and not meta['config']['smoke']
        if stage == 1:
            return legacy_arrays(path), meta
        with np.load(path, allow_pickle=False) as z:
            x = z['eval_prob']
        assert x.shape[0] == 5000 and x.dtype == np.float32 and np.isfinite(x).all() and ((x >= 0) & (x <= 1)).all()
        return dict(eval_prob=x), meta

    @lru_cache(None)
    def ami(kind, settings, cut, name, current):
        if kind == 'd2':
            return score_means('k1', d2(name, current)[0]['img'], context()[1])
        return score_means('percept', percept(settings, cut, name, current)[0]['eval_prob'], context()[1])

    def escape(actual, expected, key, dependents):
        """compare_arrays; on a difference the named cause and every dependent's three AMIs decide."""
        result = compare_arrays(actual, expected)
        if result['status'] == 'PASS':
            return result
        fallbacks = {label: compare_arrays(actual, expected, ami(*args, True), ami(*args, False), causes.get(key, ''))
                     for label, args in dependents}
        result.update(status='PASS' if all(v['status'] == 'PASS' for v in fallbacks.values()) else 'FAIL',
                      cause=causes.get(key, ''), image_ami=fallbacks)
        return result

    def regression_d2(aspect):
        return escape(d2(aspect, True)[0], d2(aspect, False)[0], f'regression/d2/{aspect}',
                      [(aspect, ('d2', None, None, aspect))])

    def regression_stage1(settings, version):
        dependents = [(arm if cut == 'k40' else f'{cut}/{arm}', ('percept', s, cut, arm))
                      for s, cut, arm in MAPPERS if s == settings and VERSION_OF[arm] == version]
        return escape(percept(settings, 'k40', version, True)[0], percept(settings, 'k40', version, False)[0],
                      f'regression/stage1/{settings}/{version}', dependents)

    def regression_stage2(settings, cut, arm):
        # The dependency chain (verify_dependency) is checked when the current mapper is loaded.
        return escape(percept(settings, cut, arm, True)[0], percept(settings, cut, arm, False)[0],
                      f'regression/stage2/{mapper_key(settings, cut, arm)}',
                      [(arm, ('percept', settings, cut, arm))])

    def regression_rcol(version):
        recorded = read_json(Path(reference) / 'percept/rcol_stage1.json')['runs']
        expected = [r for r in recorded if r['version'] == version and r['seed'] == 0]
        actual = [r for r in read_rcol_runs(remember(stage1_path(root, 'paper', version))) if r['seed'] == 0]
        return compare_judgments(actual, expected)

    def replay_d2(aspect):
        expected = d2(aspect, True)[0]['img']
        names = context()[0]
        actual = infer_d2(root, aspect, names, str(device))
        assert len(actual) == 5000
        return compare_replay(actual, expected)

    def replay_mapper(settings, cut, arm):
        expected = percept(settings, cut, arm, True)[0]['eval_prob']
        actual = infer_mapper(root, settings, cut, arm, context()[0], str(device))
        assert len(actual) == 5000
        return compare_replay(actual, expected)

    def run_item(key, callback):
        if only is not None and not key.startswith(tuple(only)):
            return
        try:
            items[key] = callback()
        except (AssertionError, OSError, ValueError, KeyError, RuntimeError) as error:
            items[key] = dict(status='FAIL', error=f'{type(error).__name__}: {error}')
        print(f'STAGE2A {key} {items[key]["status"]}', flush=True)

    with threadpool_limits(limits=8):
        run_item('training/code', lambda: training_code_item(root))
        for a in ASPECTS:
            run_item(f'regression/d2/{a}', lambda a=a: regression_d2(a))
        for s, v in STAGE1:
            run_item(f'regression/stage1/{s}/{v}', lambda s=s, v=v: regression_stage1(s, v))
        for m in MAPPERS:
            run_item(f'regression/stage2/{mapper_key(*m)}', lambda m=m: regression_stage2(*m))
        for v in VERSIONS:
            run_item(f'regression/rcol/{v}', lambda v=v: regression_rcol(v))
        for a in ASPECTS:
            run_item(f'replay/d2/{a}', lambda a=a: replay_d2(a))
        for m in MAPPERS:
            run_item(f'replay/percept/{mapper_key(*m)}', lambda m=m: replay_mapper(*m))
        report_chunked = dict(gating=False, skipped=True)
        if chunked:
            report_chunked = dict(gating=False, items={
                v: dict(gating=False, status='ERROR', error=chunked_error) if chunked_error
                else chunked_item(chunked_root(root), reference, v) for v in VERSIONS})
            if chunked_error:
                report_chunked['error'] = chunked_error
    remember(ct_d2.SAMPLE)
    return write_report(destination, items, device, sources, root, report_chunked)


def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # determinism() needs it before CUDA work
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=S2A, help='artifact folder (default outputs/stage2/s2a)')
    p.add_argument('--report', type=Path, default=REPORT)
    p.add_argument('--device', default='cuda')
    p.add_argument('--causes', type=Path, help='JSON mapping item IDs to traced hash-mismatch causes')
    p.add_argument('--check-only', action='store_true', help='compare existing artifacts; train nothing')
    p.add_argument('--no-chunked', action='store_true', help='skip the non-gating chunked-DEC training and item')
    p.add_argument('--require-pass', action='store_true')
    args = p.parse_args(argv)
    if args.require_pass:
        try:
            require_pass(args.report, args.device)
        except (AssertionError, OSError, ValueError, KeyError) as error:
            print(f'STAGE2A required: FAIL {error}', flush=True)
            return 1
        print('STAGE2A required: PASS', flush=True)
        return 0
    chunked_error = None
    root = check_root(args.root)
    if args.report.resolve() == REPORT.resolve():
        assert root == S2A.resolve(), 'the default report needs the default root; give --report for a dry run'
    if not args.check_only:
        train(root, args.device)
        if not args.no_chunked:
            try:
                train_chunked(chunked_root(root), args.device)
            except Exception as error:  # non-gating: the gating check and the report go on
                chunked_error = f'{type(error).__name__}: {error}'
                print(f'STAGE2A chunked training failed (non-gating): {chunked_error}', flush=True)
    causes = json.loads(args.causes.read_text()) if args.causes else {}
    report = check(root, args.report, device=args.device, causes=causes, chunked=not args.no_chunked,
                   chunked_error=chunked_error)
    print(f'STAGE2A {report["status"]} failed={report["failed"]} missing={report["missing"]}', flush=True)
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
