"""Development-only regression gate; no held paintings are read in Stage 0."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MAIN_TEST = Path('/project/CoSiR/src/test')
MS = MAIN_TEST / '20261019_model_sheet/outputs'
PC = MAIN_TEST / '20261018_percept_core/outputs'
ASPECTS = ('emotion', 'style', 'genre')
VERSIONS = ('clip_roberta', 'qwen_qwen')
ARMS = ('P1', 'P2', 'P3', 'P4')
GRAPHS = dict(emotion='Gres', style='Fres05', genre='R0o')


def compare_hashes(actual, expected, actual_ami=None, expected_ami=None, cause=''):
    """Exact payload hashes, or a named cause and all three AMIs within 1e-4."""
    equal = bool(expected) and actual == expected
    result = dict(status='PASS' if equal else 'FAIL', mode='hash' if equal else 'ami',
                  actual=actual, expected=expected, cause=cause)
    if not equal:
        actual_ami, expected_ami = actual_ami or {}, expected_ami or {}
        complete = set(actual_ami) == set(expected_ami) == set(ASPECTS)
        errors = {a: abs(actual_ami[a] - expected_ami[a]) for a in expected_ami if a in actual_ami}
        passed = complete and bool(cause.strip()) and all(np.isfinite(e) and e <= 1e-4 for e in errors.values())
        result.update(status='PASS' if passed else 'FAIL', actual_ami=actual_ami,
                      expected_ami=expected_ami, absolute_errors=errors)
    return result


def compare_scores(values, expected):
    """Compare score_dev's three-seed means with recorded six-decimal AMIs."""
    actual = {a: float(np.mean(xs)) for a, xs in values.items()}
    passed = set(values) == set(expected) == set(ASPECTS)
    passed = passed and all(len(values[a]) == 3 and np.isfinite(actual[a]) and
                            abs(actual[a] - expected[a]) < 5e-7 for a in ASPECTS)
    return dict(status='PASS' if passed else 'FAIL', actual=actual, expected=expected)


def array_sha256(x):
    return hashlib.sha256(x.tobytes(order='C')).hexdigest()


def compare_replay(actual, expected):
    """Held-path inference must reproduce the saved training array bit for bit."""
    passed = actual.shape == expected.shape and actual.dtype == expected.dtype
    passed = passed and np.isfinite(actual).all() and np.isfinite(expected).all()
    passed = passed and array_sha256(actual) == array_sha256(expected)
    return dict(status='PASS' if passed else 'FAIL', actual=array_sha256(actual),
                expected=array_sha256(expected), rows=len(actual))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def required_items():
    return ({f'regression/d2/{a}' for a in ASPECTS} |
            {f'regression/stage1/{v}' for v in VERSIONS} |
            {f'regression/stage2/{a}' for a in ARMS} |
            {f'{check}/d2/{a}' for check in ('scorer', 'replay') for a in ASPECTS} |
            {f'{check}/percept/{a}' for check in ('scorer', 'replay') for a in ARMS} |
            {'training/code'})


def code_identity():
    """Bind the gate to the rule, adapters, trainers and scoring code in this tree."""
    paths = list(HERE.glob('ct_*.py')) + list(HERE.glob('ct_queue*.sh')) + [HERE / 'DECISION_RULE.md']
    for folder, pattern in (('20261019_model_sheet', 'ms_*.py'),
                            ('20261018_percept_core', 'pc_*.py'),
                            ('20261018_aspect_graph_sheet', 'ags_*.py'),
                            ('20261017_image_transfer_pilot', 'itp_*.py'),
                            ('20261017_aspect_premise_checks', 'apc_*.py'),
                            ('20261017_prompted_aspect_views', 'pav_*.py')):
        paths.extend((HERE.parent / folder).glob(pattern))
    return {str(p.relative_to(HERE.parent)): file_sha256(p) for p in sorted(paths)}


TRAINING_CODE = 'stage0_training_code.json'
TRAINING_STEPS = ('d2', 'core')


def read_training_code(outputs):
    """The code identity each Stage 0 training step last started under, or {} if none."""
    path = Path(outputs) / TRAINING_CODE
    return json.loads(path.read_text()) if path.exists() else {}


def write_training_code(outputs, step, identity):
    """Record the identity of the code that trains `step` (d2 or core) in this outputs tree."""
    assert step in TRAINING_STEPS
    record = dict(read_training_code(outputs), **{step: identity})
    path = Path(outputs) / TRAINING_CODE
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + '\n')
    pending.replace(path)


def training_code_item(outputs):
    """Seed 0 must come from the code the gate checks; a rerun may reuse older artifacts."""
    recorded, current = read_training_code(outputs), code_identity()
    stale = [step for step in TRAINING_STEPS if recorded.get(step) != current]
    return dict(status='FAIL' if stale else 'PASS', stale_steps=stale)


def training_host():
    import socket
    return socket.gethostname()


def write_report(path, items, device, sources):
    """Persist every individual result, including failures; never imply a partial pass."""
    import datetime
    from zoneinfo import ZoneInfo
    missing = sorted(required_items() - set(items))
    failed = sorted(k for k, v in items.items() if v['status'] != 'PASS')
    report = dict(schema=1, stage=0, smoke=False, status='FAIL' if missing or failed else 'PASS',
                  device=str(device), training_host=training_host(), items=items,
                  missing=missing, failed=failed, sources=sources,
                  code=code_identity(), written=datetime.datetime.now(ZoneInfo('Europe/Amsterdam')).isoformat())
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix('.pending.json')
    pending.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    pending.replace(path)
    return report


def require_pass(path=HERE / 'stage0.json', device='cuda'):
    """V1: fail closed on partial/stale reports or another training device."""
    report = json.loads(Path(path).read_text())
    assert report['schema'] == 1 and report['stage'] == 0 and report['smoke'] is False
    assert report['status'] == 'PASS' and not report['missing'] and not report['failed'], 'Stage 0 did not pass'
    assert set(report['items']) == required_items(), 'incomplete Stage 0'
    assert all(item['status'] == 'PASS' for item in report['items'].values()), 'failed Stage 0 item'
    assert report['training_host'] == training_host(), 'Stage 0 training host differs'
    assert report['device'] == str(device), 'Stage 0 device differs from training device'
    assert report['code'] == code_identity(), 'Stage 0 code changed'
    for source, sha256 in report['sources'].items():
        assert file_sha256(source) == sha256, f'Stage 0 source changed: {source}'
    return report


def verify_dependency(mapper, source, source_sha256):
    """A checked mapper must descend from the checked Stage 1 run and weights."""
    assert mapper['stage1_array_sha256s'] == source['array_sha256s'], 'Stage 1 arrays changed'
    assert mapper['stage1_weights_sha256'] == source['weights_sha256'], 'Stage 1 weights changed'
    assert mapper['stage1_sha256'] == source_sha256, 'Stage 1 file changed'


def check(outputs=HERE / 'outputs', destination=HERE / 'stage0.json', *, device='cuda', causes=None,
          ms=MS, pc=PC, only=None):
    """Compare seed 0 with saved runs, replay the scorer and image inference.

    Stage 1 hashes cover every legacy array. Its AMI escape checks both dependent
    Stage 2 image mappers on every aspect, since Stage 1 has no image-only mapper.
    causes maps regression item IDs to explanations established by the controller.
    only (tests) limits the run to item IDs starting with one of the given prefixes; the
    report then lists the others as missing and cannot pass.
    All inputs here are development data; no held list or held feature is read.
    """
    from functools import lru_cache
    from unittest.mock import patch
    from threadpoolctl import threadpool_limits
    import ct_d2
    import ct_percept as pt
    import ct_score as sc
    import ct_held
    from pbp_graphs import label_vectors

    outputs, ms, pc = Path(outputs).resolve(), Path(ms), Path(pc)
    items, sources = {}, {}
    causes = causes or {}

    def remember(path):
        path = Path(path).resolve()
        sources[str(path)] = file_sha256(path)
        return path

    def read_json(path):
        return json.loads(remember(path).read_text())

    @lru_cache(None)
    def context():
        s = pt.load_frozen()
        remember(ct_d2.SAMPLE)
        names = s['all_painting_ids'][s['subset_pid']].tolist()
        assert len(names) == len(set(names)) == 5000
        labels = ct_held.painting_labels(names)
        old_labels = label_vectors(s, s['subset_pid'])
        assert all(np.array_equal(labels[a], old_labels[a]) for a in ASPECTS), 'development label mismatch'
        return s, names, labels

    @lru_cache(None)
    def records():
        return read_json(ms / 'ms_results.json'), read_json(pc / 'pc_results.json')

    @lru_cache(None)
    def d2(aspect, current):
        path = ((outputs / 'd2/train/D2') if current else (ms / 'D2')) / f'{aspect}__{GRAPHS[aspect]}__s0.npz'
        meta = read_json(path.with_suffix('.json'))
        assert file_sha256(remember(path)) == meta['output_sha256'], 'D2 output hash mismatch'
        c = meta['config']
        assert (c['aspect'], c['seed'], c['graph_row']) == (aspect, 0, GRAPHS[aspect])
        assert c['row'] == ct_d2.ROWS['D2'] and not c['smoke']
        assert c['sample_sha256'] == pt.pc_data.fingerprint(context()[0])
        if current:
            assert c['device'] == device, 'D2 training device mismatch'
            weight = outputs / 'd2/weights' / f'{aspect}__s0.pt'
            wm = read_json(weight.with_suffix('.json'))
            assert file_sha256(remember(weight)) == wm['weights_sha256']
            assert wm['output_sha256'] == meta['output_sha256'] and wm['config'] == c
        else:
            assert records()[0]['runs'][f'D2/{aspect}/s0']['identity']['output_sha256'] == meta['output_sha256']
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == {'img', 'cap'}
            arrays = {k: z[k] for k in z.files}
        for x in arrays.values():
            assert x.shape == (5000, 2048) and x.dtype == np.float16 and np.isfinite(x).all()
            assert np.all(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1) < 1e-3)
        return arrays, meta

    @lru_cache(None)
    def percept(stage, name, current):
        path = ((outputs / f'percept/core/stage{stage}' / ('k40' if stage == 2 else ''))
                if current else pc / f'stage{stage}') / f'{name}__s0.npz'
        meta = read_json(path.with_suffix('.json'))
        remember(path)
        assert meta['config']['seed'] == 0 and not meta['config']['smoke']
        if current:
            pt.verify_output(path)
            remember(path.with_suffix('.pt'))
            assert meta['config']['device'] == device, 'PercepT training device mismatch'
            if stage == 2:
                version = pt.ARMS[name][0]
                source = percept(1, version, True)[1]
                verify_dependency(meta, source, file_sha256(pt.stage1_path('core', version, 0)))
            old = percept(stage, name, False)[1]
            assert all(meta['config'][k] == v for k, v in old['config'].items()), 'core recipe changed'
        else:
            arm = name if stage == 2 else ('P1' if name == 'clip_roberta' else 'P3')
            recorded = records()[1]['arms'][arm]['seeds']['0']
            assert file_sha256(path) == recorded['source_sha256'][f'stage{stage}'], 'core reference changed'
        if stage == 1:
            assert meta['sample_sha256'] == pt.pc_data.fingerprint(context()[0])
        with np.load(path, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        expected_keys = {'eval_prob'} if stage == 2 else {'center_norms', 'fit_counts', 'kept', 'topic', 'q_paint_eval'}
        assert expected_keys <= set(arrays)
        assert all(np.isfinite(x).all() for x in arrays.values())
        x = arrays['eval_prob' if stage == 2 else 'q_paint_eval']
        assert x.shape == (5000, 40) and x.dtype == np.float32
        assert ((x >= 0) & (x <= 1)).all()
        return {k: arrays[k] for k in expected_keys}, meta

    @lru_cache(None)
    def scores(kind, name, current):
        x = d2(name, current)[0]['img'] if kind == 'd2' else percept(2, name, current)[0]['eval_prob']
        return sc.score_dev(x, context()[2], kind='k1' if kind == 'd2' else 'percept')

    def recorded(kind, name):
        if kind == 'd2':
            row = records()[0]['runs'][f'D2/{name}/s0']['scores']['img']
            return {a: row[a]['ami']['mean'] for a in ASPECTS}
        row = records()[1]['arms'][name]['seeds']['0']['img']
        return {a: row[a]['mean'] for a in ASPECTS}

    def regression(kind, name):
        key = f'regression/{kind}/{name}'
        if kind == 'd2':
            actual, old = d2(name, True), d2(name, False)
            ah, eh = dict(output_sha256=actual[1]['output_sha256']), dict(output_sha256=old[1]['output_sha256'])
        else:
            stage = int(kind[-1])
            actual, old = percept(stage, name, True), percept(stage, name, False)
            ah = {k: array_sha256(v) for k, v in actual[0].items()}
            eh = {k: array_sha256(v) for k, v in old[0].items()}
        result = compare_hashes(ah, eh)
        if result['status'] == 'PASS':
            return result
        dependents = ([name] if kind != 'stage1' else
                      ['P1', 'P2'] if name == 'clip_roberta' else ['P3', 'P4'])
        metric_kind = 'd2' if kind == 'd2' else 'percept'
        fallbacks = {arm: compare_hashes(ah, eh,
                     {a: float(np.mean(xs)) for a, xs in scores(metric_kind, arm, True).items()},
                     recorded(metric_kind, arm), causes.get(key, '')) for arm in dependents}
        result.update(status='PASS' if all(v['status'] == 'PASS' for v in fallbacks.values()) else 'FAIL',
                      cause=causes.get(key, ''), image_ami=fallbacks)
        return result

    def run_item(key, callback):
        if only is not None and not key.startswith(tuple(only)):
            return
        try:
            items[key] = callback()
        except (AssertionError, OSError, ValueError, KeyError, RuntimeError) as error:
            items[key] = dict(status='FAIL', error=f'{type(error).__name__}: {error}')
        print(f'STAGE0 {key} {items[key]["status"]}', flush=True)

    with patch.object(pt, 'OUTPUTS', outputs), threadpool_limits(limits=8):
        run_item('training/code', lambda: training_code_item(outputs))
        for kind, names in (('d2', ASPECTS), ('stage1', VERSIONS), ('stage2', ARMS)):
            for name in names:
                run_item(f'regression/{kind}/{name}', lambda k=kind, n=name: regression(k, n))
        for kind, names in (('d2', ASPECTS), ('percept', ARMS)):
            for name in names:
                run_item(f'scorer/{kind}/{name}', lambda k=kind, n=name:
                         compare_scores(scores(k, n, False), recorded(k, n)))
                def replay(k=kind, n=name):
                    names = context()[1]
                    if k == 'd2':
                        expected = d2(n, True)[0]['img']
                        actual = ct_d2.infer(n, 0, names, outdir=outputs)
                    else:
                        expected = percept(2, n, True)[0]['eval_prob']
                        actual = pt.infer_stage2('core', 'k40', n, 0, names)
                    assert len(actual) == 5000
                    return compare_replay(actual, expected)
                run_item(f'replay/{kind}/{name}', replay)
    return write_report(destination, items, device, sources)


def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')  # determinism() needs it before CUDA work
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs', type=Path, default=HERE / 'outputs')
    p.add_argument('--report', type=Path, default=HERE / 'stage0.json')
    p.add_argument('--device', default='cuda')
    p.add_argument('--causes', type=Path, help='JSON mapping regression item IDs to traced hash-mismatch causes')
    p.add_argument('--require-pass', action='store_true')
    args = p.parse_args(argv)
    if args.require_pass:
        try:
            require_pass(args.report, args.device)
        except (AssertionError, OSError, ValueError, KeyError) as error:
            print(f'STAGE0 required: FAIL {error}', flush=True)
            return 1
        print('STAGE0 required: PASS', flush=True)
        return 0
    causes = json.loads(args.causes.read_text()) if args.causes else {}
    return int(check(args.outputs, args.report, device=args.device, causes=causes)['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
