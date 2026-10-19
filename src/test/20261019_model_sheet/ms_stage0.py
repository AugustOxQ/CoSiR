"""CPU separate-head regeneration and the three stage 0 regression gates."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
import ms_score as score
import ms_train

HERE = Path(__file__).resolve().parent
REFERENCE = score.MAIN_TEST / '20261018_aspect_graph_sheet/ags_head_results.json'


def check(outputs):
    base = outputs / 'stage0'
    required = [(a, seed, base / f'{a}__{score.STAGE1[a]}__s{seed}.npz')
                for a in score.ASPECTS for seed in (range(1) if a == 'genre' else range(3))]
    missing = [str(p) for _, _, p in required if not p.exists()]
    for path in missing:
        print(f'MISSING {path}', flush=True)
    s = score.load_frozen()
    labels = score.label_vectors(s, s['subset_pid'])
    reference = json.loads(REFERENCE.read_text())
    assert reference['identity']['sample_sha256'] == score.fingerprint(s)
    result = dict(missing=missing, stale=[], a={}, b={}, best_epochs={}, sample_sha256=score.fingerprint(s),
                  code_version=ms_train.code_version(), scorer_version=score.SCORER_VERSION)
    for aspect, seed, path in required:
        if not path.exists():
            continue
        meta = json.loads(path.with_suffix('.json').read_text())
        c = meta['config']
        if c['code_version'] != result['code_version']:
            result['stale'].append(str(path))
            print(f'STALE {path}', flush=True)
        assert c['sample_sha256'] == score.fingerprint(s)
        assert (c['aspect'], c['graph_row'], c['seed']) == (aspect, score.STAGE1[aspect], seed)
        assert c['device'] == 'cpu' and c['threads'] == 8 and c['row']['share'] == 'separate'
        assert c['row']['inputs'] == 'default' and not c['smoke']
        assert score.sha_file(path) == meta['output_sha256']
        name = path.stem
        refmeta = (json.loads((score.GENRE / path.with_suffix('.json').name).read_text())
                   if aspect == 'genre' else reference['outputs'][name])
        result['best_epochs'][name] = dict(actual=meta['best_epoch'], expected=refmeta['best_epoch'])
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == {'img', 'cap'}
            for side in ('img', 'cap'):
                x = z[side]
                assert x.dtype == np.float16 and x.shape == (5000, 2048)
                score.check_view(x)
                if aspect == 'genre':
                    refpath = score.GENRE / path.name
                    assert score.sha_file(refpath) == refmeta['output_sha256']
                    assert refmeta['config']['sample_sha256'] == score.fingerprint(s)
                    with np.load(refpath, allow_pickle=False) as old:
                        assert old[side].dtype == np.float16 and old[side].shape == x.shape
                        result['a'][side] = dict(actual=score.array_sha(x.view(np.uint16)),
                            expected=score.array_sha(old[side].view(np.uint16)),
                            equal=bool(np.array_equal(x.view(np.uint16), old[side].view(np.uint16))),
                            reference=str(refpath))
                else:
                    got = score.measures(x, labels, outputs / 'cache')
                    for label in labels:
                        expected = reference['outputs'][name]['ami'][side][label]['mean']
                        actual = got[label]['ami']['mean']
                        result['b'][f'{name}/{side}/{label}'] = dict(actual=actual, expected=expected,
                            equal=round(actual, 6) == round(expected, 6))
    try:
        result['c'] = score.check_percept(s, labels, outputs)
    except AssertionError as error:
        if str(error) != 'PercepT scorer regression; see stage0_scorer.json':
            raise
        result['c'] = json.loads((outputs / 'stage0_scorer.json').read_text())
        if result['c']['status'] == 'PASS':
            raise
    failed = []
    if missing:
        failed.append('missing')
    if result['stale']:
        failed.append('stale')
    if len(result['a']) != 2 or not all(v['equal'] for v in result['a'].values()):
        failed.append('a')
    if len(result['b']) != 36 or not all(v['equal'] for v in result['b'].values()):
        failed.append('b')
    if result['c']['status'] != 'PASS':
        failed.append('c')
    result.update(status='FAIL' if failed else 'PASS', failed=failed)
    score.atomic_json(outputs / 'stage0.json', result)
    print('STAGE0 ' + result['status'] + (' ' + ','.join(failed) if failed else ''))
    return bool(failed)


def require_pass(outputs):
    path = outputs / 'stage0.json'
    if not path.exists():
        print(f'STAGE0 required: missing {path}')
        return 1
    result = json.loads(path.read_text())
    reasons = []
    if result.get('status') != 'PASS':
        reasons.append(f"status={result.get('status')}; failed={result.get('failed')}")
    if result.get('code_version') != ms_train.code_version():
        reasons.append('trainer code_version is stale or missing')
    if result.get('scorer_version') != score.SCORER_VERSION:
        reasons.append('SCORER_VERSION is stale or missing')
    print('STAGE0 required: ' + ('; '.join(reasons) if reasons else 'PASS'))
    return int(bool(reasons))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='check existing arrays without training')
    mode.add_argument('--require-pass', action='store_true', help='require PASS from the current trainer and scorer; no training/scoring')
    p.add_argument('--aspects', nargs='+', choices=score.ASPECTS)
    p.add_argument('--outputs', type=Path, default=HERE / 'outputs')
    args = p.parse_args()
    if args.require_pass:
        return require_pass(args.outputs)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', CUDA_VISIBLE_DEVICES='')
    if args.check:
        return check(args.outputs)
    for a in args.aspects or score.ASPECTS:
        subprocess.run([sys.executable, str(HERE / 'ms_train.py'), '--row', 'A1', '--set',
                        f'graphs.{a}={score.STAGE1[a]}', '--name', 'stage0', '--aspects', a,
                        '--seeds', *(['0'] if a == 'genre' else ['0', '1', '2']), '--device', 'cpu',
                        '--outdir', str(args.outputs)], env=env, check=True)
    # A per-aspect worker only regenerates; the controller checks after all workers finish.
    return 0 if args.aspects else check(args.outputs)


if __name__ == '__main__':
    with threadpool_limits(8):
        sys.exit(main())
