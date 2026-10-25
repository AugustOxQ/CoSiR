"""Stage i students K6, K7, K8, K5 (recipe R6-5) at the sample and full scale, and the student half of X1 and X2.

Training and inference are `rb_student`'s own functions (arm and root given explicitly, so neither the K5 recipe guard
nor the pick file applies); this module adds the CLI, this round's gate for held and ArtELingo-28 inference (rule §4.1,
V1), the eg_ code hashes as provenance, and `regress x1|x2`. Files are 12-24's layout under `--outputs`:
`part2/<scale>/{weights,train,held,a28}/K<k>/...`; the regression writes `part2/regression/x{1,2}_student.json`.

    python eg_student.py train --scale sample --arm K6 --seeds 0 1 2 3 4 --device cuda     # GPU, under the lock
    python eg_student.py infer --scale sample --arm K6                                       # needs the gate
    python eg_student.py regress x1 --device cuda                                            # K2 against 12-24
"""
import argparse
import json
import os
import subprocess
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import eg_paths
import eg_encoders

eg_encoders.extend_rb_maps()
import rb_encoders  # noqa: E402
import rb_paths  # noqa: E402
import rb_score  # noqa: E402
import rb_student  # noqa: E402
import ct_d2  # noqa: E402
import ms_train  # noqa: E402

RECIPE = eg_encoders.RECIPE            # R6-5
ASPECTS = rb_student.ASPECTS
SCALES = rb_student.SCALES
SCALE_OF = {'x1': 'sample', 'x2': 'full'}
STAGE_OF = {'sample': 3, 'full': 4}
REGRESSION_ARM = 'K2'                  # X1 and X2 train K2 (SigLIP 2, R6-5) seed 0
SMOKE_ONLY_ARMS = ('K2', 'K3')         # existing stores: a smoke can run before the new stores are extracted
EG_CODE_FILES = ('eg_student.py', 'eg_encoders.py', 'eg_paths.py')
_REPO = Path(__file__).resolve().parents[3]


def code_commit():
    return subprocess.run(['git', '-C', str(_REPO), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                          text=True).stdout.strip()


_RB_CODE_HASHES = rb_student.code_hashes


def code_hashes():
    """rb_student's writing-code hashes plus this round's (eg_ modules), recorded in every file written here."""
    hashes = _RB_CODE_HASHES()
    hashes.update({name: ms_train.head.sha_file(eg_paths.ROUND / name) for name in EG_CODE_FILES})
    return hashes


# --- gate -------------------------------------------------------------------------------------------------------

def _read_pass(path):
    path = Path(path)
    assert path.is_file(), f'V1: {path} missing'
    record = json.loads(path.read_text())
    assert record.get('pass') is True, f'V1: {path} does not pass'
    return record


def require_gate(outputs, scale, arm, kinds):
    """V1 (rule §4.1): stage 3 (sample) needs X0, V5 for this candidate and X1; stage 4 (full) needs X2.

    V5 is read per kind: held outputs need the candidate's ArtELingo store, ArtELingo-28 outputs its 28 store.
    """
    outputs = Path(outputs)
    reg = outputs / 'part2/regression'
    x = 'x1' if scale == 'sample' else 'x2'
    if scale == 'sample':
        _read_pass(outputs / 'stores/x0.json')
        v5 = json.loads((outputs / 'stores/v5.json').read_text())
        store = eg_encoders.BY_STUDENT[arm].store
        for kind in kinds:
            key = f'{"artelingo_full" if kind == "held" else "artelingo28"}/{store}'
            assert key in v5['stores'], f'V1: V5 has no entry {key}'
            assert v5['stores'][key].get('pass') is True, f'V1: V5 fails for {key}'
    for part in ('student', 'matched', 'scorer'):
        _read_pass(reg / f'{x}_{part}.json')


# --- train and infer --------------------------------------------------------------------------------------------

def _patched():
    stack = ExitStack()
    stack.enter_context(patch.object(rb_student, 'code_hashes', code_hashes))
    return stack


def train(scale, arm, seeds, device, outputs, *, smoke=False, max_epochs=None, root=None):
    """R6-5 student of `arm` for each seed into `root` (default `outputs/part2/<scale>`); refused once held exists."""
    assert arm in eg_encoders.BY_STUDENT and (arm in eg_encoders.CANDIDATES or arm == REGRESSION_ARM
                                              or (smoke and arm in SMOKE_ONLY_ARMS)), arm
    eg_paths.check_imports()
    encoder = eg_encoders.BY_STUDENT[arm].store
    with _patched():
        return rb_student.train(scale, encoder, RECIPE, seeds, device, outputs, smoke=smoke, max_epochs=max_epochs,
                                arm=arm, root=root)


def infer(scale, arm, seeds, kinds, outputs, device=None, *, smoke=False, root=None, gated=True):
    """Held and/or ArtELingo-28 outputs from saved best views, only when the gate holds (not in a smoke)."""
    assert arm in eg_encoders.BY_STUDENT and (arm in eg_encoders.CANDIDATES or not gated
                                              or (smoke and arm in SMOKE_ONLY_ARMS)), arm
    eg_paths.check_imports()
    outputs = Path(outputs)
    if gated and not smoke:
        require_gate(outputs, scale, arm, kinds)
    with _patched(), patch.object(rb_student, 'require_regression', lambda *a, **k: None):
        return rb_student.infer_arm(scale, arm, seeds, kinds, outputs, device, smoke=smoke, root=root)


# --- X1 and X2, student half ------------------------------------------------------------------------------------

def _bits(x):
    return rb_student._bits(x)


def compare_arrays(ours, theirs):
    equal = (ours.shape == theirs.shape and ours.dtype == theirs.dtype
             and bool(np.array_equal(_bits(ours), _bits(theirs))))
    return {'pass': bool(equal), 'rows': int(len(ours)), 'shape': list(ours.shape), 'dtype': str(ours.dtype),
            'ours_sha256': rb_student.sha_array(ours), 'rb_sha256': rb_student.sha_array(theirs)}


def regression_items(ours_root, rb_root, seed=0, arm=REGRESSION_ARM):
    """Our K2 R6-5 files (`ours_root`) against 12-24's (`rb_root`), per aspect: weights (every tensor, key order),
    best epoch and dev loss, held and ArtELingo-28 outputs (bits, shape, dtype)."""
    items = {}
    for aspect in ASPECTS:
        parts = {}
        ours_pt = rb_student.weights_path(ours_root, arm, aspect, seed)
        theirs_pt = rb_student.weights_path(rb_root, arm, aspect, seed)
        a, b = (torch.load(p, map_location='cpu', weights_only=True) for p in (ours_pt, theirs_pt))
        same = list(a) == list(b) and all(a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]) for k in a)
        parts['weights'] = {'pass': bool(same), 'keys': list(a) == list(b), 'tensors': len(a)}
        oi, ti = (json.loads(p.with_suffix('.json').read_text()) for p in (ours_pt, theirs_pt))
        parts['best_epoch'] = {
            'pass': bool(oi['best_epoch'] == ti['best_epoch'] and oi['best_dev_loss'] == ti['best_dev_loss']),
            'ours': [oi['best_epoch'], oi['best_dev_loss']], 'rb': [ti['best_epoch'], ti['best_dev_loss']]}
        for kind in ('held', 'a28'):
            name = f'{aspect}__s{seed}.npz'
            o, oinfo = rb_student.read_outputs(Path(ours_root) / kind / arm / name)
            t, tinfo = rb_student.read_outputs(Path(rb_root) / kind / arm / name)
            assert list(o) == list(t) == ['img']
            parts[kind] = dict(compare_arrays(o['img'], t['img']),
                               list_sha256=oinfo['list_sha256'] == tinfo['list_sha256'])
            parts[kind]['pass'] = parts[kind]['pass'] and parts[kind]['list_sha256']
        items[f'student/{aspect}'] = dict(**{'pass': all(v['pass'] for v in parts.values())}, **parts)
    return items


def regress(which, device, outputs, *, smoke=False, rb_part2=None):
    """Train K2 (SigLIP 2, R6-5) seed 0 at the scale with this code, infer held and ArtELingo-28, compare with 12-24's
    `part2/<scale>/`, and write `part2/regression/<which>_student.json`."""
    assert which in SCALE_OF
    scale = SCALE_OF[which]
    outputs = Path(outputs)
    rb_root = Path(rb_part2 or eg_paths.RB_PART2) / scale
    recorded = json.loads(rb_student.weights_path(rb_root, REGRESSION_ARM, ASPECTS[0], 0)
                          .with_suffix('.json').read_text())['config']['device']
    assert str(torch.device(device)) == recorded, f'12-24 trained on {recorded}; the regression must run on it'
    run = outputs / 'part2/regression' / f'{which}_run'
    train(scale, REGRESSION_ARM, [0], device, outputs, root=run)
    infer(scale, REGRESSION_ARM, [0], ('held', 'a28'), outputs, device, root=run, gated=False)
    items = regression_items(run, rb_root)
    names, sha256 = rb_student.held_names()
    items['student/held_list'] = {'pass': sha256 == eg_paths.HELD_SHA256, 'sha256': sha256}
    path = outputs / 'part2/regression' / f'{which}_student.json'
    record = rb_score.write_regression_items(path, items)
    record.update(code_commit=code_commit(), rb_part2=str(rb_root), scale=scale, arm=REGRESSION_ARM, seed=0)
    rb_score._atomic_json(path, record)
    print(f'{which} student: pass={record["pass"]}; failing: {[k for k, v in record["items"].items() if not v["pass"]]}',
          flush=True)
    return record


# --- command line -----------------------------------------------------------------------------------------------

def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')    # the CT queue's setting, before any CUDA work
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)

    def common(p):
        p.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
        p.add_argument('--smoke', action='store_true', help='outputs under /tmp/claude-0; seed 0; at most 2 epochs')

    arms = list(eg_encoders.CANDIDATES) + list(SMOKE_ONLY_ARMS)
    p = commands.add_parser('train', help='train the R6-5 student of one arm at one scale (K2, K3: --smoke only)')
    p.add_argument('--scale', required=True, choices=SCALES)
    p.add_argument('--arm', required=True, choices=arms)
    p.add_argument('--seeds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    p.add_argument('--device', default='cuda')
    p.add_argument('--max-epochs', type=int)
    common(p)
    p = commands.add_parser('infer', help='held and/or ArtELingo-28 outputs from saved best views (gated)')
    p.add_argument('--scale', required=True, choices=SCALES)
    p.add_argument('--arm', required=True, choices=arms)
    p.add_argument('--kinds', nargs='+', choices=['held', 'a28'], default=['held', 'a28'])
    p.add_argument('--seeds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    p.add_argument('--device', help='default: the device the weights were trained on')
    common(p)
    p = commands.add_parser('regress', help='K2 R6-5 seed 0 against 12-24: x1 (sample) or x2 (full)')
    p.add_argument('which', choices=list(SCALE_OF))
    p.add_argument('--device', default='cuda')
    common(p)
    args = parser.parse_args(argv)
    assert not args.smoke or eg_paths.smoke_ok(args.outputs), '--smoke outputs must be under /tmp/claude-0'
    if args.command == 'regress':
        assert not args.smoke, 'the regression has no smoke'
        record = regress(args.which, args.device, args.outputs)
        raise SystemExit(0 if record['pass'] else 1)
    seeds = [0] if args.smoke else args.seeds
    if args.command == 'train':
        return train(args.scale, args.arm, seeds, args.device, args.outputs, smoke=args.smoke,
                     max_epochs=args.max_epochs)
    return infer(args.scale, args.arm, seeds, args.kinds, args.outputs, args.device, smoke=args.smoke)


if __name__ == '__main__':
    main()
