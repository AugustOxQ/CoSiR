"""Matched arms M5 to M8 (PercepT core P3 on a candidate's image, the CT's Qwen Stage 1 topics), B1 and B6 on
ArtELingo-28 at the sample scale, and the matched half of X1 and X2 (rule §2.1 stage 5, §4.1).

Training, inference and the writers are `rb_matched`'s own (World, train_stage2, infer with its second normalization of
the inputs, write_output, run_matched, run_ct_a28); this module adds the CLI, this round's gate (V1), the eg_ code
hashes as provenance, and `regress x1|x2`. Files are 12-24's layout under `--outputs`:
  part2/<scale>/matched/M<k>/P3__s<seed>.{npz,json,pt}   part2/<scale>/held/M<k>/P3__s<seed>.npz
  part2/<scale>/a28/M<k>/P3__s<seed>.npz                 part2/sample/a28/{B1/P3,B6/P2}__s<seed>.npz
The regression writes `part2/regression/x{1,2}_matched.json`.

    python eg_matched.py --scale sample --arm M6 --device cuda      # GPU, under the lock; needs the gate
    python eg_matched.py --scale sample --arm B1                    # B1 (CT P3) and B6 (CT P2) on ArtELingo-28
    python eg_matched.py regress x1 --device cuda                   # M2 seed 0 against 12-24
"""
import argparse
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import eg_paths
import eg_encoders

eg_encoders.extend_rb_maps()
import rb_matched  # noqa: E402
import rb_score  # noqa: E402
import rb_student  # noqa: E402
import ct_percept as base  # noqa: E402
import ms_train  # noqa: E402

SCALE_OF = {'x1': 'sample', 'x2': 'full'}
REGRESSION_ARM = 'M2'                  # X1 and X2 train M2 (SigLIP 2 matched arm) seed 0
SMOKE_ONLY_ARMS = ('M2', 'M3')         # existing stores: a smoke can run before the new stores are extracted
NEW_ARMS = tuple(eg_encoders.BY_STUDENT[k].own for k in eg_encoders.CANDIDATES)   # M6, M7, M8, M5
CT_ARMS = tuple(rb_matched.CT_ARMS)    # B1, B6 (sample scale only; 12-24 wrote the full scale)
EG_CODE_FILES = ('eg_matched.py', 'eg_encoders.py', 'eg_paths.py')
# 'source_commit' is the git HEAD recorded in a mapper config (provenance, differs between commits by design).
CONFIG_IGNORED = ('source_commit',)
_REPO = Path(__file__).resolve().parents[3]
_RB_CODE = rb_matched._code
_RB_OUTPUT_PATH = rb_score.output_path


def code_commit():
    return subprocess.run(['git', '-C', str(_REPO), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                          text=True).stdout.strip()


def _code():
    """rb_matched's writing-code hashes plus this round's (eg_ modules) and the git commit, in every sidecar."""
    hashes = _RB_CODE()
    hashes.update({name: ms_train.head.sha_file(eg_paths.ROUND / name) for name in EG_CODE_FILES})
    hashes['git_commit'] = code_commit()
    return hashes


# --- gate -------------------------------------------------------------------------------------------------------

def _read_pass(path):
    path = Path(path)
    assert path.is_file(), f'V1: {path} missing'
    record = json.loads(path.read_text())
    assert record.get('pass') is True, f'V1: {path} does not pass'
    return record


def require_gate(outputs, scale, arm):
    """V1 (rule §4.1): stage 3 (sample) needs X0, V5 for this candidate (held: its ArtELingo store, ArtELingo-28: its
    28 store) and X1; stage 4 (full) needs X2. `arm` is M<k>, or B1 / B6 (no V5 entry of their own)."""
    outputs = Path(outputs)
    reg = outputs / 'part2/regression'
    x = 'x1' if scale == 'sample' else 'x2'
    if scale == 'sample':
        _read_pass(outputs / 'stores/x0.json')
        if arm in eg_encoders.BY_OWN and eg_encoders.BY_OWN[arm].new:      # V5 covers new stores only (rule §4.1)
            v5 = json.loads((outputs / 'stores/v5.json').read_text())
            store = eg_encoders.BY_OWN[arm].store
            for key in (f'artelingo_full/{store}', f'artelingo28/{store}'):
                assert key in v5['stores'], f'V1: V5 has no entry {key}'
                assert v5['stores'][key].get('pass') is True, f'V1: V5 fails for {key}'
    for part in ('student', 'matched', 'scorer'):
        _read_pass(reg / f'{x}_{part}.json')


def held_path(scale, arm, seed, aspect='emotion', outputs=None):
    """`rb_score.output_path` for every matched arm M2 to M8 (its body knows only M2, M3, M5)."""
    if arm in eg_encoders.BY_OWN and arm.startswith('M'):
        return Path(outputs) / f'part2/{scale}/held/{arm}/P3__s{seed}.npz'
    return _RB_OUTPUT_PATH(scale, arm, seed, aspect, outputs)


def _patched():
    stack = ExitStack()
    stack.enter_context(patch.object(rb_matched, '_code', _code))
    stack.enter_context(patch.object(rb_score, 'output_path', held_path))
    return stack


# --- run --------------------------------------------------------------------------------------------------------

def run(scale, arm, seeds, outputs, device, *, smoke=False, epochs=None, gated=True):
    """Matched arm `arm` (train, then held and ArtELingo-28 outputs), or B1 / B6 on ArtELingo-28 (sample scale)."""
    allowed = set(NEW_ARMS) | set(CT_ARMS) | (set(SMOKE_ONLY_ARMS) if (smoke or not gated) else set())
    assert arm in allowed, arm
    assert arm not in CT_ARMS or scale == 'sample', 'B1 and B6 at full scale are 12-24\'s files'
    eg_paths.check_imports()
    outputs = Path(outputs)
    if gated and not smoke:
        require_gate(outputs, scale, arm)
    world = rb_matched.load_world(scale)
    with _patched():
        if arm in CT_ARMS:
            return rb_matched.run_ct_a28(world, arm, seeds, outputs, gate=False)
        return rb_matched.run_matched(world, arm, seeds, outputs, device, smoke, epochs=epochs, gate=False)


# --- X1 and X2, matched half ------------------------------------------------------------------------------------

def compare_arrays(ours, theirs):
    equal = (ours.shape == theirs.shape and ours.dtype == theirs.dtype
             and bool(np.array_equal(np.ascontiguousarray(ours).view(np.uint8),
                                     np.ascontiguousarray(theirs).view(np.uint8))))
    return {'pass': bool(equal), 'rows': int(len(ours)), 'shape': list(ours.shape), 'dtype': str(ours.dtype),
            'ours_sha256': rb_score.array_sha256(ours), 'rb_sha256': rb_score.array_sha256(theirs)}


def _prob(path):
    with np.load(path, allow_pickle=False) as z:
        assert z.files == ['prob'], path
        return z['prob']


def _list_sha(path):
    return json.loads(Path(path).with_suffix('.json').read_text())['list_sha256']


def regression_items(ours_root, rb_root, seed=0, arm=REGRESSION_ARM):
    """Our M2 files (`ours_root`) against 12-24's (`rb_root`): eval probabilities, mapper tensors (every tensor, key
    order), config, loss history, held and ArtELingo-28 probabilities (bits, shape, dtype, list hash)."""
    ours_root, rb_root = Path(ours_root), Path(rb_root)
    stage2 = f'matched/{arm}/P3__s{seed}.npz'
    items = {}
    ours_meta, rb_meta = (base.verify_output(r / stage2) for r in (ours_root, rb_root))
    with np.load(ours_root / stage2, allow_pickle=False) as o, np.load(rb_root / stage2, allow_pickle=False) as t:
        assert o.files == t.files == ['eval_prob']
        items['matched/eval_prob'] = compare_arrays(o['eval_prob'], t['eval_prob'])
    a, b = (torch.load((r / stage2).with_suffix('.pt'), map_location='cpu', weights_only=True)['mapper']
            for r in (ours_root, rb_root))
    same = list(a) == list(b) and all(a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]) for k in a)
    items['matched/mapper'] = {'pass': bool(same), 'keys': list(a) == list(b), 'tensors': len(a)}
    different = sorted(k for k in set(ours_meta['config']) | set(rb_meta['config'])
                       if k not in CONFIG_IGNORED and ours_meta['config'].get(k) != rb_meta['config'].get(k))
    items['matched/config'] = {'pass': not different, 'different_keys': different}
    items['matched/loss_history'] = {'pass': bool(ours_meta['loss_history'] == rb_meta['loss_history']),
                                     'epochs': len(ours_meta['loss_history'])}
    for kind in ('held', 'a28'):
        name = f'{kind}/{arm}/P3__s{seed}.npz'
        item = compare_arrays(_prob(ours_root / name), _prob(rb_root / name))
        item['list_sha256'] = _list_sha(ours_root / name) == _list_sha(rb_root / name)
        item['pass'] = item['pass'] and item['list_sha256']
        items[f'matched/{kind}_prob'] = item
    return items


def regress(which, device, outputs, *, rb_part2=None):
    """Train M2 (SigLIP 2) seed 0 at the scale with this code, infer held and ArtELingo-28, compare with 12-24's
    `part2/<scale>/`, and write `part2/regression/<which>_matched.json`."""
    assert which in SCALE_OF
    scale = SCALE_OF[which]
    outputs = Path(outputs)
    rb_root = Path(rb_part2 or eg_paths.RB_PART2) / scale
    recorded = base.verify_output(rb_root / f'matched/{REGRESSION_ARM}/P3__s0.npz')['config']['device']
    assert str(torch.device(device)) == recorded, f'12-24 trained on {recorded}; the regression must run on it'
    run_dir = outputs / 'part2/regression' / f'{which}_matched_run'
    run(scale, REGRESSION_ARM, [0], run_dir, device, gated=False)
    ours_root = run_dir / f'part2/{scale}'
    items = regression_items(ours_root, rb_root)
    names, sha256 = rb_student.held_names()
    items['matched/held_list'] = {'pass': sha256 == eg_paths.HELD_SHA256, 'sha256': sha256}
    path = outputs / 'part2/regression' / f'{which}_matched.json'
    record = rb_score.write_regression_items(path, items)
    record.update(code_commit=code_commit(), rb_part2=str(rb_root), scale=scale, arm=REGRESSION_ARM, seed=0)
    rb_score._atomic_json(path, record)
    print(f'{which} matched: pass={record["pass"]}; failing: {[k for k, v in record["items"].items() if not v["pass"]]}',
          flush=True)
    return record


# --- command line -----------------------------------------------------------------------------------------------

def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')    # the CT queue's setting, before any CUDA work
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--outputs', type=Path, default=eg_paths.OUTPUTS)
    if 'regress' in argv:
        parser.add_argument('command', choices=['regress'])
        parser.add_argument('which', choices=list(SCALE_OF))
        parser.add_argument('--device', default='cuda')
        args = parser.parse_args(argv)
        record = regress(args.which, args.device, args.outputs)
        raise SystemExit(0 if record['pass'] else 1)
    parser.add_argument('--smoke', action='store_true', help='outputs under /tmp/claude-0; seed 0; 2 epochs')
    parser.add_argument('--scale', required=True, choices=sorted(set(SCALE_OF.values())))
    parser.add_argument('--arm', required=True, choices=list(NEW_ARMS) + list(SMOKE_ONLY_ARMS) + list(CT_ARMS))
    parser.add_argument('--seeds', type=int, nargs='+', choices=range(5), default=list(range(5)))
    parser.add_argument('--device', default=None)
    args = parser.parse_args(argv)
    assert not args.smoke or eg_paths.smoke_ok(args.outputs), '--smoke outputs must be under /tmp/claude-0'
    assert len(args.seeds) == len(set(args.seeds)), 'duplicate seeds'
    seeds = [0] if args.smoke else args.seeds
    device = args.device or ('cuda' if torch.cuda.is_available() or not args.smoke else 'cpu')
    assert args.smoke or args.arm in CT_ARMS or device.startswith('cuda'), 'full Stage 2 requires GPU'
    return run(args.scale, args.arm, seeds, args.outputs, device, smoke=args.smoke,
               epochs=2 if args.smoke else None)


if __name__ == '__main__':
    main()
