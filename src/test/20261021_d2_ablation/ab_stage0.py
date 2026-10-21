"""Compare R0/R2 exports with the frozen model-sheet D2/T1 exports."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ab_paths import MAIN_TEST, OUTPUTS

SHEET = MAIN_TEST / '20261019_model_sheet/outputs'
GRAPHS = {'emotion': 'Gres', 'style': 'Fres05', 'genre': 'R0o'}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(outputs):
    items = []
    for row, sheet_row in (('R0', 'D2'), ('R2', 'T1')):
        for aspect, graph in GRAPHS.items():
            for seed in range(3):
                name = f'{aspect}__{graph}__s{seed}.npz'
                actual = outputs / 'train' / row / name
                reference = SHEET / sheet_row / name
                item = dict(row=row, sheet_row=sheet_row, aspect=aspect, seed=seed,
                            file_name=name, sha256=None, sheet_sha256=None,
                            arrays_equal=False, best_epoch=None, sheet_best_epoch=None)
                reasons = []
                for path, key in ((actual, 'sha256'), (reference, 'sheet_sha256')):
                    if path.is_file():
                        item[key] = sha256(path)
                    else:
                        reasons.append(f'missing {path}')
                if actual.is_file() and reference.is_file():
                    with np.load(actual, allow_pickle=False) as a, np.load(reference, allow_pickle=False) as b:
                        if set(a.files) != set(b.files):
                            reasons.append('array names differ')
                        else:
                            unequal = [k for k in a.files if a[k].dtype != b[k].dtype or
                                       not np.array_equal(a[k], b[k])]
                            if unequal:
                                reasons.append('arrays differ (values or dtype): ' + ', '.join(unequal))
                            else:
                                item['arrays_equal'] = True
                for path, key in ((actual.with_suffix('.json'), 'best_epoch'),
                                  (reference.with_suffix('.json'), 'sheet_best_epoch')):
                    if path.is_file():
                        item[key] = json.loads(path.read_text())['best_epoch']
                    else:
                        reasons.append(f'missing {path}')
                if item['best_epoch'] != item['sheet_best_epoch']:
                    reasons.append('best_epoch differs')
                if reasons:
                    item['reason'] = '; '.join(reasons)
                items.append(item)
    passed = sum('reason' not in item for item in items)
    result = {'pass': passed == 18, 'items': items}
    outputs.mkdir(parents=True, exist_ok=True)
    pending = outputs / f'stage0.json.{os.getpid()}.pending'
    pending.write_text(json.dumps(result, indent=2) + '\n')
    pending.replace(outputs / 'stage0.json')
    reasons = [f"{i['row']}/{i['file_name']}: {i['reason']}" for i in items if 'reason' in i]
    print(f"AB_STAGE0 {'PASS' if result['pass'] else 'FAIL'} {passed}/18" +
          (' ' + ' | '.join(reasons[:2]) if reasons else ''))
    return int(not result['pass'])


def require_pass(outputs):
    path = outputs / 'stage0.json'
    passed = path.is_file() and json.loads(path.read_text()).get('pass') is True
    print('AB_STAGE0 REQUIRE ' + ('PASS' if passed else f'FAIL missing or failed {path}'))
    return int(not passed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs', type=Path, default=OUTPUTS)
    parser.add_argument('--require-pass', action='store_true')
    args = parser.parse_args()
    return require_pass(args.outputs) if args.require_pass else compare(args.outputs)


if __name__ == '__main__':
    raise SystemExit(main())
