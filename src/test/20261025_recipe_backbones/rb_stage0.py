"""Stage 0 (V0): part 1's code reproduces 12-22's R0, R6 and R7 seed 0 exports on the old 5,000, bit for bit.

Trains R0-5, R6-3 and R7-3 at seed 0 with the P exports off (`rb_part1`), compares each aspect's old-5,000 `img` and
`cap` arrays and best epoch with `AB_OUT/train/{R0,R6,R7}/<aspect>__<graph>__s0.npz|json`, counts P's genre labels
and times one export's scoring on a synthetic unit-norm [29232, 2048] array. Writes `<outputs>/stage0.json`
`{pass, items, genre_count, timing, smoke, code_sha256}`. A smoke run (2 epochs) cannot equal 12-22's best epochs:
it runs every step, records the comparison and never passes.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

import rb_paths
import rb_rows
import rb_part1

GRAPHS = {'emotion': 'Gres', 'style': 'Fres05', 'genre': 'R0o'}   # 12-22's graph row per aspect
N_AMI = 10
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_PARTITIONS = 3


def compare(outputs, ab_out=rb_paths.AB_OUT):
    """One item per recipe and aspect; pass requires equal key sets, dtypes, values and best epochs."""
    items = []
    for recipe, old in rb_rows.REGRESSION.items():
        for aspect, graph in GRAPHS.items():
            mine_npz = Path(outputs) / 'part1' / 'train' / recipe / f'{aspect}__s0__old5000.npz'
            mine_json = Path(outputs) / 'part1' / 'train' / recipe / f'{aspect}__s0.json'
            ref_npz = Path(ab_out) / 'train' / old / f'{aspect}__{graph}__s0.npz'
            item = dict(recipe=recipe, ab_row=old, aspect=aspect, arrays_equal=False,
                        best_epoch=None, ab_best_epoch=None, ref=str(ref_npz))
            reasons = [f'missing {p}' for p in (mine_npz, mine_json, ref_npz, ref_npz.with_suffix('.json'))
                       if not p.is_file()]
            if not reasons:
                with np.load(mine_npz, allow_pickle=False) as a, np.load(ref_npz, allow_pickle=False) as b:
                    if set(a.files) != set(b.files) or set(a.files) != {'img', 'cap'}:
                        reasons.append('array names differ')
                    else:
                        bad = [k for k in a.files if a[k].dtype != b[k].dtype or a[k].shape != b[k].shape
                               or not np.array_equal(a[k], b[k])]
                        item['arrays_equal'] = not bad
                        if bad:
                            reasons.append('arrays differ: ' + ', '.join(bad))
                item['best_epoch'] = json.loads(mine_json.read_text())['best_epoch']
                item['ab_best_epoch'] = json.loads(ref_npz.with_suffix('.json').read_text())['best_epoch']
                if item['best_epoch'] != item['ab_best_epoch']:
                    reasons.append('best_epoch differs')
            if reasons:
                item['reason'] = '; '.join(reasons)
            items.append(item)
    return items


def time_scoring(n=rb_part1.N_P, seed=0):
    """Time one export's scoring on a synthetic unit-norm array: partitions, N_AMI AMI calls, bootstrap estimate."""
    from pav_score import partitions
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 2048)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    labels = rng.integers(0, 9, size=n)
    with threadpool_limits(limits=8):
        t0 = time.monotonic()
        parts = partitions(x, 20)
        t_parts = time.monotonic() - t0
        durations = []
        for i in range(N_AMI):
            idx = rng.integers(0, n, size=n)
            t0 = time.monotonic()
            adjusted_mutual_info_score(labels[idx], parts[i % len(parts)][idx])
            durations.append(time.monotonic() - t0)
    per_ami = float(np.mean(durations))
    calls = BOOTSTRAP_DRAWS * BOOTSTRAP_PARTITIONS * len(rb_part1.ASPECTS) * len(rb_rows.ORDER)
    return dict(n=n, partitions_seconds=t_parts, ami_calls=N_AMI, ami_seconds=durations, ami_mean_seconds=per_ami,
                bootstrap_ami_calls=calls, bootstrap_seconds_estimate=per_ami * calls,
                basis=f'{BOOTSTRAP_DRAWS} draws x {BOOTSTRAP_PARTITIONS} partitions x {len(rb_part1.ASPECTS)} '
                      f'aspects x {len(rb_rows.ORDER)} recipes AMI calls at the mean call time (random 9-class '
                      f'labels; genre has fewer labelled rows, so this is an upper bound); single process')


def run(outputs, device='cuda', *, smoke=False):
    outputs = Path(outputs)
    recipes = list(rb_rows.REGRESSION)
    rb_part1.train(recipes, outputs, device, smoke=smoke, with_p=False)
    items = compare(outputs)
    import rb_data
    genre_count = rb_data.genre_label_count()
    timing = time_scoring()
    passed = all('reason' not in i for i in items) and len(items) == 9 and not smoke
    record = dict(**{'pass': passed}, items=items, genre_count=genre_count, genre_of=rb_part1.N_P,
                  timing=timing, smoke=smoke, code_sha256=rb_part1.code_hashes()['all'])
    outputs.mkdir(parents=True, exist_ok=True)
    rb_part1.atomic_json(outputs / 'stage0.json', record)
    bad = [f"{i['recipe']}/{i['aspect']}: {i['reason']}" for i in items if 'reason' in i]
    print(f"RB_STAGE0 {'PASS' if passed else 'FAIL'} {len(items) - len(bad)}/{len(items)} smoke={smoke} "
          f"genre_count={genre_count} bootstrap_estimate_s={timing['bootstrap_seconds_estimate']:.0f}"
          + (' | ' + ' | '.join(bad[:3]) if bad else ''))
    return int(not passed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--recipes', nargs='+', choices=list(rb_rows.REGRESSION),
                        default=list(rb_rows.REGRESSION), help='fixed: R0-5, R6-3, R7-3')
    parser.add_argument('--outputs', type=Path, default=rb_paths.OUTPUTS)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args(argv)
    assert args.recipes == list(rb_rows.REGRESSION), 'stage 0 trains exactly R0-5, R6-3 and R7-3'
    return run(args.outputs, args.device, smoke=args.smoke)


if __name__ == '__main__':
    raise SystemExit(main())
