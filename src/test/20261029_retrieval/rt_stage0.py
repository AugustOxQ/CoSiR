"""Stage 0 gates: validation seed ranges and a replay of frozen r10 checks.

  python rt_stage0.py {0a,0b} [--outputs DIR] [--workers 8] [--smoke N]

Smoke slices numerical inputs after the frozen loaders validate their sources.
Full-source hashes/counts (0b ii/iii) remain integrity checks in smoke mode.
Only a full 0a writes RANGES_FILE; committing that file is the queue's job.
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import rt_paths
import rt_data
import rt_metrics
import rt_stats
import rs_vh


# Spec 3.5, row 0b; emotion / style / genre, frozen before this replay.
AMI_REFERENCES = {
    'cstar': dict(zip(rt_paths.ASPECTS, (0.10658729378204021, 0.40664322288431654,
                                      0.5001038781216712))),
    'instructed': dict(zip(rt_paths.ASPECTS, (0.09207439052277983, 0.40256525412113753,
                                           0.5061513009972589))),
    'percept': {'emotion': 0.08742922158584242},
}


class Inputs:
    """Every stage data call goes here; 0a cannot request a held split."""

    def __init__(self, smoke=None, validation_only=False):
        self.smoke = smoke
        self.validation_only = validation_only
        self.hashes = {}

    def read(self, kind, split, *args):
        assert split in ('val', 'held'), split
        assert not self.validation_only or split == 'val', '0a must read validation only'
        fn = getattr(rt_data, kind)
        value = fn(args[0], split, *args[1:]) if kind in ('raw', 'embed') else fn(split)
        if isinstance(value, dict):
            value = {a: x[:self.smoke] for a, x in value.items()}
            for a, x in value.items():
                self.hashes[f'{kind}/{split}/{a}'] = rs_vh.array_sha(x)
        else:
            value = value[:self.smoke]
            key = '/'.join(map(str, (kind, split, *args)))
            self.hashes[key] = (rs_vh.names_sha(value) if kind == 'names'
                                else rs_vh.array_sha(value))
        return value


def _write(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')


def validation_seed(data, aspect, seed, labels, dist, workers):
    """One complete R0 seed, also callable for the bounded timing run."""
    x = data.read('embed', 'val', 'cstar', aspect, seed)
    score = float(rt_metrics.retrieval(x, labels[aspect], workers=workers)['ap'].mean())
    ndcg = float(rt_metrics.graded(x, dist, workers=workers).mean()) if aspect == 'emotion' else None
    return score, ndcg


def stage0a(outputs, workers=8, smoke=None):
    start = time.perf_counter()
    outputs = Path(outputs)
    assert smoke is None or outputs.resolve() != rt_paths.OUTPUTS.resolve(), 'smoke needs separate outputs'
    data = Inputs(smoke, validation_only=True)
    data.read('names', 'val')
    labels = data.read('labels', 'val')
    dist = data.read('emotion_dist', 'val')
    doc = {'map': {}, 'ndcg': {}}
    timing = {}
    for aspect in rt_paths.ASPECTS:
        maps, ndcgs, seconds = [], [], []
        for seed in rt_paths.SEEDS:
            tick = time.perf_counter()
            score, ndcg = validation_seed(data, aspect, seed, labels, dist, workers)
            maps.append(score)
            if ndcg is not None:
                ndcgs.append(ndcg)
            seconds.append(time.perf_counter() - tick)
        doc['map'][aspect] = dict(per_seed=maps, r=rt_stats.seed_range(maps))
        if ndcgs:
            doc['ndcg'][aspect] = dict(per_seed=ndcgs, r=rt_stats.seed_range(ndcgs))
        timing[aspect] = seconds
    doc.update(inputs=data.hashes, smoke=smoke, seed_seconds=timing,
               total_seconds=time.perf_counter() - start)
    path = outputs / 'stage0a/ranges.json'
    _write(path, doc)
    if smoke is None:
        _write(rt_paths.RANGES_FILE, doc)
        assert path.read_bytes() == rt_paths.RANGES_FILE.read_bytes()
    print(f'0a: PASS; {len(rt_paths.SEEDS)} seeds/aspect; {doc["total_seconds"]:.2f}s; {path}')
    return 0


def _init_partition_worker():
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '1'
    threadpool_limits(1)


def _partition(task):
    arm, aspect, seed = task
    tick = time.perf_counter()
    data = Inputs()
    # Keep c* float16 and PercepT probabilities in r10's stored representation.
    kind = 'embed' if arm == 'instructed' else 'raw'
    x = data.read(kind, 'held', arm, aspect, seed)
    reading = 'own' if arm == 'percept' else 'r20'
    with threadpool_limits(1):
        parts = rs_vh.own_partition(x) if reading == 'own' else rs_vh.r20_partition(x)
    return task, {reading: parts}, time.perf_counter() - tick


def held_ami(workers=8, pairs=None):
    """Replay selected (arm, aspect) pairs, defaulting to every 0b(i) check.

    Uses r10's point_table (including the negative-label mask), then its own
    state and seed averaging helpers. No partition cache or frozen scores enter.
    """
    assert 1 <= workers <= 8
    if pairs is None:
        pairs = [(arm, aspect) for arm, refs in AMI_REFERENCES.items() for aspect in refs]
    labels = Inputs().read('labels', 'held')
    tasks = [(arm, aspect, seed) for arm, aspect in pairs
             for seed in (rt_paths.SEEDS if arm in rt_data.SEEDED else (None,))]
    with mp.get_context('fork').Pool(workers, initializer=_init_partition_worker) as pool:
        results = pool.map(_partition, tasks, chunksize=1)
    values, timings = {}, {}
    for arm, aspect in pairs:
        parts = {seed if seed is not None else 0: ps
                 for (a, h, seed), ps, _ in results if (a, h) == (arm, aspect)}
        table = rs_vh.point_table(labels[aspect], {arm: parts})
        reading = 'own' if arm == 'percept' else 'r20'
        values[f'{arm}/{aspect}'] = rs_vh._mean_over(rs_vh._point_val(table), arm, sorted(parts), reading)
        timings[f'{arm}/{aspect}'] = [seconds for (a, h, _), _, seconds in results
                                     if (a, h) == (arm, aspect)]
    return values, timings


def _item(name, value, reference, tolerance=0):
    if isinstance(value, (float, int)) and isinstance(reference, (float, int)):
        passed = bool(np.isfinite(value) and abs(value - reference) <= tolerance)
    else:
        passed = value == reference
    return dict(name=name, value=value, reference=reference, tolerance=tolerance, **{'pass': passed})


def stage0b(outputs, workers=8, smoke=None):
    start = time.perf_counter()
    assert smoke is None or Path(outputs).resolve() != rt_paths.OUTPUTS.resolve(), 'smoke needs separate outputs'
    items, timing = [], {}
    tick = time.perf_counter()
    values, partition_seconds = held_ami(workers) if smoke is None else ({}, {})
    for arm, refs in AMI_REFERENCES.items():
        for aspect, ref in refs.items():
            name = f'i/ami/{arm}/{aspect}'
            if smoke is None:
                items.append(_item(name, values[f'{arm}/{aspect}'], ref, 1e-9))
            else:
                items.append(dict(name=name, value='skipped (smoke)', reference=ref,
                                  tolerance=1e-9, **{'pass': True}))
    timing['i'] = time.perf_counter() - tick

    # These records describe full held inputs. Check before any smoke slicing;
    # a prefix's hash/count cannot be compared with a full-source sidecar.
    tick = time.perf_counter()
    source = Inputs()
    image_sha = rs_vh.array_sha(source.read('raw', 'held', 'plain', 'emotion'))
    sidecars = {}
    for aspect in rt_paths.ASPECTS:
        for seed in rt_paths.SEEDS:
            path = rt_data.CSTAR_ROOT / f'held/k11=40/{aspect}__s{seed}.json'
            side = json.loads(path.read_text())
            assert (side['aspect'], side['seed'], side['rows'], side['list_sha256']) == (
                aspect, seed, rt_paths.N_HELD, rs_vh.names_sha(source.read('names', 'held')))
            sidecars[aspect, seed] = side
            items.append(_item(f'ii/image_sha256/{aspect}/s{seed}', image_sha, side['image_sha256']))
    timing['ii'] = time.perf_counter() - tick
    tick = time.perf_counter()
    genre = source.read('labels', 'held')['genre']
    items.append(_item('iii/genre/labelled', int((genre >= 0).sum()), 9949))
    items.append(_item('iii/genre/masked', int((genre < 0).sum()), 2332))
    timing['iii'] = time.perf_counter() - tick

    tick = time.perf_counter()
    import rb_student

    data = Inputs(smoke)
    x = data.read('embed', 'val', 'plain', 'emotion')
    infer_seconds = {}
    for aspect in rt_paths.ASPECTS:
        for seed in rt_paths.SEEDS:
            head_start = time.perf_counter()
            weights = rt_data.CSTAR_ROOT / f'weights/k11=40/{aspect}__s{seed}.pt'
            items.append(_item(f'iv/weights_sha256/{aspect}/s{seed}', rs_vh.file_sha(weights),
                               sidecars[aspect, seed]['weights_sha256']))
            pred = rb_student.infer(weights, aspect, 2048, x, 'cpu')
            saved = data.read('raw', 'val', 'cstar', aspect, seed)
            assert pred.shape == saved.shape == (len(x), 2048)
            assert pred.dtype == saved.dtype == np.float16
            diff = float(np.max(np.abs(pred.astype(np.float32) - saved.astype(np.float32))))
            items.append(_item(f'iv/max_abs_difference/{aspect}/s{seed}', diff, 0.0, 1e-3))
            infer_seconds[f'{aspect}/s{seed}'] = time.perf_counter() - head_start
    timing['iv'] = time.perf_counter() - tick
    passed = all(item['pass'] for item in items)
    doc = dict(items=items, smoke=smoke, workers=workers, item_seconds=timing,
               partition_seconds=partition_seconds, head_seconds=infer_seconds,
               total_seconds=time.perf_counter() - start, **{'pass': passed})
    path = Path(outputs) / 'stage0b/check.json'
    _write(path, doc)
    print(f'0b: {"PASS" if passed else "FAIL"}; {sum(i["pass"] for i in items)}/{len(items)} checks; '
          f'{doc["total_seconds"]:.2f}s; {path}')
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('0a', '0b'))
    parser.add_argument('--outputs', type=Path, default=rt_paths.OUTPUTS)
    parser.add_argument('--workers', type=int, choices=range(1, 9), default=8)
    parser.add_argument('--smoke', type=int)
    args = parser.parse_args()
    assert args.smoke is None or 0 < args.smoke <= min(rt_paths.N_VAL, rt_paths.N_HELD)
    fn = stage0a if args.stage == '0a' else stage0b
    return fn(args.outputs, args.workers, args.smoke)


if __name__ == '__main__':
    raise SystemExit(main())
