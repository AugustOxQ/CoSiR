"""Clustering AMI at the class count for the four ablation removals (held split of ArtELingo).

Reads the r10 held exports (same files, same sidecar check and row binding as `r20_ret`), clusters with the r17
path (`ns_metrics.partitions`: MiniBatchKMeans, n_init 3, batch 4,096, states 0 to 2, 8 threads) and scores with
`ns_clu.metric_table` (AMI, mean over the 3 states per seed, then over seeds). Gate: arm `k11=40` must reproduce the
`cstar` AMI rows of the r17 sweep record within 1e-9, per seed and mean.

    python t5_ablation_ami.py gate [--out DIR]
    python t5_ablation_ami.py full [--out DIR]
    python t5_ablation_ami.py smoke [--arm A] [--aspect X] [--seeds 0] [--out DIR]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
sys.path.insert(0, str(TEST / '20261109_r20_figures'))
import r20_ret  # noqa: E402  (also puts r10's folder and the r17 folder on sys.path)
import ns_clu  # noqa: E402
import ns_data  # noqa: E402
import ns_metrics  # noqa: E402
import ns_paths  # noqa: E402
import rb_part2_report  # noqa: E402

RECORD = Path('/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/sweep/artelingo__qwen.json')
SET, SPLIT, ENC = 'artelingo', 'held', 'qwen'
GATE_ARM = 'k11=40'
REMOVALS = ('instructed_graphs', 'content_correction', 'buddy', 'relational')
ARMS = (GATE_ARM,) + tuple(f'{GATE_ARM}-no_{r}' for r in REMOVALS)
ASPECTS = ns_paths.ASPECTS
SEEDS = ns_paths.SEEDS
TOL = 1e-9
REPO = Path('/project/CoSiR')


def git_head():
    return subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()


def record_rows():
    """The record's cstar AMI rows at the class count: {aspect: row} plus the record's class counts."""
    rec = json.loads(RECORD.read_text())
    assert rec['complete'] and rec['split'] == SPLIT and rec['set'] == SET and rec['encoder'] == ENC
    assert rec['kmeans_threads'] == ns_paths.KMEANS_THREADS[SET]
    rows = {}
    for r in rec['rows']:
        if r['arm'] == 'cstar' and r['metric'] == 'ami' and r['k'] == rec['class_count'][r['aspect']]:
            assert r['aspect'] not in rows
            rows[r['aspect']] = r
    assert set(rows) == set(ASPECTS) and all(len(r['per_seed']) == len(SEEDS) for r in rows.values())
    return rec, rows


def bind_labels():
    """Painting names and labels, with the same binding as r20_ret.run (held_labels, list_sha256) checked against
    the labels the r17 record used (ns_data.labels)."""
    names, list_sha, lab = rb_part2_report.held_labels()
    names = np.asarray(names)
    assert len(names) == 12281 and names.tolist() == sorted(names.tolist())
    assert names.tolist() == ns_data.keys(SET, SPLIT)
    y = {}
    for a in ASPECTS:
        y[a] = ns_data.labels(SET, SPLIT, a)
        assert np.array_equal(y[a], np.asarray(lab[a], dtype=np.int64)), f'labels differ for {a}'
    assert {a: int((y[a] >= 0).sum()) for a in ASPECTS} == dict(emotion=12281, style=12281, genre=9949)
    return names, list_sha, y


def load_rows(arm, aspect, seed, hashes):
    """The clustering form of r17 (stored float16 cast to float32, not renormalised: `ns_data.cstar(unit=False)`),
    from r20_ret's export path. Sidecars were bound to the list by r20_ret.check_sidecars."""
    path = r20_ret.export_path(arm, aspect, seed)
    for p in (path, path.with_suffix('.json')):
        hashes[str(p)] = r20_ret.sha256(p)
    with np.load(path, allow_pickle=False) as z:
        x16 = z['img']
    assert x16.shape == (12281, 2048) and x16.dtype == np.float16, path
    x = x16.astype(np.float32)
    assert np.isfinite(x).all()
    return x


def arm_aspect(arm, aspect, y, seeds, hashes, log):
    k = ns_paths.CLASS_COUNT[(SET, aspect)]
    assert ns_metrics.class_count(y) == k, (aspect, ns_metrics.class_count(y), k)
    parts, secs = [], []
    for s in seeds:
        t = time.monotonic()
        x = load_rows(arm, aspect, s, hashes)
        if arm == GATE_ARM:   # the gate arm also equals the loader the record used, bit for bit
            assert np.array_equal(x, ns_data.cstar(SET, SPLIT, ENC, aspect, s, unit=False)), (aspect, s)
        parts.append(ns_metrics.partitions(x, k, SET))
        secs.append(time.monotonic() - t)
        log(f'  {arm} {aspect} k={k} seed {s}: {secs[-1]:.1f}s')
    (row,) = ns_clu.metric_table(arm, aspect, np.stack(parts), y, k, metrics=('ami',))
    per_seed = [float(v) for v in row['per_seed']]
    assert np.isfinite(per_seed).all()
    return dict(k=k, seeds=list(seeds), ami_per_seed=per_seed, ami_mean=float(np.mean(per_seed)),
                n_labelled=int((y >= 0).sum()), fit_seconds=secs)


def gate_check(cells, rows):
    """Compare every stored value (per seed and mean) of the gate arm with the record."""
    out = {}
    for a, cell in cells.items():
        rec = rows[a]
        exp = [rec['per_seed'][s] for s in cell['seeds']]
        got = list(cell['ami_per_seed'])
        if len(cell['seeds']) == len(SEEDS):   # all seeds: the mean too (smoke runs fewer seeds, no mean)
            exp.append(rec['mean'])
            got.append(cell['ami_mean'])
        diff = float(np.max(np.abs(np.array(exp) - np.array(got))))
        out[a] = dict(k=cell['k'], expected=exp, got=got, max_abs_diff=diff, **{'pass': bool(diff <= TOL)})
    return out


def run(arms, aspects, seeds, out_dir, mode):
    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    rec, rows = record_rows()
    for a in aspects:
        assert rec['class_count'][a] == ns_paths.CLASS_COUNT[(SET, a)]
    names, list_sha, y = bind_labels()
    r20_ret.check_sidecars(list_sha)
    hashes = {}
    doc = dict(mode=mode, complete=False, set=SET, split=SPLIT, encoder=ENC, metric='ami', seeds=list(seeds),
               kmeans=dict(estimator='MiniBatchKMeans', n_init=ns_metrics.N_INIT, batch_size=ns_metrics.BATCH,
                           states=list(ns_paths.KMEANS_STATES), threads=ns_paths.KMEANS_THREADS[SET],
                           path='ns_metrics.partitions + ns_clu.metric_table'),
               class_count={a: ns_paths.CLASS_COUNT[(SET, a)] for a in aspects},
               class_count_source=dict(table='ns_paths.CLASS_COUNT', record=str(RECORD),
                                       labelled_classes='ns_metrics.class_count(labels) checked equal'),
               list_sha256=list_sha, git_head=git_head(), record=str(RECORD), record_sha256=r20_ret.sha256(RECORD),
               code_sha256=r20_ret.sha256(__file__), input_sha256=hashes, arms={}, gate=None)
    path = out_dir / ('ablation_ami.json' if mode in ('gate', 'full') else f'{mode}.json')

    def save():
        doc['seconds'] = time.monotonic() - t0
        path.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')

    def log(msg):
        print(msg, flush=True)

    for arm in arms:
        doc['arms'][arm] = {}
        for a in aspects:
            doc['arms'][arm][a] = arm_aspect(arm, a, y[a], seeds, hashes, log)
            log(f'{arm} {a}: AMI mean {doc["arms"][arm][a]["ami_mean"]:.5f}')
            save()
        if arm == GATE_ARM:
            doc['gate'] = gate_check(doc['arms'][arm], rows)
            doc['gate_pass'] = all(g['pass'] for g in doc['gate'].values())
            save()
            for a, g in doc['gate'].items():
                log(f'gate {a}: max abs diff {g["max_abs_diff"]:.3e} pass={g["pass"]}')
            if mode in ('gate', 'full') and not doc['gate_pass']:
                save()
                sys.exit('GATE FAILED: removals not run')
    doc['complete'] = True
    save()
    log(f'{doc["seconds"]:.1f}s -> {path}')
    return doc


def table(doc):
    aspects = [a for a in ASPECTS if a in next(iter(doc['arms'].values()))]
    lines = ['| arm | ' + ' | '.join(aspects) + ' |', '|---|' + '---:|' * len(aspects)]
    for arm, cells in doc['arms'].items():
        lines.append(f'| {arm} | ' + ' | '.join(f'{cells[a]["ami_mean"]:.5f}' for a in aspects) + ' |')
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=('gate', 'full', 'smoke'))
    p.add_argument('--out', type=Path, default=HERE / 'out')
    p.add_argument('--arm', default=GATE_ARM, choices=ARMS)
    p.add_argument('--aspect', default='emotion', choices=ASPECTS)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    a = p.parse_args()
    if a.cmd == 'gate':
        doc = run((GATE_ARM,), ASPECTS, SEEDS, a.out, 'gate')
    elif a.cmd == 'full':
        doc = run(ARMS, ASPECTS, SEEDS, a.out, 'full')
    else:
        doc = run((a.arm,), (a.aspect,), tuple(a.seeds), a.out, 'smoke')
    print(table(doc))


if __name__ == '__main__':
    main()
