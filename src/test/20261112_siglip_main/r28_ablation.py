"""r28 job F, part 2: Table 5 (the ablation) on SigLIP 1, ArtELingo held split (12,281 paintings; genre labelled on 9,949).

The Qwen3 version is `20261111_ablation_classcount/t5_ablation_ami.py` (AMI at the class counts 9 / 27 / 10) with the
plain-cosine mAP of `20261109_r20_figures/r20_ret.py` (the paper's Table 5 columns). This module imports their
functions and changes only the roots and the encoder:

  AMI  : stored float16 rows cast to float32 (not renormalised), `ns_metrics.partitions` (MiniBatchKMeans, n_init 3,
         batch 4,096, states 0 to 2, 8 threads, all rows clustered), `ns_clu.metric_table` (mean over the 3 states per
         seed, labels masked after clustering). Seeds 0 to 4; seed mean and per-seed values.
  mAP  : `r20_ret.score_rows` (unit float32 rows, `rt_metrics.retrieval(qe_k=0)`, queries without a relevant item
         dropped), per seed the mean AP; seed mean and per-seed values.

Roots (r14's layout, the sidecar's `arm` is the row name):
  reference c* (arm `k11=40`, not retrained):  REF_ROOT  = r14 `outputs/cstar/siglip1`      `held/k11=40/<aspect>__s<seed>.npz|json`
  removal rows (trained in r28):               R28_ROOT  = `/data/SSD2/cosir_r28/20261112_siglip_main/cstar/siglip1`
                                               `held/<row>/<aspect>__s<seed>.npz|json`, row in REMOVALS
The test split is scored here, never used to choose.

Gates (c* from r14's files): the AMI per-seed values and mean equal r17's sweep record
`/data/SSD2/cosir_r17/.../stage3/sweep/artelingo__siglip1.json` (rows arm `cstar`, metric `ami`, k = class count) within
1e-9, and the seed mean equals r17's `records/verdicts.json` cell `SC1/artelingo/siglip1/<aspect>/ami/<k>/plain`
(`cand`) within 1e-9. The mAP has no earlier record for this read-out on SigLIP 1: it is reported.

    python r28_ablation.py replay  [--out DIR] [--records DIR]      c* only (r14's held files): the gates, the replay values
    python r28_ablation.py full    [--out DIR] [--records DIR]      c* + the four removals (needs the r28 held exports)
    python r28_ablation.py smoke   [--arm A] [--aspect X] [--seeds 0] [--out DIR]
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
sys.path.insert(0, str(TEST / '20261111_ablation_classcount'))
import t5_ablation_ami as t5  # noqa: E402  (imports r20_ret, ns_*, rb_part2_report; puts r10, r17 folders on sys.path)
import r20_ret  # noqa: E402
import ns_clu  # noqa: E402
import ns_data  # noqa: E402
import ns_metrics  # noqa: E402
import ns_paths  # noqa: E402

SET, SPLIT, ENC = 'artelingo', 'held', 'siglip1'
ASPECTS = ns_paths.ASPECTS
SEEDS = ns_paths.SEEDS
CLASS_COUNT = {a: ns_paths.CLASS_COUNT[(SET, a)] for a in ASPECTS}
REF_ARM = 'k11=40'
REMOVALS = ('instructed_graphs', 'content_correction', 'buddy', 'relational')
ARMS = (REF_ARM,) + REMOVALS
N_HELD, WIDTH = 12281, 2048
# r28's arm names are the row names; t5 / r20_ret know r10's legacy names (`k11=40-no_<removal>`, eight removals, Qwen
# root). The mapping is explicit and asserted; t5's own arm list, root and sidecar checker (`r20_ret.check_sidecars`) are
# never called here (they read the Qwen tree and expect all eight removals).
T5_NAME = {REF_ARM: t5.GATE_ARM, **{r: f'{t5.GATE_ARM}-no_{r}' for r in REMOVALS}}
assert REF_ARM == t5.GATE_ARM and set(t5.REMOVALS) == set(REMOVALS) and set(T5_NAME.values()) <= set(t5.ARMS) \
    and len(T5_NAME) == len(ARMS) == len(set(T5_NAME.values())), 'r28 arms do not map onto t5 arms'
TOL = 1e-9

REF_ROOT = ns_paths.R14_OUT / 'cstar/siglip1'
R28_ROOT = Path('/data/SSD2/cosir_r28/20261112_siglip_main/cstar/siglip1')
R17_SWEEP = Path('/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage3/sweep/artelingo__siglip1.json')
R17_VERDICTS = TEST / '20261103_new_scores/records/verdicts.json'
OUT = Path('/data/SSD2/cosir_r28/20261112_siglip_main/ablation')
RECORDS = HERE / 'records'
REPO = Path('/project/CoSiR')


# ------------------------------------------------------------------------------------------------ roots and joins
def export_dir(arm, ref_root=REF_ROOT, r28_root=R28_ROOT):
    """Folder of an arm's held exports: c* from r14's tree, a removal from r28's tree."""
    assert arm in ARMS, arm
    return Path(ref_root if arm == REF_ARM else r28_root) / 'held' / arm


def export_path(arm, aspect, seed, ref_root=REF_ROOT, r28_root=R28_ROOT):
    return export_dir(arm, ref_root, r28_root) / f'{aspect}__s{seed}.npz'


def accepted_sidecar_arms(arm):
    """The sidecar `arm` of r14's c* is `k11=40`; a removal's is its row name (T1's brief) or, as r10 named it,
    `k11=40-no_<row>`."""
    return {T5_NAME[arm]} if arm == REF_ARM else {arm, T5_NAME[arm]}


def check_sidecar(side, x, arm, aspect, seed, n, list_sha):
    """The join: the export's sidecar names this arm, aspect and seed, the held list (hash and row count), and the
    array's hash (C6: rows are bound to painting names through the list hash, never by position alone)."""
    assert side['kind'] == 'held' and side['aspect'] == aspect and side['seed'] == seed, (side['kind'], aspect, seed)
    assert side['arm'] in accepted_sidecar_arms(arm), (side['arm'], arm)
    assert side['rows'] == n == len(x) and side['list_sha256'] == list_sha, (side['rows'], n, len(x))
    assert side['arrays']['img']['shape'] == list(x.shape) and side['arrays']['img']['dtype'] == str(x.dtype)
    assert side['arrays']['img']['sha256'] == hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(), 'array hash'


def load_rows(arm, aspect, seed, list_sha, hashes, n=N_HELD, width=WIDTH, ref_root=REF_ROOT, r28_root=R28_ROOT):
    """float16 export -> (x32 for the clustering, not renormalised; the float16 array). Sidecar-checked."""
    path = export_path(arm, aspect, seed, ref_root, r28_root)
    side = json.loads(path.with_suffix('.json').read_text())
    for p in (path, path.with_suffix('.json')):
        hashes[str(p)] = r20_ret.sha256(p)
    with np.load(path, allow_pickle=False) as z:
        assert z.files == ['img'], (path, z.files)
        x16 = z['img']
    assert x16.shape == (n, width) and x16.dtype == np.float16, (path, x16.shape, x16.dtype)
    check_sidecar(side, x16, arm, aspect, seed, n, list_sha)
    x = x16.astype(np.float32)
    assert np.isfinite(x).all(), path
    return x


# ------------------------------------------------------------------------------------------------ pure parts
def seed_mean(v):
    v = np.asarray(v, dtype=np.float64)
    assert v.ndim == 1 and np.isfinite(v).all()
    return float(v.mean())


def deltas(arm_cells, ref_cells):
    """Removal minus c*, per aspect and metric: the difference of seed means and the seed-paired differences."""
    out = {}
    for a in ASPECTS:
        out[a] = {}
        for m in ('ami', 'map'):
            r, c = np.asarray(ref_cells[a][f'{m}_per_seed']), np.asarray(arm_cells[a][f'{m}_per_seed'])
            assert r.shape == c.shape
            out[a][m] = dict(delta=float(c.mean() - r.mean()), per_seed=[float(v) for v in (c - r)])
    return out


def table_md(doc):
    arms = list(doc['arms'])
    ref = doc['arms'][REF_ARM]
    lines = ['| arm | AMI emo | AMI style | AMI genre | mAP emo | mAP style | mAP genre |', '|---|' + '---:|' * 6]
    for arm in arms:
        cells = doc['arms'][arm]
        row = [f'{cells[a]["ami_mean"]:.4f}' for a in ASPECTS] + [f'{cells[a]["map_mean"]:.4f}' for a in ASPECTS]
        lines.append(f'| {arm} | ' + ' | '.join(row) + ' |')
        if arm != REF_ARM and REF_ARM in doc['arms']:
            d = deltas(cells, ref)
            row = [f'{d[a][m]["delta"]:+.4f}' for m in ('ami', 'map') for a in ASPECTS]
            lines.append(f'| {arm} minus c* | ' + ' | '.join(row) + ' |')
    return '\n'.join(lines)


# ------------------------------------------------------------------------------------------------ scoring
def score_arm_aspect(arm, aspect, y, names, list_sha, seeds, hashes, log, ref_root=REF_ROOT, r28_root=R28_ROOT):
    k = CLASS_COUNT[aspect]
    assert ns_metrics.class_count(y) == k, (aspect, ns_metrics.class_count(y), k)
    parts, aps, secs = [], [], []
    for s in seeds:
        t = time.monotonic()
        x = load_rows(arm, aspect, s, list_sha, hashes, ref_root=ref_root, r28_root=r28_root)
        if arm == REF_ARM and ref_root == REF_ROOT:   # the loader equals r17's bit for bit
            assert np.array_equal(x, ns_data.cstar(SET, SPLIT, ENC, aspect, s, unit=False)), (aspect, s)
        parts.append(ns_metrics.partitions(x, k, SET))
        res = r20_ret.score_rows(r20_ret.unit(x), y, names, 8)     # unit float32 of the float16 export, qe_k = 0
        aps.append(dict(map=float(res['ap'].mean()), n_queries=len(res['query_ids']), dropped=int(res['dropped'])))
        secs.append(time.monotonic() - t)
        log(f'  {arm} {aspect} k={k} seed {s}: {secs[-1]:.1f}s  map {aps[-1]["map"]:.5f}')
    (row,) = ns_clu.metric_table(arm, aspect, np.stack(parts), y, k, metrics=('ami',))
    ami = [float(v) for v in row['per_seed']]
    mp = [a['map'] for a in aps]
    assert np.isfinite(ami).all() and np.isfinite(mp).all()
    assert len({a['n_queries'] for a in aps}) == 1
    return dict(k=k, seeds=list(seeds), ami_per_seed=ami, ami_mean=seed_mean(ami), map_per_seed=mp, map_mean=seed_mean(mp),
                n_labelled=int((y >= 0).sum()), n_queries=aps[0]['n_queries'], dropped=aps[0]['dropped'],
                fit_seconds=secs)


def r17_rows():
    """r17's sweep record rows for SigLIP 1 c* at the class counts: {aspect: row}."""
    rec = json.loads(R17_SWEEP.read_text())
    assert rec['complete'] and rec['split'] == SPLIT and rec['set'] == SET and rec['encoder'] == ENC
    assert rec['kmeans_threads'] == ns_paths.KMEANS_THREADS[SET]
    rows = {}
    for r in rec['rows']:
        if r['arm'] == 'cstar' and r['metric'] == 'ami' and r['k'] == rec['class_count'][r['aspect']]:
            assert r['aspect'] not in rows
            rows[r['aspect']] = r
    assert set(rows) == set(ASPECTS) and all(len(r['per_seed']) == len(SEEDS) for r in rows.values())
    return rows


def verdict_cands():
    """r17's verdicts cell values (SigLIP 1 c*, AMI at the class count against plain): {aspect: cand}."""
    v = json.loads(R17_VERDICTS.read_text())
    out = {}
    for c in v['cells']:
        if (c['encoder'], c['candidate'], c['metric'], c['kind'], c['comparator'], c['set']) == \
                (ENC, 'cstar', 'ami', 'class_count', 'plain', SET):
            assert c['aspect'] not in out and c['k'] == CLASS_COUNT[c['aspect']] and c['n_rows'] == N_HELD
            out[c['aspect']] = c['cand']
    assert set(out) == set(ASPECTS)
    return out


def gates(cells):
    """c* against r17's two records (full seed set only)."""
    rows, cands, out = r17_rows(), verdict_cands(), {}
    full = {a: c for a, c in cells.items() if len(c['seeds']) == len(SEEDS)}
    out['sweep_record'] = dict(source=str(R17_SWEEP), per_aspect=t5.gate_check(full, rows))
    out['verdicts_cell'] = {a: dict(record=cands[a], got=c['ami_mean'], abs_diff=abs(cands[a] - c['ami_mean']),
                                    rounded_3=round(c['ami_mean'], 3), **{'pass': bool(abs(cands[a] - c['ami_mean']) <= TOL)})
                            for a, c in full.items()}
    out['all_pass'] = bool(all(g['pass'] for g in out['sweep_record']['per_aspect'].values()) and
                           all(g['pass'] for g in out['verdicts_cell'].values()))
    return out


def git_head():
    return subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True,
                          check=True).stdout.strip()


def run(arms, aspects, seeds, out_dir, mode, ref_root=REF_ROOT, r28_root=R28_ROOT, log=print):
    t0 = time.monotonic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, list_sha, y = t5.bind_labels()      # held names, list hash, labels (asserted equal to r17's)
    for a in aspects:
        assert CLASS_COUNT[a] == t5.ns_paths.CLASS_COUNT[(SET, a)]
    hashes = {}
    doc = dict(mode=mode, complete=False, set=SET, split=SPLIT, encoder=ENC, seeds=list(seeds), metric='ami + plain-cosine map',
               roots=dict(ref=str(ref_root), r28=str(r28_root)), class_count={a: CLASS_COUNT[a] for a in aspects},
               kmeans=dict(estimator='MiniBatchKMeans', n_init=ns_metrics.N_INIT, batch_size=ns_metrics.BATCH,
                           states=list(ns_paths.KMEANS_STATES), threads=ns_paths.KMEANS_THREADS[SET],
                           path='ns_metrics.partitions + ns_clu.metric_table (t5 / r17)'),
               map_path='r20_ret.score_rows (rt_metrics.retrieval qe_k=0, unit float32 rows)',
               list_sha256=list_sha, git_head=git_head(), code_sha256=r20_ret.sha256(__file__), input_sha256=hashes,
               arms={}, gate=None)
    path = out_dir / ('ablation_siglip1.json' if mode in ('replay', 'full') else f'{mode}.json')

    def save():
        doc['seconds'] = time.monotonic() - t0
        path.write_text(json.dumps(doc, indent=2, allow_nan=False) + '\n')

    for arm in arms:
        doc['arms'][arm] = {}
        for a in aspects:
            doc['arms'][arm][a] = score_arm_aspect(arm, a, y[a], names, list_sha, seeds, hashes, log, ref_root, r28_root)
            log(f'{arm} {a}: AMI {doc["arms"][arm][a]["ami_mean"]:.5f}  mAP {doc["arms"][arm][a]["map_mean"]:.5f}')
            save()
        if arm == REF_ARM and len(seeds) == len(SEEDS) and tuple(aspects) == ASPECTS:
            doc['gate'] = gates(doc['arms'][arm])
            doc['gate_pass'] = doc['gate']['all_pass']
            save()
            log(f'gate (r17 sweep record and verdicts cell): pass={doc["gate_pass"]}')
            if mode in ('replay', 'full') and not doc['gate_pass']:
                sys.exit('GATE FAILED: removals not run')
    if REF_ARM in doc['arms'] and len(doc['arms']) > 1:
        ref = doc['arms'][REF_ARM]
        doc['deltas_vs_cstar'] = {arm: deltas(c, ref) for arm, c in doc['arms'].items() if arm != REF_ARM}
    doc['complete'] = True
    save()
    log(f'{doc["seconds"]:.1f}s -> {path}')
    return doc


def replay_record(doc):
    """The c* values for records/ablation_ref_replay.json."""
    cells = doc['arms'][REF_ARM]
    return dict(kind='r28_ablation_ref_replay', encoder=ENC, split=SPLIT, arm=REF_ARM, root=doc['roots']['ref'],
                seeds=doc['seeds'], list_sha256=doc['list_sha256'], git_head=doc['git_head'],
                class_count=doc['class_count'],
                ami={a: dict(k=c['k'], per_seed=c['ami_per_seed'], mean=c['ami_mean'], rounded_3=round(c['ami_mean'], 3))
                     for a, c in cells.items()},
                map_plain_cosine={a: dict(per_seed=c['map_per_seed'], mean=c['map_mean'], rounded_3=round(c['map_mean'], 3),
                                          n_queries=c['n_queries'], dropped=c['dropped']) for a, c in cells.items()},
                gate=doc['gate'], gate_pass=doc['gate_pass'], seconds=doc['seconds'])


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('cmd', choices=('replay', 'full', 'smoke'))
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--records', type=Path, default=None, help='folder for records/ablation_ref_replay.json (replay and full)')
    p.add_argument('--arm', default=REF_ARM, choices=ARMS)
    p.add_argument('--aspect', default='emotion', choices=ASPECTS)
    p.add_argument('--seeds', type=int, nargs='+', default=[0])
    a = p.parse_args()
    if a.cmd == 'replay':
        doc = run((REF_ARM,), ASPECTS, SEEDS, a.out, 'replay')
    elif a.cmd == 'full':
        doc = run(ARMS, ASPECTS, SEEDS, a.out, 'full')
    else:
        doc = run((a.arm,), (a.aspect,), tuple(a.seeds), a.out, 'smoke')
    if a.cmd in ('replay', 'full') and a.records is not None:
        write_json(Path(a.records) / 'ablation_ref_replay.json', replay_record(doc))
    print(table_md(doc))


if __name__ == '__main__':
    main()
