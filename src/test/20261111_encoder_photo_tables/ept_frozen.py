"""Frozen c* (the ArtELingo-trained emotion head, no photo fine-tuning) on the photo test sets: class-count AMI.

Reuses r17's code by import (src/test/20261103_new_scores/): ns_photo (keys, labels, plain, teacher), ns_metrics.partitions
(mini-batch k-means, 3 states), ns_clu.metric_table (per seed = mean over states; mean over seeds).
  gate   recompute r17's four SC4 Qwen cells (photo model against plain / teacher) with r17's own paths
         (ns_clu_cells._cand_parts, _comp_parts, ns_clu.paired_cell point values); pass = max abs diff <= 1e-9.
  run    frozen c* rows (r14 exports/<set>/cstar_qwen/test__s<seed>.npz, joined by key as ns_photo.photo_stored does),
         class-count AMI per seed and mean; Affection also at 20 clusters (r12/r13 protocol check) from r12's exports.
Usage: ept_frozen.py gate | run
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R17 = Path('/project/CoSiR/src/test/20261103_new_scores')
sys.path.insert(0, str(R17))
import ns_paths as NP  # noqa: E402

NP.add_rounds()
import ns_photo  # noqa: E402
import ns_clu  # noqa: E402
import ns_clu_cells as CC  # noqa: E402
import ns_metrics  # noqa: E402
import rs_vh  # noqa: E402

TOL = 1e-9
VERD = R17 / 'records/verdicts.json'
GATE_CELLS = [('affection', 'plain', 9), ('affection', 'teacher', 9), ('emoset', 'plain', 8), ('emoset', 'teacher', 8)]
R13_VERDICT = Path('/data/SSD2/cosir_r13/20261031_photo_finetune/outputs/stage3/verdict.json')
R14_DIAG = NP.R14_OUT / 'diag/photo_minus_cstar.json'


def gate():
    cells = {c['id']: c for c in json.loads(VERD.read_text())['cells']}
    out, worst = {}, 0.0
    for s, comp, k in GATE_CELLS:
        cid = f'SC4/{s}/qwen/emotion/ami/{k}/{comp}'
        y = ns_photo.labels(s, 'test')
        cand = CC._cand_parts(s, 'test', 'qwen', 'emotion', k)
        cp = CC._comp_parts(s, 'test', 'qwen', 'emotion', comp, k)
        old = ns_clu.WORKERS
        ns_clu.WORKERS = 1
        try:
            cell = ns_clu.paired_cell(cand, cp, y, k, metrics=('ami',), draws=1)['ami']   # point values, 1 draw
        finally:
            ns_clu.WORKERS = old
        # second route: metric_table means (the form used below for frozen c*)
        tc = ns_clu.metric_table('photo', 'emotion', cand, y, k, ('ami',))[0]
        tp = ns_clu.metric_table(comp, 'emotion', cp, y, k, ('ami',))[0]
        rec = cells[cid]
        d = dict(cand_recorded=rec['cand'], comp_recorded=rec['comp'], cand_paired=cell['cand'], comp_paired=cell['comp'],
                 cand_table=tc['mean'], comp_table=tp['mean'])
        d['max_abs_diff_paired'] = max(abs(cell['cand'] - rec['cand']), abs(cell['comp'] - rec['comp']))
        d['max_abs_diff_table'] = max(abs(tc['mean'] - rec['cand']), abs(tp['mean'] - rec['comp']))
        d['pass'] = bool(d['max_abs_diff_paired'] <= TOL)
        worst = max(worst, d['max_abs_diff_paired'])
        out[cid] = d
        print(cid, {k_: (round(v, 12) if isinstance(v, float) else v) for k_, v in d.items()}, flush=True)
    res = dict(cells=out, max_abs_diff=worst, tol=TOL, passed=all(v['pass'] for v in out.values()))
    (HERE / 'records/frozen_cstar_gate.json').write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')
    print('GATE', 'PASS' if res['passed'] else 'FAIL', 'max abs diff', worst)
    return res


def cstar_rows(s, seed):
    """Frozen c* float32 rows [n, 2048] on `s` test in ns_photo.keys order (stored float16 cast, not renormalised, the
    form fed to k-means for the photo model). r14 export cstar_qwen, joined by key; mirrors ns_photo.photo_stored."""
    path = NP.R14_OUT / 'exports' / s / 'cstar_qwen' / f'test__s{seed}.npz'
    side = json.loads(path.with_suffix('.json').read_text())
    assert side['sha256'] == rs_vh.file_sha(path), path
    assert side['arm'] == 'cstar_qwen' and side['split'] == 'test' and side['seed'] == seed, side
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'keys', 'emb'}
        fk, emb = z['keys'].tolist(), z['emb']
    assert emb.dtype == np.float16 and emb.shape[1] == 2048 and np.isfinite(emb).all()
    out = ns_photo.join_rows(fk, emb, ns_photo.keys(s, 'test'), str(path), exact=ns_photo._uncut_keys(s, 'test'))
    assert np.abs(np.linalg.norm(out.astype(np.float32), axis=1) - 1).max() < NP.TOL_UNIT
    return out.astype(np.float32), side['checkpoint_sha256']


def run():
    g = json.loads((HERE / 'records/frozen_cstar_gate.json').read_text())
    assert g['passed'], 'gate has not passed'
    res = {'gate_max_abs_diff': g['max_abs_diff']}
    for s, k in (('affection', 9), ('emoset', 8)):
        y = ns_photo.labels(s, 'test')
        parts, ck = [], []
        for seed in NP.SEEDS:
            x, c = cstar_rows(s, seed)
            ck.append(c)
            parts.append(ns_metrics.partitions(x, k, s))
        parts = np.stack(parts)
        row = ns_clu.metric_table('frozen_cstar', 'emotion', parts, y, k, ('ami',))[0]
        res[f'{s}_k{k}'] = dict(per_seed=row['per_seed'], mean=row['mean'], k=k, n_rows=int(len(y)),
                                n_labelled=int((y >= 0).sum()), checkpoint_sha256=ck,
                                source='r14 exports/%s/cstar_qwen/test__s<seed>.npz' % s)
        print(s, k, row['per_seed'], row['mean'], flush=True)
    # Affection k=20 check: same rows, 20 clusters, against r13's recorded 0.13838
    y = ns_photo.labels('affection', 'test')
    parts20 = np.stack([ns_metrics.partitions(cstar_rows('affection', sd)[0], 20, 'affection') for sd in NP.SEEDS])
    r20 = ns_clu.metric_table('frozen_cstar', 'emotion', parts20, y, 20, ('ami',))[0]
    r13 = json.loads(R13_VERDICT.read_text())['rows']['cstar']
    res['affection_k20'] = dict(per_seed=r20['per_seed'], mean=r20['mean'], r13_recorded_ami=r13['ami'],
                                diff=r20['mean'] - r13['ami'])
    print('affection k20', r20['mean'], 'r13', r13['ami'], 'diff', r20['mean'] - r13['ami'], flush=True)
    # Retrieval mAP at the validation-picked read-out, from records
    res['map_picked'] = dict(
        affection=dict(value=r13['map_picked']['map'], per_seed=r13['map_picked']['per_seed'], readout=r13['readout'],
                       source=f'{R13_VERDICT}: rows.cstar.map_picked.map'),
        emoset=None)
    d = json.loads(R14_DIAG.read_text())['sections']['emoset']
    res['map_picked']['emoset'] = dict(value=d['map_at_pick']['cstar'], readout=d['readouts']['cstar'],
                                       source=f'{R14_DIAG}: sections.emoset.map_at_pick.cstar')
    (HERE / 'records/frozen_cstar.json').write_text(json.dumps(res, indent=1, sort_keys=True) + '\n')


if __name__ == '__main__':
    {'gate': gate, 'run': run}[sys.argv[1]]()
