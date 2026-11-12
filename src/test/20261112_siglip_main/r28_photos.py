"""r28 job B part 2: Table 7 (photos) cells on SigLIP 1. Scores only; nothing is trained here.

Cells (the protocol of the Qwen3 rows of Table 7, reusing their code by import, nothing rewritten):
  a  photo_siglip1   EmoSet   the photo model (r14 exports `exports/emoset/photo_siglip1`), 8 clusters, read-out picked on EmoSet val
  b  cstar_siglip1   Affection (9 clusters) and EmoSet (8): "paintings only" = the ArtELingo c* SigLIP 1 emotion checkpoints,
                     inferred on val and test by `bs_photo.py export --arm cstar_siglip1 --root <OUT>` (CPU, r14's procedure)
  c  plain_siglip1   EmoSet   the plain SigLIP 1 store, unseeded
  gates (same code, Qwen / recorded arms; the run refuses to call itself checked unless they pass):
     cstar_qwen     Affection, EmoSet   against ept_frozen's records (AMI) and r13 / r14 (mAP at the picked read-out)
     photo_siglip1  EmoSet, plain_siglip1 EmoSet   against r14's `stage3/sheet_emoset.json`

Protocol per cell (arm x set):
  AMI      mean over seeds (five; one for plain) of the mean over three k-means states of AMI at the class count (Affection 9,
           EmoSet 8), on the TEST rows: `ns_metrics.partitions` (r17's mini-batch k-means) and `ns_clu.metric_table`; seeded
           rows are the stored float16 export cast to float32, not renormalised (r17's form); plain rows unit float32.
  read-out picked on the dataset's VALIDATION split only: `bs_emoset_score.readout_doc` (EmoSet) / `pf_stage2.readout`
           (Affection) = `ac_select.arm_table` (r11 cross-fit, guard, 0.001 r tie rule), with r = the seed range of val mAP
           at R0 (EmoSet: the photo model's r_E,mAP from r14's committed stage 2 record, as r14 used for every descriptive
           arm; Affection: the arm's own five-seed range, as r12 for c*).
  mAP      `bs_emoset_score.test_retrieval` = `ac_stats.arm_retrieval` (map fitted on all val rows, query expansion on the test
           gallery) at that pick; mean over seeds.
Joins are by key (`ns_photo.join_rows`, the split's key list), never by position.

CLI (python r28_photos.py -h):  score [--cells a b c ...] [--gates]   -> OUT/cells/<set>__<arm>.json, then `collect` -> records/photos.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R17 = HERE.parent / '20261103_new_scores'
sys.path.insert(0, str(R17))
import ns_paths as NP  # noqa: E402

NP.add_rounds()
import ns_photo  # noqa: E402  (also puts r14's bs_* and r13's pf_* on the path)
import ns_clu  # noqa: E402
import ns_metrics  # noqa: E402
import bs_paths as BP  # noqa: E402
import bs_stores  # noqa: E402
import bs_emoset_score  # noqa: E402
import pf_stage2  # noqa: E402
import ac_data  # noqa: E402
import ac_select  # noqa: E402
import ac_stats  # noqa: E402
import rt_metrics  # noqa: E402
import rt_readout  # noqa: E402
import rs_vh  # noqa: E402

SEEDS = tuple(NP.SEEDS)
K_CLASS = {'affection': 9, 'emoset': 8}
OUT = Path('/data/SSD2/cosir_r28/20261112_siglip_main/photos')
RECORD = HERE / 'records/photos.json'
R14_EXPORTS = BP.OUT / 'exports'
R14_STAGE2 = BP.RECORDS / 'stage2_emoset.json'
R14_SHEET = BP.OUT / 'stage3/sheet_emoset.json'
R13_VERDICT = BP.R13_OUT / 'stage3/verdict.json'
R12_RANGES = BP.R12_OUT / 'stage0b/ranges.json'
EPT_RECORD = HERE.parent / '20261111_encoder_photo_tables/records/frozen_cstar.json'
TOL_GATE = 1e-9
TIE_TOL_NOTE = 'tie rule and guard are ac_select.pick_readout (0.001 r)'

# arm name -> (kind, encoder view for plain, export root)
ARMS = {
    'photo_siglip1': dict(seeded=True, root=R14_EXPORTS),
    'cstar_siglip1': dict(seeded=True, root=OUT / 'exports'),
    'plain_siglip1': dict(seeded=False, view='siglip1'),
    'cstar_qwen': dict(seeded=True, root=R14_EXPORTS),
}
# the cells asked for (a, b, c) and the gate cells
CELLS = {
    'a': ('emoset', 'photo_siglip1'),
    'b_aff': ('affection', 'cstar_siglip1'),
    'b_emo': ('emoset', 'cstar_siglip1'),
    'c': ('emoset', 'plain_siglip1'),
}
GATES = {
    'g_aff_cstar_qwen': ('affection', 'cstar_qwen'),
    'g_emo_cstar_qwen': ('emoset', 'cstar_qwen'),
}


# ------------------------------------------------------------------------------------------------ rows
def stored(set_name, split, arm, seed):
    """Checked float16 export rows [n, 2048] of a seeded arm, joined by key to `ns_photo.keys(set, split)`.
    Asserts: sidecar sha256 equals the file's, arm / split / seed in the sidecar, npz holds exactly keys and emb, float16
    finite width 2048, key set equals the split's uncut key set, unit norm within NP.TOL_UNIT."""
    spec = ARMS[arm]
    path = Path(spec['root']) / set_name / arm / f'{split}__s{seed}.npz'
    side = json.loads(path.with_suffix('.json').read_text())
    assert side['sha256'] == rs_vh.file_sha(path), f'{path}: file differs from its sidecar'
    assert side['arm'] == arm and side['split'] == split and side['seed'] == seed and side['dataset'] == set_name, (path, side)
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'keys', 'emb'}, (path, z.files)
        file_keys, emb = z['keys'].tolist(), z['emb']
    assert emb.dtype == np.float16 and emb.ndim == 2 and emb.shape[1] == NP.CSTAR_WIDTH and np.isfinite(emb).all(), (path, emb.shape)
    out = ns_photo.join_rows(file_keys, emb, ns_photo.keys(set_name, split), str(path),
                             exact=ns_photo._uncut_keys(set_name, split))
    assert np.abs(np.linalg.norm(out.astype(np.float32), axis=1) - 1).max() < NP.TOL_UNIT, f'{path}: rows are not unit'
    return out


def plain_unit(set_name, split, view):
    """Unit float32 default-image rows in `ns_photo.keys` order (r12 / r14's `_unit`), joined by key."""
    x = ac_data._unit(bs_stores.load(set_name, view, ns_photo.keys(set_name, split)))
    assert x.shape[0] == NP.N_ROWS[(set_name, split)] and x.dtype == np.float32
    return x


def rows(set_name, split, arm):
    """({seed or None: unit float32 rows}, {seed or None: raw float32 rows for k-means}) of an arm on a split."""
    spec = ARMS[arm]
    if spec['seeded']:
        st = {s: stored(set_name, split, arm, s) for s in SEEDS}
        return {s: ac_data._unit(v) for s, v in st.items()}, {s: v.astype(np.float32) for s, v in st.items()}
    x = plain_unit(set_name, split, spec['view'])
    return {None: x}, {None: x}


# ------------------------------------------------------------------------------------------------ scoring
def r_map_for(set_name, xs_val, labels_val, workers):
    """(r, source). EmoSet: r14's committed stage 2 record, `ranges.map.r` (the photo model's r_E,mAP; r14 used it for every
    descriptive arm). Affection: the arm's own five-seed validation range of mAP at R0 (r12's definition for c*), needs
    a seeded arm."""
    if set_name == 'emoset':
        rec = json.loads(R14_STAGE2.read_text())
        return float(rec['ranges']['map']['r']), f'{R14_STAGE2}: ranges.map.r (r_E,mAP, photo_qwen)'
    assert None not in xs_val, 'Affection r needs a seeded arm'
    per = []
    for s in SEEDS:
        res = rt_metrics.retrieval(xs_val[s], labels_val, qe_k=0, workers=workers)
        assert res['dropped'] == 0 and len(res['rows']) == len(labels_val)
        per.append(float(res['ap'].mean()))
    r = ac_stats.seed_range(per)
    assert np.isfinite(r) and r > 0
    return float(r), f'own five-seed val mAP range at R0 (per seed {per})'


def pick_val(set_name, xs_val, labels_val, r, workers):
    """The read-out picked on validation: {'pick', 'scores'} (`ac_select.arm_table` through r13's / r14's wrappers)."""
    if set_name == 'emoset':
        keys = ns_photo.keys('emoset', 'val')
        bs_emoset_score.check_order(keys)
        return bs_emoset_score.readout_doc(xs_val, labels_val, r, workers)
    assert set(xs_val) == set(SEEDS), 'Affection arms here are seeded'
    return pf_stage2.readout(xs_val, r, labels=labels_val, keys=ns_photo.keys('affection', 'val'), workers=workers)


def pick_under(table, r):
    """The pick of a recorded table under another r (pure re-application of r12's rule: guard, 0.001 r tie, tie order)."""
    rows_ = {ac_select.readout_from_name(n): {'emotion': v['map']['emotion']} for n, v in table.items()}
    return rt_readout.name(ac_select.pick_readout(rows_, {'emotion': r}))


def score_cell(set_name, arm, workers=8, log=print):
    """One arm on one set: validation pick, then test AMI at the class count and test mAP at the pick."""
    t0 = time.time()
    k = K_CLASS[set_name]
    spec = ARMS[arm]
    seeded = spec['seeded']
    y_val, y_test = ns_photo.labels(set_name, 'val'), ns_photo.labels(set_name, 'test')
    xv, _ = rows(set_name, 'val', arm)
    r, r_source = r_map_for(set_name, xv, y_val, workers)
    doc = pick_val(set_name, xv, y_val, r, workers)
    pick = doc['pick']
    log(f'  {set_name}/{arm}: validation pick {pick} (r {r:.6f}) {time.time() - t0:.0f}s', flush=True)
    xt, xt_raw = rows(set_name, 'test', arm)
    seeds = SEEDS if seeded else (None,)
    parts = [ns_metrics.partitions(xt_raw[s], k, set_name) for s in seeds]
    parts = np.stack(parts) if seeded else parts[0]
    ami = ns_clu.metric_table(arm, 'emotion', parts, y_test, k, ('ami',))[0]
    log(f'  {set_name}/{arm}: test AMI k={k} {ami["mean"]:.6f} {time.time() - t0:.0f}s', flush=True)
    ret = bs_emoset_score.test_retrieval(xv, xt, pick, y_test, seeded, workers)
    cell = dict(
        set=set_name, arm=arm, k=k, seeded=seeded, n_val=int(len(y_val)), n_test=int(len(y_test)),
        readout=pick, r_map=r, r_source=r_source,
        pick_other_r={n: pick_under(doc['scores'], rr) for n, rr in (('r_half', r / 2), ('r_double', r * 2))},
        val_scores={n: v['map']['emotion'] for n, v in doc['scores'].items()},
        ami=dict(per_seed=ami['per_seed'], mean=ami['mean']),
        map=dict(per_seed=ret['per_seed'], mean=ret['point']),
        seconds=round(time.time() - t0, 1), tie_note=TIE_TOL_NOTE)
    return cell


# ------------------------------------------------------------------------------------------------ gates
def recorded(set_name, arm):
    """(AMI, mAP, pick) recorded by an earlier round for a gate cell, read from its files."""
    if arm == 'cstar_qwen':
        e = json.loads(EPT_RECORD.read_text())
        v = json.loads(R13_VERDICT.read_text())['rows']['cstar']
        sheet = {r['arm']: r for r in json.loads(R14_SHEET.read_text())['rows']}
        if set_name == 'affection':
            return e['affection_k9']['mean'], v['map_picked']['map'], v['readout']
        row = sheet['cstar_qwen']
        return e['emoset_k8']['mean'], row['map_picked'], row['readout']
    sheet = {r['arm']: r for r in json.loads(R14_SHEET.read_text())['rows']}
    row = sheet[{'plain_siglip1': 'frozen_siglip1'}.get(arm, arm)]      # r14's sheet calls the plain arm frozen_<enc>
    assert set_name == 'emoset'
    return row['ami_k8'], row['map_picked'], row['readout']


def gate(cell):
    ami, mp, pick = recorded(cell['set'], cell['arm'])
    d = dict(ami_recorded=ami, ami_here=cell['ami']['mean'], map_recorded=mp, map_here=cell['map']['mean'],
             pick_recorded=pick, pick_here=cell['readout'])
    d['max_abs_diff'] = max(abs(ami - cell['ami']['mean']), abs(mp - cell['map']['mean']))
    d['pass'] = bool(d['max_abs_diff'] <= TOL_GATE and pick == cell['readout'])
    return d


# ------------------------------------------------------------------------------------------------ CLI
def cell_path(set_name, arm):
    return OUT / 'cells' / f'{set_name}__{arm}.json'


def run_cells(names, workers=8):
    table = {**CELLS, **GATES}
    for name in names:
        set_name, arm = table[name]
        print(f'[{name}] {set_name} / {arm}', flush=True)
        cell = score_cell(set_name, arm, workers, log=print)
        if arm in ('photo_siglip1', 'plain_siglip1') or name in GATES:
            cell['gate'] = gate(cell)
            print('  gate', cell['gate'], flush=True)
        p = cell_path(set_name, arm)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cell, indent=1, sort_keys=True) + '\n')


def collect():
    """records/photos.json from the cell files (a missing cell is an error)."""
    out = {}
    for name, (set_name, arm) in {**CELLS, **GATES}.items():
        p = cell_path(set_name, arm)
        assert p.exists(), f'missing {p}'
        out[name] = json.loads(p.read_text())
    gates = {n: c['gate'] for n, c in out.items() if 'gate' in c}
    assert gates and all(g['pass'] for g in gates.values()), f'gates: {gates}'
    RECORD.parent.mkdir(exist_ok=True)
    RECORD.write_text(json.dumps(dict(cells=out, out_dir=str(OUT)), indent=1, sort_keys=True) + '\n')
    for n, c in out.items():
        print(n, c['set'], c['arm'], 'pick', c['readout'], 'AMI %.4f' % c['ami']['mean'], 'mAP %.4f' % c['map']['mean'])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('score')
    s.add_argument('--cells', nargs='+', default=None, choices=list(CELLS) + list(GATES))
    s.add_argument('--workers', type=int, default=8)
    sub.add_parser('collect')
    a = ap.parse_args(argv)
    if a.cmd == 'score':
        run_cells(a.cells or list(CELLS) + list(GATES), a.workers)
    else:
        collect()


if __name__ == '__main__':
    main()
