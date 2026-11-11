"""r21 scoring (ticket 05): gate 0a, the validation scorer, the stage 1 number sheet and picks.

CLI (from the worktree root; Python of the CoSiR env):
  python src/test/20261111_r21_dec_heads/r21_score.py gate0a [--aspects emotion style genre] [--seeds 0 1 2 3 4]
      Test-set regression: saved c* seeds and the teacher, AMI at the class count (r17's procedure, `ns_data` rows and
      labels, `ns_metrics.partitions`, `ns_clu.metric_table`) against r17's cells file (6 decimals); c* mAP per seed
      (r20's `r20_ret.load_export` / `score_rows`) against r20's `ablation_ret.json`. Full run only when both defaults
      are kept; a subset is written as `gate0a_partial.json` and never says PASS.
  python .../r21_score.py val --arm ARM --seed S
      Validation AMI at the class count and image-to-image label mAP per aspect, one arm and seed (ARM `cstar` is
      the saved c*, r10's validation exports; any r21 arm name reads R21/train/<arm>/R6-5). For `cstar` the AMI is
      checked against r17's `records/stage1_ranges.json` to 6 decimals.
  python .../r21_score.py sheet [--seed 0]
      Stage 1 number sheet and picks over every arm in R21/train: `scores/stage1_sheet.{json,md}`,
      `scores/stage1_picks.json`.

Held-data rule: `val` and `sheet` read validation exports only (`check_dev_path`); the test set is read only by
`gate0a`, which is the scoring regression of §3.5 and shapes no choice.

Row order and labels: ArtELingo sorted painting names (`rt_data.names`), labels `ns_data.labels` (r11's `rt_data.labels`,
identical to r20's `ct_held.painting_labels` on the held list, checked). AMI reads the stored float16 rows cast to float32
and not renormalised (r17's clustering form, `ns_data.cstar(unit=False)`); mAP reads unit rows (r11's retrieval form).
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEST = HERE.parent
sys.path.insert(0, str(HERE))
import r21_arms  # noqa: E402

ASPECTS = ('emotion', 'style', 'genre')
SEEDS = (0, 1, 2, 3, 4)
SET = 'artelingo'
GRAPH = dict(emotion='Gres', style='Fres05', genre='R0o')          # the validation export file names (rt_data.GRAPHS)
N_VAL = 6152
TIE_BAND = 0.002                                                   # spec §3.5, selection rule
TOL_AMI = 5e-7                                                     # "equal to 6 decimals"
TOL_MAP = 1e-9
TOL_0C_AMI = 0.004
TOL_0C_EPOCHS = 2
R21 = Path('/data/SSD2/cosir_r21/20261111_dec_heads')
R10_FULL = Path('/data/SSD2/cosir_r10/20261027_recipe_sweep/outputs/full')
SCORES = R21 / 'scores'
R17_CLU = Path('/data/SSD2/cosir_r17/20261103_new_scores/outputs/stage2/cells/clu__artelingo__qwen.json')
STAGE1_RANGES = TEST / '20261103_new_scores/records/stage1_ranges.json'
ABLATION_RET = TEST / '20261109_r20_figures/ret_full/ablation_ret.json'
CSTAR = 'cstar'                                                    # the saved c*: r10's `k11=40` tree


def _deps():
    """The frozen older rounds' code, imported on first use (an import takes seconds; the pure parts need none)."""
    global ns_paths, ns_data, ns_metrics, ns_clu, rt_data, r20_ret
    if 'ns_data' in globals():
        return
    sys.path.insert(0, str(TEST / '20261103_new_scores'))
    sys.path.insert(0, str(TEST / '20261109_r20_figures'))
    import ns_paths, ns_data, ns_metrics, ns_clu  # noqa: E401,F811
    ns_paths.add_rounds()
    import rt_data, r20_ret  # noqa: E401,F811


# ================================================================================================ selection (§3.5)
def _mean(d):
    assert set(d) == set(ASPECTS), sorted(d)
    v = [float(d[a]) for a in ASPECTS]
    assert all(math.isfinite(x) for x in v), d
    return float(np.mean(v))


def select(rows, ref_ami):
    """The selection rule over one form's candidate rows (rows: {arm, ami{aspect}, map{aspect}}; ref_ami: c* seed 0
    validation AMI per aspect). Score = mean over the three aspects of (validation AMI - c* seed 0's). Candidates within
    `TIE_BAND` of the best score (measured from the best, edge included) are tied; ties go to the higher mean validation
    mAP, then to the smaller lambda. Beyond the spec (flagged `beyond_spec`): still tied after that, the smaller W, then
    balance off, then the arm name. Returns {pick, band, candidates (all, best first), beyond_spec}."""
    assert rows, 'no candidate'
    assert len({r['arm'] for r in rows}) == len(rows), 'duplicate arm'
    cand = []
    for r in rows:
        s = r21_arms.parse_arm(r['arm'])
        assert s['form'] != 'off', 'dec-off is not a candidate'
        cand.append(dict(arm=r['arm'], lam=s['lam'], warmup=s['warmup'], balance=s['balance'],
                         mean_d_ami=_mean({a: r['ami'][a] - ref_ami[a] for a in ASPECTS}), mean_map=_mean(r['map'])))
    cand.sort(key=lambda c: -c['mean_d_ami'])
    best = cand[0]['mean_d_ami']
    band = [c for c in cand if best - c['mean_d_ami'] <= TIE_BAND + 1e-12]
    band.sort(key=lambda c: (-c['mean_map'], c['lam']))
    top = [c for c in band if (c['mean_map'], c['lam']) == (band[0]['mean_map'], band[0]['lam'])]
    beyond = len(top) > 1
    pick = min(top, key=lambda c: (c['warmup'], c['balance'], c['arm'])) if beyond else top[0]
    return dict(pick=pick['arm'], band=[c['arm'] for c in band], candidates=cand, beyond_spec=beyond)


def grid_1a():
    """Stage 1a: the plain form, lambda in (1, 10) x W in (1, 5) x stop rule in (S1, S2): 8 arm names."""
    return [r21_arms.arm_name(dict(form='plain', lam=float(lam), warmup=w, stop=stop, balance=False))
            for lam in (1, 10) for w in (1, 5) for stop in r21_arms.STOPS]


def grid_1b(base):
    """Stage 1b at 1a's pick `base` (settings): the three forms x balance off/on; the plain, balance-off arm is `base`'s."""
    assert base['form'] == 'plain' and base['balance'] is False, base
    return [r21_arms.arm_name(dict(base, form=f, balance=b)) for f in ('plain', 'start', 'teacher') for b in (False, True)]


def pick_stage1a(rows, ref_ami):
    """rows: {arm: sheet row}. Needs all eight 1a runs."""
    names = grid_1a()
    assert set(names) <= set(rows), f'stage 1a missing runs: {sorted(set(names) - set(rows))}'
    return select([rows[n] for n in names], ref_ami)


def pick_stage1b(rows, ref_ami, base_arm):
    """One pick per form between its balance-off and balance-on run at 1a's pick `base_arm`."""
    names = grid_1b(r21_arms.parse_arm(base_arm))
    assert set(names) <= set(rows), f'stage 1b missing runs: {sorted(set(names) - set(rows))}'
    return {f: select([rows[n] for n in names if r21_arms.parse_arm(n)['form'] == f], ref_ami)
            for f in ('plain', 'start', 'teacher')}


# ================================================================================================ validation exports
def check_dev_path(path):
    """Guard of `val` and `sheet` (held-data rule): only R6-5 validation exports and sidecars under a `train` folder; any
    path that names a held or test folder is refused."""
    parts = Path(path).parts
    assert 'train' in parts and 'R6-5' in parts, f'not a validation export path: {path}'
    assert not ({'held', 'test'} & set(parts)), f'a test-set path in a validation read: {path}'
    assert Path(path).suffix in ('.npz', '.json'), path


def _train_root(arm, train_root):
    if train_root is not None:
        return Path(train_root)
    return R10_FULL / 'train' if arm == CSTAR else R21 / 'full' / 'train'


def arm_dir(arm):
    """The arm's folder name in the weights/train trees: saved c* is r10's `k11=40`."""
    if arm == CSTAR:
        return 'k11=40'
    r21_arms.parse_arm(arm)
    return arm


def val_export_path(arm, aspect, seed, train_root=None):
    assert aspect in ASPECTS and seed in SEEDS
    path = _train_root(arm, train_root) / arm_dir(arm) / 'R6-5' / f'{aspect}__{GRAPH[aspect]}__s{seed}.npz'
    check_dev_path(path)
    return path


def val_names():
    _deps()
    names = rt_data.names('val')
    assert len(names) == N_VAL and names == sorted(names)
    return names


def load_val(arm, aspect, seed, train_root=None):
    """(img float16 [6152, 2048], sidecar) of one validation export, rows in sorted validation painting-name order.
    Checked as r11's `rt_data._cstar` checks the saved c*: file keys, shape, dtype, finite, unit norm within 1e-3; the
    sidecar's aspect, seed, graph row, full mode, row count and validation-names hash; the sidecar's file hash."""
    _deps()
    path = val_export_path(arm, aspect, seed, train_root)
    side = json.loads(path.with_suffix('.json').read_text())
    names = val_names()
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'img', 'cap'}, (path, z.files)
        x = z['img']
        assert z['cap'].shape == (N_VAL, 2048) and z['cap'].dtype == np.float16, path
    assert x.shape == (N_VAL, 2048) and x.dtype == np.float16 and np.isfinite(x).all(), (path, x.shape, x.dtype)
    cfg = side['config']
    assert cfg['mode'] == 'full' and cfg['smoke'] is False, path
    assert cfg['aspect'] == aspect and cfg['seed'] == seed and cfg['graph_row'] == GRAPH[aspect], path
    assert cfg['output_rows'] == N_VAL and cfg['names_sha256']['dev'] == rt_data.rs_vh.names_sha(names), path
    assert side['output_sha256'] == rt_data.rs_vh.file_sha(path), f'{path}: file hash differs from its sidecar'
    dev = np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1).max()
    assert dev < 1e-3, f'{path}: rows not unit norm (max deviation {dev})'
    return x, side
