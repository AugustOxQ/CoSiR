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
      `scores/stage1_picks.json`. Picks only when gate 0c is PASS (else exit non-zero, no picks); each pick level is
      write-once (a different pick exits non-zero unless `--force`, which logs the override).

Held-data rule: `val` and `sheet` read validation exports only (`check_dev_path`); the test set is read only by
`gate0a`, which is the scoring regression of §3.5 and shapes no choice.

Row order and labels: ArtELingo sorted painting names (`rt_data.names`), labels `ns_data.labels` (r11's `rt_data.labels`,
identical to r20's `ct_held.painting_labels` on the held list, checked). AMI reads the stored float16 rows cast to float32
and not renormalised (r17's clustering form, `ns_data.cstar(unit=False)`); mAP reads unit rows (r11's retrieval form).
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
import sys
import time
import zoneinfo
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
EPS = 1e-12                                                        # float slack on the edges that are inclusive
STOP_REASONS = ('changed', 'dec_epochs', 'patience', 'max_epochs')   # r21_train's `dec['stop_reason']`
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
    band = [c for c in cand if best - c['mean_d_ami'] <= TIE_BAND + EPS]
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
    got = (cfg.get('rg') or {}).get('arm')      # the arm name is `config['rg']['arm']` ('name' is the recipe, R6-5); rg_full checks it too
    assert got == arm_dir(arm), f"{path}: sidecar config rg.arm {got!r}, the arm's folder is {arm_dir(arm)!r}"
    assert cfg['aspect'] == aspect and cfg['seed'] == seed and cfg['graph_row'] == GRAPH[aspect], path
    assert cfg['output_rows'] == N_VAL and cfg['names_sha256']['dev'] == rt_data.rs_vh.names_sha(names), path
    assert side['output_sha256'] == rt_data.rs_vh.file_sha(path), f'{path}: file hash differs from its sidecar'
    dev = np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1).max()
    assert dev < 1e-3, f'{path}: rows not unit norm (max deviation {dev})'
    return x, side


# ================================================================================================ validation scores
def largest_share(parts):
    """Largest cluster's share of all rows, per k-means state: its mean over the states and its maximum."""
    parts = np.asarray(parts)
    assert parts.ndim == 2
    shares = [np.bincount(p).max() / len(p) for p in parts]
    return dict(mean=float(np.mean(shares)), max=float(np.max(shares)))


def val_aspect(arm, aspect, seed, train_root=None, workers=8):
    """One head of one arm and seed on the validation split: AMI at the class count (r17's clustering: the stored float16
    rows cast to float32, `ns_metrics.partitions`, all 6,152 rows clustered, unlabelled rows masked in scoring, mean over
    the 3 states) and image-to-image label mAP (r20's read-out: unit float32 rows, plain cosine, query excluded, labelled
    rows only, `rt_metrics.retrieval(qe_k=0)` through `r20_ret.score_rows`)."""
    _deps()
    x16, side = load_val(arm, aspect, seed, train_root)
    k = ns_paths.CLASS_COUNT[(SET, aspect)]
    y = ns_data.labels(SET, 'val', aspect)
    assert len(y) == N_VAL and ns_metrics.class_count(y) == k, (aspect, ns_metrics.class_count(y), k)
    parts = ns_metrics.partitions(x16.astype(np.float32), k, SET)
    states = [ns_metrics.score_partitions('ami', y, parts[i:i + 1]) for i in range(len(parts))]
    ami = ns_metrics.score_partitions('ami', y, parts)
    assert abs(ami - float(np.mean(states))) < 1e-12
    ap = r20_ret.score_rows(r20_ret.unit(x16), y, np.asarray(val_names()), workers)
    return dict(k=int(k), ami=float(ami), ami_states=[float(s) for s in states], map=float(ap['ap'].mean()),
                n_queries=int(len(ap['ap'])), dropped=int(ap['dropped']), n_labelled=int((y >= 0).sum()),
                largest_share=largest_share(parts), input_sha256=side['output_sha256'])


def val_scores(arm, seed, aspects=ASPECTS, train_root=None, workers=8):
    return dict(arm=arm, seed=int(seed), split='val', rows=N_VAL,
                aspects={a: val_aspect(arm, a, seed, train_root, workers) for a in aspects})


def check_cstar_ranges(res):
    """The `val` check of saved c*: AMI per aspect against r17's `records/stage1_ranges.json` (the file `ns_clu_cells.
    stage1_ranges` wrote), to 6 decimals. Returns {aspect: {computed, recorded, diff, ok}}."""
    rec = json.loads(STAGE1_RANGES.read_text())
    out = {}
    for a, r in res['aspects'].items():
        want = rec[f'cstar/artelingo/qwen/{a}/ami']['per_seed'][str(res['seed'])]
        out[a] = dict(computed=r['ami'], recorded=want, diff=r['ami'] - want, ok=abs(r['ami'] - want) < TOL_AMI)
    return out


def code_hashes():
    """SHA-256 of the files whose code computes a validation number (`val_aspect`): this file; r17's `ns_metrics` (k-means
    partitions, AMI) and `ns_data` (labels); r11's `rt_data` (names, labels) and `rt_metrics` (retrieval AP); r20's `r20_ret`
    (unit rows, `score_rows`). Keyed by file name."""
    _deps()
    files = [Path(__file__), Path(ns_metrics.__file__), Path(ns_data.__file__), Path(rt_data.__file__),
             Path(r20_ret.rt_metrics.__file__), Path(r20_ret.__file__)]
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def val_cached(arm, seed, scores_dir=None, train_root=None, workers=8):
    """`val_scores`, read from `scores/val/<arm>__s<seed>.json` when its recorded input hashes equal the current exports' and
    its recorded `code_sha256` equals the scoring code's now (every export is loaded and checked either way); else computed
    and written there with the code hashes."""
    scores_dir = Path(scores_dir or SCORES)
    path = scores_dir / 'val' / f'{arm}__s{seed}.json'
    code = code_hashes()
    if path.exists():
        old = json.loads(path.read_text())
        if old.get('code_sha256') == code and all(
                old['aspects'][a]['input_sha256'] == load_val(arm, a, seed, train_root)[1]['output_sha256'] for a in ASPECTS):
            return old
    res = val_scores(arm, seed, ASPECTS, train_root, workers)
    res['code_sha256'] = code
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')
    return res


# ================================================================================================ stage 1 sheet
def head_dec(arm, aspect, side):
    """(DEC epochs, DEC start epoch, stop reason) of one head from its sidecar's history, in r21_train's format: with DEC on
    (any form but `off`) warm-up entries carry `dec={'phase': 'warmup'}` (epoch W also `start_epoch`), DEC entries
    `dec={'phase': 'dec', 'start_epoch': ...}`, the last entry `dec['stop_reason']`. DEC off and saved c*: no `dec` key
    anywhere, (0, None, None)."""
    hist = side['history']
    if arm == CSTAR or r21_arms.parse_arm(arm)['form'] == 'off':
        assert not any('dec' in h for h in hist), f'{arm}/{aspect}: DEC entries in a DEC-off history'
        return 0, None, None
    where = f'{arm}/{aspect}'
    assert all(isinstance(h.get('dec'), dict) and h['dec'].get('phase') in ('warmup', 'dec') for h in hist), where
    phases = [h['dec']['phase'] for h in hist]
    n_dec = phases.count('dec')
    assert n_dec >= 1 and phases == ['warmup'] * (len(hist) - n_dec) + ['dec'] * n_dec, f'{where}: no DEC epoch, or not warm-up then DEC'
    first = hist[len(hist) - n_dec]
    start = first['dec']['start_epoch']
    assert start == first['epoch'] and all(h['dec']['start_epoch'] == start for h in hist[len(hist) - n_dec:]), where
    if n_dec < len(hist):
        assert hist[len(hist) - n_dec - 1]['dec'].get('start_epoch') == start, f'{where}: warm-up epoch W has another start_epoch'
    reason = hist[-1]['dec'].get('stop_reason')
    assert reason in STOP_REASONS, f'{where}: the last history entry has stop_reason {reason!r}'
    return n_dec, int(start), reason


def make_row(arm, val, sides, record, ref_val):
    """One sheet row. val: `val_scores` of the arm; sides: {aspect: training sidecar}; record: the arm's `run__s<seed>.json`
    (or None); ref_val: `val_scores` of saved c* with the same seed. DEC epochs = history entries with `dec['phase'] ==
    'dec'`; DEC start epoch and stop reason are read from the sidecar's history (`head_dec`). When the run record has
    `heads.<aspect>.stop_reason` / `dec_start_epoch` they must equal the sidecar's; with DEC off (no DEC stop rule) the
    reason is the record's (c*'s: patience or max_epochs) or None. DEC shares are the training paintings' argmax cluster
    shares at the last DEC epoch (not the best epoch); they are None for DEC off and saved c*, and decide no picks."""
    assert val['aspects'].keys() == ref_val['aspects'].keys() == set(ASPECTS) == set(sides)
    v, r = val['aspects'], ref_val['aspects']
    dec = {a: head_dec(arm, a, sides[a]) for a in ASPECTS}
    heads = (record or {}).get('heads') or {}
    stop = {}
    for a in ASPECTS:
        rec_a = heads.get(a) or {}
        n, start, reason = dec[a]
        if arm != CSTAR and r21_arms.parse_arm(arm)['form'] != 'off':
            assert rec_a.get('stop_reason', reason) == reason, (arm, a, 'run record stop_reason', rec_a.get('stop_reason'), reason)
            assert rec_a.get('dec_start_epoch', start) == start, (arm, a, 'run record dec_start_epoch', rec_a.get('dec_start_epoch'), start)
            stop[a] = reason
        else:
            stop[a] = rec_a.get('stop_reason')
    row = dict(arm=arm, settings=None if arm == CSTAR else r21_arms.parse_arm(arm), seed=val['seed'],
               ami={a: v[a]['ami'] for a in ASPECTS}, map={a: v[a]['map'] for a in ASPECTS},
               d_ami={a: v[a]['ami'] - r[a]['ami'] for a in ASPECTS}, d_map={a: v[a]['map'] - r[a]['map'] for a in ASPECTS},
               largest_share={a: v[a]['largest_share']['max'] for a in ASPECTS},
               dec_share_max={a: sides[a]['history'][-1]['dec']['share_max'] if dec[a][0] else None for a in ASPECTS},
               dec_share_min={a: sides[a]['history'][-1]['dec']['share_min'] if dec[a][0] else None for a in ASPECTS},
               best_epoch={a: int(sides[a]['best_epoch']) for a in ASPECTS},
               epochs_run={a: len(sides[a]['history']) for a in ASPECTS},
               dec_epochs={a: dec[a][0] for a in ASPECTS}, dec_start_epoch={a: dec[a][1] for a in ASPECTS}, stop_reason=stop)
    for a in ASPECTS:
        assert sides[a]['epochs_run'] == row['epochs_run'][a], (arm, a)
    row['mean_d_ami'] = _mean(row['d_ami'])
    row['mean_map'] = _mean(row['map'])
    return row


def zero_c_line(off_row, cstar_row):
    """Gate 0c (§3.5): the dec-off rerun's best epochs within `TOL_0C_EPOCHS` of the saved run's, and its validation AMI
    within `TOL_0C_AMI` of the saved run's (both edges inclusive, `EPS` float slack on the AMI one), per aspect (seed 0).
    'pending' while the rerun has not run."""
    if off_row is None:
        return dict(status='pending')
    assert off_row['seed'] == cstar_row['seed'] == 0
    epochs = {a: off_row['best_epoch'][a] - cstar_row['best_epoch'][a] for a in ASPECTS}
    ami = {a: off_row['ami'][a] - cstar_row['ami'][a] for a in ASPECTS}
    ok = all(abs(e) <= TOL_0C_EPOCHS for e in epochs.values()) and all(abs(d) <= TOL_0C_AMI + EPS for d in ami.values())
    return dict(status='PASS' if ok else 'FAIL', best_epoch_diff=epochs, ami_diff=ami, rerun_best_epoch=off_row['best_epoch'],
                saved_best_epoch=cstar_row['best_epoch'], tol_epochs=TOL_0C_EPOCHS, tol_ami=TOL_0C_AMI)


def sheet_markdown(doc):
    f = lambda x: f'{x:.4f}'
    lines = [f"# r21 stage 1 number sheet (validation, seed {doc['seed']})", '',
             'Validation split (6,152 paintings), AMI at the class count (9, 27, 10), image-to-image label mAP. '
             'd = change from saved c* seed 0. Largest share = largest k-means cluster (class count) as a share of all rows. '
             'DEC shares max/min = training argmax cluster shares at the last DEC epoch; diagnostic only, unused in picks.',
             '', '| run | ' + ' | '.join(f'{a} AMI (d) / mAP' for a in ASPECTS) +
             ' | mean d AMI | mean mAP | largest share | DEC shares max/min | best epoch | DEC epochs | stop |',
             '|---|' + '---|' * (len(ASPECTS) + 7)]
    for r in [doc['cstar']] + doc['rows']:
        cells = ' | '.join(f"{f(r['ami'][a])} ({r['d_ami'][a]:+.4f}) / {f(r['map'][a])}" for a in ASPECTS)
        share = '/'.join('%.2f' % r['largest_share'][a] for a in ASPECTS)
        dec_share = ', '.join(f"{r['dec_share_max'][a]:.2f}/{r['dec_share_min'][a]:.2f}"
                              if r['dec_share_max'][a] is not None else '-' for a in ASPECTS)
        best = '/'.join(str(r['best_epoch'][a]) for a in ASPECTS)
        dec = '/'.join(str(r['dec_epochs'][a]) for a in ASPECTS)
        why = '/'.join(str(r['stop_reason'][a] or 'n/a') for a in ASPECTS)
        lines.append(f"| {r['arm']} | {cells} | {r['mean_d_ami']:+.4f} | {f(r['mean_map'])} | {share} | {dec_share} | {best} | {dec} | {why} |")
    lines += ['', 'Columns of the last five: per aspect in the order ' + ', '.join(ASPECTS) + '.', '']
    z = doc['zero_c']
    if z['status'] == 'pending':
        lines.append('Gate 0c: pending (no dec-off run yet).')
    else:
        lines.append(f"Gate 0c: {z['status']}. Best epochs of the rerun {z['rerun_best_epoch']} against saved "
                     f"{z['saved_best_epoch']} (within {z['tol_epochs']}); validation AMI differences "
                     f"{ {a: round(d, 6) for a, d in z['ami_diff'].items()} } (within {z['tol_ami']}).")
    if doc['missing']:
        lines += ['', 'Arms with no complete export set at this seed: ' + ', '.join(doc['missing']) + '.']
    lines.append('')
    p = doc['picks']
    if z['status'] != 'PASS':
        lines.append(f"No picks: gate 0c is {z['status']} (spec 3.5: stop when 0c fails; picks need a PASS).")
    for key in ('stage1a',):
        if p.get(key):
            lines.append(f"Selection rule: highest mean over the aspects of (AMI - c* seed 0 AMI); ties within {TIE_BAND} go to "
                         'the higher mean mAP, then to the smaller lambda.')
            lines.append(f"stage 1a pick: {p[key]['pick']} (tied band: {', '.join(p[key]['band'])}; "
                         f"beyond the rule: {p[key]['beyond_spec']})")
    if p.get('stage1b'):
        for form, sel in p['stage1b'].items():
            lines.append(f"stage 1b pick, {form}: {sel['pick']} (tied band: {', '.join(sel['band'])}; "
                         f"beyond the rule: {sel['beyond_spec']})")
    elif p.get('stage1a'):
        lines.append('stage 1b: not all six runs present yet.')
    return '\n'.join(lines) + '\n'


def check_record(record, arm, seed):
    """The run record `run__s<seed>.json` of folder `arm` is that arm's and that seed's: its arm, settings and seed."""
    assert record['arm'] == arm, f"run record of arm {record['arm']!r} in folder {arm!r}"
    assert record['settings'] == r21_arms.parse_arm(arm), f"{arm}: run record settings {record['settings']} differ from the name's"
    assert record['seed'] == seed, f"{arm}: run record seed {record['seed']}, expected {seed}"


def collect_rows(seed=0, train_root=None, scores_dir=None, workers=8):
    """Every arm folder under R21/full/train with its seed-`seed` validation exports: (cstar row, {arm: row}, missing).
    An arm with some but not all three exports is refused; complete exports require an ok run record."""
    train_root = Path(train_root or R21 / 'full' / 'train')
    ref = val_cached(CSTAR, seed, scores_dir, workers=workers)
    chk = check_cstar_ranges(ref)
    assert all(c['ok'] for c in chk.values()), f'saved c* seed {seed} validation AMI differs from stage1_ranges.json: {chk}'
    cstar = make_row(CSTAR, ref, {a: load_val(CSTAR, a, seed)[1] for a in ASPECTS}, None, ref)
    rows, missing = {}, []
    for d in sorted(p for p in train_root.iterdir() if p.is_dir()):
        arm = d.name
        r21_arms.parse_arm(arm)
        have = [val_export_path(arm, a, seed, train_root).exists() for a in ASPECTS]
        if not any(have):
            missing.append(arm)
            continue
        assert all(have), f'{arm}: validation exports of only some aspects at seed {seed}'
        rec_path = d / f'run__s{seed}.json'
        assert 'train' in rec_path.parts and not ({'held', 'test'} & set(rec_path.parts)), rec_path
        assert rec_path.exists(), f'missing run record: {rec_path}'
        record = json.loads(rec_path.read_text())
        assert record['status'] == 'ok', f"{rec_path}: run status {record['status']!r}, expected 'ok'"
        check_record(record, arm, seed)
        val = val_cached(arm, seed, scores_dir, train_root, workers)
        sides = {a: load_val(arm, a, seed, train_root)[1] for a in ASPECTS}
        rows[arm] = make_row(arm, val, sides, record, ref)
    return cstar, rows, missing


def build_sheet(cstar, rows, missing, seed=0):
    ref_ami = cstar['ami']
    picks = dict(stage1a=None, stage1b=None)
    zero_c = zero_c_line(rows.get('dec-off'), cstar) if seed == 0 else dict(status='pending')
    if zero_c['status'] == 'PASS' and set(grid_1a()) <= set(rows):          # §3.5: no picks unless gate 0c passes
        picks['stage1a'] = pick_stage1a(rows, ref_ami)
        if set(grid_1b(r21_arms.parse_arm(picks['stage1a']['pick']))) <= set(rows):
            picks['stage1b'] = pick_stage1b(rows, ref_ami, picks['stage1a']['pick'])
    order = sorted(rows, key=lambda n: (n != 'dec-off', n))
    return dict(seed=seed, cstar=cstar, rows=[rows[n] for n in order], missing=missing, picks=picks,
                zero_c=zero_c,
                n_settings=len([n for n in rows if n != 'dec-off']), tie_band=TIE_BAND)


def pick_levels(picks):
    """{level: selection} of a picks dict: 'stage1a', and 'stage1b.<form>' for each form picked after 1b."""
    out = {'stage1a': picks['stage1a']} if picks.get('stage1a') else {}
    out.update({f'stage1b.{f}': sel for f, sel in (picks.get('stage1b') or {}).items()})
    return out


def merge_picks(old, new, force=False):
    """Stage 1's picks are frozen once written (spec 3.7), each level (the 1a pick, each form's pick after 1b) on its own.
    Returns (merged, conflicts, overrides). A level absent from `old` is added; one present with the same pick is left as
    it was written; one with another pick is a conflict {level, frozen, new} and, only with `force`, is replaced and logged
    in `merged['overrides']` (kept from `old` and appended to)."""
    merged = copy.deepcopy(old) if old else dict(stage1a=None, stage1b=None)
    merged.setdefault('stage1a', None)
    merged.setdefault('stage1b', None)
    have = pick_levels(merged)
    conflicts, overrides = [], []
    when = datetime.datetime.now(zoneinfo.ZoneInfo('Europe/Amsterdam')).strftime('%Y-%m-%d %H:%M')
    for level, sel in pick_levels(new).items():
        if level in have and have[level]['pick'] == sel['pick']:
            continue
        if level in have:
            conflicts.append(dict(level=level, frozen=have[level]['pick'], new=sel['pick']))
            if not force:
                continue
            overrides.append(dict(time=when, level=level, frozen_pick=have[level]['pick'], new_pick=sel['pick'], reason='--force'))
        if level == 'stage1a':
            merged['stage1a'] = sel
        else:
            merged['stage1b'] = merged['stage1b'] or {}
            merged['stage1b'][level.split('.', 1)[1]] = sel
    if overrides:
        merged['overrides'] = merged.get('overrides', []) + overrides
    return merged, conflicts, overrides


def cmd_sheet(args):
    """Writes the sheet (JSON and markdown) always. Picks need gate 0c = PASS: else no picks are written, the 0c status is
    printed and the exit is non-zero. Picks are write-once per level (`merge_picks`): a different pick exits non-zero
    with both shown and nothing is written, unless `--force`, which logs the override in the picks file."""
    cstar, rows, missing = collect_rows(0, workers=args.workers)
    doc = build_sheet(cstar, rows, missing)
    SCORES.mkdir(parents=True, exist_ok=True)
    picks_path = SCORES / 'stage1_picks.json'
    status = doc['zero_c']['status']
    merged = None
    if status == 'PASS':
        old = json.loads(picks_path.read_text()) if picks_path.exists() else None
        merged, conflicts, overrides = merge_picks(old, doc['picks'], getattr(args, 'force', False))
        if conflicts and not overrides:
            lines = [f"  {c['level']}: frozen {c['frozen']}, new {c['new']}" for c in conflicts]
            raise SystemExit('stage 1 picks are frozen once written (spec 3.7); the new picks differ, nothing written '
                             '(use --force to override and log it):\n' + '\n'.join(lines))
    (SCORES / 'stage1_sheet.json').write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')
    (SCORES / 'stage1_sheet.md').write_text(sheet_markdown(doc))
    print(sheet_markdown(doc))
    if merged is None:
        raise SystemExit(f"gate 0c is {status}: sheet written to {SCORES}/stage1_sheet.json, stage1_sheet.md; no picks "
                         f"(spec 3.5: stop; trace gate 0c first)")
    picks_path.write_text(json.dumps(merged, indent=1, allow_nan=False) + '\n')
    print(f'-> {SCORES}/stage1_sheet.json, stage1_sheet.md, stage1_picks.json')


# ================================================================================================ gate 0a
def gate0a_ami(aspect, seeds):
    """Test-set AMI at the class count of saved c* (`seeds`, held exports) and the teacher: r17's procedure (rows and labels
    from `ns_data`, `ns_metrics.partitions`, `ns_clu.metric_table`). Compared with r17's cells file: the 5-seed c* mean
    (only when all five seeds ran) and the teacher."""
    _deps()
    k = ns_paths.CLASS_COUNT[(SET, aspect)]
    y = ns_data.labels(SET, 'held', aspect)
    assert len(y) == 12281 and ns_metrics.class_count(y) == k and int((y >= 0).sum()) == dict(
        emotion=12281, style=12281, genre=9949)[aspect]
    cells = {c['id']: c for c in json.loads(R17_CLU.read_text())}
    ref = cells[f'SC3/{SET}/qwen/{aspect}/ami/{k}/teacher']
    assert ref['aspect'] == aspect and ref['k'] == k
    parts = np.stack([ns_metrics.partitions(ns_data.cstar(SET, 'held', 'qwen', aspect, s, unit=False), k, SET)
                      for s in seeds])
    row = ns_clu.metric_table('cstar', aspect, parts, y, k, ('ami',))[0]
    tparts = ns_metrics.partitions(ns_data.teacher(SET, 'held', aspect), k, SET)
    trow = ns_clu.metric_table('teacher', aspect, tparts, y, k, ('ami',))[0]
    out = dict(k=int(k), seeds=list(seeds), cstar_per_seed=row['per_seed'], cstar_mean=row['mean'],
               teacher=trow['mean'], r17_cstar_mean=ref['cand'], r17_teacher=ref['comp'], r17_cell=ref['id'],
               teacher_diff=trow['mean'] - ref['comp'], teacher_ok=abs(trow['mean'] - ref['comp']) < TOL_AMI)
    if tuple(seeds) == SEEDS:
        out.update(cstar_diff=row['mean'] - ref['cand'], cstar_ok=abs(row['mean'] - ref['cand']) < TOL_AMI)
    return out


def gate0a_map(aspect, seeds, workers=8):
    """Test-set c* mAP per seed by r20's code (`r20_ret.load_export`, `score_rows`, labels and names from
    `rb_part2_report.held_labels`) against `ablation_ret.json`'s per-seed values."""
    _deps()
    names, sha, labels = r20_ret.rb_part2_report.held_labels()
    names = np.asarray(names)
    assert len(names) == 12281 and r20_ret.hashlib.sha256('\n'.join(names).encode()).hexdigest() == sha
    ref = json.loads(ABLATION_RET.read_text())['aspects'][aspect]['arms']['k11=40']
    assert len(ref['per_seed']) == 5
    got, hashes = [], {}
    for s in seeds:
        side = json.loads(r20_ret.export_path('k11=40', aspect, s).with_suffix('.json').read_text())
        assert side['rows'] == 12281 and side['list_sha256'] == sha and (side['aspect'], side['seed']) == (aspect, s)
        x = r20_ret.load_export('k11=40', aspect, s, 12281, hashes)
        res = r20_ret.score_rows(x, np.asarray(labels[aspect], dtype=np.int64), names, workers)
        got.append(float(res['ap'].mean()))
    want = [ref['per_seed'][s] for s in seeds]
    diff = [g - w for g, w in zip(got, want)]
    return dict(seeds=list(seeds), per_seed=got, r20_per_seed=want, diff=diff, ok=bool(max(abs(d) for d in diff) < TOL_MAP))


def cmd_gate0a(args):
    t = time.monotonic()
    full = tuple(args.seeds) == SEEDS and tuple(args.aspects) == ASPECTS and set(args.parts) == {'ami', 'map'}
    doc = dict(complete=full, aspects={}, tolerances=dict(ami=TOL_AMI, map=TOL_MAP), r17_file=str(R17_CLU),
               r20_file=str(ABLATION_RET))
    oks = []
    for a in args.aspects:
        d = doc['aspects'][a] = {}
        if 'ami' in args.parts:
            d['ami'] = gate0a_ami(a, args.seeds)
            m = d['ami']
            print(f"{a} AMI k={m['k']}: c* per seed {[round(v, 6) for v in m['cstar_per_seed']]} mean {m['cstar_mean']:.6f} "
                  f"(r17 {m['r17_cstar_mean']:.6f}); teacher {m['teacher']:.6f} (r17 {m['r17_teacher']:.6f}) "
                  f"diffs {m.get('cstar_diff', float('nan')):+.1e} {m['teacher_diff']:+.1e}", flush=True)
            oks += [m['teacher_ok']] + ([m['cstar_ok']] if 'cstar_ok' in m else [])
        if 'map' in args.parts:
            d['map'] = gate0a_map(a, args.seeds, args.workers)
            m = d['map']
            print(f"{a} mAP per seed {[round(v, 6) for v in m['per_seed']]} r20 {[round(v, 6) for v in m['r20_per_seed']]} "
                  f"max |diff| {max(abs(x) for x in m['diff']):.1e}", flush=True)
            oks.append(m['ok'])
    doc['checks_ok'] = bool(all(oks))
    doc['pass'] = bool(full and all(oks))
    doc['seconds'] = time.monotonic() - t
    SCORES.mkdir(parents=True, exist_ok=True)
    out = SCORES / ('gate0a.json' if full else 'gate0a_partial.json')
    out.write_text(json.dumps(doc, indent=1, allow_nan=False) + '\n')
    print(f"gate 0a {'PASS' if doc['pass'] else ('FAIL' if not all(oks) else 'partial (checks ok, not the full gate)')}; "
          f"{doc['seconds']:.0f}s -> {out}")
    assert all(oks), 'gate 0a failed: stop, trace, tell the user'


# ================================================================================================ CLI
def cmd_val(args):
    t = time.monotonic()
    arm = args.arm
    assert arm == CSTAR or r21_arms.parse_arm(arm), arm
    res = val_cached(arm, args.seed, workers=args.workers)
    for a, r in res['aspects'].items():
        print(f"{arm} s{args.seed} {a}: AMI={r['ami']:.6f} mAP={r['map']:.6f} (k={r['k']}, queries={r['n_queries']}, "
              f"largest share {r['largest_share']['mean']:.3f})", flush=True)
    if arm == CSTAR:
        chk = check_cstar_ranges(res)
        for a, c in chk.items():
            print(f"check vs stage1_ranges.json {a}: computed {c['computed']:.9f} recorded {c['recorded']:.9f} "
                  f"diff {c['diff']:+.2e} {'ok' if c['ok'] else 'FAIL'}")
        assert all(c['ok'] for c in chk.values()), 'saved c* validation AMI differs from stage1_ranges.json'
    print(f'{time.monotonic() - t:.1f}s -> {SCORES / "val" / f"{arm}__s{args.seed}.json"}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('val')
    p.add_argument('--arm', required=True)
    p.add_argument('--seed', type=int, required=True, choices=SEEDS)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sh = sub.add_parser('sheet')
    sh.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sh.add_argument('--force', action='store_true', help='override frozen stage 1 picks (logged in the picks file)')
    g = sub.add_parser('gate0a')
    g.add_argument('--aspects', nargs='+', default=list(ASPECTS), choices=ASPECTS)
    g.add_argument('--seeds', nargs='+', type=int, default=list(SEEDS), choices=SEEDS)
    g.add_argument('--parts', nargs='+', default=['ami', 'map'], choices=['ami', 'map'])
    g.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    args = ap.parse_args(argv)
    {'val': cmd_val, 'gate0a': cmd_gate0a, 'sheet': cmd_sheet}[args.cmd](args)


if __name__ == '__main__':
    main()
