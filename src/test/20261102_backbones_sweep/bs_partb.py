"""r14 ticket 11: the part b sheets (spec §3.2 part b, §3.4 "Part b effects", §6 "Licensed (part b)"; D6, D7, D13).

Analysis only: each row is a one-change run compared, paired, with its reference on the same machine. The sheets select
nothing and name no better setting (D6). CPU only. Every number comes from an inherited function:

  knob sweep   39 rows (`bs_cstar.knob_rows()`) against `ref`, node402, seeds 1 to 3 (`OUT/sweeps/knob/<row>/s<seed>/`).
               AMI: the stored partitions (`partitions.npz`) through r10's dev bootstrap `rs_stage3.joint_bootstrap` (seeds
               1 to 3, 2,000 draws, M with its interval); its point values are checked against `metrics.json`.
               mAP at R0: the stored per-query AP through `paired_ap` (r11's `rt_stats.paired` for three seeds, 10,000 draws).
  ablation     r10's eight removals against r10's c* (`R10_OUT/full`, seeds 0 to 4, one 3090 host), validation and held.
               AMI: partitions by `bs_artclu.validation_ami`'s path (validation) and r10's cached held partitions (held, the
               path of `bs_artclu.replay_0vii_removals`), both through `rs_ablation.held_effect`. mAP: `bs_artret.
               ablation_retrieval` per run (R0, and centering + k = 10 fitted on the run's own validation export), then
               `rt_stats.paired` (10,000 draws) on the held per-query AP. Validation mAP is a point value (no interval).
  photo sweep  14 rows (`bs_photo.SWEEP_ROWS`) against `ref`, node402, seeds 0 to 2 (`OUT/sweeps/photo/<row>/s<seed>/`).
               AMI: one-aspect form of `joint_bootstrap` on the stored partitions. mAP: r13's cross-fit number
               (`metrics.json` `map`) for the point value, the stored per-query AP through `paired_ap` for the interval.

Units: AMI and mAP in the glossary seed ranges (knob sweep, ablation) or r13's photo model ranges
(`R13_ROUND/stage2_records.json`, `ranges`; photo sweep). M = mean over aspects of effect / r (r10's M score, `rs_sweep.
mean_gain`); the photo sweep has one aspect, so its M is the mean of the AMI and mAP gains in ranges.
A row without `metrics.json` in every seed, a row with `failed.json`, a row never started: "not run" (D21), kept.
Outputs under `bs_paths.OUT/stage4/`: `partb_knob.{json,md}`, `partb_ablation.{json,md}`, `partb_photo.{json,md}`,
`partb_knob_curves.json` (ticket 12's figure). CLI: `python bs_partb.py -h`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import bs_paths as P
import bs_artclu as AC
import bs_artret as AR

import rs_paths  # noqa: E402  (already on sys.path through bs_artclu)
import rs_sheet  # noqa: E402
import rs_sweep  # noqa: E402
import rs_stage3  # noqa: E402
import rs_ablation  # noqa: E402
import rs_vh  # noqa: E402
import rg_held  # noqa: E402
import rt_data  # noqa: E402
import rt_metrics  # noqa: E402
import rt_stats  # noqa: E402
import rb_ami  # noqa: E402
import eg_score3  # noqa: E402
import ct_score  # noqa: E402

ASPECTS = tuple(P.ASPECTS)
R_AMI = dict(AC.GLOSSARY_R)                                    # glossary seed ranges (spec §3.4)
R_MAP = {'emotion': 0.0014, 'style': 0.0022, 'genre': 0.0038}
N_CLU, N_RET = 2000, 10000                                     # the inherited draws
NOT_RUN = 'not run'
TOL = 1e-9
REMOVALS = tuple(rs_ablation.REMOVALS)


def stage4_dir():
    d = P.OUT / 'stage4'
    d.mkdir(parents=True, exist_ok=True)
    return d


# ====================================================================================== statistics
def paired_ap(cand, comp, draws=N_RET):
    """r11's `rt_stats.paired` for k seeds: cand, comp float64 [k, n] (rows = seeds in the same order), the same draw order
    (one `default_rng(0)`; per draw I = integers(0, n, n), then S = choice(k seeds, size=k, replace=True), shared by both
    sides). `rt_stats.paired` itself fixes k = 5; for k = 5 this equals it (`selfcheck`). -> d, interval, boot."""
    cand, comp = np.asarray(cand, np.float64), np.asarray(comp, np.float64)
    assert cand.ndim == comp.ndim == 2 and cand.shape == comp.shape and cand.shape[1] > 0, (cand.shape, comp.shape)
    k, n = cand.shape
    cols = np.vstack([cand, comp])
    rng = np.random.default_rng(0)
    boot = np.empty(draws)
    chunk = 250
    for lo in range(0, draws, chunk):
        m = min(chunk, draws - lo)
        counts = np.empty((m, n))
        sel = np.empty((m, k), dtype=np.int64)
        for b in range(m):
            counts[b] = np.bincount(rng.integers(0, n, size=n), minlength=n)
            sel[b] = rng.choice(list(range(k)), size=k, replace=True)
        means = counts @ cols.T / n
        boot[lo:lo + m] = (np.take_along_axis(means[:, :k], sel, axis=1).mean(axis=1)
                           - np.take_along_axis(means[:, k:], sel, axis=1).mean(axis=1))
    return dict(d=float(cand.mean() - comp.mean()), interval=np.percentile(boot, [2.5, 97.5]).tolist(), boot=boot)


def selfcheck(draws=300):
    """`paired_ap` with k = 5 against `rt_stats.paired` on random data (difference must be 0)."""
    rng = np.random.default_rng(1)
    a, b = rng.random((5, 400)), rng.random((5, 400))
    ours, theirs = paired_ap(a, b, draws), rt_stats.paired(a, b, draws)
    diff = max(abs(ours['d'] - theirs['d']), np.abs(ours['boot'] - theirs['boot']).max())
    return dict(max_abs_diff=float(diff), ok=bool(diff <= TOL))


def ami_effect_one(labels, parts_row, parts_ref, seeds, n_draws, workers=8):
    """The one-aspect form of `rs_stage3.joint_bootstrap` (same draws `joint_draws`, `boot_scores`, `paired_gains`, same
    percentiles): row minus reference AMI over three seeds. parts_*: {seed: int [3, n]}."""
    labels = np.asarray(labels)
    seeds = tuple(seeds)
    n = len(labels)
    L = np.flatnonzero(labels >= 0)
    A = {nm: {s: float(rb_ami.score(labels, parts[s], L)) for s in seeds} for nm, parts in (('row', parts_row), ('ref', parts_ref))}
    I, Sb = rs_stage3.joint_draws(n, n_draws, seeds)
    boot = {nm: {} for nm in ('row', 'ref')}
    for s in seeds:
        got = rs_stage3.boot_scores(labels, {'row': parts_row[s], 'ref': parts_ref[s]}, I, workers)
        for nm in boot:
            boot[nm][s] = got[nm]
    e_b = eg_score3.paired_gains(boot['row'], boot['ref'], Sb)
    lo, hi = np.percentile(e_b, [rs_stage3.Z_LOW, rs_stage3.Z_HIGH])
    abar = {nm: float(np.mean(list(A[nm].values()))) for nm in A}
    return dict(d=abar['row'] - abar['ref'], lo=float(lo), hi=float(hi), row_mean=abar['row'], ref_mean=abar['ref'],
                per_seed=A)


def eff(d, lo, hi, r, row_mean=None, ref_mean=None):
    """One effect cell: the difference, its interval, and all three in seed ranges (`r`)."""
    out = dict(d=d, lo=lo, hi=hi, r=r, d_r=d / r, lo_r=None if lo is None else lo / r, hi_r=None if hi is None else hi / r)
    if row_mean is not None:
        out.update(row_mean=row_mean, ref_mean=ref_mean)
    return out


def m_point(d_by_aspect, units):
    """r10's M score without the interval: mean over aspects of effect / r (`rs_sweep.mean_gain`'s formula)."""
    return float(np.mean([d_by_aspect[a] / units[a] for a in d_by_aspect]))


# ====================================================================================== sweep folders
def run_state(kind, row, seeds):
    """('done', None) or ('not run', reason): every seed folder has metrics.json; failed.json or no folder -> not run."""
    base = P.OUT / 'sweeps' / kind / row
    if (base / 'failed.json').is_file():
        try:
            why = json.loads((base / 'failed.json').read_text()).get('status', 'failed')
        except ValueError:
            why = 'failed'
        return NOT_RUN, f'failed.json ({why})'
    if not base.is_dir():
        return NOT_RUN, 'no folder (never started)'
    missing = [s for s in seeds if not (base / f's{s}' / 'metrics.json').is_file()]
    if missing:
        return NOT_RUN, f'no metrics.json for seeds {missing}'
    return 'done', None


def load_seed(kind, row, seed):
    folder = P.OUT / 'sweeps' / kind / row / f's{seed}'
    metrics = json.loads((folder / 'metrics.json').read_text())
    with np.load(folder / 'partitions.npz', allow_pickle=False) as z:
        parts = {k: z[k] for k in z.files}
    with np.load(folder / 'perquery.npz', allow_pickle=False) as z:
        perq = {k: z[k] for k in z.files}
    return metrics, parts, perq


def load_row(kind, row, seeds, names):
    """{seed: (metrics, parts, perquery)}; the partitions' key order is asserted against `names` (joined by key, C6)."""
    out = {}
    for s in seeds:
        m, parts, perq = load_seed(kind, row, s)
        assert [str(x) for x in parts['names']] == list(names), f'{kind}/{row}/s{s}: partition rows are not the validation list'
        out[s] = (m, parts, perq)
    return out


# ====================================================================================== knob sweep
def knob_value(row):
    import bs_cstar
    knob = row.partition('=')[0]
    return knob, bs_cstar.row_config(row)[knob]


def knob_sheet(rows=None, n_clu=N_CLU, n_ret=N_RET, workers=8, seeds=None):
    import bs_cstar
    seeds = tuple(seeds or P.SWEEP_SEEDS_KNOB)
    assert len(seeds) == 3
    rows = list(rows or bs_cstar.knob_rows())
    names, labels = rt_data.names('val'), rt_data.labels('val')
    ref_state, ref_why = run_state('knob', 'ref', seeds)
    ref = load_row('knob', 'ref', seeds, names) if ref_state == 'done' else None
    out = dict(kind='knob', reference='ref', seeds=list(seeds), units=dict(ami=R_AMI, map=R_MAP), draws=dict(clu=n_clu, ret=n_ret),
               reference_state=ref_state, reference_reason=ref_why, rows={})
    if ref:
        out['reference_values'] = {m: {a: float(np.mean([ref[s][0][key][a] for s in seeds])) for a in ASPECTS}
                                   for m, key in (('ami', 'ami'), ('map', 'map_r0'))}
    for row in rows:
        knob, value = knob_value(row)
        rec = dict(row=row, knob=knob, value=value)
        state, why = run_state('knob', row, seeds)
        if ref is None:
            state, why = NOT_RUN, f'reference not run ({ref_why}); row state was {state}'
        rec.update(status=state, reason=why)
        if state == 'done':
            data = load_row('knob', row, seeds, names)
            t0 = time.time()
            for s in seeds:                                    # the stored partitions must give the stored AMI
                for a in ASPECTS:
                    for nm, d in (('row', data), ('ref', ref)):
                        got = float(rb_ami.score(labels[a], d[s][1][a], np.flatnonzero(labels[a] >= 0)))
                        assert abs(got - d[s][0]['ami'][a]) <= TOL, (row, nm, s, a, got, d[s][0]['ami'][a])
            parts = {a: {'row': {s: data[s][1][a] for s in seeds}, 'ref': {s: ref[s][1][a] for s in seeds}} for a in ASPECTS}
            jb = rs_stage3.joint_bootstrap(labels, parts, 'row', 'ref', seeds, n_clu, workers)
            rec['ami'] = {a: eff(jb['effects'][a]['e'], jb['effects'][a]['lo'], jb['effects'][a]['hi'], R_AMI[a],
                                 jb['effects'][a]['Abar']['row'], jb['effects'][a]['Abar']['ref']) for a in ASPECTS}
            rec['M_ami'] = dict(M=jb['M']['M'], lo=jb['M']['lo'], hi=jb['M']['hi'])
            rec['map'] = {}
            for a in ASPECTS:
                rows_a = data[seeds[0]][2][f'{a}__rows']
                for s in seeds:
                    assert np.array_equal(rows_a, data[s][2][f'{a}__rows']) and np.array_equal(rows_a, ref[s][2][f'{a}__rows']), (row, a, s)
                cand = np.stack([data[s][2][f'{a}__ap'] for s in seeds]).astype(np.float64)
                comp = np.stack([ref[s][2][f'{a}__ap'] for s in seeds]).astype(np.float64)
                for nm, arr, d in (('row', cand, data), ('ref', comp, ref)):     # per-query mean = stored mAP
                    for i, s in enumerate(seeds):
                        assert abs(float(arr[i].mean()) - d[s][0]['map_r0'][a]) <= TOL, (row, nm, s, a)
                p = paired_ap(cand, comp, n_ret)
                rec['map'][a] = eff(p['d'], *p['interval'], R_MAP[a], float(cand.mean()), float(comp.mean()))
            rec['M_map'] = dict(M=m_point({a: rec['map'][a]['d'] for a in ASPECTS}, R_MAP), note='point value only')
            print(f'knob {row}: done ({time.time() - t0:.0f} s)', flush=True)
        out['rows'][row] = rec
    return out


def knob_curves(sheet):
    """Ticket 12's figure data: per knob, the points (value, effect in seed ranges with its interval) in value order, the
    reference value (c*'s own, effect 0) included; rows not run are listed with status and no numbers."""
    import bs_cstar
    curves = {}
    for knob in rs_sheet.KNOBS:
        pts = []
        for v in bs_cstar.knob_values(knob):
            row = bs_cstar.row_name(knob, v)
            rec = sheet['rows'].get(row)
            if rec is None:
                continue
            pt = dict(row=row, value=v, status=rec['status'])
            if rec['status'] == 'done':
                pt.update(ami={a: {k: rec['ami'][a][k] for k in ('d_r', 'lo_r', 'hi_r')} for a in ASPECTS},
                          map={a: {k: rec['map'][a][k] for k in ('d_r', 'lo_r', 'hi_r')} for a in ASPECTS},
                          M_ami=rec['M_ami']['M'], M_ami_lo=rec['M_ami']['lo'], M_ami_hi=rec['M_ami']['hi'], M_map=rec['M_map']['M'])
            else:
                pt['reason'] = rec['reason']
            pts.append(pt)
        curves[knob] = dict(star=rs_sheet.G.get(knob) if knob != 'k11' else 40, points=pts)
    return dict(units='seed ranges (AMI 0.0046 / 0.0082 / 0.0128; mAP 0.0014 / 0.0022 / 0.0038)', seeds=sheet['seeds'],
                note='analysis only; no point is a better setting (D6)', curves=curves)


# ====================================================================================== photo sweep
def photo_ranges():
    rec = json.loads((P.R13_ROUND / 'stage2_records.json').read_text())['ranges']
    return dict(clu=float(rec['clu']['r']), map=float(rec['map']['r']))


def photo_sheet(rows=None, n_clu=N_CLU, n_ret=N_RET, workers=8, seeds=None):
    import bs_photo
    import ac_data
    seeds = tuple(seeds or P.SWEEP_SEEDS_PHOTO)
    assert len(seeds) == 3
    rows = list(rows or bs_photo.SWEEP_ROWS)
    keys, labels = list(ac_data.keys('val')), np.asarray(ac_data.labels('val'))
    rng_ = photo_ranges()
    ref_state, ref_why = run_state('photo', 'ref', seeds)
    ref = load_row('photo', 'ref', seeds, keys) if ref_state == 'done' else None
    out = dict(kind='photo', reference='ref', seeds=list(seeds), units=dict(ami=rng_['clu'], map=rng_['map'], source='r13 stage2_records.json ranges (photo model, five final runs)'),
               draws=dict(clu=n_clu, ret=n_ret), reference_state=ref_state, reference_reason=ref_why, rows={})
    if ref:
        out['reference_values'] = dict(ami=float(np.mean([ref[s][0]['ami'] for s in seeds])), map=float(np.mean([ref[s][0]['map'] for s in seeds])))
    for row in rows:
        rec = dict(row=row)
        state, why = run_state('photo', row, seeds)
        if ref is None:
            state, why = NOT_RUN, f'reference not run ({ref_why}); row state was {state}'
        rec.update(status=state, reason=why)
        if state == 'done':
            data = load_row('photo', row, seeds, keys)
            for nm, d in (('row', data), ('ref', ref)):
                for s in seeds:
                    got = float(rb_ami.score(labels, d[s][1]['emotion'], np.flatnonzero(labels >= 0)))
                    assert abs(got - d[s][0]['ami']) <= TOL, (row, nm, s, got, d[s][0]['ami'])
            ami = ami_effect_one(labels, {s: data[s][1]['emotion'] for s in seeds}, {s: ref[s][1]['emotion'] for s in seeds}, seeds, n_clu, workers)
            rec['ami'] = eff(ami['d'], ami['lo'], ami['hi'], rng_['clu'], ami['row_mean'], ami['ref_mean'])
            mask = np.isfinite(data[seeds[0]][2]['ap'])
            for s in seeds:
                assert np.array_equal(mask, np.isfinite(data[s][2]['ap'])) and np.array_equal(mask, np.isfinite(ref[s][2]['ap'])), (row, s)
                assert [str(k) for k in data[s][2]['keys']] == keys
            cand = np.stack([data[s][2]['ap'][mask] for s in seeds]).astype(np.float64)
            comp = np.stack([ref[s][2]['ap'][mask] for s in seeds]).astype(np.float64)
            p = paired_ap(cand, comp, n_ret)
            row_map = float(np.mean([data[s][0]['map'] for s in seeds]))
            ref_map = float(np.mean([ref[s][0]['map'] for s in seeds]))
            rec['map'] = eff(row_map - ref_map, p['interval'][0], p['interval'][1], rng_['map'], row_map, ref_map)
            rec['map']['note'] = 'd: r13 cross-fit mAP difference (metrics.json); interval: per-query bootstrap of the AP difference'
            rec['M'] = dict(M=float(np.mean([rec['ami']['d_r'], rec['map']['d_r']])), note='mean of the AMI and mAP gains in seed ranges; point value')
            print(f'photo {row}: done', flush=True)
        out['rows'][row] = rec
    return out


# ====================================================================================== ablation table
def _val_file(arm, aspect, seed):
    found = sorted((AC.R10_FULL / f'train/{arm}/R6-5').glob(f'{aspect}__*__s{seed}.npz'))
    assert len(found) == 1, f'{arm} {aspect} s{seed}: {len(found)} validation exports'
    return found[0]


def load_val(arm, aspect, seed):
    """r10's validation export of a run (float16 [6152, 2048], r11's sorted painting order), sidecar checked as
    `bs_artclu.replay_0vii_sha` does (output sha256) and the dev list by its names hash."""
    f = _val_file(arm, aspect, seed)
    side = json.loads(f.with_suffix('.json').read_text())
    assert side['output_sha256'] == AC.sha_file(f), f'{f}: output hash differs from its sidecar'
    assert side['config']['names_sha256']['dev'] == rs_vh.names_sha(rt_data.names('val')), f'{f}: dev list differs'
    with np.load(f, allow_pickle=False) as z:
        x = z['img']
    assert x.shape == (6152, 2048) and x.dtype == np.float16
    return x, side['output_sha256']


def load_held(arm, aspect, seed):
    f = AC.R10_FULL / f'held/{arm}/{aspect}__s{seed}.npz'
    x = rg_held.load_held_output(f, arm, seed, rs_paths.HELD_SHA256, aspect=aspect)       # float32, unit rows
    return x, rg_held.held_output_source(f)['output_sha256']


def _unit(x):
    x = x.astype(np.float32)
    return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)


def run_retrieval(arm, aspect, seed, ylab, workers=8):
    """`bs_artret.ablation_retrieval` of one run, cached under `OUT/stage4/partb_cache/` (resume after a timeout)."""
    cache = stage4_dir() / 'partb_cache' / f'ret__{arm}__{aspect}__s{seed}.npz'
    xv16, vsha = load_val(arm, aspect, seed)
    xh, hsha = load_held(arm, aspect, seed)
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as z:
            if str(z['val_sha']) == vsha and str(z['held_sha']) == hsha:
                return {k: z[k] for k in z.files}
    res = AR.ablation_retrieval(_unit(xv16), _unit(xh), ylab['val'], ylab['held'], aspect, workers=workers)
    out = dict(held_r0_ap=res['held']['r0']['ap'], held_ck_ap=res['held']['center_k10']['ap'], held_rows=res['held']['r0']['rows'],
               val_r0=np.float64(res['val']['r0_map']), val_ck=np.float64(res['val']['center_k10_crossfit_map']),
               val_sha=np.asarray(vsha), held_sha=np.asarray(hsha))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache.with_name(cache.stem + '.pending.npz'), **out)
    cache.with_name(cache.stem + '.pending.npz').replace(cache)
    return out


def ablation_sheet(removals=None, aspects=None, n_clu=N_CLU, n_ret=N_RET, workers=8, smoke=False):
    removals = list(removals or REMOVALS)
    aspects = tuple(aspects or ASPECTS)
    assert set(removals) <= set(REMOVALS) and set(aspects) <= set(ASPECTS)
    seeds = tuple(P.SEEDS)
    held_labels = AC.held_setup()[2]
    ylab = {a: dict(val=rt_data.labels('val'), held=rt_data.labels('held')) for a in ASPECTS}
    labels_v = rt_data.labels('val')
    assert rs_vh.names_sha(rt_data.names('held')) == rs_paths.HELD_SHA256
    star = 'k11=40'
    cache_root = P.R10_OUT / 'ablation/cache/held'
    held_json = json.loads((P.R10_OUT / 'ablation/held.json').read_text())

    def val_parts(arm, aspect):
        return {s: {'r20': np.asarray(ct_score.output_partitions(load_val(arm, aspect, s)[0].astype(np.float32), 'k1')['r20'])} for s in seeds}

    def held_parts(arm, aspect):
        return {s: {'r20': AC.read_stored_set(cache_root / arm / f'{aspect}__s{s}__r20.npz', rs_paths.N_HELD, 3,
                                              AC.R10_FULL / f'held/{arm}/{aspect}__s{s}.npz')} for s in seeds}

    ref_val = {a: val_parts(star, a) for a in aspects}
    ref_held = {a: held_parts(star, a) for a in aspects}
    ref_ret = {a: [run_retrieval(star, a, s, ylab[a], workers) for s in seeds] for a in aspects}
    out = dict(kind='ablation', reference='k11=40 (r10 c*)', seeds=list(seeds), aspects=list(aspects), smoke=bool(smoke),
               units=dict(ami=R_AMI, map=R_MAP), draws=dict(clu=n_clu, ret=n_ret), readout_mAP='R0 and centering + k = 10 fitted on each run\'s own validation export',
               rows={})
    for x in removals:
        arm = f'k11=40-no_{x}'
        t0 = time.time()
        rec = dict(removal=x, arm=arm, status='done', val={}, held={})
        try:
            pv = {a: val_parts(arm, a) for a in aspects}
            ph = {a: held_parts(arm, a) for a in aspects}
            rr = {a: [run_retrieval(arm, a, s, ylab[a], workers) for s in seeds] for a in aspects}
        except (AssertionError, FileNotFoundError, KeyError) as err:
            out['rows'][x] = dict(removal=x, arm=arm, status=NOT_RUN, reason=f'{type(err).__name__}: {err}'[:300])
            continue
        for a in aspects:
            v = rs_ablation.held_effect(pv[a], ref_val[a], labels_v[a], n_clu, workers)
            rec['val'][a] = dict(ami=eff(v['effect'], v['lo'], v['hi'], R_AMI[a], v['arm_ami'], v['final_ami']))
            h = rs_ablation.held_effect(ph[a], ref_held[a], held_labels[a], n_clu, workers)
            rec['held'][a] = dict(ami=eff(h['effect'], h['lo'], h['hi'], R_AMI[a], h['arm_ami'], h['final_ami']))
            # r10's own held.json carries these AMI numbers (0(vii)): points always, intervals when the draws are r10's 2,000
            ref_cell = held_json['results'][x]['aspects'][a]
            keys_cmp = ('effect', 'arm_ami', 'final_ami') + (('lo', 'hi') if n_clu == N_CLU else ())
            rec['held'][a]['ami']['vs_r10_held_json_max_abs'] = max(abs(h[k] - ref_cell[k]) for k in keys_cmp)
            rec['held'][a]['ami']['vs_r10_held_json_compared'] = list(keys_cmp)
            rows_a = rr[a][0]['held_rows']
            for r in rr[a] + ref_ret[a]:
                assert np.array_equal(r['held_rows'], rows_a), (arm, a)
            for key, name in (('held_r0_ap', 'map_r0'), ('held_ck_ap', 'map_center_k10')):
                cand = np.stack([r[key] for r in rr[a]]).astype(np.float64)
                comp = np.stack([r[key] for r in ref_ret[a]]).astype(np.float64)
                p = rt_stats.paired(cand, comp, n_ret)
                rec['held'][a][name] = eff(p['d'], *p['interval'], R_MAP[a], float(cand.mean()), float(comp.mean()))
            for key, name in (('val_r0', 'map_r0'), ('val_ck', 'map_center_k10')):
                row_m = float(np.mean([r[key] for r in rr[a]]))
                ref_m = float(np.mean([r[key] for r in ref_ret[a]]))
                rec['val'][a][name] = dict(eff(row_m - ref_m, None, None, R_MAP[a], row_m, ref_m), note='point value; no interval')
        for split in ('val', 'held'):
            rec[f'M_{split}'] = {nm: m_point({a: rec[split][a][nm]['d'] for a in aspects}, R_AMI if nm == 'ami' else R_MAP) for nm in
                                 ('ami', 'map_r0', 'map_center_k10')}
        out['rows'][x] = rec
        print(f'ablation {x}: done ({time.time() - t0:.0f} s)', flush=True)
    return out


# ====================================================================================== tables
def _f(v, n=4):
    return 'n/a' if v is None else f'{v:+.{n}f}'


def _cell(e):
    if e is None:
        return ''
    iv = '' if e['lo'] is None else f" [{e['lo']:+.4f}, {e['hi']:+.4f}]"
    ir = '' if e['lo'] is None else f" [{e['lo_r']:+.2f}, {e['hi_r']:+.2f}]"
    return f"{e['d']:+.4f}{iv} = {e['d_r']:+.2f} r{ir}"


NOTE = ('Analysis only (D6): row minus its same-machine reference, mean over seeds, 95% paired interval, and in seed ranges (r). '
        'Nothing here names a better setting than c* or the photo model.')


def knob_md(s):
    head = ['# Part b: knob sweep (c* against its node402 rerun, validation)', '', NOTE, '',
            f"Seeds {s['seeds']}; draws AMI {s['draws']['clu']}, mAP {s['draws']['ret']}. Reference: {s['reference_state']}.", '']
    cols = ['row', 'status'] + [f'{a} AMI' for a in ASPECTS] + ['M (AMI)'] + [f'{a} mAP@R0' for a in ASPECTS] + ['M (mAP)']
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for row, r in s['rows'].items():
        if r['status'] != 'done':
            lines.append(f"| {row} | {NOT_RUN}: {r['reason']} |" + ' |' * (len(cols) - 2))
            continue
        m = r['M_ami']
        lines.append('| ' + ' | '.join([row, 'done'] + [_cell(r['ami'][a]) for a in ASPECTS]
                                        + [f"{m['M']:+.2f} r [{m['lo']:+.2f}, {m['hi']:+.2f}]"] + [_cell(r['map'][a]) for a in ASPECTS]
                                        + [f"{r['M_map']['M']:+.2f} r"]) + ' |')
    return '\n'.join(head + lines) + '\n'


def photo_md(s):
    head = ['# Part b: photo sweep (photo model against its node402 rerun, Affection validation)', '', NOTE, '',
            f"Seeds {s['seeds']}; ranges AMI {s['units']['ami']:.6f}, mAP {s['units']['map']:.6f} (r13's photo model). Reference: {s['reference_state']}.", '']
    cols = ['row', 'status', 'AMI', 'mAP (r13 read-out)', 'M']
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for row, r in s['rows'].items():
        if r['status'] != 'done':
            lines.append(f"| {row} | {NOT_RUN}: {r['reason']} | | | |")
        else:
            lines.append(f"| {row} | done | {_cell(r['ami'])} | {_cell(r['map'])} | {r['M']['M']:+.2f} r |")
    return '\n'.join(head + lines) + '\n'


def ablation_md(s):
    head = ['# Part b: ablation table (r10 removals against r10 c*, seeds 0 to 4, one 3090 host)', '', NOTE, '',
            'Validation mAP is a point value (no interval). Held mAP and AMI carry intervals.', '']
    out = head
    for split in ('val', 'held'):
        cols = ['removal'] + [f'{a} AMI' for a in s['aspects']] + [f'{a} {m}' for m in ('mAP@R0', 'mAP center+k10') for a in s['aspects']] + ['M AMI']
        out += [f'## {split}', '', '| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
        for x, r in s['rows'].items():
            if r['status'] != 'done':
                out.append(f"| {x} | {NOT_RUN}: {r['reason']} |" + ' |' * (len(cols) - 2))
                continue
            d = r[split]
            out.append('| ' + ' | '.join([x] + [_cell(d[a]['ami']) for a in s['aspects']]
                                          + [_cell(d[a]['map_r0']) for a in s['aspects']] + [_cell(d[a]['map_center_k10']) for a in s['aspects']]
                                          + [f"{r['M_' + split]['ami']:+.2f} r"]) + ' |')
        out.append('')
    return '\n'.join(out)


def write_sheet(name, sheet, md):
    d = stage4_dir()
    (d / f'{name}.json').write_text(json.dumps(sheet, indent=1, allow_nan=False) + '\n')
    if md is not None:
        (d / f'{name}.md').write_text(md)
    return d / f'{name}.json'


# ====================================================================================== CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    common.add_argument('--n-clu', type=int, default=N_CLU, help='AMI bootstrap draws (inherited: 2000)')
    common.add_argument('--n-ret', type=int, default=N_RET, help='mAP bootstrap draws (inherited: 10000)')
    sub = p.add_subparsers(dest='cmd', required=True)
    for nm in ('knob', 'photo'):
        q = sub.add_parser(nm, parents=[common])
        q.add_argument('--rows', nargs='+', default=None, help='only these rows (smoke)')
    a = sub.add_parser('ablation', parents=[common])
    a.add_argument('--smoke', action='store_true', help='explicit smoke: first removal only (or --removals), emotion only; draws as given')
    a.add_argument('--removals', nargs='+', default=None, choices=list(REMOVALS))
    a.add_argument('--aspects', nargs='+', default=None, choices=list(ASPECTS))
    sub.add_parser('selfcheck')
    args = p.parse_args(argv)
    if args.cmd == 'selfcheck':
        res = selfcheck()
        print(json.dumps(res))
        return 0 if res['ok'] else 1
    if args.cmd == 'knob':
        sheet = knob_sheet(args.rows, args.n_clu, args.n_ret, args.workers)
        write_sheet('partb_knob', sheet, knob_md(sheet))
        write_sheet('partb_knob_curves', knob_curves(sheet), None)
    elif args.cmd == 'photo':
        sheet = photo_sheet(args.rows, args.n_clu, args.n_ret, args.workers)
        write_sheet('partb_photo', sheet, photo_md(sheet))
    else:
        removals = args.removals or (list(REMOVALS[:1]) if args.smoke else None)
        aspects = args.aspects or (['emotion'] if args.smoke else None)
        sheet = ablation_sheet(removals, aspects, args.n_clu, args.n_ret, args.workers, smoke=args.smoke)
        write_sheet('partb_ablation' + ('_smoke' if args.smoke else ''), sheet, ablation_md(sheet))
    return 0


if __name__ == '__main__':
    sys.exit(main())
