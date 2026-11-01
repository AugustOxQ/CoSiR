"""Stage 1 of r13: the sweep score, row 9, the pick (D9), the sheet, and gate (ii)'s sweep part.

Spec 2026-10-31-photo-finetune-design.md §3.4 (sweep scores), §3.5 (grid, row 9, changed settings), §6 (stage 1
selection rule), D5, D6, D9, D13. Rows are the sheet's rows (index, "Sheet"): dicts with `settings` (a GRID row),
`status` (`ok` | `failed` | `skipped` | `reference`) and `score` (None unless `ok` or `reference`).
"""
import json
import time
from pathlib import Path

import numpy as np

import pf_paths as P  # puts r12's round (and every older one) on sys.path
import ac_cluster
import ac_data
import ac_select
import rt_readout

ROW1 = P.GRID[1]
OK, FAILED, SKIPPED, REFERENCE = 'ok', 'failed', 'skipped', 'reference'


# ---- the score (spec §3.4) -----------------------------------------------------------------------------------------

def gaps(ami, map_):
    """The sweep score from an AMI and an mAP: g_clu = (AMI - teacher AMI) / r_clu, g_map = (mAP - teacher mAP) /
    r_mAP, score = min(g_clu, g_map). The teacher values and the ranges are r12's (pf_paths). Unrounded."""
    g_clu = (ami - P.TEACHER_AMI) / P.R12_R_CLU
    g_map = (map_ - P.TEACHER_MAP) / P.R12_R_MAP
    return {'ami': float(ami), 'map': float(map_), 'g_clu': float(g_clu), 'g_map': float(g_map),
            'score': float(min(g_clu, g_map))}


def unit_rows(x_stored):
    """Retrieval input from a stored float16 export: float32, L2-normalized (`ac_data._unit`, as `ac_data.embed`)."""
    return ac_data._unit(x_stored)


def score(x_stored, labels=None, workers=8):
    """One run's sweep score on the validation photos. `x_stored`: the run's float16 validation export as stored,
    rows in `ac_data.keys('val')` order. AMI: r10's k = 20 partitions of the float16 rows as stored (`ac_cluster`),
    mean over three states, against `labels` (default `ac_data.labels('val')`). mAP: r12's cross-fit (r11's
    `rt_readout.crossfit`) of the fixed read-out centering plus k = 10 on float32 unit rows, emotion only; one run
    goes in as seed None, because `crossfit_all` accepts only the seed sets [None] and [0..4] (code map C3d).
    Returns {ami, map, g_clu, g_map, score}."""
    x_stored = np.asarray(x_stored)
    assert x_stored.dtype == np.float16 and x_stored.ndim == 2, (x_stored.dtype, x_stored.shape)
    labels = ac_data.labels('val') if labels is None else np.asarray(labels)
    assert labels.shape == (len(x_stored),) and labels.dtype == np.int64, (labels.shape, labels.dtype)
    ami = ac_cluster.ami(ac_cluster.partitions(x_stored), labels)
    ro = ac_select.readout_by_name(P.SWEEP_READOUT, x_stored.shape[1])
    map_ = rt_readout.crossfit({('emotion', None): unit_rows(x_stored)}, {'emotion': labels}, ro, workers=workers)['emotion']
    return gaps(ami, map_)


# ---- changed settings and row 9 (spec §3.5, D9) --------------------------------------------------------------------

def changed(settings):
    """D9: how many of the five grid columns differ from row 1's value. `settings` must have exactly the columns."""
    assert set(settings) == set(P.COLUMNS), sorted(settings)
    return sum(settings[c] != ROW1[c] for c in P.COLUMNS)


def _scored(rows, n):
    return n in rows and rows[n]['status'] == OK


def _require_row1(rows):
    if not _scored(rows, 1):
        raise RuntimeError('sweep row 1 failed or is missing: stop (spec §3.4)')


def row9(rows):
    """Row 9's settings (spec §3.5): for each column, the value of the highest-scoring row among 2 to 7 that sets it
    to a value other than row 1's, if that score is strictly above row 1's (equal scores: the lower row number);
    otherwise row 1's value. Failed rows are out. Returns (settings, skipped, equals): `skipped` when the settings
    equal an earlier (scored) row's, `equals` the lowest such row number or None. Row 1 failed: RuntimeError."""
    _require_row1(rows)
    base = rows[1]['settings']
    out = dict(base)
    for c in P.COLUMNS:
        cands = [n for n in P.ROW9_SOURCES if _scored(rows, n) and rows[n]['settings'][c] != base[c]
                 and rows[n]['score'] > rows[1]['score']]
        if cands:
            best = min(cands, key=lambda n: (-rows[n]['score'], n))
            out[c] = rows[best]['settings'][c]
    equal = [n for n in range(1, 9) if _scored(rows, n) and rows[n]['settings'] == out]
    return out, bool(equal), (equal[0] if equal else None)


# ---- the pick (spec §6 stage 1, D9) --------------------------------------------------------------------------------

def pick(rows):
    """Stage 1's pick: among the scored rows 1 to 9 (row 0, failed and skipped rows are never candidates), those with
    score >= best - TIE_MARGIN (inclusive), then the fewest changed settings, then the lowest row number.
    Returns {row, settings, best_score, tied_rows}: `best_score` is the highest score, `tied_rows` the sorted rows
    within the margin. Row 1 failed: RuntimeError."""
    _require_row1(rows)
    cands = {n: r for n, r in rows.items() if n != 0 and r['status'] == OK}
    best = max(r['score'] for r in cands.values())
    tied = sorted(n for n, r in cands.items() if r['score'] >= best - P.TIE_MARGIN)
    row = min(tied, key=lambda n: (changed(cands[n]['settings']), n))
    return {'row': row, 'settings': dict(cands[row]['settings']), 'best_score': best, 'tied_rows': tied}


# ---- the sheet (index, "Sheet") ------------------------------------------------------------------------------------

SCORE_FIELDS = ('ami', 'map', 'g_clu', 'g_map', 'score')


def entry(settings, status, scores=None):
    """One row of the sweep as the pick and row 9 read it: settings, status and the five score fields (`score`'s
    output; None for a failed or skipped row). Row 0 (frozen c*, status `reference`) has settings None."""
    assert status in (OK, FAILED, SKIPPED, REFERENCE), status
    if status in (OK, REFERENCE):
        assert scores is not None and set(scores) == set(SCORE_FIELDS), scores
        assert status == REFERENCE or settings is not None
    else:
        assert scores is None, 'a failed or skipped row carries no scores'
    scores = scores or dict.fromkeys(SCORE_FIELDS)
    return {'settings': None if settings is None else dict(settings), 'status': status,
            **{f: scores[f] for f in SCORE_FIELDS}}


def sheet(rows):
    """The stage 1 document (`stage1_sheet.json`): {rows: {"0".."9": {settings, changed, status, ami, map, g_clu, g_map,
    score}}, row9: {settings, skipped, equals}, pick: {row, settings, best_score, tied_rows}}.
    `rows`: {n: entry} for row 0 (reference) and rows 1 to 8, plus row 9 when it ran (scored or failed). Row 9's
    settings come from `row9` and must match the entry given; when `row9` says skipped there must be no row 9 entry
    (the sheet records it as `skipped`, scores null). Row 1 failed: RuntimeError."""
    assert rows[0]['status'] == REFERENCE and all(n in rows for n in range(1, 9)), sorted(rows)
    settings9, skipped, equals = row9(rows)
    if skipped:
        assert 9 not in rows, 'row 9 equals an earlier row and must not be run'
        rows = {**rows, 9: entry(settings9, SKIPPED)}
    else:
        assert 9 in rows and rows[9]['status'] in (OK, FAILED) and rows[9]['settings'] == settings9, \
            'row 9 must be run with the settings of the row 9 rule'
    doc_rows = {str(n): {'settings': r['settings'], 'changed': None if n == 0 else changed(r['settings']),
                         'status': r['status'], **{f: r[f] for f in SCORE_FIELDS}}
                for n, r in sorted(rows.items())}
    return {'rows': doc_rows, 'row9': {'settings': settings9, 'skipped': skipped, 'equals': equals},
            'pick': pick(rows)}


# ---- gate (ii), sweep part (spec §3.5 0a(ii)) ----------------------------------------------------------------------

def replay_check(per_seed):
    """Compare the sweep score path on frozen c*'s export to r12's numbers. `per_seed`: {seed: score dict (ami, map)}.
    Seed 0's AMI against FROZEN_AMI_S0; with all five seeds, the mean mAP against CSTAR_MAP_5SEED; each within
    TOL['ii']. `ok` needs all five seeds (`complete`)."""
    tol = P.TOL['ii']
    complete = sorted(per_seed) == list(P.SEEDS)
    ami_diff = per_seed[0]['ami'] - P.FROZEN_AMI_S0
    map_mean = float(np.mean([per_seed[s]['map'] for s in P.SEEDS])) if complete else None
    map_diff = None if map_mean is None else map_mean - P.CSTAR_MAP_5SEED
    return {'complete': complete, 'ami_seed0': per_seed[0]['ami'], 'ami_ref': P.FROZEN_AMI_S0, 'ami_diff': ami_diff,
            'map_seed0': per_seed[0]['map'], 'map_5seed_mean': map_mean, 'map_ref': P.CSTAR_MAP_5SEED,
            'map_diff': map_diff, 'tol': tol,
            'ok': bool(complete and abs(ami_diff) <= tol and abs(map_diff) <= tol)}


def gate_ii_sweep(seeds=P.SEEDS, out=None, workers=8):
    """Gate (ii), the sweep-score part: frozen c*'s r12 validation exports (`ac_data.stored('cstar', 'val', s)`) go
    through `score`; seed 0's AMI must equal FROZEN_AMI_S0 and the mean mAP of the five seeds CSTAR_MAP_5SEED, each
    within 1e-9. Writes `OUTPUTS/stage0a/sweep_replay.json` (or `out`) and returns the record. `seeds` may be cut
    to a subset for a plumbing check (the record is then incomplete and not ok)."""
    t0 = time.time()
    per_seed = {s: score(ac_data.stored('cstar', 'val', s), workers=workers) for s in seeds}
    res = replay_check(per_seed)
    res.update(per_seed={str(s): v for s, v in per_seed.items()}, seconds=time.time() - t0)
    out = Path(out) if out else P.OUTPUTS / 'stage0a' / 'sweep_replay.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    return res
