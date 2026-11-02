"""ArtELingo clustering scores of r14 (ticket 05): SC1, SC3 and SC6 (clustering) for X-c*, X-c*'s validation seed ranges,
and the replays 0(vi) (report 12-28 part), 0(vii) and 0(xii) (F and K parts). CPU only.

Spec: docs/superpowers/specs/2026-11-02-backbones-sweep-design.md, SC1, SC3, SC6, §3.4 ("ArtELingo clustering", "SC3's
statistic", "Seed ranges for LEVEL"), §3.5 0(vi), 0(vii), 0(xii), D16. Nothing here is rewritten from older rounds; each
statistic calls the function that produced the replayed number:

  SC1  X-c* against X's frozen image   `rb_part2_report.margin(..., seedless=(F,))` (10-25's margin "c_frozen"): the
       frozen image has one partition set and draws no seeds (paintings only).
  SC3  X-c* against X's matched arm    `rb_part2_report.margin(..., (K, M))` (10-25's and 10-26's margin "b_matched"):
       one `default_rng(0)`; per draw the paintings once (I_b), then seeds for X-c* (S_b), then seeds for the matched arm
       (S'_b), independent; d = A(X-c*) minus the larger of the matched arm's two readings (own topics; 20 clusters),
       the larger taken again in each draw (`rb_score.score_indices`).
  SC6  X-c* against the teacher        `rg_held.bootstrap_d_t` (r10's `rs_stage4.d_t_block`, report 12-28): I_b, then S_b
       for X-c*; the teacher T_a (10-26 `partitions/full/T`, equal to 10-25's) draws nothing.
  ranges  max - min over seeds 0 to 4 of the validation AMI (20 clusters, mean of three states) per aspect, read through
       r11's validation reader `rt_data._cstar` with only its root changed (index, "X-c* runs"), unrounded.

Held data are read only by `score_encoder` and the replays; `clustering_ranges` reads validation only.

CLI (`python bs_artclu.py -h`): ranges | score | replay vi | replay vii | replay xii. Outputs go under `bs_paths.OUT`
(BS_OUTPUTS for a smoke run).
"""
import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import bs_paths

for _name in ('20261027_recipe_sweep', '20261029_retrieval'):       # this checkout's copies of the inherited code
    if str(bs_paths.TEST / _name) not in sys.path:
        sys.path.insert(0, str(bs_paths.TEST / _name))
import rs_paths  # noqa: E402  (puts r10's, 12-27's, 12-25's, 12-24's and the older rounds' folders on sys.path)
import rb_score  # noqa: E402
import rb_part2_report  # noqa: E402
import rg_held  # noqa: E402
import rs_stage0 as S0  # noqa: E402
import rt_data  # noqa: E402

SEEDS = tuple(bs_paths.SEEDS)                                       # (0, 1, 2, 3, 4)
ASPECTS = tuple(bs_paths.ASPECTS)
N_DRAWS = 2000
TOL = 1e-9                                                          # 0(vi), 0(vii), 0(xii)
GLOSSARY_R = {'emotion': 0.0046, 'style': 0.0082, 'genre': 0.0128}  # glossary "seed range", clustering AMI
CELLS_NAME = 'cells_artelingo_clu.json'
CELL_KEYS = ('sc', 'encoder', 'aspect', 'task', 'data', 'candidate', 'comparator', 'candidate_value',
             'comparator_value', 'd', 'lo', 'hi', 'r', 'r_source', 'n_draws', 'seeds')

R10_FULL = bs_paths.R10_OUT / 'full'
RB_PART2 = bs_paths.RB_ROUND / 'outputs/part2'                      # 10-25 (frozen, main checkout only)
EG_PART2 = bs_paths.EG_ROUND / 'outputs/part2'                      # 10-26

# Per encoder: the matched arm and the frozen image of the older rounds. `qwen` is the stand-in of the smoke (r10's c*,
# K4's matched arm B1 and frozen image F4 of 10-25); it is not an encoder of part a.
ENC = {
    'siglip2': dict(store='siglip2_so400m', root=RB_PART2, cand='K2', matched='M2', frozen='F2'),
    'siglip1': dict(store='siglip_so400m', root=EG_PART2, cand='K8', matched='M8', frozen='F8'),
    'clip_l14': dict(store='clip_l14', root=EG_PART2, cand='K7', matched='M7', frozen='F7'),
    'qwen': dict(store='qwen3vl_emb_2b', root=RB_PART2, cand='K4', matched='B1', frozen='F4'),
}
TEACHER_ROOT = EG_PART2                                             # rule review Q3: 10-26 `partitions/full/T`
GRAPHS = rt_data.GRAPHS                                             # c*'s graph row per aspect (validation file names)


# ============================================================================================ pure statistics
def seed_range(per_seed):
    """max - min over seeds 0 to 4 of a per-seed score, unrounded (spec §3.4 "Seed ranges for LEVEL")."""
    assert sorted(per_seed) == list(SEEDS), f'seeds {sorted(per_seed)}, expected {list(SEEDS)}'
    values = np.array([per_seed[s] for s in SEEDS], dtype=np.float64)
    assert np.isfinite(values).all()
    return float(values.max() - values.min())


def _check_set(fixed, seeds, n_rows, readings, what):
    assert sorted(fixed) == list(seeds), f'{what}: seeds {sorted(fixed)}, expected {list(seeds)}'
    for seed, got in fixed.items():
        assert set(got) == set(readings), (what, seed, sorted(got))
        for reading, p in got.items():
            sets = 1 if reading == 'own' else 3
            assert p.shape == (sets, n_rows) and np.issubdtype(p.dtype, np.integer), (what, seed, reading, p.shape)


def frozen_margin(fixed_c, fixed_f, labels, n_resamples=N_DRAWS, workers=8, labels_c='C', labels_f='F'):
    """SC1: X-c* (seeds 0 to 4) against X's frozen image `fixed_f = {0: {'r20': 3 partitions}}`. The frozen image is
    seedless: every draw resamples the paintings and draws seeds for X-c* only (10-25's "c_frozen")."""
    _check_set(fixed_c, SEEDS, len(labels), ('r20',), 'X-c*')
    _check_set(fixed_f, (0,), len(labels), ('r20',), 'frozen image')
    fixed = {labels_c: fixed_c, labels_f: fixed_f}
    return rb_part2_report.margin(fixed, labels, {labels_c: list(SEEDS)}, (labels_c, labels_f), n_resamples, workers,
                                  seedless=(labels_f,))


def matched_margin(fixed_c, fixed_m, labels, n_resamples=N_DRAWS, workers=8, labels_c='C', labels_m='M'):
    """SC3: 10-26's `b_matched`: d = A(X-c*, r20) minus the larger of the matched arm's r20 and own readings, each a
    mean over its seeds 0 to 4; independent seed draws per arm; the larger reading taken again in each draw."""
    _check_set(fixed_c, SEEDS, len(labels), ('r20',), 'X-c*')
    _check_set(fixed_m, SEEDS, len(labels), ('r20', 'own'), 'matched arm')
    fixed = {labels_c: fixed_c, labels_m: fixed_m}
    eligible = {labels_c: list(SEEDS), labels_m: list(SEEDS)}
    return rb_part2_report.margin(fixed, labels, eligible, (labels_c, labels_m), n_resamples, workers)


def teacher_margin(fixed_c, fixed_t, labels, n_resamples=N_DRAWS, workers=8):
    """SC6: report 12-28's d_T,a = A(X-c*) - A(T_a): I_b, then S_b for X-c*; T draws nothing (`fixed_t` = {0: {'r20'}})."""
    res = rg_held.bootstrap_d_t(fixed_c, fixed_t, labels, n_resamples, workers)
    return dict(d=res['d'], interval=res['interval'], A_cand=res['A_G'], A_T=res['A_T'])


def arm_means(fixed_arm, labels):
    """{reading: mean over the seeds of the mean AMI over the reading's partitions} on all rows (the point estimate of
    `rb_score.score_indices` with rows = 0..n-1; labels < 0 masked)."""
    mask = labels >= 0
    out = {}
    for reading in next(iter(fixed_arm.values())):
        out[reading] = float(np.mean([np.mean([float(rb_score.ami(labels[mask], p[mask])) for p in fixed_arm[s][reading]])
                                      for s in sorted(fixed_arm)]))
    return out


# ============================================================================================ cells
def make_cell(sc, encoder, aspect, candidate, comparator, candidate_value, comparator_value, d, lo, hi, r, r_source,
              n_draws, seeds=SEEDS, data='artelingo_held', **extra):
    """One cell of `cells_artelingo_clu` (index format). `r` is the unrounded LEVEL unit, None for a WIN-only bar."""
    assert sc in ('SC1', 'SC3', 'SC6') and aspect in ASPECTS
    values = [candidate_value, comparator_value, d, lo, hi]
    assert all(np.isfinite(v) for v in values), (sc, encoder, aspect, values)
    assert (r is None) == (sc != 'SC6'), 'only SC6 cells carry a LEVEL unit'
    assert r is None or (np.isfinite(r) and r > 0)
    cell = dict(sc=sc, encoder=encoder, aspect=aspect, task='clu', data=data, candidate=candidate, comparator=comparator,
                candidate_value=float(candidate_value), comparator_value=float(comparator_value), d=float(d),
                lo=float(lo), hi=float(hi), r=None if r is None else float(r), r_source=r_source, n_draws=int(n_draws),
                seeds=list(seeds), **extra)
    assert set(CELL_KEYS) <= set(cell)
    return cell


def sc6_unit(aspect, ranges):
    """SC6 cell unit: X-c*'s own validation range decides; the glossary's unit goes beside it (§3.4)."""
    return dict(r=float(ranges[aspect]), r_source='X-c* own validation seed range (clustering AMI, seeds 0 to 4)',
                r_glossary=GLOSSARY_R[aspect])


def aspect_cells(enc, aspect, c_label, margins, ranges, n_resamples, matched_name, frozen_name):
    """The three cells of one aspect from the three margins (frozen, matched, teacher)."""
    cand = f'cstar_{enc}'
    fz, mt, tc = margins['frozen'], margins['matched'], margins['teacher']
    unit = sc6_unit(aspect, ranges)
    cells = [
        make_cell('SC1', enc, aspect, cand, f'frozen_{enc}', fz['A_cand'], fz['A_e'][f'{frozen_name}-r20'], fz['d'],
                  fz['interval'][0], fz['interval'][1], None, None, n_resamples),
        make_cell('SC3', enc, aspect, cand, f'matched_{matched_name}', mt['A_cand'], mt['A_e'][mt['max_entry']], mt['d'],
                  mt['interval'][0], mt['interval'][1], None, None, n_resamples,
                  comparator_reading=mt['max_entry'], comparator_readings=mt['A_e']),
        make_cell('SC6', enc, aspect, cand, f'teacher_{aspect}', tc['A_cand'], tc['A_T'], tc['d'], tc['interval'][0],
                  tc['interval'][1], unit['r'], unit['r_source'], n_resamples, r_glossary=unit['r_glossary']),
    ]
    return cells


def write_cells(path, cells, encoders):
    """Merge `cells` into the list file: cells of `encoders` are replaced, other encoders' cells kept (atomic)."""
    path = Path(path)
    old = json.loads(path.read_text()) if path.exists() else []
    assert isinstance(old, list)
    kept = [c for c in old if c['encoder'] not in set(encoders)]
    for c in cells:
        assert set(CELL_KEYS) <= set(c) and c['encoder'] in set(encoders)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(kept + list(cells), indent=1, allow_nan=False) + '\n')
    tmp.replace(path)
    return kept + list(cells)


# ============================================================================================ readers
def sha_file(path):
    return rb_score.sha_file(path)


def read_stored_set(path, n_rows, n_sets, source_file=None):
    """A stored fixed-partition file, read only (never written): its sidecar's partition hash, H's hash, the row count
    and, with `source_file`, the SHA-256 of the output file it came from (`rg_held.read_frozen_partitions`)."""
    assert Path(path).is_file(), f'{path}: missing'
    return rg_held.read_frozen_partitions(path, n_rows, source_file, n_sets)


def stored_k(root, arm, aspect, n_rows, seeds=SEEDS, check_source=True):
    """{seed: {'r20'}} of a K arm (X-c* of 10-25/10-26, or r10's c*) from `root/partitions/full/<arm>/`."""
    out = {}
    for s in seeds:
        src = Path(root) / f'full/held/{arm}/{aspect}__s{s}.npz' if check_source else None
        out[s] = {'r20': read_stored_set(Path(root) / f'partitions/full/{arm}/{aspect}__s{s}__r20.npz', n_rows, 3, src)}
    return out


def stored_m(root, arm, n_rows, seeds=SEEDS, check_source=True):
    """{seed: {'r20', 'own'}} of a matched arm from `root/partitions/full/<arm>/percept__s{seed}__{reading}.npz`."""
    out = {}
    for s in seeds:
        src = Path(root) / f'full/held/{arm}/P3__s{s}.npz' if check_source else None
        out[s] = {r: read_stored_set(Path(root) / f'partitions/full/{arm}/percept__s{s}__{r}.npz', n_rows,
                                     1 if r == 'own' else 3, src) for r in ('r20', 'own')}
    return out


def stored_frozen(root, arm, n_rows):
    return {0: {'r20': read_stored_set(Path(root) / f'partitions/full/{arm}/frozen__s0__r20.npz', n_rows, 3)}}


def stored_teacher(root, aspect, n_rows):
    return {0: {'r20': read_stored_set(Path(root) / f'partitions/full/T/{aspect}__s0__r20.npz', n_rows, 3)}}


def cstar_fixed(held_dir, cache, aspect, sidecar_arm='k11=40'):
    """X-c*'s fixed r20 partitions of seeds 0 to 4 from its held exports (r10's `rs_stage0.candidate_fixed`: sidecar and
    unit-norm checks, then `rb_score.fixed_partitions` into `cache`). Returns (fixed, sources)."""
    cand = S0.Candidate('C', Path(held_dir), sidecar_arm)
    return S0.candidate_fixed(cand, 'full', Path(cache), aspect)


def legacy_loader(enc, names):
    """X's default ArtELingo image of `names` as in 10-25 and 10-26 (`rb_encoders.full_images`, float32, row-normalized)."""
    import eg_encoders
    import rb_encoders
    eg_encoders.extend_rb_maps()
    return rb_encoders.full_images(ENC[enc]['store'], names)


def store_loader(enc, names):
    """The r14 store reader (ticket 01): `bs_stores.load('artelingo', view, keys)`."""
    import bs_stores
    return bs_stores.load('artelingo', enc, list(names))


def frozen_fixed(enc, names, cache, held_sha, loader=legacy_loader):
    """X's frozen image clustered like an output (20 clusters, three states) and fixed once under `cache`; {0: {'r20'}}."""
    x = loader(enc, names)
    assert x.shape[0] == len(names) and x.dtype == np.float32
    out = rb_score.fixed_partitions(x, 'K', Path(cache), 'frozen', 0, held_sha, rb_score.array_sha256(x))
    return {0: {'r20': out['r20']}}


@contextlib.contextmanager
def cstar_root(root):
    """r11's validation reader with only the root changed (`rt_data.CSTAR_ROOT`), restored afterwards."""
    old = rt_data.CSTAR_ROOT
    rt_data.CSTAR_ROOT = Path(root)
    try:
        yield
    finally:
        rt_data.CSTAR_ROOT = old


def validation_ami(x, y):
    """Mean over three k-means states of the AMI (20 clusters) of an export `x` against labels `y` (< 0 masked)."""
    import ct_score
    ps = ct_score.output_partitions(x, 'k1')['r20']
    mask = y >= 0
    return float(np.mean([float(rb_score.ami(y[mask], p[mask])) for p in ps]))


def clustering_ranges(root, seeds=SEEDS):
    """X-c*'s validation clustering range per aspect: AMI (20 clusters, three states) of each seed's validation export
    (`root/train/k11=40/R6-5/<aspect>__<graph>__s<seed>.npz`, 6,152 paintings) against the validation labels, then max - min
    over seeds. Validation only. Returns {'ranges': {aspect: float}, 'per_seed': {aspect: {seed: ami}}, 'sources': {...}}."""
    names = rt_data.names('val')
    labels = rt_data.labels('val')
    per_seed, sources = {}, {}
    with cstar_root(root):
        for aspect in ASPECTS:
            per_seed[aspect], sources[aspect] = {}, {}
            for s in seeds:
                x = rt_data._cstar('val', aspect, s, names).astype(np.float32)
                per_seed[aspect][s] = validation_ami(x, labels[aspect])
                f = Path(root) / f'train/k11=40/R6-5/{aspect}__{GRAPHS[aspect]}__s{s}.npz'
                sources[aspect][str(s)] = dict(path=str(f), sha256=sha_file(f))
    return dict(ranges={a: seed_range(per_seed[a]) for a in ASPECTS}, per_seed=per_seed, sources=sources,
                n_val=len(names), labelled={a: int((labels[a] >= 0).sum()) for a in ASPECTS})


def write_ranges(path, enc, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(encoder=enc, kind='artelingo_clustering_validation_ranges', **result)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(record, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)
    return record


def read_ranges(path, enc):
    rec = json.loads(Path(path).read_text())
    assert rec['encoder'] == enc and set(rec['ranges']) == set(ASPECTS), path
    assert all(np.isfinite(v) and v > 0 for v in rec['ranges'].values()), path
    return rec['ranges']


# ============================================================================================ scoring
def held_setup():
    """(names, sha256, {aspect: labels}) of the 12,281 held paintings: the CT's list and labels, as r10's scorer."""
    names, sha, labels = rb_part2_report.held_labels()
    assert sha == rs_paths.HELD_SHA256 and len(names) == rs_paths.N_HELD
    return names, sha, labels


def score_encoder(enc, held_dir, ranges, cache, n_resamples=N_DRAWS, workers=8, loader=legacy_loader,
                  frozen='compute', sidecar_arm='k11=40', check_source=True):
    """SC1, SC3 and SC6 cells of one encoder's X-c* on ArtELingo held (three aspects). `held_dir` is X-c*'s
    `held/k11=40/`; `ranges` its validation ranges (stage 1, written before this runs); `cache` where X-c*'s and X's
    frozen partitions are fixed. The matched arm and the teacher are 10-25's and 10-26's stored partitions.
    `frozen`: 'compute' (cluster X's default image) or 'stored' (the older round's F partitions: smoke and replay)."""
    spec = ENC[enc]
    names, sha, labels = held_setup()
    n = len(names)
    cells, detail = [], {}
    if frozen == 'compute':
        fz_fixed = frozen_fixed(enc, names, Path(cache) / 'frozen' / enc, sha, loader)
    else:
        fz_fixed = stored_frozen(spec['root'], spec['frozen'], n)
    m_fixed = stored_m(spec['root'], spec['matched'], n, check_source=check_source and enc != 'qwen')
    for aspect in ASPECTS:
        y = labels[aspect]
        c_fixed, c_src = cstar_fixed(held_dir, Path(cache) / 'cstar' / enc, aspect, sidecar_arm)
        t_fixed = stored_teacher(TEACHER_ROOT, aspect, n)
        margins = dict(
            frozen=frozen_margin(c_fixed, fz_fixed, y, n_resamples, workers, 'C', spec['frozen']),
            matched=matched_margin(c_fixed, m_fixed, y, n_resamples, workers, 'C', spec['matched']),
            teacher=teacher_margin(c_fixed, t_fixed, y, n_resamples, workers))
        cells += aspect_cells(enc, aspect, 'C', margins, ranges, n_resamples, spec['matched'], spec['frozen'])
        detail[aspect] = dict(sources=c_src)
        print(f'{enc} {aspect}: ' + ', '.join(f'{k} d={m["d"]:+.4f}' for k, m in margins.items()), flush=True)
    return cells, detail


# ============================================================================================ replays
def _item(ours, theirs):
    diff = float(abs(ours - theirs))
    return dict(ours=float(ours), theirs=float(theirs), abs_diff=diff, ok=bool(diff < TOL))


def _result(items):
    worst = max(i['abs_diff'] for i in items.values())
    return dict(max_abs_diff=worst, passed=bool(all(i['ok'] for i in items.values())), items=items)


def replay_0vi(n_resamples=N_DRAWS, workers=8, recompute_cache=None):
    """0(vi), report 12-28 part: c*'s and the teacher's held AMI and the d_T intervals of r10's `stage4/verdict.json`.
    c*'s partitions are r10's stored ones (read only, source hashes checked); with `recompute_cache` (a scratch folder)
    they are fixed again from the held exports and must equal the stored ones. Uses `teacher_margin` (SC6's path)."""
    n = rs_paths.N_HELD
    verdict = json.loads((bs_paths.R10_OUT / 'stage4/verdict.json').read_text())
    labels = held_setup()[2]
    items = {}
    for aspect in ASPECTS:
        c_fixed = {s: {'r20': read_stored_set(bs_paths.R10_OUT / f'stage4/partitions/full/k11=40/{aspect}__s{s}__r20.npz',
                                              n, 3, R10_FULL / f'held/k11=40/{aspect}__s{s}.npz')} for s in SEEDS}
        if recompute_cache is not None:
            again, _ = cstar_fixed(R10_FULL / 'held/k11=40', recompute_cache, aspect)
            items[f'partitions/{aspect}'] = dict(ours=0.0, theirs=0.0, ok=all(
                np.array_equal(again[s]['r20'], c_fixed[s]['r20']) for s in SEEDS),
                abs_diff=0.0 if all(np.array_equal(again[s]['r20'], c_fixed[s]['r20']) for s in SEEDS) else 1.0)
        res = teacher_margin(c_fixed, stored_teacher(rs_paths.RB_PART2, aspect, n), labels[aspect],
                             n_resamples, workers)
        v = verdict['aspects'][aspect]
        for key, ours, theirs in (('A_cand', res['A_cand'], v['A_cand']), ('A_T', res['A_T'], v['A_T']),
                                  ('d', res['d'], v['d']), ('lo', res['interval'][0], v['lo']),
                                  ('hi', res['interval'][1], v['hi'])):
            items[f'{aspect}/{key}'] = _item(ours, theirs)
    return _result(items)


def replay_standin_matched(n_resamples=N_DRAWS, workers=8, aspects=ASPECTS):
    """The SC3 path on r10's c* against B1 (K4's matched arm), against r10's `stage4/reported.json` margin "b_matched"
    (the same statistic, stored with 2,000 draws): d, interval, A_cand and both entries within 1e-9. Smoke of the
    interface with the stand-in run, not one of the spec's replays."""
    n = rs_paths.N_HELD
    labels = held_setup()[2]
    rep = json.loads((bs_paths.R10_OUT / 'stage4/reported.json').read_text())['aspects']
    m_fixed = stored_m(RB_PART2, 'B1', n, check_source=False)
    items = {}
    for aspect in aspects:
        c_fixed = {s: {'r20': read_stored_set(bs_paths.R10_OUT / f'stage4/partitions/full/k11=40/{aspect}__s{s}__r20.npz',
                                              n, 3, R10_FULL / f'held/k11=40/{aspect}__s{s}.npz')} for s in SEEDS}
        res = matched_margin(c_fixed, m_fixed, labels[aspect], n_resamples, workers, 'k11=40', 'B1')
        v = rep[aspect]['b_matched']
        items.update({f'{aspect}/d': _item(res['d'], v['d']), f'{aspect}/lo': _item(res['interval'][0], v['lo']),
                      f'{aspect}/hi': _item(res['interval'][1], v['hi']), f'{aspect}/A_cand': _item(res['A_cand'], v['A_cand'])})
        for k, val in v['A_e'].items():
            items[f'{aspect}/{k}'] = _item(res['A_e'][k], val)
    return _result(items)


def reported_stage4(root):
    return json.loads((Path(root) / 'reported_stage4.json').read_text())


def replay_0vii_matched(encs=('siglip2', 'siglip1', 'clip_l14')):
    """0(vii), matched arms: each arm's own and 20-cluster entries from its stored held outputs' partitions (sources hashed)
    against `reported_stage4` (style, genre: margin b_matched's A_e; emotion: the seed spread's mean)."""
    n = rs_paths.N_HELD
    labels = held_setup()[2]
    items = {}
    for enc in encs:
        spec = ENC[enc]
        rep = reported_stage4(spec['root'])['blocks'][spec['cand']]['aspects']
        m_fixed = stored_m(spec['root'], spec['matched'], n)
        for aspect in ASPECTS:
            ours = arm_means(m_fixed, labels[aspect])
            if aspect == 'emotion':
                theirs = {r: rep['emotion']['point_spread'][spec['matched']][r]['mean'] for r in ('r20', 'own')}
            else:
                theirs = {r: rep[aspect]['margins']['b_matched']['A_e'][f'{spec["matched"]}-{r}'] for r in ('r20', 'own')}
            for r in ('r20', 'own'):
                items[f'{spec["matched"]}/{aspect}/{r}'] = _item(ours[r], theirs[r])
    return _result(items)


def replay_0xii_f(encs=('siglip2', 'clip_l14', 'siglip1'), recompute_cache=None, loader=legacy_loader):
    """0(xii), F part: the F2, F7, F8 20-cluster held AMIs on emotion (0.0569, 0.0659, 0.0672), style and genre against
    the `c_frozen` entries of `reported_stage4`, from the stored frozen partitions. With `recompute_cache` X's default
    image is clustered again by `frozen_fixed` (the path SC1 uses) and its partitions must equal the stored ones."""
    n = rs_paths.N_HELD
    names, sha, labels = held_setup()
    items = {}
    for enc in encs:
        spec = ENC[enc]
        rep = reported_stage4(spec['root'])['blocks'][spec['cand']]['aspects']
        fz = stored_frozen(spec['root'], spec['frozen'], n)
        if recompute_cache is not None:
            again = frozen_fixed(enc, names, Path(recompute_cache) / enc, sha, loader)
            same = bool(np.array_equal(again[0]['r20'], fz[0]['r20']))
            items[f'{spec["frozen"]}/partitions'] = dict(ours=float(same), theirs=1.0, abs_diff=0.0 if same else 1.0, ok=same)
        for aspect in ASPECTS:
            ours = arm_means(fz, labels[aspect])['r20']
            items[f'{spec["frozen"]}/{aspect}'] = _item(ours, rep[aspect]['margins']['c_frozen']['A_e'][f'{spec["frozen"]}-r20'])
    return _result(items)


def replay_0xii_k(encs=('siglip2', 'clip_l14', 'siglip1'), aspects=('style', 'genre'), n_resamples=N_DRAWS, workers=8):
    """0(xii), K part: K2, K7 and K8 `b_matched` d, interval, X-c* score and entries (style, genre), by `matched_margin`
    on the stored partitions, against `reported_stage4`."""
    n = rs_paths.N_HELD
    labels = held_setup()[2]
    items = {}
    for enc in encs:
        spec = ENC[enc]
        rep = reported_stage4(spec['root'])['blocks'][spec['cand']]['aspects']
        m_fixed = stored_m(spec['root'], spec['matched'], n)
        for aspect in aspects:
            k_fixed = stored_k(spec['root'], spec['cand'], aspect, n)
            res = matched_margin(k_fixed, m_fixed, labels[aspect], n_resamples, workers, spec['cand'], spec['matched'])
            v = rep[aspect]['margins']['b_matched']
            tag = f'{spec["cand"]}/{aspect}'
            items.update({f'{tag}/d': _item(res['d'], v['d']), f'{tag}/lo': _item(res['interval'][0], v['interval'][0]),
                          f'{tag}/hi': _item(res['interval'][1], v['interval'][1]),
                          f'{tag}/A_cand': _item(res['A_cand'], v['A_cand'])})
            for k, val in v['A_e'].items():
                items[f'{tag}/{k}'] = _item(res['A_e'][k], val)
            assert res['max_entry'] == v['max_entry'], (tag, res['max_entry'], v['max_entry'])
    return _result(items)


ABLATION_ARMS = ('k11=40', 'k11=40-no_relational', 'k11=40-no_sharp_row', 'k11=40-no_direct', 'k11=40-no_buddy',
                 'k11=40-no_caption_endpoints', 'k11=40-no_content_correction', 'k11=40-no_instructed_graphs',
                 'k11=40-no_shared_trunk')


def replay_0vii_sha():
    """0(vii), exports: every r10 ablation export (c* and the eight removals; held, validation and weights, 3 aspects, 5 seeds)
    hashes to its sidecar. Held: the sidecar's array hash (`rg_held.load_held_output`, the check the ablation scorer used);
    validation: the sidecar's `output_sha256` against the file."""
    held_sha = rs_paths.HELD_SHA256
    items = {}
    for arm in ABLATION_ARMS:
        bad = []
        for aspect in ASPECTS:
            for s in SEEDS:
                f = R10_FULL / f'held/{arm}/{aspect}__s{s}.npz'
                try:
                    rg_held.load_held_output(f, arm, s, held_sha, aspect=aspect)
                except AssertionError as err:
                    bad.append(f'held {aspect} s{s}: {err}')
                # validation export: the graph row in its name differs by arm (removal arms), so glob it (exactly one)
                found = sorted((R10_FULL / f'train/{arm}/R6-5').glob(f'{aspect}__*__s{s}.npz'))
                assert len(found) == 1, f'{arm} {aspect} s{s}: {len(found)} validation exports'
                v = found[0]
                side = json.loads(v.with_suffix('.json').read_text())
                if side['output_sha256'] != sha_file(v):
                    bad.append(f'validation {aspect} s{s}: hash differs')
                # weights export: the weights sidecar names the weights file's hash and the validation output's hash
                w = R10_FULL / f'weights/{arm}/{aspect}__s{s}.pt'
                wside = json.loads(w.with_suffix('.json').read_text())
                if wside['weights_sha256'] != sha_file(w):
                    bad.append(f'weights {aspect} s{s}: hash differs')
                if wside['output_sha256'] != side['output_sha256']:
                    bad.append(f'weights {aspect} s{s}: output_sha256 differs from the validation sidecar')
        items[arm] = dict(ours=float(len(bad)), theirs=0.0, abs_diff=float(len(bad)), ok=not bad, bad=bad)
    return _result(items)


def replay_0vii_removals(n_resamples=N_DRAWS, workers=8, removals=None):
    """0(vii), removals: each removal's held change against c* (Delta, interval, both AMIs) by `rs_ablation.held_effect` on
    r10's stored held partitions (`ablation/cache/held/<arm>/`, read only, sources hashed), against `ablation/held.json`."""
    import rs_ablation
    n = rs_paths.N_HELD
    labels = held_setup()[2]
    held = json.loads((bs_paths.R10_OUT / 'ablation/held.json').read_text())
    removals = list(removals or rs_ablation.REMOVALS)
    cache = bs_paths.R10_OUT / 'ablation/cache/held'

    def fixed_of(arm, aspect):
        return {s: {'r20': read_stored_set(cache / arm / f'{aspect}__s{s}__r20.npz', n, 3,
                                           R10_FULL / f'held/{arm}/{aspect}__s{s}.npz')} for s in SEEDS}
    items = {}
    for x in removals:
        arm = f'k11=40-no_{x}'
        for aspect in ASPECTS:
            res = rs_ablation.held_effect(fixed_of(arm, aspect), fixed_of('k11=40', aspect), labels[aspect], n_resamples,
                                          workers)
            v = held['results'][x]['aspects'][aspect]
            for key, ours in (('effect', res['effect']), ('lo', res['lo']), ('hi', res['hi']),
                              ('arm_ami', res['arm_ami']), ('final_ami', res['final_ami'])):
                items[f'{x}/{aspect}/{key}'] = _item(ours, v[key])
    return _result(items)


# ============================================================================================ CLI
def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    p.add_argument('--n-draws', type=int, default=N_DRAWS)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('ranges', help='X-c* validation seed ranges -> OUT/stage1/artclu_ranges_<enc>.json')
    r.add_argument('--enc', required=True, choices=sorted(ENC))
    r.add_argument('--root', type=Path, help='default OUT/cstar/<enc>; the qwen stand-in uses r10 full/')
    s = sub.add_parser('score', help='SC1, SC3, SC6 cells -> OUT/stage3/cells_artelingo_clu.json')
    s.add_argument('--enc', required=True, choices=sorted(ENC))
    s.add_argument('--root', type=Path)
    s.add_argument('--loader', choices=('legacy', 'bs_stores'), default='legacy')
    s.add_argument('--frozen', choices=('compute', 'stored'), default='compute')
    rp = sub.add_parser('replay', help='0(vi) | 0(vii) | 0(xii)')
    rp.add_argument('gate', choices=('vi', 'vii-matched', 'vii-sha', 'vii-removals', 'xii-f', 'xii-k', 'standin'))
    rp.add_argument('--recompute', action='store_true', help='fix partitions again into OUT/replays/cache and compare')
    rp.add_argument('--removals', nargs='+')
    args = p.parse_args(argv)
    t0 = time.monotonic()
    out = bs_paths.OUT
    assert args.cmd == 'replay' or not str(out).startswith(str(bs_paths.R10_OUT)), 'never write into an older round'
    if args.cmd in ('ranges', 'score'):
        root = args.root or (R10_FULL if args.enc == 'qwen' else out / 'cstar' / args.enc)
        rpath = out / f'stage1/artclu_ranges_{args.enc}.json'
        if args.cmd == 'ranges':
            rec = write_ranges(rpath, args.enc, clustering_ranges(root))
            print(f'{args.enc}: ranges {rec["ranges"]} -> {rpath} ({time.monotonic() - t0:.0f}s)')
            return 0
        ranges = read_ranges(rpath, args.enc)          # stage 1 first: no held number before the ranges are written
        loader = store_loader if args.loader == 'bs_stores' else legacy_loader
        cells, _ = score_encoder(args.enc, Path(root) / 'held/k11=40', ranges, out / 'stage3/partitions', args.n_draws,
                                 args.workers, loader, args.frozen)
        write_cells(out / 'stage3' / CELLS_NAME, cells, [args.enc])
        print(f'{args.enc}: {len(cells)} cells -> {out / "stage3" / CELLS_NAME} ({time.monotonic() - t0:.0f}s)')
        return 0
    cache = out / 'replays/cache' if args.recompute else None
    g = args.gate
    if g == 'vi':
        res = replay_0vi(args.n_draws, args.workers, cache)
    elif g == 'standin':
        res = replay_standin_matched(args.n_draws, args.workers)
    elif g == 'vii-matched':
        res = replay_0vii_matched()
    elif g == 'vii-sha':
        res = replay_0vii_sha()
    elif g == 'vii-removals':
        res = replay_0vii_removals(args.n_draws, args.workers, args.removals)
    elif g == 'xii-f':
        res = replay_0xii_f(recompute_cache=cache)
    else:
        res = replay_0xii_k(n_resamples=args.n_draws, workers=args.workers)
    path = out / f'replays/replay_{g}.json'
    _write_json(path, dict(gate=g, n_draws=args.n_draws, **res))
    print(f'replay {g}: {"PASS" if res["passed"] else "FAIL"} max_abs_diff {res["max_abs_diff"]:.3g} '
          f'({time.monotonic() - t0:.0f}s) {path}')
    return 0 if res['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
