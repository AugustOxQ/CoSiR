"""Affection scores of r14 (ticket 07): SC4, SC5 and the SC6 photo cells for the X photo models, their validation seed
ranges and read-out picks (stage 1), the frozen-image bars' picks (D19), and the replay 0(vi), r13 part. CPU only.

Spec: docs/superpowers/specs/2026-11-02-backbones-sweep-design.md, SC4, SC5, SC6, §3.4 ("Affection", "Seed ranges for
LEVEL"), §6 ("Selection rules", stage 1), §3.5 0(vi), D19. Nothing is rewritten from older rounds; each statistic calls the
function that produced the replayed number (r13's verdict code, which itself calls r12's):

  ranges, read-out pick  `pf_stage2.ranges`, `pf_stage2.readout` (r13), over r12's `ac_stats.measure_seed_ranges` and
                         `ac_select.arm_table` (r11's cross-fit, guard and 0.001 r tie rule). The X photo model uses its own
                         r'_mAP, never r12's.
  frozen-image bar pick  CLIP L/14 (and the Qwen stand-in) keep r12's committed pick (`stage1/picks.json`); SigLIP 2 and
                         SigLIP 1 pick by `ac_select.arm_table` over the frozen image with r12's r_ret,mAP (D19).
  clustering             `pf_stage3.candidate_partitions` (r12's `ac_cluster.cached_partitions`), then
                         `ac_cluster.paired` (2,000 draws: photos, then seeds; the frozen image and the teacher draw no seeds).
  retrieval              `pf_stage3.candidate_retrieval` (r12's `ac_stats.arm_retrieval`: W fitted on all validation
                         photos, expansion on the test gallery), then `rt_stats.paired` (10,000 joint draws, queries and seeds).
  teacher                r12's artwork-wording outputs (per-query AP, partitions, pick) through `pf_stage3.load_comparators`.

  SC4  X photo AMI  against X frozen image            task clu   r null (WIN bar)
  SC5  X photo mAP  against X frozen image, picks     task ret   r null (WIN bar)
  SC6  X photo AMI  against the teacher               task clu   r = the X photo model's own r'_clu
  SC6  X photo mAP  against the teacher, picks        task ret   r = the X photo model's own r'_mAP

Held data (the test photos) are read only by `score` and the replay; `stage1` reads validation only, and `score` refuses to
start without the stage 1 file (no held number before the ranges and picks are written).

Files (under `bs_paths.OUT`; BS_OUTPUTS for a smoke run):
  in   exports/affection/<arm>/<split>__s<seed>.npz   `keys`, `emb` float16 (+ .json with `sha256`); arm `photo_<enc>`
  out  stage1/affection_ranges_<enc>.json             `ranges` {clu, map, ndcg}, `ranges_doc` (per-seed values),
                                                      `photo_readout` {pick, scores}, `frozen` {pick, source, r, scores}
  out  stage3/cells_affection.json                    list of cells (the index's format; same pattern as ticket 05)

CLI (`python bs_affection.py -h`): ranges | score | replay vi | standin prepare | standin check.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import bs_paths as P
import bs_stores

if str(P.TEST / '20261031_photo_finetune') not in sys.path:     # this checkout's copy of r13 (puts r12, r11, r10 on the path)
    sys.path.insert(0, str(P.TEST / '20261031_photo_finetune'))
import pf_paths  # noqa: E402  (asserts AC_SMOKE / AC_OUTPUTS / PF_OUTPUTS are not redirected)
import pf_stage2  # noqa: E402
import pf_stage3  # noqa: E402
import ac_cluster  # noqa: E402
import ac_data  # noqa: E402
import ac_select  # noqa: E402
import ac_stats  # noqa: E402
import rs_vh  # noqa: E402
import rt_readout  # noqa: E402
import rt_stats  # noqa: E402

SEEDS = tuple(P.SEEDS)
N_CLU, N_RET = 2000, 10000                                  # r12's paired draws (photos and seeds; queries and seeds)
TOL = 1e-9                                                  # 0(vi)
WIDTH_OUT = 2048                                            # every photo model's output width (§3.2)
CELLS_NAME = 'cells_affection.json'
CELL_KEYS = ('sc', 'encoder', 'aspect', 'task', 'data', 'candidate', 'comparator', 'candidate_value',
             'comparator_value', 'd', 'lo', 'hi', 'r', 'r_source', 'n_draws', 'seeds')
ENCODERS = ('siglip2', 'siglip1', 'clip_l14', 'qwen')       # `qwen` is the stand-in of the smoke, not an encoder of part a
VIEW = {'qwen': 'qwen_default', 'siglip2': 'siglip2', 'siglip1': 'siglip1', 'clip_l14': 'clip_l14'}
R12_PICK_ARM = {'qwen': 'plain', 'clip_l14': 'clip'}        # frozen arms that keep r12's committed pick (D19)
R12_PICKS = pf_paths.R12_OUT / 'stage1/picks.json'
R12_RANGES = pf_paths.R12_OUT / 'stage0b/ranges.json'
R12_VERDICT = pf_paths.R12_OUT / 'stage2/verdict.json'
R12_BAR_ARM = {'clip_l14': 'clip'}                          # frozen bars that r12 scored (row of its stage 2 verdict)
SC_ROWS = (('SC4', 'clu', 'frozen'), ('SC5', 'ret', 'frozen'), ('SC6', 'clu', 'teacher'), ('SC6', 'ret', 'teacher'))


# ============================================================================================ stores and exports
def export_path(arm, split, seed, root=None):
    return Path(P.OUT if root is None else root) / 'exports/affection' / arm / f'{split}__s{seed}.npz'


def join_rows(file_keys, rows, keys, what='rows'):
    """`rows` re-ordered to `keys` by key (never by position): every requested key must be listed once in `file_keys`."""
    file_keys = [str(k) for k in file_keys]
    keys = list(keys)
    assert len(file_keys) == len(rows), f'{what}: {len(file_keys)} keys for {len(rows)} rows'
    assert len(set(file_keys)) == len(file_keys), f'{what}: duplicate keys in the file'
    assert keys and len(set(keys)) == len(keys), f'{what}: requested keys must be unique'
    lookup = {k: i for i, k in enumerate(file_keys)}
    missing = [k for k in keys if k not in lookup]
    assert not missing, f'{what}: {len(missing)} keys not in the file, e.g. {missing[:2]}'
    idx = np.fromiter((lookup[k] for k in keys), dtype=np.int64, count=len(keys))
    return rows[idx]


def load_export(arm, split, seed, keys, root=None):
    """Checked float16 rows of an emotion-pathway export (`keys`, `emb`; ticket 04's layout), joined to `keys` by key.
    Asserts: the sidecar's sha256 equals the file's, float16 2-D finite rows, unit norm within 1e-3."""
    path = export_path(arm, split, seed, root)
    side = json.loads(path.with_suffix('.json').read_text())
    assert side['sha256'] == rs_vh.file_sha(path), f'{path}: file differs from its sidecar'
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'keys', 'emb'}, (path, z.files)
        file_keys, emb = z['keys'].tolist(), z['emb']
    assert emb.dtype == np.float16 and emb.ndim == 2 and np.isfinite(emb).all(), (path, emb.dtype, emb.shape)
    out = join_rows(file_keys, emb, keys, str(path))
    assert np.abs(np.linalg.norm(out.astype(np.float32), axis=1) - 1).max() < 1e-3, f'{path}: rows are not unit'
    return out


def frozen_embed(enc, keys):
    """X's default image, unit float32 [n, d] in `keys` order (r12's `ac_data.embed('plain'|'clip')` form: the stored
    float16 rows cast to float32 and normalized)."""
    return ac_data._unit(bs_stores.load('affection', VIEW[enc], keys))


class PhotoArm:
    """The five seeds of an X photo model's exports as arrays in `ac_data.keys(split)` order."""

    def __init__(self, arm, root=None):
        self.arm, self.root = arm, root
        self.keys = {'val': ac_data.keys('val')}                # the test split is loaded only when scoring (`test_keys`)
        self._cache = {}

    def test_keys(self):
        if 'test' not in self.keys:
            self.keys['test'] = ac_data.keys('test')
        return self.keys['test']

    def stored(self, split, seed):
        if (split, seed) not in self._cache:
            keys = self.test_keys() if split == 'test' else self.keys[split]
            self._cache[(split, seed)] = load_export(self.arm, split, seed, keys, self.root)
        return self._cache[(split, seed)]

    def embed(self, split, seed):
        return ac_data._unit(self.stored(split, seed))

    def candidate(self, readout):
        return pf_stage3.candidate_from_data(self.arm, readout, _Split(self))


class _Split:
    """`pf_stage3.candidate_from_data`'s loader over a `PhotoArm`."""

    def __init__(self, arm):
        self.arm = arm

    def stored(self, split, seed):
        return self.arm.stored(split, seed)

    def embed(self, split, seed):
        return self.arm.embed(split, seed)


# ============================================================================================ stage 1
def photo_stage1(embed_val, stored_val, labels, dist, keys, *, r12_ranges, width=WIDTH_OUT, workers=8):
    """The X photo model's seed ranges (AMI; mAP at R0; nDCG) and its read-out pick on Affection validation, by r13's
    code over r12's: the pick uses the model's OWN r'_mAP (`pf_stage2.readout`), not r12's. embed_val / stored_val:
    {seed: unit float32 / float16 rows} in sorted validation-key order. Returns `pf_stage2.record`'s dict
    ({'ranges', 'r12_ranges', 'readout': {'pick', 'scores'}}), unrounded."""
    ranges = pf_stage2.ranges(stored_val.__getitem__, embed_val.__getitem__, labels=labels, dist=dist, workers=workers)
    selected = pf_stage2.readout(embed_val, ranges['map']['r'], labels=labels, keys=keys, workers=workers)
    return pf_stage2.record(ranges, selected, r12_ranges=r12_ranges, width=width)


def frozen_readout(enc, x_val, labels, keys, r12_ranges, r12_picks, workers=8, force_rule=False):
    """The frozen-image bar's read-out (D19). CLIP L/14 (and the Qwen stand-in) keep r12's committed pick (r12's `clip`
    and `plain` arms); SigLIP 2 and SigLIP 1 pick by r12's rule, `ac_select.arm_table` over the unseeded frozen
    validation image with r12's r_ret,mAP (`r12_ranges['map']['r']`), as the reused pick did. `force_rule` runs the rule
    for any encoder (the stand-in's check that the rule reproduces r12's pick).
    Returns {'pick', 'source', 'r' (the range used, null for a kept pick), 'scores' (the table, null for a kept pick)}."""
    arm = R12_PICK_ARM.get(enc)
    if arm is not None and not force_rule:
        return dict(pick=r12_picks['readout'][arm], source=f"r12 pick of the {arm} arm (stage1/picks.json)", r=None,
                    scores=None)
    r = float(r12_ranges['map']['r'])
    assert np.isfinite(r) and r > 0
    table, pick = ac_select.arm_table({('emotion', None): x_val}, {'emotion': labels}, {'emotion': r},
                                      workers=workers, keys=keys)
    return dict(pick=rt_readout.name(pick), source="r12's rule with r12's r_ret,mAP", r=r, scores=table)


def stage1_record(enc, photo, frozen, sources=None):
    """The stage 1 file's content for one encoder: photo ranges and pick, frozen pick (index: `affection_ranges_<enc>`)."""
    ranges = photo['ranges']
    return dict(encoder=enc, kind='affection_validation_ranges_and_readouts',
                ranges={m: float(ranges[m]['r']) for m in ('clu', 'map', 'ndcg')},
                ranges_doc={m: ranges[m] for m in ('clu', 'map', 'ndcg')},
                photo_readout=photo['readout'], frozen=frozen, sources=sources or {})


def write_record(path, record, overwrite=False):
    """Atomic write; an existing file with other content is refused unless `overwrite` (picks are frozen once written)."""
    path = Path(path)
    text = json.dumps(record, indent=1, allow_nan=False) + '\n'
    if path.exists() and not overwrite:
        assert path.read_text() == text, f'{path} holds other content: picks are frozen once written (use --overwrite)'
        return record
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(text)
    tmp.replace(path)
    return record


def read_record(path, enc):
    """The stage 1 file, checked: encoder, positive finite ranges equal to the per-seed max - min, picks present."""
    rec = json.loads(Path(path).read_text())
    assert rec['encoder'] == enc and rec['kind'] == 'affection_validation_ranges_and_readouts', path
    for m in ('clu', 'map', 'ndcg'):
        per_seed = rec['ranges_doc'][m]['per_seed']
        assert len(per_seed) == 5 and rec['ranges'][m] == rec['ranges_doc'][m]['r'] == ac_stats.seed_range(per_seed), (path, m)
        assert np.isfinite(rec['ranges'][m]) and rec['ranges'][m] > 0, (path, m)
    assert rec['photo_readout']['pick'] in rec['photo_readout']['scores'] and rec['frozen']['pick'], path
    r12_r = ac_stats.load_ranges(R12_RANGES)['map']['r']      # the frozen pick uses r12's range (D19), never the photo model's
    assert rec['frozen']['r'] in (None, r12_r), (path, 'frozen pick range is not r12 r_ret,mAP', rec['frozen']['r'], r12_r)
    return rec


def ranges_path(enc, root=None):
    return Path(P.OUT if root is None else root) / f'stage1/affection_ranges_{enc}.json'


def stage1(enc, root=None, workers=8, overwrite=False):
    """Stage 1 for one encoder (validation only): X photo model's ranges and pick, and X's frozen-image pick."""
    arm = f'photo_{enc}'
    keys, labels, dist = ac_data.keys('val'), ac_data.labels('val'), ac_data.emotion_dist('val')
    pa = PhotoArm(arm, root)
    embed_val = {s: pa.embed('val', s) for s in SEEDS}
    stored_val = {s: pa.stored('val', s) for s in SEEDS}
    r12_ranges = ac_stats.load_ranges(R12_RANGES)
    photo = photo_stage1(embed_val, stored_val, labels, dist, keys, r12_ranges=r12_ranges, workers=workers)
    r12_picks = json.loads(R12_PICKS.read_text())['affection']
    frozen = frozen_readout(enc, frozen_embed(enc, keys), labels, keys, r12_ranges, r12_picks, workers)
    frozen['view'] = VIEW[enc]
    sources = {f'val__s{s}': json.loads(export_path(arm, 'val', s, root).with_suffix('.json').read_text())['sha256']
               for s in SEEDS}
    rec = stage1_record(enc, photo, frozen, sources)
    return write_record(ranges_path(enc, root), rec, overwrite)


# ============================================================================================ scoring
def make_cell(sc, task, encoder, candidate, comparator, candidate_value, comparator_value, d, lo, hi, r, r_source,
              n_draws, seeds=SEEDS, data='affection_test', **extra):
    """One cell of `cells_affection` (index format). `r` is the unrounded LEVEL unit, None for a WIN-only bar (SC4, SC5)."""
    assert sc in ('SC4', 'SC5', 'SC6') and task in ('clu', 'ret')
    assert all(np.isfinite(v) for v in (candidate_value, comparator_value, d, lo, hi)), (sc, task, encoder)
    assert (r is None) == (sc != 'SC6'), 'only SC6 cells carry a LEVEL unit'
    assert r is None or (np.isfinite(r) and r > 0)
    cell = dict(sc=sc, encoder=encoder, aspect='emotion', task=task, data=data, candidate=candidate,
                comparator=comparator, candidate_value=float(candidate_value), comparator_value=float(comparator_value),
                d=float(d), lo=float(lo), hi=float(hi), r=None if r is None else float(r), r_source=r_source,
                n_draws=int(n_draws), seeds=list(seeds), **extra)
    assert set(CELL_KEYS) <= set(cell)
    return cell


def sc6_unit(task, r):
    """SC6 cell unit: the X photo model's own validation seed range for the task (§3.4)."""
    what = 'clustering AMI' if task == 'clu' else 'mAP at R0'
    return float(r['clu' if task == 'clu' else 'map']), f"X photo model own validation seed range ({what}, seeds 0 to 4)"


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


def frozen_comparator(enc, readout, val_keys, test_keys, labels, partition_root, workers=8):
    """X's frozen image as a comparator in `pf_stage3.load_comparators`' form: arm, read-out, per-query AP and rows under
    the pick (W fitted on all validation photos), mean mAP, and its test partitions (int32 [3, n], fitted once)."""
    x_val, x_test = frozen_embed(enc, val_keys), frozen_embed(enc, test_keys)
    name = f'frozen_{enc}'
    res = ac_stats.arm_retrieval(lambda s: x_val, lambda s: x_test, readout, labels, seeds=None, workers=workers)
    assert res['dropped'] == 0, (readout, 'queries without a relevant item', res['dropped'])
    parts = ac_cluster.cached_partitions(x_test, f'affection__test__{name}__s-', partition_root)
    out = dict(arm=name, readout=readout, ap=res['ap'], rows=res['rows'], point=res['point'], labels=parts)
    check_against_r12(enc, out, labels)
    return out


def check_against_r12(enc, comp, labels, verdict_path=None):
    """CLIP L/14's frozen bar is r12's `clip` arm: its read-out, mAP at the pick and AMI must equal r12's stored row of
    `stage2/verdict.json` within 1e-9 (as r13's `pf_stage3.load_comparators` does for its arms). No-op for other encoders."""
    arm = R12_BAR_ARM.get(enc)
    if arm is None:
        return
    row = json.loads(Path(verdict_path or R12_VERDICT).read_text())['rows'][arm]
    assert comp['readout'] == row['readout'], f"{enc}: read-out {comp['readout']} differs from r12's {row['readout']}"
    assert abs(comp['point'] - row['map_picked']['map']) <= TOL, f"{enc}: mAP point differs from r12 row {arm}"
    assert abs(ac_cluster.ami(comp['labels'], labels) - row['ami']) <= TOL, f"{enc}: AMI differs from r12 row {arm}"


def score_cells(enc, cand, r, labels, comparators, partition_root, draws_clu=N_CLU, draws_ret=N_RET, workers=8):
    """SC4, SC5 and the two SC6 cells of an X photo model on the test photos.
    cand: `pf_stage3.Candidate` (five seeds, read-out = the stage 1 pick); r: {'clu': r'_clu, 'map': r'_mAP}, the model's
    own validation ranges; comparators: {'frozen': ..., 'teacher': ...} in `frozen_comparator`'s form.
    Returns {'cells': [4 cells], 'partitions': int32 [5, 3, n], 'perquery': {...}}."""
    pf_stage3._check_candidate(cand)
    pf_stage3._check_r(r)
    labels = np.asarray(labels)
    assert cand.test_stored.shape[1] == len(labels), ('candidate rows', cand.test_stored.shape, len(labels))
    parts = pf_stage3.candidate_partitions(cand, partition_root, workers)
    picked = pf_stage3.candidate_retrieval(cand, cand.readout, labels, workers)
    cells = []
    for sc, task, which in SC_ROWS:
        comp = comparators[which]
        r_cell, r_source = (None, None) if sc != 'SC6' else sc6_unit(task, r)
        if task == 'clu':
            res = ac_cluster.paired(parts, comp['labels'], labels, draws_clu, workers)
            c_val, t_val, d, (lo, hi), n_draws, extra = res['A_cand'], res['A_comp'], res['d'], res['interval'], draws_clu, {}
        else:
            assert np.array_equal(picked['rows'], comp['rows']), (sc, 'the two arms were scored on different queries')
            p = rt_stats.paired(picked['ap'], comp['ap'], draws_ret)
            c_val, t_val, d, (lo, hi), n_draws = picked['point'], comp['point'], p['d'], p['interval'], draws_ret
            extra = dict(readout_candidate=cand.readout, readout_comparator=comp['readout'])
        cells.append(make_cell(sc, task, enc, cand.name, comp['arm'], c_val, t_val, d, lo, hi, r_cell, r_source, n_draws,
                               **extra))
    return dict(cells=cells, partitions=parts, picked=picked)


def score(enc, root=None, workers=8, draws_clu=N_CLU, draws_ret=N_RET, cache=None):
    """The Affection test cells of one encoder -> `OUT/stage3/cells_affection.json`. Stage 1 first: the ranges and picks
    come from `OUT/stage1/affection_ranges_<enc>.json`, which must exist before any test photo is read."""
    root = Path(P.OUT if root is None else root)
    rec = read_record(ranges_path(enc, root), enc)
    r = {'clu': rec['ranges']['clu'], 'map': rec['ranges']['map']}
    r12_picks = json.loads(R12_PICKS.read_text())['affection']
    pa = PhotoArm(f'photo_{enc}', root)
    val_keys, test_keys, labels = pa.keys['val'], pa.test_keys(), ac_data.labels('test')
    cache = Path(cache or root / 'stage3/partitions')
    comps = pf_stage3.load_comparators(r12_picks, len(labels))
    comparators = dict(teacher=comps['teacher'],
                       frozen=frozen_comparator(enc, rec['frozen']['pick'], val_keys, test_keys, labels, cache, workers))
    cand = pa.candidate(rec['photo_readout']['pick'])
    res = score_cells(enc, cand, r, labels, comparators, cache, draws_clu, draws_ret, workers)
    write_cells(root / 'stage3' / CELLS_NAME, res['cells'], [enc])
    return res, comps


# ============================================================================================ replay 0(vi), r13 part
def _item(ours, theirs):
    d = abs(float(ours) - float(theirs))
    return dict(ours=float(ours), theirs=float(theirs), abs_diff=d, ok=bool(d <= TOL))


def _result(items):
    return dict(max_abs_diff=max(i['abs_diff'] for i in items.values()),
                passed=bool(all(i['ok'] for i in items.values())), items=items)


def replay_vi(n_clu=N_CLU, n_ret=N_RET, workers=8, cache=None):
    """0(vi), r13 part: r13's SC1 to SC5 values and intervals from r13's stored final exports (`pf_export.stored`, the five
    runs), r13's committed read-out and r' (`stage2_records.json`), through r13's own `pf_stage3.stage3` (partitions fitted
    again into `cache`, not r13's stored ones). Compared with r13's `stage3/verdict.json`: point values, d, r and
    interval ends within 1e-9, outcomes and names equal, the verdict equal, the photo row. Returns {'max_abs_diff',
    'passed', 'items'}."""
    import pf_export
    rec = json.loads(pf_paths.STAGE2_FILE.read_text())
    pick = rec['readout']['pick']
    r = {'clu': rec['ranges']['clu']['r'], 'map': rec['ranges']['map']['r']}

    class Loader:
        @staticmethod
        def stored(split, seed):
            return pf_export.stored(f'final/s{seed}', split)

        @staticmethod
        def embed(split, seed):
            return pf_export.embed(f'final/s{seed}', split)

    cand = pf_stage3.candidate_from_data('photo', pick, Loader)
    cache = Path(cache) if cache is not None else P.OUT / 'replays/cache_vi'
    res = pf_stage3.stage3(cand, r, workers=workers, draws_clu=n_clu, draws_ret=n_ret, partition_root=cache)
    ref = json.loads((P.R13_OUT / 'stage3/verdict.json').read_text())
    items = {}
    for sid in pf_stage3.SC_IDS:
        mine, theirs = res['doc']['sc'][sid], ref['sc'][sid]
        for k in pf_stage3.REPLAY_NUMBERS:
            items[f'{sid}/{k}'] = _item(mine[k], theirs[k])
        for j, k in enumerate(('lo', 'hi')):
            items[f'{sid}/{k}'] = _item(mine['interval'][j], theirs['interval'][j])
        same = all(mine[k] == theirs[k] for k in ('candidate', 'comparator', 'dataset', 'aspect', 'metric', 'readout_c',
                                                  'readout_t', 'outcome', 'pass'))
        items[f'{sid}/outcome_and_names'] = dict(ours=mine['outcome'], theirs=theirs['outcome'], abs_diff=0.0 if same else 1.0,
                                                 ok=bool(same))
    items['verdict'] = dict(ours=res['doc']['verdict'], theirs=ref['verdict'], abs_diff=float(res['doc']['verdict'] != ref['verdict']),
                            ok=bool(res['doc']['verdict'] == ref['verdict']))
    row, ref_row = res['doc']['rows']['photo'], ref['rows']['photo']
    for k in ('ami', 'ndcg_r0'):
        items[f'row/{k}'] = _item(row[k], ref_row[k])
    for k in ('map_picked', 'map_r0'):
        items[f'row/{k}'] = _item(row[k]['map'], ref_row[k]['map'])
        for s in SEEDS:
            items[f'row/{k}/s{s}'] = _item(row[k]['per_seed'][s], ref_row[k]['per_seed'][s])
    return _result(items)


# ============================================================================================ Qwen stand-in
STANDIN_ARM = 'photo_qwen'


def standin_prepare(root=None, seed=0):
    """Write r13's five final photo models' stored Affection val/test exports (`pf_export.stored`, sidecar-checked) in
    ticket 04's layout `exports/affection/photo_qwen/<split>__s<seed>.npz` under the OUT root, rows in a shuffled order, so
    the key join is exercised. (Ticket 04's `export_arm` reproduced these exports with max abs diff 0.0.)"""
    import hashlib
    import pf_export
    rng = np.random.default_rng(seed)
    written = []
    for split in ('val', 'test'):
        keys = ac_data.keys(split)
        order = rng.permutation(len(keys))
        for s in SEEDS:
            emb = pf_export.stored(f'final/s{s}', split)[order]
            path = export_path(STANDIN_ARM, split, s, root)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, keys=np.asarray([keys[i] for i in order]), emb=emb)
            path.with_suffix('.json').write_text(json.dumps(dict(
                dataset='affection', arm=STANDIN_ARM, split=split, seed=s, rows=len(keys), width=WIDTH_OUT,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(), note='r13 stored final export, shuffled rows'),
                indent=1) + '\n')
            written.append(path)
    return written


def standin_check(root=None, n_clu=N_CLU, n_ret=N_RET, workers=8):
    """Compare the stand-in run (`ranges --enc qwen`, `score --enc qwen`) with r13's reported values: the photo model's
    r'_clu, r'_mAP and pick (`stage2_records.json`); SC4 against r13's SC1 (plain), SC5 against SC2, SC6 clu against SC3
    (teacher), SC6 ret against SC4: candidate and comparator values, d, interval ends within 1e-9 and SC6's r equal to
    r13's r'. Also: the pick rule on the frozen Qwen image with r12's r_ret,mAP gives r12's `plain` pick."""
    root = Path(P.OUT if root is None else root)
    rec = read_record(ranges_path('qwen', root), 'qwen')
    r13rec = json.loads(pf_paths.STAGE2_FILE.read_text())
    ref = json.loads((P.R13_OUT / 'stage3/verdict.json').read_text())['sc']
    items = {'ranges/clu': _item(rec['ranges']['clu'], r13rec['ranges']['clu']['r']),
             'ranges/map': _item(rec['ranges']['map'], r13rec['ranges']['map']['r']),
             'photo_pick': dict(ours=rec['photo_readout']['pick'], theirs=r13rec['readout']['pick'],
                                abs_diff=float(rec['photo_readout']['pick'] != r13rec['readout']['pick']),
                                ok=rec['photo_readout']['pick'] == r13rec['readout']['pick'])}
    for k, v in rec['photo_readout']['scores'].items():
        items[f'photo_scores/{k}'] = _item(v['score'], r13rec['readout']['scores'][k]['score'])
    cells = {(c['sc'], c['task']): c for c in json.loads((root / 'stage3' / CELLS_NAME).read_text()) if c['encoder'] == 'qwen'}
    for (sc, task), theirs in (((('SC4', 'clu')), 'SC1'), (('SC5', 'ret'), 'SC2'), (('SC6', 'clu'), 'SC3'), (('SC6', 'ret'), 'SC4')):
        c, t = cells[(sc, task)], ref[theirs]
        for name, a, b in (('cand', c['candidate_value'], t['point_c']), ('comp', c['comparator_value'], t['point_t']),
                           ('d', c['d'], t['d']), ('lo', c['lo'], t['interval'][0]), ('hi', c['hi'], t['interval'][1])):
            items[f'{sc}/{task}/{name}'] = _item(a, b)
        if sc == 'SC6':
            items[f'{sc}/{task}/r'] = _item(c['r'], t['r'])
    r12_picks = json.loads(R12_PICKS.read_text())['affection']
    keys, labels = ac_data.keys('val'), ac_data.labels('val')
    rule = frozen_readout('qwen', frozen_embed('qwen', keys), labels, keys, ac_stats.load_ranges(R12_RANGES), r12_picks,
                          workers, force_rule=True)
    items['frozen_rule_pick'] = dict(ours=rule['pick'], theirs=r12_picks['readout']['plain'],
                                     abs_diff=float(rule['pick'] != r12_picks['readout']['plain']),
                                     ok=rule['pick'] == r12_picks['readout']['plain'])
    return _result(items)


# ============================================================================================ CLI
def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    p.add_argument('--n-clu', type=int, default=N_CLU)
    p.add_argument('--n-ret', type=int, default=N_RET)
    sub = p.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('ranges', help="stage 1: X photo model's ranges and pick, frozen pick -> OUT/stage1/affection_ranges_<enc>.json")
    r.add_argument('--enc', required=True, choices=ENCODERS)
    r.add_argument('--overwrite', action='store_true')
    s = sub.add_parser('score', help='SC4, SC5, SC6 cells -> OUT/stage3/cells_affection.json')
    s.add_argument('--enc', required=True, choices=ENCODERS)
    rp = sub.add_parser('replay', help='0(vi), r13 part')
    rp.add_argument('gate', choices=('vi',))
    st = sub.add_parser('standin', help='the Qwen stand-in: prepare | check')
    st.add_argument('what', choices=('prepare', 'check'))
    args = p.parse_args(argv)
    if P.OUT == P.DEFAULT_OUT and (args.cmd == 'standin' or getattr(args, 'enc', None) == 'qwen'):
        p.error('the Qwen stand-in writes cells that ticket 09 would stop on: set BS_OUTPUTS to a scratch folder first '
                f'(OUT is the production folder {P.DEFAULT_OUT})')
    t0 = time.monotonic()
    out = P.OUT
    assert not str(out).startswith(str(P.R13_OUT)) and not str(out).startswith(str(P.R12_OUT)), 'never write into an older round'
    if args.cmd == 'ranges':
        rec = stage1(args.enc, workers=args.workers, overwrite=args.overwrite)
        print(f"{args.enc}: r'_clu {rec['ranges']['clu']:.10g} r'_mAP {rec['ranges']['map']:.10g} photo pick "
              f"{rec['photo_readout']['pick']} frozen pick {rec['frozen']['pick']} -> {ranges_path(args.enc)} "
              f"({time.monotonic() - t0:.0f}s)")
        return 0
    if args.cmd == 'score':
        res, _ = score(args.enc, workers=args.workers, draws_clu=args.n_clu, draws_ret=args.n_ret)
        print(f'{args.enc}: {len(res["cells"])} cells -> {out / "stage3" / CELLS_NAME} ({time.monotonic() - t0:.0f}s)')
        return 0
    if args.cmd == 'standin' and args.what == 'prepare':
        print(f'{len(standin_prepare())} exports -> {out / "exports/affection" / STANDIN_ARM} ({time.monotonic() - t0:.0f}s)')
        return 0
    if args.cmd == 'standin':
        res, name = standin_check(n_clu=args.n_clu, n_ret=args.n_ret, workers=args.workers), 'standin'
    else:
        res, name = replay_vi(args.n_clu, args.n_ret, args.workers), 'replay_vi'
    path = out / f'replays/{name}.json'
    _write_json(path, dict(gate=name, n_clu=args.n_clu, n_ret=args.n_ret, **res))
    print(f'{name}: {"PASS" if res["passed"] else "FAIL"} max_abs_diff {res["max_abs_diff"]:.3g} '
          f'({time.monotonic() - t0:.0f}s) {path}')
    return 0 if res['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
