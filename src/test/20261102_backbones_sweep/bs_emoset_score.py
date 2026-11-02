"""EmoSet stage 2 and part c of r14 (ticket 08): r_E, read-out picks, wording picks, SC7 to SC10, descriptive rows. CPU only.

Spec: docs/superpowers/specs/2026-11-02-backbones-sweep-design.md, §3.4 "EmoSet (part c)" steps 1 to 4 ("Each map is fitted on
all EmoSet validation features"), §3.5 stage 2 and 0(xi), §6 SC7 to SC10, D4, D5, D10. The protocol is r12's (Affection),
on EmoSet's cut sets (`bs_emoset.keys`) in the sha256 order (`bs_emoset.order`). Nothing is rewritten from r12; each
statistic calls the function that produced the replayed number:

  clustering        `ac_cluster.partitions` / `cached_partitions` (r10's MiniBatchKMeans, k = 20, states 0 to 2), `ac_cluster.ami`,
                    `ac_cluster.paired` (2,000 draws: photos, then seeds; frozen arms draw no seeds). K = 8 (D10) is the same
                    fit with k = 8 (`partitions_k`; its k = 20 call equals `ac_cluster.partitions`, tested).
  retrieval         `rt_metrics.retrieval` (r11), `ac_stats.arm_retrieval` (W fitted on all validation rows, query expansion on
                    the test gallery), `rt_stats.paired` (10,000 joint draws over queries and seeds).
  read-out pick     `ac_select.arm_table` (r11's cross-fit by position parity, the guard, the 0.001 r tie rule), with r_E,mAP.
                    The parity is of the position in the sha256 order (D17), so `keys=` is NOT passed (r12 passes its sorted
                    keys there); `check_order` asserts the order instead.
  wording           `ac_select.pick_wording`: clustering by validation AMI with r_E,clu; retrieval by the cross-fit mAP at each
                    wording's OWN pick with r_E,mAP; the photo wording only when better by more than 0.001 r.
  r_E               max minus min over the photo model's five final runs, EmoSet validation: emotion AMI (20 clusters, mean of
                    three states; the stored float16 rows, as r12 clusters c*) and emotion mAP at R0 (no nDCG: EmoSet has
                    one label per photo). `ac_stats.measure_seed_ranges` needs a label distribution, so its two lines are
                    repeated in `seed_ranges`.

Steps in order (`stage2_record`): 1 r_E; 2 read-outs of the photo model, plain Qwen and both wordings; 3 wording per task;
4 read-outs of the descriptive arms. The record is a plain dict (unrounded floats); ticket 12 commits it as
`RECORDS/stage2_emoset.json`. `run_stage3` refuses to start unless that file equals its content in git HEAD
(r13's `pf_run.require_records` pattern), before any key, label or feature is read.

Arms (names are the exports' and the cells'):
  bar arms          photo_qwen (r13's five, seeded), plain_qwen, teacher_emotion_focus (artwork wording),
                    teacher_emotion_focus_photo (photo wording)
  descriptive arms  cstar_qwen (frozen c*, seeded), crl_clip_l14 (r12's Affection pick, CLIP L/14, D10),
                    photo_siglip2 / photo_siglip1 / photo_clip_l14 (seeded), frozen_siglip2 / frozen_siglip1 / frozen_clip_l14

Cells (index format, `OUT/stage3/cells_emoset.json`; encoder 'qwen', aspect 'emotion', data 'emoset_test'):
  SC7  photo AMI against plain Qwen                          task clu, r null (WIN bar)
  SC8  photo mAP at its pick against plain Qwen at its pick  task ret, r null (WIN bar)
  SC9  photo AMI against the teacher, clustering wording     task clu, r = r_E,clu
  SC10 photo mAP against the teacher, retrieval wording      task ret, r = r_E,mAP
The sheet (`OUT/stage3/sheet_emoset.json`): one row per arm, bar and descriptive: AMI with 20 clusters and with 8 (K = 8), mAP at
R0 and at the picked read-out. No cell, no interval.

CLI (`python bs_emoset_score.py -h`; the controller runs it, CPU):
  stage2 [--write-record]   -> OUT/stage2/stage2_emoset.json (+ RECORDS/stage2_emoset.json with --write-record)
  stage3                    -> OUT/stage3/cells_emoset.json, sheet_emoset.json (needs the committed record)
  gate0xi [--recompute]     -> OUT/stage0/gate_0xi.json (r12's validation outputs; no EmoSet data)
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bs_paths as BP  # noqa: E402
import bs_emoset  # noqa: E402

if str(BP.TEST / '20261031_photo_finetune') not in sys.path:     # this checkout's copy of r13 (puts r12, r11, r10 on the path)
    sys.path.insert(0, str(BP.TEST / '20261031_photo_finetune'))
import pf_paths  # noqa: E402,F401  (asserts AC_SMOKE / AC_OUTPUTS / PF_OUTPUTS are not redirected)
import ac_cluster  # noqa: E402
import ac_select  # noqa: E402
import ac_stats  # noqa: E402
import rs_vh  # noqa: E402
import rt_metrics  # noqa: E402
import rt_readout  # noqa: E402
import rt_stats  # noqa: E402

SEEDS = tuple(BP.SEEDS)
N_CLU, N_RET = 2000, 10000                                  # r12's paired draws
K_ROW = 8                                                   # D10: the class count, as a plain row
BAR_ARMS = ('photo_qwen', 'plain_qwen', 'teacher_emotion_focus', 'teacher_emotion_focus_photo')
DESCRIPTIVE_ARMS = ('cstar_qwen', 'crl_clip_l14', 'photo_siglip2', 'photo_siglip1', 'photo_clip_l14', 'frozen_siglip2',
                    'frozen_siglip1', 'frozen_clip_l14')
SEEDED = frozenset({'photo_qwen', 'cstar_qwen', 'photo_siglip2', 'photo_siglip1', 'photo_clip_l14'})
FROZEN_VIEW = {'plain_qwen': 'qwen_default', 'teacher_emotion_focus': 'emotion_focus',
               'teacher_emotion_focus_photo': 'emotion_focus_photo', 'frozen_siglip2': 'siglip2',
               'frozen_siglip1': 'siglip1', 'frozen_clip_l14': 'clip_l14', 'crl_clip_l14': 'clip_l14'}
WORDING_VIEW = {'artwork': 'emotion_focus', 'photo': 'emotion_focus_photo'}     # r12's names for the two wordings
TEACHER_ARM = {w: f'teacher_{v}' for w, v in WORDING_VIEW.items()}
EXPORT_NAMES = tuple(f'{sp}__s{s}' for sp in ('val', 'test') for s in SEEDS)
CELLS_NAME, SHEET_NAME = 'cells_emoset.json', 'sheet_emoset.json'
CELL_KEYS = ('sc', 'encoder', 'aspect', 'task', 'data', 'candidate', 'comparator', 'candidate_value',
             'comparator_value', 'd', 'lo', 'hi', 'r', 'r_source', 'n_draws', 'seeds')
R_SOURCE = {'clu': "r_E,clu: max - min over the photo model's five final runs, EmoSet validation emotion AMI "
                   "(20 clusters, mean of three states)",
            'ret': "r_E,mAP: max - min over the photo model's five final runs, EmoSet validation emotion mAP at R0"}
TOL_GATE = 1e-9                                             # 0(xi)
R12_PICKS = pf_paths.R12_OUT / 'stage1/picks.json'
R12_SCORES = pf_paths.R12_OUT / 'stage1/scores.json'
R12_RANGES = pf_paths.R12_OUT / 'stage0b/ranges.json'
STAGE2_NAME = 'stage2_emoset.json'


def record_file():
    return BP.RECORDS / STAGE2_NAME


# ============================================================================================ boundaries
def check_order(keys):
    """The rows are in D17's order: unique keys, sorted by the sha256 of the key string. r12's sorted-key order would
    group classes in the parity halves."""
    keys = list(keys)
    assert keys and keys == bs_emoset.order(keys), 'rows must be in bs_emoset.order (sha256) order'


def keys_sha(keys):
    return hashlib.sha256('\n'.join(keys).encode()).hexdigest()


def encode_labels(names):
    """int64 codes of the class names: position in `bs_paths.EMOSET_CLASSES` (alphabetical)."""
    codes = []
    for n in names:
        assert n in BP.EMOSET_CLASSES, f'unknown class {n!r}'
        codes.append(BP.EMOSET_CLASSES.index(n))
    return np.asarray(codes, dtype=np.int64)


def unit(x):
    """float32 unit rows from stored rows (r12's `ac_data._unit`: cast to float32, divide by the row norm)."""
    return bs_emoset.unit(x)


def _check_labels(labels, n):
    labels = np.asarray(labels)
    assert labels.dtype == np.int64 and labels.shape == (n,), (labels.dtype, labels.shape, n)
    assert set(np.unique(labels)) == set(range(len(BP.EMOSET_CLASSES))), 'every class must be present'
    return labels


def _check_stored(x, n, name):
    assert x.dtype == np.float16 and x.ndim == 2 and len(x) == n and np.isfinite(x).all(), (name, x.dtype, x.shape)
    return x


# ============================================================================================ sources
class RealSource:
    """The real data: cut sets and labels of `bs_emoset`, frozen views of `bs_stores`, exports of ticket 04
    (`OUT/exports/emoset/<arm>/<split>__s<seed>.npz`). Every row is joined by key."""

    def keys(self, split):
        return bs_emoset.keys(split)

    def labels(self, keys):
        return encode_labels(bs_emoset.labels(keys))

    def _export(self, arm, split, seed):
        return Path(BP.OUT) / 'exports/emoset' / arm / f'{split}__s{seed}.npz'

    def stored(self, arm, split, seed, keys):
        """Checked float16 rows of an emotion-pathway export joined to `keys` by key: the sidecar's sha256 equals the
        file's, float16 2-D finite rows, unit norm within 1e-3."""
        assert arm in SEEDED, arm
        path = self._export(arm, split, seed)
        side = json.loads(path.with_suffix('.json').read_text())
        assert side['sha256'] == rs_vh.file_sha(path), f'{path}: file differs from its sidecar'
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == {'keys', 'emb'}, (path, z.files)
            file_keys, emb = z['keys'].tolist(), z['emb']
        assert emb.dtype == np.float16 and emb.ndim == 2 and len(emb) == len(file_keys) and np.isfinite(emb).all(), path
        assert len(set(file_keys)) == len(file_keys), f'{path}: duplicate keys'
        lookup = {k: i for i, k in enumerate(file_keys)}
        keys = list(keys)
        missing = [k for k in keys if k not in lookup]
        assert not missing, f'{path}: {len(missing)} keys not in the file, e.g. {missing[:2]}'
        out = emb[np.fromiter((lookup[k] for k in keys), dtype=np.int64, count=len(keys))]
        assert np.abs(np.linalg.norm(out.astype(np.float32), axis=1) - 1).max() < 1e-3, f'{path}: rows are not unit'
        return out

    def frozen(self, arm, keys):
        """Unit float32 rows of a frozen arm. `crl_clip_l14` is r12's CRL vector of the CLIP L/14 image, painting
        wording (`ac_data._crl`, the function that built r12's `crl_clip_painting` arm)."""
        import bs_stores
        x = unit(bs_stores.load('emoset', FROZEN_VIEW[arm], keys))
        if arm == 'crl_clip_l14':
            import ac_data
            return ac_data._crl(x, 'clip', 'painting', 'emotion')
        return x

    def export_sha(self, arm):
        out = {}
        for name in EXPORT_NAMES:
            side = self._export(arm, name.split('__s')[0], int(name.split('__s')[1])).with_suffix('.json')
            if side.exists():
                out[name] = json.loads(side.read_text())['sha256']
        return out

    def descriptive_available(self):
        """(arms that exist, {arm: reason} for those that do not). A seeded arm needs all ten export files; a frozen arm
        its store. A missing descriptive arm is reported, not an error (D18: an encoder may have failed)."""
        import bs_stores
        have, missing = [], {}
        for arm in DESCRIPTIVE_ARMS:
            if arm in SEEDED:
                got = self.export_sha(arm)
                (have.append(arm) if set(got) == set(EXPORT_NAMES) else missing.update({arm: f'{len(got)} of 10 exports'}))
            else:
                folder = bs_stores.folders('emoset', FROZEN_VIEW[arm])[0]
                (have.append(arm) if Path(folder).exists() else missing.update({arm: f'no store {folder}'}))
        if 'crl_clip_l14' in have:
            picks = json.loads(R12_PICKS.read_text())['affection']
            assert picks['crl_version'] == 'clip_painting', f"r12's CRL pick is {picks['crl_version']}, D10 carries clip_painting"
        return have, missing


# ============================================================================================ clustering and retrieval pieces
def partitions_k(x, k):
    """r10's MiniBatchKMeans partitions with k clusters: states 0, 1, 2, n_init 3, batch 4096, one thread, float32, unit
    norm checked (`rs_vh.r20_partition` with k as an argument). int32 [3, n]. k = 20 equals `ac_cluster.partitions` (tested)."""
    x = np.asarray(x).astype(np.float32)
    assert np.isfinite(x).all() and np.abs(np.linalg.norm(x, axis=1) - 1).max() < rs_vh.UNIT_TOL, 'rows are not unit norm'
    out = []
    with threadpool_limits(limits=1):
        for state in rs_vh.STATES:
            km = MiniBatchKMeans(n_clusters=k, random_state=state, n_init=rs_vh.N_INIT, batch_size=rs_vh.BATCH)
            out.append(km.fit(x).labels_)
    return np.stack(out).astype(np.int32)


_SHARED = {}


def _fit_task(i):
    """One fit in a forked worker: (array, k, cache name or None, cache root)."""
    x, k, name, root = _SHARED['tasks'][i]
    with threadpool_limits(1):
        if k == ac_cluster.K and name is not None:
            return ac_cluster.cached_partitions(x, name, root)
        return partitions_k(x, k)


def fit_partitions(tasks, workers):
    """Run `tasks` [(x, k, cache_name, cache_root)] in forked workers; the arrays reach the workers through the fork."""
    _SHARED['tasks'] = tasks
    try:
        if workers <= 1:
            return [_fit_task(i) for i in range(len(tasks))]
        with mp.get_context('fork').Pool(min(workers, len(tasks))) as pool:
            return pool.map(_fit_task, range(len(tasks)), chunksize=1)
    finally:
        _SHARED.clear()


def range_entry(per_seed):
    per_seed = [float(v) for v in per_seed]
    r = ac_stats.seed_range(per_seed)
    assert len(per_seed) == 5 and np.isfinite(r) and r > 0, f'seed range {r} must be finite and positive'
    return dict(per_seed=per_seed, r=r)


def seed_ranges(stored, labels, workers=8):
    """Step 1: r_E,clu and r_E,mAP from the photo model's five validation exports. stored: {seed: float16 [n, d]} in the
    sha256 order. Per seed: the mean over three states of the emotion AMI of the 20-cluster partitions of the stored rows;
    mAP of plain cosine (R0) of the unit rows, every photo a query. Returns {'clu': {per_seed, r}, 'map': {per_seed, r}}."""
    assert sorted(stored) == list(SEEDS), 'five runs (seeds 0 to 4) are required'
    n = len(labels)
    labels = _check_labels(labels, n)
    clu, mp_ = [], []
    for s in SEEDS:
        st = _check_stored(stored[s], n, f'seed {s}')
        clu.append(ac_cluster.ami(ac_cluster.partitions(st), labels))
        res = rt_metrics.retrieval(unit(st), labels, qe_k=0, workers=workers)
        assert res['dropped'] == 0 and len(res['rows']) == n, 'every validation photo must be a query'
        mp_.append(float(res['ap'].mean()))
    return dict(clu=range_entry(clu), map=range_entry(mp_))


def readout_doc(xs, labels, r_map, workers=8):
    """One arm's validation read-out table and pick (`ac_select.arm_table` with r_E,mAP): the cross-fit halves are the
    parity of the position in `xs`' rows, which must be in the sha256 order (the caller checks it). xs: {seed or None:
    unit float32 [n, d]}. Returns {'pick': 'center+k10', 'scores': {name: {'map': {'emotion': f}, 'score': f, 'kept': b}}}."""
    assert np.isfinite(r_map) and r_map > 0
    table, pick = ac_select.arm_table({('emotion', s): x for s, x in xs.items()}, {'emotion': labels},
                                      {'emotion': r_map}, workers=workers)
    return dict(pick=rt_readout.name(pick), scores=table)


def map_at_pick(doc):
    return float(doc['scores'][doc['pick']]['map']['emotion'])


def wording_picks(ami, docs, r_clu, r_map):
    """Step 3, r12's wording rule (`ac_select.pick_wording`), per task. ami: {'artwork': f, 'photo': f}, the validation AMI
    of the two wordings; docs: {'artwork': readout_doc, 'photo': readout_doc}, each wording's own table and pick; the retrieval
    score of a wording is its cross-fit mAP at ITS OWN pick. The photo wording is picked only when better by more than
    0.001 r (r_E,clu for clustering, r_E,mAP for retrieval)."""
    assert set(ami) == set(docs) == {'artwork', 'photo'}
    at_pick = {w: map_at_pick(docs[w]) for w in ('artwork', 'photo')}
    return dict(clustering=ac_select.pick_wording(ami['artwork'], ami['photo'], r_clu),
                retrieval=ac_select.pick_wording(at_pick['artwork'], at_pick['photo'], r_map),
                ami={w: float(ami[w]) for w in ami}, map_at_own_pick=at_pick,
                readout={w: docs[w]['pick'] for w in docs})


def test_retrieval(val, test, readout, labels, seeded, workers=8):
    """Per-query AP of one arm on the test photos under its read-out (`ac_stats.arm_retrieval`): the map W is fitted on ALL
    validation rows (cut set, no labels) and applied to the test rows; query expansion runs on the test gallery.
    val, test: {seed or None: unit float32 rows}."""
    res = ac_stats.arm_retrieval(lambda s: val[s], lambda s: test[s], readout, np.asarray(labels),
                                 seeds=SEEDS if seeded else None, workers=workers)
    assert res['dropped'] == 0, (readout, 'queries without a relevant item', res['dropped'])
    return res


# ============================================================================================ stage 2
def _val_inputs(source, arm, keys):
    """{seed or None: unit float32 rows} of an arm on the validation keys."""
    if arm in SEEDED:
        return {s: unit(source.stored(arm, 'val', s, keys)) for s in SEEDS}
    return {None: source.frozen(arm, keys)}


def stage2_record(source, workers=8):
    """Stage 2 (development only, EmoSet validation, cut set): §3.4's steps 1 to 4 in order. Returns the record (plain dict,
    unrounded floats): sets, ranges, bar arms' read-outs, teacher wording, descriptive read-outs, export hashes."""
    val_keys = source.keys('val')
    check_order(val_keys)
    test_keys = source.keys('test')                         # the key list only (no label, no feature): recorded for stage 3's check
    check_order(test_keys)
    labels = _check_labels(source.labels(val_keys), len(val_keys))
    sets = {sp: dict(n=len(k), keys_sha256=keys_sha(k)) for sp, k in (('val', val_keys), ('test', test_keys))}
    # step 1: r_E
    stored = {s: source.stored('photo_qwen', 'val', s, val_keys) for s in SEEDS}
    ranges = seed_ranges(stored, labels, workers=workers)
    r_clu, r_map = ranges['clu']['r'], ranges['map']['r']
    # step 2: the bar arms' read-outs by r12's rule with r_E,mAP
    readout = {}
    for arm in BAR_ARMS:
        xs = {s: unit(stored[s]) for s in SEEDS} if arm == 'photo_qwen' else _val_inputs(source, arm, val_keys)
        readout[arm] = readout_doc(xs, labels, r_map, workers)
    # step 3: the wording per task
    ami = {w: float(ac_cluster.ami(ac_cluster.partitions(source.frozen(TEACHER_ARM[w], val_keys)), labels))
           for w in ('artwork', 'photo')}
    wording = wording_picks(ami, {w: readout[TEACHER_ARM[w]] for w in ('artwork', 'photo')}, r_clu, r_map)
    # step 4: the descriptive arms' read-outs, same rule, same r_E,mAP
    have, not_run = source.descriptive_available()
    assert set(have) | set(not_run) == set(DESCRIPTIVE_ARMS) and not set(have) & set(not_run)
    descriptive = {arm: readout_doc(_val_inputs(source, arm, val_keys), labels, r_map, workers) for arm in DESCRIPTIVE_ARMS if arm in have}
    exports = {}
    for arm in sorted(SEEDED & (set(BAR_ARMS) | set(descriptive))):
        exports[arm] = source.export_sha(arm)
        assert set(exports[arm]) == set(EXPORT_NAMES), f'{arm}: exports {sorted(exports[arm])}'
    rec = dict(kind='emoset_stage2', sets=sets, ranges=ranges, readout=readout,
               teacher=dict(ami=wording['ami'], map_at_own_pick=wording['map_at_own_pick'], readout=wording['readout'],
                            views=dict(WORDING_VIEW)),
               wording=dict(clustering=wording['clustering'], retrieval=wording['retrieval']),
               descriptive=descriptive, descriptive_not_run=not_run, exports=exports)
    rec = json.loads(json.dumps(rec, allow_nan=False))
    check_record(rec)
    return rec


def check_record(rec):
    """Re-derive what the record says: the ranges are max - min over five runs (positive); every table's scores follow from
    its mAPs and r_E,mAP and its pick from its table (r12's rule); the wording picks follow from the stored AMI and tables."""
    assert rec['kind'] == 'emoset_stage2'
    for name in ('clu', 'map'):
        e = rec['ranges'][name]
        assert len(e['per_seed']) == 5 and e['r'] == ac_stats.seed_range(e['per_seed']) and e['r'] > 0, ('range', name)
    r_clu, r_map = rec['ranges']['clu']['r'], rec['ranges']['map']['r']
    assert set(rec['readout']) == set(BAR_ARMS) and set(rec['descriptive']) <= set(DESCRIPTIVE_ARMS)
    for arm, doc in {**rec['readout'], **rec['descriptive']}.items():
        assert set(doc) == {'pick', 'scores'} and doc['pick'] in doc['scores'], arm
        ac_select._check_table(doc['scores'], {'emotion': r_map}, ('emotion',),
                               pick=ac_select.readout_from_name(doc['pick']))
    docs = {w: rec['readout'][TEACHER_ARM[w]] for w in ('artwork', 'photo')}
    want = wording_picks(rec['teacher']['ami'], docs, r_clu, r_map)
    assert rec['wording'] == dict(clustering=want['clustering'], retrieval=want['retrieval']), 'wording picks differ from the rule'
    assert rec['teacher']['map_at_own_pick'] == want['map_at_own_pick'] and rec['teacher']['readout'] == want['readout']
    assert set(rec['sets']) == {'val', 'test'} and set(rec['descriptive_not_run']) | set(rec['descriptive']) == set(DESCRIPTIVE_ARMS)
    return rec


# ============================================================================================ the held-data guard
def guard_reason(f):
    """'' when the record file `f` passes, else the reason it does not: it exists and (not in a smoke run) its bytes equal
    `git show HEAD:<file>` (r13's `pf_run.guard_reason`)."""
    f = Path(f)
    if not f.is_file():
        return f'{f} does not exist'
    if BP.SMOKE > 0:
        print(f'guard: smoke run, git check of {f} skipped', flush=True)
        return ''
    r = subprocess.run(['git', '-C', str(f.parent), 'show', f'HEAD:./{f.name}'], capture_output=True)
    if r.returncode:
        return f'{f} is not in git HEAD: commit it first'
    if r.stdout != f.read_bytes():
        return f'{f} differs from its content in git HEAD: commit it first'
    return ''


def require_record():
    """Test scoring needs the committed stage 2 record. SystemExit (nonzero) before any data is read."""
    why = guard_reason(record_file())
    if why:
        raise SystemExit(f'REFUSED: stage 3: stage 2 record: {why}')


# ============================================================================================ stage 3
def make_cell(sc, task, candidate, comparator, candidate_value, comparator_value, d, interval, r, r_source, n_draws, **extra):
    """One cell of `cells_emoset` (index format). r: the unrounded LEVEL unit, None for a WIN-only bar (SC7, SC8)."""
    lo, hi = interval
    assert sc in ('SC7', 'SC8', 'SC9', 'SC10') and task in ('clu', 'ret')
    assert all(np.isfinite(v) for v in (candidate_value, comparator_value, d, lo, hi)), sc
    assert (r is None) == (sc in ('SC7', 'SC8')), 'SC9 and SC10 carry the LEVEL unit; SC7 and SC8 are WIN-only'
    assert r is None or (np.isfinite(r) and r > 0)
    cell = dict(sc=sc, encoder='qwen', aspect='emotion', task=task, data='emoset_test', candidate=candidate,
                comparator=comparator, candidate_value=float(candidate_value), comparator_value=float(comparator_value),
                d=float(d), lo=float(lo), hi=float(hi), r=None if r is None else float(r), r_source=r_source,
                n_draws=int(n_draws), seeds=list(SEEDS), **extra)
    assert set(CELL_KEYS) <= set(cell)
    return cell


def write_cells(path, cells):
    """Atomic write of the four cells of part c (all or nothing)."""
    assert [c['sc'] for c in cells] == ['SC7', 'SC8', 'SC9', 'SC10'], 'part c has exactly SC7 to SC10'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(cells, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def _test_inputs(source, arm, val_keys, test_keys):
    """(stored float16 test rows {seed: ...} or None, unit float32 val rows, unit float32 test rows), each {seed or None: ...}."""
    if arm in SEEDED:
        st = {s: source.stored(arm, 'test', s, test_keys) for s in SEEDS}
        return st, {s: unit(source.stored(arm, 'val', s, val_keys)) for s in SEEDS}, {s: unit(st[s]) for s in SEEDS}
    return None, {None: source.frozen(arm, val_keys)}, {None: source.frozen(arm, test_keys)}


def score_arm(source, arm, pick, val_keys, test_keys, labels, cache_root, workers):
    """Everything stage 3 needs of one arm on the test photos: the 20-cluster partitions (cached) and the 8-cluster ones, mean
    AMI of each, per-query AP at R0 and at the committed pick. Clustering input: the stored float16 rows of a seeded arm (as
    r13 clusters its photo model), the unit float32 rows of a frozen one (as r12)."""
    stored, val, test = _test_inputs(source, arm, val_keys, test_keys)
    seeded = arm in SEEDED
    seeds = SEEDS if seeded else (None,)
    cluster_in = {s: (stored[s] if seeded else test[s]) for s in seeds}
    tasks = [(cluster_in[s], ac_cluster.K, f'emoset__test__{arm}__s{"-" if s is None else s}', cache_root) for s in seeds]
    tasks += [(cluster_in[s], K_ROW, None, None) for s in seeds]
    fitted = fit_partitions(tasks, workers)
    p20, p8 = fitted[:len(seeds)], fitted[len(seeds):]
    parts20 = np.stack(p20) if seeded else p20[0]
    parts8 = np.stack(p8) if seeded else p8[0]
    picked = test_retrieval(val, test, pick, labels, seeded, workers)
    r0 = picked if pick == 'none+k0' else test_retrieval(val, test, 'none+k0', labels, seeded, workers)
    return dict(arm=arm, seeded=seeded, parts20=parts20, parts8=parts8, picked=picked, r0=r0, pick=pick,
                ami_k20=ac_cluster.ami(parts20, labels), ami_k8=ac_cluster.ami(parts8, labels))


def sheet_row(res, role):
    per_seed = lambda v: None if v is None else [float(x) for x in v]
    return dict(arm=res['arm'], role=role, seeded=res['seeded'], readout=res['pick'], ami_k20=float(res['ami_k20']),
                ami_k8=float(res['ami_k8']), map_r0=float(res['r0']['point']), map_picked=float(res['picked']['point']),
                map_r0_per_seed=per_seed(res['r0']['per_seed']), map_picked_per_seed=per_seed(res['picked']['per_seed']))


def stage3(source, rec, cache_root, workers=8, n_clu=N_CLU, n_ret=N_RET, log=print):
    """SC7 to SC10 on the test photos and the sheet, from a stage 2 record. `rec` must be the committed record (the caller
    checked the guard). It is re-derived (`check_record`) and checked against the data on hand: the cut sets' key hashes and
    the exports' sha256 must be the ones stage 2 used."""
    check_record(rec)
    val_keys, test_keys = source.keys('val'), source.keys('test')
    check_order(val_keys)
    check_order(test_keys)
    for sp, k in (('val', val_keys), ('test', test_keys)):
        assert rec['sets'][sp] == dict(n=len(k), keys_sha256=keys_sha(k)), f'{sp}: the cut set differs from the record'
    for arm, want in rec['exports'].items():
        assert source.export_sha(arm) == want, f'{arm}: exports differ from the record'
    labels = _check_labels(source.labels(test_keys), len(test_keys))
    picks = {**{a: d['pick'] for a, d in rec['readout'].items()}, **{a: d['pick'] for a, d in rec['descriptive'].items()}}
    res, rows = {}, []
    for arm in list(BAR_ARMS) + list(rec['descriptive']):
        t0 = time.perf_counter()
        res[arm] = score_arm(source, arm, picks[arm], val_keys, test_keys, labels, cache_root, workers)
        rows.append(sheet_row(res[arm], 'bar' if arm in BAR_ARMS else 'descriptive'))
        log(f'  stage3 {arm}: pick {picks[arm]}, AMI20 {res[arm]["ami_k20"]:.4f}, mAP {res[arm]["picked"]["point"]:.4f}, '
            f'{time.perf_counter() - t0:.0f}s')
    ph = res['photo_qwen']
    w_clu, w_ret = (TEACHER_ARM[rec['wording'][t]] for t in ('clustering', 'retrieval'))
    r_clu, r_map = rec['ranges']['clu']['r'], rec['ranges']['map']['r']
    cells = []
    for sc, comp, r, r_src in (('SC7', 'plain_qwen', None, None), ('SC9', w_clu, r_clu, R_SOURCE['clu'])):
        b = ac_cluster.paired(ph['parts20'], res[comp]['parts20'], labels, n_clu, workers)
        cells.append(make_cell(sc, 'clu', 'photo_qwen', comp, b['A_cand'], b['A_comp'], b['d'], b['interval'], r, r_src, n_clu))
    for sc, comp, r, r_src in (('SC8', 'plain_qwen', None, None), ('SC10', w_ret, r_map, R_SOURCE['ret'])):
        c = res[comp]['picked']
        assert np.array_equal(ph['picked']['rows'], c['rows']), (sc, 'the two arms were scored on different queries')
        p = rt_stats.paired(ph['picked']['ap'], c['ap'], n_ret)
        cells.append(make_cell(sc, 'ret', 'photo_qwen', comp, ph['picked']['point'], c['point'], p['d'], p['interval'], r, r_src,
                               n_ret, readout_candidate=ph['pick'], readout_comparator=res[comp]['pick']))
    cells.sort(key=lambda c: ('SC7', 'SC8', 'SC9', 'SC10').index(c['sc']))
    return dict(cells=json.loads(json.dumps(cells, allow_nan=False)), sheet=json.loads(json.dumps(rows, allow_nan=False)))


def run_stage3(source, out, workers=8, n_clu=N_CLU, n_ret=N_RET):
    """The stage 3 entry: the guard first (before any key, label or feature is read), then `stage3`, then the two files."""
    require_record()
    rec = json.loads(record_file().read_text())
    out = Path(out)
    doc = stage3(source, rec, out / 'stage3/partitions_emoset', workers, n_clu, n_ret)
    write_cells(out / 'stage3' / CELLS_NAME, doc['cells'])
    sheet_path = out / 'stage3' / SHEET_NAME
    sheet_path.write_text(json.dumps(dict(note='descriptive rows and K = 8 (D10): no interval, no cell, no outcome',
                                          n_test=rec['sets']['test']['n'], rows=doc['sheet']), indent=1, allow_nan=False) + '\n')
    return doc


# ============================================================================================ gate 0(xi)
def gate_xi_from_documents(ranges, tables, picks, scores):
    """0(xi): r12's wording rule, fed r12's validation outputs, gives r12's wording picks and scores within 1e-9. ranges: r12's
    `stage0b/ranges.json`; tables: {'teacher_artwork': table, 'teacher_photo': table} (r12's `stage1/scores.json`
    readout.affection); picks: r12's `picks['affection']`; scores: r12's scores.json (`teacher_wording`). The two tables' picks
    are re-derived from the tables by r12's rule before they are used. Returns {'ok', 'items': {name: {value, reference, ok}}}."""
    items = {}

    def item(name, value, ref, exact=False):
        ok = value == ref if exact else abs(value - ref) <= TOL_GATE
        items[name] = dict(value=value, reference=ref, ok=bool(ok))

    r_clu, r_map = float(ranges['clu']['r']), float(ranges['map']['r'])
    docs = {}
    for w in ('artwork', 'photo'):
        arm = f'teacher_{w}'
        table = tables[arm]
        ac_select._check_table(table, {'emotion': r_map}, ('emotion',))
        rows = {ac_select.readout_from_name(n): row['map'] for n, row in table.items()}
        pick = rt_readout.name(ac_select.pick_readout(rows, {'emotion': r_map}))
        item(f'pick.{arm}', pick, picks['readout'][arm], exact=True)
        docs[w] = dict(pick=picks['readout'][arm], scores=table)
    ami = {w: float(scores['teacher_wording']['clustering'][w]) for w in ('artwork', 'photo')}
    got = wording_picks(ami, docs, r_clu, r_map)
    for task in ('clustering', 'retrieval'):
        item(f'wording.{task}', got[task], picks['teacher_wording'][task], exact=True)
    for w in ('artwork', 'photo'):
        item(f'score.clustering.{w}', got['ami'][w], scores['teacher_wording']['clustering'][w])
        item(f'score.retrieval.{w}', got['map_at_own_pick'][w], scores['teacher_wording']['retrieval'][w])
    return dict(ok=all(i['ok'] for i in items.values()), items=items)


def gate_xi(recompute=False, workers=8):
    """0(xi) on r12's stored validation outputs (no EmoSet data). With `recompute`, the two teacher wordings' validation
    tables and AMI are also rebuilt from r12's validation features by this module's own functions (`readout_doc`,
    `ac_cluster.ami`, with r12's sorted-key order and r12's r_ret,mAP) and compared with r12's stored numbers within 1e-9."""
    ranges = ac_stats.load_ranges(R12_RANGES)
    picks = json.loads(R12_PICKS.read_text())['affection']
    scores = json.loads(R12_SCORES.read_text())
    tables = {a: scores['readout']['affection'][a] for a in ('teacher_artwork', 'teacher_photo')}
    res = gate_xi_from_documents(ranges, tables, picks, scores)
    if recompute:
        import ac_data
        keys, labels = ac_data.keys('val'), ac_data.labels('val')
        assert keys == sorted(keys), 'r12 validation keys are sorted'
        for w in ('artwork', 'photo'):
            arm = f'teacher_{w}'
            x = ac_data.embed(arm, 'val')
            doc = readout_doc({None: x}, labels, float(ranges['map']['r']), workers)
            ami = float(ac_cluster.ami(ac_cluster.partitions(x), labels))
            res['items'][f'recompute.pick.{arm}'] = dict(value=doc['pick'], reference=picks['readout'][arm], ok=doc['pick'] == picks['readout'][arm])
            res['items'][f'recompute.ami.{w}'] = dict(value=ami, reference=scores['teacher_wording']['clustering'][w],
                                                      ok=abs(ami - scores['teacher_wording']['clustering'][w]) <= TOL_GATE)
            for name, row in tables[arm].items():
                v, ref = doc['scores'][name]['map']['emotion'], row['map']['emotion']
                res['items'][f'recompute.{arm}.{name}'] = dict(value=v, reference=ref, ok=abs(v - ref) <= TOL_GATE)
        res['ok'] = all(i['ok'] for i in res['items'].values())
    return res


# ============================================================================================ CLI
def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.pending.json')
    tmp.write_text(json.dumps(value, indent=1, allow_nan=False) + '\n')
    tmp.replace(path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = p.add_subparsers(dest='cmd', required=True)
    s2 = sub.add_parser('stage2', help='r_E, read-out picks, wording picks -> OUT/stage2/stage2_emoset.json')
    s2.add_argument('--write-record', action='store_true', help='also write RECORDS/stage2_emoset.json (the controller commits it)')
    s3 = sub.add_parser('stage3', help='SC7 to SC10 cells and the sheet (needs the committed record)')
    s3.add_argument('--n-clu', type=int, default=N_CLU)
    s3.add_argument('--n-ret', type=int, default=N_RET)
    g = sub.add_parser('gate0xi', help="0(xi): r12's wording rule on r12's validation outputs")
    g.add_argument('--recompute', action='store_true')
    args = p.parse_args(argv)
    t0 = time.monotonic()
    out = Path(BP.OUT)
    if args.cmd == 'stage2':
        rec = stage2_record(RealSource(), workers=args.workers)
        text = json.dumps(rec, indent=1, allow_nan=False) + '\n'
        _write_json(out / 'stage2' / STAGE2_NAME, rec)
        if args.write_record:
            f = record_file()
            assert BP.SMOKE == 0, 'a smoke run never writes the committed record'
            assert not f.exists() or f.read_text() == text, f'{f} holds another record: picks are frozen once written'
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
        print(f"r_E,clu {rec['ranges']['clu']['r']:.10g} r_E,mAP {rec['ranges']['map']['r']:.10g} photo pick "
              f"{rec['readout']['photo_qwen']['pick']} wording {rec['wording']} -> {out / 'stage2' / STAGE2_NAME} "
              f"({time.monotonic() - t0:.0f}s)")
        return 0
    if args.cmd == 'stage3':
        doc = run_stage3(RealSource(), out, args.workers, args.n_clu, args.n_ret)
        for c in doc['cells']:
            print(f"{c['sc']}: d {c['d']:.5f} [{c['lo']:.5f}, {c['hi']:.5f}] vs {c['comparator']}")
        print(f'{len(doc["cells"])} cells, {len(doc["sheet"])} sheet rows -> {out / "stage3"} ({time.monotonic() - t0:.0f}s)')
        return 0
    res = gate_xi(args.recompute, args.workers)
    _write_json(out / 'stage0' / 'gate_0xi.json', dict(gate='0(xi)', recompute=args.recompute, **res))
    print(f'0(xi): {"PASS" if res["ok"] else "FAIL"} ({len(res["items"])} items, {time.monotonic() - t0:.0f}s)')
    return 0 if res['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
