"""r29 ticket 06: image test cells of a new model on ArtELingo: AMI and image mAP, with their validation picks and ranges.

Spec docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md §3.4 (cells table rows 1, 2, 7; outcome rule; order
and C13), §3.5 gate (v) image part, D18 (each cell type keeps the protocol that made its paper cell), D23.

Three cell types per aspect (emotion 9, style 27, genre 10 classes), each with its own protocol (D18; never mixed):
  ami (vs plain, vs teacher)  r17 SC1 / SC3: the aspect head's stored float16 rows cast to float32, MiniBatchKMeans at the
                              class count (`ns_metrics.partitions`: n_init 3, batch 4096, states 0 to 2), 2,000 paired draws
                              (rows, then seeds), `ns_clu.paired_cell`. r = the candidate's five-seed range of validation AMI
                              at the class count (`ns_clu_cells.stage1_ranges`).
  map_plain (SC7 protocol)    r17 SC7: the "untold" candidate (three unit heads joined, divided by sqrt(3)) against the plain
                              image, each at its own validation-fitted read-out (`ns_ret._Rows.ap`, `ns_ret.paired_ret`,
                              10,000 draws, queries then seeds). Candidate read-out: r11's rule (`ns_picks.readout_doc`) in
                              units of the candidate's own validation R0 mAP range; plain read-out: r14's recorded pick.
  map_teacher (r14 protocol)  r14 `bs_artret`: the aspect head at r11's stage-4 read-out rule (own ranges), held per-query AP
                              through `rt_run.score_arm`, against r11's stored `instructed` per-query files
                              (`bs_artret.r11_stored`), `rt_stats.paired` (10,000 draws, queries and seeds).
  LEVEL unit r of the two mAP types: the candidate's own five-seed range of validation mAP at R0 per aspect
  (`validation_ranges`, `bs_artret.validation_ranges`'s computation).

Wiring: `Wiring(root, arm, encoder)` is a context manager that points r17's `ns_data.cstar` / `_head_id` and r14's
`bs_artret.cstar_embed` at a model root and arm (`mock.patch`; older files are not edited). Layout (r14's):
  held/<arm>/<aspect>__s<seed>.npz   {img}  float16 [n, 2048], sidecar .json with arm, aspect, seed, rows, list_sha256
  val/<arm>/<aspect>__s<seed>.npz    {img, cap}  sidecar: arm, aspect, seed, names_sha256   (ticket 03's validation export)
  train/<arm>/R6-5/<aspect>__<graph>__s<seed>.npz  r14's training-time validation file, used when val/<arm>/ has no file
                                      (K0: arm k11=40); sidecar config.{aspect, seed, names_sha256.dev, rg.arm}
Rows join by names: the sidecar's names hash is checked against r11's sorted painting names (C6), never positions.

Order rule (§3.4): `picks --arm A` reads validation only and writes `records/image_picks_<arm>.json`; `cells --arm A`
refuses without that file (and, outside smoke, unless it is committed unchanged in HEAD) and writes
`records/image_cells_<arm>.json`. `gate-v-image` replays r17's SC1, SC3, SC7 and r14's Q1 image cells on K0 (gate (v)).
"""
import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo
from datetime import datetime

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_paths as P  # noqa: E402

sys.path.insert(0, str(P.TEST / '20261103_new_scores'))      # r17 (this checkout's copy; it puts the older rounds on sys.path)
import ns_paths  # noqa: E402

ns_paths.add_rounds()
import bs_artret  # noqa: E402  (r14)
import ns_clu  # noqa: E402
import ns_clu_cells  # noqa: E402
import ns_data  # noqa: E402
import ns_picks  # noqa: E402
import ns_ret  # noqa: E402
import rs_vh  # noqa: E402
import rt_data  # noqa: E402
import rt_metrics  # noqa: E402
import rt_readout  # noqa: E402
import rt_stats  # noqa: E402

ASPECTS = tuple(P.ASPECTS)
SEEDS = tuple(P.SEEDS)
GRAPHS = rt_data.GRAPHS
TOL = 1e-9                                            # D23: replayed test cells
N_DRAWS = {'ami': 2000, 'map': 10000}                 # r17 SC1 / SC3; r17 SC7 and r14 Q1
CELL_TYPES = ('ami', 'map_plain', 'map_teacher')
SET = 'artelingo'
SPLIT_HELD = ns_paths.HELD_SPLIT[SET]                 # 'held'
AMI_RNG_SEED = ns_paths.BOOT_SEED                     # r17: a fresh default_rng(0) per cell
TZ = ZoneInfo('Europe/Amsterdam')
ENCODER_OF = {'siglip1': 'siglip1', 'siglip2': 'siglip2', 'clipl14': 'clip_l14'}   # caption source prefix -> image encoder


def now():
    return datetime.now(TZ).strftime('%F %H:%M')


def encoder_of_arm(arm):
    """Image encoder of an arm named after its caption source (`siglip1`, `siglip1_prefix`, `clipl14`, ...)."""
    base = arm.removesuffix('_prefix')
    assert base in ENCODER_OF, f'arm {arm!r}: not a caption source of siglip1, siglip2, clipl14'
    return ENCODER_OF[base]


def records_dir():
    """Committed records, or OUT/records under a smoke run (R29_OUTPUTS set): smoke never writes the committed ones."""
    return P.RECORDS if P.OUT == P.DEFAULT_OUT else P.OUT / 'records'


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ================================================================================================ outcome rule
def outcome(lo, r):
    """WIN when the interval's lower bound is above 0; LEVEL when not WIN and it is above -r; BELOW otherwise (spec §3.4;
    r11's `rt_stats.outcome`). r is the LEVEL unit: finite and not negative; a missing r crashes (no silent WIN-only)."""
    assert r is not None and np.isfinite(r) and r >= 0, f'LEVEL unit r must be finite and >= 0, got {r}'
    assert np.isfinite(lo), f'interval lower bound {lo}'
    if lo > 0:
        return 'WIN'
    if lo > -r:
        return 'LEVEL'
    return 'BELOW'


# ================================================================================================ wiring
class Wiring:
    """One model root and arm for the older scorers (module docstring). `splits`: the splits it may read (picks: ('val',),
    so a held row never reaches a pick). `names(split)`: the painting names the rows must follow (default: r11's sorted
    names through `ns_data.keys`; tests pass small lists)."""

    def __init__(self, root, arm, encoder, splits=('val', 'held'), names=None):
        assert encoder in ns_paths.ENCODERS and encoder != 'qwen', encoder
        assert set(splits) <= {'val', 'held'}, splits
        self.root, self.arm, self.encoder, self.splits = Path(root), arm, encoder, tuple(splits)
        self._names = names or (lambda split: ns_data.keys(SET, split))
        self._stack = None

    # --- reading
    def path(self, split, aspect, seed):
        if split == 'held':
            return self.root / f'held/{self.arm}/{aspect}__s{seed}.npz'
        p = self.root / f'val/{self.arm}/{aspect}__s{seed}.npz'
        if p.exists():
            return p
        q = self.root / f'train/{self.arm}/R6-5/{aspect}__{GRAPHS[aspect]}__s{seed}.npz'
        assert q.exists(), f'no validation file: neither {p} nor {q}'
        return q

    def _identity(self, side):
        """(arm, aspect, seed, names hash) of a sidecar, held or validation (both layouts)."""
        cfg = side.get('config')
        if cfg is not None:
            return cfg.get('rg', {}).get('arm', self.arm), cfg['aspect'], cfg['seed'], cfg['names_sha256']['dev']
        return side['arm'], side['aspect'], side['seed'], side.get('list_sha256', side.get('names_sha256'))

    def head(self, split, aspect, seed):
        """Stored float16 rows [n, 2048] of one head, joined by the sidecar's names hash to the sorted painting names."""
        assert split in ('val', 'held') and aspect in ASPECTS and seed in SEEDS, (split, aspect, seed)
        assert split in self.splits, f'this wiring reads {self.splits} only (validation picks never see held rows): {split!r}'
        path = self.path(split, aspect, seed)
        side = json.loads(path.with_suffix('.json').read_text())
        arm, a, s, digest = self._identity(side)
        names = list(self._names(split))
        assert (arm, a, s) == (self.arm, aspect, seed), f'{path}: sidecar says {(arm, a, s)}, asked {(self.arm, aspect, seed)}'
        assert digest == rs_vh.names_sha(names), f'{path}: sidecar names hash is not the {split} painting names in sorted order'
        with np.load(path, allow_pickle=False) as z:
            assert 'img' in z.files and set(z.files) <= {'img', 'cap'}, (path, z.files)
            x = z['img']
        assert x.dtype == np.float16 and x.shape == (len(names), ns_paths.CSTAR_WIDTH), (path, x.dtype, x.shape)
        assert np.isfinite(x).all(), f'{path}: non-finite rows'
        dev = float(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1).max())
        assert dev < ns_paths.TOL_UNIT, f'{path}: rows not unit norm (max deviation {dev})'
        return x

    # --- the scorers' entry points (signatures of the functions they replace)
    def _check(self, set_, enc):
        assert set_ == SET and enc == self.encoder, f'wired for {SET}/{self.encoder}, asked {set_}/{enc}'

    def cstar(self, set_, split, enc, aspect, seed, unit=True):
        """`ns_data.cstar` at this root: unit=True the retrieval form, unit=False the stored values cast to float32."""
        self._check(set_, enc)
        x16 = self.head(split, aspect, seed)
        x = ns_data._unit32(x16) if unit else np.asarray(x16).astype(np.float32)
        return ns_data._check_rows(x, len(x16), ns_paths.CSTAR_WIDTH, f'cstar {split}/{aspect}/s{seed}')

    def head_id(self, set_, split, enc, aspect, seed):
        """`ns_data._head_id`: the names hash of the head's sidecar (the three heads of `untold` must agree)."""
        self._check(set_, enc)
        self.head(split, aspect, seed)
        side = json.loads(self.path(split, aspect, seed).with_suffix('.json').read_text())
        return self._identity(side)[3]

    def embed(self, root, split, aspect, seed):
        """`bs_artret.cstar_embed` (r11's `rt_data.embed('cstar')`): unit float32 rows; `root` is ignored (this root)."""
        x = self.head(split, aspect, seed).astype(np.float32)
        return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)

    def __enter__(self):
        assert self._stack is None, 'wiring already active'
        self._stack = contextlib.ExitStack()
        for obj, name, fn in ((ns_data, 'cstar', self.cstar), (ns_data, '_head_id', self.head_id),
                              (bs_artret, 'cstar_embed', self.embed)):
            self._stack.enter_context(mock.patch.object(obj, name, fn))
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self._stack = None


# ================================================================================================ picks (validation only)
def validation_ranges(w, aspects=ASPECTS):
    """The candidate's validation mAP at R0 per aspect and seed (whole validation set, r11's stage 0a computation, as
    `bs_artret.validation_ranges`) and the seed ranges (max minus min over seeds 0 to 4, unrounded). Validation only."""
    labels = rt_data.labels('val')
    per_seed, files = {}, {}
    for a in aspects:
        per_seed[a], files[a] = {}, {}
        for s in SEEDS:
            x = w.embed(w.root, 'val', a, s)
            per_seed[a][str(s)] = float(rt_metrics.retrieval(x, labels[a])['ap'].mean())
            files[a][str(s)] = dict(path=str(w.path('val', a, s)), sha256=sha_file(w.path('val', a, s)))
    ranges = {}
    for a in aspects:
        ranges[a] = rt_stats.seed_range([per_seed[a][str(s)] for s in SEEDS])
        assert np.isfinite(ranges[a]) and ranges[a] > 0, (a, ranges[a])
    return dict(ranges=ranges, per_seed=per_seed, files=files, n_val=len(next(iter(labels.values()))))


def compute_picks(w, workers=8, tables_dir=None):
    """Everything the test cells need from validation, as the record's body (no held row is read: `w.splits` is ('val',)).
    ami_ranges   `ns_clu_cells.stage1_ranges` AMI per aspect: r, per_seed, k (r17).
    map_ranges   `validation_ranges`: R0 mAP per aspect and seed, r.
    readouts     untold: r11's rule on the three joined heads, units = map_ranges (as r17's `ns_picks`, whose units are
                 the told c*'s recorded range); cstar: r11's stage-4 rule for the aspect heads with own ranges (r14's
                 `pick_cstar`); plain: r14's recorded pick; teacher: r11's stored `instructed` pick."""
    assert w.splits == ('val',), 'picks read validation only'
    enc = w.encoder
    with w:
        stage1 = ns_clu_cells.stage1_ranges(SET, enc)
        ami = {a: stage1[f'cstar/{SET}/{enc}/{a}/ami'] for a in ASPECTS}
        mr = validation_ranges(w)
        unit = mr['ranges']
        xs, labels = ns_picks.inputs('untold', SET, enc)
        udoc = ns_picks.readout_doc(xs, labels, unit, workers)
        cdoc = bs_artret.pick_cstar(w.root, unit, workers)
    row = udoc['table'][udoc['pick']]
    ro = udoc['readout']
    readouts = dict(
        untold=dict(pick=udoc['pick'], k=int(ro.k), score=row['score'], kept=row['kept'], map=row['map'],
                    r0_map=udoc['table']['none+k0']['map']),
        cstar=dict(pick=cdoc['pick'], score=cdoc['table'][cdoc['pick']]['score'], kept=cdoc['table'][cdoc['pick']]['kept'],
                   map=cdoc['table'][cdoc['pick']]['map']),
        plain=dict(pick=ns_ret.reused_pick('plain', SET, enc), source=f'r14 stage1_{enc}.json artret_picks.frozen'),
        teacher=dict(pick=bs_artret.read_json(ns_paths.R11_OUT / 'stage4/picks.json')['instructed'],
                     source='r11 stage4/picks.json instructed'))
    if tables_dir is not None:
        ns_paths.write_json(Path(tables_dir) / f'tables_{w.arm}.json',
                            dict(untold=udoc['table'], cstar=cdoc['table'], units=unit))
    return dict(ami_ranges={a: dict(r=float(ami[a]['r']), per_seed=ami[a]['per_seed'], k=int(ami[a]['k']),
                                    n_val=int(ami[a]['n_val'])) for a in ASPECTS},
                map_ranges={a: dict(r=float(mr['ranges'][a]), per_seed=mr['per_seed'][a]) for a in ASPECTS},
                readouts=readouts, validation_files=mr['files'], n_val=mr['n_val'])


def write_picks(arm, encoder, root, body, records=None):
    rec = dict(arm=arm, encoder=encoder, root=str(root), split='val', written=now(), **body)
    path = Path(records or records_dir()) / f'image_picks_{arm}.json'
    ns_paths.write_json(path, rec)
    return path, rec


def read_picks(arm, records=None, require_committed=True, repo=None):
    """The picks record of `arm`. Refuses when absent (the order rule: picks before any test number), when it names another
    arm or encoder, and (require_committed) unless it equals its content in git HEAD of `repo` (default: this checkout)."""
    path = Path(records or records_dir()) / f'image_picks_{arm}.json'
    if not path.exists():
        raise FileNotFoundError(f'no picks record {path}: run `picks --arm {arm}` (validation) before any test cell')
    text = path.read_text()
    rec = json.loads(text)
    assert rec['arm'] == arm, (path, rec['arm'])
    assert rec['encoder'] in ns_paths.ENCODERS, (path, rec['encoder'])
    repo = Path(repo or P.REPO)
    if require_committed and path.resolve().is_relative_to(repo.resolve()):
        rel = path.resolve().relative_to(repo.resolve())
        head = subprocess.run(['git', '-C', str(repo), 'show', f'HEAD:{rel}'], capture_output=True, text=True)
        if head.returncode != 0 or head.stdout != text:
            raise RuntimeError(f'{rel} is not committed unchanged in HEAD: commit the picks before any test cell (C13)')
    return rec


# ================================================================================================ test cells
def _cell(type_, protocol, enc, arm, aspect, comparator, k, p, r, n_rows, **extra):
    """One cell: value (candidate), comparator value, d, interval, r, outcome."""
    vals = [p['cand'], p['comp'], p['d'], p['lo'], p['hi']]
    assert all(np.isfinite(v) for v in vals), (type_, aspect, vals)
    metric = 'ami' if type_ == 'ami' else 'map'
    return dict(id=f'{SET}/{enc}/{arm}/{aspect}/{metric}{"" if k is None else k}/{comparator}', type=type_, protocol=protocol,
                set=SET, encoder=enc, arm=arm, aspect=aspect, metric=metric, k=k, comparator=comparator,
                value=float(p['cand']), comparator_value=float(p['comp']), d=float(p['d']), lo=float(p['lo']), hi=float(p['hi']),
                r=float(r), outcome=outcome(p['lo'], r), n_draws=int(p['n_draws']), n_rows=int(n_rows), **extra)


def ami_cells(w, picks, aspects, draws):
    """r17 SC1 (plain) and SC3 (teacher) AMI cells, one pair per aspect."""
    enc, out = w.encoder, []
    for a in aspects:
        k = ns_paths.CLASS_COUNT[(SET, a)]
        y = ns_data.labels(SET, SPLIT_HELD, a)
        assert ns_clu_cells.ns_metrics.class_count(y) == k, (a, k)
        cand = ns_clu_cells._cand_parts(SET, SPLIT_HELD, enc, a, k)
        for comp, proto in (('plain', 'r17_SC1'), ('teacher', 'r17_SC3')):
            cp = ns_clu_cells._comp_parts(SET, SPLIT_HELD, enc, a, comp, k)
            res = ns_clu.paired_cell(cand, cp, y, k, r=None, metrics=('ami',), draws=draws, rng_seed=AMI_RNG_SEED)['ami']
            out.append(_cell('ami', proto, enc, w.arm, a, comp, k, res, picks['ami_ranges'][a]['r'], len(y),
                             n_labelled=int((y >= 0).sum())))
    return out


def map_plain_cells(w, picks, aspects, draws, workers=8):
    """r17 SC7: untold candidate against the plain image (`ns_ret`), each at its own validation pick."""
    enc, out = w.encoder, []
    rows = ns_ret._Rows(SET, enc)
    for a in aspects:
        y = ns_data.labels(SET, SPLIT_HELD, a)
        cand = rows.ap('untold', a, picks['readouts']['untold']['pick'], y, workers)
        comp = rows.ap('plain', a, picks['readouts']['plain']['pick'], y, workers)
        assert np.array_equal(cand['rows'], comp['rows']), (a, 'the arms were scored on other queries')
        p = ns_ret.paired_ret(cand['ap'], comp['ap'], picks['map_ranges'][a]['r'], draws, comp_seeds='none')
        out.append(_cell('map_plain', 'r17_SC7', enc, w.arm, a, 'plain', None, p, picks['map_ranges'][a]['r'], len(y),
                         candidate_readout=picks['readouts']['untold']['pick'],
                         comparator_readout=picks['readouts']['plain']['pick'], n_queries=int(len(cand['rows']))))
    return out


def map_teacher_cells(w, picks, aspects, draws, outputs, workers=8):
    """r14 Q1: the aspect head at r11's read-out against r11's stored `instructed` per-query AP."""
    enc, out = w.encoder, []
    pick = picks['readouts']['cstar']['pick']
    ro = bs_artret.readout_of(pick, rt_data.WIDTHS['cstar'])
    labels = rt_data.labels(SPLIT_HELD)

    def get_x(head, seed):
        return rt_readout.stage5_apply(ro, w.embed(w.root, 'val', head, seed), w.embed(w.root, 'held', head, seed))
    res = bs_artret.rt_run.score_arm(bs_artret._args(outputs, workers), 5, 'cstar', [(a, a) for a in aspects], get_x, labels,
                                     qe_k=ro.k)
    for a in aspects:
        cand, teacher = res[(a, a)], bs_artret.r11_stored('instructed', a)
        assert np.array_equal(cand['rows'], teacher['rows']), (a, 'the arms were scored on other queries')
        assert teacher['pick'] == picks['readouts']['teacher']['pick'], (a, teacher['pick'])
        pr = rt_stats.paired(cand['ap'], teacher['ap'], draws)
        p = dict(cand=float(cand['ap'].mean()), comp=float(teacher['ap'].mean()), d=pr['d'], lo=pr['interval'][0],
                 hi=pr['interval'][1], n_draws=draws)
        out.append(_cell('map_teacher', 'r14_Q1', enc, w.arm, a, 'teacher', None, p, picks['map_ranges'][a]['r'],
                         len(labels[a]), candidate_readout=pick, comparator_readout=teacher['pick'],
                         n_queries=int(len(cand['rows']))))
    return out


def score_cells(w, picks, aspects=ASPECTS, types=CELL_TYPES, draws=None, outputs=None, workers=8):
    """The cells of the given types and aspects, from a picks body (record or recorded K0 picks). `draws`: per-metric
    override of N_DRAWS (replay checks only)."""
    draws = dict(N_DRAWS, **(draws or {}))
    outputs = Path(outputs or P.OUT / 'image_cells' / w.encoder / w.arm / 'perquery')
    cells = []
    with w:
        if 'ami' in types:
            cells += ami_cells(w, picks, aspects, draws['ami'])
        if 'map_plain' in types:
            cells += map_plain_cells(w, picks, aspects, draws['map'], workers)
        if 'map_teacher' in types:
            cells += map_teacher_cells(w, picks, aspects, draws['map'], outputs, workers)
    return cells


def held_files(w, aspects=ASPECTS):
    return {a: {str(s): dict(path=str(w.path('held', a, s)), sha256=sha_file(w.path('held', a, s))) for s in SEEDS}
            for a in aspects}


# ================================================================================================ gate (v), image part
def recorded_k0_picks():
    """K0's picks and ranges as the older rounds recorded them (the inputs of the replay of the test cells)."""
    enc = 'siglip1'
    s17 = json.loads((P.TEST / '20261103_new_scores/records/stage1_ranges.json').read_text())
    pk17 = json.loads((P.TEST / '20261103_new_scores/records/stage1_picks.json').read_text())[f'untold/{SET}/{enc}']
    r14 = json.loads((P.R14_ROUND / f'records/stage1_{enc}.json').read_text())
    return dict(
        ami_ranges={a: s17[f'cstar/{SET}/{enc}/{a}/ami'] for a in ASPECTS},
        map_ranges={a: dict(r=float(r14['artret_ranges']['ranges'][a]), per_seed=r14['artret_ranges']['per_seed'][a])
                    for a in ASPECTS},
        readouts=dict(untold=dict(pick=pk17['pick'], map=pk17['scores']['map']),
                      cstar=dict(pick=r14['artret_picks']['cstar']['pick']),
                      plain=dict(pick=ns_ret.reused_pick('plain', SET, enc)),
                      teacher=dict(pick=r14['artret_picks']['teacher']['pick'])))


def _recorded_cell(cell, r17, r14):
    """The older record's values for one of our cells: (record id, {value, comparator_value, d, lo, hi, r, outcome})."""
    if cell['type'] == 'map_teacher':
        (c,) = [c for c in r14['cells'] if (c['sc'], c['encoder'], c['aspect'], c['data'], c['task']) ==
                ('SC6', cell['encoder'], cell['aspect'], 'artelingo_held', 'ret')]
        return f"r14 SC6 {cell['encoder']}/{cell['aspect']}", dict(
            value=c['candidate_value'], comparator_value=c['comparator_value'], d=c['d'], lo=c['lo'], hi=c['hi'], r=c['r'],
            outcome=c['outcome'])
    if cell['type'] == 'ami':
        rid = f"{'SC1' if cell['comparator'] == 'plain' else 'SC3'}/{SET}/{cell['encoder']}/{cell['aspect']}/ami/{cell['k']}/{cell['comparator']}"
    else:
        rid = f"SC7/{SET}/{cell['encoder']}/{cell['aspect']}/map/-/plain"
    (c,) = [c for c in r17['cells'] if c['id'] == rid]
    return f'r17 {rid}', dict(value=c['cand'], comparator_value=c['comp'], d=c['d'], lo=c['lo'], hi=c['hi'], r=c['r'],
                              outcome=c['outcome'])


def compare_cells(cells, r17, r14, check_interval):
    items = []
    for c in cells:
        rid, rec = _recorded_cell(c, r17, r14)
        diffs = {k: abs(c[k] - rec[k]) for k in ('value', 'comparator_value', 'd', 'r') + (('lo', 'hi') if check_interval else ())}
        items.append(dict(id=c['id'], record=rid, max_abs_diff=float(max(diffs.values())), diffs=diffs,
                          outcome=c['outcome'], recorded_outcome=rec['outcome'],
                          ok=bool(max(diffs.values()) <= TOL and (not check_interval or c['outcome'] == rec['outcome']))))
    return items


def compare_picks(body, rec):
    """Recomputed validation picks and ranges against the older records (r17 stage1_ranges and picks, r14 stage 1)."""
    items = []
    for a in ASPECTS:
        for s in SEEDS:
            items.append(dict(name=f'ami per_seed {a} s{s}', diff=abs(body['ami_ranges'][a]['per_seed'][str(s)] - rec['ami_ranges'][a]['per_seed'][str(s)])))
            items.append(dict(name=f'map R0 per_seed {a} s{s}', diff=abs(body['map_ranges'][a]['per_seed'][str(s)] - float(rec['map_ranges'][a]['per_seed'][str(s)]))))
        items.append(dict(name=f'ami r {a}', diff=abs(body['ami_ranges'][a]['r'] - rec['ami_ranges'][a]['r'])))
        items.append(dict(name=f'map r {a}', diff=abs(body['map_ranges'][a]['r'] - rec['map_ranges'][a]['r'])))
        items.append(dict(name=f'untold crossfit map {a}', diff=abs(body['readouts']['untold']['map'][a] - rec['readouts']['untold']['map'][a])))
    for i in items:
        i['ok'] = bool(i['diff'] <= TOL)
    picks_equal = {k: body['readouts'][k]['pick'] == rec['readouts'][k]['pick'] for k in ('untold', 'cstar', 'plain', 'teacher')}
    return items, picks_equal


def gate_v_image(aspects=ASPECTS, draws=None, validation=True, workers=8, root=P.K0_ROOT, arm=P.K0_ARM, outputs=None):
    """Gate (v), image part, on K0 (r14's SigLIP 1 c* held exports): our cells from the recorded picks and ranges against
    r17's SC1, SC3, SC7 and r14's SC6 (Q1) ArtELingo SigLIP 1 cells within 1e-9; with `validation`, our picks and
    ranges recomputed from the validation exports against the recorded ones. Returns the gate record's details."""
    draws = dict(N_DRAWS, **(draws or {}))
    check_interval = draws == N_DRAWS
    r17 = json.loads((P.TEST / '20261103_new_scores/records/verdicts.json').read_text())
    r14 = json.loads((P.R14_ROUND / 'records/verdicts.json').read_text())
    rec_picks = recorded_k0_picks()
    t0 = time.monotonic()
    wv = Wiring(root, arm, 'siglip1', splits=('val',))
    wh = Wiring(root, arm, 'siglip1')
    details = dict(aspects=list(aspects), draws=draws, intervals_checked=check_interval, tol=TOL)
    ok = True
    if validation:                                       # first: a crash here costs minutes, not the cells' half hour
        body = compute_picks(wv, workers)
        pitems, picks_equal = compare_picks(body, rec_picks)
        details.update(picks=pitems, picks_equal=picks_equal, seconds_validation=round(time.monotonic() - t0),
                       recomputed_readouts={k: v['pick'] for k, v in body['readouts'].items()})
        ok = all(i['ok'] for i in pitems) and all(picks_equal.values())
    t1 = time.monotonic()
    cells = score_cells(wh, rec_picks, aspects, draws=draws, outputs=outputs, workers=workers)
    items = compare_cells(cells, r17, r14, check_interval)
    details.update(cells=items, seconds_cells=round(time.monotonic() - t1))
    ok = ok and all(i['ok'] for i in items) and len(items) == 4 * len(aspects)
    details['max_abs_diff'] = float(max([i['max_abs_diff'] for i in items] + [i['diff'] for i in details.get('picks', [])]))
    return ok, details


def write_gate(ok, details, partial_dir=None):
    """records/gates/v_image.json (`tries` counts earlier writes). A reduced check (a subset of aspects, fewer draws, no
    validation part) goes to `partial_dir` and never to the gate record."""
    full = (set(details['aspects']) == set(ASPECTS) and details['intervals_checked'] and 'picks' in details)
    path = (P.GATES if full else Path(partial_dir or P.OUT / 'gates_partial')) / 'v_image.json'
    tries = 1
    if path.exists():
        tries = int(json.loads(path.read_text()).get('tries', 0)) + 1
    rec = dict(gate='v_image', status='pass' if ok else 'fail', tries=tries, details=details, written=now())
    if not full:
        rec['status'] = 'partial-' + rec['status']
    ns_paths.write_json(path, rec)
    return path, rec


# ================================================================================================ CLI
def _wiring(args, splits):
    arm = args.arm
    enc = args.encoder or encoder_of_arm(arm)
    root = args.root or P.OUT / 'cstar' / enc
    return Wiring(root, arm, enc, splits=splits)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, h in (('picks', 'validation picks and ranges -> records/image_picks_<arm>.json'),
                    ('cells', 'test cells -> records/image_cells_<arm>.json (refuses without the picks record)')):
        s = sub.add_parser(name, help=h)
        s.add_argument('--arm', required=True, help='caption source name, e.g. siglip1')
        s.add_argument('--encoder', choices=[e for e in ns_paths.ENCODERS if e != 'qwen'])
        s.add_argument('--root', type=Path, help='model root, default OUT/cstar/<encoder>')
    sub.choices['cells'].add_argument('--types', nargs='+', choices=CELL_TYPES, default=list(CELL_TYPES))
    sub.choices['cells'].add_argument('--allow-uncommitted', action='store_true',
                                      help='smoke only: do not require the picks record to be committed')
    g = sub.add_parser('gate-v-image', help='replay r17 SC1, SC3, SC7 and r14 Q1 ArtELingo SigLIP 1 cells on K0 exports')
    g.add_argument('--aspects', nargs='+', choices=ASPECTS, default=list(ASPECTS))
    g.add_argument('--draws-ami', type=int, default=N_DRAWS['ami'])
    g.add_argument('--draws-map', type=int, default=N_DRAWS['map'])
    g.add_argument('--no-validation', action='store_true', help='skip the recomputation of picks and ranges')
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    if args.cmd == 'gate-v-image':
        ok, details = gate_v_image(tuple(args.aspects), dict(ami=args.draws_ami, map=args.draws_map),
                                   not args.no_validation, args.workers)
        path, rec = write_gate(ok, details)
        print(f'gate v_image: {rec["status"]} max_abs_diff {details["max_abs_diff"]:.3g} ({time.monotonic() - t0:.0f}s) {path}')
        return 0 if ok else 1
    if args.cmd == 'picks':
        w = _wiring(args, ('val',))
        body = compute_picks(w, args.workers, tables_dir=P.OUT / 'image_picks')
        path, rec = write_picks(w.arm, w.encoder, w.root, body)
        print(f'picks {w.arm}: untold {rec["readouts"]["untold"]["pick"]}, cstar {rec["readouts"]["cstar"]["pick"]}, '
              f'plain {rec["readouts"]["plain"]["pick"]} -> {path} ({time.monotonic() - t0:.0f}s)')
        return 0
    w = _wiring(args, ('val', 'held'))
    picks = read_picks(args.arm, require_committed=not args.allow_uncommitted)
    assert picks['encoder'] == w.encoder and picks['root'] == str(w.root), 'the picks record is for another model root'
    cells = score_cells(w, picks, types=tuple(args.types), workers=args.workers)
    rec = dict(arm=w.arm, encoder=w.encoder, root=str(w.root), picks_sha256=sha_file(records_dir() / f'image_picks_{w.arm}.json'),
               held_files=held_files(w), written=now(), cells=cells)
    path = records_dir() / f'image_cells_{w.arm}.json'
    ns_paths.write_json(path, rec)
    print(f'cells {w.arm}: {len(cells)} -> {path} ({time.monotonic() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
