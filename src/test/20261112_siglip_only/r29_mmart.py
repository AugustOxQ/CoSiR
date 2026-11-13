"""r29 ticket 11: the MMArt cells of K* (SC5), their exports, validation picks and ranges, and gates (v) MMArt and (vi).

Spec docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md §3.4 (cells table row 4 and the rebuild paragraph; order
and C13), §6 SC5, D18, D23, D3, §3.5 gates (v) (MMArt rebuilds) and (vi). Scoring map item 5.

Six cells per comparator side, each cell type with its own protocol (D18; never swapped):
  ami (plain, teacher)        r17 SC1 / SC3 on MMArt: the aspect head's stored float16 rows cast to float32, k-means at 9 / 20 / 10,
                              2,000 paired draws; r = the candidate's five-seed AMI range on MMArt validation (r17 `stage1_ranges`).
  map, untold vs plain        r17 SC7: the three unit heads joined (divided by sqrt(3)) against the plain SigLIP 1 image, each at its own
                              validation-fitted read-out (r15's rule on MMArt validation, as r17): untold at the pick of this module,
                              plain at r15's recorded `frozen` pick; 10,000 draws, queries then seeds.
  map, told vs teacher        r15 SC4: the aspect head at r15's rule (own ranges) against the teacher's aspect view at r15's
                              `picks_teacher.json` pick (one pick for the three aspects, picked with Qwen's ranges); 10,000 joint draws.
  r of the two mAP types      the candidate's own five-seed range of validation mAP at R0 on MMArt validation (r15's `ret_ranges`).
All of it goes through ticket 06's cell code (`r29_image_cells.ami_cells`, `map_plain_cells`, `_cell`; its `SET` and `SPLIT_HELD`
are patched to MMArt for the call and unwind) and r15's `mt_ret` for the told cells. No older file is edited.

Exports (r15's `mt_export` layout and checks, checkpoint root and arm as parameters): `export` runs K*'s saved heads on the default
image of MMArt's `val` (5,769) and `new` (16,782; the test rows are a subset) keysets on the CPU (as r15's stored exports were made),
into `OUT/mmart/<arm>/exports/<keyset>/siglip1/cstar/<aspect>__s<seed>.npz` (`keys`, `emb` float16 [n, 2048]) with r15's sidecar.

Order rule (§3.4): `picks --arm A` reads validation only and writes `records/mmart_picks_<arm>.json` (once per arm); `cells --arm A`
refuses without it (committed unchanged in HEAD) and without `records/pass_rule.json` committed unchanged in HEAD with status pass
(D3: MMArt is stage 2); all three commands refuse an arm that is not `kstar` of `records/prefix_pick.json`. `cells` writes
`records/mmart_cells_<arm>.json` (ticket 06's cell shape, ticket 08's reader).

Gates (stage 0, on K0 = r14's SigLIP 1 weights, never on a candidate):
  gate-v-mmart   the teacher's and plain SigLIP 1's per-query AP rebuilt from the stores at the recorded picks; each rebuilt point value
                 equals its record within 1e-9; `records/gates/v_mmart.json` has one flag per comparator.
  gate-vi export / score   K0's weights through `export` equal r15's stored exports bit for bit (CPU, as r15's); the scorer on them
                 reproduces r28's `mmart.json` SigLIP 1 c* cells, r17's SC1, SC3, SC7 and r15's SC4 cells, and the picks and ranges of
                 r17 and r15, within 1e-9; `records/gates/vi.json` (a gate record is written at most three times).

CLI (from the round folder):
  export --arm A [--root DIR] [--export-root DIR] [--device cpu]
  picks  --arm A [--root DIR] [--export-root DIR]
  cells  --arm A [--root DIR] [--export-root DIR] [--types ami map_plain map_teacher] [--allow-uncommitted]
  gate-v-mmart
  gate-vi export [--try N] [--device cpu]  |  gate-vi score --try N [--aspects ..] [--draws-ami N] [--draws-map N] [--no-validation]
"""
import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_paths as P  # noqa: E402
import r29_image_cells as IC  # noqa: E402  (ticket 06: puts the older rounds on sys.path, r17's and r14's modules imported)
import r29_captions as C  # noqa: E402
import r29_trainer as T  # noqa: E402

import bs_cstar  # noqa: E402  (r14)
import mt_data  # noqa: E402  (r15)
import mt_export  # noqa: E402
import mt_ret  # noqa: E402
import ns_clu_cells  # noqa: E402  (r17)
import ns_data  # noqa: E402
import ns_paths  # noqa: E402
import ns_picks  # noqa: E402
import ns_ret  # noqa: E402
import rt_metrics  # noqa: E402  (r11)
P.check_older_imports()

SET = 'mmart'
HELD = ns_paths.HELD_SPLIT[SET]                          # 'test'
ENCODER = 'siglip1'
ASPECTS = tuple(P.ASPECTS)
SEEDS = tuple(P.SEEDS)
KEYSETS = ('val', 'new')                                 # r15's export keysets this ticket needs; `test` is a subset of `new`
EXPORT_KEYSET = dict(ns_data.EXPORT_KEYSET)              # {'test': 'new', 'val': 'val'}
CELL_TYPES = IC.CELL_TYPES
N_DRAWS = IC.N_DRAWS
TOL = 1e-9                                               # D23: replayed test cells
CSTAR_WIDTH = ns_paths.CSTAR_WIDTH
R15 = P.OLDER / '20261102_mmart_test/records'
R15_TEACHER_RECORD = R15 / 'stage1_teacher.json'
N_VAL = 5769


# ================================================================================================ small helpers
def now():
    return IC.now()


def records_dir():
    return IC.records_dir()


def jsonable(x):
    """The JSON round trip of a record body: numpy scalars and arrays to Python types, keys to strings."""
    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, np.ndarray):
            return conv(o.tolist())
        if isinstance(o, np.generic):
            return o.item()
        return o
    return conv(x)


def write_json(path, obj):
    ns_paths.write_json(path, jsonable(obj))


def read_prefix_pick(records):
    """`records/prefix_pick.json` (ticket 05 writes it): the four keys of the index's shared interfaces, asserted."""
    path = Path(records) / 'prefix_pick.json'
    if not path.exists():
        raise RuntimeError(f'no {path}: ticket 05 writes it (K* is its `kstar`)')
    doc = json.loads(path.read_text())
    for key in ('kstar', 'kstar_form', 'p1_form', 'flag'):
        assert key in doc, f'{path}: key {key} missing (keys {sorted(doc)})'
    assert doc['kstar'] in ('siglip1', 'siglip1_prefix'), f'{path}: kstar {doc["kstar"]!r}'
    assert doc['kstar_form'] in ('plain', 'prefixed'), f'{path}: kstar_form {doc["kstar_form"]!r}'
    return doc


def check_kstar(arm, records):
    doc = read_prefix_pick(records)
    assert arm == doc['kstar'], f'arm {arm!r} is not K* ({doc["kstar"]!r} in prefix_pick.json): the MMArt cells are K*\'s'
    return doc


def require_stage2(records):
    """D3: MMArt is stage 2. `pass_rule.json` committed unchanged in HEAD (`P.require_committed`) and its status is pass."""
    path = Path(records) / 'pass_rule.json'
    P.require_committed(path, 'commit records/pass_rule.json before any MMArt export, pick or test cell (spec §3.4, §3.5 row 1b)')
    status = json.loads(path.read_text()).get('status')
    if status != 'pass':
        raise RuntimeError(f'{path}: stage 2 (MMArt) needs status pass (D3), it says {status!r}')


def arm_root(arm):
    assert arm in ('siglip1', 'siglip1_prefix'), f'arm {arm!r}: K* is siglip1 or siglip1_prefix'
    return P.OUT / 'cstar' / ENCODER


def default_export_root(arm):
    return P.OUT / 'mmart' / arm


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ================================================================================================ exports
def checkpoint_path(root, arm, aspect, seed):
    return bs_cstar.rb_student.weights_path(root, arm, aspect, seed)


def check_sidecar_source(weights, arm):
    """The head's sidecar names the caption source, width and store of the arm (K0's arm: `qwen`), as `r29_trainer.export` asserts."""
    source = 'qwen' if arm == P.K0_ARM else arm
    _, rb = T._sidecar(weights)
    assert (rb['cap_source'], rb['cap_width'], rb['cap_store_sha256']) == (source, P.CAPTION_SOURCES[source][1], C.store_hash(source)), \
        f'{weights}: trained on another caption source or store than arm {arm}'


def _heads_inference(stack, root, arm, device):
    """r14's inference of one saved head's image tower (`r29_trainer.infer_sides`: batches of 512, float16, the saved best view, at
    the widths the sidecar records) after r14's weights checks, inside r14's export context (as `r29_trainer.export`)."""
    stack.enter_context(mock.patch.object(bs_cstar, 'row_arm', lambda row: arm))
    ctx = stack.enter_context(bs_cstar.training_context(ENCODER, bs_cstar.REF, export=True))

    def infer(weights, aspect, seed, x):
        got, _, sha = bs_cstar._check_weights(root, arm, aspect, seed, ctx.backbone, False)
        assert Path(got) == Path(weights) and sha == sha_file(weights), (got, weights)
        check_sidecar_source(weights, arm)
        assert x.shape[1] == ctx.width == ns_paths.WIDTH[ENCODER], (x.shape, ctx.width)
        emb = T.infer_sides(weights, {'img': x}, device)['img']
        problem = bs_cstar.rg_full.vc3_problem(emb, n_rows=len(x))
        assert problem is None, f'{weights}: {problem}'
        return emb
    return infer


def export_arm(root, arm, export_root, *, keysets=KEYSETS, seeds=SEEDS, aspects=ASPECTS, device='cpu', rows=None, infer=None,
               log=print):
    """Run the saved heads of (root, arm) on the default image of each keyset and write r15's export files (module docstring).
    `rows(keyset) -> (keys in mt_data.order, float32 [n, width])` (default: `mt_export.keyset_rows`); `infer(weights, aspect, seed, x)
    -> float16 [n, 2048]` (default: r14's inference above). An existing export is kept when its sidecar names this checkpoint and
    the npz is its sidecar's; otherwise the run stops (nothing is overwritten)."""
    root, export_root = Path(root), Path(export_root)
    rows = rows or (lambda keyset: mt_export.keyset_rows(ENCODER, keyset))
    written = []
    with contextlib.ExitStack() as stack:
        if infer is None:
            infer = _heads_inference(stack, root, arm, device)
        for keyset in keysets:
            keys, x = rows(keyset)
            for seed in seeds:
                for aspect in aspects:
                    ckpt = checkpoint_path(root, arm, aspect, seed)
                    path = mt_export.export_path(export_root, keyset, ENCODER, aspect, seed)
                    if path.exists():
                        meta = json.loads(path.with_suffix('.json').read_text())
                        assert meta['sha256'] == sha_file(path) and meta['checkpoint'] == str(ckpt) \
                            and meta['checkpoint_sha256'] == sha_file(ckpt), f'{path}: exists and is not this checkpoint\'s export'
                        log(f'{keyset} {aspect} s{seed}: kept')
                        written.append(path)
                        continue
                    t0 = time.monotonic()
                    emb = infer(ckpt, aspect, seed, x)
                    assert emb.shape == (len(keys), CSTAR_WIDTH) and emb.dtype == np.float16, (emb.shape, emb.dtype)
                    with mock.patch.object(mt_export, 'DEVICE', device):
                        mt_export.write_export(export_root, keyset, ENCODER, aspect, seed, keys, emb, checkpoint=ckpt,
                                               view=mt_export.P.DEFAULT_VIEW[ENCODER])
                    side_path = path.with_suffix('.json')
                    side = json.loads(side_path.read_text())
                    side.update(arm=arm, r29_code_sha256=sha_file(__file__))
                    side_path.write_text(json.dumps(side, indent=2, allow_nan=False) + '\n')
                    log(f'{keyset} {aspect} s{seed}: n={len(keys)} {time.monotonic() - t0:.1f}s')
                    written.append(path)
    return written


# ================================================================================================ wiring
class Wiring:
    """One model root, arm and export root for r17's loaders (`mock.patch`; no older file is edited). `splits`: the splits it may
    read (picks: ('val',), so a test row never reaches a pick). `names(split)`: the MMArt keys the rows must follow (default: r17's
    `ns_data.keys`, sha256 order; tests pass small lists). Rows are joined by key (r15's `mt_export.load`), never by position (C6)."""

    def __init__(self, root, arm, export_root, splits=('val', 'test'), names=None):
        assert set(splits) <= {'val', 'test'}, splits
        self.root, self.arm, self.export_root, self.splits = Path(root), arm, Path(export_root), tuple(splits)
        self.encoder = ENCODER
        self._names = names or (lambda split: ns_data.keys(SET, split))
        self._stack = None

    def checkpoint(self, aspect, seed):
        return checkpoint_path(self.root, self.arm, aspect, seed)

    def path(self, split, aspect, seed):
        return mt_export.export_path(self.export_root, EXPORT_KEYSET[split], ENCODER, aspect, seed)

    def head(self, split, aspect, seed):
        """Stored float16 rows [n, 2048] of one head in the order of the split's keys."""
        assert split in ('val', 'test') and aspect in ASPECTS and seed in SEEDS, (split, aspect, seed)
        assert split in self.splits, f'this wiring reads {self.splits} only (validation picks never see test rows): {split!r}'
        names = list(self._names(split))
        with mock.patch.object(mt_export, 'checkpoint_path', lambda enc, a, s: self.checkpoint(a, s)):
            x = mt_export.load(EXPORT_KEYSET[split], ENCODER, aspect, seed, names, root=self.export_root)
        assert x.dtype == np.float16 and x.shape == (len(names), CSTAR_WIDTH), (x.dtype, x.shape)
        assert np.isfinite(x).all(), f'{self.path(split, aspect, seed)}: non-finite rows'
        dev = float(np.abs(np.linalg.norm(x.astype(np.float32), axis=1) - 1).max())
        assert dev < ns_paths.TOL_UNIT, f'{self.path(split, aspect, seed)}: rows not unit norm (max deviation {dev})'
        return x

    def _check(self, set_, enc):
        assert set_ == SET and enc == ENCODER, f'wired for {SET}/{ENCODER}, asked {set_}/{enc}'

    def cstar(self, set_, split, enc, aspect, seed, unit=True):
        """`ns_data.cstar` at this root: unit=True the retrieval form, unit=False the stored values cast to float32."""
        self._check(set_, enc)
        x16 = self.head(split, aspect, seed)
        x = ns_data._unit32(x16) if unit else np.asarray(x16).astype(np.float32)
        return ns_data._check_rows(x, len(x16), CSTAR_WIDTH, f'cstar {set_}/{split}/{aspect}/s{seed}')

    def head_id(self, set_, split, enc, aspect, seed):
        """`ns_data._head_id` for MMArt: the `keys` array inside the export (the three heads of `untold` must agree)."""
        self._check(set_, enc)
        self.head(split, aspect, seed)
        with np.load(self.path(split, aspect, seed), allow_pickle=False) as z:
            return z['keys'].tolist()

    def __enter__(self):
        assert self._stack is None, 'wiring already active'
        self._stack = contextlib.ExitStack()
        for name, fn in (('cstar', self.cstar), ('_head_id', self.head_id)):
            self._stack.enter_context(mock.patch.object(ns_data, name, fn))
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self._stack = None


def validation_files(w):
    return {a: {str(s): dict(path=str(w.path('val', a, s)), sha256=sha_file(w.path('val', a, s))) for s in SEEDS} for a in ASPECTS}


def held_files(w, aspects=ASPECTS):
    return {a: {str(s): dict(path=str(w.path('test', a, s)), sha256=sha_file(w.path('test', a, s))) for s in SEEDS} for a in aspects}


def check_validation_files(w, picks):
    """The validation exports the cells read again (to fit the read-outs) are the ones the picks read: sha256 against the record."""
    want = picks['validation_files']
    for a in ASPECTS:
        for s in SEEDS:
            got = sha_file(w.path('val', a, s))
            assert got == want[a][str(s)]['sha256'], f'validation export {w.path("val", a, s)} changed since the picks were written'


# ================================================================================================ picks (validation only)
def teacher_pick():
    """r15's committed teacher record: one pick for the three aspect views, picked with Qwen's ranges (D12)."""
    rec = json.loads(R15_TEACHER_RECORD.read_text())
    assert rec['ranges_encoder'] == 'qwen' and rec['n_val'] == N_VAL, R15_TEACHER_RECORD
    return rec['arms']['teacher']['pick']


def compute_picks(w, workers=8):
    """Everything the test cells need from validation, as the record's body (no test row is read: `w.splits` is ('val',)).
    ami_ranges   `ns_clu_cells.stage1_ranges('mmart', ...)` AMI per aspect: r, per_seed, k (r17).
    map_ranges   R0 mAP per aspect and seed on the 5,769 validation paintings (`mt_ret.r0_map_per_seed`) and `own_ranges` (r15).
    readouts     untold: r15's rule (`ns_picks.readout_doc`, r11's `ac_select.arm_table`) on the three joined heads, in units of
                 map_ranges (as ticket 06); cstar: `mt_ret.readout_doc` (r15's rule, own ranges); plain: r15's recorded `frozen`
                 pick; teacher: r15's `picks_teacher.json` pick."""
    assert w.splits == ('val',), 'picks read validation only'
    with w:
        stage1 = ns_clu_cells.stage1_ranges(SET, ENCODER)
        ami = {a: stage1[f'cstar/{SET}/{ENCODER}/{a}/ami'] for a in ASPECTS}
        keys = ns_data.keys(SET, 'val')
        mt_ret.check_order(keys)
        labels = {a: ns_data.labels(SET, 'val', a) for a in ASPECTS}
        xs = {(a, s): ns_data.cstar(SET, 'val', ENCODER, a, s) for a in ASPECTS for s in SEEDS}
        per_seed = mt_ret.r0_map_per_seed(lambda a, s: xs[(a, s)], labels, workers)
        unit = mt_ret.own_ranges(per_seed)
        cdoc = mt_ret.readout_doc(xs, labels, unit, workers)
        del xs
        uxs, ulabels = ns_picks.inputs('untold', SET, ENCODER)
        udoc = ns_picks.readout_doc(uxs, ulabels, unit, workers)
    urow = udoc['table'][udoc['pick']]
    crow = cdoc['table'][cdoc['pick']]
    readouts = dict(
        untold=dict(pick=udoc['pick'], k=int(udoc['readout'].k), score=urow['score'], kept=urow['kept'], map=urow['map'],
                    r0_map=udoc['table']['none+k0']['map'], ranges=udoc['ranges'], table=udoc['table']),
        cstar=dict(pick=cdoc['pick'], score=crow['score'], kept=crow['kept'], map=crow['map'], table=cdoc['table']),
        plain=dict(pick=ns_ret.reused_pick('plain', SET, ENCODER), source=f'r15 stage1_{ENCODER}.json picks.arms.frozen'),
        teacher=dict(pick=teacher_pick(), source='r15 records/stage1_teacher.json arms.teacher'))
    return jsonable(dict(
        ami_ranges={a: dict(r=float(ami[a]['r']), per_seed=ami[a]['per_seed'], k=int(ami[a]['k']), n_val=int(ami[a]['n_val']))
                    for a in ASPECTS},
        map_ranges={a: dict(r=float(unit[a]), per_seed=per_seed[a]) for a in ASPECTS},
        readouts=readouts, validation_files=validation_files(w), n_val=len(keys)))


def picks_path(arm, records=None):
    return Path(records or records_dir()) / f'mmart_picks_{arm}.json'


def write_picks(arm, root, export_root, body, records=None, **extra):
    rec = dict(arm=arm, encoder=ENCODER, root=str(root), export_root=str(export_root), split='val', written=now(), **extra, **body)
    path = picks_path(arm, records)
    if path.exists():            # order rule: one record per arm; a new try needs the old file moved aside by the controller
        raise FileExistsError(f'{path} exists: a picks record is written once per arm (the controller moves it for a new try)')
    write_json(path, rec)
    return path, rec


def read_picks(arm, records=None, require_committed=True):
    """The picks record of `arm`: refuses when absent (the order rule), when it names another arm, and (require_committed) unless
    it is committed unchanged in git HEAD (C13)."""
    path = picks_path(arm, records)
    if not path.exists():
        raise FileNotFoundError(f'no picks record {path}: run `picks --arm {arm}` (validation) before any test cell')
    if require_committed:
        P.require_committed(path, 'commit the MMArt picks record before any test cell (C13)')
    rec = json.loads(path.read_text())
    assert rec['arm'] == arm and rec['encoder'] == ENCODER, (path, rec['arm'], rec['encoder'])
    return rec


# ================================================================================================ test cells
def map_teacher_cells(w, picks, aspects, draws, workers=8):
    """r15 SC4 (told against the teacher): the aspect head at r15's rule against the teacher's aspect view at r15's
    `picks_teacher.json` pick, `mt_ret.arm_ap` and `mt_ret.aspect_cells` (`rt_stats.paired`, 10,000 joint draws)."""
    out = []
    cpick, tpick = picks['readouts']['cstar']['pick'], picks['readouts']['teacher']['pick']
    for a in aspects:
        y = ns_data.labels(SET, HELD, a)
        cand = mt_ret.arm_ap(lambda s: ns_data.cstar(SET, 'val', ENCODER, a, s), lambda s: ns_data.cstar(SET, HELD, ENCODER, a, s),
                             cpick, y, True, workers)
        tv, te = mt_ret.unit(ns_data.teacher(SET, 'val', a)), mt_ret.unit(ns_data.teacher(SET, HELD, a))
        teacher = mt_ret.arm_ap(lambda s: tv, lambda s: te, tpick, y, False, workers)
        r = picks['map_ranges'][a]['r']
        (c,) = mt_ret.aspect_cells(ENCODER, a, cand, None, teacher, r, draws, dict(cstar=cpick, frozen=None, teacher=tpick), len(y),
                                   'mmart_test', scs=('SC4',))
        p = dict(cand=c['candidate_value'], comp=c['comparator_value'], d=c['d'], lo=c['lo'], hi=c['hi'], n_draws=c['n_draws'])
        out.append(IC._cell('map_teacher', 'r15_SC4', ENCODER, w.arm, a, 'teacher', None, p, r, len(y), candidate_readout=cpick,
                            comparator_readout=tpick, n_queries=c['n_queries']))
    return out


def score_cells(w, picks, aspects=ASPECTS, types=CELL_TYPES, draws=None, workers=8, check_files=True):
    """The cells of the given types and aspects from a picks body (record, or K0's recorded picks for the gate). `draws`:
    per-metric override of N_DRAWS (replay checks only). `check_files`: the validation exports against the picks record."""
    if check_files:
        check_validation_files(w, picks)
    draws = dict(N_DRAWS, **(draws or {}))
    cells = []
    with w, mock.patch.multiple(IC, SET=SET, SPLIT_HELD=HELD):       # ticket 06's cell code, pointed at MMArt for this call
        if 'ami' in types:
            cells += IC.ami_cells(w, picks, aspects, draws['ami'])
        if 'map_plain' in types:
            cells += IC.map_plain_cells(w, picks, aspects, draws['map'], workers)
        if 'map_teacher' in types:
            cells += map_teacher_cells(w, picks, aspects, draws['map'], workers)
    return cells


def cells_path(arm, records=None):
    return Path(records or records_dir()) / f'mmart_cells_{arm}.json'


# ================================================================================================ commands
def run_export(arm, *, records=None, root=None, export_root=None, device='cpu', **kw):
    records = Path(records or records_dir())
    require_stage2(records)
    check_kstar(arm, records)
    return export_arm(root or arm_root(arm), arm, export_root or default_export_root(arm), device=device, **kw)


def run_picks(arm, *, records=None, root=None, export_root=None, workers=8, tables_dir=None):
    records = Path(records or records_dir())
    require_stage2(records)
    pp = check_kstar(arm, records)
    root, export_root = Path(root or arm_root(arm)), Path(export_root or default_export_root(arm))
    if picks_path(arm, records).exists():
        raise FileExistsError(f'{picks_path(arm, records)} exists: a picks record is written once per arm')
    body = compute_picks(Wiring(root, arm, export_root, splits=('val',)), workers)
    return write_picks(arm, root, export_root, body, records=records, kstar=pp['kstar'], kstar_form=pp['kstar_form'], flag=pp['flag'])[0]


def run_cells(arm, *, records=None, root=None, export_root=None, types=CELL_TYPES, workers=8, allow_uncommitted=False):
    """The order rule's refusals first (pass_rule committed and pass; K*; the picks record present and committed), then the cells."""
    assert not allow_uncommitted or P.OUT != P.DEFAULT_OUT, '--allow-uncommitted is for smoke runs only (set R29_OUTPUTS to a smoke root)'
    records = Path(records or records_dir())
    require_stage2(records)
    pp = check_kstar(arm, records)
    picks = read_picks(arm, records, require_committed=not allow_uncommitted)
    root, export_root = Path(root or arm_root(arm)), Path(export_root or default_export_root(arm))
    assert picks['root'] == str(root) and picks['export_root'] == str(export_root), 'the picks record is for another model root'
    path = cells_path(arm, records)
    if path.exists():
        raise FileExistsError(f'{path} exists: a cells record is written once (the controller moves it for a new try)')
    w = Wiring(root, arm, export_root)
    cells = score_cells(w, picks, types=tuple(types), workers=workers)
    rec = dict(arm=arm, encoder=ENCODER, root=str(root), export_root=str(export_root), kstar=pp['kstar'], kstar_form=pp['kstar_form'],
               flag=pp['flag'], picks_sha256=sha_file(picks_path(arm, records)), held_files=held_files(w), written=now(), cells=cells)
    write_json(path, rec)
    return path


# ================================================================================================ comparisons (gates)
def compare_export_files(ours, stored):
    """Two export npz files bit for bit: the `keys` lists (in order), and `emb` (shape, dtype, every value)."""
    with np.load(ours, allow_pickle=False) as x, np.load(stored, allow_pickle=False) as y:
        kx, ky, a, b = x['keys'].tolist(), y['keys'].tolist(), x['emb'], y['emb']
    same_shape = a.shape == b.shape and a.dtype == b.dtype
    differ = int((a != b).sum()) if same_shape else None
    return dict(equal=bool(kx == ky and same_shape and differ == 0), keys_equal=bool(kx == ky), shape=list(a.shape),
                differing_values=differ,
                max_abs_diff=float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if same_shape else None)


def compare_rebuilds(rebuilt, recorded, tol=TOL):
    """Gate (v): `rebuilt` {comparator: {aspect: mAP}}; `recorded` {comparator: {aspect: [(source, value), ...]}}. Every recorded
    value of a cell must equal the rebuilt one within `tol` (inclusive). Returns (items, {comparator: ok}): a failed comparator
    fails only its own flag."""
    items = []
    ok = {c: True for c in rebuilt}
    for comparator, per_aspect in rebuilt.items():
        for aspect, value in per_aspect.items():
            assert np.isfinite(value), (comparator, aspect, value)
            assert recorded[comparator][aspect], (comparator, aspect, 'no recorded value')
            for source, want in recorded[comparator][aspect]:
                good = bool(abs(value - want) <= tol)
                ok[comparator] &= good
                items.append(dict(comparator=comparator, aspect=aspect, source=source, rebuilt=float(value), recorded=float(want),
                                  abs_diff=float(abs(value - want)), ok=good))
    return items, ok


def compare_cell(cell, record_id, rec, check_interval):
    """One of our cells against an older record's values `rec` {value, comparator_value, d, lo, hi, r, outcome}."""
    keys = ('value', 'comparator_value', 'd', 'r') + (('lo', 'hi') if check_interval else ())
    diffs = {k: abs(cell[k] - rec[k]) for k in keys}
    worst = max(diffs.values())
    return dict(id=cell['id'], record=record_id, max_abs_diff=float(worst), diffs=diffs, outcome=cell['outcome'],
                recorded_outcome=rec['outcome'], ok=bool(worst <= TOL and (not check_interval or cell['outcome'] == rec['outcome'])))


def write_gate(folder, gate, passed, details):
    """`<folder>/<gate>.json` through `r29_captions.write_gate_record`, which counts the tries (one per call) and raises on a fourth
    (§3.5: three tries per gate). Status rule shared with tickets 05, 08, 14: `pass`; a failure is `fail` before try 3 and `parked` at
    try 3 (the gate then parks what it serves)."""
    path = Path(folder) / f'{gate}.json'
    prior = int(json.loads(path.read_text())['tries']) if path.exists() else 0
    status = 'pass' if passed else ('parked' if prior == 2 else 'fail')
    return C.write_gate_record(folder, dict(status=status, details=jsonable(details)), gate=gate)


# ================================================================================================ gate (v) MMArt rebuilds
def _json(path):
    return json.loads(Path(path).read_text())


def recorded_comparators():
    """{comparator: {aspect: [(source, value)]}}: the records the rebuilt point values must equal. Teacher: r15's SC4 cell
    `comparator_value` of each of the four encoders (one teacher view per aspect, so the same value four times); plain SigLIP 1:
    r17's SC7 `comp` and r15's SC2 `comparator_value`."""
    r15 = _json(R15 / 'verdicts.json')['cells']
    r17 = {c['id']: c for c in _json(P.R17_ROUND / 'records/verdicts.json')['cells']}
    out = dict(teacher={}, plain={})
    for a in ASPECTS:
        out['teacher'][a] = [(f'r15 verdicts SC4 {e}/{a} comparator_value', c['comparator_value']) for e in ns_paths.ENCODERS
                             for c in r15 if (c['sc'], c['encoder'], c['aspect'], c['data'], c['task']) == ('SC4', e, a, 'mmart_test', 'ret')]
        assert len(out['teacher'][a]) == len(ns_paths.ENCODERS), (a, out['teacher'][a])
        cid = f'SC7/{SET}/{ENCODER}/{a}/map/-/plain'
        (sc2,) = [c for c in r15 if (c['sc'], c['encoder'], c['aspect'], c['data'], c['task']) == ('SC2', ENCODER, a, 'mmart_test', 'ret')]
        out['plain'][a] = [(f'r17 verdicts {cid} comp', r17[cid]['comp']), (f'r15 verdicts SC2 {ENCODER}/{a} comparator_value',
                                                                           sc2['comparator_value'])]
    return out


def rebuild_comparators(aspects=ASPECTS, workers=8, perquery_dir=None):
    """The teacher's and plain SigLIP 1's per-query AP on the MMArt test rows, rebuilt from the stores at the recorded picks (r15's
    teacher pick for the three aspect views; r15's `frozen` pick for plain): the teacher as r15's `score_encoder` (`mt_ret.arm_ap`
    on `mt_ret.unit` of the view), plain as r17's `_Rows.ap`. Returns {comparator: {aspect: mAP}}; per-query AP to `perquery_dir`."""
    tpick, ppick = teacher_pick(), ns_ret.reused_pick('plain', SET, ENCODER)
    rows = ns_ret._Rows(SET, ENCODER)
    out = dict(teacher={}, plain={})
    for a in aspects:
        y = ns_data.labels(SET, HELD, a)
        tv, te = mt_ret.unit(ns_data.teacher(SET, 'val', a)), mt_ret.unit(ns_data.teacher(SET, HELD, a))
        res = dict(teacher=mt_ret.arm_ap(lambda s: tv, lambda s: te, tpick, y, False, workers),
                   plain=rows.ap('plain', a, ppick, y, workers))
        for comparator, r in res.items():
            out[comparator][a] = float(r['ap'].mean())
            assert r['ap'].ndim == 1 and np.isfinite(r['ap']).all()
            if perquery_dir is not None:
                Path(perquery_dir).mkdir(parents=True, exist_ok=True)
                np.savez(Path(perquery_dir) / f'{comparator}__{a}.npz', ap=r['ap'], rows=r['rows'])
    return out


def gate_v_mmart(aspects=ASPECTS, workers=8, perquery_dir=None):
    rebuilt = rebuild_comparators(aspects, workers, perquery_dir)
    recorded = recorded_comparators()
    items, ok = compare_rebuilds(rebuilt, {c: {a: recorded[c][a] for a in aspects} for c in recorded}, TOL)
    details = dict(aspects=list(aspects), tol=TOL, picks=dict(teacher=teacher_pick(), plain=ns_ret.reused_pick('plain', SET, ENCODER)),
                   comparators_ok=ok, failed_comparators=[c for c, v in ok.items() if not v], items=items,
                   max_abs_diff=float(max(i['abs_diff'] for i in items)))
    return all(ok.values()), details


# ================================================================================================ gate (vi)
def recorded_k0_picks():
    """K0's picks and ranges as r17 and r15 recorded them (the inputs of the replay of the test cells)."""
    s17 = _json(P.R17_ROUND / 'records/stage1_ranges.json')
    pk17 = _json(P.R17_ROUND / 'records/stage1_picks.json')[f'untold/{SET}/{ENCODER}']
    r15 = _json(R15 / f'stage1_{ENCODER}.json')
    return dict(
        ami_ranges={a: s17[f'cstar/{SET}/{ENCODER}/{a}/ami'] for a in ASPECTS},
        map_ranges={a: dict(r=float(r15['ret_ranges']['ranges'][a]), per_seed=r15['ret_ranges']['per_seed'][a]) for a in ASPECTS},
        readouts=dict(untold=dict(pick=pk17['pick'], map=pk17['scores']['map'], score=pk17['scores']['score'], kept=pk17['scores']['kept']),
                      cstar=dict(pick=r15['picks']['arms']['cstar']['pick'], table=r15['picks']['arms']['cstar']['table']),
                      plain=dict(pick=ns_ret.reused_pick('plain', SET, ENCODER)), teacher=dict(pick=teacher_pick())))


def _recorded_cell(cell, r17, r15):
    """(record id, {value, comparator_value, d, lo, hi, r, outcome}) of the older round's cell that ours replays."""
    a = cell['aspect']
    if cell['type'] == 'map_teacher':
        (c,) = [c for c in r15 if (c['sc'], c['encoder'], c['aspect'], c['data'], c['task']) == ('SC4', ENCODER, a, 'mmart_test', 'ret')]
        return f'r15 SC4 {ENCODER}/{a}', dict(value=c['candidate_value'], comparator_value=c['comparator_value'], d=c['d'], lo=c['lo'],
                                              hi=c['hi'], r=c['r'], outcome=c['outcome'])
    if cell['type'] == 'ami':
        rid = f"{'SC1' if cell['comparator'] == 'plain' else 'SC3'}/{SET}/{ENCODER}/{a}/ami/{cell['k']}/{cell['comparator']}"
    else:
        rid = f'SC7/{SET}/{ENCODER}/{a}/map/-/plain'
    c = r17[rid]
    return f'r17 {rid}', dict(value=c['cand'], comparator_value=c['comp'], d=c['d'], lo=c['lo'], hi=c['hi'], r=c['r'], outcome=c['outcome'])


def compare_r28(cells, r28):
    """Our cells against r28's `mmart.json` SigLIP 1 values (the paper's MMArt cells): AMI value (c*) and the plain AMI; the told c*
    mAP (`value` of the told-against-teacher cell) and the plain mAP (`comparator_value` of the untold-against-plain cell)."""
    sig, order = r28['siglip1'], {a: i for i, a in enumerate(ASPECTS)}
    items = []
    for c in cells:
        i = order[c['aspect']]
        pairs = []
        if c['type'] == 'ami':
            pairs.append(('value', f'siglip1/cstar_ami/values[{i}]', sig['cstar_ami']['values'][i]))
            if c['comparator'] == 'plain':
                pairs.append(('comparator_value', f'siglip1/plain_ami/values[{i}]', sig['plain_ami']['values'][i]))
        elif c['type'] == 'map_teacher':
            pairs.append(('value', f'siglip1/cstar_map/values[{i}]', sig['cstar_map']['values'][i]))
        else:
            pairs.append(('comparator_value', f'siglip1/plain_map/values[{i}]', sig['plain_map']['values'][i]))
        for field, name, want in pairs:
            items.append(dict(id=c['id'], field=field, record=f'r28 mmart.json {name}', ours=float(c[field]), recorded=float(want),
                              abs_diff=float(abs(c[field] - want)), ok=bool(abs(c[field] - want) <= TOL)))
    return items


def gate_vi_export(out_root, device='cpu', seeds=SEEDS, keysets=KEYSETS, log=print):
    """Part 1: K0's weights through `export_arm` into `out_root`; every file equal bit for bit to r15's stored export of the same
    keyset, aspect and seed (keys in order, every float16 value)."""
    paths = export_arm(P.K0_ROOT, P.K0_ARM, out_root, keysets=keysets, seeds=seeds, device=device, log=log)
    items = {}
    for path in paths:
        keyset = path.parents[2].name
        aspect, seed = path.stem.split('__s')
        stored = mt_export.export_path(ns_paths.R15_OUT, keyset, ENCODER, aspect, int(seed))
        items[f'{keyset}/{path.stem}'] = dict(stored=str(stored), **compare_export_files(path, stored))
    details = dict(root=str(P.K0_ROOT), arm=P.K0_ARM, out_root=str(out_root), device=device, stored_root=str(ns_paths.R15_OUT),
                   n_files=len(items), items=items, equal=bool(items) and all(i['equal'] for i in items.values()))
    return details['equal'], details


def gate_vi_score(export_root, aspects=ASPECTS, draws=None, validation=True, workers=8):
    """Part 2: the scorer on the exports under `export_root` (K0's, equal to r15's by part 1): the cells from K0's recorded picks and
    ranges against r28's `mmart.json`, r17's SC1, SC3, SC7 and r15's SC4 within 1e-9 (intervals when the draws are r17's and r15's);
    with `validation`, our picks and ranges recomputed from the validation exports against the recorded ones."""
    draws = dict(N_DRAWS, **(draws or {}))
    check_interval = draws == N_DRAWS
    r17 = {c['id']: c for c in _json(P.R17_ROUND / 'records/verdicts.json')['cells']}
    r15 = _json(R15 / 'verdicts.json')['cells']
    r28 = _json(P.R28_RECORDS / 'mmart.json')
    rec_picks = recorded_k0_picks()
    t0 = time.monotonic()
    details = dict(aspects=list(aspects), draws=draws, intervals_checked=check_interval, tol=TOL, export_root=str(export_root))
    ok = True
    if validation:                       # first: a crash here costs minutes, not the cells' half hour
        body = jsonable(compute_picks(Wiring(P.K0_ROOT, P.K0_ARM, export_root, splits=('val',)), workers))
        pitems, picks_equal = IC.compare_picks(body, rec_picks)
        details.update(picks=pitems, picks_equal=picks_equal, seconds_validation=round(time.monotonic() - t0),
                       recomputed_readouts={k: v['pick'] for k, v in body['readouts'].items()})
        ok = all(i['ok'] for i in pitems) and all(picks_equal.values())
    t1 = time.monotonic()
    cells = score_cells(Wiring(P.K0_ROOT, P.K0_ARM, export_root), rec_picks, aspects, draws=draws, workers=workers, check_files=False)
    items = [compare_cell(c, *_recorded_cell(c, r17, r15), check_interval) for c in cells]
    r28_items = compare_r28(cells, r28)
    details.update(cells=items, r28=r28_items, seconds_cells=round(time.monotonic() - t1))
    ok = ok and all(i['ok'] for i in items) and all(i['ok'] for i in r28_items) and len(items) == 4 * len(aspects)
    details['max_abs_diff'] = float(max([i['max_abs_diff'] for i in items] + [i['abs_diff'] for i in r28_items]
                                        + [i['diff'] for i in details.get('picks', [])]))
    return ok, details


# ================================================================================================ CLI
def _gate_try_dir(n=None):
    base = P.OUT / 'gate_vi'
    if n is None:
        n = 1
        while (base / f'try{n}').exists():
            n += 1
    return base / f'try{n}', n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('export', 'picks', 'cells'):
        s = sub.add_parser(name)
        s.add_argument('--arm', required=True, choices=('siglip1', 'siglip1_prefix'))
        s.add_argument('--root', type=Path, help='model root, default OUT/cstar/siglip1')
        s.add_argument('--export-root', type=Path, help='export root, default OUT/mmart/<arm>')
    sub.choices['export'].add_argument('--device', default='cpu')
    sub.choices['cells'].add_argument('--types', nargs='+', choices=CELL_TYPES, default=list(CELL_TYPES))
    sub.choices['cells'].add_argument('--allow-uncommitted', action='store_true', help='smoke only: the picks record need not be committed')
    sub.add_parser('gate-v-mmart', help='rebuild the teacher\'s and plain SigLIP 1\'s per-query AP; records/gates/v_mmart.json')
    g = sub.add_parser('gate-vi', help='part export (K0 weights vs r15 exports) or score (the scorer vs r28, r17, r15)')
    g.add_argument('part', choices=('export', 'score'))
    g.add_argument('--try', dest='n', type=int, help='try folder number (export: default the next free one; score: required)')
    g.add_argument('--device', default='cpu')
    g.add_argument('--aspects', nargs='+', choices=ASPECTS, default=list(ASPECTS))
    g.add_argument('--draws-ami', type=int, default=N_DRAWS['ami'])
    g.add_argument('--draws-map', type=int, default=N_DRAWS['map'])
    g.add_argument('--no-validation', action='store_true')
    args = ap.parse_args(argv)
    t0 = time.monotonic()
    rec_dir = records_dir()
    if args.cmd == 'export':
        paths = run_export(args.arm, root=args.root, export_root=args.export_root, device=args.device)
        print(f'export {args.arm}: {len(paths)} files ({time.monotonic() - t0:.0f}s)')
    elif args.cmd == 'picks':
        path = run_picks(args.arm, root=args.root, export_root=args.export_root, workers=args.workers)
        print(f'picks {args.arm} -> {path} ({time.monotonic() - t0:.0f}s)')
    elif args.cmd == 'cells':
        path = run_cells(args.arm, root=args.root, export_root=args.export_root, types=args.types, workers=args.workers,
                         allow_uncommitted=args.allow_uncommitted)
        print(f'cells {args.arm} -> {path} ({time.monotonic() - t0:.0f}s)')
    elif args.cmd == 'gate-v-mmart':
        ok, details = gate_v_mmart(workers=args.workers, perquery_dir=P.OUT / 'gate_v_mmart' / 'perquery')
        rec = write_gate(rec_dir / 'gates', 'v_mmart', ok, details)
        print(f'gate v_mmart: {rec["status"]} tries {rec["tries"]} max_abs_diff {details["max_abs_diff"]:.3g} '
              f'failed {details["failed_comparators"]} ({time.monotonic() - t0:.0f}s)')
        return 0 if ok else 1
    elif args.part == 'export':
        out, n = _gate_try_dir(args.n)
        ok, details = gate_vi_export(out, device=args.device)
        write_json(out / 'export_result.json', dict(try_number=n, ok=ok, written=now(), **details))
        print(f'gate vi export try{n}: {"equal" if ok else "DIFFERENT"} {details["n_files"]} files ({time.monotonic() - t0:.0f}s) {out}')
        return 0 if ok else 1
    else:
        assert args.n, 'gate-vi score needs --try N (the export part\'s folder)'
        out, n = _gate_try_dir(args.n)
        exp = json.loads((out / 'export_result.json').read_text())
        ok, details = gate_vi_score(out, tuple(args.aspects), dict(ami=args.draws_ami, map=args.draws_map), not args.no_validation,
                                    args.workers)
        full = (set(args.aspects) == set(ASPECTS) and details['intervals_checked'] and 'picks' in details
                and exp['n_files'] == len(KEYSETS) * len(ASPECTS) * len(SEEDS))
        both = ok and exp['ok']
        details = dict(export=exp, score=details)
        if full:
            rec = write_gate(rec_dir / 'gates', 'vi', both, details)
            print(f'gate vi: {rec["status"]} tries {rec["tries"]} ({time.monotonic() - t0:.0f}s)')
        else:
            write_json(out / 'score_partial.json', dict(ok=both, **details))
            print(f'gate vi (reduced check, no gate record): score {"ok" if ok else "FAIL"} -> {out / "score_partial.json"} '
                  f'({time.monotonic() - t0:.0f}s)')
        return 0 if both else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
