"""r29 ticket 13: photo cells of F1 (SigLIP 1 photo model, claim SC6): Affection AMI (k = 9) and mAP, EmoSet AMI (k = 8) and mAP,
each against P1 (plain SigLIP 1) and Q1 (the Qwen3 teacher), with their validation picks, ranges and the gates that check the
comparators and the scorer.

Spec docs/superpowers/specs/2026-11-12-siglip-only-adapter-design.md §3.4 (cells table rows 5 and 6, rebuild paragraph, order),
§3.5 gates (v) photo rebuilds and (vii) scorer part, §6 SC6, D18, D19, D23.

Rows (F1's exports, r14's photo layout `<root>/exports/<set>/photo_siglip1/<split>__s<seed>.npz`; `r29_photo.py export --dataset`):
  AMI  the stored float16 rows cast to float32 (not renormalised), `ns_metrics.partitions` k-means at the class count, five seeds
       (r28 `r28_photos.score_cell`, r17 `ns_clu`). Interval: `ns_clu.paired_cell`, 2,000 draws (rows, then seeds), r17's form, for
       both sets. LEVEL unit r: F1's five-seed range of validation AMI at the class count.
  mAP  unit float32 rows, the read-out picked on validation (Affection: r12's rule `pf_stage2.readout`; EmoSet: r14 part c step 2
       `bs_emoset_score.readout_doc`), fitted on all validation rows and applied to test (`bs_emoset_score.test_retrieval`), paired
       10,000 draws (`rt_stats.paired`, queries then seeds). r: Affection, F1's own five-seed validation range of mAP at R0 (as r28
       for c*); EmoSet, r14's committed r_E (D19).
  Comparators (unseeded), rebuilt from the stores: plain = the SigLIP 1 default image, teacher = Qwen3 image in the "artwork" wording
       (`ns_photo.plain`, `ns_photo.teacher`'s views). mAP at the older round's recorded pick (`center+k10`), AMI by k-means.
Order rule: `picks --arm K` reads validation only and writes `records/photo_picks_<K>.json` once; `cells --arm K` refuses without
`records/pass_rule.json` saying pass, without the picks record committed unchanged in HEAD, and without `records/gates/v_photo.json`
("gate v_photo not run"). A comparator unit that gate (v) did not rebuild parks only the cells that use it (INCONCLUSIVE).
Gates (stage 0, no pass rule): `gate-v-photo` rebuilds the comparator values from the stores against the older records (1e-9);
`gate-vii-photo` runs r14's photo exports through this scorer against r28 `photos.json` cell a and r20's Affection photo cells.
Gate status: pass; fail before try 3; parked at try 3 (`r29_captions.write_gate_record` counts the tries).

CLI (from the round folder):  python r29_photo_cells.py [--workers N] picks --arm K [--root R] | cells --arm K [--root R]
    [--allow-uncommitted] | gate-v-photo [--no-record] | gate-vii-photo [--no-record]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r29_paths as P  # noqa: E402

P.use_older_rounds()
import ns_paths  # noqa: E402

ns_paths.add_rounds()
import ns_photo  # noqa: E402  (r17; puts r14's bs_*, r13's pf_* on the path)
import ns_clu  # noqa: E402
import ns_metrics  # noqa: E402
import bs_paths  # noqa: E402
import bs_stores  # noqa: E402
import bs_emoset_score  # noqa: E402
import pf_stage2  # noqa: E402
import ac_data  # noqa: E402
import rt_metrics  # noqa: E402
import rt_stats  # noqa: E402
import rs_vh  # noqa: E402
import r29_image_cells as IC  # noqa: E402  (outcome rule, now, records_dir, sha_file)
import r29_captions as C  # noqa: E402  (write_gate_record counts the tries)
P.check_older_imports()

SETS = ('affection', 'emoset')
SEEDS = tuple(P.SEEDS)
K_CLASS = {'affection': 9, 'emoset': 8}
N_DRAWS = dict(ami=2000, map=10000)               # r17 SC4; r14 SC5 / SC10
TOL = 1e-9                                        # D23
ENC = 'siglip1'                                   # F1's run folder `<root>/photo/siglip1/s<seed>`
ARM_NAME = 'photo_siglip1'                        # r14's arm name in the export folder and sidecar
COMPARATORS = ('plain', 'teacher')
CAP_WIDTH = 1152
MAX_TRIES = 3
BRIEF_LITERALS = {'affection/map/teacher': 0.2945474692, 'emoset/map/teacher': 0.4518753521, 'affection/map/plain': 0.2545391742,
                  'emoset/map/plain': 0.3638545779, 'emoset/ami/plain': 0.2672607188}   # spec §3.5 (v), 10 decimals
REQUIRED_UNITS = tuple(BRIEF_LITERALS)

outcome = IC.outcome


def records_dir():
    return IC.records_dir()


def gates_dir():
    return P.GATES if P.OUT == P.DEFAULT_OUT else P.OUT / 'records' / 'gates'


# ================================================================================================ the export reader
def read_export(path, set_name, split, seed, keys, uncut_keys, checkpoint=None):
    """F1's float16 export rows in `keys` order, joined by key. Asserts: sidecar sha256 equals the file's; sidecar arm
    `photo_siglip1`, split, seed, dataset; optional checkpoint path (`checkpoint`); npz holds exactly `keys` and `emb`; float16,
    finite, 2,048 wide; the file's key set equals `uncut_keys`; unit norm."""
    path = Path(path)
    side = json.loads(path.with_suffix('.json').read_text())
    assert side['sha256'] == rs_vh.file_sha(path), f'{path}: file differs from its sidecar'
    got = (side['arm'], side['split'], side['seed'], side['dataset'])
    assert got == (ARM_NAME, split, seed, set_name), f'{path}: sidecar says {got}, asked {(ARM_NAME, split, seed, set_name)}'
    if checkpoint is not None:
        assert side['checkpoint'] == str(checkpoint), f'{path}: sidecar checkpoint {side["checkpoint"]} is not {checkpoint}'
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) == {'keys', 'emb'}, (path, z.files)
        file_keys, emb = z['keys'].tolist(), z['emb']
    assert emb.dtype == np.float16 and emb.ndim == 2 and emb.shape[1] == ns_paths.CSTAR_WIDTH and np.isfinite(emb).all(), \
        (path, emb.dtype, emb.shape)
    out = ns_photo.join_rows(file_keys, emb, keys, str(path), exact=uncut_keys)
    dev = float(np.abs(np.linalg.norm(out.astype(np.float32), axis=1) - 1).max())
    assert dev < ns_paths.TOL_UNIT, f'{path}: rows are not unit (max deviation {dev})'
    return out


def f1_identity(root, arm):
    """The five run records under `<root>/photo/siglip1/s<seed>/run.json` say F1 was trained on caption source `arm`, 1,152 wide,
    from one caption store (`cap_store_sha256`)."""
    runs, shas = {}, {}
    for s in SEEDS:
        path = Path(root) / 'photo' / ENC / f's{s}' / 'run.json'
        runs[s] = json.loads(path.read_text())
        shas[str(s)] = IC.sha_file(path)
        assert runs[s].get('cap_source') == arm, f'{path}: cap_source {runs[s].get("cap_source")!r}, not {arm!r}'
        assert runs[s].get('cap_width') == CAP_WIDTH, f'{path}: cap_width {runs[s].get("cap_width")}'
    stores = {r.get('cap_store_sha256') for r in runs.values()}
    assert len(stores) == 1 and None not in stores, f'one cap_store_sha256 across the five seeds, got {stores}'
    return dict(cap_source=arm, cap_width=CAP_WIDTH, cap_store_sha256=stores.pop(), run_json_sha256=shas)


# ================================================================================================ data
class RealData:
    """The stores and F1's exports as the module reads them (tests pass a fake with the same methods). `export_root`: where the
    photo exports are (default `OUT/exports`: ticket 12's; gate (vii) passes r14's); `root`: F1's model root, when given the
    exports' checkpoint paths are checked against it."""

    def __init__(self, export_root=None, root=None):
        self.root = None if root is None else Path(root)
        self.export_root = Path(export_root) if export_root is not None else P.OUT / 'exports'

    def keys(self, set_name, split):
        return ns_photo.keys(set_name, split)

    def labels(self, set_name, split):
        return ns_photo.labels(set_name, split)

    def _path(self, set_name, split, seed):
        return self.export_root / set_name / ARM_NAME / f'{split}__s{seed}.npz'

    def f1(self, set_name, split, seed):
        ckpt = None if self.root is None else self.root / 'photo' / ENC / f's{seed}' / 'emotion.pt'
        return read_export(self._path(set_name, split, seed), set_name, split, seed, self.keys(set_name, split),
                           ns_photo._uncut_keys(set_name, split), checkpoint=ckpt)

    def f1_sha(self, set_name, split, seed):
        return IC.sha_file(self._path(set_name, split, seed))

    def plain(self, set_name, split):
        return ac_data._unit(bs_stores.load(set_name, 'siglip1', self.keys(set_name, split)))

    def teacher(self, set_name, split):
        return ac_data._unit(bs_stores.load(set_name, ns_photo.teacher_wording(set_name), self.keys(set_name, split)))


# ================================================================================================ picks
def read_r_E():
    """(r_E,mAP, source): r14's committed stage 2 record `ranges.map.r` (EmoSet's LEVEL unit for mAP, D19)."""
    path = Path(ns_photo.R14_STAGE2_EMOSET)
    r = float(json.loads(path.read_text())['ranges']['map']['r'])
    assert np.isfinite(r) and r > 0, r
    return r, f'r14 stage2_emoset.json ({path}): ranges.map.r (r_E,mAP)'


def pick_readout(set_name, xs_val, y_val, keys_val, r_map, workers):
    """The validation read-out: Affection r12's rule (`pf_stage2.readout`, sorted keys); EmoSet r14 part c step 2
    (`bs_emoset_score.readout_doc`, after the sha256 order is asserted). {'pick', 'scores'}."""
    if set_name == 'emoset':
        bs_emoset_score.check_order(keys_val)
        return bs_emoset_score.readout_doc(xs_val, y_val, r_map, workers)
    return pf_stage2.readout(xs_val, r_map, labels=y_val, keys=keys_val, workers=workers)


def _raw(x16):
    return np.asarray(x16).astype(np.float32)           # the clustering form: stored float16 cast, not renormalised


def _parts(rows_by_seed, k, set_name):
    return np.stack([ns_metrics.partitions(_raw(rows_by_seed[s]), k, set_name) for s in SEEDS])


def compute_picks(data, arm, workers=8, sets=SETS):
    """F1's validation picks and ranges per set (validation rows only: `data` is asked for 'val' alone). AMI range at the class
    count; mAP range (Affection only: F1's own five-seed R0 validation mAP); the read-out pick."""
    out = {}
    for s in sets:
        k = K_CLASS[s]
        keys, y = data.keys(s, 'val'), data.labels(s, 'val')
        x16 = {sd: data.f1(s, 'val', sd) for sd in SEEDS}
        unit = {sd: ac_data._unit(x16[sd]) for sd in SEEDS}
        ami = ns_clu.ranges(_parts(x16, k, s), y, k, ('ami',))['ami']
        ami_range = dict(r=float(ami['r']), per_seed={str(sd): float(v) for sd, v in ami['per_seed'].items()}, k=k)
        if s == 'affection':
            per = []
            for sd in SEEDS:
                res = rt_metrics.retrieval(unit[sd], y, qe_k=0, workers=workers)
                assert res['dropped'] == 0 and len(res['rows']) == len(y), (s, sd, res['dropped'])
                per.append(float(res['ap'].mean()))
            r_map = rt_stats.seed_range(per)
            assert np.isfinite(r_map) and r_map > 0, r_map
            map_range = dict(r=float(r_map), per_seed={str(sd): v for sd, v in zip(SEEDS, per)})
            source = 'own five-seed validation mAP range at R0'
        else:
            r_map, source = read_r_E()
            map_range = None
        doc = pick_readout(s, unit, y, keys, r_map, workers)
        out[s] = dict(k=k, n_val=int(len(y)), ami_range=ami_range, map_range=map_range, r_map=float(r_map), r_map_source=source,
                      readout=dict(pick=doc['pick'], val_scores={n: float(v['map']['emotion']) for n, v in doc['scores'].items()}),
                      val_files={str(sd): data.f1_sha(s, 'val', sd) for sd in SEEDS})
    return dict(arm=arm, sets=out)


# ================================================================================================ comparator records
def _json(path):
    return json.loads(Path(path).read_text())


def comparator_records():
    """{(set, 'plain'|'teacher'): dict(map_value, map_pick, ami_value, source)} from the older records (spec §3.4's table):
    Affection plain r14 SC5, teacher r13 stage 3 `teacher_artwork`; EmoSet plain r28 `photos.json` cell c, teacher r14 SC10.
    AMI values: r17 SC4 comparators (cell c for EmoSet plain)."""
    v14 = _json(P.R14_ROUND / 'records/verdicts.json')['cells']
    v17 = {c['id']: c for c in _json(P.R17_ROUND / 'records/verdicts.json')['cells']}

    def one(**want):
        (c,) = [c for c in v14 if all(c.get(k) == v for k, v in want.items())]
        return c
    sc5 = one(sc='SC5', encoder='siglip1', task='ret', data='affection_test', candidate='photo_siglip1', comparator='frozen_siglip1')
    sc10 = one(sc='SC10', encoder='qwen', task='ret', data='emoset_test', candidate='photo_qwen', comparator='teacher_emotion_focus')
    r13 = _json(Path(bs_paths.R13_OUT) / 'stage3/verdict.json')['rows']['teacher_artwork']
    c28 = _json(P.R28_RECORDS / 'photos.json')['cells']['c']
    ami = lambda i: v17[i]['comp']  # noqa: E731
    out = {
        ('affection', 'plain'): dict(map_value=sc5['comparator_value'], map_pick=sc5['readout_comparator'],
                                     ami_value=ami('SC4/affection/siglip1/emotion/ami/9/plain'),
                                     source='r14 verdicts.json SC5 siglip1 ret affection_test; r17 SC4 plain comp'),
        ('affection', 'teacher'): dict(map_value=r13['map_picked']['map'], map_pick=r13['readout'],
                                       ami_value=ami('SC4/affection/siglip1/emotion/ami/9/teacher'),
                                       source='r13 stage3/verdict.json rows.teacher_artwork; r17 SC4 teacher comp'),
        ('emoset', 'plain'): dict(map_value=c28['map']['mean'], map_pick=c28['readout'], ami_value=c28['ami']['mean'],
                                  source='r28 photos.json cells.c'),
        ('emoset', 'teacher'): dict(map_value=sc10['comparator_value'], map_pick=sc10['readout_comparator'],
                                    ami_value=ami('SC4/emoset/qwen/emotion/ami/8/teacher'),
                                    source='r14 verdicts.json SC10 emoset_test; r17 SC4 qwen emoset teacher comp'),
    }
    return out


# ================================================================================================ gate records
def _prior_tries(path):
    return json.loads(Path(path).read_text())['tries'] if Path(path).exists() else 0


def write_gate(gate, ok, details, folder):
    """`<folder>/<gate>.json`: pass; fail before try 3; parked at try 3 (`r29_captions.write_gate_record` counts the tries and
    raises on a fourth)."""
    prior = _prior_tries(Path(folder) / f'{gate}.json')
    assert prior < MAX_TRIES, f'gate {gate}: three tries exhausted, parked'
    status = 'pass' if ok else ('parked' if prior == MAX_TRIES - 1 else 'fail')
    return C.write_gate_record(folder, dict(status=status, details=details), gate=gate)


def read_gate(folder, gate):
    path = Path(folder) / f'{gate}.json'
    if not path.exists():
        raise RuntimeError(f'gate {gate} not run: no {path}')
    rec = _json(path)
    assert rec.get('gate') == gate, path
    return rec


def compare_unit(cell, rebuilt, recorded, required, **extra):
    """One unit of a gate: rebuilt value against the record within 1e-9 (finite). `literal` (extra): the record must round to it
    at 10 decimals (the brief's printed value)."""
    diff = abs(float(rebuilt) - float(recorded))
    ok = bool(np.isfinite(diff) and diff <= TOL)
    if 'literal' in extra:
        ok = ok and round(float(recorded), 10) == extra['literal']
    return dict(cell=cell, rebuilt=float(rebuilt), recorded=float(recorded), diff=float(diff), ok=ok, required=bool(required), **extra)


def judge_units(units):
    """(all units ok, sorted names of the units that are not ok). An extra unit that differs also parks its cells."""
    bad = sorted(n for n, u in units.items() if not u['ok'])
    return not bad, bad


# ================================================================================================ scoring
def _cand_ret(data, s, pick, y_test, workers):
    val = {sd: ac_data._unit(data.f1(s, 'val', sd)) for sd in SEEDS}
    test = {sd: ac_data._unit(data.f1(s, 'test', sd)) for sd in SEEDS}
    return bs_emoset_score.test_retrieval(val, test, pick, y_test, True, workers)


def _comp_rows(data, s, comp):
    return (data.plain if comp == 'plain' else data.teacher)(s, 'val'), (data.plain if comp == 'plain' else data.teacher)(s, 'test')


def _comp_ret(data, s, comp, pick, y_test, workers):
    xv, xt = _comp_rows(data, s, comp)
    return bs_emoset_score.test_retrieval({None: xv}, {None: xt}, pick, y_test, False, workers)


def _comp_ami_value(data, s, comp, y_test):
    k = K_CLASS[s]
    _, xt = _comp_rows(data, s, comp)
    return ns_clu.metric_table(comp, 'emotion', ns_metrics.partitions(xt, k, s), y_test, k, ('ami',))[0]['mean']


def _cell(s, arm, comp, metric, res, r, n_rows, **extra):
    k = K_CLASS[s] if metric == 'ami' else None
    vals = [res['cand'], res['comp'], res['d'], res['lo'], res['hi']]
    assert all(np.isfinite(v) for v in vals), (s, comp, metric, vals)
    return dict(id=f'{s}/siglip1/{arm}/emotion/{metric}{"" if k is None else k}/{comp}', type=metric,
                protocol='r17_SC4' if metric == 'ami' else ('r14_SC5' if s == 'affection' else 'r14_SC10'),
                set=s, encoder='siglip1', arm=arm, aspect='emotion', metric=metric, k=k, comparator=comp,
                value=float(res['cand']), comparator_value=float(res['comp']), d=float(res['d']), lo=float(res['lo']),
                hi=float(res['hi']), r=float(r), outcome=outcome(res['lo'], r), n_draws=int(res['n_draws']), n_rows=int(n_rows),
                **extra)


def _parked_cell(s, arm, comp, metric, r, unit):
    k = K_CLASS[s] if metric == 'ami' else None
    return dict(id=f'{s}/siglip1/{arm}/emotion/{metric}{"" if k is None else k}/{comp}', type=metric, protocol=None, set=s,
                encoder='siglip1', arm=arm, aspect='emotion', metric=metric, k=k, comparator=comp, value=None,
                comparator_value=None, d=None, lo=None, hi=None, r=None if r is None else float(r), outcome='INCONCLUSIVE',
                n_draws=None, n_rows=None, reason=f'gate (v) photo: the rebuilt comparator {unit} differs from its record')


def score_cells(data, picks, comps, parked=(), draws=None, workers=8, sets=SETS):
    """The photo cells from a picks body (or record): per set, AMI and mAP against plain and teacher. `parked`: unit names
    `<set>/<ami|map>/<plain|teacher>` that gate (v) did not confirm; their cells are INCONCLUSIVE and read no rows."""
    draws = dict(N_DRAWS, **(draws or {}))
    parked = set(parked)
    arm, cells = picks['arm'], []
    for s in sets:
        sp, k = picks['sets'][s], K_CLASS[s]
        y_test = data.labels(s, 'test')
        live = {m: [c for c in COMPARATORS if f'{s}/{m}/{c}' not in parked] for m in ('ami', 'map')}
        if live['ami']:
            cand_parts = _parts({sd: data.f1(s, 'test', sd) for sd in SEEDS}, k, s)
        if live['map']:
            cand = _cand_ret(data, s, sp['readout']['pick'], y_test, workers)
        for metric in ('ami', 'map'):
            for comp in COMPARATORS:
                unit = f'{s}/{metric}/{comp}'
                r = sp['ami_range']['r'] if metric == 'ami' else sp['r_map']
                if unit in parked:
                    cells.append(_parked_cell(s, arm, comp, metric, r, unit))
                elif metric == 'ami':
                    _, xt = _comp_rows(data, s, comp)
                    res = ns_clu.paired_cell(cand_parts, ns_metrics.partitions(xt, k, s), y_test, k, r=None, metrics=('ami',),
                                             draws=draws['ami'], rng_seed=ns_paths.BOOT_SEED)['ami']
                    cells.append(_cell(s, arm, comp, 'ami', res, r, len(y_test)))
                else:
                    pick = comps[(s, comp)]['map_pick']
                    cmp_ = _comp_ret(data, s, comp, pick, y_test, workers)
                    assert np.array_equal(cand['rows'], cmp_['rows']), (s, comp, 'the arms were scored on other queries')
                    p = rt_stats.paired(cand['ap'], cmp_['ap'], draws['map'])
                    res = dict(cand=cand['point'], comp=cmp_['point'], d=p['d'], lo=p['interval'][0], hi=p['interval'][1],
                               n_draws=draws['map'])
                    cells.append(_cell(s, arm, comp, 'map', res, r, len(y_test), candidate_readout=sp['readout']['pick'],
                                       comparator_readout=pick, n_queries=int(len(cand['rows']))))
    return cells


# ================================================================================================ the commands
def _require_pass(records):
    path = Path(records) / 'pass_rule.json'
    if not path.exists() or _json(path).get('status') != 'pass':
        raise RuntimeError(f'stage 2 command refused: {path} must exist and say status pass (D3, pass_rule)')


def _picks_path(records, arm):
    return Path(records) / f'photo_picks_{arm}.json'


def _comp_picks(comps):
    return {f'{s}/{c}': comps[(s, c)]['map_pick'] for s in SETS for c in COMPARATORS}


def run_picks(arm, data=None, records=None, gates=None, comps=None, workers=8, root=None):
    """`picks`: F1's validation picks and ranges -> `records/photo_picks_<arm>.json` (once). `gates` is accepted for symmetry with
    `run_cells` and unused: picks need no gate record."""
    records = Path(records or records_dir())
    _require_pass(records)
    if data is None:
        data = RealData(export_root=None if root is None else Path(root) / 'exports', root=root)
    comps = comps or comparator_records()
    path = _picks_path(records, arm)
    if path.exists():
        raise FileExistsError(f'{path} exists: a picks record is written once per arm (the controller moves it for a new try)')
    body = compute_picks(data, arm, workers)
    r_E, r_E_source = read_r_E()
    rec = dict(encoder='siglip1', split='val', written=IC.now(), code_sha256=IC.sha_file(__file__), **body,
               comparator_picks=_comp_picks(comps), r_E=dict(value=r_E, source=r_E_source))
    if isinstance(data, RealData) and data.root is not None:
        rec['f1'] = f1_identity(data.root, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1) + '\n')
    return path


def run_cells(arm, data=None, records=None, gates=None, comps=None, draws=None, workers=8, check_committed=True, root=None):
    """`cells`: the eight photo cells -> `records/photo_cells_<arm>.json`. Refuses, in this order: the pass rule not pass; no picks
    record, or (check_committed) not committed unchanged in HEAD; no gate v_photo record. Units gate (v) did not confirm park
    their cells."""
    records = Path(records or records_dir())
    gates = Path(gates or records / 'gates')
    _require_pass(records)
    pick_file = _picks_path(records, arm)
    why = f'run `picks --arm {arm}` and commit {pick_file.name} before any test cell (order rule, C13)'
    if check_committed:
        P.require_committed(pick_file, why)
    elif not pick_file.exists():
        raise RuntimeError(f'no {pick_file.name} at {pick_file}: {why}')
    picks = _json(pick_file)
    assert picks['arm'] == arm and picks['encoder'] == 'siglip1', (pick_file, picks['arm'], picks['encoder'])
    gate = read_gate(gates, 'v_photo')
    units = gate['details']['units']
    _, parked = judge_units(units)
    if data is None:
        data = RealData(export_root=None if root is None else Path(root) / 'exports', root=root)
    comps = comps or comparator_records()
    assert picks['comparator_picks'] == _comp_picks(comps), 'the comparators\' recorded picks differ from the picks record\'s'
    for s in SETS:
        for sd in SEEDS:
            assert data.f1_sha(s, 'val', sd) == picks['sets'][s]['val_files'][str(sd)], \
                f'{s}: validation export of seed {sd} changed since the picks were written'
    if isinstance(data, RealData) and data.root is not None and 'f1' in picks:
        assert f1_identity(data.root, arm) == picks['f1'], 'F1\'s run records changed since the picks were written'
    cells = score_cells(data, picks, comps, parked=parked, draws=draws, workers=workers)
    rec = dict(arm=arm, encoder='siglip1', picks_sha256=IC.sha_file(pick_file), f1=picks.get('f1'),
               test_files={s: {str(sd): data.f1_sha(s, 'test', sd) for sd in SEEDS} for s in SETS},
               v_photo=dict(status=gate['status'], tries=gate['tries'], parked=parked), written=IC.now(), cells=cells)
    path = records / f'photo_cells_{arm}.json'
    path.write_text(json.dumps(rec, indent=1) + '\n')
    return path


# ================================================================================================ gate (v) photo
def rebuild_units(data, comps, workers=8):
    """Every comparator value rebuilt from the stores at the recorded pick, as compare_unit dicts keyed `<set>/<ami|map>/<comp>`.
    Required: the brief's five; the three other r17 AMI comparators are extras (they also park their cells)."""
    units = {}
    for s in SETS:
        y_test = data.labels(s, 'test')
        for comp in COMPARATORS:
            c = comps[(s, comp)]
            res = _comp_ret(data, s, comp, c['map_pick'], y_test, workers)
            name = f'{s}/map/{comp}'
            units[name] = compare_unit(name, res['point'], c['map_value'], True, readout=c['map_pick'], source=c['source'],
                                       literal=BRIEF_LITERALS[name])
            name = f'{s}/ami/{comp}'
            units[name] = compare_unit(name, _comp_ami_value(data, s, comp, y_test), c['ami_value'], name in REQUIRED_UNITS,
                                       source=c['source'], **({'literal': BRIEF_LITERALS[name]} if name in BRIEF_LITERALS else {}))
    return units


def gate_v_photo(data=None, workers=8):
    data = data or RealData()
    t0 = time.monotonic()
    units = rebuild_units(data, comparator_records(), workers)
    ok, parked = judge_units(units)
    required_ok = all(units[n]['ok'] for n in REQUIRED_UNITS)
    return ok, dict(units=units, parked=parked, required_ok=required_ok, tol=TOL, seconds=round(time.monotonic() - t0),
                    code_sha256=IC.sha_file(__file__))


# ================================================================================================ gate (vii) photo
def _same(units, name, rebuilt, recorded, **extra):
    units[name] = compare_unit(name, rebuilt, recorded, extra.pop('required', True), **extra)


def gate_vii_photo(data=None, workers=8):
    """r14's photo exports through THIS scorer (picks, then the candidate's test scalars) against r28 `photos.json` cell a (EmoSet:
    pick, AMI and mAP means and per seed) and r20's Affection photo cells (AMI, mAP, with r14 SC5's pick). Extras: Affection's
    AMI range against r17 `stage1_ranges.json`, and r17's SC4 AMI cells and r14's SC5 mAP cell replayed through `score_cells`."""
    data = data or RealData(export_root=ns_photo.EXPORT_ROOT)
    t0 = time.monotonic()
    comps = comparator_records()
    picks = compute_picks(data, 'siglip1', workers)
    units = {}
    a = _json(P.R28_RECORDS / 'photos.json')['cells']['a']
    r20 = _json(P.OLDER / '20261111_encoder_photo_tables/records/cells.json')
    v14 = _json(P.R14_ROUND / 'records/verdicts.json')['cells']
    (sc5,) = [c for c in v14 if (c['sc'], c['encoder'], c['task'], c['data'], c['candidate'], c['comparator']) ==
              ('SC5', 'siglip1', 'ret', 'affection_test', 'photo_siglip1', 'frozen_siglip1')]
    for s in SETS:
        sp, k = picks['sets'][s], K_CLASS[s]
        y_test = data.labels(s, 'test')
        want_pick = a['readout'] if s == 'emoset' else sc5['readout_candidate']
        units[f'{s}/photo/pick'] = compare_unit(f'{s}/photo/pick', float(sp['readout']['pick'] == want_pick), 1.0, True,
                                                pick=sp['readout']['pick'], recorded_pick=want_pick)
        parts = _parts({sd: data.f1(s, 'test', sd) for sd in SEEDS}, k, s)
        ami = ns_clu.metric_table('photo_siglip1', 'emotion', parts, y_test, k, ('ami',))[0]
        ret = _cand_ret(data, s, sp['readout']['pick'], y_test, workers)
        if s == 'emoset':
            _same(units, 'emoset/photo/ami/mean', ami['mean'], a['ami']['mean'])
            _same(units, 'emoset/photo/map/mean', ret['point'], a['map']['mean'])
            for sd in SEEDS:
                _same(units, f'emoset/photo/ami/s{sd}', ami['per_seed'][sd], a['ami']['per_seed'][sd])
                _same(units, f'emoset/photo/map/s{sd}', ret['per_seed'][sd], a['map']['per_seed'][sd])
        else:
            _same(units, 'affection/photo/ami/mean', ami['mean'], r20['photo/affection/siglip1/photo/ami']['value'])
            _same(units, 'affection/photo/map/mean', ret['point'], r20['photo/affection/siglip1/photo/map']['value'])
    rng = _json(P.R17_ROUND / 'records/stage1_ranges.json')['photo/affection/siglip1/emotion/ami']
    ar = picks['sets']['affection']['ami_range']
    _same(units, 'affection/photo/ami_range/r', ar['r'], rng['r'], required=False)
    for sd in SEEDS:
        _same(units, f'affection/photo/ami_range/s{sd}', ar['per_seed'][str(sd)], rng['per_seed'][str(sd)], required=False)
    v17 = {c['id']: c for c in _json(P.R17_ROUND / 'records/verdicts.json')['cells']}
    cells = {c['id']: c for c in score_cells(data, picks, comps, workers=workers, sets=('affection',))}
    for comp in COMPARATORS:
        mine, rec = cells[f'affection/siglip1/siglip1/emotion/ami9/{comp}'], v17[f'SC4/affection/siglip1/emotion/ami/9/{comp}']
        for f, g in (('value', 'cand'), ('comparator_value', 'comp'), ('d', 'd'), ('lo', 'lo'), ('hi', 'hi'), ('r', 'r')):
            _same(units, f'affection/SC4/ami9/{comp}/{f}', mine[f], rec[g], required=False)
    mine = cells['affection/siglip1/siglip1/emotion/map/plain']
    for f, g in (('value', 'candidate_value'), ('comparator_value', 'comparator_value'), ('d', 'd'), ('lo', 'lo'), ('hi', 'hi')):
        _same(units, f'affection/SC5/map/plain/{f}', mine[f], sc5[g], required=False)
    ok, bad = judge_units(units)
    return ok, dict(units=units, parked=bad, required_ok=all(u['ok'] for u in units.values() if u['required']), tol=TOL,
                    picks={s: dict(pick=picks['sets'][s]['readout']['pick'], r_map=picks['sets'][s]['r_map'],
                                   ami_r=picks['sets'][s]['ami_range']['r']) for s in SETS},
                    seconds=round(time.monotonic() - t0), code_sha256=IC.sha_file(__file__))


# ================================================================================================ CLI
def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name, h in (('picks', 'validation picks and ranges -> records/photo_picks_<arm>.json (once)'),
                    ('cells', 'the eight photo cells -> records/photo_cells_<arm>.json')):
        s = sub.add_parser(name, help=h)
        s.add_argument('--arm', required=True, help='F1\'s caption source arm, e.g. siglip1')
        s.add_argument('--root', type=Path, help='model root (default OUT): exports under <root>/exports, runs under <root>/photo')
    sub.choices['cells'].add_argument('--allow-uncommitted', action='store_true',
                                      help='smoke only: do not require the picks record to be committed')
    for name in ('gate-v-photo', 'gate-vii-photo'):
        sub.add_parser(name).add_argument('--no-record', action='store_true', help='run the check, write no gate record')
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    t0 = time.monotonic()
    if args.cmd in ('gate-v-photo', 'gate-vii-photo'):
        gate, fn, data = (('v_photo', gate_v_photo, RealData()) if args.cmd == 'gate-v-photo'
                          else ('vii_photo', gate_vii_photo, RealData(export_root=ns_photo.EXPORT_ROOT)))
        ok, details = fn(data, args.workers)
        status = 'pass' if ok else 'fail'
        if not args.no_record:
            status = write_gate(gate, ok, details, gates_dir())['status']
        worst = max(u['diff'] for u in details['units'].values())
        print(f'gate {gate}: {status} (not recorded: {args.no_record}) units {len(details["units"])} max diff {worst:.3g} '
              f'not ok {details["parked"]} ({time.monotonic() - t0:.0f}s)')
        return 0 if ok else 1
    root = args.root or P.OUT
    if args.cmd == 'picks':
        path = run_picks(args.arm, root=root, workers=args.workers)
    else:
        path = run_cells(args.arm, root=root, workers=args.workers, check_committed=not args.allow_uncommitted)
    print(f'{args.cmd} {args.arm}: {path} ({time.monotonic() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
