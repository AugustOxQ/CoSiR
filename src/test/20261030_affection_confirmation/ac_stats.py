"""Statistics and the verdict for r12 (spec 2026-10-30 §3.4 "Seed ranges", "Retrieval", "Outcomes"; §6; D4, D7, D10).

Wrappers of r11's code (imported read-only, not copied): `rt_stats.seed_range`, `outcome` and `paired`;
`rt_metrics.retrieval` and `graded`; `rt_readout.stage5_apply`. What differs from r11 is wrapped here: the table is
SC1 to SC8, the comparators follow the stage 1 picks, each SC has its own r, and emotion is the only Affection aspect.

  seed_range, outcome        r11's, re-exported
  ranges_doc                 the three seed ranges in `stage0b/ranges.json`'s form (unrounded, per-seed values kept)
  measure_seed_ranges        r_clu, r_ret,mAP, r_ret,nDCG from c*'s per-seed validation emotion arrays
  SC, r_for, comparator_arm, readouts
                             the SC1 to SC8 table; the r, comparator arm and read-out names of each row
  sc_pass, verdict, repeats_line, claims
                             bars, the stage 2 verdict, the yes/no line, the stage 3 CRL claims
  arm_retrieval, retrieval_sc, cluster_sc, sc_entry, stage2_doc, stage3_doc
                             the retrieval comparison with picked read-outs and the verdict files' `sc` entries
  replay_r11_stage5          spec §3.5 0a(iii), r11 stage 5 part

Pure functions: no file access except `load_ranges` (reads a ranges file), `r11_r_a` and `replay_r11_stage5` (read r11's
frozen inputs), `write_ranges` (writes) and `validation_seed_ranges` (reads c*'s validation arrays through `ac_data`).
"""
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np

import ac_paths  # puts r11's round on sys.path
import rt_metrics
import rt_readout
import rt_stats
from rt_stats import outcome, seed_range  # noqa: F401  (re-exported: r11's functions)

SEEDS = tuple(ac_paths.SEEDS)
OUTCOMES = rt_stats.OUTCOMES
BARS = rt_stats.BARS


# ---- seed ranges (spec §3.4) ---------------------------------------------------------------------------------------

def ranges_doc(clu_per_seed, map_per_seed, ndcg_per_seed):
    """`stage0b/ranges.json`: {"clu": {"per_seed": [5], "r": f}, "map": ..., "ndcg": ...}, unrounded. Every range must
    be positive (a zero range would make LEVEL impossible and the read-out scores undefined: stop and note it)."""
    doc = {}
    for name, values in (('clu', clu_per_seed), ('map', map_per_seed), ('ndcg', ndcg_per_seed)):
        per_seed = [float(v) for v in values]
        r = seed_range(per_seed)
        assert np.isfinite(r) and r > 0, f'seed range {name} is {r}'
        doc[name] = {'per_seed': per_seed, 'r': r}
    return doc


def measure_seed_ranges(get_x, get_cluster, labels, dist, workers=8):
    """The three seed ranges of spec §3.4 from c*'s emotion head on all validation images. get_x(seed) -> float32 unit
    rows for retrieval; get_cluster(seed) -> checked stored float16 rows for clustering, seed 0 to 4.
    The rows are queries and gallery together; labels: int64 [n]; dist: float64 [n, 9], the
    graded relevance. Per seed: the mean over three states of the AMI of r10's k = 20 partitions (r_clu); mAP of
    plain cosine R0 (r_ret,mAP); nDCG of plain cosine with the overlap gain (r_ret,nDCG). Returns `ranges_doc`."""
    import ac_cluster
    labels = np.asarray(labels)
    clu, mp, nd = [], [], []
    for seed in SEEDS:
        x = get_x(seed)
        clu.append(ac_cluster.ami(ac_cluster.partitions(get_cluster(seed)), labels))
        res = rt_metrics.retrieval(x, labels, qe_k=0, workers=workers)
        assert res['dropped'] == 0 and len(res['rows']) == len(x), 'every validation image must be a query'
        mp.append(float(res['ap'].mean()))
        nd.append(float(rt_metrics.graded(x, dist, workers=workers).mean()))
    return ranges_doc(clu, mp, nd)


def validation_seed_ranges(workers=8):
    """Stage 0b: `measure_seed_ranges` on c*'s emotion head and the labels of Affection validation."""
    import ac_data
    return measure_seed_ranges(lambda seed: ac_data.embed('cstar', 'val', seed, head='emotion'),
                               lambda seed: ac_data.stored('cstar', 'val', seed, head='emotion'),
                               ac_data.labels('val'), ac_data.emotion_dist('val'), workers=workers)


def load_ranges(path):
    """Read a `ranges.json` and check it: the three entries, five finite per-seed values each, and r equal to their
    max minus min (so a hand-edited range is refused)."""
    doc = json.loads(Path(path).read_text())
    assert set(doc) == {'clu', 'map', 'ndcg'}, tuple(doc)
    for name, e in doc.items():
        assert set(e) == {'per_seed', 'r'} and len(e['per_seed']) == 5, name
        assert e['r'] == seed_range(e['per_seed']) and e['r'] > 0, (name, e['r'])
    return doc


def write_ranges(doc, outputs, copy_to=None):
    """Write <outputs>/stage0b/ranges.json (indent 1, no NaN). copy_to (the committed ac_paths.RANGES_FILE): the ranges
    are frozen once written, so a copy that exists with other content refuses the write; the same content is
    rewritten unchanged. Nothing is written when a check fails."""
    if copy_to is not None:
        ac_paths.check_record_write(copy_to)
    for name, e in doc.items():
        assert e['r'] == seed_range(e['per_seed']), name
    assert set(doc) == {'clu', 'map', 'ndcg'}
    text = json.dumps(doc, indent=1, allow_nan=False) + '\n'
    if copy_to is not None and Path(copy_to).exists():
        assert Path(copy_to).read_text() == text, f'{copy_to} holds other ranges: they are frozen once committed'
    out = Path(outputs) / 'stage0b'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'ranges.json').write_text(text)
    if copy_to is not None:
        Path(copy_to).write_text(text)


def r11_r_a():
    """r11's seed range per aspect (mAP, c* under R0 on ArtELingo validation): SC5 to SC7 and the ArtELingo picks."""
    doc = json.loads(ac_paths.R11_RANGES_FILE.read_text())
    return {a: float(doc['map'][a]['r']) for a in ac_paths.ASPECTS}


# ---- the SC table (spec §6) ----------------------------------------------------------------------------------------

class Sc(NamedTuple):
    """One success criterion. comparator: 'plain', 'teacher' (arm = teacher_<wording picked for `wording_task`>) or
    'crl' (arm = crl_<version picked for the dataset>). r: 'clu' (r_clu), 'map' (r_ret,mAP) or 'r_a' (r11's r for
    the aspect)."""
    id: str
    stage: int
    dataset: str
    aspect: str
    metric: str             # 'ami' | 'map'
    candidate: str
    comparator: str
    wording_task: object    # 'clustering' | 'retrieval' for a teacher comparator, else None
    bar: str                # a key of BARS
    r: str


def _sc():
    rows = [
        Sc('SC1', 2, 'affection', 'emotion', 'ami', 'cstar', 'plain', None, 'WIN', 'clu'),
        Sc('SC2', 2, 'affection', 'emotion', 'map', 'cstar', 'plain', None, 'WIN', 'map'),
        Sc('SC3', 2, 'affection', 'emotion', 'ami', 'cstar', 'teacher', 'clustering', 'LEVEL_OR_WIN', 'clu'),
        Sc('SC4', 2, 'affection', 'emotion', 'map', 'cstar', 'teacher', 'retrieval', 'LEVEL_OR_WIN', 'map'),
        Sc('SC5', 3, 'artelingo', 'emotion', 'map', 'cstar', 'crl', None, 'WIN', 'r_a'),
        Sc('SC6', 3, 'artelingo', 'style', 'map', 'cstar', 'crl', None, 'WIN', 'r_a'),
        Sc('SC7', 3, 'artelingo', 'genre', 'map', 'cstar', 'crl', None, 'WIN', 'r_a'),
        Sc('SC8', 3, 'affection', 'emotion', 'map', 'cstar', 'crl', None, 'WIN', 'map'),
    ]
    return {sc.id: sc for sc in rows}


SC = _sc()
STAGE2_IDS = ('SC1', 'SC2', 'SC3', 'SC4')
STAGE3_IDS = ('SC5', 'SC6', 'SC7', 'SC8')


def r_for(sc_id, ranges, r_a):
    """The seed range of an SC. ranges: the `stage0b/ranges.json` document (or any {'clu'|'map': {'r': f}});
    r_a: {aspect: r} (r11's `stage0a_ranges.json` mAP ranges; see `r11_r_a`). The range must be finite and positive."""
    sc = SC[sc_id]
    r = float(r_a[sc.aspect]) if sc.r == 'r_a' else float(ranges[sc.r]['r'])
    assert np.isfinite(r) and r > 0, f'{sc_id}: seed range {r} must be finite and positive'   # r11's outcome accepts 0
    return r


def comparator_arm(sc_id, picks):
    """The comparator's arm name given the stage 1 picks document: 'plain'; teacher_<wording picked for the SC's task>;
    crl_<version picked for the SC's dataset> (ArtELingo: crl_clip_painting or crl_qwen_painting)."""
    sc = SC[sc_id]
    if sc.comparator == 'plain':
        return 'plain'
    if sc.comparator == 'teacher':
        return f'teacher_{picks[sc.dataset]["teacher_wording"][sc.wording_task]}'
    return f'crl_{picks[sc.dataset]["crl_version"]}'


def readouts(sc_id, picks):
    """(candidate's, comparator's) picked read-out names; (None, None) for a clustering SC (no read-out there)."""
    sc = SC[sc_id]
    if sc.metric == 'ami':
        return None, None
    table = picks[sc.dataset]['readout']
    return table[sc.candidate], table[comparator_arm(sc_id, picks)]


# ---- bars, the verdict, the yes/no line ----------------------------------------------------------------------------

def sc_pass(sc_id, outcome_):
    """True when the SC's outcome meets its bar (WIN for a WIN bar; LEVEL or WIN for a LEVEL_OR_WIN bar)."""
    assert outcome_ in OUTCOMES, outcome_
    return outcome_ in BARS[SC[sc_id].bar]


def verdict(outcomes):
    """Stage 2: 'holds' when SC1 to SC4 pass; 'qualified' when SC1 and SC2 pass and exactly one of SC3 and SC4 does;
    'fails' otherwise. outcomes: {sc_id: outcome} for exactly SC1 to SC4."""
    assert set(outcomes) == set(STAGE2_IDS), f'need exactly {STAGE2_IDS}, got {sorted(outcomes)}'
    ok = {i: sc_pass(i, outcomes[i]) for i in STAGE2_IDS}
    if all(ok.values()):
        return 'holds'
    if ok['SC1'] and ok['SC2'] and ok['SC3'] != ok['SC4']:
        return 'qualified'
    return 'fails'


def repeats_line(sc3_outcome):
    """'yes' when the ArtELingo clustering WIN over the teacher repeats, i.e. SC3 is WIN, else 'no'. Decides nothing."""
    assert sc3_outcome in OUTCOMES, sc3_outcome
    return 'yes' if sc3_outcome == 'WIN' else 'no'


# ---- the retrieval comparison with picked read-outs (spec §3.4 "Retrieval", D10) ---------------------------------------

def arm_retrieval(get_val, get_eval, readout, labels, seeds=None, workers=8):
    """Per-query AP of one arm on the scored split under its picked read-out, as r11's stage 5: W is fitted on all
    validation images (`rt_readout.stage5_apply`), applied to the scored split, and the query expansion with the
    pick's k runs on that split's gallery (`rt_metrics.retrieval(qe_k=k)`; queries and gallery are the split's
    labelled images).
    get_val(seed), get_eval(seed): float32 unit rows of validation / the scored split (seed None for an unseeded arm);
    readout: a ReadOut or its name ('center+k10'); labels: int64 [n] of the scored split; seeds: None or (0, 1, 2, 3, 4).
    Returns {'ap': float64 [nq] or [5, nq], 'rows': int64 [nq], 'dropped': int, 'point': mean mAP (over seeds),
    'per_seed': [5 mAPs] or None}."""
    import ac_select
    ro = ac_select.readout_from_name(readout) if isinstance(readout, str) else readout
    assert seeds is None or tuple(seeds) == SEEDS, f'seeds must be None or {SEEDS}, got {seeds}'
    aps, rows, dropped = [], None, None
    for seed in ((None,) if seeds is None else SEEDS):
        x = rt_readout.stage5_apply(ro, get_val(seed), get_eval(seed))
        res = rt_metrics.retrieval(x, labels, qe_k=ro.k, workers=workers)
        assert res['ap'].dtype == np.float64 and res['rows'].dtype == np.int64
        if rows is None:
            rows, dropped = res['rows'], res['dropped']
        assert np.array_equal(rows, res['rows']) and dropped == res['dropped'], 'queries differ between seeds'
        aps.append(res['ap'])
    if seeds is None:
        return {'ap': aps[0], 'rows': rows, 'dropped': dropped, 'point': float(aps[0].mean()), 'per_seed': None}
    per_seed = [float(a.mean()) for a in aps]
    return {'ap': np.stack(aps), 'rows': rows, 'dropped': dropped, 'point': float(np.mean(per_seed)),
            'per_seed': per_seed}


def retrieval_sc(sc_id, cand, comp, picks, ranges, r_a, draws=10000):
    """The `sc` entry of a retrieval SC from two `arm_retrieval` results: r11's paired bootstrap (10,000 draws over
    queries and seeds), the outcome against the SC's r, the bar. ranges / r_a as `r_for`."""
    assert SC[sc_id].metric == 'map', f'{sc_id} is a clustering SC: use cluster_sc'
    assert np.array_equal(cand['rows'], comp['rows']), (sc_id, 'the two arms were scored on different queries')
    p = rt_stats.paired(cand['ap'], comp['ap'], draws)
    return sc_entry(sc_id, picks, cand['point'], comp['point'], p['d'], p['interval'], r_for(sc_id, ranges, r_a))


def cluster_sc(sc_id, res, picks, ranges, r_a=None):
    """The `sc` entry of a clustering SC from `ac_cluster.paired`'s result ({'A_cand', 'A_comp', 'd', 'interval'})."""
    assert SC[sc_id].metric == 'ami', f'{sc_id} is a retrieval SC: use retrieval_sc'
    return sc_entry(sc_id, picks, res['A_cand'], res['A_comp'], res['d'], res['interval'],
                    r_for(sc_id, ranges, r_a))


# ---- spec §3.5 0a(iii): the replay of r11's stage 5 ----------------------------------------------------------------

def replay_r11_stage5(arms=None, workers=8, draws=10000, data=None, picks=None, ranges=None):
    """r11's stage 5 recomputed through this module (`arm_retrieval`, rt_stats.paired and outcome), as `rt_run.stage5`:
    every arm's own head per aspect, its stage 4 read-out fitted on all ArtELingo validation paintings and applied to
    the held ones, mAP and SC10 to SC15 (r11's table) with 10,000 draws.
    data: the module providing labels / embed / ARMS / SEEDED / ASPECT_FREE (default rt_data); picks: {arm: read-out
    name} (default r11's stage4/picks.json); ranges: {aspect: r_a} (default r11's stage0a_ranges.json).
    Returns {'map': {arm: {aspect: {'mean', 'per_seed'}}}, 'sc': {id: {'comparisons': [...], 'pass': bool}}} in the
    form of r11's stage5/verdict.json (only the SCs whose arms were all scored)."""
    if data is None:
        import rt_data as data
    if picks is None:
        picks = json.loads((ac_paths.R11_OUT / 'stage4/picks.json').read_text())
    if ranges is None:
        ranges = r11_r_a()
    arms = tuple(data.ARMS if arms is None else arms)
    assert set(arms) <= set(picks), ('picks missing for', set(arms) - set(picks))
    labels = data.labels('held')
    cells = {}
    for arm in arms:
        for aspect in ac_paths.ASPECTS:
            head = None if arm in data.ASPECT_FREE else aspect
            cells[(arm, head, aspect)] = arm_retrieval(
                lambda s: data.embed(arm, 'val', head or 'emotion', s), lambda s: data.embed(arm, 'held', head or 'emotion', s),
                picks[arm], labels[aspect], seeds=SEEDS if arm in data.SEEDED else None, workers=workers)
    maps = {arm: {a: {'mean': cells[(arm, None if arm in data.ASPECT_FREE else a, a)]['point'],
                      'per_seed': cells[(arm, None if arm in data.ASPECT_FREE else a, a)]['per_seed']}
                  for a in ac_paths.ASPECTS} for arm in arms}
    sc_doc = {}
    for sid in rt_stats.TABLE_IDS[5]:
        sc = rt_stats.SC[sid]
        if not all(c.candidate[0] in arms and c.comparator[0] in arms for c in sc.comparisons):
            continue
        comps, outs = [], []
        for cmp in sc.comparisons:
            cand, comp = cells[(*cmp.candidate, sc.aspect)], cells[(*cmp.comparator, sc.aspect)]
            assert np.array_equal(cand['rows'], comp['rows']), (sid, 'query rows differ')
            p = rt_stats.paired(cand['ap'], comp['ap'], draws)
            o = outcome(p['interval'][0], ranges[sc.aspect])
            outs.append(o)
            comps.append({'candidate': list(cmp.candidate), 'comparator': list(cmp.comparator), 'aspect': sc.aspect,
                          'd': p['d'], 'interval': [float(v) for v in p['interval']], 'r': ranges[sc.aspect],
                          'outcome': o, 'bar': cmp.bar})
        sc_doc[sid] = {'comparisons': comps, 'pass': bool(rt_stats.sc_pass(sid, outs))}
    return {'map': maps, 'sc': sc_doc}


# ---- the verdict files (index: stage2/verdict.json, stage3/verdict.json) -------------------------------------------

def sc_entry(sc_id, picks, point_c, point_t, d, interval, r):
    """One `sc` entry of a verdict file. point_c, point_t: the two sides' scores (AMI or mAP, seeds averaged); d: the
    paired bootstrap's point difference; interval: its 95% interval; r: `r_for(sc_id, ...)`. The outcome is the
    interval's lower bound against 0 and -r; pass is the SC's bar on it. readout_c and readout_t are the picked
    read-out names (null for a clustering SC)."""
    sc = SC[sc_id]
    interval = [float(interval[0]), float(interval[1])]
    out = outcome(interval[0], r)
    readout_c, readout_t = readouts(sc_id, picks)
    return {'candidate': sc.candidate, 'comparator': comparator_arm(sc_id, picks), 'dataset': sc.dataset,
            'aspect': sc.aspect, 'metric': sc.metric, 'readout_c': readout_c, 'readout_t': readout_t,
            'point_c': float(point_c), 'point_t': float(point_t), 'd': float(d), 'interval': interval,
            'r': float(r), 'outcome': out, 'pass': bool(sc_pass(sc_id, out))}


def _check_entries(sc, ids):
    """The document's entries are exactly `ids`, in order, and each one's outcome and pass follow from its interval
    and r (a file edited by hand, or built from another rule, is refused)."""
    assert list(sc) == list(ids), f'need exactly {ids} in this order, got {list(sc)}'
    for sid, e in sc.items():
        assert e['outcome'] == outcome(e['interval'][0], e['r']), (sid, e['outcome'], e['interval'], e['r'])
        assert e['pass'] == sc_pass(sid, e['outcome']), (sid, e['pass'], e['outcome'])
        assert (e['dataset'], e['aspect'], e['metric']) == (SC[sid].dataset, SC[sid].aspect, SC[sid].metric), sid


def stage2_doc(sc, rows):
    """stage2/verdict.json: the SC1 to SC4 entries, the verdict, the yes/no line (SC3 is WIN) and the descriptive rows."""
    _check_entries(sc, STAGE2_IDS)
    outcomes = {sid: e['outcome'] for sid, e in sc.items()}
    return {'sc': sc, 'verdict': verdict(outcomes), 'repeats_artelingo_win': repeats_line(outcomes['SC3']),
            'rows': rows}


def claims(sc):
    """{sc_id: bool} for SC5 to SC8: each CRL bar is its own claim (True = passed)."""
    return {sid: bool(e['pass']) for sid, e in sc.items()}


def stage3_doc(sc, rows):
    """stage3/verdict.json: the SC5 to SC8 entries, each CRL bar's claim and the descriptive rows. A failed bar gives
    no claim (False) and keeps its entry and its row; there is no overall verdict."""
    _check_entries(sc, STAGE3_IDS)
    return {'sc': sc, 'claims': claims(sc), 'rows': rows}
