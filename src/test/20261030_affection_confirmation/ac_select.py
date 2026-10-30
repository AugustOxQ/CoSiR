"""Stage 1 selection for the r12 round (spec 2026-10-30-affection-confirmation-design, section 6 "Selection rule
(stage 1)", 3.3, 3.4 "Seed ranges"): every pick, from validation inputs only.

Pure functions on cross-fit rows and plain numbers; no file reading except the replay of r11's stage 4 and the
writer of the picks files. The menu, the cross-fit halves, the tie order and the tie tolerance are r11's
(rt_readout), imported and not copied. The scoring, the guard and the pick (readout_scores, readout_guard,
pick_readout) are the exception: they are re-implemented for one aspect (Affection has only emotion), because r11's
score, guard and select assert three aspects. The tests check them against r11's own functions on random
three-aspect tables.
"""
import json
from pathlib import Path

import ac_paths  # puts r11's round on sys.path
import rt_readout
from rt_readout import R0, SLACK, TIE_TOL


def readout_scores(rows, ranges):
    """r10's gain over R0 in units of the seed range, averaged over the aspects present:
    mean_a (mAP_a - mAP_a(R0)) / r_a. One aspect (Affection, emotion): (mAP - mAP(R0)) / r.
    rows: {ReadOut: {aspect: mAP}} (cross-fit, seeds averaged), R0 among them; ranges: {aspect: r_a}."""
    assert R0 in rows, 'the R0 row is needed'
    aspects = tuple(rows[R0])
    assert aspects and set(ranges) >= set(aspects) and all(ranges[a] > 0 for a in aspects), (aspects, ranges)
    assert all(set(m) == set(aspects) for m in rows.values())
    base = rows[R0]
    return {x: sum((m[a] - base[a]) / ranges[a] for a in aspects) / len(aspects) for x, m in rows.items()}


def readout_guard(rows, ranges):
    """{ReadOut: bool}, True = kept: no aspect more than r_a below its R0 value (r11's guard). With one aspect a
    dropped read-out has a score below -1 and can never be within 0.001 of the best (R0 scores 0), so the pick does
    not depend on it there."""
    aspects = tuple(rows[R0])
    base = rows[R0]
    return {x: all(m[a] >= base[a] - ranges[a] - SLACK for a in aspects) for x, m in rows.items()}


def pick_readout(rows, ranges):
    """The pick: among the read-outs passing the guard, those with a score within 0.001 of the best; the first of
    them in r11's tie order (R0, no expansion before expansion, none < center < whiten, smaller k, larger size)."""
    sc, keep = readout_scores(rows, ranges), readout_guard(rows, ranges)
    assert keep[R0]
    cands = [x for x in rows if keep[x]]
    best = max(sc[x] for x in cands)
    tied = [x for x in cands if sc[x] >= best - TIE_TOL - SLACK]
    return min(tied, key=rt_readout.tie_key)


# ---- teacher wording and CRL version -------------------------------------------------------------------------------

WORDING_MARGIN = TIE_TOL   # "more than 0.001 r": the margin is in units of the seed range


def pick_wording(artwork, photo, r):
    """Teacher wording for one task. artwork, photo: validation AMI (clustering) or cross-fit mAP at the wording's
    own read-out pick (retrieval); r: r_clu or r_ret,mAP. The photo wording only when it beats the artwork wording
    by more than 0.001 r; exactly 0.001 r is not more (SLACK absorbs the float rounding of that boundary)."""
    assert r > 0
    return 'photo' if (photo - artwork) / r > WORDING_MARGIN + SLACK else 'artwork'


def version_score(maps, ranges):
    """A CRL version's score on a dataset: the mean over the dataset's aspects of cross-fit validation mAP / r_a at
    the version's own read-out pick. It is in units of r, so spec 6's "within 0.001 r of the best" is TIE_TOL on it.
    maps: {aspect: mAP at the pick}; ranges: {aspect: r_a} (Affection: emotion, r_ret,mAP; ArtELingo: r11's r_a)."""
    assert maps and set(ranges) >= set(maps) and all(ranges[a] > 0 for a in maps)
    return sum(maps[a] / ranges[a] for a in maps) / len(maps)


def pick_crl_version(scores):
    """The version with the highest score; within 0.001 (score units, i.e. 0.001 r) of the best, the first of
    ac_paths.CRL_VERSIONS (clip_painting, clip_photo, qwen_painting, qwen_photo). scores: {version: float}, a subset
    of the four (ArtELingo: the two painting versions)."""
    assert scores and set(scores) <= set(ac_paths.CRL_VERSIONS), tuple(scores)
    best = max(scores.values())
    return next(v for v in ac_paths.CRL_VERSIONS if v in scores and scores[v] >= best - TIE_TOL - SLACK)


# ---- an arm's table: cross-fit, scores, pick -----------------------------------------------------------------------

def readout_by_name(name, width):
    """The read-out of r11's menu at `width` with this name (rt_readout.name's form, e.g. 'center+k10')."""
    by_name = {rt_readout.name(x): x for x in rt_readout.menu(width)}
    return by_name[name]


def arm_table(xs, labels, ranges, ros=None, fixed=None, workers=8, keys=None):
    """One arm's validation table and pick, through r11's cross-fit (rt_readout.crossfit_all: parity halves of the
    position in the sorted validation list, fit on one half and scored on the other, mean over the two halves, then
    over the seeds of a seeded arm).
    xs: {(aspect, seed): float32 [n, width] unit rows in sorted-key order}, seed None for an unseeded arm, 0..4 for a
    seeded one; labels: {aspect: int64 [n]}; ranges: {aspect: r_a}. ros: the read-outs to score (default the menu at
    the arm's width). fixed: a read-out given in advance (ArtELingo's c* and CLIP CRL): only R0 and it are scored
    and it is the pick. keys: optional, the validation keys in row order; the parity halves are right only in
    sorted-key order, so it must be sorted and have one key per row.
    Returns (table, pick): table = {name: {'map': {aspect: mAP}, 'score': gain, 'kept': guard}} in the order scored,
    pick = a ReadOut."""
    widths = {x.shape[1] for x in xs.values()}
    assert len(widths) == 1, ('one width per arm', widths)
    if keys is not None:
        keys = list(keys)
        assert keys == sorted(keys), 'the rows must be in sorted-key order'
        assert all(len(x) == len(keys) for x in xs.values()), ('one key per row', len(keys))
    width = widths.pop()
    assert set(labels) == {a for (a, _) in xs} and set(ranges) >= set(labels), (tuple(labels), tuple(ranges))
    if fixed is not None:
        assert ros is None
        ros = [R0] if fixed == R0 else [R0, fixed]
    ros = rt_readout.menu(width) if ros is None else list(ros)
    assert R0 in ros
    rows = rt_readout.crossfit_all(xs, labels, ros, workers=workers)
    sc, keep = readout_scores(rows, ranges), readout_guard(rows, ranges)
    table = {rt_readout.name(x): {'map': {a: float(m[a]) for a in m}, 'score': float(sc[x]), 'kept': bool(keep[x])}
             for x, m in rows.items()}
    return table, (fixed if fixed is not None else pick_readout(rows, ranges))


# ---- the stage 1 documents -----------------------------------------------------------------------------------------

AFFECTION_ARMS = ('cstar', 'plain', 'teacher_artwork', 'teacher_photo', 'clip', 'percept',
                  'crl_clip_painting', 'crl_clip_photo', 'crl_qwen_painting', 'crl_qwen_photo')
ARTELINGO_FIXED = {'cstar': 'center+k10', 'crl_clip_painting': 'none+k10'}   # r11's stage 4 picks (spec 6)
ARTELINGO_VERSIONS = ('clip_painting', 'qwen_painting')


def readout_from_name(name):
    """The read-out with this name (rt_readout.name's form: 'none+k0', 'center+k10', 'whiten256+k3')."""
    head, k = name.split('+k')
    w = head.rstrip('0123456789')
    return rt_readout.ReadOut(w, int(head[len(w):]) if w == 'whiten' else None, int(k))


TOL = 1e-12


def _check_table(table, ranges, aspects, pick=None):
    """Re-derive what a table says: its rows cover exactly `aspects`; each score is the mean over the aspects of
    (mAP - mAP(R0)) / r_a within 1e-12; if `pick` is given, it is pick_readout on the table's mAPs."""
    assert 'none+k0' in table, 'the R0 row is needed'
    base = table['none+k0']['map']
    for n, row in table.items():
        assert set(row['map']) == set(aspects), (n, tuple(row['map']))
        want = sum((row['map'][a] - base[a]) / ranges[a] for a in aspects) / len(aspects)
        assert abs(row['score'] - want) <= TOL, (n, row['score'], want)
    if pick is not None:
        rows = {readout_from_name(n): row['map'] for n, row in table.items()}
        assert pick_readout(rows, ranges) == pick, ('the pick is not the table\'s', rt_readout.name(pick))


def _at_pick(table, pick):
    return table[rt_readout.name(pick)]['map']


def build_stage1(aff_tables, aff_picks, ami, r_clu, r_ret, art_tables, art_picks, art_ranges):
    """Every pick of spec 6's stage 1 selection rule, and the scores behind them, in the index's formats.
    aff_tables: {arm: table} and aff_picks: {arm: ReadOut} for the ten AFFECTION_ARMS (arm_table's output on the
    validation images, emotion only); ami: {'artwork': float, 'photo': float}, the teacher wordings' validation
    emotion AMI (ticket 06); r_clu, r_ret: the seed ranges r_clu and r_ret,mAP (ticket 08).
    art_tables / art_picks: for 'crl_clip_painting' (scored at its fixed read-out) and 'crl_qwen_painting' (picked
    by r11's rule over the three aspects); art_ranges: r11's r_a. Returns (picks, scores)."""
    assert set(aff_tables) == set(aff_picks) == set(AFFECTION_ARMS), (tuple(aff_tables), tuple(aff_picks))
    assert set(ami) == {'artwork', 'photo'}
    assert set(art_tables) == set(art_picks) == {'crl_clip_painting', 'crl_qwen_painting'}
    assert art_picks['crl_clip_painting'] == readout_by_name(ARTELINGO_FIXED['crl_clip_painting'], 50), \
        'the CLIP CRL on ArtELingo keeps r11\'s pick'
    for arm in AFFECTION_ARMS:       # the tables and picks are given: re-derive them before they decide anything
        _check_table(aff_tables[arm], {'emotion': r_ret}, ('emotion',), aff_picks[arm])
    _check_table(art_tables['crl_clip_painting'], art_ranges, ac_paths.ASPECTS)            # its pick is fixed
    _check_table(art_tables['crl_qwen_painting'], art_ranges, ac_paths.ASPECTS, art_picks['crl_qwen_painting'])
    aff_maps = {arm: _at_pick(aff_tables[arm], aff_picks[arm]) for arm in AFFECTION_ARMS}
    retrieval = {w: aff_maps[f'teacher_{w}']['emotion'] for w in ('artwork', 'photo')}
    wording = {'clustering': pick_wording(ami['artwork'], ami['photo'], r_clu),
               'retrieval': pick_wording(retrieval['artwork'], retrieval['photo'], r_ret)}
    aff_versions = {v: version_score(aff_maps[f'crl_{v}'], {'emotion': r_ret}) for v in ac_paths.CRL_VERSIONS}
    art_versions = {v: version_score(_at_pick(art_tables[f'crl_{v}'], art_picks[f'crl_{v}']), art_ranges)
                    for v in ARTELINGO_VERSIONS}
    picks = {
        'affection': {'readout': {arm: rt_readout.name(aff_picks[arm]) for arm in AFFECTION_ARMS},
                      'teacher_wording': wording, 'crl_version': pick_crl_version(aff_versions)},
        'artelingo': {'readout': {**ARTELINGO_FIXED, 'crl_qwen_painting': rt_readout.name(art_picks['crl_qwen_painting'])},
                      'crl_version': pick_crl_version(art_versions)},
    }
    scores = {
        'readout': {'affection': aff_tables, 'artelingo': art_tables},
        'teacher_wording': {'clustering': dict(ami), 'retrieval': retrieval},
        'crl_version': {'affection': aff_versions, 'artelingo': art_versions},
    }
    validate_picks(picks)
    return picks, scores


def replay_r11_stage4(arms=None, workers=8, data=None, ranges=None):
    """Spec 3.5 stage 0a(iii), r11 stage 4 part: r11's stage 4 scores and picks recomputed through this module
    from r11's inputs (rt_data.embed on the ArtELingo validation paintings), as rt_run.stage4 builds them.
    arms: default every r11 arm; data: the module providing labels / embed / ARMS / SEEDED / ASPECT_FREE / WIDTHS
    (default rt_data); ranges: {aspect: r_a} (default r11's stage0a_ranges.json, mAP).
    Returns (scores, picks) in r11's stage4/scores.json form ({arm: {name: {'map', 'score', 'kept'}}}) and
    stage4/picks.json form ({arm: name})."""
    if data is None:
        import rt_data as data
    if ranges is None:
        doc = json.loads(ac_paths.R11_RANGES_FILE.read_text())
        ranges = {a: float(doc['map'][a]['r']) for a in ac_paths.ASPECTS}
    arms = tuple(data.ARMS if arms is None else arms)
    labels = data.labels('val')
    scores, picks = {}, {}
    for arm in arms:
        xs = {}
        for s in (ac_paths.SEEDS if arm in data.SEEDED else (None,)):
            if arm in data.ASPECT_FREE:
                x = data.embed(arm, 'val', 'emotion', s)
                xs.update({(a, s): x for a in ac_paths.ASPECTS})      # one matrix: its eigendecomposition is shared
            else:
                xs.update({(a, s): data.embed(arm, 'val', a, s) for a in ac_paths.ASPECTS})
        if arm != 'crl':
            assert {x.shape[1] for x in xs.values()} == {data.WIDTHS[arm]}, arm
        scores[arm], picked = arm_table(xs, labels, ranges, workers=workers)
        picks[arm] = rt_readout.name(picked)
    return scores, picks


def validate_picks(picks):
    """The picks document's shape (index, 'stage1/picks.json'); asserts, returns nothing."""
    assert set(picks) == {'affection', 'artelingo'}
    aff, art = picks['affection'], picks['artelingo']
    assert set(aff) == {'readout', 'teacher_wording', 'crl_version'} and set(art) == {'readout', 'crl_version'}
    assert set(aff['readout']) == set(AFFECTION_ARMS) and all(isinstance(v, str) for v in aff['readout'].values())
    assert set(aff['teacher_wording']) == {'clustering', 'retrieval'}
    assert all(v in ('artwork', 'photo') for v in aff['teacher_wording'].values())
    assert aff['crl_version'] in ac_paths.CRL_VERSIONS
    assert set(art['readout']) == {'cstar', 'crl_clip_painting', 'crl_qwen_painting'}
    assert all(art['readout'][a] == v for a, v in ARTELINGO_FIXED.items())
    assert art['crl_version'] in ARTELINGO_VERSIONS


def write_picks(picks, scores, outputs, copy_to=None):
    """Write <outputs>/stage1/picks.json and scores.json (r11's JSON style: indent 1, no NaN). copy_to (the
    committed ac_paths.PICKS_FILE): the picks are frozen once written, so a copy that exists with other content
    refuses the write; the same content is rewritten unchanged. Nothing is written when a check fails."""
    if copy_to is not None:
        ac_paths.check_record_write(copy_to)
    validate_picks(picks)
    text = json.dumps(picks, indent=1, allow_nan=False) + '\n'
    score_text = json.dumps(scores, indent=1, allow_nan=False) + '\n'
    if copy_to is not None and Path(copy_to).exists():
        assert Path(copy_to).read_text() == text, f'{copy_to} holds other picks: they are frozen once committed'
    out = Path(outputs) / 'stage1'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'picks.json').write_text(text)
    (out / 'scores.json').write_text(score_text)
    if copy_to is not None:
        Path(copy_to).write_text(text)
