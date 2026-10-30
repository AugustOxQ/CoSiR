"""Stage 1 selection for the r12 round (spec 2026-10-30-affection-confirmation-design, section 6 "Selection rule
(stage 1)", 3.3, 3.4 "Seed ranges"): every pick, from validation inputs only.

Pure functions on cross-fit rows and plain numbers; no file reading except the replay of r11's stage 4 and the
writer of the picks files. The menu, the cross-fit halves, the tie order and the tie tolerance are r11's
(rt_readout), imported and not copied.
"""
import ac_paths  # noqa: F401  (puts r11's round on sys.path)
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
