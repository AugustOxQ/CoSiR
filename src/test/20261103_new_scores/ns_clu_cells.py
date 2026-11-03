"""r17 ticket 07: clustering cells: SC1 to SC5 assembly (arms, partitions, r) and the stage 1 class-count ranges.

Spec docs/superpowers/specs/2026-11-03-new-scores-design.md §6 (SC1 to SC5), §3.2 arms, §3.3 data roles, §3.4 (r at the
class count, r at 20 for SC5), D7, D15, D16, D20.

  stage1_ranges(set_name, enc) -> {range key: {r, per_seed, k, n_val, n_labelled}}
      Candidate c* on the painting sets, the photo model on the photo sets; the validation split only (`RANGE_SPLIT`);
      k-means at the class count K (all rows, labels masked after); all four metrics; per seed the mean over the 3
      states (`ns_clu.ranges`). Range key `<cstar|photo>/<set>/<enc>/<aspect>/<metric>` (`ns_outcome.range_key`).
  score_cells(set_name, enc, scs, ranges, draws) -> [cell]
      One cell per claim cell of `ns_outcome.claim_cells()` with task 'clu', that set and encoder, and an SC in `scs`
      (so ids, SC applicability and k come from ticket 02). Cell: {id, sc, set, encoder, aspect, task, metric, k,
      candidate, comparator, cand, comp, d, lo, hi, r, r_source, n_draws, n_rows (+ n_labelled)}.

Which rows and partitions are paired (held or test split of the set, `ns_paths.HELD_SPLIT`):
  candidate  c* head (painting sets; `ns_data.cstar(..., unit=False)`, the stored float16 cast to float32, the form r10,
             r14 and r15 fed k-means) or the photo model (`ns_photo.photo`, same form), 5 seeds, k-means at K [5, 3, n].
  SC1        plain image (`plain`), unseeded [3, n].
  SC2        matched PercepT, only its k-means reading at K on L2-normalised probabilities (`ns_metrics.percept_partitions`,
             D16), one fit per seed 0 to 4, [5, 3, n]: `ns_clu.paired_cell` then draws its seeds apart from c*'s.
  SC3, SC4   the teacher (Qwen image instructed per aspect; photo sets the picked wording) and (SC4) the plain image.
  SC5        CRL on CLIP L/14, clustered on its 50-d vector (`ns_data.crl`): AMI at 20 clusters (k 20, id token `ami@20`)
             and the four metrics at K.
r of a cell: class-count cells, the candidate's stage 1 range for that set, encoder, aspect and metric (`ranges[range
key]['r']`; `ranges` None reads the committed `stage1_ranges.json`); SC5's AMI at 20, CLIP L/14 c*'s recorded AMI range
(r14 `artclu.ranges` on ArtELingo, r15 `clu_ranges.ranges` on MMArt), never a stage 1 value. A missing stage 1 key gives
r None with `r_source` 'missing:<key>' (ticket 02 reads the status for it).
Every k-means call goes through `ns_metrics.partitions(x, K, set_name)`; its threads come from
`ns_paths.KMEANS_THREADS` in code. Partitions are kept in memory per call (or in the caller's `cache`), never written
into an older round's partition folder.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ns_paths as P  # noqa: E402

P.add_rounds()
import ns_clu  # noqa: E402
import ns_data  # noqa: E402
import ns_metrics  # noqa: E402
import ns_outcome  # noqa: E402
import ns_photo  # noqa: E402

CLU_SCS = ('SC1', 'SC2', 'SC3', 'SC4', 'SC5')
RANGE_SPLIT = P.DEV_SPLIT            # the only split a range reads (D7)
RECORDED_AMI20 = {                   # CLIP L/14 c*'s recorded AMI-at-20 ranges: (record, path inside it)
    'artelingo': (P.ROUNDS['r14'] / 'records/stage1_clip_l14.json', ('artclu', 'ranges'), 'artelingo_clustering_validation_ranges'),
    'mmart': (P.ROUNDS['r15'] / 'records/stage1_clip_l14.json', ('clu_ranges', 'ranges'), 'mmart_clustering_validation_ranges'),
}


# ------------------------------------------------------------------------------------------------ loaders
def _check_set_enc(set_name, enc):
    assert set_name in P.HELD_SPLIT, f'unknown set {set_name!r}'
    assert enc in P.ENCODERS, f'unknown encoder {enc!r}'


def _labels(set_name, split, aspect):
    if set_name in P.PHOTO_SETS:
        assert aspect == 'emotion', aspect
        return ns_photo.labels(set_name, split)
    return ns_data.labels(set_name, split, aspect)


def _cand_rows(set_name, split, enc, aspect, seed):
    """The candidate's rows in the clustering form (float16 stored values cast to float32, not renormalised)."""
    if set_name in P.PHOTO_SETS:
        return ns_photo.photo(set_name, split, enc, seed)
    return ns_data.cstar(set_name, split, enc, aspect, seed, unit=False)


def _cand_parts(set_name, split, enc, aspect, k):
    """int32 [5, 3, n]: the candidate's seeds, k-means at k (threads per set inside `ns_metrics.partitions`)."""
    return np.stack([ns_metrics.partitions(_cand_rows(set_name, split, enc, aspect, s), k, set_name) for s in P.SEEDS])


def _plain_rows(set_name, split, enc):
    return ns_photo.plain(set_name, split, enc) if set_name in P.PHOTO_SETS else ns_data.plain(set_name, split, enc)


def _teacher_rows(set_name, split, enc, aspect):
    return ns_photo.teacher(set_name, split, enc) if set_name in P.PHOTO_SETS else ns_data.teacher(set_name, split, aspect)


def _comp_parts(set_name, split, enc, aspect, comparator, k):
    """The comparator's partitions: int32 [3, n] (unseeded) or [5, 3, n] (matched PercepT, one fit per seed)."""
    if comparator == 'plain':
        return ns_metrics.partitions(_plain_rows(set_name, split, enc), k, set_name)
    if comparator == 'teacher':
        return ns_metrics.partitions(_teacher_rows(set_name, split, enc, aspect), k, set_name)
    if comparator == 'crl':
        return ns_metrics.partitions(ns_data.crl(set_name, split, aspect), k, set_name)
    assert comparator == 'matched', comparator
    assert set_name in P.PAINTING_SETS, 'matched PercepT is scored on the painting sets'
    return np.stack([ns_metrics.percept_partitions(ns_data.percept_probs(set_name, split, enc, s), k, set_name)
                     for s in P.SEEDS])


# ------------------------------------------------------------------------------------------------ r
def recorded_ami20_range(set_name, aspect):
    """CLIP L/14 c*'s recorded AMI range at 20 clusters on the set's validation split (SC5's r at 20): r14's
    `stage1_clip_l14.json` `artclu.ranges` (ArtELingo), r15's `clu_ranges.ranges` (MMArt)."""
    path, inner, kind = RECORDED_AMI20[set_name]
    rec = json.loads(path.read_text())
    node = rec[inner[0]]
    assert node['encoder'] == P.CRL_ENCODER, path
    assert node['kind'] == kind, (path, node['kind'])
    r = float(node['ranges'][aspect])
    assert np.isfinite(r) and r >= 0, (path, aspect, r)
    return r, f'recorded:{path.parent.parent.name}/records/{path.name}:{inner[0]}.ranges.{aspect}'


def _r_of(cell, ranges):
    """(r, r_source) of one claim cell."""
    if cell['kind'] == ns_outcome.FIXED20:
        return recorded_ami20_range(cell['set'], cell['aspect'])
    key = ns_outcome.range_key(cell)
    if key not in ranges:
        return None, f'missing:{key}'
    r = float(ranges[key]['r'])
    assert np.isfinite(r) and r >= 0, (key, r)
    return r, f'stage1:{key}'


# ------------------------------------------------------------------------------------------------ stage 1
def stage1_ranges(set_name, enc):
    """Class-count seed ranges of the candidate on the validation split (module docstring)."""
    _check_set_enc(set_name, enc)
    if set_name in P.PHOTO_SETS:
        assert enc in P.PHOTO_ENCODERS[set_name], f'{set_name} is scored for {P.PHOTO_ENCODERS[set_name]}, not {enc}'
    cand = 'photo' if set_name in P.PHOTO_SETS else 'cstar'
    out = {}
    for aspect in P.SET_ASPECTS[set_name]:
        k = P.CLASS_COUNT[(set_name, aspect)]
        y = _labels(set_name, RANGE_SPLIT, aspect)
        assert ns_metrics.class_count(y) == k, f'{set_name}/{RANGE_SPLIT}/{aspect}: {ns_metrics.class_count(y)} classes, not {k}'
        rng = ns_clu.ranges(_cand_parts(set_name, RANGE_SPLIT, enc, aspect, k), y, k)
        for m in P.CLASS_METRICS:
            out[f'{cand}/{set_name}/{enc}/{aspect}/{m}'] = dict(
                r=rng[m]['r'], per_seed=rng[m]['per_seed'], k=int(k), n_val=int(len(y)), n_labelled=int((y >= 0).sum()))
    return out


# ------------------------------------------------------------------------------------------------ stage 2
def score_cells(set_name, enc, scs, ranges=None, draws=P.N_DRAWS['clu'], cache=None, rng_seed=P.BOOT_SEED):
    """Clustering cells of (set, encoder) for the SCs in `scs` (module docstring). `ranges`: the stage 1 record
    ({range key: {r, ...}}) or None for the committed `stage1_ranges.json`. `cache`: an optional dict the caller keeps to
    share partitions between calls (in memory). SCs that do not apply to (set, encoder) give no cell."""
    _check_set_enc(set_name, enc)
    scs = tuple(scs)
    assert scs and set(scs) <= set(CLU_SCS), f'clustering SCs are {CLU_SCS}, got {scs}'
    cells = [c for c in ns_outcome.claim_cells()
             if c['task'] == 'clu' and c['set'] == set_name and c['encoder'] == enc and c['sc'] in scs]
    if not cells:
        return []
    if ranges is None:
        ranges = P.read_committed_record('stage1_ranges.json')
    cache = {} if cache is None else cache
    split = P.HELD_SPLIT[set_name]
    groups = {}
    for c in cells:
        groups.setdefault((c['sc'], c['aspect'], c['comparator'], c['kind']), []).append(c)
    out = []
    for (sc, aspect, comparator, kind), group in groups.items():
        k = group[0]['k']
        assert all(c['k'] == k for c in group)
        y = _labels(set_name, split, aspect)
        if kind == ns_outcome.CLASS_COUNT:
            assert ns_metrics.class_count(y) == k, f'{set_name}/{split}/{aspect}: {ns_metrics.class_count(y)} classes, not {k}'
        ck = ('cand', set_name, enc, aspect, k)
        if ck not in cache:
            cache[ck] = _cand_parts(set_name, split, enc, aspect, k)
        mk = ('comp', set_name, enc, None if comparator == 'plain' else aspect, comparator, k)
        if mk not in cache:
            cache[mk] = _comp_parts(set_name, split, enc, aspect, comparator, k)
        metrics = tuple(c['metric'] for c in group)
        res = ns_clu.paired_cell(cache[ck], cache[mk], y, k, r=None, metrics=metrics, draws=draws, rng_seed=rng_seed)
        for c in group:
            m = res[c['metric']]
            r, source = _r_of(c, ranges)
            out.append(dict({key: c[key] for key in ns_outcome.IDENTITY_KEYS}, id=c['id'],
                            cand=m['cand'], comp=m['comp'], d=m['d'], lo=m['lo'], hi=m['hi'], r=r, r_source=source,
                            n_draws=m['n_draws'], n_rows=int(len(y)), n_labelled=int((y >= 0).sum())))
    assert [c['id'] for c in out] == [c['id'] for g in groups.values() for c in g]
    return out
