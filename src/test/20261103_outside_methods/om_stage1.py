"""r18 stage 1 (ticket 08; spec §3.4 source table, "Retrieval", "CLAY", D7, D8, D14): every range and read-out pick, on the
VALIDATION split only (§3.3). Nothing here names a held split: `ValData` fixes `val` in every call and `audit_validation_only`
asserts that the validation keys share nothing with the held ones.

Outputs (OUT = `om_paths.OUT`):
    OUT/stage1/om_ranges.json   `<set>/<enc>/<aspect>/<metric>/<k>` (clustering) and `<set>/<enc>/<aspect>/map` (retrieval)
                                -> {r, source (+ per_seed, n_val, n_labelled when this stage computed it)}; r null when its
                                source is missing (then `source` starts with `missing:`).
    OUT/stage1/om_picks.json    `tpips/<set>`, `clay/<set>` -> r11's pick form (`ns_picks`) plus `text_set`, `unit_score`;
                                `clay_textset/clu/<set>`, `clay_textset/ret/<set>` -> {set 'A'|'B', scores, rule, reason};
                                `clay_text_sha256` -> {'A': {aspect: sha}, 'B': {aspect: sha or null}} (the pin: see PIN_KEY).
    OUT/stage1/tables/          the full read-out tables behind each pick.

Sources of r (the clustering key's k is the class count, or 20):
    class count, four metrics        r17's committed `records/stage1_ranges.json` (key `cstar/<set>/<enc>/<aspect>/<metric>`);
                                     absent file or key -> r null
    AMI at 20, ArtELingo Qwen        computed here: k-means as r10 (`ns_metrics.partitions`) on Qwen c*'s validation exports,
                                     seeds 0 to 4, mean over the 3 states per seed, max minus min, unrounded
    AMI at 20, ArtELingo others      r14's `stage1_<enc>.json` `artclu`
    AMI at 20 (and MMArt style's AMI at its class count 20), MMArt, all encoders     r15's `stage1_<enc>.json` `clu_ranges`
    mAP                              `ns_ret.recorded_range` (r11 for Qwen on ArtELingo, r14, r15)

CLI: python om_stage1.py ranges | tpips --set S | clay --set S | pins | check-recorded --enc E   (OM_OUTPUTS sets OUT).
`ranges` and `clay` need no TPIPS store; `tpips` crashes if the store is missing. Every step reruns safely.
"""
import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import om_paths as op

import ac_select  # noqa: E402  (om_paths put r17's folder and the older rounds on sys.path)
import ns_clu  # noqa: E402
import ns_metrics  # noqa: E402
import ns_paths  # noqa: E402
import ns_picks  # noqa: E402
import ns_ret  # noqa: E402
import rt_readout  # noqa: E402

SPLIT = op.DEV_SPLIT                          # 'val': the only split this module reads
ASPECTS, ENCODERS, SETS = op.ASPECTS, op.ENCODERS, op.SETS
SLACK = rt_readout.SLACK                      # absorbs float rounding at a boundary, as r11's rule
CLU_MARGIN = 0.001                            # D8: B needs a mean AMI at 20 more than this above A (AMI units)
RET_MARGIN = 0.1                              # D7: B needs a mean mAP / range more than this above A
PIN_KEY = 'clay_text_sha256'                  # in om_picks.json: {'A': {aspect: sha}, 'B': {aspect: sha or null}}
TEXT_SETS = ('A', 'B')
RECORD_DIRS = {'r14': ns_paths.ROUNDS['r14'] / 'records', 'r15': ns_paths.ROUNDS['r15'] / 'records'}
R17_FILE = 'stage1_ranges.json'


def out_dir(out=None):
    return Path(op.OUT if out is None else out)


# ================================================================================================ validation data
class ValData:
    """Validation readers over `om_data`: no method takes a split, so no range or pick can read a held row. `held_keys`
    gives the held split's KEYS only, for the disjointness audit."""

    def __init__(self, od=None, tpips_roots=None):
        if od is None:
            import om_data as od
        self.od, self.tpips_roots = od, tpips_roots

    def keys(self, set_):
        return self.od.keys(set_, SPLIT)

    def held_keys(self, set_):
        return self.od.keys(set_, op.HELD_SPLIT[set_])

    def labels(self, set_, aspect):
        return self.od.labels(set_, SPLIT, aspect)

    def cstar(self, set_, enc, aspect, seed):
        """c*'s head in the clustering form (stored float16 cast to float32, as r10, r14, r15 fed k-means)."""
        return self.od.cstar(set_, SPLIT, enc, aspect, seed, unit=False)

    def tpips(self, set_, aspect):
        return self.od.tpips(set_, SPLIT, aspect, roots=self.tpips_roots)

    def clay(self, set_, aspect, text_set):
        return self.od.clay(set_, SPLIT, aspect, text_set)

    def _b_entry(self, aspect):
        doc = json.loads((op.CLAY_B_DIR / 'lists.json').read_text())
        return doc['aspects'][aspect]

    def text_count(self, text_set, aspect):
        """Number of text rows of the set for the aspect; None for set B on an aspect that fell back (no file)."""
        import om_clay
        if text_set == 'B' and self._b_entry(aspect)['fallback']:
            return None
        return int(om_clay.text_set(text_set, aspect).shape[0])

    def text_sha(self, text_set, aspect):
        """The pin: `om_data.array_sha256` of the text rows, the `text_sha256` of the CLAY cache sidecar."""
        import om_clay
        if self.text_count(text_set, aspect) is None:
            return None
        return self.od.array_sha256(om_clay.text_set(text_set, aspect))


def audit_validation_only(data, set_):
    """The validation keys of `set_` are the expected count and share no key with the held split (spec §3.3)."""
    val, held = list(data.keys(set_)), set(data.held_keys(set_))
    assert len(val) == len(set(val)) == ns_paths.N_ROWS[(set_, SPLIT)], (set_, len(val))
    shared = held & set(val)
    assert not shared, f'{set_}: {len(shared)} validation keys are also in the held split, e.g. {sorted(shared)[:3]}'
    return val


# ================================================================================================ ranges
def class_ks(set_, aspect):
    """[(k, metrics)] of the labelled clustering cells: all four metrics at the class count, AMI alone at 20."""
    cc = op.CLASS_COUNT[(set_, aspect)]
    out = [(cc, tuple(op.CLASS_METRICS))]
    if op.K20 != cc:
        out.append((op.K20, ('ami',)))
    return out


def _source_family(set_, enc, metric, k, aspect):
    cc = op.CLASS_COUNT[(set_, aspect)]
    if metric == 'map':
        return 'map'
    if k == op.K20 and metric == 'ami':
        if set_ == 'mmart':
            return 'r15'                       # includes MMArt style's AMI at its class count 20
        return 'ours' if enc == 'qwen' else 'r14'
    assert k == cc, (set_, aspect, k)
    return 'r17'


def range_keys():
    """Every range key, as dicts {key, set, enc, aspect, metric, k (None for map), source}: the order is set, encoder,
    aspect, then clustering before map."""
    out = []
    for set_ in SETS:
        for enc in ENCODERS:
            for aspect in ASPECTS:
                for k, metrics in class_ks(set_, aspect):
                    for m in metrics:
                        out.append(dict(key=f'{set_}/{enc}/{aspect}/{m}/{k}', set=set_, enc=enc, aspect=aspect, metric=m, k=k,
                                        source=_source_family(set_, enc, m, k, aspect)))
                out.append(dict(key=f'{set_}/{enc}/{aspect}/map', set=set_, enc=enc, aspect=aspect, metric='map', k=None,
                                source='map'))
    return out


def cstar_range(data, set_, enc, aspect, k, metric, partitions=ns_metrics.partitions):
    """Seed range of c*'s k-means on the validation split: per seed 0 to 4 the mean over states 0 to 2 of `metric` (labels
    masked after clustering, all rows clustered), maximum minus minimum, unrounded. `ns_clu.ranges` does the arithmetic."""
    y = np.asarray(data.labels(set_, aspect), dtype=np.int64)
    parts = np.stack([np.asarray(partitions(data.cstar(set_, enc, aspect, s), k, set_)) for s in op.SEEDS])
    rng = ns_clu.ranges(parts, y, k, metrics=(metric,))[metric]
    return dict(r=float(rng['r']), per_seed=rng['per_seed'], k=int(k), n_val=int(len(y)), n_labelled=int((y >= 0).sum()))


def qwen_ami20(data, partitions=ns_metrics.partitions):
    """{aspect: cstar_range} for Qwen on ArtELingo, AMI at 20 (no recorded range exists)."""
    return {a: cstar_range(data, 'artelingo', 'qwen', a, op.K20, 'ami', partitions) for a in ASPECTS}


def _recorded(round_, enc, inner, kind, aspect):
    path = Path(RECORD_DIRS[round_]) / f'stage1_{enc}.json'
    node = json.loads(path.read_text())[inner]
    assert node['encoder'] == enc and node['kind'] == kind, (path, node['encoder'], node['kind'])
    r = float(node['ranges'][aspect])
    per = [float(node['per_seed'][aspect][str(s)]) for s in op.SEEDS]
    assert np.isfinite(r) and r >= 0, (path, aspect, r)
    assert abs(max(per) - min(per) - r) <= 1e-12, f'{path} {aspect}: per-seed values do not give the range'
    return r, f'{round_} records/stage1_{enc}.json:{inner}.ranges.{aspect}'


def recorded_ami20(set_, enc, aspect):
    """(r, source): c*'s recorded AMI range at 20 clusters on the validation split: r14 `artclu` (ArtELingo, not Qwen),
    r15 `clu_ranges` (MMArt, four encoders)."""
    if set_ == 'mmart':
        return _recorded('r15', enc, 'clu_ranges', 'mmart_clustering_validation_ranges', aspect)
    assert set_ == 'artelingo' and enc != 'qwen', 'no recorded AMI-at-20 range exists for Qwen on ArtELingo (computed here)'
    return _recorded('r14', enc, 'artclu', 'artelingo_clustering_validation_ranges', aspect)


def map_range(set_, enc, aspect):
    """(r, source): c*'s recorded validation mAP range (`ns_ret.recorded_range`)."""
    return ns_ret.recorded_range(set_, enc, aspect)


def r17_record(path=None):
    """r17's committed class-count ranges {key: {r, ...}}, or None when its file is absent (r17's run has not happened)."""
    path = Path(ns_paths.RECORDS / R17_FILE if path is None else path)
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    for key, v in doc.items():
        assert np.isfinite(v['r']) and v['r'] >= 0, (path, key, v['r'])
    return doc


def build_ranges(qwen20, r17_path=None):
    """The whole r table: {key: {r, source, ...}}. `qwen20` is `qwen_ami20`'s result. A missing source gives r null."""
    r17 = r17_record(r17_path)
    out = {}
    for k in range_keys():
        fam, key = k['source'], k['key']
        if fam == 'r17':
            rkey = f"cstar/{k['set']}/{k['enc']}/{k['aspect']}/{k['metric']}"
            if r17 is None:
                e = dict(r=None, source=f'missing:{rkey}: r17 records/{R17_FILE} absent')
            elif rkey not in r17:
                e = dict(r=None, source=f'missing:{rkey} not in r17 records/{R17_FILE}')
            else:
                e = dict(r=float(r17[rkey]['r']), source=f'r17 records/{R17_FILE}:{rkey}')
        elif fam == 'ours':
            q = qwen20[k['aspect']]
            e = dict(r=float(q['r']), source='om_stage1: k-means as r10 on Qwen c* validation exports, seeds 0 to 4, '
                                             'mean over states 0 to 2, max minus min', per_seed=q['per_seed'],
                     n_val=q['n_val'], n_labelled=q['n_labelled'])
        elif fam in ('r14', 'r15'):
            r, src = recorded_ami20(k['set'], k['enc'], k['aspect'])
            e = dict(r=r, source=src)
        else:
            r, src = map_range(k['set'], k['enc'], k['aspect'])
            e = dict(r=float(r), source=src)
        out[key] = e
    assert len(out) == len(range_keys())
    return out


def check_recorded(data, enc, partitions=ns_metrics.partitions):
    """Replay of the method on an encoder whose range IS recorded (r14, ArtELingo AMI at 20): the computed range against the
    record, per aspect. Returns {aspect: {computed, recorded, abs_diff}}."""
    out = {}
    for a in ASPECTS:
        got = cstar_range(data, 'artelingo', enc, a, op.K20, 'ami', partitions)['r']
        want, _ = recorded_ami20('artelingo', enc, a)
        out[a] = dict(computed=got, recorded=want, abs_diff=abs(got - want))
    return out


# ================================================================================================ read-out picks
def units(arm, set_):
    """(unit {aspect: r}, source): Qwen c*'s validation mAP ranges for TPIPS, CLIP L/14 c*'s for CLAY (D7), from the record."""
    assert arm in ('tpips', 'clay'), arm
    return ns_picks.ranges(set_, 'qwen' if arm == 'tpips' else op.CLAY_ENCODER)


def clay_menu(rank):
    """r11's menu with the whitening sizes clipped to the CLAY rank: a size at or above the rank becomes the rank, repeats
    dropped (15 read-outs: none, centering and whitening at `rank`, each with the 5 expansion sizes). Tie order kept."""
    return rt_readout.menu(rank)


def readout_doc(xs, labels, unit, ros=None, workers=8):
    """r17's `ns_picks.readout_doc` with an optional menu `ros` (None: r11's menu at the arm's width). unit {aspect: r}."""
    assert set(labels) == set(ASPECTS) and set(unit) == set(ASPECTS), (tuple(labels), tuple(unit))
    assert all(np.isfinite(v) and v > 0 for v in unit.values()), unit
    table, ro = ac_select.arm_table(xs, labels, unit, ros=ros, workers=workers)
    return dict(pick=rt_readout.name(ro), readout=ro, table=table, ranges={a: float(unit[a]) for a in ASPECTS},
                dropped=rt_readout.half_dropped(labels))


def _inputs(data, set_, rows):
    """(xs, labels) after the order check: `rows` {aspect: float32 unit rows}, one array per aspect, unseeded."""
    keys = audit_validation_only(data, set_)
    ns_picks.check_order(set_, keys)
    labels = {a: np.asarray(data.labels(set_, a), dtype=np.int64) for a in ASPECTS}
    xs = {}
    for a in ASPECTS:
        x = rows(a)
        assert x.dtype == np.float32 and x.ndim == 2 and x.shape[0] == len(keys), (a, x.dtype, x.shape)
        assert np.abs(np.linalg.norm(x, axis=1) - 1).max() < 1e-4, f'{a}: rows are not unit rows'
        xs[(a, None)] = x
    assert all(y.shape == (len(keys),) for y in labels.values())
    return xs, labels


def _pick(xs, labels, unit, source, n_val, arm_name, out, ros=None, workers=8):
    doc = readout_doc(xs, labels, unit, ros=ros, workers=workers)
    rel = f'stage1/tables/{arm_name}.json'
    ns_paths.write_json(out / rel, dict(arm=arm_name, pick=doc['pick'], table=doc['table'], ranges=doc['ranges'],
                                        unit_source=source, dropped=doc['dropped'], n_val=n_val))
    return ns_picks._record(doc, source, n_val, rel)


def tpips_pick(data, set_, unit=None, source=None, workers=8, out=None):
    """TPIPS's read-out pick on the set's validation split: one pick across its three aspects, r11's menu at 4,096 wide,
    units Qwen c*'s validation mAP ranges. Crashes if the store is missing."""
    out = out_dir(out)
    if unit is None:
        unit, source = units('tpips', set_)
    xs, labels = _inputs(data, set_, lambda a: data.tpips(set_, a))
    return _pick(xs, labels, unit, source or 'given', len(next(iter(labels.values()))), f'tpips__{set_}', out, workers=workers)


def unit_score(record, unit):
    """D7's text-set score: the mean over the aspects of validation mAP at the pick divided by the unit (CLIP c*'s range)."""
    return float(np.mean([record['scores']['map'][a] / unit[a] for a in ASPECTS]))


def clay_pick(data, set_, text_set, unit=None, source=None, workers=8, out=None):
    """CLAY's read-out pick for one text set: r11's rule on the set's validation split with the whitening sizes clipped to
    the CLAY rank (the smallest rank over the aspects), the guard against this text set's own R0."""
    out = out_dir(out)
    if unit is None:
        unit, source = units('clay', set_)
    counts = [data.text_count(text_set, a) for a in ASPECTS]
    assert all(c is not None for c in counts), f'set {text_set} has no vectors for some aspect (fallback)'
    rank = min(op.clay_rank(c) for c in counts)
    xs, labels = _inputs(data, set_, lambda a: data.clay(set_, a, text_set))
    rec = _pick(xs, labels, unit, source or 'given', len(next(iter(labels.values()))), f'clay__{set_}__{text_set}', out,
                ros=clay_menu(rank), workers=workers)
    return dict(rec, text_set=text_set, text_rank=int(rank), unit_score=unit_score(rec, unit))


def choose_text_set(score_a, score_b, margin):
    """(set, reason): B only when its score exceeds A's by more than `margin` (equal to the margin is a tie, which goes to
    A); `score_b` None (set B fell back) gives A."""
    if score_b is None:
        return 'A', 'fallback'
    if score_b - score_a > margin + SLACK:
        return 'B', 'margin'
    return 'A', 'tie' if score_b - score_a >= -margin - SLACK else 'a_higher'


def _b_usable(data):
    return all(data.text_count('B', a) is not None for a in ASPECTS)


def clay_retrieval(data, set_, unit=None, source=None, workers=8, out=None):
    """CLAY's retrieval text set and pick for one set (D7): {`clay/<set>`: the chosen pick, `clay_textset/ret/<set>`: the
    choice}. Set B only when no aspect fell back; otherwise A with no pick for B."""
    out = out_dir(out)
    if unit is None:
        unit, source = units('clay', set_)
    sets = TEXT_SETS if _b_usable(data) else ('A',)
    picks = {ts: clay_pick(data, set_, ts, unit, source, workers, out) for ts in sets}
    sa = picks['A']['unit_score']
    sb = picks['B']['unit_score'] if 'B' in picks else None
    which, reason = choose_text_set(sa, sb, RET_MARGIN)
    scores = {ts: (None if ts not in picks else dict(
        score=picks[ts]['unit_score'], pick=picks[ts]['pick'], map=picks[ts]['scores']['map'],
        r0_map=picks[ts]['scores']['r0_map'], rank=picks[ts]['text_rank'])) for ts in TEXT_SETS}
    rule = (f'B only if mean over aspects of validation mAP at the set\'s own pick / CLIP L/14 c* range exceeds A\'s by more '
            f'than {RET_MARGIN}; set B fell back: A, no pick' if reason == 'fallback' else
            f'B only if mean over aspects of validation mAP at the set\'s own pick / CLIP L/14 c* range exceeds A\'s by more '
            f'than {RET_MARGIN}')
    return {f'clay/{set_}': picks[which],
            f'clay_textset/ret/{set_}': dict(set=which, scores=scores, rule=rule, reason=reason, margin=RET_MARGIN,
                                              unit=dict(unit), unit_source=source or 'given')}


def clay_ami20(data, set_, aspect, text_set, partitions=None):
    """Mean over k-means states 0 to 2 of validation AMI at 20 clusters on the CLAY vectors (k-means as r10, all rows
    clustered, labels masked after)."""
    partitions = partitions or ns_metrics.partitions
    x = data.clay(set_, aspect, text_set)
    y = np.asarray(data.labels(set_, aspect), dtype=np.int64)
    return float(ns_metrics.score_partitions('ami', y, partitions(x, op.K20, set_)))


def clay_clustering_choice(data, set_):
    """The clustering text set (D8): the mean over aspects of validation AMI at 20; B only when more than 0.001 above A."""
    audit_validation_only(data, set_)
    sets = TEXT_SETS if _b_usable(data) else ('A',)
    scores = {}
    for ts in TEXT_SETS:
        if ts not in sets:
            scores[ts] = None
            continue
        per = {a: clay_ami20(data, set_, a, ts) for a in ASPECTS}
        scores[ts] = dict(ami20=per, mean_ami20=float(np.mean(list(per.values()))))
    which, reason = choose_text_set(scores['A']['mean_ami20'], scores['B']['mean_ami20'] if scores['B'] else None, CLU_MARGIN)
    rule = (f'B only if the mean over aspects of validation AMI at 20 (k-means as r10, mean over states) exceeds A\'s by more '
            f'than {CLU_MARGIN}' + ('; set B fell back: A' if reason == 'fallback' else ''))
    return dict(set=which, scores=scores, rule=rule, reason=reason, margin=CLU_MARGIN)


# ================================================================================================ pins and the index
def text_pins(data):
    """{'A': {aspect: sha}, 'B': {aspect: sha or None}}: the hash of every text set the picks read."""
    return {ts: {a: data.text_sha(ts, a) for a in ASPECTS} for ts in TEXT_SETS}


def merge_picks(path, records, pins=None):
    """Merge `records` (and the text pins) into the index file, locked and atomic. Pins are frozen: a file that has pins
    refuses different ones, before anything is written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + '.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        have = json.loads(path.read_text()) if path.exists() else {}
        if pins is not None:
            old = have.get(PIN_KEY)
            assert old is None or old == pins, f'{PIN_KEY} are frozen: the text sets changed since stage 1 pinned them'
            have[PIN_KEY] = pins
        have.update(records)
        tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        ns_paths.write_json(tmp, have)
        os.replace(tmp, path)
    return path


# ================================================================================================ CLI
def stage1_dir(out=None):
    return out_dir(out) / 'stage1'


def step_ranges(data, out=None):
    audit_validation_only(data, 'artelingo')
    table = build_ranges(qwen_ami20(data))
    path = stage1_dir(out) / 'om_ranges.json'
    ns_paths.write_json(path, table)
    return path, table


def step_tpips(data, set_, workers, out=None):
    rec = tpips_pick(data, set_, workers=workers, out=out)
    merge_picks(stage1_dir(out) / 'om_picks.json', {f'tpips/{set_}': rec})
    return rec


def step_clay(data, set_, workers, out=None):
    recs = clay_retrieval(data, set_, workers=workers, out=out)
    recs[f'clay_textset/clu/{set_}'] = clay_clustering_choice(data, set_)
    merge_picks(stage1_dir(out) / 'om_picks.json', recs, pins=text_pins(data))
    return recs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('ranges', help='the r table -> OUT/stage1/om_ranges.json (rerun when r17\'s record arrives)')
    s = sub.add_parser('tpips', help='TPIPS read-out pick of one set (needs the TPIPS stores)')
    s.add_argument('--set', required=True, choices=SETS)
    s = sub.add_parser('clay', help='CLAY text sets and read-out picks of one set')
    s.add_argument('--set', required=True, choices=SETS)
    sub.add_parser('pins', help='write the text hashes only')
    s = sub.add_parser('check-recorded', help='replay the AMI-at-20 range of an encoder whose range is recorded (r14)')
    s.add_argument('--enc', required=True, choices=[e for e in ENCODERS if e != 'qwen'])
    args = p.parse_args(argv)
    t0, data = time.monotonic(), ValData()
    if args.cmd == 'ranges':
        path, table = step_ranges(data)
        nulls = sorted(k for k, v in table.items() if v['r'] is None)
        print(f'ranges: {len(table)} keys, {len(nulls)} with r null (source missing) -> {path} ({time.monotonic() - t0:.0f}s)')
    elif args.cmd == 'tpips':
        rec = step_tpips(data, args.set, args.workers)
        print(f'tpips/{args.set}: pick {rec["pick"]} score {rec["scores"]["score"]:.4f} ({time.monotonic() - t0:.0f}s)')
    elif args.cmd == 'clay':
        recs = step_clay(data, args.set, args.workers)
        r, c = recs[f'clay_textset/ret/{args.set}'], recs[f'clay_textset/clu/{args.set}']
        print(f'clay/{args.set}: retrieval text set {r["set"]} pick {recs[f"clay/{args.set}"]["pick"]}; clustering text set '
              f'{c["set"]} ({time.monotonic() - t0:.0f}s)')
    elif args.cmd == 'pins':
        merge_picks(stage1_dir() / 'om_picks.json', {}, pins=text_pins(data))
        print('pins written')
    else:
        res = check_recorded(data, args.enc)
        for a, v in res.items():
            print(f'{args.enc}/{a}: computed {v["computed"]!r} recorded {v["recorded"]!r} diff {v["abs_diff"]:.3g}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
