"""Validation picks and seed ranges: text-claim design §3.4, §3.5 and §6 selection rule."""
import json

import numpy as np

import tcl_paths as p
from tcl_metrics import summarize_pair


def _read_score(set_name, kind, row, head, seed, direction, readout, alignment):
    assert set_name in ('art_val', 'mm_val'), 'stage 1 reads validation only'
    path = p.score_path(set_name, kind, row, head, seed, direction, readout)
    assert path.parents[2].name == path.resolve().parents[2].name == set_name, \
        f'stage 1 needs a validation score path: {path}'
    aspects = (head,) if row in p.HEAD_ROWS else (('emotion',) if row == 'roberta' else p.ASPECTS)
    with np.load(path, allow_pickle=False) as score:
        required = {'q_pid', 'first_rank'} | {f'ap__{aspect}' for aspect in aspects}
        assert required <= set(score.files), f'missing score columns: {path}'
        pid, ranks = score['q_pid'], score['first_rank']
        n_i = min(p.SMOKE or p.N_PAINTINGS[set_name], p.N_PAINTINGS[set_name])
        text = set_name == 'art_val' and direction != 'i2t'
        if text:
            # Smoke keeps every caption of the selected paintings, so its count varies.
            n_q = len(pid) if p.SMOKE else p.N_CAPTIONS[set_name]
        else:
            n_q = n_i
        assert pid.shape == (n_q,) and pid.dtype == np.int64, f'query IDs: {path}'
        assert ((pid >= 0) & (pid < n_i)).all(), f'painting positions: {path}'
        if direction == 'i2t' or set_name == 'mm_val':
            assert np.array_equal(pid, np.arange(n_i)), f'painting order: {path}'
        reference = alignment.setdefault((kind, direction), pid)
        assert np.array_equal(pid, reference), f'query ID alignment: {path}'
        assert ranks.shape == pid.shape and ranks.dtype == np.float32, f'first_rank shape/dtype: {path}'
        valid = np.isfinite(ranks)
        assert (valid | np.isnan(ranks)).all(), f'infinite rank: {path}'
        assert ((ranks[valid] >= 1) & (ranks[valid] == np.floor(ranks[valid]))).all(), f'ranks: {path}'
        aps = {}
        for aspect in aspects:
            ap = score[f'ap__{aspect}']
            assert ap.shape == pid.shape and ap.dtype == np.float32, f'AP shape/dtype: {path}'
            valid = np.isfinite(ap)
            assert valid.any() and (valid | np.isnan(ap)).all(), f'no scored AP or infinite AP: {path}'
            assert ((ap[valid] >= 0) & (ap[valid] <= 1)).all(), f'AP bounds: {path}'
            aps[aspect] = ap
    return aps, ranks


def _pair_picks(set_name, kind, alignment):
    picks = {}
    for direction in p.PAIR_DIRECTIONS:
        candidates = {}
        for row in ('late', 'plain'):
            heads = p.ASPECTS if row == 'late' else ('all',)
            seeds = p.SEEDS if row == 'late' else ('x',)
            for head in heads:
                summaries = {}
                for readout in p.READOUTS:
                    per_seed = []
                    for seed in seeds:
                        _, ranks = _read_score(set_name, kind, row, head, seed, direction, readout, alignment)
                        per_seed.append(summarize_pair(ranks.astype(np.float64)))
                    # Query counts match across heads; integer totals preserve exact R@10 ties.
                    summaries[readout] = dict(mrr=float(np.mean([s['mrr'] for s in per_seed])),
                                              r10=sum(round(s['r10'] * s['n_query']) for s in per_seed))
                picked = max(p.READOUTS, key=lambda r: summaries[r]['mrr'])
                candidates[head] = (picked, summaries[picked]['r10'])
        head = max(p.ASPECTS, key=lambda h: candidates[h][1])
        picks[direction] = dict(late=dict(head=head, readout=candidates[head][0]),
                                plain=dict(readout=candidates['all'][0]))
    return picks


def run(set_name):
    """Read a full validation sweep, write its stage1_<group>.json, and return the record."""
    assert set_name in ('art_val', 'mm_val'), 'stage 1 reads validation only'
    group = p.GROUP[set_name]
    record = dict(group=group, picks={}, values={}, ranges={}, pair_picks={})
    alignment = {}
    rows = p.BAR_ROWS + (p.SIDE_ROWS if group == 'art' else ())
    for kind in p.kinds(group):
        picks, values, ranges = {}, {}, {}
        record['picks'][kind], record['values'][kind], record['ranges'][kind] = picks, values, ranges
        for row in rows:
            picks[row], values[row] = {}, {}
            if row in p.SEEDED_ROWS:
                ranges[row] = {}
            aspects = ('emotion',) if row == 'roberta' else p.ASPECTS
            seeds = p.SEEDS if row in p.SEEDED_ROWS else ('x',)
            directions = ('t2t',) if row == 'roberta' else p.DIRECTIONS
            for aspect in aspects:
                picks[row][aspect], values[row][aspect] = {}, {}
                if row in p.SEEDED_ROWS:
                    ranges[row][aspect] = {}
                head = aspect if row in p.HEAD_ROWS else 'all'
                for direction in directions:
                    per_seed = {}
                    for readout in p.READOUTS:
                        maps = []
                        for seed in seeds:
                            aps, _ = _read_score(set_name, kind, row, head, seed, direction, readout, alignment)
                            ap = aps[aspect]
                            maps.append(float(ap[np.isfinite(ap)].mean(dtype=np.float64)))
                        per_seed[readout] = maps
                    means = {readout: float(np.mean(per_seed[readout])) for readout in p.READOUTS}
                    picked = max(p.READOUTS, key=means.__getitem__)
                    picks[row][aspect][direction] = picked
                    values[row][aspect][direction] = means
                    if row in p.SEEDED_ROWS:
                        ranges[row][aspect][direction] = max(per_seed[picked]) - min(per_seed[picked])
        record['pair_picks'][kind] = _pair_picks(set_name, kind, alignment)
    path = p.record_path(f'stage1_{group}.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + '\n')
    return record
