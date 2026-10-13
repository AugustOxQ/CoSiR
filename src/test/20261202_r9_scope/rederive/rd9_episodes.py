"""R9 §5.1 shares, language-local indices, set-aside passes and assembly.

Only the frozen episode primitive and validator listed by R6 §8.4 are imported.
The language loop, values restriction, sub-array mapping and assembly are ours.
"""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from rd9_stats import shares
from rd9_inputs import MAIN, HERE, save_json, save_npz
from src.eval.aspect_episodes import (AspectEpisodes, PaintingValueIndex,
    eligible_values, validate_aspect_episodes, episodes_sha256)

PAIRS = (('emotion','style','genre'), ('emotion','genre','style'), ('style','genre','emotion'))
FIELDS = ('anchor','candidates','pairs_a_img','pairs_a_txt','pairs_b_img','pairs_b_txt')
LANGUAGES = ('Burmese','Darija','Emakhuwa','Hausa','Hindi','Igbo','Indonesian','IsiNdebele','IsiXhosa','IsiZulu','Kazakh','Kinyarwanda','Korean','Kyrgyz','Malay','Setswana','Swahili','Tagalog','Tamil','Thai','Turkish','Urdu','Uzbek','Vietnamese','Yoruba','arabic','chinese')


class ValuesSubsetError(AssertionError):
    """The only assertion that §5.1.6 allows us to catch."""


def failed_draw_count(message, a, b, n):
    match = re.fullmatch(re.escape(f'{a} x {b}: could not fill {n} episodes (')
                         + r'(\d+) failed draws\); the pools are too small for these constraints', message)
    if match and int(match[1]) > 10 * n:
        return int(match[1])
    return None


def pool_rows(labels, rows):
    rows = np.asarray(rows, np.int64)
    return rows[np.logical_and.reduce([labels[a][rows] >= 0 for a in ('emotion','style','genre')])]


def restricted(labels, groups, rows, a, b, n, seed, third, index, values, min_paintings=30):
    # Reuses rd_common.build_restricted's identity-based restriction, with min_paintings added.
    import src.eval.aspect_episodes as ae
    original = ae.eligible_values
    seen = []
    def eligibility(lab, grp, pool, minimum):
        valid = original(lab, grp, pool, minimum)
        if lab is index.labels[a]:
            wanted, tag = values[0], 'a'
        elif lab is index.labels[b]:
            wanted, tag = values[1], 'b'
        else:
            raise AssertionError('unexpected label array')
        if not set(wanted) <= set(valid):
            raise ValuesSubsetError(f'{tag}: missing values {sorted(set(wanted)-set(valid))}')
        seen.append(tag)
        return sorted(set(valid) & set(wanted))
    ae.eligible_values = eligibility
    try:
        ep = ae.build_aspect_episodes(labels, groups, rows, a, b, n, seed,
                                     third=third, min_paintings=min_paintings, index=index)
    finally:
        ae.eligible_values = original
    assert seen == ['a','b']
    return ep


def concat(parts, a='mixed', b='mixed'):
    assert parts
    return AspectEpisodes(a, b, *(np.concatenate([getattr(ep,f) for ep in parts]) for f in FIELDS))


def value_codes(record):
    # R6 writes per-aspect {codes, names}; accept plain code arrays for synthetic callers.
    if 'values' in record:
        record = record['values']
    result = {}
    for a in ('emotion','style','genre'):
        v = record[a]
        if isinstance(v, dict):
            v = v['codes']
        assert len(v) == len(set(v)) and all(isinstance(x, int) and x >= 0 for x in v)
        result[a] = sorted(v)
    return result


def table_arrays(table, real=True):
    expected = {'row_id','painting','language','emotion','style','genre','held','dropped'}
    assert set(table.columns) == expected, set(table.columns)
    ids = table.row_id.to_numpy()
    assert ids.dtype.kind in 'iu' and len(np.unique(ids)) == len(ids)
    assert np.all(np.diff(ids) > 0)
    assert table.held.dtype == bool and table.dropped.dtype == bool
    assert table.painting.notna().all() and table.language.notna().all()
    labels = {a: table[a].to_numpy(dtype=np.int64) for a in ('emotion','style','genre')}
    for a in labels:
        assert table[a].dtype.kind in 'iu' and (labels[a] >= -1).all()
    _, groups = np.unique(table.painting.to_numpy(), return_inverse=True)
    languages = sorted(table.language.unique())
    if real:
        assert len(table) == 74569 and len(np.unique(groups)) == 746
        assert languages == list(LANGUAGES)
    return labels, groups, languages


def build_a28(table, values, seed, total=4096, real=True):
    labels, groups, languages = table_arrays(table, real)
    codes = value_codes(values)
    if real:
        assert len(codes['emotion']) == 8
        assert codes['style'] == [9,12,17,20,21,23,24]
        assert codes['genre'] == [0,1,2,4,6,7,8,9]
    keep = ~table.dropped.to_numpy()
    lang_col = table.language.to_numpy()
    # Indices use the rows BEFORE drops; allocation counts only kept fully known rows.
    indices, global_rows, kept_positions, sizes = {}, {}, {}, {}
    for lang in languages:
        R = np.flatnonzero(lang_col == lang)
        sub = {a: labels[a][R] for a in labels}
        rows = np.flatnonzero(keep[R])
        global_rows[lang] = R
        kept_positions[lang] = rows
        indices[lang] = PaintingValueIndex(sub, groups[R])
        sizes[lang] = len(pool_rows(sub, rows))
    episodes, record = {}, {}
    for a,b,third in PAIRS:
        remaining = list(languages)
        excluded, passes = [], []
        while True:
            allocations = dict(zip(remaining, shares([sizes[l] for l in remaining], total)))
            parts, hashes, failed = [], {}, []
            for lang in remaining:
                n = allocations[lang]
                if n == 0:
                    # No draw or subset check when a language has zero allocated episodes.
                    hashes[lang] = None
                    continue
                R = global_rows[lang]
                sub = {x: labels[x][R] for x in labels}
                idx = indices[lang]
                local_seed = 1000 * seed + LANGUAGES.index(lang) + 1
                try:
                    part = restricted(sub, groups[R], kept_positions[lang], a,b,n,local_seed,third,
                                      idx,(codes[a],codes[b]),min_paintings=1)
                except ValuesSubsetError:
                    failed.append(dict(language=lang, reason='values-subset', seed=local_seed))
                    continue
                except RuntimeError as error:
                    # The frozen primitive's only RuntimeError is its failed-draw limit.
                    if failed_draw_count(str(error), a, b, n) is None:
                        raise
                    failed.append(dict(language=lang, reason='failed-draw', seed=local_seed))
                    continue
                validate_aspect_episodes(part, sub, groups[R], third=third, index=idx)
                mapped = AspectEpisodes(a,b,*(R[getattr(part,f)] for f in FIELDS))
                assert len(mapped.anchor) == n and keep[mapped.rows()].all()
                assert (lang_col[mapped.rows()] == lang).all()
                parts.append(mapped)
                hashes[lang] = episodes_sha256(mapped)
            passes.append(dict(shares=allocations, failed=failed, part_hashes=hashes))
            excluded.extend(failed)
            assert len(excluded) <= 2, f'more than two set aside: {excluded}'
            if not failed:
                combined = concat(parts,a,b)
                assert len(combined.anchor) == total
                key = f'{a}__{b}'
                episodes[key] = combined
                record[key] = dict(passes=passes, set_aside=excluded, episodes_sha256=episodes_sha256(combined))
                break
            remaining = [l for l in remaining if l not in {f['language'] for f in failed}]
    return episodes, record


def build_standard(labels, groups, rows, values, seed, total=4096):
    index = PaintingValueIndex(labels, groups)
    out = {}
    for a,b,t in PAIRS:
        ep = restricted(labels,groups,rows,a,b,total,seed,t,index,(values[a],values[b]))
        validate_aspect_episodes(ep,labels,groups,third=t,index=index)
        assert np.isin(ep.rows(), rows).all()
        out[f'{a}__{b}'] = ep
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--table', type=Path, default=HERE/'results/read_set_a28.parquet')
    p.add_argument('--values', type=Path, default=HERE/'results/value_sets_a28.json')
    p.add_argument('--seeds', type=int, nargs='+', default=[9001,9002,9003])
    p.add_argument('--total', type=int, default=4096)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    import pandas as pd
    table = pd.read_parquet(args.table)
    values = json.loads(args.values.read_text())
    arrays, rec = {}, {}
    for s in args.seeds:
        eps, rec[str(s)] = build_a28(table,values,s,total=args.total)
        arrays.update({f'seed{s}__{key}__{f}':getattr(ep,f) for key,ep in eps.items() for f in FIELDS})
    save_npz(args.out.with_suffix('.npz'), arrays)
    save_json(args.out.with_suffix('.json'), rec)
    print('episodes complete:',len(args.seeds),'seeds; labels only')


if __name__ == '__main__':
    main()
