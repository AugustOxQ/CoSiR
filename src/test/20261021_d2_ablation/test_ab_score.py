"""Fast tests for ab_score (ticket 04): draw list, self-effect, missing rows. Synthetic data, no model outputs."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_score as sc  # noqa: E402


def synthetic(n=300, rows=('R0', 'R1', 'R2'), seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, size=n)
    y[::7] = -1  # unlabelled paintings
    parts = {}
    for k, row in enumerate(rows):
        parts[row] = {s: np.stack([np.where(rng.random(n) < .5 + .1 * k, y, rng.integers(0, 8, size=n)).clip(0)
                                   for _ in range(3)]) for s in sc.SEEDS}
    return y, parts


def test_draw_list_is_deterministic_and_shared():
    labelled = np.flatnonzero(np.arange(100) % 5 != 0)
    a, b = sc.draw_list(labelled, 20), sc.draw_list(labelled, 20)
    assert len(a) == 20
    for (ia, ca), (ib, cb) in zip(a, b):
        assert np.array_equal(ia, ib) and list(ca) == list(cb) == list(sc.ROW_ORDER)
        assert all(np.array_equal(ca[r], cb[r]) for r in ca)
        assert set(ia) <= set(labelled) and len(ia) == len(labelled)  # only labelled paintings, resampled
        assert all(len(c) == 3 and set(c) <= set(sc.SEEDS) for c in ca.values())
    # a longer list starts with the shorter one; the list does not depend on which rows are scored
    longer = sc.draw_list(labelled, 30)
    assert all(np.array_equal(x[0], z[0]) for x, z in zip(a, longer))
    assert all(np.array_equal(x[1]['R12'], z[1]['R12']) for x, z in zip(a, longer))


def test_effect_against_itself_is_exactly_zero():
    y, parts = synthetic()
    draws = sc.draw_list(np.flatnonzero(y >= 0), 20)
    result = sc.bootstrap_effects(parts, y, draws, workers=1)
    r0 = result['R0']
    assert r0['effect'] == 0.0 and r0['lo'] == 0.0 and r0['hi'] == 0.0
    assert result['R1']['effect'] != 0.0 and result['R1']['lo'] <= result['R1']['hi']
    # the same draws give the same numbers whatever the pool
    again = sc.bootstrap_effects(parts, y, draws, workers=2)
    assert again['R1']['lo'] == result['R1']['lo'] and again['R2']['hi'] == result['R2']['hi']
    # a missing row leaves the other rows' intervals unchanged
    fewer = sc.bootstrap_effects({r: parts[r] for r in ('R0', 'R2')}, y, draws, workers=1)
    assert fewer['R2'] == result['R2']
    # without the reference row there is no effect
    assert sc.bootstrap_effects({'R1': parts['R1']}, y, draws, workers=1)['R1']['effect'] is None


def test_missing_rows_are_listed(tmp_path):
    train = tmp_path / 'train'
    for row, skip in (('R0', None), ('R1', ('style', 2)), ('R3', 'meta')):
        folder = train / row
        folder.mkdir(parents=True)
        for aspect in sc.ASPECTS:
            for seed in sc.SEEDS:
                if skip == (aspect, seed):
                    continue
                (folder / f'{aspect}__G__s{seed}.npz').write_bytes(b'')
                (folder / f'{aspect}__G__s{seed}.json').write_text('{}')
        if skip != 'meta':
            (folder / 'ab_row.json').write_text('{"name": "%s", "caption_side": "scored"}' % row)
    info, reasons = sc.find_row(tmp_path, 'R0')
    assert not reasons and info['caption_scored'] and len(info['files']) == 9
    assert sc.find_row(tmp_path, 'R1')[1] == ['style seed 2: 0 npz files']
    assert sc.find_row(tmp_path, 'R3')[1] == ['no ab_row.json']
    assert sc.find_row(tmp_path, 'R5')[0] is None and 'no folder' in sc.find_row(tmp_path, 'R5')[1][0]


def test_caption_side_not_scored(tmp_path):
    folder = tmp_path / 'train' / 'R7'
    folder.mkdir(parents=True)
    for aspect in sc.ASPECTS:
        for seed in sc.SEEDS:
            (folder / f'{aspect}__G__s{seed}.npz').write_bytes(b'')
            (folder / f'{aspect}__G__s{seed}.json').write_text('{}')
    (folder / 'ab_row.json').write_text('{"name": "R7", "caption_side": "not scored"}')
    assert sc.find_row(tmp_path, 'R7')[0]['caption_scored'] is False


def test_summarize_row_not_scored_caption():
    def side(base, probe):
        r = dict(kmeans={m: dict(mean=base + i, sd=0.0) for i, m in enumerate(('ami', 'v', 'ari', 'fmi'))},
                 leiden_ami=[base, base + .1, base + .2])
        if probe:
            r['probe'] = dict(balanced_accuracy=.5 + base, per_class_recall=[.1, .9], classes=[0, 1], majority_rate=.6,
                              chance=.5, n=10, convergence_warnings=0)
        return r
    per_seed = {s: dict(img=side(s, True), cap=side(s + .5, s == 0)) for s in sc.SEEDS}
    full = sc.summarize_row(per_seed, True)
    assert full['img']['kmeans']['ami']['per_seed'] == [0.0, 1.0, 2.0] and 'nmi' in full['img']['kmeans']
    assert abs(full['img']['kmeans']['ami']['sd'] - np.std([0., 1., 2.])) < 1e-12
    assert full['img']['probe']['seeds'] == [0, 1, 2] and full['cap']['probe']['seeds'] == [0]
    assert sc.summarize_row(per_seed, False)['cap'] == 'not scored'
