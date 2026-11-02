"""Tests of bs_affection (r14 ticket 07) on synthetic data: the pick with the model's own ranges, the SigLIP rule with r12's
r_ret,mAP, the SC6 `r` unit, the cell format, key joins. No real Affection data is read."""
import json

import numpy as np
import pytest

import bs_affection as A   # first: puts r13's round (and r12, r11, r10) on sys.path
import bs_stores
import ac_cluster
import ac_data
import ac_stats
import pf_stage3

N, D = 240, 16
R12_BIG = 1000.0                       # a range so large that no read-out beats R0 by 0.001 r


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def make_keys(n=N):
    return [f'src/img_{i:04d}.jpg' for i in range(n)]            # sorted


def make_labels(n=N):
    return (np.arange(n) % 9).astype(np.int64)


def make_dist(labels):
    dist = np.full((len(labels), 9), 0.02)
    dist[np.arange(len(labels)), labels] = 1 - 0.02 * 8
    return dist / dist.sum(axis=1, keepdims=True)


def make_rows(labels, seed, offset=4.0, noise=0.9, d=D):
    """Class signal plus a large common offset (centering helps) and seed-specific noise."""
    rng = np.random.default_rng(seed)
    centers = np.random.default_rng(99).normal(size=(9, d))
    x = offset * np.ones(d) + centers[labels] + noise * rng.normal(size=(len(labels), d))
    return unit(x)


def five_seeds(labels, **kw):
    embed = {s: make_rows(labels, 10 + s, **kw) for s in range(5)}
    stored = {s: e.astype(np.float16) for s, e in embed.items()}
    embed = {s: ac_data._unit(stored[s]) for s in range(5)}      # as the loader: stored rows cast and normalized
    return embed, stored


def ranges_with_map_r(r):
    """A consistent r12 `ranges.json`-form document whose mAP range is exactly r."""
    return ac_stats.ranges_doc([0.10, 0.11, 0.10, 0.10, 0.10], [0.2, 0.2 + r, 0.2, 0.2, 0.2], [0.9, 0.91, 0.9, 0.9, 0.9])


@pytest.fixture(scope='module')
def val():
    labels, keys = make_labels(), make_keys()
    embed, stored = five_seeds(labels)
    return dict(labels=labels, keys=keys, dist=make_dist(labels), embed=embed, stored=stored)


# ------------------------------------------------------------------------------------- the pick with own ranges
def test_photo_pick_uses_the_models_own_range_not_r12s(val):
    rec = A.photo_stage1(val['embed'], val['stored'], val['labels'], val['dist'], val['keys'],
                         r12_ranges=ranges_with_map_r(R12_BIG), width=D, workers=1)
    own = rec['ranges']['map']['r']
    assert 0 < own < 0.1 and own != R12_BIG
    pick, table = rec['readout']['pick'], rec['readout']['scores']
    assert pick != 'none+k0', 'with its own small range the offset-removing read-out must win'
    base = table['none+k0']['map']['emotion']
    for name, row in table.items():                    # every score is in units of the model's own range
        assert row['score'] == pytest.approx((row['map']['emotion'] - base) / own, abs=1e-12)
    assert rec['r12_ranges']['map']['r'] == pytest.approx(R12_BIG)


def test_photo_pick_range_is_measured_from_the_five_seeds(val):
    rec = A.photo_stage1(val['embed'], val['stored'], val['labels'], val['dist'], val['keys'],
                         r12_ranges=ranges_with_map_r(0.001), width=D, workers=1)
    per_seed = rec['ranges']['map']['per_seed']
    assert len(per_seed) == 5 and rec['ranges']['map']['r'] == max(per_seed) - min(per_seed)


# ------------------------------------------------------------------------------------- the frozen-image rule
R12_PICKS = {'readout': {'plain': 'center+k10', 'clip': 'none+k0', 'cstar': 'center+k10'}}


def frozen_x(val):
    return val['embed'][0]


def test_siglip_pick_uses_r12s_range(val):
    small = A.frozen_readout('siglip2', frozen_x(val), val['labels'], val['keys'], ranges_with_map_r(0.001), R12_PICKS,
                             workers=1)
    big = A.frozen_readout('siglip1', frozen_x(val), val['labels'], val['keys'], ranges_with_map_r(R12_BIG), R12_PICKS,
                           workers=1)
    assert small['pick'] != 'none+k0' and small['r'] == pytest.approx(0.001)
    assert big['pick'] == 'none+k0' and big['r'] == pytest.approx(R12_BIG)
    assert small['source'] == "r12's rule with r12's r_ret,mAP" and small['scores']['none+k0']['score'] == 0


def test_siglip_pick_matches_the_rule_called_directly(val):
    from ac_select import arm_table
    import rt_readout
    r = 0.002
    table, pick = arm_table({('emotion', None): frozen_x(val)}, {'emotion': val['labels']}, {'emotion': r},
                            keys=val['keys'], workers=1)
    got = A.frozen_readout('siglip2', frozen_x(val), val['labels'], val['keys'], ranges_with_map_r(r), R12_PICKS, workers=1)
    assert got['pick'] == rt_readout.name(pick)
    assert {k: v['score'] for k, v in got['scores'].items()} == pytest.approx({k: v['score'] for k, v in table.items()}, abs=1e-9)


def test_clip_l14_and_qwen_keep_r12s_pick_without_looking_at_data(val):
    for enc, arm in (('clip_l14', 'clip'), ('qwen', 'plain')):
        got = A.frozen_readout(enc, frozen_x(val), val['labels'], val['keys'], ranges_with_map_r(0.001), R12_PICKS,
                               workers=1)
        assert got['pick'] == R12_PICKS['readout'][arm] and got['r'] is None and got['scores'] is None
    forced = A.frozen_readout('clip_l14', frozen_x(val), val['labels'], val['keys'], ranges_with_map_r(0.001), R12_PICKS,
                              workers=1, force_rule=True)
    assert forced['scores'] is not None and forced['pick'] != 'none+k0'


# ------------------------------------------------------------------------------------- key joins
def write_export(root, arm, split, seed, keys, emb):
    path = A.export_path(arm, split, seed, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, keys=np.asarray(keys), emb=emb)
    path.with_suffix('.json').write_text(json.dumps({'sha256': A.rs_vh.file_sha(path)}))
    return path


def test_load_export_joins_by_key_not_position(tmp_path, val):
    keys = val['keys']
    perm = np.random.default_rng(1).permutation(len(keys))
    write_export(tmp_path, 'photo_siglip2', 'val', 0, [keys[i] for i in perm], val['stored'][0][perm])
    got = A.load_export('photo_siglip2', 'val', 0, keys, tmp_path)
    assert np.array_equal(got, val['stored'][0])
    sub = [keys[5], keys[2], keys[100]]                               # a subset in another order
    assert np.array_equal(A.load_export('photo_siglip2', 'val', 0, sub, tmp_path), val['stored'][0][[5, 2, 100]])


def test_load_export_refuses_missing_duplicate_and_changed_files(tmp_path, val):
    keys, emb = val['keys'], val['stored'][0]
    write_export(tmp_path, 'photo_a', 'val', 0, keys[:-1] + ['src/other.jpg'], emb)
    with pytest.raises(AssertionError, match='not in the file'):
        A.load_export('photo_a', 'val', 0, keys, tmp_path)
    write_export(tmp_path, 'photo_b', 'val', 0, keys[:-1] + [keys[0]], emb)
    with pytest.raises(AssertionError, match='duplicate'):
        A.load_export('photo_b', 'val', 0, keys, tmp_path)
    path = write_export(tmp_path, 'photo_c', 'val', 0, keys, emb)
    path.with_suffix('.json').write_text(json.dumps({'sha256': '0' * 64}))
    with pytest.raises(AssertionError, match='sidecar'):
        A.load_export('photo_c', 'val', 0, keys, tmp_path)


def test_frozen_embed_joins_a_store_listing_other_keys_in_other_order(tmp_path, monkeypatch):
    store_keys = [f'src/img_{i:04d}.jpg' for i in range(30)][::-1]    # the store lists them reversed
    raw = np.random.default_rng(2).normal(size=(30, bs_stores.WIDTH['siglip2'])).astype(np.float16)
    folder = tmp_path / 'siglip2'
    folder.mkdir()
    np.save(folder / 'img.npy', raw)
    (folder / 'keys.json').write_text(json.dumps(store_keys))
    monkeypatch.setattr(bs_stores, 'folders', lambda dataset, view: [folder])
    want = [f'src/img_{i:04d}.jpg' for i in (3, 17, 0)]
    got = A.frozen_embed('siglip2', want)
    rows = [store_keys.index(k) for k in want]
    assert np.allclose(got, ac_data._unit(raw[rows].astype(np.float32)), atol=0)


# ------------------------------------------------------------------------------------- cells and the SC6 unit
def comparator(arm, readout, x_val, x_test, labels, tmp_path):
    res = ac_stats.arm_retrieval(lambda s: x_val, lambda s: x_test, readout, labels, seeds=None, workers=1)
    return dict(arm=arm, readout=readout, ap=res['ap'], rows=res['rows'], point=res['point'],
                labels=ac_cluster.cached_partitions(x_test, f'cmp__{arm}', tmp_path))


@pytest.fixture(scope='module')
def scored(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('score')
    labels = make_labels()
    embed_v, stored_v = five_seeds(labels, noise=0.9)
    embed_t, stored_t = five_seeds(labels, noise=0.9)
    cand = pf_stage3.Candidate(name='photo_siglip2', readout='center+k3',
                               test_stored=np.stack([stored_t[s] for s in range(5)]),
                               test_embed=np.stack([embed_t[s] for s in range(5)]),
                               val_embed=np.stack([embed_v[s] for s in range(5)]))
    frozen = comparator('frozen_siglip2', 'center+k0', make_rows(labels, 500, noise=1.6), make_rows(labels, 501, noise=1.6),
                        labels, tmp)
    teacher = comparator('teacher_artwork', 'center+k10', make_rows(labels, 502, noise=1.3), make_rows(labels, 503, noise=1.3),
                         labels, tmp)
    r = {'clu': 0.0071, 'map': 0.0024}
    res = A.score_cells('siglip2', cand, r, labels, {'frozen': frozen, 'teacher': teacher}, tmp / 'parts', draws_clu=40,
                        draws_ret=60, workers=1)
    return dict(res=res, r=r, frozen=frozen, teacher=teacher, tmp=tmp)


def test_four_cells_in_the_index_format(scored):
    cells = scored['res']['cells']
    assert [(c['sc'], c['task']) for c in cells] == [('SC4', 'clu'), ('SC5', 'ret'), ('SC6', 'clu'), ('SC6', 'ret')]
    for c in cells:
        assert set(A.CELL_KEYS) <= set(c)
        assert (c['encoder'], c['aspect'], c['data'], c['candidate']) == ('siglip2', 'emotion', 'affection_test', 'photo_siglip2')
        assert c['seeds'] == [0, 1, 2, 3, 4] and c['lo'] <= c['hi']
        assert c['d'] == pytest.approx(c['candidate_value'] - c['comparator_value'], abs=1e-9)
    assert [c['comparator'] for c in cells] == ['frozen_siglip2', 'frozen_siglip2', 'teacher_artwork', 'teacher_artwork']
    assert [c['n_draws'] for c in cells] == [40, 60, 40, 60]
    assert cells[1]['readout_candidate'] == 'center+k3' and cells[1]['readout_comparator'] == 'center+k0'
    assert cells[3]['readout_comparator'] == 'center+k10'
    json.dumps(cells, allow_nan=False)


def test_sc6_carries_the_models_own_range_by_task_and_wins_carry_none(scored):
    c4, c5, c6clu, c6ret = scored['res']['cells']
    assert c4['r'] is None and c5['r'] is None                    # WIN-only bars
    assert c6clu['r'] == scored['r']['clu'] == 0.0071            # clustering unit: r'_clu
    assert c6ret['r'] == scored['r']['map'] == 0.0024            # retrieval unit: r'_mAP, not r'_clu, not r12's
    assert 'clustering AMI' in c6clu['r_source'] and 'mAP' in c6ret['r_source'] and 'own' in c6ret['r_source']


def test_cell_values_are_the_candidates_and_comparators_scores(scored):
    c4, c5, c6clu, c6ret = scored['res']['cells']
    picked = scored['res']['picked']
    assert c5['candidate_value'] == pytest.approx(picked['point']) and c6ret['candidate_value'] == pytest.approx(picked['point'])
    assert c5['comparator_value'] == pytest.approx(scored['frozen']['point'])
    assert c6ret['comparator_value'] == pytest.approx(scored['teacher']['point'])
    ami = ac_cluster.ami(scored['res']['partitions'], make_labels())
    assert c4['candidate_value'] == pytest.approx(ami) and c6clu['candidate_value'] == pytest.approx(ami)
    assert c4['comparator_value'] == pytest.approx(ac_cluster.ami(scored['frozen']['labels'], make_labels()))
    assert c6clu['comparator_value'] == pytest.approx(ac_cluster.ami(scored['teacher']['labels'], make_labels()))


def test_score_refuses_queries_that_differ_between_the_two_arms(scored):
    labels = make_labels()
    bad = dict(scored['frozen'], rows=scored['frozen']['rows'][::-1])
    cand = pf_stage3.Candidate  # the fixture's candidate is rebuilt cheaply below
    embed, stored = five_seeds(labels)
    c = cand(name='photo_siglip2', readout='none+k0', test_stored=np.stack([stored[s] for s in range(5)]),
             test_embed=np.stack([embed[s] for s in range(5)]), val_embed=np.stack([embed[s] for s in range(5)]))
    with pytest.raises(AssertionError, match='different queries'):
        A.score_cells('siglip2', c, scored['r'], labels, {'frozen': bad, 'teacher': scored['teacher']}, scored['tmp'] / 'p2',
                      draws_clu=5, draws_ret=5, workers=1)


def test_make_cell_refuses_a_level_unit_on_a_win_bar_and_none_on_sc6():
    kw = dict(task='clu', encoder='siglip2', candidate='c', comparator='f', candidate_value=0.2, comparator_value=0.1, d=0.1,
              lo=0.05, hi=0.15, r_source=None, n_draws=10)
    A.make_cell('SC4', r=None, **kw)
    with pytest.raises(AssertionError):
        A.make_cell('SC4', r=0.003, **kw)
    with pytest.raises(AssertionError):
        A.make_cell('SC6', r=None, **kw)
    with pytest.raises(AssertionError):
        A.make_cell('SC6', r=float('nan'), **kw)
    with pytest.raises(AssertionError):
        A.make_cell('SC1', r=None, **kw)


def test_write_cells_replaces_an_encoders_cells_and_keeps_the_others(tmp_path, scored):
    cells = scored['res']['cells']
    path = tmp_path / 'stage3' / A.CELLS_NAME
    A.write_cells(path, cells, ['siglip2'])
    other = [dict(c, encoder='clip_l14') for c in cells]
    A.write_cells(path, other, ['clip_l14'])
    again = [dict(c, candidate_value=0.5) for c in cells]
    out = A.write_cells(path, again, ['siglip2'])
    on_disk = json.loads(path.read_text())
    assert on_disk == out and len(on_disk) == 8
    assert {c['encoder'] for c in on_disk} == {'siglip2', 'clip_l14'}
    assert all(c['candidate_value'] == 0.5 for c in on_disk if c['encoder'] == 'siglip2')
    with pytest.raises(AssertionError):
        A.write_cells(path, cells, ['clip_l14'])                    # cells of another encoder than the one named


# ------------------------------------------------------------------------------------- stage 1 file
def test_stage1_record_roundtrip_and_freeze(tmp_path, val):
    photo = A.photo_stage1(val['embed'], val['stored'], val['labels'], val['dist'], val['keys'],
                           r12_ranges=ranges_with_map_r(0.001), width=D, workers=1)
    frozen = dict(pick='center+k10', source='r12 pick', r=None, scores=None, view='clip_l14')
    rec = A.stage1_record('siglip2', photo, frozen)
    path = A.ranges_path('siglip2', tmp_path)
    A.write_record(path, rec)
    back = A.read_record(path, 'siglip2')
    assert back['ranges']['map'] == photo['ranges']['map']['r'] and back['photo_readout']['pick'] == photo['readout']['pick']
    A.write_record(path, rec)                                       # the same content is accepted
    changed = dict(rec, frozen=dict(frozen, pick='none+k0'))
    with pytest.raises(AssertionError, match='frozen once written'):
        A.write_record(path, changed)
    A.write_record(path, changed, overwrite=True)
    with pytest.raises(AssertionError):
        A.read_record(path, 'siglip1')                              # another encoder's file
    doc = json.loads(path.read_text())
    doc['ranges']['map'] = doc['ranges']['map'] * 2                 # a hand-edited range
    path.write_text(json.dumps(doc))
    with pytest.raises(AssertionError):
        A.read_record(path, 'siglip2')


def test_score_needs_the_stage1_file_before_any_test_photo(tmp_path, monkeypatch):
    def touched(*a, **k):
        raise RuntimeError('test data were read before the stage 1 file')
    for obj, name in ((pf_stage3, 'load_comparators'), (ac_data, 'labels'), (ac_data, 'keys'), (A, 'frozen_comparator')):
        monkeypatch.setattr(obj, name, touched)
    with pytest.raises(FileNotFoundError):
        A.score('siglip2', root=tmp_path)


# ------------------------------------------------------------------------------------- review fixes (ticket 07)
def write_r12_files(tmp_path, monkeypatch, r12_r, picks=R12_PICKS):
    """r12's ranges and picks files replaced by small tmp copies (the loaders' module constants)."""
    rp, pp = tmp_path / 'r12_ranges.json', tmp_path / 'r12_picks.json'
    rp.write_text(json.dumps(ranges_with_map_r(r12_r)))
    pp.write_text(json.dumps({'affection': picks}))
    monkeypatch.setattr(A, 'R12_RANGES', rp)
    monkeypatch.setattr(A, 'R12_PICKS', pp)


@pytest.fixture
def stage1_world(tmp_path, val, monkeypatch):
    """Five-seed val exports of `photo_siglip2` on disk and patched loaders; the test split raises on any read."""
    for s in range(5):
        write_export(tmp_path, 'photo_siglip2', 'val', s, val['keys'], val['stored'][s])

    def keys(split):
        assert split == 'val', f'stage 1 read the {split} split'
        return val['keys']
    monkeypatch.setattr(ac_data, 'keys', keys)
    monkeypatch.setattr(ac_data, 'labels', lambda split: val['labels'] if split == 'val' else pytest.fail(f'labels({split})'))
    monkeypatch.setattr(ac_data, 'emotion_dist', lambda split: val['dist'] if split == 'val' else pytest.fail(split))
    monkeypatch.setattr(A, 'frozen_embed', lambda enc, ks: val['embed'][0])
    write_r12_files(tmp_path, monkeypatch, R12_BIG)
    orig = A.photo_stage1                                           # the synthetic rows are D wide, not the 2048 of a photo model
    monkeypatch.setattr(A, 'photo_stage1', lambda *a, **k: orig(*a, width=D, **k))
    return tmp_path


def test_stage1_frozen_pick_uses_r12s_range_not_the_photo_models(stage1_world):
    rec = A.stage1('siglip2', root=stage1_world, workers=1)
    own = rec['ranges']['map']
    assert own < 0.1 and rec['photo_readout']['pick'] != 'none+k0'
    assert rec['frozen']['r'] == pytest.approx(R12_BIG) and rec['frozen']['r'] != own     # r12's r_ret,mAP, D19
    assert rec['frozen']['pick'] == 'none+k0'                      # with r12's huge range nothing beats R0
    assert A.read_record(A.ranges_path('siglip2', stage1_world), 'siglip2')['frozen']['r'] == pytest.approx(R12_BIG)


def test_stage1_does_not_load_the_test_split(stage1_world):
    A.stage1('siglip2', root=stage1_world, workers=1)               # ac_data.keys('test') / labels('test') would fail here
    pa = A.PhotoArm('photo_siglip2', stage1_world)
    assert set(pa.keys) == {'val'}


def test_read_record_refuses_a_frozen_range_that_is_not_r12s(stage1_world):
    path = A.ranges_path('siglip2', stage1_world)
    A.stage1('siglip2', root=stage1_world, workers=1)
    doc = json.loads(path.read_text())
    for r in (doc['ranges']['map'], 0.5):                          # the photo model's own range; any other number
        bad = dict(doc, frozen=dict(doc['frozen'], r=r))
        path.write_text(json.dumps(bad))
        with pytest.raises(AssertionError, match='frozen'):
            A.read_record(path, 'siglip2')
    path.write_text(json.dumps(dict(doc, frozen=dict(doc['frozen'], r=None, scores=None))))
    assert A.read_record(path, 'siglip2')['frozen']['r'] is None   # a kept r12 pick has no range


def frozen_world(val, monkeypatch):
    """`frozen_embed` patched: the val key list gets one set of rows, its reversed copy (the "test" keys) another."""
    labels = val['labels']
    rows = {'val': make_rows(labels, 500, noise=1.6), 'test': make_rows(labels, 501, noise=1.6)}
    which = {tuple(val['keys']): 'val', tuple(val['keys'][::-1]): 'test'}
    monkeypatch.setattr(A, 'frozen_embed', lambda enc, ks: rows[which[tuple(ks)]])
    return rows, val['keys'], val['keys'][::-1]


def test_frozen_comparator_is_unseeded_and_equals_arm_retrieval(tmp_path, val, monkeypatch):
    rows, vk, tk = frozen_world(val, monkeypatch)
    labels = val['labels']
    comp = A.frozen_comparator('siglip2', 'center+k3', vk, tk, labels, tmp_path, workers=1)
    assert comp['arm'] == 'frozen_siglip2' and comp['readout'] == 'center+k3'
    assert comp['labels'].shape == (3, len(labels)) and comp['labels'].dtype == np.int32    # 3 k-means states, no seeds
    ref = ac_stats.arm_retrieval(lambda s: rows['val'], lambda s: rows['test'], 'center+k3', labels, seeds=None, workers=1)
    assert comp['point'] == ref['point'] and np.array_equal(comp['ap'], ref['ap']) and np.array_equal(comp['rows'], ref['rows'])
    assert np.array_equal(comp['labels'], ac_cluster.cached_partitions(rows['test'], 'x', tmp_path / 'other'))


def test_clip_l14_bar_is_checked_against_r12s_stored_row(tmp_path, val, monkeypatch):
    rows, vk, tk = frozen_world(val, monkeypatch)
    labels = val['labels']
    ref = A.frozen_comparator('siglip2', 'center+k3', vk, tk, labels, tmp_path / 'a', workers=1)
    ami = ac_cluster.ami(ref['labels'], labels)

    def verdict(**kw):
        row = dict(ami=ami, readout='center+k3', map_picked=dict(map=ref['point'], per_seed=None))
        row.update(kw)
        path = tmp_path / 'verdict.json'
        path.write_text(json.dumps(dict(rows=dict(clip=row))))
        monkeypatch.setattr(A, 'R12_VERDICT', path)
    verdict()
    got = A.frozen_comparator('clip_l14', 'center+k3', vk, tk, labels, tmp_path / 'b', workers=1)
    assert got['point'] == ref['point'] and got['arm'] == 'frozen_clip_l14'
    verdict(ami=ami + 1e-6)                                         # r12 row differs in the AMI
    with pytest.raises(AssertionError, match='r12'):
        A.frozen_comparator('clip_l14', 'center+k3', vk, tk, labels, tmp_path / 'b', workers=1)
    row = json.loads((tmp_path / 'verdict.json').read_text())
    row['rows']['clip'].update(ami=ami, map_picked=dict(map=ref['point'] + 1e-6, per_seed=None))     # ... in the point
    (tmp_path / 'verdict.json').write_text(json.dumps(row))
    with pytest.raises(AssertionError, match='r12'):
        A.frozen_comparator('clip_l14', 'center+k3', vk, tk, labels, tmp_path / 'b', workers=1)
    # the other encoders have no r12 row to compare with
    A.frozen_comparator('siglip1', 'center+k3', vk, tk, labels, tmp_path / 'c', workers=1)


@pytest.mark.parametrize('argv', [['score', '--enc', 'qwen'], ['ranges', '--enc', 'qwen'], ['standin', 'prepare'],
                                  ['standin', 'check']])
def test_main_refuses_qwen_and_standin_on_the_production_folder(monkeypatch, capsys, argv):
    import bs_paths
    monkeypatch.setattr(bs_paths, 'OUT', bs_paths.DEFAULT_OUT)
    for name in ('stage1', 'score', 'standin_prepare', 'standin_check'):
        monkeypatch.setattr(A, name, lambda *a, **k: pytest.fail('ran on the production folder'))
    with pytest.raises(SystemExit) as e:
        A.main(argv)
    assert e.value.code != 0 and 'BS_OUTPUTS' in capsys.readouterr().err


def test_main_lets_a_real_encoder_and_a_redirected_folder_through(monkeypatch, tmp_path):
    import bs_paths
    calls = []
    monkeypatch.setattr(A, 'standin_prepare', lambda *a, **k: calls.append('prepare') or [])
    monkeypatch.setattr(bs_paths, 'OUT', tmp_path)
    assert A.main(['standin', 'prepare']) == 0 and calls == ['prepare']
    monkeypatch.setattr(bs_paths, 'OUT', bs_paths.DEFAULT_OUT)
    monkeypatch.setattr(A, 'stage1', lambda *a, **k: dict(ranges=dict(clu=1, map=1), photo_readout=dict(pick='p'),
                                                          frozen=dict(pick='f')))
    assert A.main(['ranges', '--enc', 'siglip2']) == 0
