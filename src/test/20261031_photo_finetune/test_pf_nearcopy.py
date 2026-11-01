"""Near-copy rule (ticket 02, D12). Synthetic arrays for `compute`; the synthetic store of test_pf_data for `run`."""
import json

import numpy as np
import pytest

import pf_paths  # first: puts r12's and every older round on sys.path
import pf_data
import pf_nearcopy
from test_pf_data import World


def unit(rows):
    x = np.asarray(rows, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def compute(train, held, *, held_split=None, train_keys=None, held_keys=None, **kw):
    """train / held: row lists; keys and splits default to distinct names, held all 'val'."""
    train, held = unit(train), unit(held)
    train_keys = train_keys or [f's/t{i}.jpg' for i in range(len(train))]
    held_keys = held_keys or [f's/h{i}.jpg' for i in range(len(held))]
    held_split = held_split or ['val'] * len(held)
    return pf_nearcopy.compute(train_keys, train, held_keys, held_split, held, **kw)


def test_the_cut_is_inclusive():
    # cosine of (1,0,0,0) and (.5,.5,.5,.5) is exactly 0.5 in float32
    rec, _ = compute([[1, 0, 0, 0], [0, 1, 0, 0]], [[1, 1, 1, 1]], cut=0.5)
    assert rec['removed_keys'] == ['s/t0.jpg', 's/t1.jpg'] and rec['n_removed'] == 2
    rec, _ = compute([[1, 0, 0, 0]], [[1, 1, 1, 1]], cut=0.5000001)
    assert rec['removed_keys'] == [] and rec['n_removed'] == 0 and rec['cut'] == 0.5000001


def test_the_default_cut_is_the_spec_cut():
    assert pf_nearcopy.compute.__kwdefaults__['cut'] == pf_paths.NEAR_COPY_CUT == 0.85


def test_nearest_split_and_key_are_those_of_the_best_held_photo():
    train = [[1, 0, 0, 0], [0, 1, 0, 0]]
    held = [[0.6, 0.8, 0, 0], [1, 0.1, 0, 0], [0, 1, 0.2, 0]]
    _, detail = compute(train, held, held_split=['val', 'test', 'test'], cut=0.85)
    assert detail['nearest_key'] == ['s/h1.jpg', 's/h2.jpg']
    assert detail['nearest_split'] == ['test', 'test']
    assert np.allclose(detail['cosine'], [1 / np.sqrt(1.01), 1 / np.sqrt(1.04)], atol=1e-6)
    assert detail['cosine'].dtype == np.float32


def test_a_tie_between_held_photos_goes_to_the_earlier_held_row():
    _, detail = compute([[1, 0, 0, 0]], [[1, 0, 0, 0], [1, 0, 0, 0]], held_split=['val', 'test'], cut=0.85)
    assert detail['nearest_key'] == ['s/h0.jpg'] and detail['nearest_split'] == ['val']


def random_world(n_train=70, n_held=23, dim=6, seed=3):
    """Clustered data so that many train photos have a held photo at a high cosine."""
    rng = np.random.default_rng(seed)
    held = rng.normal(size=(n_held, dim))
    train = held[rng.integers(0, n_held, n_train)] + rng.normal(scale=0.7, size=(n_train, dim))
    split = ['val' if i % 3 else 'test' for i in range(n_held)]
    return train, held, split


@pytest.mark.parametrize('block', [1, 7, 70, 500])
def test_block_size_does_not_change_the_record(block):
    train, held, split = random_world()
    want, want_detail = compute(train, held, held_split=split, cut=0.8, block=10 ** 6)
    got, got_detail = compute(train, held, held_split=split, cut=0.8, block=block)
    assert 0 < want['n_removed'] < len(train)
    assert got == want
    assert got_detail['nearest_key'] == want_detail['nearest_key']
    # BLAS may round a float32 dot product differently for another block shape (1 ulp); only a cosine within 1e-7
    # of the cut could change a decision, and a run uses one fixed block.
    np.testing.assert_allclose(got_detail['cosine'], want_detail['cosine'], rtol=0, atol=1e-6)


def test_a_hand_loop_gives_the_same_cut_and_nearest():
    train, held, split = random_world(seed=11)
    rec, detail = compute(train, held, held_split=split, cut=0.8, block=9)
    t, h = unit(train), unit(held)
    for i in range(len(t)):
        cos = [float(np.dot(t[i].astype(np.float64), h[j].astype(np.float64))) for j in range(len(h))]
        j = int(np.argmax(cos))
        assert detail['nearest_key'][i] == f's/h{j}.jpg' and detail['nearest_split'][i] == split[j]
        assert (f's/t{i}.jpg' in rec['removed_keys']) == (cos[j] >= 0.8)


def test_a_held_key_never_appears_in_removed_keys_and_overlap_is_refused():
    train, held, split = random_world()
    rec, _ = compute(train, held, held_split=split, cut=0.0)          # everything is removed
    assert rec['n_removed'] == len(train) and rec['removed_keys'] == sorted(rec['removed_keys'])
    assert not set(rec['removed_keys']) & {f's/h{i}.jpg' for i in range(len(held))}
    with pytest.raises(AssertionError):
        compute(train, held, train_keys=['s/h0.jpg'] + [f's/t{i}.jpg' for i in range(1, len(train))])


def test_the_record_has_exactly_the_ticket_fields_and_n_train():
    rec, _ = compute([[1, 0, 0, 0], [0, 0, 1, 0]], [[1, 0, 0, 0]])
    assert set(rec) == {'cut', 'n_train', 'n_removed', 'removed_keys', 'by_source', 'hist', 'within_train_copies'}
    assert rec['n_train'] == 2 and rec['n_removed'] == 1


def test_by_source_counts_removed_photos_per_source_and_nearest_split():
    train = [[1, 0, 0, 0], [1, 0.01, 0, 0], [0, 1, 0, 0], [0, 1, 0.01, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    tkeys = ['a/1', 'a/2', 'b/1', 'b/2', 'b/3', 'c/1']
    held = [[1, 0, 0, 0], [0, 1, 0, 0]]
    rec, _ = compute(train, held, train_keys=tkeys, held_keys=['d/1', 'd/2'], held_split=['val', 'test'], cut=0.85)
    assert rec['by_source'] == {'a': {'val': 2, 'test': 0}, 'b': {'val': 0, 'test': 2}, 'c': {'val': 0, 'test': 0}}
    assert sum(v['val'] + v['test'] for v in rec['by_source'].values()) == rec['n_removed'] == 4


def test_histogram_has_edges_080_to_100_and_counts_every_photo_with_nearest_at_least_080():
    # nearest cosines 0.79 (outside), 0.805 (first bin), 0.855 (sixth), 1.0 (last bin, closed edge)
    held = [[1, 0]]
    train = [[c, np.sqrt(1 - c ** 2)] for c in (0.79, 0.805, 0.855)] + [[1, 0]]
    rec, detail = compute(train, held)
    hist = rec['hist']
    assert hist['edges'] == [round(0.8 + 0.01 * i, 2) for i in range(21)]
    assert len(hist['counts']) == 20 and sum(hist['counts']) == 3
    assert hist['counts'][0] == 1 and hist['counts'][5] == 1 and hist['counts'][19] == 1
    assert sum(hist['counts']) == int((detail['cosine'] >= 0.8).sum())


def test_histogram_counts_a_float_rounding_cosine_above_one_in_the_last_bin():
    x = np.float32(1.0000001)
    rec, _ = pf_nearcopy.compute(['a/1'], np.array([[x, 0]], np.float32), ['b/1'], ['val'], np.array([[x, 0]], np.float32))
    assert rec['hist']['counts'][19] == 1 and sum(rec['hist']['counts']) == 1


def test_within_train_copies_exclude_self_pairs_and_stay_in():
    train = [[1, 0, 0, 0], [1, 0.05, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0]]
    rec, _ = compute(train, [[0, 0, 0, 1]], cut=0.85)
    assert rec['within_train_copies'] == 4 and rec['n_removed'] == 0          # t0..t1 and t3..t4, nothing alone
    lone, _ = compute([[1, 0, 0, 0], [0, 1, 0, 0]], [[0, 0, 0, 1]])
    assert lone['within_train_copies'] == 0                                  # a photo is never its own copy


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    monkeypatch.setattr(pf_paths, 'OUTPUTS', tmp_path / 'out')
    return w


def make_copy(world, train_row, held_row):
    """Store rows are float16; make the train photo's image the held photo's image."""
    world.img[train_row] = world.img[held_row]
    np.save(world.tmp / 'store' / 'img.npy', world.img)


def test_run_writes_the_record_and_fit_keys_drops_the_copy(world):
    make_copy(world, 0, 5)                                   # a/1.jpg (train) is a copy of d/t1.jpg (test)
    record = pf_nearcopy.run()
    assert record['removed_keys'] == ['a/1.jpg'] and record['n_train'] == 3 and record['n_removed'] == 1
    assert record['by_source'] == {'a': {'val': 0, 'test': 1}, 'b': {'val': 0, 'test': 0}}
    assert json.loads(pf_paths.NEARCOPY_FILE.read_text()) == record
    assert pf_data.fit_keys() == ['a/2.jpg', 'b/1.jpg']
    evidence = json.loads((pf_paths.OUTPUTS / 'stage0a' / 'nearcopy.json').read_text())
    assert evidence.pop('seconds') >= 0 and evidence == record


def test_run_joins_by_key_not_by_file_name(world):
    # b/1.jpg shares its file name with a/1.jpg; only the val photo c/v1.jpg copies the image of b/1.jpg
    make_copy(world, 1, 3)
    record = pf_nearcopy.run()
    assert record['removed_keys'] == ['b/1.jpg'] and record['by_source']['b'] == {'val': 1, 'test': 0}


def test_run_under_smoke_covers_the_smoke_train_keys_only(world, monkeypatch):
    make_copy(world, 1, 5)                                 # b/1.jpg is a copy, but the smoke set is a/1, a/2
    monkeypatch.setattr(pf_paths, 'SMOKE', 2)
    record = pf_nearcopy.run()
    assert record['n_train'] == 2 and record['removed_keys'] == []


@pytest.mark.slow
@pytest.mark.real_data
def test_real_store_near_copy_count(tmp_path, monkeypatch, capsys):
    """Controller's run on the real default store (images only; the records go to a scratch folder)."""
    import time
    monkeypatch.setattr(pf_paths, 'NEARCOPY_FILE', tmp_path / 'stage0a_nearcopy.json')
    monkeypatch.setattr(pf_paths, 'OUTPUTS', tmp_path / 'out')
    started = time.time()
    record = pf_nearcopy.run()
    seconds = time.time() - started
    with capsys.disabled():
        print(f"\nnear-copy: n_train {record['n_train']}, n_removed {record['n_removed']} (rule review: about 2,328), "
              f"within_train_copies {record['within_train_copies']}, by_source {record['by_source']}, "
              f"hist {record['hist']['counts']}, {seconds:.0f} s")
    assert record['n_train'] == pf_paths.N['train'] == len(pf_data.keys('train'))
    assert 0 < record['n_removed'] < record['n_train'] // 10, 'far from the rule review count: investigate'
    assert set(record['removed_keys']) <= set(pf_data.keys('train'))


@pytest.mark.parametrize('block', [1, 4, 100])
def test_within_train_count_does_not_depend_on_the_block(block):
    train, held, split = random_world(seed=5)
    want, _ = compute(train, held, held_split=split, cut=0.8, block=10 ** 6)
    got, _ = compute(train, held, held_split=split, cut=0.8, block=block)
    assert 0 < got['within_train_copies'] < len(train) and got['within_train_copies'] == want['within_train_copies']
