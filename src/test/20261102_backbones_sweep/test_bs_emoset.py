"""Tests of bs_emoset (r14 ticket 02): the hashed order, labels, keys with and without the cut, the near-copy cut.

Order thresholds: a statistic of an order passes when it is at most the 99th percentile of the same statistic over 200
seeded random permutations of the same labels (one-sided: the question is grouping, so only too-large values count).
Statistics: (1) the longest run of one label in the order; (2) the label imbalance of the cross-fit halves by position
parity: the L1 distance between the class proportions of the even and the odd positions.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import bs_emoset as E
import bs_paths as BP
import bs_stores

REAL_VAL = Path(BP.EMOSET_ROOT) / 'val.json'
CLASSES = BP.EMOSET_CLASSES


# ------------------------------------------------------------------------------------------------ helpers
def longest_run(lab):
    best = run = 1
    for a, b in zip(lab, lab[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best


def half_gap(lab):
    lab = np.asarray(lab)
    cls = sorted(set(lab.tolist()))
    even, odd = lab[0::2], lab[1::2]
    return float(sum(abs((even == c).mean() - (odd == c).mean()) for c in cls))


def thresholds(lab, n_perm=200, q=99, seed=0):
    """99th percentile of both statistics over seeded random permutations of the labels."""
    rng = np.random.default_rng(seed)
    runs, gaps = [], []
    for _ in range(n_perm):
        p = rng.permutation(len(lab))
        shuffled = [lab[i] for i in p]
        runs.append(longest_run(shuffled))
        gaps.append(half_gap(shuffled))
    return np.percentile(runs, q), np.percentile(gaps, q)


def groups_classes(ordered_keys, truth, thr):
    """True when the order shows class grouping beyond the random range."""
    lab = [truth[k] for k in ordered_keys]
    return longest_run(lab) > thr[0] or half_gap(lab) > thr[1]


def synthetic(n_per=100):
    truth = {f'image/{c}/{c}_{n:05d}.jpg': c for c in CLASSES for n in range(n_per)}
    return truth


def write_split_root(root, entries_by_split):
    root.mkdir(parents=True, exist_ok=True)
    for split, entries in entries_by_split.items():
        (root / f'{split}.json').write_text(json.dumps(entries))


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    """A synthetic split folder; the class in the first field is deliberately NOT always the class in the path."""
    val = [[c, f'image/{c}/{c}_{n:05d}.jpg', f'annotation/{c}/{c}_{n:05d}.json'] for c in CLASSES[:4] for n in range(6)]
    test = [[c, f'image/{c}/{c}_{n:05d}.jpg', f'annotation/{c}/{c}_{n:05d}.json'] for c in CLASSES[4:] for n in range(5)]
    val[0][0] = 'fear'                         # first field disagrees with the path of image/amusement/amusement_00000
    root = tmp_path / 'emoset'
    write_split_root(root, {'val': val, 'test': test})
    monkeypatch.setattr(E, 'ROOT', root)
    monkeypatch.setattr(E, 'COUNTS', {'val': len(val), 'test': len(test)})
    return val, test


# ------------------------------------------------------------------------------------------------ order
def test_order_is_sha256_hex_digest_of_key_string():
    import hashlib
    ks = ['image/awe/awe_1.jpg', 'image/anger/anger_2.jpg', 'image/fear/fear_3.jpg', 'image/awe/awe_0.jpg']
    expect = sorted(ks, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    assert E.order(ks) == expect and E.order(reversed(ks)) == expect and E.order(ks) != sorted(ks)
    with pytest.raises(AssertionError):
        E.order(ks + ks[:1])


def test_sha_order_does_not_group_classes_but_sorted_order_does_synthetic():
    truth = synthetic()
    thr = thresholds(list(truth.values()))
    assert not groups_classes(E.order(truth), truth, thr)
    assert groups_classes(sorted(truth), truth, thr)           # plain sorted keys fail the same test
    assert longest_run([truth[k] for k in sorted(truth)]) == 100 and thr[0] < 20 and thr[1] < 0.35


@pytest.mark.skipif(not REAL_VAL.exists(), reason='EmoSet val split file is not unpacked on this machine')
def test_sha_order_does_not_group_classes_on_the_real_val_split():
    entries = json.loads(REAL_VAL.read_text())
    assert len(entries) == 5905
    truth = {e[1]: e[0] for e in entries}
    thr = thresholds(list(truth.values()))
    sha = E.order(truth)
    lab = [truth[k] for k in sha]
    assert not groups_classes(sha, truth, thr), (longest_run(lab), half_gap(lab), thr)
    assert groups_classes(sorted(truth), truth, thr)           # r12's sorted keys would group the 8 classes


# ------------------------------------------------------------------------------------------------ labels and keys
def test_labels_read_the_first_field_not_the_path(fake_root):
    val, test = fake_root
    got = E.labels(['image/amusement/amusement_00000.jpg', 'image/sadness/sadness_00004.jpg',
                    'image/anger/anger_00001.jpg'])
    assert got == ['fear', 'sadness', 'anger']                  # the first one: field says fear, path says amusement
    with pytest.raises(AssertionError):
        E.labels(['image/awe/awe_99999.jpg'])


def test_keys_uncut_is_the_split_file_keys_in_sha_order(fake_root):
    val, test = fake_root
    assert E.keys('val', cut=False) == E.order([e[1] for e in val])
    assert E.keys('test', cut=False) == E.order([e[1] for e in test])
    assert sorted(E.keys('val', cut=False)) == sorted(e[1] for e in val)


def write_record(path, val, test, removed_val=(), removed_test=(), cut=None):
    rec = dict(cut=BP.NEAR_COPY_CUT if cut is None else cut,
               splits={'val': dict(n=len(val), n_removed=len(removed_val), removed_keys=sorted(removed_val)),
                       'test': dict(n=len(test), n_removed=len(removed_test), removed_keys=sorted(removed_test))})
    path.write_text(json.dumps(rec))
    return path


def test_keys_cut_needs_the_record_and_drops_exactly_its_keys(fake_root, tmp_path, monkeypatch):
    val, test = fake_root
    monkeypatch.setattr(E, 'record_file', lambda: tmp_path / 'absent.json')
    with pytest.raises(AssertionError):
        E.keys('val')
    gone = [val[1][1], val[7][1]]
    rec = write_record(tmp_path / 'rec.json', val, test, removed_val=gone)
    got = E.keys('val', record=rec)
    assert got == E.order([e[1] for e in val if e[1] not in gone]) and len(got) == len(val) - 2
    assert E.keys('test', record=rec) == E.keys('test', cut=False)
    monkeypatch.setattr(E, 'record_file', lambda: rec)
    assert E.keys('val') == got                                  # the default record path
    bad = write_record(tmp_path / 'bad.json', val, test, removed_val=['image/awe/awe_99999.jpg'])
    with pytest.raises(AssertionError):
        E.keys('val', record=bad)                                # a removed key that is not in the split file
    wrong_cut = write_record(tmp_path / 'cut.json', val, test, cut=0.9)
    with pytest.raises(AssertionError):
        E.keys('val', record=wrong_cut)


# ------------------------------------------------------------------------------------------------ near-copy
def test_cut_is_inclusive_at_exactly_the_cut_and_uses_float32_cosine():
    # unit rows with exactly representable products: cosine of q_i with r = [1,0,0,0] is exactly q_i[0]
    r = np.array([[1, 0, 0, 0]], dtype=np.float32)
    q = np.array([[0.5, 0.5, 0.5, 0.5], [0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.5, 0.5]], dtype=np.float32)
    best, arg = E.nearest(q, r)
    assert best.tolist() == [0.5, 0.25, 0.5] and arg.tolist() == [0, 0, 0]
    assert E.removed_mask(best, cut=0.5).tolist() == [True, False, True]       # exactly at the cut leaves
    # the default cut: the float32 value nearest 0.85 is removed, the next float32 below it is kept
    c = np.float32(0.85)
    b = np.array([c, np.nextafter(c, np.float32(0))], dtype=np.float32)
    assert E.removed_mask(b).tolist() == [True, False]
    assert BP.NEAR_COPY_CUT == 0.85


def test_nearest_matches_r13s_blockwise_code_and_ties_go_to_the_earlier_reference():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(40, 16)).astype(np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    q = rng.normal(size=(25, 16)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    best, arg = E.nearest(q, ref, block=7)
    assert np.allclose(best, (q @ ref.T).max(axis=1), atol=1e-6) and (arg == (q @ ref.T).argmax(axis=1)).all()
    tie = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)   # exact ties
    tb, ta = E.nearest(np.array([[0, 1, 0, 0]], dtype=np.float32), tie)
    assert ta.tolist() == [0] and tb.tolist() == [1.0]            # rows 0 and 2 tie: the earlier one
    sys.path.insert(0, str(BP.R13_ROUND))
    try:
        import pf_nearcopy
    except Exception as exc:                                     # pragma: no cover - r13 code unavailable
        pytest.skip(f'r13 code not importable: {exc}')
    b13, a13 = pf_nearcopy._nearest(q, ref, 7)
    assert (b13 == best).all() and (a13 == arg).all()


def test_histogram_bins_clip_float32_overshoot_and_count_below():
    best = np.array([0.79, 0.80, 0.8499, 0.85, 0.999, 1.0, np.float32(1.0000001)], dtype=np.float32)
    h = E.histogram(best)
    assert len(h['edges']) == 21 and len(h['counts']) == 20 and h['n_below'] == 1
    assert sum(h['counts']) == 6 and h['counts'][0] == 1 and h['counts'][4] == 1 and h['counts'][5] == 1
    assert h['counts'][19] == 3                                  # 0.999, 1.0 and the overshoot (closed last bin)


def make_store(folder, keys, vecs):
    """An ac-format store (keys.json + img.npy float16) of the default Qwen width, rows in the given key order."""
    folder.mkdir(parents=True)
    (folder / 'keys.json').write_text(json.dumps(keys))
    np.save(folder / 'img.npy', vecs.astype(np.float16))
    return folder


def test_near_copy_record_joins_by_key_when_the_stores_list_keys_in_other_orders(tmp_path):
    rng = np.random.default_rng(3)
    d = bs_stores.WIDTH['qwen_default']
    ref_keys = [f'coco_affective/r{i}.jpg' for i in range(30)]
    ref = rng.normal(size=(30, d)).astype(np.float32)
    val_keys = [f'image/awe/awe_{i:05d}.jpg' for i in (4, 1, 5, 0, 3, 2)]      # split-file order is not sorted
    test_keys = [f'image/fear/fear_{i:05d}.jpg' for i in (3, 0, 4, 2, 1)]
    qvecs = rng.normal(size=(11, d)).astype(np.float32)
    planted = {0: 4, 3: 17, 8: 29}                               # query row -> reference row it nearly copies
    for qi, ri in planted.items():
        qvecs[qi] = ref[ri] + 0.01 * rng.normal(size=d).astype(np.float32)
    qkeys = val_keys + test_keys
    expect_removed = sorted(qkeys[i] for i in planted)

    def record(ref_order, q_order):
        ra = make_store(tmp_path / f'ref{len(list(tmp_path.iterdir()))}', [ref_keys[i] for i in ref_order],
                        ref[ref_order])
        qa = make_store(tmp_path / f'q{len(list(tmp_path.iterdir()))}', [qkeys[i] for i in q_order], qvecs[q_order])
        return E.near_copy_record(
            ref_keys, lambda ks: bs_stores.load('affection', 'qwen_default', ks, folder_list=[ra]),
            {'val': val_keys, 'test': test_keys},
            lambda ks: bs_stores.load('emoset', 'qwen_default', ks, folder_list=[qa]), block=4)

    straight = record(list(range(30)), list(range(11)))
    shuffled = record(rng.permutation(30), rng.permutation(11))
    assert straight == shuffled
    got = straight['splits']['val']['removed_keys'] + straight['splits']['test']['removed_keys']
    assert sorted(got) == expect_removed
    assert straight['splits']['val']['n'] == 6 and straight['splits']['test']['n_removed'] == 1
    assert straight['splits']['val']['n_removed'] + straight['splits']['val']['n_kept'] == 6
    assert straight['reference']['n'] == 30 and straight['cut'] == 0.85
    assert sum(straight['splits']['val']['hist']['counts']) == 2     # two planted copies (cos ~ 1); random rows < 0.80


def test_near_copy_record_asserts_overlapping_keys_and_missing_rows(tmp_path):
    x = np.eye(4, dtype=np.float32)
    with pytest.raises(AssertionError):                           # a query key that is also a reference key
        E.near_copy_record(['a', 'b'], lambda ks: x[:2], {'val': ['b']}, lambda ks: x[:1])
    with pytest.raises(AssertionError):                           # loader returns fewer rows than keys
        E.near_copy_record(['a', 'b'], lambda ks: x[:1], {'val': ['c']}, lambda ks: x[:1])
