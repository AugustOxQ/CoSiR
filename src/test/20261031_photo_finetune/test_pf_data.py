"""Photo data provider (ticket 01): keys, explanation rows, I, D, E, T. Synthetic stores in a tmp folder; the
`real_data` tests read only the real index JSON (no features, no labels)."""
import csv
import json

import numpy as np
import pytest

import pf_paths  # first: puts r12's and every older round on sys.path
import ac_data
import ac_paths
import pf_data
import rs_vh
import rt_data

ART_PROMPT = rt_data.ct2_data.itp_data.PROMPTS['emotion_focus']
QWEN = ('Qwen/Qwen3-VL-Embedding-2B', rt_data.ct2_data.QWEN_REVISION)
GATE = {'ok': True, 'n_nonfinite': 0, 'n_bad_norm': 0, 'max_norm_dev': 0.0002}

# (key, split of the photo, [row splits]); texts are shuffled below, so index order is not grouped by photo.
PHOTOS = [
    ('a/1.jpg', 'train', ['train', 'train', 'train', 'ignored']),
    ('b/1.jpg', 'train', ['train', 'train']),            # same file name under another source
    ('a/2.jpg', 'train', ['ignored', 'train']),
    ('c/v1.jpg', 'val', ['val', 'val', 'ignored']),
    ('c/v2.jpg', 'val', ['val']),
    ('d/t1.jpg', 'test', ['test', 'test']),
]


def unit16(rng, n):
    x = rng.normal(size=(n, 2048)).astype(np.float32)
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)


def write_json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))


def write_view(path, keys, x, instruction=ART_PROMPT, smoke=False, gate=GATE):
    """A train-image or teacher view: img.npy + keys.json + meta.json."""
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'img.npy', x)
    write_json(path / 'keys.json', keys)
    write_json(path / 'meta.json', dict(
        view=path.name, instruction=instruction, model_id=QWEN[0], model_revision=QWEN[1], dtype='float16',
        shape=list(x.shape), keys_sha256=rs_vh.names_sha(keys), img_sha256=rs_vh.array_sha(x), gate=gate,
        smoke=smoke, batch_size=32, seconds=1.0))


def write_txt_view(path, csv_rows, keys, x, instruction=ART_PROMPT, smoke=False, gate=GATE):
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'txt.npy', x)
    write_json(path / 'rows.json', {'csv_row': [int(c) for c in csv_rows], 'key': list(keys)})
    write_json(path / 'meta.json', dict(
        view=path.name, instruction=instruction, model_id=QWEN[0], model_revision=QWEN[1], dtype='float16',
        shape=list(x.shape), txt_sha256=rs_vh.array_sha(x),
        rows_sha256=rs_vh.names_sha([f'{int(c)}\t{k}' for c, k in zip(csv_rows, keys)]), gate=gate,
        smoke=smoke, batch_size=128, seconds=1.0))


class World:
    """The synthetic round: store, annotations, paths; the new views are written by the tests that need them."""

    def __init__(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(5)
        self.rng, self.tmp = rng, tmp_path
        images, texts, ann, order = [], [], [], []
        for i, (key, split, rows) in enumerate(PHOTOS):
            source, name = key.split('/')
            images.append(dict(source=source, image_name=name, splits=sorted(set(rows) | {split}), n_texts=len(rows)))
            for r in rows:
                order.append((i, r))
                ann.append((source, name, r, 'awe'))
        rng.shuffle(order)
        self.order = [(int(i), r) for i, r in order]
        self.csv_rows = (np.arange(len(order)) * 3 + 100).tolist()
        for (i, r), c in zip(self.order, self.csv_rows):
            texts.append(dict(img=i, csv_row=c, emotion='awe', workerid='w', split=r))
        store = tmp_path / 'store'
        write_json(store / 'index.json', dict(dataset='affection_local', backbone='qwen3vl_emb_2b', smoke=False,
                                              dtype='float16', dim=2048, n_images=len(images), n_texts=len(texts),
                                              images=images, texts=texts))
        self.img = unit16(rng, len(images))
        self.txt = (rng.normal(size=(len(texts), 2048)) * 0.05).astype(np.float16)   # raw rows, not unit
        np.save(store / 'img.npy', self.img)
        np.save(store / 'txt.npy', self.txt)
        with (tmp_path / 'annotations.csv').open('w', newline='') as h:
            w = csv.writer(h)
            w.writerow(('dataset', 'image_name', 'split', 'emotion'))
            w.writerows(ann)
        for mod in (ac_paths, pf_paths):
            monkeypatch.setattr(mod, 'STORE', store)
            monkeypatch.setattr(mod, 'VIEWS', tmp_path / 'views')
        monkeypatch.setattr(ac_paths, 'ANNOTATIONS', tmp_path / 'annotations.csv')
        monkeypatch.setattr(ac_paths, 'N', {'val': 2, 'test': 1})
        monkeypatch.setattr(pf_paths, 'N', {'train': 3, 'val': 2, 'test': 1})
        self.teacher = tmp_path / 'views' / 'emotion_focus'
        self.train_view = tmp_path / 'views' / 'emotion_focus_train'
        self.txt_view = tmp_path / 'views' / 'emotion_focus_txt'
        monkeypatch.setattr(pf_paths, 'VAL_TEST_TEACHER_VIEW', self.teacher)
        monkeypatch.setattr(pf_paths, 'TRAIN_IMG_VIEW', self.train_view)
        monkeypatch.setattr(pf_paths, 'TXT_VIEW', self.txt_view)
        monkeypatch.setattr(pf_paths, 'NEARCOPY_FILE', tmp_path / 'nearcopy.json')
        monkeypatch.setattr(pf_paths, 'SMOKE', 0)

    def split_keys(self, split):
        return sorted(k for k, s, _ in PHOTOS if s == split)

    def rows(self, split):
        """Independent oracle for text_rows: (key, csv_row, store_row, row split) of the split's photos."""
        want = set(self.split_keys(split))
        return [(PHOTOS[i][0], c, p, r) for p, ((i, r), c) in enumerate(zip(self.order, self.csv_rows))
                if PHOTOS[i][0] in want]


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


def test_keys_are_sorted_per_split_and_same_file_names_stay_distinct(world):
    assert pf_data.keys('train') == ['a/1.jpg', 'a/2.jpg', 'b/1.jpg']
    assert pf_data.keys('val') == ['c/v1.jpg', 'c/v2.jpg'] == ac_data.keys('val')
    assert pf_data.keys('test') == ['d/t1.jpg']


def test_smoke_takes_the_first_n_sorted_train_keys_and_leaves_val_full(world, monkeypatch):
    monkeypatch.setattr(pf_paths, 'SMOKE', 2)
    assert pf_data.keys('train') == ['a/1.jpg', 'a/2.jpg']
    assert pf_data.keys('val') == ['c/v1.jpg', 'c/v2.jpg']


def test_a_photo_with_two_non_ignored_splits_is_refused(world):
    path = world.tmp / 'store' / 'index.json'
    index = json.loads(path.read_text())
    victim = next(t for t in index['texts'] if t['img'] == 0 and t['split'] == 'train')
    victim['split'] = 'val'
    write_json(path, index)
    with pytest.raises(AssertionError):
        pf_data.keys('train')


def test_text_rows_cover_every_row_of_the_splits_photos_in_index_order(world):
    got = pf_data.text_rows('train')
    want = world.rows('train')
    assert got['key'] == [k for k, _, _, _ in want]
    assert got['csv_row'].tolist() == [c for _, c, _, _ in want] and got['csv_row'].dtype == np.int64
    assert got['store_row'].tolist() == [p for _, _, p, _ in want] and got['store_row'].dtype == np.int64
    assert got['eligible'].tolist() == [r != 'ignored' for _, _, _, r in want] and got['eligible'].dtype == np.bool_
    assert len(got['key']) == 8 and int(got['eligible'].sum()) == 6
    val = pf_data.text_rows('val')
    assert len(val['key']) == 4 and int(val['eligible'].sum()) == 3
    with pytest.raises(AssertionError):
        pf_data.text_rows('test')


def test_text_rows_under_smoke_keep_only_the_smoke_photos(world, monkeypatch):
    monkeypatch.setattr(pf_paths, 'SMOKE', 1)
    got = pf_data.text_rows('train')
    assert set(got['key']) == {'a/1.jpg'} and len(got['key']) == 4


def hand_mean(world, key, source):
    """Independent oracle: the key's eligible rows summed in float64 in a python loop, / count, float32, L2."""
    total, n = np.zeros(2048, dtype=np.float64), 0
    for (i, r), pos in zip(world.order, range(len(world.order))):
        if PHOTOS[i][0] == key and r != 'ignored':
            total += source[pos].astype(np.float64)
            n += 1
    mean = (total / n).astype(np.float32)
    return mean / np.linalg.norm(mean)


def test_default_images_are_unit_float32_store_rows_joined_by_key(world):
    keys = ['b/1.jpg', 'a/1.jpg', 'd/t1.jpg']                 # unsorted, same file name under two sources, a test key
    x = pf_data.default_images(keys)
    assert x.dtype == np.float32 and x.shape == (3, 2048)
    for row, key in zip(x, keys):
        raw = world.img[[k for k, _, _ in PHOTOS].index(key)].astype(np.float32)
        np.testing.assert_allclose(row, raw / np.linalg.norm(raw), atol=1e-6)
    assert not np.allclose(x[0], x[1])
    with pytest.raises(AssertionError):
        pf_data.default_images(['a/1.jpg', 'nope/1.jpg'])


def test_photo_arrays_D_equals_a_hand_loop_over_eligible_rows_only(world):
    keys = ['b/1.jpg', 'a/1.jpg', 'c/v1.jpg', 'a/2.jpg']
    ignored = [p for p, (i, r) in enumerate(world.order) if r == 'ignored']
    world.txt[ignored] = 60000.0 / 2048                         # would dominate any mean that kept them
    np.save(world.tmp / 'store' / 'txt.npy', world.txt)
    got = pf_data._default_texts(keys)
    assert got.dtype == np.float32 and got.shape == (4, 2048)
    for row, key in zip(got, keys):
        np.testing.assert_allclose(row, hand_mean(world, key, world.txt), atol=1e-6)
        assert abs(np.linalg.norm(row) - 1) < 1e-5


def test_the_mean_matches_ct2s_paint_mean_then_l2(world):
    ct2 = rt_data.ct2_data
    keys = pf_data.keys('train')
    rows = pf_data.text_rows('train')
    live = rows['eligible']
    pid = np.asarray([keys.index(k) for k, e in zip(rows['key'], live) if e])
    x = world.txt[rows['store_row'][live]].astype(np.float32).astype(np.float64)
    oracle = ct2.l2(ct2.paint_mean(x, pid, len(keys)))
    np.testing.assert_array_equal(pf_data._default_texts(keys), oracle)


def test_a_key_with_no_eligible_row_is_an_error():
    with pytest.raises(AssertionError, match='no eligible'):
        pf_data._group_mean(['p', 'q'], ['p', 'p'], np.arange(2), np.ones((2, 2048), np.float16))


def test_unknown_keys_raise(world):
    with pytest.raises(AssertionError, match='unknown key'):
        pf_data._default_texts(['a/1.jpg', 'd/t1.jpg'])       # a test photo has no train/val explanations


def write_all_views(world, perm=None):
    """TXT_VIEW with every train row then every val row (ignored included, with junk); optional row permutation."""
    rows = [pf_data.text_rows(s) for s in ('train', 'val')]
    csv_rows = np.concatenate([r['csv_row'] for r in rows])
    row_keys = rows[0]['key'] + rows[1]['key']
    store = np.concatenate([r['store_row'] for r in rows])
    live = np.concatenate([r['eligible'] for r in rows])
    x = world.txt[store].astype(np.float32)
    x[live] = unit16(world.rng, int(live.sum())).astype(np.float32)        # view rows are unit (index "New views")
    x[~live] = 7.0                                            # ignored rows: junk that would show in any mean
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float16)
    perm = np.arange(len(csv_rows)) if perm is None else perm
    write_txt_view(world.txt_view, csv_rows[perm], [row_keys[i] for i in perm], x[perm])
    return csv_rows, row_keys, x


def test_E_joins_the_view_by_csv_row_even_when_the_view_is_reordered(world):
    csv_rows, row_keys, x = write_all_views(world, perm=np.random.default_rng(1).permutation(12))
    keys = ['c/v2.jpg', 'a/1.jpg', 'b/1.jpg']
    got = pf_data._instructed_texts(keys)
    by_csv = {int(c): x[i] for i, c in enumerate(csv_rows)}   # oracle: the unpermuted view rows by csv_row
    for row, key in zip(got, keys):
        total, n = np.zeros(2048), 0
        for (i, r), c in zip(world.order, world.csv_rows):
            if PHOTOS[i][0] == key and r != 'ignored':
                total += by_csv[c].astype(np.float64)
                n += 1
        mean = (total / n).astype(np.float32)
        np.testing.assert_allclose(row, mean / np.linalg.norm(mean), atol=1e-6)


def test_E_refuses_a_view_whose_rows_disagree_with_the_index(world):
    write_all_views(world)
    rows = json.loads((world.txt_view / 'rows.json').read_text())
    rows['key'][0] = 'b/1.jpg' if rows['key'][0] != 'b/1.jpg' else 'a/1.jpg'
    write_json(world.txt_view / 'rows.json', rows)
    with pytest.raises(AssertionError, match='rows.json'):
        pf_data._instructed_texts(['a/1.jpg'])


def test_E_refuses_a_view_missing_a_row(world):
    csv_rows, row_keys, x = write_all_views(world)
    write_txt_view(world.txt_view, csv_rows[1:], row_keys[1:], x[1:])
    with pytest.raises(AssertionError, match='rows.json'):
        pf_data._instructed_texts(['a/1.jpg'])


@pytest.mark.parametrize('bad', [dict(gate={**GATE, 'ok': False}), dict(gate={**GATE, 'n_bad_norm': 1}),
                                 dict(smoke=True), dict(instruction='another wording')])
def test_every_view_must_pass_its_meta_gate(world, bad):
    csv_rows, row_keys, x = write_all_views(world)
    write_txt_view(world.txt_view, csv_rows, row_keys, x, **bad)
    with pytest.raises(AssertionError):
        pf_data._instructed_texts(['a/1.jpg'])


def write_teacher(world, **bad):
    keys = pf_data.keys('val') + pf_data.keys('test')
    x = unit16(world.rng, len(keys))
    write_view(world.teacher, keys, x, **bad)
    return dict(zip(keys, x))


def test_T_takes_train_keys_from_the_train_view_and_val_keys_from_the_teacher_view_by_key(world):
    train_keys = ['b/1.jpg', 'a/2.jpg', 'a/1.jpg']            # a reordered train view still joins by key
    train_x = unit16(world.rng, 3)
    write_view(world.train_view, train_keys, train_x)
    teacher = write_teacher(world)
    asked = ['c/v2.jpg', 'a/1.jpg', 'c/v1.jpg', 'b/1.jpg']
    got = pf_data._teacher_images(asked)
    want = [teacher['c/v2.jpg'], train_x[2], teacher['c/v1.jpg'], train_x[0]]
    for row, w in zip(got, want):
        w = w.astype(np.float32)
        np.testing.assert_allclose(row, w / np.linalg.norm(w), atol=1e-6)
    assert got.dtype == np.float32


def test_T_refuses_a_train_view_with_other_keys_or_a_failed_gate(world):
    write_teacher(world)
    write_view(world.train_view, ['a/1.jpg', 'a/2.jpg'], unit16(world.rng, 2))
    with pytest.raises(AssertionError):
        pf_data._teacher_images(['a/1.jpg'])
    write_view(world.train_view, pf_data.keys('train'), unit16(world.rng, 3), gate={**GATE, 'ok': False})
    with pytest.raises(AssertionError):
        pf_data._teacher_images(['a/1.jpg'])


def test_photo_arrays_returns_I_D_E_T_in_keys_order_and_refuses_unknown_keys(world):
    write_all_views(world)
    write_view(world.train_view, pf_data.keys('train'), unit16(world.rng, 3))
    write_teacher(world)
    keys = ['c/v1.jpg', 'b/1.jpg', 'a/1.jpg']
    got = pf_data.photo_arrays(keys)
    assert set(got) == {'I', 'D', 'E', 'T'}
    for name, x in got.items():
        assert x.dtype == np.float32 and x.shape == (3, 2048)
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)
    np.testing.assert_array_equal(got['I'], pf_data.default_images(keys))
    np.testing.assert_array_equal(got['D'], pf_data._default_texts(keys))
    np.testing.assert_array_equal(got['E'], pf_data._instructed_texts(keys))
    np.testing.assert_array_equal(got['T'], pf_data._teacher_images(keys))
    with pytest.raises(AssertionError, match='unknown key'):
        pf_data.photo_arrays(['a/1.jpg', 'd/t1.jpg'])


def test_fit_keys_are_the_sorted_train_keys_minus_the_near_copies(world):
    write_json(world.tmp / 'nearcopy.json', {'n_train': 3, 'removed_keys': ['b/1.jpg']})
    assert pf_data.fit_keys() == ['a/1.jpg', 'a/2.jpg']
    write_json(world.tmp / 'nearcopy.json', {'n_train': 3, 'removed_keys': ['c/v1.jpg']})   # not a train key
    with pytest.raises(AssertionError):
        pf_data.fit_keys()


def test_fit_keys_refuses_a_near_copy_file_written_for_another_train_count(world, monkeypatch):
    write_json(world.tmp / 'nearcopy.json', {'n_train': 2, 'removed_keys': ['b/1.jpg']})   # e.g. written under PF_SMOKE
    with pytest.raises(AssertionError, match='n_train'):
        pf_data.fit_keys()
    write_json(world.tmp / 'nearcopy.json', {'removed_keys': ['b/1.jpg']})                  # no count at all
    with pytest.raises(AssertionError):
        pf_data.fit_keys()
    write_json(world.tmp / 'nearcopy.json', {'n_train': 3, 'removed_keys': ['b/1.jpg']})   # full file, smoke run
    monkeypatch.setattr(pf_paths, 'SMOKE', 2)
    with pytest.raises(AssertionError, match='n_train'):
        pf_data.fit_keys()


def tamper_meta(path, **change):
    meta = json.loads((path / 'meta.json').read_text())
    meta.update(change)
    write_json(path / 'meta.json', meta)


def test_train_view_refuses_a_wrong_keys_hash(world):
    write_view(world.train_view, pf_data.keys('train'), unit16(world.rng, 3))
    pf_data._train_view()
    tamper_meta(world.train_view, keys_sha256='0' * 64)
    with pytest.raises(AssertionError, match='view keys hash'):
        pf_data._train_view()


def test_train_view_refuses_a_wrong_image_array_hash(world):
    write_view(world.train_view, pf_data.keys('train'), unit16(world.rng, 3))
    tamper_meta(world.train_view, img_sha256='0' * 64)
    with pytest.raises(AssertionError, match='view image array hash'):
        pf_data._train_view()


def test_txt_view_refuses_a_wrong_array_hash(world):
    write_all_views(world)
    pf_data._instructed_texts(['a/1.jpg'])
    tamper_meta(world.txt_view, txt_sha256='0' * 64)
    with pytest.raises(AssertionError, match='view text array hash'):
        pf_data._instructed_texts(['a/1.jpg'])


def test_txt_view_refuses_a_wrong_rows_hash(world):
    write_all_views(world)
    tamper_meta(world.txt_view, rows_sha256='0' * 64)
    with pytest.raises(AssertionError, match='view rows hash'):
        pf_data._instructed_texts(['a/1.jpg'])


def test_txt_view_rows_hash_follows_rows_json_order(world):
    write_all_views(world, perm=np.random.default_rng(2).permutation(12))      # permuted view, hash of that order
    assert pf_data._instructed_texts(['a/1.jpg']).shape == (1, 2048)


def test_txt_view_refuses_non_unit_rows(world):
    csv_rows, row_keys, x = write_all_views(world)
    write_txt_view(world.txt_view, csv_rows, row_keys, (x.astype(np.float32) * 2).astype(np.float16))
    with pytest.raises(AssertionError):
        pf_data._instructed_texts(['a/1.jpg'])


def test_new_views_must_carry_their_own_view_name(world):
    write_view(world.train_view, pf_data.keys('train'), unit16(world.rng, 3))
    tamper_meta(world.train_view, view='emotion_focus_txt')
    with pytest.raises(AssertionError, match='view name'):
        pf_data._train_view()
    write_all_views(world)
    tamper_meta(world.txt_view, view='emotion_focus_train')
    with pytest.raises(AssertionError, match='view name'):
        pf_data._instructed_texts(['a/1.jpg'])


@pytest.mark.real_data
def test_real_train_keys_and_text_row_counts():
    """CPU, index JSON (and r12's annotation CSV for the val/test cross-check) only."""
    train = pf_data.keys('train')
    assert len(train) == 72255 == len(set(train)) and train == sorted(train)
    assert (len(pf_data.keys('val')), len(pf_data.keys('test'))) == (4251, 8501)
    assert not set(train) & set(pf_data.keys('val')) and not set(train) & set(pf_data.keys('test'))
    for split, n, eligible in (('train', 447680, 443258), ('val', 26371, 26113)):
        rows = pf_data.text_rows(split)
        assert len(rows['key']) == n and int(rows['eligible'].sum()) == eligible
        assert len(set(rows['csv_row'].tolist())) == n
        assert rows['store_row'].dtype == np.int64 and (np.diff(rows['store_row']) > 0).all()
        assert set(rows['key']) == set(pf_data.keys(split))


@pytest.mark.real_data
def test_real_smoke_takes_the_first_sorted_train_keys(monkeypatch):
    full = pf_data.keys('train')
    monkeypatch.setattr(pf_paths, 'SMOKE', 50)
    assert pf_data.keys('train') == full[:50]
    assert set(pf_data.text_rows('train')['key']) == set(full[:50])
