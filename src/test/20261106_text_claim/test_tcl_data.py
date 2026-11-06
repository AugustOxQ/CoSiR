"""Tests of the r23 data door (ticket 02). Real stores for everything that exists; tiny fixture stores in tmp_path for
the new text stores (ticket 03 makes the real ones), read through TCL_STORES's parameter `tcl_paths.STORE_ROOT`."""
import hashlib
import json
import re

import numpy as np
import pytest

import tcl_paths as P
import tcl_data as D
import tx_data
import mt_data
import mt_paths
import rt_data
import rs_vh
from pav_prompts import PROMPTS

R22 = P.R22_OUT


def write_store(path, x, instruction, ids=None, keys=None, smoke=False, dtype=np.float16):
    path.mkdir(parents=True)
    np.save(path / 'txt.npy', np.asarray(x, dtype=dtype))
    if ids is not None:
        np.savez_compressed(path / 'ids.npz', sample_ids=np.asarray(ids, dtype=np.int64))
    if keys is not None:
        (path / 'keys.json').write_text(json.dumps(list(keys)))
    (path / 'done.json').write_text(json.dumps(dict(instruction=instruction, count=len(x), smoke=smoke)))


def shuffled_with_extras(wanted, rng, n_extra=5):
    """Stored order of a fixture: the wanted IDs plus unrelated ones, shuffled; and the row of each ID."""
    stored = list(wanted) + [10**9 + i for i in range(n_extra)]
    order = rng.permutation(len(stored))
    return [stored[i] for i in order]


def unit_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def direct_art_rows(row, section, wanted):
    path = tx_data.FULL / tx_data.STORES[row]
    index = json.loads((path / 'index.json').read_text())[section]
    key = 'sample_id' if section == 'texts' else 'painting'
    positions = {r[key]: i for i, r in enumerate(index)}
    assert len(positions) == len(index)
    raw = np.load(path / ('txt.npy' if section == 'texts' else 'img.npy'), mmap_mode='r')
    return unit_rows(raw[[positions[k] for k in wanted]])


def slow_names_label(texts, own_names, aspect):
    """Direct per-text scan by label name: no tcl codebooks, class maps or mask helpers."""
    import ewm_masks
    table = tx_data.STYLE_PATTERNS if aspect == 'style' else tx_data.GENRE_PATTERNS
    patterns = {k.lower(): v for k, v in table.items()}
    out = []
    for text, own in zip(texts, own_names):
        if own is None:
            out.append(False)
        elif aspect == 'emotion':
            words = [m.group().lower() for m in ewm_masks.PATTERNS['label'].finditer(text)]
            named = {ewm_masks.EMOTION_NAMES[ewm_masks.WORD_LABEL[w]] for w in words}
            out.append(own.lower() in named)
        else:
            out.append(re.search(r'\b(?:' + patterns[own.lower()] + r')\b', text, re.IGNORECASE) is not None)
    assert len(out) == len(texts) == len(own_names)
    return np.asarray(out, dtype=bool)


@pytest.fixture(scope='module')
def mm_test_raw():
    import pandas as pd
    keys = mt_data.keys('test')
    cols = ['image_id', 'style', 'dominant_emotion', *['e_' + k for k in P.KINDS]]
    raw = pd.read_parquet('/data/SSD/mmart/raw/data/data.parquet', columns=cols,
                          filters=[('image_id', 'in', keys)]).set_index('image_id')
    assert raw.index.is_unique
    return raw.loc[keys]


# ------------------------------------------------------------------------------------------ real stores
@pytest.mark.parametrize('set_name,kind', [('art_val', 'caption'), ('art_held', 'caption'),
                                         ('mm_val', 'narrative'), ('mm_test', 'unified')])
@pytest.mark.parametrize('smoke', [0, 3])
def test_frame_edits_do_not_poison_later_calls(monkeypatch, set_name, kind, smoke):
    monkeypatch.setattr(P, 'SMOKE', smoke)
    f = D.frame(set_name, kind)
    arrays = {k: v for k, v in f.items() if isinstance(v, np.ndarray)}
    arrays.update({(a, row): v for a, lab in f['labels'].items() for row, v in lab.items()})
    for key, value in arrays.items():
        original = value.copy()
        value[:] = -1
        fresh = D.frame(set_name, kind)
        got = fresh['labels'][key[0]][key[1]] if isinstance(key, tuple) else fresh[key]
        assert np.array_equal(got, original), key
        value[:] = original
    for key in ('names', 'text'):
        original = list(f[key])
        f[key][0] = 'edited by a scoring cell'
        assert D.frame(set_name, kind)[key] == original


def test_gate_iv():
    got = D.gate_iv()
    assert got == dict(art_held_paintings=12281, art_held_captions=61744, art_held_genre_labelled=9949,
                       mm_test_keys=16384, art_val_paintings=6152, art_val_captions=30872, mm_val_keys=5769)


@pytest.mark.parametrize('fault', ['membership', 'order', 'duplicate'])
def test_gate_iv_rejects_wrong_keys_independently_of_r15_order(monkeypatch, fault):
    keys = {s: mt_data.keys(s) for s in ('val', 'test')}
    if fault == 'membership':
        keys['test'][0] = 'not a member of the frozen r15 cutset'
        keys['test'].sort(key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    elif fault == 'order':
        keys['test'].reverse()
    else:
        keys['test'][0] = keys['test'][1]
    # A broken upstream order helper must not make the gate tautological.
    monkeypatch.setattr(mt_data, 'keys', lambda split: list(keys[split]))
    monkeypatch.setattr(mt_data, 'order', lambda ks: list(ks))
    with pytest.raises(AssertionError, match='MMArt test'):
        D.gate_iv()


def test_art_val_frame_equals_r22():
    f = D.frame('art_val', 'caption')
    c, lab = tx_data.val_captions(), tx_data.caption_labels()
    assert f['n_i'] == 6152 and f['n_t'] == 30872
    assert f['names'] == rt_data.names('val')
    assert np.array_equal(f['sample_id'], c['sample_id'])
    assert np.array_equal(f['tp'], c['pid']) and f['tp'].dtype == np.int64
    assert np.array_equal(f['ip'], np.arange(6152))
    assert f['text'] == c['text'].tolist()
    for a in P.ASPECTS:
        assert np.array_equal(f['labels'][a]['t'], lab[a]) and f['labels'][a]['t'].dtype == np.int64
        assert np.array_equal(f['labels'][a]['i'], rt_data.labels('val')[a])


def test_art_val_names_label_equals_r22():
    for a in P.ASPECTS:
        got = D.names_label('art_val', 'caption', a)
        assert got.dtype == bool and np.array_equal(got, tx_data.names_label(a))


def test_art_val_plain_and_teacher_equal_r22():
    text, image = D.plain('art_val', 'caption')
    assert np.array_equal(text, tx_data.text_rows('plain')) and np.array_equal(image, tx_data.image_rows('plain'))
    for a in P.ASPECTS:
        text, image = D.teacher('art_val', 'caption', a)
        assert np.array_equal(text, tx_data.text_rows('teacher', a, out=R22))
        assert np.array_equal(image, tx_data.image_rows('teacher', a))


def test_art_val_side_rows_equal_r22():
    for row in ('clip_l14', 'siglip2_so400m', 'siglip_so400m'):
        text, image = D.side('art_val', row)
        assert np.array_equal(text, tx_data.text_rows(row)) and np.array_equal(image, tx_data.image_rows(row))
    text, image = D.side('art_val', 'roberta')
    assert image is None and text.shape == (30872, 768) and text.dtype == np.float32
    assert np.array_equal(text, tx_data.text_rows('roberta'))


def test_art_held_frame_and_plain():
    f = D.frame('art_held', 'caption')
    assert f['n_i'] == 12281 and f['n_t'] == 61744 and f['names'] == rt_data.names('held')
    assert (np.diff(f['sample_id']) > 0).all() and f['tp'].min() == 0 and f['tp'].max() == 12280
    held = rt_data.labels('held')
    assert int((f['labels']['genre']['i'] >= 0).sum()) == 9949
    for a in P.ASPECTS:
        assert np.array_equal(f['labels'][a]['i'], held[a])
        assert f['labels'][a]['t'].shape == (61744,)
        if a != 'emotion':
            assert np.array_equal(f['labels'][a]['t'], held[a][f['tp']])
    # a painting's plurality emotion is the plurality of its captions' own emotions
    counts = np.zeros((12281, 9), dtype=np.int64)
    np.add.at(counts, (f['tp'], f['labels']['emotion']['t']), 1)
    assert np.array_equal(counts.argmax(1), held['emotion'])
    text, image = D.plain('art_held', 'caption')
    assert text.shape == (61744, 2048) and image.shape == (12281, 2048)
    assert text.dtype == image.dtype == np.float32
    assert np.allclose(np.linalg.norm(text, axis=1), 1, atol=1e-3) and np.allclose(np.linalg.norm(image, axis=1), 1, atol=1e-3)
    assert np.array_equal(text, direct_art_rows('plain', 'texts', f['sample_id']))
    assert np.array_equal(image, direct_art_rows('plain', 'images', f['names']))


@pytest.mark.parametrize('row', ['clip_l14', 'siglip2_so400m', 'siglip_so400m'])
def test_art_held_side_rows_match_direct_store(row):
    f = D.frame('art_held', 'caption')
    text, image = D.side('art_held', row)
    assert np.array_equal(text, direct_art_rows(row, 'texts', f['sample_id']))
    assert np.array_equal(image, direct_art_rows(row, 'images', f['names']))


@pytest.mark.parametrize('aspect', P.ASPECTS)
def test_art_held_teacher_images(aspect):
    assert np.array_equal(D.image('art_held', 'teacher', aspect), rt_data.embed('instructed', 'held', aspect))


def test_art_held_names_label_matches_direct_recompute():
    f = D.frame('art_held', 'caption')
    annotations = json.loads(rs_vh.ANNOTATIONS.read_text())
    rows = [annotations[int(s)] for s in f['sample_id']]
    genre = dict(zip(rt_data.names('held'), rt_data.labels('held')['genre']))
    own = dict(emotion=[r['emotion'] for r in rows], style=[r['art_style'] for r in rows],
               genre=[tx_data.GENRE_NAMES[genre[r['painting']]] if genre[r['painting']] >= 0 else None for r in rows])
    for a in P.ASPECTS:
        got = D.names_label('art_held', 'caption', a)
        assert got.dtype == bool and got.shape == (61744,)
        assert np.array_equal(got, slow_names_label([r['caption'] for r in rows], own[a], a))
        assert not got[f['labels'][a]['t'] < 0].any()


def test_mm_val_frame_equals_r15_and_parquet():
    import pandas as pd
    keys = mt_data.keys('val')
    f = D.frame('mm_val', 'narrative')
    assert f['names'] == keys and f['n_i'] == f['n_t'] == 5769
    assert np.array_equal(f['tp'], np.arange(5769)) and np.array_equal(f['ip'], np.arange(5769))
    for a in P.ASPECTS:
        y = mt_data.labels(keys, a)
        assert np.array_equal(f['labels'][a]['t'], y) and np.array_equal(f['labels'][a]['i'], y)
    raw = pd.read_parquet('/data/SSD/mmart/raw/data/data.parquet', columns=['image_id', 'e_unified'],
                          filters=[('image_id', 'in', keys[:50])]).set_index('image_id')['e_unified']
    u = D.frame('mm_val', 'unified')
    assert u['text'][:50] == [raw[k] for k in keys[:50]]
    assert D.frame('mm_val', 'formal')['text'] != u['text']


def test_mm_val_plain_joins_by_key():
    keys = mt_data.keys('val')
    for kind in ('narrative', 'unified'):
        text, image = D.plain('mm_val', kind)
        store = R22 / 'text/mmart_val' / ('e_' + kind)
        stored = json.loads((store / 'keys.json').read_text())
        assert stored != keys                       # r22 stored them in the index's order, so the join matters
        x = np.load(store / 'txt.npy').astype(np.float32)
        want = unit_rows(np.stack([x[stored.index(k)] for k in keys[:20]]))
        assert np.array_equal(text[:20], want) and text.shape == (5769, 2048)
    assert np.array_equal(image, mt_data.features('qwen_default', keys))


def test_mm_test_frame_gate_and_images():
    keys = mt_data.keys('test')
    f = D.frame('mm_test', 'unified')
    assert f['names'] == keys == sorted(keys, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    assert f['n_t'] == 16384 and len(f['text']) == 16384
    image = D.image('mm_test', 'plain')
    assert image.shape == (16384, 2048)
    assert np.array_equal(image[:5], mt_data.features('qwen_default', keys[:5]))


@pytest.mark.parametrize('kind', P.KINDS)
def test_mm_test_labels_text_and_names_label_against_raw(kind, mm_test_raw):
    import pandas as pd
    f = D.frame('mm_test', kind)
    assert f['names'] == mm_test_raw.index.tolist()
    assert f['text'] == mm_test_raw['e_' + kind].tolist()
    index = pd.read_parquet(mt_paths.MMART_INDEX).set_index('key').loc[f['names']]
    own = dict(emotion=[name or None for name in mm_test_raw['dominant_emotion'].str.lower()],
               style=[name or None for name in mm_test_raw['style'].str.lower()],
               genre=[mt_paths.CLASSES['genre'][y] if y >= 0 else None for y in index['genre']])
    for a in P.ASPECTS:
        codes = {name: i for i, name in enumerate(mt_paths.CLASSES[a])}
        want = np.asarray([codes[name] if name is not None else -1 for name in own[a]], dtype=np.int64)
        for row in ('t', 'i'):
            assert f['labels'][a][row].dtype == np.int64
            assert np.array_equal(f['labels'][a][row], want)
        assert np.array_equal(D.names_label('mm_test', kind, a),
                              slow_names_label(mm_test_raw['e_' + kind].tolist(), own[a], a))


@pytest.mark.parametrize('set_name', ['mm_val', 'mm_test'])
@pytest.mark.parametrize('aspect', P.ASPECTS)
def test_mm_teacher_images_all_aspects(set_name, aspect):
    keys = mt_data.keys(D.SPLIT[set_name])
    view = dict(emotion='emotion_focus', style='style_word', genre='genre_word')[aspect]
    assert np.array_equal(D.image(set_name, 'teacher', aspect), mt_data.features(view, keys))


def test_d9_shares_on_mm_val():
    # Spec D9: own-label word share, labelled validation rows, in per cent: (emotion, style, genre).
    expected = dict(narrative=(2.3, 0.0, 4.7), unified=(15.7, 83.2, 37.6))
    got = {k: tuple(round(100 * D.label_share('mm_val', k, a), 3) for a in P.ASPECTS) for k in expected}
    print('D9 shares measured', got)
    for kind, want in expected.items():
        for g, w in zip(got[kind], want):
            assert abs(g - w) <= 0.05, (kind, got[kind], want)
    art = tuple(round(100 * D.label_share('art_val', 'caption', a), 3) for a in P.ASPECTS)
    print('ArtELingo captions', art)


def test_mm_names_label_maps_every_class():
    for a in ('style', 'genre', 'emotion'):
        mapping = D.class_map(a)
        assert set(mapping) == set(mt_paths.CLASSES[a])
        assert all(v in D.r22_names(a) for v in mapping.values())
    for a in P.ASPECTS:
        got = D.names_label('mm_val', 'unified', a)
        f = D.frame('mm_val', 'unified')
        codes = mt_data.labels(f['names'], a)
        own = [mt_paths.CLASSES[a][y] if y >= 0 else None for y in codes]
        assert got.shape == (5769,) and np.array_equal(got, slow_names_label(f['text'], own, a))


def test_val_calls_read_no_held_label(monkeypatch):
    calls = []
    real_rt, real_mt = rt_data.labels, mt_data.labels
    test_keys = set(mt_data.keys('test'))

    def rt_labels(split):
        calls.append(('rt', split))
        return real_rt(split)

    def mt_labels(keys, aspect, **kw):
        calls.append(('mt', bool(set(keys) & test_keys)))
        return real_mt(keys, aspect, **kw)

    monkeypatch.setattr(rt_data, 'labels', rt_labels)
    monkeypatch.setattr(mt_data, 'labels', mt_labels)
    for fn in (D._art_captions, D._frame_full, D._names_label_full):
        fn.cache_clear()
    for a in P.ASPECTS:
        D.frame('art_val', 'caption')
        D.names_label('art_val', 'caption', a)
        D.frame('mm_val', 'narrative')
        D.names_label('mm_val', 'unified', a)
        D.label_share('mm_val', 'unified', a)
    assert ('rt', 'val') in calls and ('mt', False) in calls
    assert ('rt', 'held') not in calls and ('mt', True) not in calls


# ------------------------------------------------------------------------------------------ fixture stores
def smoke_frame(monkeypatch, n):
    monkeypatch.setattr(P, 'SMOKE', n)
    return n


@pytest.mark.parametrize('set_name,kind', [('art_val', 'caption'), ('art_held', 'caption'),
                                         ('mm_val', 'narrative'), ('mm_test', 'unified')])
def test_smoke_takes_first_paintings(monkeypatch, set_name, kind):
    full = D.frame(set_name, kind)
    masks = {a: D.names_label(set_name, kind, a) for a in P.ASPECTS}
    images = D.image(set_name, 'plain')
    monkeypatch.setattr(P, 'SMOKE', 3)
    f = D.frame(set_name, kind)
    keep = full['tp'] < 3
    assert f['n_i'] == 3 and f['names'] == full['names'][:3]
    if 'sample_id' in full:
        assert np.array_equal(f['sample_id'], full['sample_id'][keep])
    assert f['n_t'] == int(keep.sum()) and f['text'] == [t for t, k in zip(full['text'], keep) if k]
    assert np.array_equal(f['tp'], full['tp'][keep])
    assert np.array_equal(f['ip'], np.arange(3))
    for a in P.ASPECTS:
        assert np.array_equal(f['labels'][a]['t'], full['labels'][a]['t'][keep])
        assert np.array_equal(f['labels'][a]['i'], full['labels'][a]['i'][:3])
        assert np.array_equal(D.names_label(set_name, kind, a), masks[a][keep])
    assert np.array_equal(D.image(set_name, 'plain'), images[:3])


def test_teacher_art_held_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'SMOKE', 3)
    monkeypatch.setattr(P, 'STORE_ROOT', tmp_path)
    f = D.frame('art_held', 'caption')
    rng = np.random.default_rng(0)
    for a, instr in P.INSTRUCTIONS.items():
        stored = shuffled_with_extras(f['sample_id'].tolist(), rng)
        x = rng.normal(size=(len(stored), 2048)).astype(np.float16)
        write_store(tmp_path / P.ART_TEACHER_STORE.name / f'{instr}_heldtxt', x, PROMPTS[instr], ids=stored, smoke=True)
        got = D.teacher_text('art_held', 'caption', a)
        want = unit_rows(np.stack([x[stored.index(int(s))] for s in f['sample_id']]))
        assert np.array_equal(got, want) and got.shape == (f['n_t'], 2048)


def test_teacher_art_held_wrong_instruction(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'SMOKE', 3)
    monkeypatch.setattr(P, 'STORE_ROOT', tmp_path)
    f = D.frame('art_held', 'caption')
    stored = f['sample_id'].tolist()
    x = np.ones((len(stored), 2048))
    write_store(tmp_path / P.ART_TEACHER_STORE.name / 'style_word_heldtxt', x, PROMPTS['genre_word'], ids=stored, smoke=True)
    with pytest.raises(AssertionError):
        D.teacher_text('art_held', 'caption', 'style')


def test_smoke_store_refused_outside_smoke(tmp_path):
    write_store(tmp_path / 's', np.ones((4, 2048)), PROMPTS['default'], ids=[1, 2, 3, 4], smoke=True)
    assert P.SMOKE == 0
    with pytest.raises(AssertionError):
        D.read_store(tmp_path / 's', [1, 2], 'ids', PROMPTS['default'], 2048, np.float16)


def test_store_root_refused_even_for_non_smoke_store(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'SMOKE', 0)
    monkeypatch.setattr(P, 'STORE_ROOT', tmp_path)
    keys = mt_data.keys('test')
    write_store(tmp_path / P.MMART_TEXT_STORE.name / 'narrative/default/test',
                np.ones((len(keys), 2048)), PROMPTS['default'], keys=keys, smoke=False)
    with pytest.raises(AssertionError, match='TCL_STORES requires TCL_SMOKE'):
        D.plain_text('mm_test', 'narrative')


def test_store_with_missing_id_or_bad_width_crashes(tmp_path):
    write_store(tmp_path / 's', np.ones((4, 2048)), PROMPTS['default'], ids=[1, 2, 3, 4])
    with pytest.raises(AssertionError):
        D.read_store(tmp_path / 's', [1, 9], 'ids', PROMPTS['default'], 2048, np.float16)
    with pytest.raises(AssertionError):
        D.read_store(tmp_path / 's', [1, 2], 'ids', PROMPTS['default'], 1152, np.float16)
    write_store(tmp_path / 'z', np.zeros((4, 2048)), PROMPTS['default'], ids=[1, 2, 3, 4])
    with pytest.raises(AssertionError):
        D.read_store(tmp_path / 'z', [1, 2], 'ids', PROMPTS['default'], 2048, np.float16)


def test_store_with_wrong_done_count_crashes(tmp_path):
    path = tmp_path / 'bad_count'
    write_store(path, np.ones((4, 2048)), PROMPTS['default'], ids=[1, 2, 3, 4])
    done = json.loads((path / 'done.json').read_text())
    done['count'] = 5
    (path / 'done.json').write_text(json.dumps(done))
    with pytest.raises(AssertionError, match='store rows, ids and count differ'):
        D.read_store(path, [1, 2], 'ids', PROMPTS['default'], 2048, np.float16)


def test_roberta_art_held_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'SMOKE', 3)
    monkeypatch.setattr(P, 'STORE_ROOT', tmp_path)
    f = D.frame('art_held', 'caption')
    rng = np.random.default_rng(1)
    stored = shuffled_with_extras(f['sample_id'].tolist(), rng)
    x = rng.normal(size=(len(stored), 768)).astype(np.float32)
    write_store(tmp_path / P.ART_ROBERTA_STORE.name, x, None, ids=stored, dtype=np.float32, smoke=True)
    text, image = D.side('art_held', 'roberta')
    want = unit_rows(np.stack([x[stored.index(int(s))] for s in f['sample_id']]))
    assert image is None and text.dtype == np.float32 and np.array_equal(text, want)


@pytest.mark.parametrize('split', ['val', 'test'])
def test_mmart_text_stores_fixture(monkeypatch, tmp_path, split):
    monkeypatch.setattr(P, 'SMOKE', 4)
    monkeypatch.setattr(P, 'STORE_ROOT', tmp_path)
    set_name = 'mm_' + split
    keys = D.frame(set_name, 'narrative')['names']
    assert len(keys) == 4
    rng = np.random.default_rng(2)
    base = tmp_path / P.MMART_TEXT_STORE.name
    stored = shuffled_with_extras(keys, rng)
    stored = [str(k) for k in stored]
    # teacher stores: any kind, any instruction; plain test store: instruction default
    for kind in ('narrative', 'unified'):
        for a, instr in P.INSTRUCTIONS.items():
            x = rng.normal(size=(len(stored), 2048)).astype(np.float16)
            write_store(base / kind / instr / split, x, PROMPTS[instr], keys=stored, smoke=True)
            got = D.teacher_text(set_name, kind, a)
            want = unit_rows(np.stack([x[stored.index(k)] for k in keys]))
            assert np.array_equal(got, want)
        if split == 'test':
            x = rng.normal(size=(len(stored), 2048)).astype(np.float16)
            write_store(base / kind / 'default' / split, x, PROMPTS['default'], keys=stored, smoke=True)
            got = D.plain_text(set_name, kind)
            assert np.array_equal(got, unit_rows(np.stack([x[stored.index(k)] for k in keys])))


def test_mm_plain_val_uses_r22_store_in_smoke(monkeypatch):
    monkeypatch.setattr(P, 'SMOKE', 6)
    keys = D.frame('mm_val', 'formal')['names']
    got = D.plain_text('mm_val', 'formal')
    store = R22 / 'text/mmart_val/e_formal'
    stored = json.loads((store / 'keys.json').read_text())
    x = np.load(store / 'txt.npy').astype(np.float32)
    assert np.array_equal(got, unit_rows(np.stack([x[stored.index(k)] for k in keys])))
