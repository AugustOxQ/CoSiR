"""Ticket 04 tests: late fusion algebra on real rows, row identities and real gate fault checks."""
import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import tcl_paths as P
import tcl_forward as F
import rt_data
import tcl_data
import tx_model
import mt_data
import mt_export

DEVICE = os.environ.get('TCL_TEST_DEVICE', 'cpu')

N = 400   # first paintings' rows


@pytest.fixture(scope='module', autouse=True)
def cached_plain():
    # rows() and its independent side reference share immutable plain inputs.
    with pytest.MonkeyPatch.context() as patch:
        cached = lru_cache(maxsize=None)(tcl_data.plain)
        patch.setattr(tcl_data, 'plain', cached)
        yield
        cached.cache_clear()


@pytest.fixture(scope='module')
def art():
    """art_val emotion seed 0: c*, plain, late rows (full), text and image."""
    out = {r: F.rows('art_val', 'caption', r, 'emotion' if r != 'plain' else 'all', 0 if r != 'plain' else 'x', DEVICE)
           for r in ('cstar', 'late', 'plain')}
    return out


def test_fuse_endpoints_and_mixture(art):
    c_t, c_i = art['cstar']
    p_t, p_i = art['plain']
    for c, p in ((c_t[:N], p_t[:N]), (c_i[:N], p_i[:N])):
        assert np.allclose(np.linalg.norm(c, axis=1), 1, atol=1e-5) and np.allclose(np.linalg.norm(p, axis=1), 1, atol=1e-5)
        zero, one = F.fuse(c, p, 0.0), F.fuse(c, p, 1.0)
        assert zero.shape[1] == 4096
        assert np.array_equal(zero[:, :2048], c) and not zero[:, 2048:].any()      # a = 0: c* up to the weight 1
        assert np.array_equal(one[:, 2048:], p) and not one[:, :2048].any()        # a = 1: plain
        assert np.allclose(zero @ zero[::-1].T, c @ c[::-1].T, atol=1e-6)
        assert np.allclose(one @ one[::-1].T, p @ p[::-1].T, atol=1e-6)


def test_late_rows_cosine_is_the_mixture(art):
    a = P.A
    assert a == 0.7
    (c_t, c_i), (p_t, p_i), (l_t, l_i) = art['cstar'], art['plain'], art['late']
    assert l_t.shape == (len(c_t), 4096) and l_i.shape == (len(c_i), 4096) and l_t.dtype == l_i.dtype == np.float32
    q, g = slice(0, N), slice(1000, 1000 + N)
    for lq, lg, cq, cg, pq, pg in ((l_t, l_i, c_t, c_i, p_t, p_i), (l_i, l_t, c_i, c_t, p_i, p_t),
                                   (l_t, l_t, c_t, c_t, p_t, p_t)):
        late = lq[q] @ lg[g].T
        want = (1 - a) * (cq[q] @ cg[g].T) + a * (pq[q] @ pg[g].T)
        assert np.abs(late - want).max() < 1e-5
        assert np.allclose(lq[q, :2048], np.sqrt(1 - a) * cq[q]) and np.allclose(lq[q, 2048:], np.sqrt(a) * pq[q])


def test_cstar_image_rows_match_saved_export(art):
    got = art['cstar'][1]
    saved = rt_data.embed('cstar', 'val', 'emotion', 0)
    assert got.shape == saved.shape == (P.N_PAINTINGS['art_val'], 2048)
    assert np.abs(got - saved).max() < 1e-3


def test_rows_refuse_bad_arguments():
    with pytest.raises(AssertionError):
        F.rows('art_val', 'caption', 'cstar', 'all', 0, 'cpu')
    with pytest.raises(AssertionError):
        F.rows('art_val', 'caption', 'plain', 'emotion', 'x', 'cpu')
    with pytest.raises(AssertionError):
        F.rows('art_val', 'caption', 'late', 'emotion', 'x', 'cpu')
    with pytest.raises(AssertionError):
        F.rows('mm_val', 'narrative', 'free', 'emotion', 0, 'cpu')       # free is ArtELingo only
    with pytest.raises(AssertionError):
        F.rows('art_val', 'narrative', 'plain', 'all', 'x', 'cpu')


def test_forward_names_no_label_reader():
    """Source-only guard: delegates in rows() can still load labels through tcl_data.frame."""
    src = Path(F.__file__).read_text()
    code = '\n'.join(re.sub(r'#.*', '', line) for line in src.splitlines())
    body = code.split('"""', 2)[2]                      # drop the module doc
    for banned in ('labels', 'label_share', 'names_label', 'frame(', '.labels'):
        assert banned not in body.replace("'labels'", ''), banned


CASES = [('art_val', 'style', 3), ('art_held', 'genre', 2),
         ('mm_val', 'emotion', 4), ('mm_test', 'style', 1)]


@pytest.fixture(params=CASES, ids=lambda x: '-'.join(map(str, x)))
def identity(request, monkeypatch):
    set_name, head, seed = request.param
    kind = 'caption' if set_name.startswith('art_') else 'narrative'
    if set_name == 'mm_test':
        # Ticket 03 has not supplied this store. Keep the real image/key path.
        keys = mt_data.keys('test')
        assert len(keys) == 16384
        synthetic = tcl_data.unit(np.random.default_rng(4).standard_normal((len(keys), 2048)).astype(np.float32))
        original = tcl_data.plain_text

        def plain_text(s, k):
            if s == 'mm_test':
                assert k == kind and tcl_data.frame(s, k)['names'] == keys
                return synthetic
            return original(s, k)

        monkeypatch.setattr(tcl_data, 'plain_text', plain_text)
    plain = tcl_data.plain(set_name, kind)
    cstar = F.rows(set_name, kind, 'cstar', head, seed, DEVICE)
    return set_name, kind, head, seed, plain, cstar


def test_image_identity_by_row(identity):
    set_name, kind, head, seed, plain, cstar = identity
    if set_name.startswith('art_'):
        saved = rt_data.embed('cstar', tcl_data.SPLIT[set_name], head, seed)
    else:
        keys = tcl_data.frame(set_name, kind)['names']
        saved = mt_export.load('val' if set_name == 'mm_val' else 'new', 'qwen', head, seed, keys, root=P.R15_OUT)
    assert cstar[1].shape == saved.shape == (P.N_PAINTINGS[set_name], 2048)
    assert np.max(np.abs(cstar[1] - saved.astype(np.float32))) <= 1e-3


def test_text_identity_by_row(identity):
    set_name, kind, head, seed, plain, cstar = identity
    image_side, caption_side = tx_model.load_sides(head, seed, DEVICE)
    np.testing.assert_array_equal(cstar[0], caption_side(plain[0]))
    assert np.max(np.abs(cstar[0] - image_side(plain[0]))) > 1e-3


@pytest.mark.parametrize('set_name,head,seed', CASES[:2])
def test_free_variant(set_name, head, seed):
    plain = tcl_data.plain(set_name, 'caption')
    cstar = F.rows(set_name, 'caption', 'cstar', head, seed, DEVICE)
    free = F.rows(set_name, 'caption', 'free', head, seed, DEVICE)
    image_side, _ = tx_model.load_sides(head, seed, DEVICE)
    np.testing.assert_array_equal(free[0], image_side(plain[0]))
    np.testing.assert_array_equal(free[1], cstar[1])


def test_teacher_delegates_head():
    want = tcl_data.teacher('art_val', 'caption', 'style')
    got = F.rows('art_val', 'caption', 'teacher', 'style', 'x', DEVICE)
    for actual, expected in zip(got, want):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('row', ('clip_l14', 'siglip2_so400m', 'siglip_so400m', 'roberta'))
def test_side_delegates_row(monkeypatch, row):
    # Exercise dispatch without requiring ticket 03's held text stores.
    calls = []
    want = (np.ones((3, 5), dtype=np.float32), None if row == 'roberta' else np.zeros((2, 5), dtype=np.float32))

    def side(set_name, selected):
        calls.append((set_name, selected))
        return want

    monkeypatch.setattr(tcl_data, 'side', side)
    got = F.rows('art_held', 'caption', row, 'all', 'x', DEVICE)
    assert got is want and calls == [('art_held', row)]


def test_late_rows_are_not_normalised(art):
    for late, cstar, plain in zip(art['late'], art['cstar'], art['plain']):
        assert np.array_equal(late[:, :2048], np.float32(np.sqrt(0.3)) * cstar)
        assert np.array_equal(late[:, 2048:], np.float32(np.sqrt(0.7)) * plain)


GATE_SETS = {'art_val_image', 'art_held_image', 'mm_test_image', 'art_val_caption'}


@pytest.fixture(scope='module')
def gate_runner():
    # The first call runs all real weights/inputs. Reuse only identical frozen
    # forwards in later fault runs. Stored arrays are independent real reads,
    # cached below so repeated faults need neither disk writes nor rereads.
    original = tx_model.load_sides
    cache = {}

    @lru_cache(maxsize=None)
    def load_sides(head, seed, device):
        sides = original(head, seed, device)

        def cached(side):
            def forward(x):
                key = (head, seed, side, x.shape, hashlib.sha256(x.tobytes()).digest())
                if key not in cache:
                    cache[key] = sides[side](x)
                return cache[key]
            return forward

        return cached(0), cached(1)

    def run(out):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(tx_model, 'load_sides', load_sides)
            result = F.gate_ii(DEVICE, out=out)
        assert json.loads((out / 'stage0/gate_ii.json').read_text()) == result
        return result

    # Fault wrappers copy these arrays before changing one entry. They wrap the
    # cached loaders, so every fault still reaches the real gate comparison.
    with pytest.MonkeyPatch.context() as patch:
        embed = lru_cache(maxsize=None)(rt_data.embed)
        caption = lru_cache(maxsize=None)(F._stored_caption)
        original_export = mt_export.load
        original_features = mt_data.features
        text_rows = lru_cache(maxsize=None)(F.tx_data.text_rows)

        @lru_cache(maxsize=None)
        def features(view, keys):
            return original_features(view, list(keys))

        @lru_cache(maxsize=None)
        def export(keyset, enc, head, seed, keys, root):
            return original_export(keyset, enc, head, seed, list(keys), root=root)

        def load_export(keyset, enc, head, seed, keys, root=None):
            return export(keyset, enc, head, seed, tuple(keys), root)

        patch.setattr(rt_data, 'embed', embed)
        patch.setattr(F, '_stored_caption', caption)
        patch.setattr(mt_export, 'load', load_export)
        patch.setattr(F.tx_data, 'text_rows', text_rows)
        patch.setattr(mt_data, 'features', lambda view, keys: features(view, tuple(keys)))
        yield run
        cache.clear()
        embed.cache_clear()
        caption.cache_clear()
        export.cache_clear()
        features.cache_clear()
        text_rows.cache_clear()
        load_sides.cache_clear()


def test_gate_ii_clean(gate_runner, tmp_path):
    result = gate_runner(tmp_path)
    assert result['tolerance'] == 1e-3
    assert set(result['files']) == {f'{head}__s{seed}' for head in P.ASPECTS for seed in P.SEEDS}
    for item in result['files'].values():
        assert set(item) == GATE_SETS
        assert all(1e-5 < value <= 1e-3 for value in item.values()), item
    assert set(result['largest']) == GATE_SETS
    for name in GATE_SETS:
        assert result['largest'][name] == max(item[name] for item in result['files'].values())
    assert result['pass'] == {'art': True, 'mm': True}


@pytest.mark.parametrize('target', sorted(GATE_SETS))
def test_gate_ii_stored_fault(gate_runner, monkeypatch, tmp_path, target):
    # One entry, one file, one set. Never change a real store or cached array.
    head, seed = 'style', 3
    embed, load, caption = rt_data.embed, mt_export.load, F._stored_caption

    def changed(x):
        out = x.astype(np.float32).copy()
        out[7, 11] += np.float32(2e-3)
        return out

    def faulty_embed(arm, split, aspect, selected_seed=None):
        x = embed(arm, split, aspect, selected_seed)
        if (arm, aspect, selected_seed) == ('cstar', head, seed) and target == f'art_{split}_image':
            return changed(x)
        return x

    def faulty_load(keyset, enc, aspect, selected_seed, keys, root=None):
        x = load(keyset, enc, aspect, selected_seed, keys, root=root)
        return changed(x) if target == 'mm_test_image' and (aspect, selected_seed) == (head, seed) else x

    def faulty_caption(aspect, selected_seed):
        x = caption(aspect, selected_seed)
        return changed(x) if target == 'art_val_caption' and (aspect, selected_seed) == (head, seed) else x

    monkeypatch.setattr(rt_data, 'embed', faulty_embed)
    monkeypatch.setattr(mt_export, 'load', faulty_load)
    monkeypatch.setattr(F, '_stored_caption', faulty_caption)
    result = gate_runner(tmp_path)
    for file, item in result['files'].items():
        for name, value in item.items():
            if (file, name) == (f'{head}__s{seed}', target):
                assert 1e-3 < value < 3e-3
            else:
                assert 1e-5 < value <= 1e-3
    assert result['pass'] == {
        'art': target not in ('art_val_image', 'art_held_image', 'art_val_caption'),
        'mm': target not in ('mm_test_image', 'art_val_caption'),
    }
