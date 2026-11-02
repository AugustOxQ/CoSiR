"""Tests of mt_export (ticket 02): the key join, the stale-file refusal, the file format, the 0(ii) tolerance logic and
the export loop's wiring. Synthetic fixtures; the real-data checks are the CLI `gate-ii` and the smoke export."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mt_export as E  # noqa: E402
import mt_paths as P  # noqa: E402

ENC, ASPECT, SEED, KEYSET = 'siglip2', 'style', 1, 'val'


def sha_key(k):
    return hashlib.sha256(k.encode('utf-8')).hexdigest()


def keys_in_order(n=7):
    return sorted((f'Style/p{i}.jpg' for i in range(n)), key=sha_key)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A fake checkpoint root for ENC and an output root; returns (out_root, checkpoint path)."""
    root = tmp_path / 'ckpt'
    path = root / 'weights' / P.CSTAR_ARM / f'{ASPECT}__s{SEED}.pt'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'weights-v1' * 100)
    monkeypatch.setitem(P.CSTAR_ROOT, ENC, root)
    return tmp_path / 'out', path


def rows(keys, d=8):
    """Distinct rows, so a wrong join shows."""
    return np.stack([np.full(d, i + 1, np.float16) * np.float16(.25) for i in range(len(keys))])


def write(out, keys, emb, **kw):
    args = dict(checkpoint=E.checkpoint_path(ENC, ASPECT, SEED), view=P.DEFAULT_VIEW[ENC])
    args.update(kw)
    return E.write_export(out, KEYSET, ENC, ASPECT, SEED, keys, emb, **args)


# ---- the key join ---------------------------------------------------------------------------------------------
def test_load_returns_rows_in_the_order_asked_and_joins_by_key(world):
    out, _ = world
    keys = keys_in_order()
    emb = rows(keys)
    write(out, keys, emb)
    # asked in another order, and a subset: each row is the one of its key
    ask = [keys[4], keys[0], keys[6], keys[2]]
    got = E.load(KEYSET, ENC, ASPECT, SEED, ask, root=out)
    assert got.dtype == np.float16 and got.shape == (4, emb.shape[1])
    assert np.array_equal(got, emb[[4, 0, 6, 2]])


def test_load_of_a_file_stored_in_another_order_still_returns_the_asked_rows(world):
    """The stored `keys` array is the truth: a file whose rows were shuffled (and keys with them) joins the same."""
    out, _ = world
    keys = keys_in_order()
    emb = rows(keys)
    path = write(out, keys, emb)
    perm = np.random.default_rng(0).permutation(len(keys))
    with np.load(path) as z:
        k, e = z['keys'][perm], z['emb'][perm]
    np.savez(path, keys=k, emb=e)
    side = json.loads(path.with_suffix('.json').read_text())
    side['sha256'] = E.sha_file(path)                     # a legitimate rewrite: the sidecar follows the file
    path.with_suffix('.json').write_text(json.dumps(side))
    assert np.array_equal(E.load(KEYSET, ENC, ASPECT, SEED, keys, root=out), emb)


def test_load_refuses_a_key_that_was_not_exported(world):
    out, _ = world
    keys = keys_in_order()
    write(out, keys, rows(keys))
    with pytest.raises(AssertionError, match='not exported'):
        E.load(KEYSET, ENC, ASPECT, SEED, keys + ['Style/other.jpg'], root=out)


# ---- stale and wrong files ------------------------------------------------------------------------------------
def test_load_refuses_a_file_whose_checkpoint_changed(world):
    out, ckpt = world
    keys = keys_in_order()
    write(out, keys, rows(keys))
    assert E.load(KEYSET, ENC, ASPECT, SEED, keys, root=out).shape[0] == len(keys)
    ckpt.write_bytes(b'weights-v2' * 100)                  # retrained: the export is stale
    with pytest.raises(AssertionError, match='stale'):
        E.load(KEYSET, ENC, ASPECT, SEED, keys, root=out)


def test_load_refuses_a_file_changed_after_its_sidecar(world):
    out, _ = world
    keys = keys_in_order()
    path = write(out, keys, rows(keys))
    np.savez(path, keys=np.asarray(keys), emb=rows(keys) + np.float16(1))
    with pytest.raises(AssertionError, match='sha256 differs'):
        E.load(KEYSET, ENC, ASPECT, SEED, keys, root=out)


@pytest.mark.parametrize('field,value,msg', [('encoder', 'clip_l14', 'encoder'), ('aspect', 'genre', 'aspect'),
                                             ('seed', 3, 'seed'), ('keyset', 'new', 'keyset'),
                                             ('view', 'siglip1', 'view')])
def test_load_refuses_a_sidecar_that_names_another_encoder_aspect_seed_keyset_or_view(world, field, value, msg):
    out, _ = world
    keys = keys_in_order()
    path = write(out, keys, rows(keys))
    side = json.loads(path.with_suffix('.json').read_text())
    side[field] = value
    path.with_suffix('.json').write_text(json.dumps(side))
    with pytest.raises(AssertionError, match=msg):
        E.load(KEYSET, ENC, ASPECT, SEED, keys, root=out)


def test_sidecar_records_what_the_ticket_asks(world):
    out, ckpt = world
    keys = keys_in_order()
    path = write(out, keys, rows(keys))
    side = json.loads(path.with_suffix('.json').read_text())
    assert side['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert side['checkpoint'] == str(ckpt) and side['checkpoint_sha256'] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert side['view'] == P.DEFAULT_VIEW[ENC] and side['n'] == len(keys)
    assert path == out / 'exports' / KEYSET / ENC / 'cstar' / f'{ASPECT}__s{SEED}.npz'


# ---- write-time asserts ---------------------------------------------------------------------------------------
def test_write_refuses_keys_out_of_the_sha256_order_duplicates_and_wrong_arrays(world):
    out, _ = world
    keys = keys_in_order()
    emb = rows(keys)
    with pytest.raises(AssertionError, match='order'):
        write(out, keys[::-1], emb)
    with pytest.raises(AssertionError):
        write(out, keys[:3] + keys[:1], emb[:4])
    with pytest.raises(AssertionError):
        write(out, keys, emb.astype(np.float32))
    with pytest.raises(AssertionError):
        write(out, keys, emb[:-1])
    bad = emb.copy()
    bad[2, 0] = np.nan
    with pytest.raises(AssertionError):
        write(out, keys, bad)


# ---- gate 0(ii): tolerance logic ------------------------------------------------------------------------------
def test_compare_passes_within_tolerance_and_fails_beyond_it():
    names = ['a', 'b', 'c']
    base = np.zeros((3, 4), np.float16)
    near = base.copy()
    near[1, 2] = np.float16(2 ** -10)                      # 9.77e-4 < 1e-3
    far = base.copy()
    far[1, 2] = np.float16(2 ** -9)                        # 1.95e-3 > 1e-3
    ok = E.compare_by_name(names, near, names, base)
    assert ok['pass'] and ok['max_abs_diff'] == pytest.approx(2 ** -10)
    bad = E.compare_by_name(names, far, names, base)
    assert not bad['pass'] and bad['max_abs_diff'] == pytest.approx(2 ** -9)


def test_compare_tolerance_is_inclusive_and_nan_fails():
    names = ['a', 'b']
    a, b = np.zeros((2, 3), np.float16), np.zeros((2, 3), np.float16)
    a[0, 0] = np.float16(2 ** -10)
    assert E.compare_by_name(names, a, names, b, tol=2 ** -10)['pass']
    assert not E.compare_by_name(names, a, names, b, tol=2 ** -10 - 1e-6)['pass']
    a[1, 1] = np.nan
    assert not E.compare_by_name(names, a, names, b)['pass']


def test_compare_joins_by_painting_name_not_by_position():
    names = ['a', 'b', 'c', 'd']
    x = np.arange(4 * 3, dtype=np.float16).reshape(4, 3)
    order = [2, 0, 3, 1]
    got = E.compare_by_name(names, x, [names[i] for i in order], x[order])
    assert got['pass'] and got['max_abs_diff'] == 0.0
    shifted = E.compare_by_name(names, x, names, x[order])           # same rows by position would be wrong
    assert not shifted['pass']


def test_compare_refuses_other_name_sets_and_shapes():
    x = np.zeros((3, 2), np.float16)
    with pytest.raises(AssertionError):
        E.compare_by_name(['a', 'b', 'c'], x, ['a', 'b', 'z'], x)
    with pytest.raises(AssertionError):
        E.compare_by_name(['a', 'b', 'c'], x, ['a', 'b', 'c'], x[:, :1])
    with pytest.raises(AssertionError):
        E.compare_by_name(['a', 'a', 'c'], x, ['a', 'a', 'c'], x)


def test_gate_result_fails_when_one_item_fails():
    items = {k: {'pass': True, 'max_abs_diff': 0.0} for k in E.GATE_ITEMS}
    items.update({'emotion__s0': {'pass': True, 'max_abs_diff': 1e-4}, 'style__s0': {'pass': False, 'max_abs_diff': 5e-3}})
    res = E.gate_result('siglip2', items, tol=1e-3)
    assert res['pass'] is False and res['max_abs_diff'] == pytest.approx(5e-3) and res['failed'] == ['style__s0']
    items['style__s0'] = {'pass': True, 'max_abs_diff': 2e-4}
    res = E.gate_result('siglip2', items, tol=1e-3)
    assert res['pass'] is True and res['max_abs_diff'] == pytest.approx(2e-4) and res['failed'] == []
    with pytest.raises(AssertionError):
        E.gate_result('siglip2', {}, tol=1e-3)                        # no items is not a pass


def test_gate_covers_all_five_seeds_and_three_aspects():
    assert sorted(E.GATE_ITEMS) == sorted(f'{a}__s{s}' for s in range(5) for a in P.ASPECTS)
    items = {k: {'pass': True, 'max_abs_diff': 0.0} for k in E.GATE_ITEMS[:-1]}
    with pytest.raises(AssertionError, match='15'):
        E.gate_result('qwen', items, tol=1e-3)


# ---- the export loop's wiring ---------------------------------------------------------------------------------
def test_export_runs_the_default_view_on_each_keyset_and_writes_every_seed_and_aspect(world, monkeypatch):
    out, _ = world
    import mt_data
    keysets = {'val': keys_in_order(5), 'new': keys_in_order(9)}
    seen = {}

    def fake_features(view, keys):
        seen.setdefault('views', set()).add(view)
        return (np.ones((len(keys), P.WIDTH[ENC])) / np.sqrt(P.WIDTH[ENC])).astype(np.float32)

    def fake_infer(enc, aspect, seed, x):
        assert enc == ENC and x.dtype == np.float32
        emb = np.full((len(x), 2048), seed + 1, np.float16)
        return emb, E.sha_file(E.checkpoint_path(enc, aspect, seed))

    for a in P.ASPECTS:
        for s in (0, 1):
            p = E.checkpoint_path(ENC, a, s)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f'{a}{s}'.encode())
    monkeypatch.setattr(mt_data, 'keys', lambda name: list(keysets[name]))
    monkeypatch.setattr(mt_data, 'features', fake_features)
    monkeypatch.setattr(E, 'infer', fake_infer)
    paths = E.export(ENC, keysets=('val', 'new'), seeds=(0, 1), root=out)
    assert seen['views'] == {P.DEFAULT_VIEW[ENC]}
    assert len(paths) == 2 * 2 * 3 and all(p.exists() for p in paths)
    got = E.load('new', ENC, 'genre', 1, keysets['new'], root=out)
    assert got.shape == (9, 2048) and (got == 2).all()
