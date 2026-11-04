"""Tests of om_data (ticket 06): synthetic stores and rows; the real keys are replaced by short fake lists.

Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 /root/miniconda3/envs/CoSiR/bin/python -m pytest -q
     src/test/20261103_outside_methods/test_om_data.py
"""
import json

import numpy as np
import pytest

import om_clay as oc
import om_data as od
import om_paths as op

W = op.TPIPS_WIDTH


def unit(a):
    a = np.asarray(a, dtype=np.float64)
    return (a / np.linalg.norm(a, axis=1, keepdims=True)).astype(np.float32)


def rows(n, d, seed, shift=0.0):
    x = np.random.default_rng(seed).standard_normal((n, d))
    x[:, 0] += shift
    return unit(x)


def write_store(root, factors, names, seed0):
    """A synthetic TPIPS store: `names` in the given (shuffled) order, rows that differ per name and factor."""
    out = {}
    for j, f in enumerate(factors):
        x = rows(len(names), W, seed0 + j)
        d = root / f
        d.mkdir(parents=True)
        np.save(d / 'emb.npy', x)
        (d / 'keys.json').write_text(json.dumps(list(names)))
        (d / 'meta.json').write_text(json.dumps({'n': len(names)}))
        out[f] = dict(zip(names, x))
    return out


@pytest.fixture
def fake(monkeypatch, tmp_path):
    """Fake keys: ArtELingo held p0..p3, val p4..p8 (sorted); MMArt test m0..m2, val (not sorted) m7, m5, m6 whose
    ArtELingo names are p6, p4, p8."""
    ks = {('artelingo', 'held'): [f'p{i}' for i in range(4)], ('artelingo', 'val'): [f'p{i}' for i in range(4, 9)],
          ('mmart', 'test'): ['m0', 'm1', 'm2'], ('mmart', 'val'): ['m7', 'm5', 'm6']}
    monkeypatch.setattr(od, 'keys', lambda s, sp: list(ks[(s, sp)]))
    names = {'m7': 'p6', 'm5': 'p4', 'm6': 'p8'}
    shuffled = ['p7', 'p2', 'p4', 'p0', 'p8', 'p1', 'p6', 'p3', 'p5']
    a = write_store(tmp_path / 'store_a', od.FACTORS, shuffled, 100)
    m = write_store(tmp_path / 'store_m', od.FACTORS, ['m2', 'm0', 'm1', 'm9'], 200)
    roots = {'artelingo': tmp_path / 'store_a', 'mmart': tmp_path / 'store_m'}
    return dict(keys=ks, name_of=names.__getitem__, a=a, m=m, roots=roots, tmp=tmp_path)


# ------------------------------------------------------------------------------------------------ TPIPS rows
def test_tpips_artelingo_rows_follow_keys_not_store_order(fake):
    for split in ('held', 'val'):
        x = od.tpips('artelingo', split, 'style', roots=fake['roots'])
        want = np.stack([fake['a']['style'][k] for k in fake['keys'][('artelingo', split)]])
        assert x.dtype == np.float32 and x.shape == want.shape and np.array_equal(x, want)


def test_tpips_mmart_test_from_its_own_store(fake):
    x = od.tpips('mmart', 'test', 'genre', roots=fake['roots'])
    assert np.array_equal(x, np.stack([fake['m']['genre'][k] for k in ('m0', 'm1', 'm2')]))


def test_tpips_mmart_val_joined_by_painting_name_not_position(fake):
    x = od.tpips('mmart', 'val', 'emotion', roots=fake['roots'], name_of=fake['name_of'])
    want = np.stack([fake['a']['emotion'][fake['name_of'](k)] for k in ('m7', 'm5', 'm6')])
    assert np.array_equal(x, want)
    # by position in the ArtELingo val list (m7 -> p4, ...) would give other rows
    wrong = np.stack([fake['a']['emotion'][k] for k in ('p4', 'p5', 'p6')])
    assert not np.array_equal(x, wrong)


def test_tpips_mmart_val_refuses_a_name_outside_artelingo_val(fake):
    with pytest.raises(AssertionError):
        od.tpips('mmart', 'val', 'emotion', roots=fake['roots'], name_of={'m7': 'p6', 'm5': 'p1', 'm6': 'p8'}.__getitem__)


def test_tpips_aspect_selects_its_factor(fake):
    a = od.tpips('artelingo', 'held', 'emotion', roots=fake['roots'])
    b = od.tpips('artelingo', 'held', 'style', roots=fake['roots'])
    assert not np.array_equal(a, b)
    with pytest.raises(AssertionError):
        od.tpips('artelingo', 'held', 'colour', roots=fake['roots'])


# ------------------------------------------------------------------------------------------------ CLAY
@pytest.fixture
def clay_fake(fake, monkeypatch):
    d = 768
    img = rows(40, d, 1, shift=2.0)
    lab = np.full(40, -1)
    lab[:10] = np.arange(10) % 3                       # only the first 10 rows carry a label
    text = rows(8, d, 2, shift=-1.5)
    fake['keys'][('artelingo', 'held')] = [f'q{i}' for i in range(40)]
    calls = []
    monkeypatch.setattr(od, 'plain', lambda s, sp, enc: calls.append((s, sp, enc)) or img.copy())
    monkeypatch.setattr(od, 'labels', lambda *a: (_ for _ in ()).throw(AssertionError('labels must not be read')))
    monkeypatch.setattr(oc, 'text_set', lambda name, aspect: text)
    return dict(img=img, lab=lab, text=text, calls=calls, **fake)


def test_clay_mean_uses_unlabelled_rows_too(clay_fake):
    img, text = clay_fake['img'], clay_fake['text']
    out = od.clay('artelingo', 'held', 'emotion', 'A', out=clay_fake['tmp'] / 'o')
    assert out.dtype == np.float32 and out.shape == (40, 768)
    assert clay_fake['calls'] == [('artelingo', 'held', 'clip_l14')]
    P, mu = oc.subspace(text, op.clay_rank(8))
    assert np.array_equal(out, oc.condition(img, P, mu))                     # mean over all 40 rows
    labelled_only = oc.condition(img[:10], P, mu)
    assert not np.allclose(out[:10], labelled_only, atol=1e-3)               # a labelled-only mean would differ


def test_clay_cache_written_checked_and_reused(clay_fake, monkeypatch):
    o = clay_fake['tmp'] / 'o'
    first = od.clay('artelingo', 'held', 'style', 'A', out=o)
    f = o / 'clay' / 'artelingo_held' / 'A' / 'style.npz'
    side = json.loads(f.with_suffix('.json').read_text())
    with np.load(f, allow_pickle=False) as z:
        assert z.files == ['keys', 'emb'] and z['keys'].tolist() == clay_fake['keys'][('artelingo', 'held')]
        assert np.array_equal(z['emb'], first)
    assert side['rank'] == op.clay_rank(8) and side['n'] == 40
    assert side['input_sha256'] == od.array_sha256(clay_fake['img']) and side['text_sha256'] == od.array_sha256(clay_fake['text'])
    # second call reads the cache: the conditioning function is not called
    monkeypatch.setattr(oc, 'vectors', lambda *a: (_ for _ in ()).throw(AssertionError('recomputed')))
    assert np.array_equal(od.clay('artelingo', 'held', 'style', 'A', out=o), first)


def test_clay_cache_is_recomputed_when_inputs_change(clay_fake, monkeypatch):
    o = clay_fake['tmp'] / 'o'
    first = od.clay('artelingo', 'held', 'style', 'A', out=o)
    new_text = rows(8, 768, 9, shift=-1.5)
    monkeypatch.setattr(oc, 'text_set', lambda name, aspect: new_text)
    second = od.clay('artelingo', 'held', 'style', 'A', out=o)
    assert not np.array_equal(first, second)
    assert np.array_equal(second, od.clay('artelingo', 'held', 'style', 'A', out=o))
    side = json.loads((o / 'clay' / 'artelingo_held' / 'A' / 'style.json').read_text())
    assert side['text_sha256'] == od.array_sha256(new_text)


def test_clay_cache_is_recomputed_when_one_image_row_changes(clay_fake, monkeypatch):
    o = clay_fake['tmp'] / 'o'
    first = od.clay('artelingo', 'held', 'style', 'A', out=o)
    img2 = clay_fake['img'].copy()
    img2[39] = unit(img2[39:40] + 0.5)[0]                         # one row of the images, text unchanged
    monkeypatch.setattr(od, 'plain', lambda s, sp, enc: img2.copy())
    second = od.clay('artelingo', 'held', 'style', 'A', out=o)
    assert not np.array_equal(first, second)                      # the stale cache was not returned
    side = json.loads((o / 'clay' / 'artelingo_held' / 'A' / 'style.json').read_text())
    assert side['input_sha256'] == od.array_sha256(img2)


def test_clay_refuses_a_cache_with_reordered_keys(clay_fake):
    o = clay_fake['tmp'] / 'o'
    od.clay('artelingo', 'held', 'style', 'A', out=o)
    f = o / 'clay' / 'artelingo_held' / 'A' / 'style.npz'
    with np.load(f, allow_pickle=False) as z:
        k, e = z['keys'], z['emb']
    np.savez(f, keys=k[::-1].copy(), emb=e[::-1].copy())          # a consistent file with the rows in another order
    sp = f.with_suffix('.json')
    side = json.loads(sp.read_text())
    side['emb_sha256'] = op.sha256_file(f)                        # so only the key check can catch it
    sp.write_text(json.dumps(side))
    with pytest.raises(AssertionError):
        od.clay('artelingo', 'held', 'style', 'A', out=o)


def test_clay_refuses_a_corrupted_cache_file(clay_fake):
    o = clay_fake['tmp'] / 'o'
    od.clay('artelingo', 'held', 'style', 'A', out=o)
    f = o / 'clay' / 'artelingo_held' / 'A' / 'style.npz'
    with np.load(f, allow_pickle=False) as z:
        k, e = z['keys'], z['emb'].copy()
    e[0, 0] += 0.5
    np.savez(f, keys=k, emb=e)
    with pytest.raises(AssertionError):
        od.clay('artelingo', 'held', 'style', 'A', out=o)


def test_clay_refuses_unknown_text_set_and_aspect(clay_fake):
    with pytest.raises(AssertionError):
        od.clay('artelingo', 'held', 'style', 'C', out=clay_fake['tmp'] / 'o')
    with pytest.raises(AssertionError):
        od.clay('artelingo', 'held', 'colour', 'A', out=clay_fake['tmp'] / 'o')


# ------------------------------------------------------------------------------------------------ partition files
FIT = dict(loss=1.5, entropy=0.5, n_clusters=3, seconds=2.0, device='cpu')


@pytest.fixture
def parts(fake, monkeypatch):
    ks = [f'q{i}' for i in range(12)]
    fake['keys'][('artelingo', 'held')] = ks
    fake['ks'] = ks
    fake['out'] = fake['tmp'] / 'o'
    monkeypatch.setattr(od, 'HELD_SPLIT', {'artelingo': 'held', 'mmart': 'test'})
    return fake


def lab(seed, k=3, n=12):
    return np.random.default_rng(seed).integers(0, k, n).astype(np.int32)


def save_plain(p, k=9, start=0, labels=None, **kw):
    labels = lab(start, k) if labels is None else labels
    return od.save_parts(labels, method='turtle', set_='artelingo', enc='qwen', arm='plain', k=k, start=start,
                         input_sha256='ab' * 32, out=p['out'], **FIT, **kw)


def save_cstar(p, seed, aspect='emotion', k=9, labels=None):
    labels = lab(10 + seed, k) if labels is None else labels
    return od.save_parts(labels, method='turtle', set_='artelingo', enc='qwen', arm='cstar', k=k, aspect=aspect,
                         seed=seed, start=0, input_sha256='cd' * 32, out=p['out'], **FIT)


def test_plain_parts_round_trip_with_names_and_sidecar(parts):
    for s in (0, 1, 2):
        path = save_plain(parts, start=s)
    f = parts['out'] / 'parts' / 'turtle' / 'artelingo' / 'qwen' / 'plain' / 'k9__start1.npz'
    assert f.exists() and (f.with_suffix('.json')).exists()
    side = json.loads(f.with_suffix('.json').read_text())
    for key in ('method', 'set', 'split', 'enc', 'arm', 'aspect', 'K', 'start', 'seed', 'n', 'sha256', 'input_sha256',
                'loss', 'entropy', 'n_clusters', 'seconds', 'device'):
        assert key in side, key
    assert side['sha256'] == op.sha256_file(f) and side['n'] == 12 and side['split'] == 'held' and side['K'] == 9
    with np.load(f, allow_pickle=False) as z:
        assert z.files == ['keys', 'labels'] and z['keys'].tolist() == parts['ks'] and z['labels'].dtype == np.int32
    got = od.turtle_parts('artelingo', 'qwen', 'plain', 9, out=parts['out'])
    assert got.shape == (3, 12) and got.dtype == np.int32
    assert all(np.array_equal(got[s], lab(s, 9)) for s in (0, 1, 2))
    assert path.name == 'k9__start2.npz'


def test_arm_names_with_method_suffix_are_accepted(parts):
    for s in (0, 1, 2):
        save_plain(parts, start=s)
    got = od.turtle_parts('artelingo', 'qwen', 'plain_turtle', 9, out=parts['out'])
    assert got.shape == (3, 12)


def test_cstar_parts_round_trip_seed_order(parts):
    for s in (4, 0, 3, 1, 2):
        save_cstar(parts, s)
    f = parts['out'] / 'parts' / 'turtle' / 'artelingo' / 'qwen' / 'cstar' / 'emotion__s3__k9.npz'
    assert f.exists()
    got = od.turtle_parts('artelingo', 'qwen', 'cstar', 9, aspect='emotion', out=parts['out'])
    assert got.shape == (5, 1, 12) and got.dtype == np.int32
    assert all(np.array_equal(got[s, 0], lab(10 + s, 9)) for s in range(5))


def test_reader_refuses_missing_start_or_seed(parts):
    for s in (0, 1):
        save_plain(parts, start=s)
    with pytest.raises(FileNotFoundError):
        od.turtle_parts('artelingo', 'qwen', 'plain', 9, out=parts['out'])


def test_reader_refuses_keys_out_of_order(parts):
    for s in (0, 1, 2):
        save_plain(parts, start=s)
    f = parts['out'] / 'parts' / 'turtle' / 'artelingo' / 'qwen' / 'plain' / 'k9__start1.npz'
    side = json.loads(f.with_suffix('.json').read_text())
    keys_, labels_ = parts['ks'][::-1], lab(1, 9)[::-1].copy()          # a consistent but reordered file
    np.savez(f, keys=np.array(keys_), labels=labels_)
    side['sha256'] = op.sha256_file(f)                                  # the sha is right: only the key check can refuse
    f.with_suffix('.json').write_text(json.dumps(side))
    with pytest.raises(AssertionError, match='keys'):
        od.turtle_parts('artelingo', 'qwen', 'plain', 9, out=parts['out'])


def test_reader_refuses_a_wrong_sha(parts):
    for s in (0, 1, 2):
        save_plain(parts, start=s)
    f = parts['out'] / 'parts' / 'turtle' / 'artelingo' / 'qwen' / 'plain' / 'k9__start2.npz'
    np.savez(f, keys=np.array(parts['ks']), labels=lab(99, 9))          # right keys, other labels, sidecar unchanged
    with pytest.raises(AssertionError, match='sha256'):
        od.turtle_parts('artelingo', 'qwen', 'plain', 9, out=parts['out'])


def test_reader_refuses_a_sidecar_for_another_fit(parts):
    for s in (0, 1, 2):
        save_plain(parts, start=s)
    f = parts['out'] / 'parts' / 'turtle' / 'artelingo' / 'qwen' / 'plain' / 'k9__start0.json'
    side = json.loads(f.read_text())
    side['K'] = 10
    f.write_text(json.dumps(side))
    with pytest.raises(AssertionError):
        od.turtle_parts('artelingo', 'qwen', 'plain', 9, out=parts['out'])


def test_writer_refuses_bad_labels_and_overwrite(parts):
    with pytest.raises(AssertionError):
        save_plain(parts, labels=np.full(12, 9, dtype=np.int32))        # label 9 with K 9
    with pytest.raises(AssertionError):
        save_plain(parts, labels=lab(0, 9)[:-1])                         # wrong length
    with pytest.raises(AssertionError):
        save_plain(parts, labels=np.full(12, -1, dtype=np.int32))
    save_plain(parts, start=0)
    with pytest.raises(AssertionError):
        save_plain(parts, start=0)
    save_plain(parts, start=0, overwrite=True)


def test_writer_refuses_wrong_arm_arguments(parts):
    with pytest.raises(AssertionError):                                  # c* needs an aspect and a seed
        od.save_parts(lab(0, 9), method='turtle', set_='artelingo', enc='qwen', arm='cstar', k=9, start=0,
                      input_sha256='cd' * 32, out=parts['out'], **FIT)
    with pytest.raises(AssertionError):                                  # plain has start 0 to 2 and no aspect or seed
        save_plain(parts, start=3)
    with pytest.raises(AssertionError):
        save_cstar(parts, 5)


def test_plain_k_outside_the_planned_counts_is_refused(parts):
    assert 11 not in op.plain_ks('artelingo')
    with pytest.raises(AssertionError):
        save_plain(parts, k=11, labels=lab(0, 11))


def test_cstar_k_outside_the_label_counts_is_refused(parts):
    assert 16 in op.plain_ks('artelingo') and 16 not in op.label_ks('artelingo', 'emotion')
    with pytest.raises(AssertionError, match='c\\* K'):
        save_cstar(parts, 0, k=16, labels=lab(0, 16))
    with pytest.raises(AssertionError, match='c\\* K'):
        od.turtle_parts('artelingo', 'qwen', 'cstar', 16, aspect='emotion', out=parts['out'])
    save_cstar(parts, 0, aspect='style', k=27, labels=lab(0, 27))         # ArtELingo style has 27 classes, MMArt 20
    with pytest.raises(AssertionError, match='c\\* K'):
        od.save_parts(np.zeros(3, dtype=np.int32), method='turtle', set_='mmart', enc='qwen', arm='cstar', k=27,
                      aspect='style', seed=0, start=0, input_sha256='cd' * 32, out=parts['out'], **FIT)


def test_cstar_label_counts_give_the_spec_fit_count():
    pairs = sum(len(op.label_ks(s, a)) for s in op.SETS for a in op.ASPECTS)
    assert pairs == 11 and pairs * len(op.ENCODERS) * len(op.SEEDS) == op.N_TURTLE_FITS - 192


def test_replay_method_of_gate_iii_b_round_trips(parts):
    for s in (0, 1, 2):
        od.save_parts(lab(s, 20), method='replay', set_='artelingo', enc='qwen', arm='plain', k=20, start=s,
                      input_sha256='ab' * 32, out=parts['out'], **FIT)
    assert od.turtle_parts('artelingo', 'qwen', 'plain', 20, method='replay', out=parts['out']).shape == (3, 12)


def test_temi_takes_the_same_form(parts):
    for s in (0, 1, 2):
        od.save_parts(lab(s, 9), method='temi', set_='artelingo', enc='qwen', arm='plain', k=9, start=s,
                      input_sha256='ab' * 32, out=parts['out'], **FIT)
    assert (parts['out'] / 'parts' / 'temi' / 'artelingo' / 'qwen' / 'plain' / 'k9__start0.npz').exists()
    got = od.turtle_parts('artelingo', 'qwen', 'plain', 9, method='temi', out=parts['out'])
    assert got.shape == (3, 12)
    with pytest.raises(FileNotFoundError):
        od.turtle_parts('artelingo', 'qwen', 'plain', 9, out=parts['out'])   # nothing under turtle


# ------------------------------------------------------------------------------------------------ extra, check_part
def test_extra_is_merged_into_the_sidecar(parts):
    p = save_plain(parts, extra={'torch': '2.9.0', 'turtle': dict(op.TURTLE)})
    side = json.loads(p.with_suffix('.json').read_text())
    assert side['torch'] == '2.9.0' and side['turtle'] == op.TURTLE and side['K'] == 9


def test_extra_refuses_a_standard_sidecar_key(parts):
    for key in ('loss', 'sha256', 'K', 'device'):
        with pytest.raises(AssertionError):
            save_plain(parts, extra={key: 1})
    assert not list(parts['out'].rglob('*.npz'))                  # refused before anything is written


def _chk(parts, **kw):
    return od.check_part('artelingo', 'qwen', 'plain', 9, start=0, out=parts['out'], **kw)


def test_check_part_accepts_a_good_fit_and_expect_keys(parts):
    save_plain(parts, start=0)
    assert _chk(parts) is None
    assert _chk(parts, expect={'input_sha256': 'ab' * 32}) is None
    assert _chk(parts, expect={'input_sha256': 'ee' * 32}) is not None      # input changed: refit


def test_check_part_reports_missing_npz_alone_and_corrupt_files(parts):
    assert _chk(parts) is not None                                           # nothing written
    p = save_plain(parts, start=0)
    p.with_suffix('.json').unlink()
    assert _chk(parts) is not None                                           # npz alone (crash before the sidecar)
    p2 = save_plain(parts, start=0, overwrite=True)                          # the resume rewrite
    assert _chk(parts) is None
    with np.load(p2, allow_pickle=False) as z:
        k, y = z['keys'], z['labels'].copy()
    y[0] = (y[0] + 1) % 9
    np.savez(p2, keys=k, labels=y)
    assert 'sha256' in _chk(parts)                                           # npz changed under its sidecar


def test_check_part_cstar_and_refuses_a_nonfit(parts):
    save_cstar(parts, 2)
    assert od.check_part('artelingo', 'qwen', 'cstar', 9, aspect='emotion', start=0, seed=2, out=parts['out']) is None
    assert od.check_part('artelingo', 'qwen', 'cstar', 9, aspect='emotion', start=0, seed=3, out=parts['out']) is not None
    with pytest.raises(AssertionError):
        od.check_part('artelingo', 'qwen', 'plain', 7, start=0, out=parts['out'])   # K not in plain_ks
