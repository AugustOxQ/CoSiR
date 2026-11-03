"""Tests of pp_export (r19 ticket 06): the c* exports, their sidecars and readers, gate (iii).

Fake checkpoints in r14's layout (random weights, a real sidecar that r14's `_check_weights` accepts) stand in for the
new three's X-c*, which do not exist yet; the `real` tests use the stored r10 and r14 files."""
import json
import shutil

import numpy as np
import pytest
import torch

import pp_paths as PP
import pp_register  # noqa: F401  (registers the new three before the older rounds' modules are used)
import pp_export as E

import bs_cstar
import bs_photo
import mt_export
import mt_paths
import rt_data

ENC = 'clip_b32'                       # a new encoder: ArtELingo store exists, width 512
ASPECTS = PP.ASPECTS


# ------------------------------------------------------------------------------------------------ fake checkpoints
def fake_checkpoints(root, enc, seeds=(0,), aspects=ASPECTS, width=None):
    """r14's `weights/k11=40/<aspect>__s<seed>.{pt,json}` with random weights and a sidecar `_check_weights` accepts."""
    width = width or PP.WIDTH[enc]
    out = {}
    with bs_cstar.training_context(enc, bs_cstar.REF, export=True) as ctx:
        switches = bs_cstar.V.switches(ctx.arm)
        _, views = bs_cstar.rb_student.build_model(
            bs_cstar.rb_rows.recipe(bs_cstar.rb_rows.REFERENCE)['base'], list(ASPECTS), torch.device('cpu'), width=width)
        for seed in seeds:
            for aspect in aspects:
                torch.manual_seed(1000 * seed + ASPECTS.index(aspect))
                for p in views[aspect].parameters():
                    torch.nn.init.normal_(p, std=0.05)
                path = bs_cstar.rb_student.weights_path(root, ctx.arm, aspect, seed)
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(views[aspect].state_dict(), path)
                config = dict(aspect=aspect, seed=seed, smoke=False, device='cuda', max_epochs=bs_cstar.ms_train.CONFIG['max_epochs'],
                              rb=dict(scale=bs_cstar.rg_full.SCALE, encoder=ctx.backbone, width=width,
                                      recipe=bs_cstar.rg_full.RECIPE),
                              rg=dict(arm=ctx.arm, switches=switches))
                path.with_suffix('.json').write_text(json.dumps(dict(
                    weights_sha256=E.sha_file(path), config=config, best_epoch=3, best_dev_loss=1.0)))
                out[aspect, seed] = path
    return out


@pytest.fixture
def cstar_root(tmp_path, monkeypatch):
    """A fake X-c* for ENC under tmp_path, registered as that encoder's c* root."""
    root = tmp_path / 'cstar' / ENC
    monkeypatch.setitem(mt_paths.CSTAR_ROOT, ENC, root)
    monkeypatch.setitem(PP.CSTAR_ROOT, ENC, root)
    fake_checkpoints(root, ENC, seeds=(0, 1))
    return root


def unit_rows(n, width, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, width)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def keys_n(n):
    return PP.order([f'k{i:03d}.jpg' for i in range(n)])


# ------------------------------------------------------------------------------------------------ SemArt round trip
def test_semart_round_trip_matches_direct_inference(cstar_root, tmp_path):
    keys, x = keys_n(7), unit_rows(7, 512)
    out = tmp_path / 'out'
    paths = E.export_semart(ENC, seeds=[0, 1], root=out, keys=keys, rows=x)
    assert len(paths) == 2
    assert paths[0] == out / 'exports/semart' / ENC / 'cstar/genre__s0.npz'
    # the head applied by hand (r14's inference), on the same rows
    weights = bs_cstar.rb_student.weights_path(cstar_root, 'k11=40', 'genre', 1)
    want = bs_cstar._infer_sides(weights, 'genre', 512, {'img': x}, 'cpu')['img']
    got = E.load('semart', ENC, 'genre', 1, keys, root=out)
    assert got.dtype == np.float16 and got.shape == (7, 2048)
    assert np.array_equal(got, want)
    side = json.loads(paths[1].with_suffix('.json').read_text())
    assert side['sha256'] == E.sha_file(paths[1]) and side['n'] == 7 and side['view'] == ENC
    assert side['checkpoint'] == str(weights) and side['checkpoint_sha256'] == E.sha_file(weights)
    assert (side['encoder'], side['aspect'], side['seed'], side['set']) == (ENC, 'genre', 1, 'semart')


def test_semart_export_refuses_keys_out_of_sha_order(cstar_root, tmp_path):
    keys = keys_n(5)[::-1]
    with pytest.raises(AssertionError, match='sha256 order'):
        E.export_semart(ENC, seeds=[0], root=tmp_path / 'o', keys=keys, rows=unit_rows(5, 512))


def test_semart_export_refuses_wrong_width_rows(cstar_root, tmp_path):
    with pytest.raises(AssertionError):
        E.export_semart(ENC, seeds=[0], root=tmp_path / 'o', keys=keys_n(5), rows=unit_rows(5, 768))


# ------------------------------------------------------------------------------------------------ the join and the sidecar
@pytest.fixture
def semart_files(cstar_root, tmp_path):
    keys, x = keys_n(9), unit_rows(9, 512, seed=3)
    out = tmp_path / 'out'
    E.export_semart(ENC, seeds=[0], root=out, keys=keys, rows=x)
    return keys, out


def test_load_joins_by_key_in_the_order_asked(semart_files):
    keys, out = semart_files
    full = E.load('semart', ENC, 'genre', 0, keys, root=out)
    ask = [keys[5], keys[0], keys[8]]
    part = E.load('semart', ENC, 'genre', 0, ask, root=out)
    assert np.array_equal(part, full[[5, 0, 8]])
    assert not np.array_equal(part[0], part[1])


def test_load_asserts_every_key_present(semart_files):
    keys, out = semart_files
    with pytest.raises(AssertionError, match='not exported'):
        E.load('semart', ENC, 'genre', 0, keys + ['zzz.jpg'], root=out)


def test_load_refuses_duplicate_request_keys(semart_files):
    keys, out = semart_files
    with pytest.raises(AssertionError, match='unique'):
        E.load('semart', ENC, 'genre', 0, [keys[0], keys[0]], root=out)


def test_load_refuses_a_file_changed_after_writing(semart_files):
    keys, out = semart_files
    path = E.semart_path(ENC, 'genre', 0, out)
    with np.load(path) as z:
        emb = z['emb'].copy()
    emb[0, 0] = np.float16(0.5)
    np.savez(path, keys=np.asarray(keys), emb=emb)
    with pytest.raises(AssertionError, match='sha256'):
        E.load('semart', ENC, 'genre', 0, keys, root=out)


def test_load_refuses_a_stale_checkpoint(semart_files, cstar_root):
    keys, out = semart_files
    weights = bs_cstar.rb_student.weights_path(cstar_root, 'k11=40', 'genre', 0)
    torch.save({k: v + 1 for k, v in torch.load(weights, weights_only=True).items()}, weights)   # retrained
    with pytest.raises(AssertionError, match='stale'):
        E.load('semart', ENC, 'genre', 0, keys, root=out)


@pytest.mark.parametrize('field,value,match', [('encoder', 'clip_b16', 'encoder'), ('seed', 3, 'seed'),
                                               ('aspect', 'style', 'aspect'), ('view', 'siglip2', 'view'),
                                               ('n', 4, ' n ')])
def test_load_refuses_a_sidecar_naming_something_else(semart_files, field, value, match):
    keys, out = semart_files
    side = E.semart_path(ENC, 'genre', 0, out).with_suffix('.json')
    doc = json.loads(side.read_text())
    doc[field] = value
    side.write_text(json.dumps(doc))
    with pytest.raises(AssertionError, match=match):
        E.load('semart', ENC, 'genre', 0, keys, root=out)


def test_load_refuses_a_missing_sidecar(semart_files):
    keys, out = semart_files
    E.semart_path(ENC, 'genre', 0, out).with_suffix('.json').unlink()
    with pytest.raises(FileNotFoundError):
        E.load('semart', ENC, 'genre', 0, keys, root=out)


def test_semart_exports_only_genre(cstar_root, tmp_path):
    with pytest.raises(AssertionError):
        E.semart_path(ENC, 'emotion', 0, tmp_path)


# ------------------------------------------------------------------------------------------------ ArtELingo held (real store)
def test_artelingo_held_round_trip(cstar_root, tmp_path):
    out_root = tmp_path / 'held_out'
    E.export_artelingo(ENC, seeds=[0], out_root=out_root)
    names, _ = bs_cstar.rb_student.held_names()
    path = E.held_path(ENC, 'genre', 0, out_root)
    assert path == out_root / 'held/k11=40/genre__s0.npz'
    arrays, info = bs_cstar.rb_student.read_outputs(path)
    assert arrays['img'].shape == (12281, 2048) and info['list_sha256'] == bs_cstar.rb_student.held_names()[1]
    side = json.loads(path.with_name('genre__s0.pp.json').read_text())
    weights = bs_cstar.rb_student.weights_path(cstar_root, 'k11=40', 'genre', 0)
    assert side['sha256'] == E.sha_file(path) and side['n'] == 12281 and side['view'] == ENC
    assert side['checkpoint'] == str(weights) and side['checkpoint_sha256'] == E.sha_file(weights)
    ask = [names[100], names[7], names[12280]]
    got = E.load('artelingo_held', ENC, 'genre', 0, ask, root=out_root)
    assert np.array_equal(got, arrays['img'][[100, 7, 12280]])
    with pytest.raises(AssertionError, match='not exported'):
        E.load('artelingo_held', ENC, 'genre', 0, ['no such painting'], root=out_root)


def test_held_export_refuses_other_than_the_held_list(cstar_root, tmp_path):
    out_root = tmp_path / 'held_out'
    E.export_artelingo(ENC, seeds=[0], out_root=out_root)
    side = out_root / 'held/k11=40/genre__s0.json'
    doc = json.loads(side.read_text())
    doc['list_sha256'] = '0' * 64
    side.write_text(json.dumps(doc))
    with pytest.raises(AssertionError):
        E.load('artelingo_held', ENC, 'genre', 0, ['x'], root=out_root)


# ------------------------------------------------------------------------------------------------ MMArt (val keys read ArtELingo stores)
def test_mmart_val_export_layout_and_reader(cstar_root, tmp_path):
    out = tmp_path / 'out'
    paths = E.export_mmart(ENC, keysets=['val'], seeds=[0], root=out)
    assert len(paths) == 3
    assert paths[0] == out / 'mmart/exports/val' / ENC / 'cstar' / f'{ASPECTS[0]}__s0.npz'
    side = json.loads(paths[0].with_suffix('.json').read_text())
    assert side['n'] == 5769 and side['view'] == ENC and side['sha256'] == E.sha_file(paths[0])
    import mt_data
    keys = mt_data.keys('val')
    ask = [keys[3], keys[1]]
    full = E.load('mmart_val', ENC, ASPECTS[0], 0, keys, root=out)
    assert full.shape == (5769, 2048)
    assert np.array_equal(E.load('mmart_val', ENC, ASPECTS[0], 0, ask, root=out), full[[3, 1]])


# ------------------------------------------------------------------------------------------------ Affection (clip_l14 store exists)
def test_affection_photo_round_trip(tmp_path):
    enc, root = 'clip_l14', tmp_path / 'out'
    state = bs_photo.build_emotion(torch.device('cpu'), 768)[1].state_dict()
    for p in state.values():
        p.copy_(torch.randn_like(p) * 0.05) if p.is_floating_point() else None
    ck = root / 'photo' / enc / 's0'
    ck.mkdir(parents=True)
    torch.save(state, ck / 'emotion.pt')
    (ck / 'run.json').write_text(json.dumps(dict(status='ok', weights_sha256=E.sha_file(ck / 'emotion.pt'))))
    paths = E.export_affection(enc, seeds=[0], splits=['val'], kinds=['photo'], root=root)
    assert paths == [root / 'exports/affection/photo_clip_l14/val__s0.npz']
    keys = bs_photo.pf_data.keys('val')
    with np.load(paths[0]) as z:
        assert z['keys'].tolist() == keys and z['emb'].shape == (4251, 2048)
        emb = z['emb']
    side = json.loads(paths[0].with_name('val__s0.pp.json').read_text())
    assert side['sha256'] == E.sha_file(paths[0]) and side['n'] == 4251 and side['view'] == enc
    ask = [keys[9], keys[2]]
    got = E.load('affection_val', enc, 'emotion', 0, ask, root=root, kind='photo', old_ok=False)
    assert np.array_equal(got, emb[[9, 2]])
    # a retrained checkpoint makes the export stale
    torch.save({k: v + 1 for k, v in state.items()}, ck / 'emotion.pt')
    (ck / 'run.json').write_text(json.dumps(dict(status='ok', weights_sha256=E.sha_file(ck / 'emotion.pt'))))
    with pytest.raises(AssertionError, match='stale'):
        E.load('affection_val', enc, 'emotion', 0, ask, root=root, kind='photo', old_ok=False)


# ------------------------------------------------------------------------------------------------ validation reader (real files)
@pytest.mark.parametrize('enc', ['qwen', 'clip_l14'])
def test_validation_reads_the_stored_validation_export(enc):
    names, x = E.validation(enc, 'genre', 0)
    assert names == rt_data.names('val')
    assert x.dtype == np.float32 and x.shape == (6152, 2048)
    assert abs(np.linalg.norm(x, axis=1) - 1).max() < 1e-3


def test_validation_refuses_a_changed_file(tmp_path):
    src = PP.R10_OUT / 'full/train/k11=40/R6-5'
    dst = tmp_path / 'train/k11=40/R6-5'
    dst.mkdir(parents=True)
    f = 'genre__R0o__s0'
    shutil.copy(src / f'{f}.json', dst / f'{f}.json')
    with np.load(src / f'{f}.npz') as z:
        img, cap = z['img'].copy(), z['cap']
    img[0, 0] = np.float16(0.25)
    np.savez(dst / f'{f}.npz', img=img, cap=cap)
    with pytest.raises(AssertionError, match='hash'):
        E.validation('qwen', 'genre', 0, root=tmp_path)


# ------------------------------------------------------------------------------------------------ gate (iii) logic
def items(diff=0.0, drop=None, rows=3):
    out = {k: {'pass': diff <= E.TOL, 'max_abs_diff': diff, 'rows': rows} for k in E.GATE_ITEMS['r14']}
    if drop:
        out.pop(drop)
    return out


def test_gate_items_are_all_seeds_and_aspects():
    assert len(E.GATE_ITEMS['r10']) == 15 and len(E.GATE_ITEMS['r14']) == 15 + 5
    assert 'affection_test/emotion__s4' in E.GATE_ITEMS['r14']
    assert 'artelingo_held/genre__s0' in E.GATE_ITEMS['r10']


def test_compare_is_inclusive_at_the_tolerance_and_fails_beyond():
    a = np.zeros((4, 3), dtype=np.float16)
    b = a.copy()
    b[1, 1] = np.float16(0.5)
    assert E.compare(a, b, tol=0.5)['pass'] and not E.compare(a, b, tol=0.4)['pass']
    assert E.compare(a, b, tol=0.5)['max_abs_diff'] == 0.5


def test_compare_refuses_other_shapes_and_nonfinite():
    a = np.zeros((4, 3), dtype=np.float16)
    with pytest.raises(AssertionError):
        E.compare(a, np.zeros((5, 3), dtype=np.float16))
    bad = a.copy()
    bad[0, 0] = np.float16(np.nan)
    assert not E.compare(a, bad)['pass']


def test_gate_part_passes_only_with_every_item_passing():
    ok = E.gate_part('r14', items())
    assert ok['pass'] and ok['max_abs_diff'] == 0.0 and len(ok['items']) == 20
    bad = items()
    key = 'affection_test/emotion__s2'
    bad[key] = {'pass': False, 'max_abs_diff': 0.5, 'rows': 3}
    res = E.gate_part('r14', bad)
    assert not res['pass'] and res['failed'] == [key]


def test_gate_part_refuses_a_missing_item():
    with pytest.raises(AssertionError, match='items'):
        E.gate_part('r14', items(drop='artelingo_held/emotion__s0'))


def test_write_gate_keeps_the_other_part(tmp_path):
    E.write_gate_part(tmp_path, 'r10', {'pass': True, 'marker': 1})
    doc = E.write_gate_part(tmp_path, 'r14', {'pass': False, 'marker': 2})
    assert doc['r10']['marker'] == 1 and doc['r14']['marker'] == 2 and doc['pass'] is False
    on_disk = json.loads((tmp_path / 'stage0/gate_iii.json').read_text())
    assert on_disk == doc


def test_gate_stored_reads_nothing_from_the_wrong_root(tmp_path):
    # the gate's work folder is under the given out root, never under the stored round's folders
    work = E.gate_work(tmp_path, 'r14', 'clip_l14')
    assert tmp_path in work.parents
    with pytest.raises(AssertionError):
        E.gate_work(PP.R14_OUT, 'r14', 'clip_l14')
