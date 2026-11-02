"""Ticket 04 (verdict code): the generalized photo trainer and the emotion-pathway export, on tiny synthetic data (CPU).

Each test guards one piece of logic and fails when that piece is broken (the sabotage list is in the ticket's hand-back).
"""
import json
import socket
from pathlib import Path

import numpy as np
import pytest
import torch

import bs_paths as P
import bs_photo as BP
import pf_export
import pf_paths
import pf_train
import ms_train
import rs_vh

N_FIT, N_VAL = 300, 80
CPU = torch.device('cpu')
FIRST = 'img.trunk.0.weight'


def _unit(rng, n, d):
    x = rng.standard_normal((n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _part(rng, n, tag, d_img=2048):
    return dict(keys=[f'{tag}/img{i:04d}.jpg' for i in range(n)], I=_unit(rng, n, d_img), D=_unit(rng, n, 2048),
                E=_unit(rng, n, 2048), T=_unit(rng, n, 2048))


def _pairs(rng, n, m):
    a, b = rng.integers(0, n, 4 * m), rng.integers(0, n, 4 * m)
    keep = a != b
    p = np.unique(np.stack([np.minimum(a, b), np.maximum(a, b)], 1)[keep], axis=0)
    return p[:m].astype(np.int64)


def _data(d_img=2048):
    rng = np.random.default_rng(0)
    fit, val = _part(rng, N_FIT, 'fit', d_img), _part(rng, N_VAL, 'val', d_img)
    graph = dict(fit_pairs=_pairs(rng, N_FIT, 700), dev_pairs=_pairs(rng, N_VAL, 150), manifest_sha256='ab' * 32)
    return fit, val, graph


@pytest.fixture()
def data():
    return _data()


def _ref(**kw):
    return dict(BP.ROWS['ref'], **kw)


def _fake_cstar(root, width, seed=3, cstar_seed=0):
    root.mkdir(parents=True)
    _, view = BP.build_emotion(CPU, width, seed=seed)
    path = root / f'emotion__s{cstar_seed}.pt'
    torch.save({k: v.detach().clone() for k, v in view.state_dict().items()}, path)
    path.with_suffix('.json').write_text(json.dumps(dict(weights_sha256=rs_vh.file_sha(path))))
    return path


# ------------------------------------------------------------------------------------------- the rows and settings

def test_fourteen_rows_and_reference():
    assert list(BP.ROWS) == ['ref'] + BP.SWEEP_ROWS and len(BP.SWEEP_ROWS) == 14 and len(set(BP.SWEEP_ROWS)) == 14
    expect = {'k=20': ('k', 20), 'k=40': ('k', 40), 'k=160': ('k', 160),
              'lr=0.0001': ('lr', 0.0001), 'lr=0.0003': ('lr', 0.0003), 'lr=0.003': ('lr', 0.003),
              'epochs=2': ('epochs', 2), 'epochs=5': ('epochs', 5), 'epochs=20': ('epochs', 20),
              'teacher_weight=0': ('teacher_weight', 0.0), 'teacher_weight=0.5': ('teacher_weight', 0.5),
              'teacher_weight=1': ('teacher_weight', 1.0), 'start=random': ('start', 'random'),
              'correction=off': ('correction', False)}
    assert set(BP.SWEEP_ROWS) == set(expect)
    ref = BP.ROWS['ref']
    for name, (column, value) in expect.items():
        row = BP.ROWS[name]
        assert set(row) == set(ref)
        assert row[column] == value and type(row[column]) is type(value), name
        assert [c for c in ref if row[c] != ref[c]] == [column], name        # one change at a time


def test_reference_is_r13s_picked_photo_model_settings():
    pick = json.loads((P.R13_ROUND / 'stage1_sheet.json').read_text())['pick']['settings']
    ref = dict(BP.ROWS['ref'])
    assert ref.pop('epochs') is None
    assert ref == pick == dict(start='cstar', lr=0.001, k=80, correction=True, teacher_weight=0.25)
    assert BP.train_seed(3) == 103 == pf_paths.train_seed(3)


def test_recipe_constants_are_r13s():
    assert BP.RECIPE == pf_paths.RECIPE == dict(weight_decay=0.05, batch_pairs=512, max_epochs=200, patience=15)
    for w in (0.25, 0.5, 1.0):
        assert BP.loss_recipe(w) == pf_train.loss_recipe(w)
    row, switches = BP.loss_recipe(0.0)                                   # teacher weight 0: no teacher term at all
    assert row['distill'] == 'none' and row['distill_weight'] == 0.0 and switches == pf_train.loss_recipe(0.25)[1]


# ---------------------------------------------------------------------------------------------- the model

def test_qwen_model_is_r13s():
    _, mine = BP.build_emotion(CPU, 2048, seed=7)
    _, theirs = pf_train.build_emotion(CPU, seed=7)
    a, b = mine.state_dict(), theirs.state_dict()
    assert list(a) == list(b) == list(pf_train.CKPT_KEYS)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_other_width_changes_only_the_first_layer():
    _, qwen = BP.build_emotion(CPU, 2048, seed=7)
    for width in (768, 1152):
        _, other = BP.build_emotion(CPU, width, seed=7)
        a, b = qwen.state_dict(), other.state_dict()
        assert list(a) == list(b)
        assert [k for k in a if a[k].shape != b[k].shape] == [FIRST]
        assert b[FIRST].shape == (1024, width) and a[FIRST].shape == (1024, 2048)
        assert [type(m) for m in qwen.modules()] == [type(m) for m in other.modules()]


def test_input_width_must_match_the_model(data):
    fit, val, graph = data
    fit_x = dict(fit, I=_unit(np.random.default_rng(1), N_FIT, 1152))
    val_x = dict(val, I=_unit(np.random.default_rng(2), N_VAL, 1152))
    pairs = (graph['fit_pairs'], graph['dev_pairs'])
    with pytest.raises(AssertionError):
        BP.fit_photo('r', _ref(start='random'), 2048, 0, 100, fit_x, val_x, pairs, CPU, max_epochs=1)
    with pytest.raises(AssertionError):
        BP.fit_photo('r', _ref(start='random'), 1152, 0, 100, fit, val, pairs, CPU, max_epochs=1)


# ------------------------------------------------------------------------ training: the Qwen path is r13's

@pytest.mark.parametrize('settings', [dict(start='random'), dict(start='cstar', teacher_weight=1.0),
                                      dict(start='random', lr=0.0003, teacher_weight=0.5)])
def test_qwen_path_reproduces_r13s_trainer(tmp_path, data, settings):
    fit, val, graph = data
    cstar = _fake_cstar(tmp_path / 'cstar', 2048)
    s = _ref(**settings)
    theirs = pf_train.train_run('t', {c: s[c] for c in pf_paths.COLUMNS}, 0, 105, fit, val, graph, 'cpu',
                                max_epochs=3, outputs=tmp_path / 'r13', cstar_root=cstar.parent)
    mine = BP.train_run('t', s, 2048, 0, 105, fit, val, graph, 'cpu', max_epochs=3, keep='all',
                        out_dir=tmp_path / 'mine', cstar_root=cstar.parent)
    assert theirs['status'] == 'ok'
    assert mine.record['best_epoch'] == theirs['best_epoch']
    assert mine.record['best_dev_loss'] == theirs['best_dev_loss']
    assert [h['dev_loss'] for h in mine.record['history']] == [h['dev_loss'] for h in theirs['history']]
    assert [h['train_loss'] for h in mine.record['history']] == [h['train_loss'] for h in theirs['history']]
    assert mine.record['weights_sha256'] == theirs['weights_sha256']
    assert mine.record['cstar_weights_sha256'] == theirs['cstar_weights_sha256']
    assert mine.record['fit_keys_sha256'] == theirs['fit_keys_sha256']


def test_start_checkpoint_is_used_and_checked(tmp_path, data):
    fit, val, graph = data
    path = _fake_cstar(tmp_path / 'cstar', 2048)
    a = BP.train_run('a', _ref(), 2048, 0, 105, fit, val, graph, 'cpu', max_epochs=1, keep='none', cstar_root=path.parent)
    b = BP.train_run('b', _ref(start='random'), 2048, 0, 105, fit, val, graph, 'cpu', max_epochs=1, keep='none')
    assert a.record['weights_sha256'] is None and a.record['cstar_weights_sha256'] == rs_vh.file_sha(path)
    assert b.record['cstar_weights_sha256'] is None
    assert not torch.equal(a.state[FIRST], b.state[FIRST])
    path.with_suffix('.json').write_text(json.dumps(dict(weights_sha256='0' * 64)))
    with pytest.raises(AssertionError, match='sidecar'):
        BP.train_run('c', _ref(), 2048, 0, 105, fit, val, graph, 'cpu', max_epochs=1, keep='none', cstar_root=path.parent)


def test_other_width_trains_from_its_own_cstar_checkpoint(tmp_path):
    fit, val, graph = _data(1152)
    path = _fake_cstar(tmp_path / 'cstar', 1152)
    run = BP.train_run('x', _ref(), 1152, 0, 100, fit, val, graph, 'cpu', max_epochs=2, keep='none', cstar_root=path.parent)
    assert run.state[FIRST].shape == (1024, 1152) and run.record['encoder_width'] == 1152 and run.record['status'] == 'ok'
    wrong = _fake_cstar(tmp_path / 'wrong', 2048)                          # a 2048-wide start for a 1152-wide model
    with pytest.raises(RuntimeError):
        BP.train_run('y', _ref(), 1152, 0, 100, fit, val, graph, 'cpu', max_epochs=1, keep='none', cstar_root=wrong.parent)


# ---------------------------------------------------------------- training: fixed epochs, no stopping

def _rising_dev(monkeypatch):
    """Validation loss rises every epoch: the best epoch is always epoch 1, and the patience (15) runs out at epoch 16."""
    calls = []

    def dev_loss(*a, **k):
        calls.append(1)
        return 1.0 + len(calls)
    monkeypatch.setattr(pf_train, 'dev_loss', dev_loss)
    return calls


@pytest.mark.parametrize('n', [2, 5, 20])
def test_fixed_epoch_row_stops_at_exactly_that_checkpoint(monkeypatch, data, n):
    fit, val, graph = data
    calls = _rising_dev(monkeypatch)
    snaps = {}
    run = BP.train_run('f', _ref(start='random', epochs=n), 2048, 0, 100, fit, val, graph, 'cpu', keep='none',
                       on_epoch=lambda e, state: snaps.__setitem__(e, {k: v.clone() for k, v in state.items()}))
    assert len(calls) == n and [h['epoch'] for h in run.record['history']] == list(range(1, n + 1))
    assert run.record['best_epoch'] == n and run.record['fixed_epochs'] == n          # not the best-dev epoch (1)
    assert run.record['best_dev_loss'] == 1.0 + n == run.record['history'][-1]['dev_loss']
    assert all(torch.equal(run.state[k], snaps[n][k]) for k in snaps[n])                # the state after epoch n
    assert not all(torch.equal(snaps[1][k], snaps[n][k]) for k in snaps[n])


def test_standard_row_still_stops_by_patience_and_keeps_best(monkeypatch, data):
    fit, val, graph = data
    calls = _rising_dev(monkeypatch)
    run = BP.train_run('s', _ref(start='random'), 2048, 0, 100, fit, val, graph, 'cpu', keep='none')
    assert len(calls) == 16 and run.record['best_epoch'] == 1 and run.record['fixed_epochs'] is None


def test_standard_row_needs_strict_improvement(monkeypatch, data):
    fit, val, graph = data
    calls = []
    monkeypatch.setattr(pf_train, 'dev_loss', lambda *a, **k: calls.append(1) or 1.0)       # never improves after epoch 1
    run = BP.train_run('s', _ref(start='random'), 2048, 0, 100, fit, val, graph, 'cpu', keep='none')
    assert len(calls) == 16 and run.record['best_epoch'] == 1


def test_teacher_weight_zero_has_no_teacher_term(monkeypatch, data):
    fit, val, graph = data
    seen = []
    original = ms_train.distill_loss
    monkeypatch.setattr(ms_train, 'distill_loss', lambda *a, **k: seen.append(1) or original(*a, **k))
    BP.train_run('z', _ref(start='random', teacher_weight=0.0), 2048, 0, 100, fit, val, graph, 'cpu', max_epochs=1,
                 keep='none')
    assert not seen
    BP.train_run('z', _ref(start='random'), 2048, 0, 100, fit, val, graph, 'cpu', max_epochs=1, keep='none')
    assert seen


def test_epochs_setting_and_max_epochs_are_exclusive(data):
    fit, val, graph = data
    with pytest.raises(AssertionError):
        BP.train_run('f', _ref(start='random', epochs=2), 2048, 0, 100, fit, val, graph, 'cpu', max_epochs=3, keep='none')


# --------------------------------------------------------------------------- failures, records, files

def test_nonfinite_train_and_validation_loss_raise(data):
    fit, val, graph = data
    bad = dict(fit, I=fit['I'].copy())
    bad['I'][3] = np.nan
    with pytest.raises(BP.NonFiniteLoss, match='train'):
        BP.train_run('n', _ref(start='random'), 2048, 0, 100, bad, val, graph, 'cpu', max_epochs=2, keep='none')
    badv = dict(val, I=val['I'].copy())
    badv['I'][3] = np.nan
    with pytest.raises(BP.NonFiniteLoss, match='validation'):
        BP.train_run('n', _ref(start='random'), 2048, 0, 100, fit, badv, graph, 'cpu', max_epochs=2, keep='none')


def test_failed_run_leaves_a_note_and_no_checkpoint(tmp_path, data):
    fit, val, graph = data
    bad = dict(fit, I=fit['I'].copy())
    bad['I'][3] = np.nan
    with pytest.raises(BP.NonFiniteLoss):
        BP.train_run('n', _ref(start='random'), 2048, 0, 100, bad, val, graph, 'cpu', max_epochs=2, keep='all',
                     out_dir=tmp_path / 's0')
    assert not (tmp_path / 's0/emotion.pt').exists() and not (tmp_path / 's0/run.json').exists()
    assert json.loads((tmp_path / 's0/failed.json').read_text())['status'] == 'failed'


def test_keep_all_and_keep_none_files(tmp_path, data):
    fit, val, graph = data
    none = BP.train_run('a', _ref(start='random'), 2048, 0, 100, fit, val, graph, 'cpu', max_epochs=1, keep='none',
                        out_dir=tmp_path / 'none')
    assert sorted(p.name for p in (tmp_path / 'none').iterdir()) == ['run.json']
    assert none.state is not None and none.record['weights_sha256'] is None
    allk = BP.train_run('a', _ref(start='random'), 2048, 0, 100, fit, val, graph, 'cpu', max_epochs=1, keep='all',
                        out_dir=tmp_path / 'all')
    assert sorted(p.name for p in (tmp_path / 'all').iterdir()) == ['emotion.pt', 'run.json']
    rec = json.loads((tmp_path / 'all/run.json').read_text())
    assert rec == json.loads(json.dumps(allk.record))
    assert rs_vh.file_sha(tmp_path / 'all/emotion.pt') == rec['weights_sha256']
    state = torch.load(tmp_path / 'all/emotion.pt', weights_only=True)
    assert tuple(state) == pf_train.CKPT_KEYS and all(torch.equal(state[k], allk.state[k]) for k in state)
    with pytest.raises(AssertionError):                                       # a done run is never overwritten
        BP.train_run('a', _ref(start='random'), 2048, 0, 100, fit, val, graph, 'cpu', max_epochs=1, keep='all',
                     out_dir=tmp_path / 'all')


def test_run_record_fields(data):
    fit, val, graph = data
    run = BP.train_run('a', _ref(start='random'), 2048, 0, 100, fit, val, graph, 'cpu', max_epochs=2, keep='none')
    r = run.record
    assert r['status'] == 'ok' and r['host'] == socket.gethostname() and r['device'] == 'cpu'
    assert r['seconds'] > 0 and r['best_epoch'] in (1, 2) and r['train_seed'] == 100 and r['cstar_seed'] == 0
    assert r['best_dev_loss'] == min(h['dev_loss'] for h in r['history'])
    assert r['settings'] == _ref(start='random') and r['encoder_width'] == 2048
    assert r['fit_keys_sha256'] == rs_vh.names_sha(fit['keys']) and r['graph_manifest_sha256'] == 'ab' * 32


def test_images_replace_only_the_image_input(data):
    fit, val, _ = data
    x = _unit(np.random.default_rng(5), N_FIT, 1152)
    out = BP.with_images(fit, x)
    assert out['I'] is x and out['keys'] is fit['keys']
    assert all(out[k] is fit[k] for k in ('D', 'E', 'T'))
    with pytest.raises(AssertionError):
        BP.with_images(fit, x[:-1])


# ------------------------------------------------------------------------------------------------- export

def test_qwen_inference_is_r13s_and_other_width_runs():
    rng = np.random.default_rng(3)
    _, view = pf_train.build_emotion(CPU, seed=11)
    state = {k: v.detach().clone() for k, v in view.state_dict().items()}
    x = _unit(rng, 700, 2048)
    mine, theirs = BP.infer(state, x, 2048), pf_export.infer(state, x)
    assert mine.dtype == np.float16 and mine.shape == (700, 2048) and np.array_equal(mine, theirs)
    _, narrow = BP.build_emotion(CPU, 768, seed=11)
    state768 = {k: v.detach().clone() for k, v in narrow.state_dict().items()}
    out = BP.infer(state768, _unit(rng, 10, 768), 768)
    assert out.shape == (10, 2048) and out.dtype == np.float16
    with pytest.raises(AssertionError):
        BP.infer(state768, _unit(rng, 10, 2048), 768)
    with pytest.raises(RuntimeError):
        BP.infer(state768, _unit(rng, 10, 2048), 2048)                        # a 768-wide state in a 2048-wide model


def test_export_files_and_sidecar(tmp_path):
    rng = np.random.default_rng(4)
    ckpt = tmp_path / 'emotion.pt'
    _, view = BP.build_emotion(CPU, 768, seed=2)
    torch.save({k: v.detach().clone() for k, v in view.state_dict().items()}, ckpt)
    sha = rs_vh.file_sha(ckpt)
    keys = [f'image/awe/awe_{i:05d}.jpg' for i in range(40)]
    x = _unit(rng, 40, 768)
    path = BP.export_checkpoint('emoset', 'photo_siglip2', 'val', 3, ckpt, sha, 768, keys, x, root=tmp_path / 'out')
    assert path == tmp_path / 'out/exports/emoset/photo_siglip2/val__s3.npz'
    with np.load(path, allow_pickle=False) as z:
        assert sorted(z.files) == ['emb', 'keys'] and z['emb'].dtype == np.float16 and z['emb'].shape == (40, 2048)
        assert z['keys'].tolist() == keys
        emb = z['emb']
    side = json.loads(path.with_suffix('.json').read_text())
    assert side['sha256'] == rs_vh.file_sha(path)
    assert side['checkpoint'] == str(ckpt) and side['checkpoint_sha256'] == sha
    assert side['emb_sha256'] == rs_vh.array_sha(emb) and side['keys_sha256'] == rs_vh.names_sha(keys)
    again = BP.export_checkpoint('emoset', 'photo_siglip2', 'val', 3, ckpt, sha, 768, keys, x, root=tmp_path / 'out')
    assert again == path                                                      # same content: accepted, not rewritten
    with pytest.raises(AssertionError):                                      # another content under the same name
        BP.export_checkpoint('emoset', 'photo_siglip2', 'val', 3, ckpt, sha, 768, keys, x[::-1].copy(),
                             root=tmp_path / 'out')
    with pytest.raises(AssertionError):                                      # the checkpoint hash is checked
        BP.export_checkpoint('emoset', 'photo_siglip2', 'test', 3, ckpt, '0' * 64, 768, keys, x, root=tmp_path / 'out')


def test_export_joins_by_key_not_position(tmp_path):
    ckpt = tmp_path / 'emotion.pt'
    _, view = BP.build_emotion(CPU, 768, seed=2)
    torch.save({k: v.detach().clone() for k, v in view.state_dict().items()}, ckpt)
    sha = rs_vh.file_sha(ckpt)
    rng = np.random.default_rng(6)
    keys = [f'k{i}' for i in range(30)]
    table = dict(zip(keys, _unit(rng, 30, 768)))
    shuffled = keys[::-1]
    a = BP.export_checkpoint('emoset', 'photo_clip_l14', 'val', 0, ckpt, sha, 768, keys, np.stack([table[k] for k in keys]),
                             root=tmp_path / 'a')
    b = BP.export_checkpoint('emoset', 'photo_clip_l14', 'val', 0, ckpt, sha, 768, shuffled,
                             np.stack([table[k] for k in shuffled]), root=tmp_path / 'b')
    with np.load(a) as za, np.load(b) as zb:
        assert dict(zip(za['keys'].tolist(), za['emb'])).keys() == dict(zip(zb['keys'].tolist(), zb['emb'])).keys()
        ea, eb = dict(zip(za['keys'].tolist(), za['emb'])), dict(zip(zb['keys'].tolist(), zb['emb']))
        assert all(np.array_equal(ea[k], eb[k]) for k in keys)
    with pytest.raises(AssertionError):
        BP.export_checkpoint('emoset', 'photo_clip_l14', 'val', 1, ckpt, sha, 768, keys[:-1] + keys[:1],
                             np.stack([table[k] for k in keys]), root=tmp_path / 'c')      # duplicate key


def test_arm_names_and_checkpoint_paths(tmp_path):
    assert BP.arm_checkpoint('photo_qwen', 2)[0] == P.R13_OUT / 'runs/final/s2/emotion.pt'
    assert BP.arm_checkpoint('cstar_qwen', 2)[0] == pf_paths.CSTAR_WEIGHTS / 'emotion__s2.pt'
    assert BP.arm_checkpoint('photo_siglip1', 1, root=tmp_path)[0] == tmp_path / 'photo/siglip1/s1/emotion.pt'
    assert BP.arm_checkpoint('cstar_clip_l14', 1, cstar_root=tmp_path / 'c')[0] == \
        tmp_path / 'c/weights/k11=40/emotion__s1.pt'
    assert BP.arm_checkpoint('cstar_clip_l14', 1, root=tmp_path)[0] == \
        P.OUT / 'cstar/clip_l14/weights/k11=40/emotion__s1.pt'                # the photo root does not move c*
    assert [BP.arm_checkpoint(a, 0, root=tmp_path)[1] for a in ('photo_qwen', 'photo_siglip2', 'cstar_clip_l14')] == \
        [2048, 1152, 768]
    with pytest.raises(AssertionError):
        BP.arm_checkpoint('photo_dino', 0)


# ------------------------------------------------------------------------------------ review fixes (t04 round 2)

def _entries(keys):
    return [['awe', k, k.replace('image/', 'annotation/').replace('.jpg', '.json')] for k in keys]


def test_emoset_export_keys_come_from_the_split_files_not_store_position(monkeypatch):
    val = [f'image/awe/awe_{i:05d}.jpg' for i in range(0, 12, 2)]
    test = [f'image/fear/fear_{i:05d}.jpg' for i in range(9)]
    store = [k for pair in zip(test[:6], val) for k in pair] + test[6:]            # val and test interleaved
    assert store[:len(val)] != val and len(store) == len(val) + len(test)
    monkeypatch.setattr(BP.bs_extract, 'read_split_entries', lambda root=None, zf=None: dict(val=_entries(val), test=_entries(test)))
    monkeypatch.setattr(BP.bs_stores, 'folders', lambda dataset, view: ['store'])
    monkeypatch.setattr(BP.bs_stores, 'folder_keys', lambda folder, dataset: store)
    assert BP.export_keys('emoset', 'val') == val and BP.export_keys('emoset', 'test') == test
    monkeypatch.setattr(BP.bs_stores, 'folder_keys', lambda folder, dataset: [k for k in store if k != val[2]])
    with pytest.raises(AssertionError, match='store'):                            # a split key missing from the store
        BP.export_keys('emoset', 'val')


class _OneGraph:
    def __init__(self, graph):
        self.graph = graph

    def get(self, k, correction):
        return self.graph


def test_start_checkpoint_default_is_tickets_03_root_whatever_the_photo_root(tmp_path, monkeypatch):
    import bs_cstar
    monkeypatch.setattr(P, 'OUT', tmp_path / 'out')
    assert BP.start_root('siglip2') == bs_cstar.default_root('siglip2') / 'weights/k11=40' == \
        tmp_path / 'out/cstar/siglip2/weights/k11=40'
    assert BP.start_root('siglip2', cstar_root=tmp_path / 'c') == tmp_path / 'c/weights/k11=40'
    assert BP.start_root('qwen', cstar_root=tmp_path / 'c') == Path(pf_paths.CSTAR_WEIGHTS)
    fit, val, graph = _data(1152)
    data = (fit, val, fit, val)
    path = _fake_cstar(tmp_path / 'out/cstar/siglip2/weights/k11=40', 1152)
    other_photo_root = tmp_path / 'elsewhere'                                 # a photo root with no c* under it
    out = BP.train('siglip2', [BP.REF], [0], 'cpu', keep='none', root=other_photo_root, max_epochs=1, data=data,
                   graphs=_OneGraph(graph))
    assert out[(BP.REF, 0)].record['cstar_weights_sha256'] == rs_vh.file_sha(path)
    explicit = _fake_cstar(tmp_path / 'c2/weights/k11=40', 1152)
    out = BP.train('siglip2', [BP.REF], [0], 'cpu', keep='none', root=tmp_path / 'third', max_epochs=1, data=data,
                   graphs=_OneGraph(graph), cstar_root=tmp_path / 'c2')
    assert out[(BP.REF, 0)].record['cstar_weights_sha256'] == rs_vh.file_sha(explicit)


def test_default_run_folders_are_distinct_and_the_indexs(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'OUT', tmp_path / 'out')
    out = tmp_path / 'out'
    part_a = BP.run_dir('qwen', BP.REF, 0)
    sweep_ref = BP.run_dir('qwen', BP.REF, 0, sweep=True)
    v = BP.run_dir('qwen', BP.REF, 0, root=BP.regress_v_root())
    assert part_a == out / 'photo/qwen/s0'
    assert sweep_ref == out / 'sweeps/photo/ref/s0'
    assert BP.run_dir('qwen', 'k=20', 2, sweep=True) == out / 'sweeps/photo/k=20/s2'
    assert v == out / 'stage0/photo_regress_v/photo/qwen/s0'
    assert len({part_a, sweep_ref, v}) == 3
    with pytest.raises(AssertionError):                                       # a sweep row has no part-a folder
        BP.run_dir('qwen', 'k=20', 0)
    with pytest.raises(AssertionError):                                       # the sweep is Qwen's (index: no encoder)
        BP.run_dir('siglip2', BP.REF, 0, sweep=True)


def test_regress_v_uses_its_own_root(monkeypatch, tmp_path):
    monkeypatch.setattr(P, 'OUT', tmp_path / 'out')
    ref = json.loads((P.R13_OUT / 'runs/final/s0/run.json').read_text())
    seen = {}

    def fake_train(enc, rows, seeds, device, **kw):
        seen.update(kw, enc=enc, rows=rows, seeds=seeds)
        return {(BP.REF, 0): BP.PhotoRun(dict(best_epoch=ref['best_epoch'], best_dev_loss=ref['best_dev_loss'],
                                              weights_sha256=ref['weights_sha256'], device='cuda', host='h', seconds=1.),
                                         None)}
    monkeypatch.setattr(BP, 'train', fake_train)
    assert BP.regress_v(device='cuda') == 0
    assert seen['root'] == tmp_path / 'out/stage0/photo_regress_v' and seen['keep'] == 'all' and seen['seeds'] == [0]
    assert json.loads((tmp_path / 'out/stage0/photo_regress_v.json').read_text())['ok'] is True


def test_sweep_train_writes_the_sweep_folder_and_keeps_no_weights(monkeypatch, tmp_path, data):
    monkeypatch.setattr(P, 'OUT', tmp_path / 'out')
    fit, val, graph = data
    out = BP.train('qwen', ['start=random'], [0], 'cpu', root=None, max_epochs=1, data=(fit, val, fit, val),
                   graphs=_OneGraph(graph), sweep=True)
    folder = tmp_path / 'out/sweeps/photo/start=random/s0'
    assert sorted(p.name for p in folder.iterdir()) == ['run.json']          # D15: sweep rows keep no weights
    assert json.loads((folder / 'run.json').read_text())['row'] == 'start=random'
    assert out[('start=random', 0)].state is not None
