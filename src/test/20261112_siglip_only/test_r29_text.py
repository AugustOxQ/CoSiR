"""Tests of r29_text on small synthetic stores, towers and score files (no real data)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from unittest import mock

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_text as X  # noqa: E402
import r28_text as T  # noqa: E402
import r29_captions as C  # noqa: E402
import r29_model as M  # noqa: E402
import r29_paths as P29  # noqa: E402
import tcl_paths as P23  # noqa: E402
import tcl_data  # noqa: E402
import tcl_score  # noqa: E402
from tcl_metrics import fit_means  # noqa: E402

W = 1152
ASPECTS = list(P29.ASPECTS)
# 6 paintings, two captions each; extraction IDs and the store's row order deliberately unlike the frame's order
NAMES = ['p0', 'p1', 'p2', 'p3', 'p4', 'p5']
CAP_IDS = [11, 5, 8, 20, 2, 9, 14, 30, 1, 17, 25, 3]
CAP_PAINT = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
QWEN_HASH = 'ab' * 32                     # stands in for the Qwen store's index hash (the real one needs the CT's data)


def make_store(path, seed=0):
    """A store holding exactly CAP_IDS, in reversed row order."""
    rng = np.random.default_rng(seed)
    ids = CAP_IDS[::-1]
    names = [NAMES[CAP_PAINT[CAP_IDS.index(i)]] for i in ids]
    x = rng.normal(size=(len(ids), W)).astype(np.float16)
    path.mkdir(parents=True, exist_ok=True)
    texts = [dict(sample_id=i, painting=n, img=0) for i, n in zip(ids, names)]
    raw = json.dumps(dict(backbone='siglip_so400m', smoke=False, dim=W, n_texts=len(ids), texts=texts, images=[]))
    (path / 'index.json').write_text(raw)                      # sealed as scripts/extract_features.py does (ticket 01's reader)
    np.save(path / 'txt.npy', x)
    (path / 'progress.json').write_text(json.dumps(dict(txt_done=len(ids), gate=dict(ok=True),
                                                         index_sha256=hashlib.sha256(raw.encode()).hexdigest())))
    return dict(zip(ids, x.astype(np.float32)))


def frame(n_captions_per=2):
    """A tcl_data-style frame in sample-ID order (as r23's frames)."""
    order = np.argsort(CAP_IDS)
    ids = np.asarray(CAP_IDS)[order]
    tp = np.asarray(CAP_PAINT, dtype=np.int64)[order]
    rng = np.random.default_rng(1)
    labels = {a: dict(t=rng.integers(0, 3, len(NAMES))[tp].astype(np.int64), i=None) for a in ASPECTS}
    for a in ASPECTS:
        labels[a]['i'] = rng.integers(0, 3, len(NAMES)).astype(np.int64)
        labels[a]['t'] = labels[a]['i'][tp]
    return dict(n_i=len(NAMES), n_t=len(ids), tp=tp, ip=np.arange(len(NAMES), dtype=np.int64), labels=labels,
                names=list(NAMES), text=['t'] * len(ids), sample_id=ids)


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Synthetic store as 'siglip1' and 'siglip1_prefix', OUT and the text outputs in tmp_path."""
    rows = make_store(tmp_path / 'store')
    make_store(tmp_path / 'store_prefix', seed=3)
    monkeypatch.setitem(C.SOURCES, 'siglip1', (tmp_path / 'store', W))
    monkeypatch.setitem(C.SOURCES, 'siglip1_prefix', (tmp_path / 'store_prefix', W))
    monkeypatch.setattr(P29, 'OUT', tmp_path / 'out')
    monkeypatch.setattr(X, 'TEXT_OUT', tmp_path / 'text')
    monkeypatch.setattr(X, 'RECORDS_DIR', tmp_path / 'records')
    monkeypatch.setattr(X, 'SMOKE', True)                              # explicit smoke mode: no git check
    real_hash = C.store_hash
    monkeypatch.setattr(C, 'store_hash', lambda source: QWEN_HASH if source == 'qwen' else real_hash(source))
    return rows


# --- rows --------------------------------------------------------------------------------------------------------

def test_text_rows_join_by_sample_id_and_are_unit(world):
    fr = frame()
    got = X.text_rows('siglip1', fr)
    assert got.shape == (12, W) and got.dtype == np.float32
    for k, i in enumerate(fr['sample_id']):
        want = world[int(i)] / np.linalg.norm(world[int(i)])
        np.testing.assert_allclose(got[k], want, atol=1e-6)


def test_text_rows_wrong_painting_raises(world):
    fr = frame()
    fr['names'][0], fr['names'][1] = fr['names'][1], fr['names'][0]
    with pytest.raises(AssertionError, match='painting'):
        X.text_rows('siglip1', fr)


def test_p1_rows_are_the_stores_rows_and_the_image_rows(world, monkeypatch):
    fr = frame()
    img = np.random.default_rng(2).normal(size=(6, W)).astype(np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    monkeypatch.setattr(tcl_data, 'frame', lambda s, k: fr)
    monkeypatch.setattr(tcl_data, 'image', lambda s, r, aspect='emotion': img)
    text, image = X.rows('p1_siglip1', 'art_val', 'all', 'x')
    np.testing.assert_array_equal(text, X.text_rows('siglip1', fr))
    np.testing.assert_array_equal(image, img)


# --- towers ------------------------------------------------------------------------------------------------------

def save_weights(arm, aspect, seed, store_hash=None, source=None):
    torch.manual_seed(seed)
    _, views = M.build_model(ASPECTS, torch.device('cpu'), width=W, cap_width=W)
    w = X.weights_path(arm, aspect, seed)
    w.parent.mkdir(parents=True, exist_ok=True)
    torch.save(views[aspect].state_dict(), w)
    rb = dict(width=W, cap_width=W, cap_source=source or arm, cap_store_sha256=store_hash or C.store_hash(arm))
    w.with_suffix('.json').write_text(json.dumps(dict(config=dict(aspect=aspect, seed=seed, rb=rb))))
    return w


def test_towers_equal_r28s_side_on_the_same_weights(world):
    w = save_weights('siglip1', 'style', 2)
    image_side, caption_side = X.load_towers('siglip1', 'style', 2)
    # r28's own tower class and unit-normalisation, built from the same state
    _, Side = X.T._towers_module()
    state = torch.load(w, map_location='cpu', weights_only=True)
    ref = {}
    for side in ('img', 'cap'):
        m = Side(W)
        m.load_state_dict({k[len(side) + 1:]: v for k, v in state.items() if k.startswith(side + '.')})
        ref[side] = m.eval()
    x = np.random.default_rng(0).normal(size=(5, W)).astype(np.float32)
    xu = x / np.linalg.norm(x, axis=1, keepdims=True)
    for fn, side in ((image_side, 'img'), (caption_side, 'cap')):
        with torch.no_grad():
            want = ref[side](torch.as_tensor(xu)).numpy()
        np.testing.assert_allclose(fn(x), want, atol=1e-6)


def test_tower_refuses_a_sidecar_naming_another_store(world):
    save_weights('siglip1', 'emotion', 0, store_hash='cd' * 32)
    with pytest.raises(AssertionError, match='not the expected'):
        X.load_towers('siglip1', 'emotion', 0)


def test_tower_refuses_a_sidecar_naming_another_source(world):
    save_weights('siglip1_prefix', 'emotion', 0, source='siglip1')
    with pytest.raises(AssertionError, match='not the store'):
        X.load_towers('siglip1_prefix', 'emotion', 0)


def test_candidate_rows_run_the_towers_on_the_stores_rows(world, monkeypatch):
    save_weights('siglip1', 'genre', 1)
    fr = frame()
    img = np.random.default_rng(2).normal(size=(6, W)).astype(np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    monkeypatch.setattr(tcl_data, 'frame', lambda s, k: fr)
    monkeypatch.setattr(tcl_data, 'image', lambda s, r, aspect='emotion': img)
    text, image = X.rows('siglip1', 'art_val', 'genre', 1)
    image_side, caption_side = X.load_towers('siglip1', 'genre', 1)
    np.testing.assert_array_equal(text, caption_side(X.text_rows('siglip1', fr)))
    np.testing.assert_array_equal(image, image_side(img))
    assert text.shape == (12, 2048) and image.shape == (6, 2048)


# --- float16 rule ------------------------------------------------------------------------------------------------

def test_f16_check_one_step_and_larger():
    stored = np.linspace(-0.1, 0.1, 1000).astype(np.float16).reshape(10, 100)
    mine = stored.astype(np.float32)
    share, d32, one = X.f16_check(mine, stored)
    assert share == 1.0 and d32 == 0.0 and one
    bumped = stored.copy()
    bumped[0, 0] = np.nextafter(stored[0, 0], np.float16(1))
    share, _, one = X.f16_check(bumped.astype(np.float32), stored)
    assert share == pytest.approx(0.999) and one
    bumped[0, 1] = np.nextafter(np.nextafter(stored[0, 1], np.float16(1)), np.float16(1))
    assert not X.f16_check(bumped.astype(np.float32), stored)[2]


# --- score files, picks, ranges ---------------------------------------------------------------------------------

def write_scores(root, row, head, seed, values, n_q=4):
    """Per-query AP files of one unit: values[(direction, readout)] = the AP every scored query has (aspect column)."""
    for d in X.DIRECTIONS:
        for r in X.READOUTS:
            path = P23.score_path('art_val', X.KIND, row, head, seed, d, r, out=root)
            path.parent.mkdir(parents=True, exist_ok=True)
            ap = np.full(n_q, values[(d, r)], dtype=np.float32)
            np.savez(path, q_pid=np.arange(n_q, dtype=np.int64), first_rank=np.ones(n_q, np.float32),
                     **{f'ap__{a}': ap for a in ASPECTS})


def fill(arm, seeds, value):
    """value(aspect, seed, direction, readout) -> float, for every unit of the arm."""
    info = X.arm_info(arm)
    for head, seed in X.units(arm, seeds=seeds):
        write_scores(X.score_root(arm), info['row'], head, seed,
                     {(d, r): value(head, seed, d, r) for d in X.DIRECTIONS for r in X.READOUTS})
        if info['kind'] == 'candidate':                          # the weights the scores came from (only their bytes matter here)
            w = X.weights_path(arm, head, seed)
            w.parent.mkdir(parents=True, exist_ok=True)
            w.write_bytes(f'{arm}/{head}/{seed}'.encode())
        X.write_provenance(arm, 'art_val', head, seed)


def test_seed0_only_record_has_per_seed_values_and_no_picks(world):
    fill('siglip1', [0], lambda a, s, d, r: 0.2 + 0.01 * X.READOUTS.index(r))
    doc = X.write_val_record('siglip1')
    assert doc['seeds'] == [0] and 'picks' not in doc
    assert doc['val']['cells']['emotion']['t2i']['per_seed']['center'] == [pytest.approx(0.21)]
    assert doc['val']['group_r0']['per_seed'] == [pytest.approx(0.2)]
    assert doc['store_sha256'] == C.store_hash('siglip1')
    assert json.loads(X.record_path('siglip1').read_text())['seeds'] == [0]
    with pytest.raises(AssertionError, match='picks need seeds'):
        X.picks('siglip1')


def test_a_seed_missing_one_aspect_does_not_count(world):
    fill('siglip1', [0], lambda a, s, d, r: 0.2)
    X.write_val_record('siglip1')
    write_scores(X.score_root('siglip1'), 'cstar', 'emotion', 1, {(d, r): 0.3 for d in X.DIRECTIONS for r in X.READOUTS})
    assert X.seeds_done('siglip1') == [0]


def test_picks_highest_mean_ties_to_first_and_range_at_the_pick(world):
    def value(a, s, d, r):
        if (a, d) == ('emotion', 't2i'):        # center_k10 has the highest five-seed mean
            return {'r0': 0.20, 'center': 0.21, 'center_k10': 0.22}[r] + 0.001 * s
        if (a, d) == ('emotion', 'i2t'):        # exact tie r0 = center, both above center_k10: the first (r0)
            return {'r0': 0.30, 'center': 0.30, 'center_k10': 0.29}[r]
        if (a, d) == ('style', 't2t'):          # a seed with a high r0 only: mean decides, not the best seed
            return 0.5 if (r == 'r0' and s == 4) else 0.2 + 0.001 * (r == 'center')
        return 0.1
    fill('siglip1', range(5), value)
    doc = X.picks('siglip1')
    pk = doc['picks']
    assert pk['emotion']['t2i']['pick'] == 'center_k10'
    assert pk['emotion']['t2i']['range'] == pytest.approx(0.004, abs=1e-6)           # max - min of the per-seed values at the pick
    assert pk['emotion']['t2i']['mean'] == pytest.approx(0.222, abs=1e-6)
    assert pk['emotion']['i2t']['pick'] == 'r0'                              # tie to the first read-out
    assert pk['style']['t2t']['pick'] == 'r0'                                # 0.26 beats 0.2 and 0.201
    assert pk['genre']['t2i']['pick'] == 'r0' and pk['genre']['t2i']['range'] == 0.0
    assert doc['val']['group_r0']['mean'] == pytest.approx(np.mean(doc['val']['group_r0']['per_seed']))


def test_p1_record_and_picks(world, monkeypatch, tmp_path):
    stage1 = tmp_path / 'stage1_art.json'                     # r23's committed picks, here all `center`
    stage1.write_text(json.dumps(dict(picks={X.KIND: {'siglip_so400m': {a: {d: 'center' for d in X.DIRECTIONS} for a in ASPECTS}}})))
    monkeypatch.setattr(X.T, 'R23_STAGE1', stage1)
    fill('p1_siglip1', [0], lambda a, s, d, r: {'r0': 0.1, 'center': 0.2, 'center_k10': 0.15}[r])
    doc = X.picks('p1_siglip1')
    assert doc['seeds'] == ['x'] and doc['kind'] == 'p1'
    assert doc['picks']['emotion']['t2i'] == dict(pick='center', mean=pytest.approx(0.2), range=None)
    assert doc['val']['group_r0']['mean'] == pytest.approx(0.1)


# --- the order rule ----------------------------------------------------------------------------------------------

def test_held_refuses_without_a_pick_record_and_with_one_without_picks(world):
    with pytest.raises(RuntimeError, match='no pick record'):
        X.score('siglip1', 'art_held')
    fill('siglip1', [0], lambda a, s, d, r: 0.2)
    X.write_val_record('siglip1')                                  # a record, but no picks
    with pytest.raises(RuntimeError, match='no picks'):
        X.score('siglip1', 'art_held')
    with pytest.raises(RuntimeError, match='no picks'):
        X.held_record('siglip1')


def test_held_refuses_when_the_store_changed_after_the_picks(world, tmp_path):
    fill('siglip1', range(5), lambda a, s, d, r: 0.2)
    X.picks('siglip1')
    X.require_pick_record('siglip1')                               # fine
    idx = tmp_path / 'store' / 'index.json'
    doc = json.loads(idx.read_text())
    doc['note'] = 'changed'
    idx.write_text(json.dumps(doc))
    with pytest.raises(AssertionError, match='store changed'):
        X.require_pick_record('siglip1')


def git(repo, *args):
    subprocess.run(['git', '-C', str(repo), '-c', 'user.name=t', '-c', 'user.email=t@t', *args], check=True, capture_output=True)


@pytest.fixture
def repo_world(world, monkeypatch, tmp_path):
    """Not smoke: records in a throw-away git repo that stands in for the project (P29.REPO)."""
    repo = tmp_path / 'repo'
    (repo / 'records').mkdir(parents=True)
    git(repo, 'init', '-q')
    monkeypatch.setattr(P29, 'REPO', repo)
    monkeypatch.setattr(X, 'RECORDS_DIR', repo / 'records')
    monkeypatch.setattr(X, 'SMOKE', False)
    fill('siglip1', range(5), lambda a, s, d, r: 0.2)
    X.picks('siglip1')
    return repo


def test_the_pick_record_must_be_committed_unchanged_outside_smoke(repo_world):
    repo = repo_world
    with pytest.raises(RuntimeError, match='not committed'):             # never committed
        X.require_pick_record('siglip1')
    git(repo, 'add', 'records')
    git(repo, 'commit', '-q', '-m', 'picks')
    assert X.require_pick_record('siglip1')['picks']                       # committed unchanged: fine
    path = X.record_path('siglip1')
    doc = json.loads(path.read_text())
    doc['picks']['emotion']['t2i']['pick'] = 'center'                     # edited after the commit
    path.write_text(json.dumps(doc, indent=1) + '\n')
    with pytest.raises(RuntimeError, match='not committed unchanged'):
        X.require_pick_record('siglip1')


def test_the_pick_record_seeds_must_be_the_arms_five_seeds(world):
    fill('siglip1', range(5), lambda a, s, d, r: 0.2)
    X.picks('siglip1')
    assert json.loads(X.record_path('siglip1').read_text())['seeds'] == [0, 1, 2, 3, 4]    # ticket 05 asserts seeds[0] == 0
    path = X.record_path('siglip1')
    doc = json.loads(path.read_text())
    doc['seeds'] = [0, 1, 2, 3]
    path.write_text(json.dumps(doc))
    with pytest.raises(AssertionError, match='the pick needs'):
        X.require_pick_record('siglip1')


def test_the_pick_record_must_name_the_weights_as_they_are_now(world):
    fill('siglip1', range(5), lambda a, s, d, r: 0.2)
    X.picks('siglip1')
    X.require_pick_record('siglip1')
    X.weights_path('siglip1', 'genre', 3).write_bytes(b'retrained')
    with pytest.raises(AssertionError, match='weights'):
        X.require_pick_record('siglip1')


def test_smoke_redirect_needs_the_explicit_flag(monkeypatch):
    """R29_TEXT_OUT alone does not switch the git check off: the module refuses to load."""
    import importlib
    monkeypatch.setenv('R29_TEXT_OUT', '/tmp/somewhere')
    monkeypatch.delenv('R29_TEXT_SMOKE', raising=False)
    with pytest.raises(AssertionError, match='R29_TEXT_SMOKE'):
        importlib.reload(X)
    monkeypatch.delenv('R29_TEXT_OUT')
    importlib.reload(X)
    assert X.SMOKE is False and Path(X.RECORDS_DIR).resolve() == P29.RECORDS.resolve()
    monkeypatch.setenv('R29_TEXT_OUT', '/tmp/somewhere')
    monkeypatch.setenv('R29_TEXT_SMOKE', '1')
    importlib.reload(X)
    assert X.SMOKE is True and Path(X.RECORDS_DIR) == Path('/tmp/somewhere/records')
    monkeypatch.delenv('R29_TEXT_OUT')
    monkeypatch.delenv('R29_TEXT_SMOKE')
    importlib.reload(X)


# --- provenance: scores belong to the weights and store they were made from ---------------------------------------

def test_a_scored_unit_is_not_reused_after_its_weights_change(world, monkeypatch):
    fr = frame()
    rng = np.random.default_rng(5)

    def fake_rows(arm, set_name, head, seed, device='cpu'):
        t, i = (rng.normal(size=(n, 2048)).astype(np.float32) for n in (fr['n_t'], fr['n_i']))
        return t / np.linalg.norm(t, axis=1, keepdims=True), i / np.linalg.norm(i, axis=1, keepdims=True)
    monkeypatch.setattr(tcl_data, 'frame', lambda s, k: fr)
    monkeypatch.setattr(tcl_data, 'names_label', lambda s, k, a: np.zeros(fr['n_t'], dtype=bool))
    monkeypatch.setattr(X, 'rows', fake_rows)
    w = X.weights_path('siglip1', 'emotion', 3)
    w.parent.mkdir(parents=True, exist_ok=True)
    w.write_bytes(b'first training')
    X.score('siglip1', 'art_val', aspects=('emotion',), seeds=[3], block=8)
    prov = json.loads((X.score_root('siglip1') / 'provenance/art_val/emotion__s3.json').read_text())
    assert prov == dict(weights_sha256=hashlib.sha256(b'first training').hexdigest(), store_sha256=C.store_hash('siglip1'))
    X.score('siglip1', 'art_val', aspects=('emotion',), seeds=[3], block=8)            # unchanged: skipped, fine
    w.write_bytes(b'retrained into the same folder')
    with pytest.raises(AssertionError, match='weights'):
        X.score('siglip1', 'art_val', aspects=('emotion',), seeds=[3], block=8)
    w.write_bytes(b'first training')
    idx = Path(C.SOURCES['siglip1'][0]) / 'index.json'
    idx.write_text(idx.read_text() + ' ')                                              # the store's index bytes changed
    with pytest.raises(AssertionError, match='store'):
        X.score('siglip1', 'art_val', aspects=('emotion',), seeds=[3], block=8)


def test_a_complete_unit_without_provenance_is_refused(world, monkeypatch):
    fill('siglip1', [0], lambda a, s, d, r: 0.2)
    (X.score_root('siglip1') / 'provenance/art_val/style__s0.json').unlink()
    with pytest.raises(AssertionError, match='provenance'):
        X.write_val_record('siglip1')


def test_the_val_record_refuses_scores_of_other_weights(world):
    fill('siglip1', range(5), lambda a, s, d, r: 0.2)
    X.write_val_record('siglip1')
    X.weights_path('siglip1', 'style', 2).write_bytes(b'retrained')
    with pytest.raises(AssertionError, match='weights'):
        X.write_val_record('siglip1')


def test_the_val_record_lists_each_units_provenance(world):
    fill('siglip1', [0], lambda a, s, d, r: 0.2)
    doc = X.write_val_record('siglip1')
    assert sorted(doc['provenance']) == [f'{a}__s0' for a in sorted(ASPECTS)]
    assert doc['provenance']['style__s0']['weights_sha256'] == hashlib.sha256(b'siglip1/style/0').hexdigest()


# --- P1 picks against r23's committed ones ------------------------------------------------------------------------

def test_p1_picks_must_equal_r23s_committed_picks(world, monkeypatch, tmp_path):
    fill('p1_siglip1', [0], lambda a, s, d, r: {'r0': 0.1, 'center': 0.2, 'center_k10': 0.15}[r])
    stage1 = tmp_path / 'stage1_art.json'
    picks = {a: {d: 'center' for d in X.DIRECTIONS} for a in ASPECTS}
    stage1.write_text(json.dumps(dict(picks={X.KIND: {'siglip_so400m': picks}})))
    monkeypatch.setattr(X.T, 'R23_STAGE1', stage1)
    assert X.picks('p1_siglip1')['picks']['style']['i2t']['pick'] == 'center'
    picks['style']['i2t'] = 'r0'
    stage1.write_text(json.dumps(dict(picks={X.KIND: {'siglip_so400m': picks}})))
    with pytest.raises(AssertionError, match='r23'):
        X.picks('p1_siglip1')


# --- the arm for gate (i)(a)'s new trainer run (diagnostic (e)) -------------------------------------------------------

class Reached(Exception):
    """Raised by a stub placed after the checks: the call got past the pass-rule refusal."""


def pass_rule_path():
    return Path(X.RECORDS_DIR) / 'pass_rule.json'


@pytest.mark.parametrize('arm', ['siglip1', 'siglip1_prefix'])
@pytest.mark.parametrize('call', ['score_held', 'cells'])
def test_candidate_test_numbers_refuse_without_a_committed_pass_rule(repo_world, monkeypatch, arm, call):
    """`score --set art_held` and `cells` of a candidate arm: no pass_rule.json, an uncommitted one, a modified one are
    refusals before the pick record is even read; a committed one (pass or fail: D9) lets the call through."""
    repo = repo_world
    monkeypatch.setattr(X, 'require_pick_record', mock.Mock(side_effect=Reached))     # the next check after the pass rule
    run = X.held_record if call == 'cells' else lambda a: X.score(a, 'art_held')
    with pytest.raises(RuntimeError, match='no pass_rule.json'):
        run(arm)                                                      # absent
    pass_rule_path().write_text('{"status": "fail"}\n')
    with pytest.raises(RuntimeError, match='pass_rule.json is not committed'):
        run(arm)                                                      # present, not committed
    git(repo, 'add', 'records')
    git(repo, 'commit', '-q', '-m', 'rule')
    with pytest.raises(Reached):
        run(arm)                                                      # committed unchanged, status fail: through
    pass_rule_path().write_text('{"status": "pass"}\n')
    with pytest.raises(RuntimeError, match='pass_rule.json is not committed'):
        run(arm)                                                      # modified since the commit


@pytest.mark.parametrize('arm', ['qwen', 'gate_i_a_new', 'p1_siglip1'])
def test_k0_and_gate_paths_run_without_a_pass_rule(repo_world, monkeypatch, arm):
    """K0 (qwen) and the gate (i)(a) arm are replays, not candidate test numbers: held scoring of them goes past the pass
    rule with none on record (it stops at the next check, the pick record, here a stub)."""
    assert not pass_rule_path().exists()
    monkeypatch.setattr(X, 'require_pick_record', mock.Mock(side_effect=Reached))
    with pytest.raises(Reached):
        X.score(arm, 'art_held')
    with pytest.raises(Reached):
        X.held_record(arm)


def test_gate_ii_runs_without_a_pass_rule(repo_world, monkeypatch):
    """Gate (ii) reads K0's validation scores only and never asks for pass_rule.json."""
    assert not pass_rule_path().exists()
    monkeypatch.setattr(P29, 'require_committed', mock.Mock(side_effect=AssertionError('asked for a committed record')))
    monkeypatch.setattr(X, 'seeds_done', lambda arm: list(X.SEEDS))
    monkeypatch.setattr(X, 'check_provenance', mock.Mock())
    monkeypatch.setattr(T, 'validation_picks', mock.Mock(side_effect=Reached))
    with pytest.raises(Reached):
        X.gate_ii(replay_too=False)


def test_gate_i_a_new_arm_scores_k0s_protocol_on_the_new_trainers_weights(world, monkeypatch):
    info = X.arm_info('gate_i_a_new')
    assert info['kind'] == 'candidate' and info['source'] == 'qwen' and info['row'] == 'cstar' and tuple(info['seeds']) == (0,)
    assert X.weights_path('gate_i_a_new', 'style', 0) == P29.OUT / 'gate_i_a/new/weights/qwen/style__s0.pt'
    assert X.units('gate_i_a_new') == [(a, 0) for a in ASPECTS]
    fill('gate_i_a_new', [0], lambda a, s, d, r: 0.2)
    assert X.seeds_done('gate_i_a_new') == [0]                                   # one seed is the whole arm
    assert X.write_val_record('gate_i_a_new')['val']['group_r0']['per_seed'] == [pytest.approx(0.2)]
    # towers and rows: the caption tower on Qwen3's plain rows (as r28), built from the sidecar of the new trainer
    fake_hash = QWEN_HASH
    torch.manual_seed(0)
    _, views = M.build_model(ASPECTS, torch.device('cpu'), width=W, cap_width=2048)
    w = X.weights_path('gate_i_a_new', 'genre', 0)
    torch.save(views['genre'].state_dict(), w)
    w.with_suffix('.json').write_text(json.dumps(dict(config=dict(aspect='genre', seed=0, rb=dict(
        width=W, cap_width=2048, cap_source='qwen', cap_store_sha256=fake_hash)))))
    fr = frame()
    img = np.random.default_rng(2).normal(size=(6, W)).astype(np.float32)
    plain = np.random.default_rng(3).normal(size=(12, 2048)).astype(np.float32)
    monkeypatch.setattr(tcl_data, 'frame', lambda s, k: fr)
    monkeypatch.setattr(tcl_data, 'image', lambda s, r, aspect='emotion': img)
    monkeypatch.setattr(tcl_data, 'plain_text', lambda s, k: plain)
    text, image = X.rows('gate_i_a_new', 'art_val', 'genre', 0)
    image_side, caption_side = X.load_towers('gate_i_a_new', 'genre', 0)
    np.testing.assert_array_equal(text, caption_side(plain))
    np.testing.assert_array_equal(image, image_side(img))


# --- end to end on tiny frames: means refit on validation, held reads them -------------------------------------------

def test_score_refits_means_on_validation_rows_and_held_reads_them(world, monkeypatch):
    fr = frame()
    rng = np.random.default_rng(5)

    def unit_rows(n):
        x = rng.normal(size=(n, 2048)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    made = {}

    def fake_rows(arm, set_name, head, seed, device='cpu'):
        made[(set_name, head, seed)] = (unit_rows(fr['n_t']), unit_rows(fr['n_i']))
        return made[(set_name, head, seed)]
    monkeypatch.setattr(tcl_data, 'frame', lambda s, k: fr)
    monkeypatch.setattr(tcl_data, 'names_label', lambda s, k, a: np.zeros(fr['n_t'], dtype=bool))
    monkeypatch.setattr(X, 'rows', fake_rows)
    w = X.weights_path('siglip1', 'emotion', 3)
    w.parent.mkdir(parents=True, exist_ok=True)
    w.write_bytes(b'weights')
    X.score('siglip1', 'art_val', aspects=('emotion',), seeds=[3], block=8)
    means = np.load(P23.means_path('art_val', X.KIND, 'cstar', 'emotion', 3, out=X.score_root('siglip1')))
    t, i = made[('art_val', 'emotion', 3)]
    want_t, want_i = fit_means(t, i)
    np.testing.assert_array_equal(means['t'], want_t)
    np.testing.assert_array_equal(means['i'], want_i)
    for d in X.DIRECTIONS:
        for r in X.READOUTS:
            assert P23.score_path('art_val', X.KIND, 'cstar', 'emotion', 3, d, r, out=X.score_root('siglip1')).exists()
    assert X.seeds_done('siglip1') == []                              # style and genre not scored: no record, no seed counts
    assert not X.record_path('siglip1').exists()


def test_held_cells_at_the_recorded_pick(world, monkeypatch):
    """Held files exist for the picked read-out only; `held_record` reads exactly those."""
    def value(a, s, d, r):
        return {'r0': 0.1, 'center': 0.3, 'center_k10': 0.2}[r]
    fill('siglip1', range(5), value)
    X.picks('siglip1')
    root = X.score_root('siglip1')
    for a in ASPECTS:
        for s in range(5):
            for d in X.DIRECTIONS:
                path = P23.score_path('art_held', X.KIND, 'cstar', a, s, d, 'center', out=root)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, q_pid=np.arange(3, dtype=np.int64), first_rank=np.ones(3, np.float32),
                         **{f'ap__{b}': np.full(3, 0.5 + 0.01 * s, dtype=np.float32) for b in ASPECTS})
    for a in ASPECTS:
        for s in range(5):
            X.write_provenance('siglip1', 'art_held', a, s)
    out = X.held_record('siglip1')
    cell = out['cells']['emotion']['t2i']
    assert cell['readout'] == 'center' and cell['mean'] == pytest.approx(0.52) and cell['n_queries'] == 3
    assert json.loads(X.record_path('siglip1', held=True).read_text())['picks']['genre']['t2t'] == 'center'


# --- hash check --------------------------------------------------------------------------------------------------

def make_trainer_records(arm, hash_, seeds=(0,), where='top'):
    info = X.arm_info(arm)
    root = Path(info['root'])
    (root / 'train' / arm).mkdir(parents=True, exist_ok=True)
    (root / 'weights' / arm).mkdir(parents=True, exist_ok=True)
    for s in seeds:
        run = dict(seed=s, cap_store_sha256=hash_) if where == 'top' else dict(seed=s, config=dict(rb=dict(cap_store_sha256=hash_)))
        (root / 'train' / arm / f'run__s{s}.json').write_text(json.dumps(run))
        for a in ASPECTS:
            (root / 'weights' / arm / f'{a}__s{s}.json').write_text(json.dumps(dict(config=dict(rb=dict(cap_store_sha256=hash_)))))


def test_hash_check_writes_its_own_part_file(world):
    fill('siglip1', range(5), lambda a, s, d, r: 0.2)
    X.write_val_record('siglip1')
    make_trainer_records('siglip1', C.store_hash('siglip1'), seeds=(0, 1))
    rec = X.hash_check(('siglip1',))
    assert rec['gate'] == 'iv_hash' and rec['status'] == 'pass' and rec['tries'] == 1 and rec['written']
    assert set(rec) == {'gate', 'status', 'tries', 'details', 'written'}
    assert rec['details']['arms']['siglip1']['n_run_records'] == 2
    gates = Path(X.RECORDS_DIR) / 'gates'
    assert sorted(p.name for p in gates.iterdir()) == ['iv_hash.json']             # ticket 02's iv_store.json is not touched
    assert json.loads((gates / 'iv_hash.json').read_text()) == rec
    make_trainer_records('siglip1', 'ee' * 32, seeds=(2,), where='config')               # a run trained on another store
    rec = X.hash_check(('siglip1',))
    assert rec['status'] == 'fail' and rec['tries'] == 2


def test_hash_check_fails_without_a_scorer_record_or_trainer_records(world):
    rec = X.hash_check(('siglip1',))
    assert rec['status'] == 'fail'
    assert rec['details']['arms']['siglip1']['scorer_record'] is None


# --- gate (ii): K0 against r28's records -------------------------------------------------------------------------

def k0_values(shift=0.0, picks=None):
    """What `validation_picks` returns, for all cells: per-seed values at every read-out, and the pick."""
    out = {}
    for a in ASPECTS:
        out[a] = {}
        for d in X.DIRECTIONS:
            per_seed = {r: [0.2 + 0.01 * k + 0.001 * s + shift for s in range(5)] for k, r in enumerate(X.READOUTS)}
            out[a][d] = dict(per_seed=per_seed, means={r: float(np.mean(v)) for r, v in per_seed.items()},
                             pick=(picks or {}).get((a, d), 'center_k10'))
    return out


def test_gate_ii_passes_on_equal_values_and_picks_and_reports_the_worst_gap():
    details = X.compare_k0(k0_values(), k0_values())
    assert details['ok'] and details['mismatched'] == [] and details['picks_equal'] and details['max_abs_diff_per_seed_map'] == 0.0
    assert details['n_values'] == 3 * 3 * 3 * 5


def test_gate_ii_tolerance_is_one_in_a_million():
    inside = k0_values(shift=5e-7)
    assert X.compare_k0(inside, k0_values())['ok']
    off = k0_values()
    off['style']['t2i']['per_seed']['center'][3] += 2e-6                            # one value off by 2e-6
    details = X.compare_k0(off, k0_values())
    assert not details['ok'] and details['mismatched'] == ['style/t2i/center']
    assert details['max_abs_diff_per_seed_map'] == pytest.approx(2e-6, rel=1e-3)


def test_gate_ii_fails_on_one_changed_pick():
    details = X.compare_k0(k0_values(picks={('genre', 'i2t'): 'r0'}), k0_values())
    assert not details['ok'] and not details['picks_equal'] and details['mismatched'] == []


def test_gate_ii_a_nan_value_is_a_failure_not_a_pass():
    nan = k0_values()
    nan['emotion']['t2t']['per_seed']['r0'][2] = float('nan')
    with pytest.raises(AssertionError, match='finite'):
        X.compare_k0(nan, k0_values())
    with pytest.raises(AssertionError, match='finite'):
        X.compare_k0(k0_values(), nan)


def test_gate_ii_needs_the_same_number_of_seeds_on_both_sides():
    short = k0_values()
    short['emotion']['t2t']['per_seed']['r0'] = short['emotion']['t2t']['per_seed']['r0'][:4]
    with pytest.raises(AssertionError, match='seeds'):
        X.compare_k0(short, k0_values())


def test_replay_rule_needs_99_5_percent_equal_and_one_float16_step():
    assert X.replay_passed(0.998, True)
    assert not X.replay_passed(0.99, True)             # r28's rule: at least 99.5% of the elements equal
    assert not X.replay_passed(0.999, False)


# --- the gate writers count their tries through r29_captions.write_gate_record (final review fix A, ruling 3) -----------

@pytest.mark.parametrize('name', ['ii', 'iv_hash'])
def test_a_third_failure_parks_and_a_fourth_try_raises(world, name):
    got = [X.write_gate(name, 'fail', dict(n=i)) for i in range(3)]
    assert [(r['status'], r['tries']) for r in got] == [('fail', 1), ('fail', 2), ('parked', 3)]
    assert set(got[2]) == {'gate', 'status', 'tries', 'details', 'written'}
    with pytest.raises(AssertionError, match='fourth'):
        X.write_gate(name, 'fail', dict(n=3))
    assert json.loads((Path(X.RECORDS_DIR) / 'gates' / f'{name}.json').read_text())['tries'] == 3


def test_a_pass_after_two_failures_stays_a_pass(world):
    X.write_gate('ii', 'fail', {}); X.write_gate('ii', 'fail', {})
    assert X.write_gate('ii', 'pass', {})['status'] == 'pass'
