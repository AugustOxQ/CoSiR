"""Tests of r29_text on small synthetic stores, towers and score files (no real data)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r29_text as X  # noqa: E402
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


def make_store(path, seed=0):
    """A store holding exactly CAP_IDS, in reversed row order."""
    rng = np.random.default_rng(seed)
    ids = CAP_IDS[::-1]
    names = [NAMES[CAP_PAINT[CAP_IDS.index(i)]] for i in ids]
    x = rng.normal(size=(len(ids), W)).astype(np.float16)
    path.mkdir(parents=True, exist_ok=True)
    texts = [dict(sample_id=i, painting=n, img=0) for i, n in zip(ids, names)]
    (path / 'index.json').write_text(json.dumps(dict(dim=W, n_texts=len(ids), texts=texts, images=[])))
    np.save(path / 'txt.npy', x)
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
    C._read.cache_clear()
    monkeypatch.setattr(P29, 'OUT', tmp_path / 'out')
    monkeypatch.setattr(X, 'TEXT_OUT', tmp_path / 'text')
    monkeypatch.setattr(X, 'RECORDS_DIR', tmp_path / 'records')       # not P29.RECORDS: smoke, no git check
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
    with pytest.raises(AssertionError, match='not the store'):
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


def test_p1_record_and_picks(world):
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


def test_the_pick_record_must_be_committed_outside_smoke(world, monkeypatch, tmp_path):
    fill('siglip1', range(5), lambda a, s, d, r: 0.2)
    X.picks('siglip1')
    monkeypatch.setattr(X, 'RECORDS_DIR', P29.RECORDS)               # not smoke now: the git check applies
    rec = P29.RECORDS / 'text_siglip1.json'
    if rec.exists():
        pytest.skip('a real record exists')
    rec.write_text((tmp_path / 'records' / 'text_siglip1.json').read_text())
    try:
        with pytest.raises(RuntimeError, match='not committed'):
            X.require_pick_record('siglip1')
    finally:
        rec.unlink()


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


def test_hash_check_pass_and_fail_and_merge_into_the_gate_record(world):
    fill('siglip1', range(5), lambda a, s, d, r: 0.2)
    X.write_val_record('siglip1')
    make_trainer_records('siglip1', C.store_hash('siglip1'), seeds=(0, 1))
    gate = Path(X.RECORDS_DIR) / 'gates' / 'iv.json'
    gate.parent.mkdir(parents=True, exist_ok=True)
    gate.write_text(json.dumps(dict(gate='iv', status='pass', tries=1, details=dict(store_part=dict(status='pass', n=3)),
                                    written='x')))
    rec = X.hash_check(('siglip1',))
    assert rec['status'] == 'pass' and rec['tries'] == 2
    assert rec['details']['store_part'] == dict(status='pass', n=3)                       # the other ticket's part is kept
    assert rec['details']['parts_present'] == ['hash_check', 'store_part']
    assert rec['details']['hash_check']['arms']['siglip1']['n_run_records'] == 2
    make_trainer_records('siglip1', 'ee' * 32, seeds=(2,), where='config')               # a run trained on another store
    rec = X.hash_check(('siglip1',))
    assert rec['status'] == 'fail' and rec['details']['hash_check']['status'] == 'fail'


def test_hash_check_fails_without_a_scorer_record_or_trainer_records(world):
    rec = X.hash_check(('siglip1',))
    assert rec['status'] == 'fail'
    assert rec['details']['hash_check']['arms']['siglip1']['scorer_record'] is None
