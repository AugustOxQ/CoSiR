"""r28 job D tests: the join, the replay, the pick and the cell of the text-direction scorer (`r28_text.py`).

Real-data tests read r23's committed score files and r14's exports (read only) on CPU. Run from anywhere:
    python -m pytest -q -p no:cacheprovider src/test/20261112_siglip_main/test_r28_text.py
"""
import sys
from math import erf, sqrt
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r28_text as T  # noqa: E402
import tcl_paths as P  # noqa: E402,F401
import tcl_data  # noqa: E402
import tx_data  # noqa: E402


# ------------------------------------------------------------------------------------------------ the pick
def test_pick_takes_highest_seed_mean_and_ties_go_to_the_first_readout():
    per = {'r0': [0.2, 0.2], 'center': [0.3, 0.1], 'center_k10': [0.1, 0.3]}     # center and center_k10 tie at 0.2
    picked, means = T.pick_readout(per)
    assert picked == 'r0' and means['center'] == means['center_k10'] == pytest.approx(0.2)
    per = {'r0': [0.1, 0.1], 'center': [0.3, 0.1], 'center_k10': [0.31, 0.3]}
    assert T.pick_readout(per)[0] == 'center_k10'
    with pytest.raises(AssertionError):
        T.pick_readout({'r0': [0.1], 'center': [0.2]})


def test_seed_mean_ap_skips_unscored_queries():
    assert T.seed_mean_ap(np.array([0.5, np.nan, 0.25], dtype=np.float32)) == pytest.approx(0.375)


# ------------------------------------------------------------------------------------------------ the join
def test_join_by_id_follows_ids_not_positions():
    stored = [30, 10, 20, 40]
    assert T.join_by_id(stored, [10, 40, 30]).tolist() == [1, 3, 0]
    assert T.join_by_id(stored, []).tolist() == []
    with pytest.raises(AssertionError):
        T.join_by_id(stored, [10, 99])
    with pytest.raises(AssertionError):
        T.join_by_id([1, 2, 2], [1])


@pytest.fixture(scope='module')
def held_text():
    f = tcl_data.frame('art_held', 'caption')
    return f, tcl_data.plain_text('art_held', 'caption')


def test_held_caption_rows_are_the_qwen_plain_rows_of_their_sample_ids(held_text):
    """The caption side on test starts from the plain Qwen3 embedding of each test caption: rebuilt independently here from
    the store's index (sample id -> row), the painting name and the float16 row."""
    f, text = held_text
    path = tcl_data.FULL / tx_data.STORES['plain']
    index = tx_data.read_json(path / 'index.json')
    where = {r['sample_id']: i for i, r in enumerate(index['texts'])}
    store = np.load(path / 'txt.npy', mmap_mode='r', allow_pickle=False)
    assert text.shape == (f['n_t'], 2048) and f['n_t'] == 61744
    assert (np.diff(f['sample_id']) > 0).all()
    rng = np.random.default_rng(0)
    for k in rng.choice(f['n_t'], size=300, replace=False):
        row = where[int(f['sample_id'][k])]
        assert index['texts'][row]['painting'] == f['names'][f['tp'][k]]
        ref = np.asarray(store[row], dtype=np.float32)
        ref = ref / np.linalg.norm(ref)
        assert np.allclose(text[k], ref, atol=1e-6)
    # every held painting has a caption and the image rows are in the same painting order
    assert set(np.unique(f['tp']).tolist()) == set(range(f['n_i']))


# ------------------------------------------------------------------------------------------------ the towers and the replay
def _numpy_tower(state, side, x):
    """Independent forward of one tower from the checkpoint: unit input, Linear, exact GELU, Linear, unit output."""
    w1, b1 = state[f'{side}.trunk.0.weight'].numpy(), state[f'{side}.trunk.0.bias'].numpy()
    w2, b2 = state[f'{side}.output.weight'].numpy(), state[f'{side}.output.bias'].numpy()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    h = x @ w1.T + b1
    h = 0.5 * h * (1 + np.vectorize(erf)(h / sqrt(2)))
    y = h @ w2.T + b2
    return y / np.linalg.norm(y, axis=1, keepdims=True)


def test_towers_equal_an_independent_numpy_forward_and_have_the_right_widths():
    import torch
    image_side, caption_side = T.load_sides('style', 1, 'cpu')
    state = torch.load(T.R14_SIGLIP1 / 'weights/k11=40/style__s1.pt', map_location='cpu', weights_only=True)
    rng = np.random.default_rng(1)
    xi = rng.normal(size=(20, 1152)).astype(np.float32)
    xc = rng.normal(size=(20, 2048)).astype(np.float32)
    assert np.allclose(image_side(xi), _numpy_tower(state, 'img', xi), atol=2e-5)
    assert np.allclose(caption_side(xc), _numpy_tower(state, 'cap', xc), atol=2e-5)
    with pytest.raises(AssertionError):
        image_side(xc)                      # the image tower reads 1152, not 2048
    with pytest.raises(AssertionError):
        caption_side(xi)


def test_replay_caption_tower_on_validation_means_equals_the_r14_export():
    """The inference path (plain Qwen3 caption means -> caption tower) against r14's stored validation caption side, on
    one head and seed, on CPU. The stores are float16: after casting, at least 99.5% of entries are equal and the rest differ
    by one float16 step; the cosine to the stored row is 1 within 1e-6."""
    import rt_data
    n = P.N_PAINTINGS['art_val']
    captions = tx_data.val_captions()
    means = tx_data.caption_mean(tx_data.text_rows('plain', raw=True), captions['pid'], n)
    _, caption_side = T.load_sides('emotion', 0, 'cpu')
    stored, _ = T._val_export('emotion', 0)
    mine = caption_side(means)
    assert stored.shape == mine.shape == (n, 2048)
    d32, d16, equal = T.f16_diff(mine, stored)
    assert equal > 0.995 and d32 < 1.3e-4 and d16 <= 1.3e-4, (d32, d16, equal)
    cos = (mine * (stored.astype(np.float32) / np.linalg.norm(stored.astype(np.float32), axis=1, keepdims=True))).sum(1)
    assert cos.min() > 1 - 1e-6, cos.min()
    assert len(rt_data.names('val')) == n


def test_replay_image_tower_on_held_images_equals_the_r14_export():
    image_side, _ = T.load_sides('genre', 2, 'cpu')
    x = tcl_data.image('art_held', 'siglip_so400m')
    stored = T._held_export('genre', 2)
    assert x.shape == (12281, 1152) and stored.shape == (12281, 2048)
    d32, d16, equal = T.f16_diff(image_side(x[:2000]), stored[:2000])
    assert equal > 0.995 and d32 < 1.3e-4, (d32, d16, equal)


# ------------------------------------------------------------------------------------------------ the cells on r23's files
def test_cells_code_reproduces_r23_picks_and_the_printed_qwen3_text_rows():
    """The pick and cell code, run on r23's Qwen3 score files, gives r23's committed stage 1 picks and Table 3's printed text
    rows (plain, instructed, ours) to three places."""
    import json
    stage1 = json.loads(T.R23_STAGE1.read_text())
    paper = T.paper_text_rows()
    assert paper['t2i']['cstar'] == [0.264, 0.172, 0.484] and paper['t2t']['teacher'] == [0.377, 0.098, 0.299]
    for row, seeds_of in (('cstar', T.SEEDS), ('plain', ('x',)), ('teacher', ('x',))):
        picks = T.validation_picks(T.R23_OUT, row, seeds_of)
        for a in T.ASPECTS:
            for d in T.DIRECTIONS:
                assert picks[a][d]['pick'] == stage1['picks']['caption'][row][a][d], (row, a, d)
        held = T.held_cells(T.R23_OUT, row, picks, seeds_of)
        for d in T.DIRECTIONS:
            built = [round(held[a][d]['mean'], 3) for a in T.ASPECTS]
            assert built == paper[d][row], (row, d, built, paper[d][row])
