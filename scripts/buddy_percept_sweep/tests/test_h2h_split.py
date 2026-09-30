import numpy as np
from scripts.buddy_percept_sweep.h2h_split import make_split

def _labels(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    emotion = rng.choice(np.array(["joy", "fear", "awe", "sad"], dtype=object), size=n)
    genre = np.where(rng.random(n) < 0.02, "portrait", "").astype(object)
    return emotion, genre

def test_split_is_disjoint_complete_sorted():
    e, g = _labels()
    s = make_split(e, g)
    assert len(np.intersect1d(s.val_idx, s.test_idx)) == 0
    assert np.array_equal(np.sort(np.concatenate([s.val_idx, s.test_idx])), np.arange(len(e)))
    assert np.all(np.diff(s.val_idx) > 0) and np.all(np.diff(s.test_idx) > 0)

def test_split_is_deterministic_and_seed_sensitive():
    e, g = _labels()
    assert make_split(e, g).digest == make_split(e, g).digest
    assert make_split(e, g, seed=1).digest != make_split(e, g).digest

def test_split_is_stratified_by_emotion_and_genre_presence():
    e, g = _labels()
    s = make_split(e, g)
    for label in np.unique(e):
        n_val = np.sum(e[s.val_idx] == label); n_all = np.sum(e == label)
        assert abs(n_val - n_all / 2) <= 1
    has_g = g != ""
    assert abs(has_g[s.val_idx].sum() - has_g.sum() / 2) <= 4   # one per emotion stratum at most
