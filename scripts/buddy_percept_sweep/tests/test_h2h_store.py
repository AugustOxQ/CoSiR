import numpy as np, torch, threading
from scripts.buddy_percept_sweep import h2h_store

def _fake_arrays():
    n, m = 12, 6
    f = lambda r, c: np.arange(r * c, dtype=np.float32).reshape(r, c)
    obj = lambda k, v: np.array([v] * k, dtype=object)
    return dict(train_paintings=obj(n, "p"), heldout_paintings=obj(m, "q"),
                train_img=f(n, 4), train_txt=f(n, 4), heldout_img=f(m, 4), heldout_txt=f(m, 4),
                train_content_raw=f(n, 8), heldout_content_raw=f(m, 8),
                train_affect28=f(n, 28), heldout_affect28=f(m, 28),
                train_percept_h=f(n, 10), heldout_percept_h=f(m, 10),
                train_emotion=obj(n, "joy"), heldout_emotion=obj(m, "fear"),
                train_genre=obj(n, ""), heldout_genre=obj(m, "portrait"))

def _patches(n, m):
    return torch.zeros(n, 3, 5), torch.zeros(m, 3, 5)

def test_build_then_load_round_trips(tmp_path):
    calls = []
    def builder():
        calls.append(1); return _fake_arrays()
    a = h2h_store.load_or_build_store(tmp_path, builder=builder, patch_loader=_patches)
    b = h2h_store.load_or_build_store(tmp_path, builder=builder, patch_loader=_patches)
    assert len(calls) == 1
    assert np.array_equal(a.train_content_raw, b.train_content_raw)
    assert list(b.heldout_genre) == ["portrait"] * 6
    assert b.train_patches.shape == (12, 3, 5)

def test_concurrent_callers_build_once(tmp_path):
    calls = []
    def builder():
        calls.append(1); return _fake_arrays()
    threads = [threading.Thread(target=h2h_store.load_or_build_store,
                                args=(tmp_path,), kwargs=dict(builder=builder, patch_loader=_patches))
               for _ in range(4)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len(calls) == 1

def test_partial_file_is_never_loaded(tmp_path):
    (tmp_path / "h2h_store.npz.tmp").write_bytes(b"garbage")
    s = h2h_store.load_or_build_store(tmp_path, builder=_fake_arrays, patch_loader=_patches)
    assert s.train_img.shape == (12, 4)
