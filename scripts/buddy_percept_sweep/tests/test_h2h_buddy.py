import importlib.util
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from scripts.buddy_percept_sweep import h2h_buddy
from scripts.buddy_percept_sweep.h2h_buddy import (
    BuddyStage1Config, fit_buddy_stage1, monitor_subsample, pilot_constants,
)
from scripts.buddy_percept_sweep.h2h_store import H2HStore
from scripts.buddy_percept_sweep.stage1 import upper_triangle_edges

_SNAPSHOT_PILOT = (Path(__file__).resolve().parents[3]
                   / "src/test/20260923_artelingo_buddy_analysis/run_attention_h1_embedding_snapshot_pilot.py")


def test_pilot_constants_map_config_to_arch_names():
    c = pilot_constants(BuddyStage1Config(d_shared=64, lr=3e-4, batch_size=2048, temperature=0.2,
                                          max_epochs=150, plateau_window=3,
                                          plateau_rel_improvement=0.02, content_pca_dim=80))
    assert c == {"D_SHARED": 64, "LEARNING_RATE": 3e-4, "BATCH_SIZE": 2048, "TEMPERATURE": 0.2,
                 "MAX_EPOCHS": 150, "PLATEAU_WINDOW": 3, "PLATEAU_REL_IMPROVEMENT": 0.02,
                 "CONTENT_PCA_DIM": 80}


def test_monitor_subsample_matches_pilot_draw_when_full():
    # pilot: rng = default_rng(seed); sampled = rng.choice(n, 2000, replace=False); rank = rng.choice(n, 5000, replace=False)
    n = 9365
    s, r = monitor_subsample(n_monitor=n, seed=42, edge_sample=2000, rank_sample=5000)
    rng = np.random.default_rng(42)
    assert np.array_equal(s, rng.choice(n, size=2000, replace=False))
    assert np.array_equal(r, rng.choice(n, size=5000, replace=False))


def test_monitor_subsample_caps_at_subset_size():
    s, r = monitor_subsample(n_monitor=1200, seed=0, edge_sample=2000, rank_sample=5000)
    assert len(s) == 1200 and len(r) == 1200


def test_invalid_impl_or_heads_raise():
    with pytest.raises(ValueError):
        pilot_constants(BuddyStage1Config(impl="nope"))
    with pytest.raises(ValueError):
        pilot_constants(BuddyStage1Config(heads="attn"))   # pilot impl needs attn1/attn4/mlp128


# ---------------------------------------------------------------- fixtures

def _obj(values) -> np.ndarray:
    out = np.empty(len(values), dtype=object)
    out[:] = list(values)
    return out


def _synthetic_split(rng, n, prefix, dim=24):
    """Clustered img/txt nodes + 28-d affect probabilities, pilot dtypes."""
    labels = rng.integers(0, 6, size=n)
    img = (rng.normal(size=(6, dim))[labels] + 0.6 * rng.normal(size=(n, dim))).astype(np.float32)
    txt = (rng.normal(size=(6, dim))[labels] + 0.6 * rng.normal(size=(n, dim))).astype(np.float32)
    affect = rng.random((n, 28))
    paintings = [f"{prefix}{i:04d}" for i in range(n)]
    emotions = ("awe", "fear", "contentment", "sadness")
    counts = [Counter({emotions[int(k) % 4]: 2, emotions[(int(k) + 1) % 4]: 1}) for k in labels]
    return dict(paintings=paintings, img=img, txt=txt, affect=affect, counts=counts)


def _store_from(data, cca_audit, affect_pilot, genre_map) -> H2HStore:
    tr, ho = data["train"], data["heldout"]
    return H2HStore(
        train_paintings=_obj(tr["paintings"]), heldout_paintings=_obj(ho["paintings"]),
        train_img=tr["img"], train_txt=tr["txt"], heldout_img=ho["img"], heldout_txt=ho["txt"],
        train_content_raw=cca_audit.content_features(tr["img"], tr["txt"], affect_pilot),
        heldout_content_raw=cca_audit.content_features(ho["img"], ho["txt"], affect_pilot),
        train_affect28=np.asarray(tr["affect"], dtype=np.float64),
        heldout_affect28=np.asarray(ho["affect"], dtype=np.float64),
        train_percept_h=np.zeros((len(tr["img"]), 2), dtype=np.float32),
        heldout_percept_h=np.zeros((len(ho["img"]), 2), dtype=np.float32),
        train_emotion=_obj([c.most_common(1)[0][0] for c in tr["counts"]]),
        heldout_emotion=_obj([c.most_common(1)[0][0] for c in ho["counts"]]),
        train_genre=_obj([genre_map[p] for p in tr["paintings"]]),
        heldout_genre=_obj([genre_map[p] for p in ho["paintings"]]),
        train_patches=torch.zeros(len(tr["img"]), 1), heldout_patches=torch.zeros(len(ho["img"]), 1),
    )


@pytest.fixture
def restore_torch_determinism():
    state = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
             torch.are_deterministic_algorithms_enabled(),
             torch.is_deterministic_algorithms_warn_only_enabled())
    yield state
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = state[0], state[1]
    torch.use_deterministic_algorithms(state[2], warn_only=state[3])


# Small-data arch constants so the real pilot helpers run in seconds on CPU.
_SMALL_ARCH = dict(D_SHARED=16, CONTENT_PCA_DIM=12, BATCH_SIZE=64, LEARNING_RATE=1e-3,
                   TEMPERATURE=0.1, MAX_EPOCHS=30, EDGE_SAMPLE_SIZE=50, EFFECTIVE_RANK_SAMPLE_SIZE=80)


@pytest.fixture
def pilot_on_synthetic_data(monkeypatch):
    """The real pilot modules (fresh instances) with only their data loaders
    replaced by synthetic 240-train / 120-held-out paintings, on CPU."""
    from scripts.buddy_percept_sweep.pilot_metrics import load_pilot_modules

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    pm = load_pilot_modules()
    rng = np.random.default_rng(0)
    data = {"train": _synthetic_split(rng, 240, "t"), "heldout": _synthetic_split(rng, 120, "h")}
    genres = ("portrait", "landscape", "abstract")
    genre_map = {p: genres[i % 3] for split in data.values() for i, p in enumerate(split["paintings"])}
    affect_by_painting = {p: row for split in data.values()
                          for p, row in zip(split["paintings"], split["affect"])}
    for pipe, key in ((pm.pipeline, "train"), (pm.heldout_pipeline, "heldout")):
        pipe.assert_extraction_complete = lambda: None
        pipe.load_dedup_features = lambda key=key: (
            data[key]["paintings"], data[key]["img"], data[key]["txt"], data[key]["counts"])
        pipe.load_genre_map = lambda: dict(genre_map)
    pm.affect_pilot.extract_affect_nodes = (
        lambda json_path, paintings, device: np.stack([affect_by_painting[p] for p in paintings]))
    for name, value in _SMALL_ARCH.items():
        setattr(pm.arch, name, value)
    store = _store_from(data, pm.cca_audit, pm.affect_pilot, genre_map)
    return pm, store


def _run_snapshot_pilot(pm, seed, tmp_path, monkeypatch):
    """Run the frozen snapshot pilot's main() unchanged, wired to `pm`."""
    spec = importlib.util.spec_from_file_location("snapshot_pilot_for_h2h_buddy_test", str(_SNAPSHOT_PILOT))
    snap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(snap)
    arch = pm.arch
    by_path = {arch.AFFECT_PILOT_PATH: pm.affect_pilot, arch.SINGLE_MODALITY_PATH: pm.single_modality,
               arch.CCA_AUDIT_PATH: pm.cca_audit}

    def load_sibling_module(name, path):
        if path == arch.PIPELINE_PATH:
            return pm.heldout_pipeline if name.endswith("_heldout") else pm.pipeline
        return by_path[path]

    messages = []
    monkeypatch.setattr(arch, "load_sibling_module", load_sibling_module)
    monkeypatch.setattr(arch, "log", messages.append)
    monkeypatch.setattr(snap, "load_module", lambda name, path: arch)
    monkeypatch.setattr(snap, "SEED", seed)
    monkeypatch.setattr(snap, "NPZ_PATH", str(tmp_path / "snapshot.npz"))
    monkeypatch.setattr(snap, "REPORT_PATH", str(tmp_path / "snapshot.md"))
    monkeypatch.setattr(snap, "REPORT_OUT_DIR", str(tmp_path))
    snap.main()
    stop = [m for m in messages if m.startswith("Training stopped: ")]
    assert len(stop) == 1
    checkpoints = [m for m in messages if m.startswith("epoch=") and "content_recall=" in m]
    with np.load(tmp_path / "snapshot.npz", allow_pickle=True) as npz:
        return (npz["train_embedding_post"], npz["heldout_embedding_post"],
                stop[0][len("Training stopped: "):-1], checkpoints)


# ---------------------------------------------------------------- pilot port

@pytest.mark.parametrize("seed,window,rel", [(42, 5, 0.01), (7, 2, 10.0)])
def test_pilot_port_reproduces_snapshot_pilot_bit_for_bit(
        seed, window, rel, pilot_on_synthetic_data, tmp_path, monkeypatch, restore_torch_determinism):
    pm, store = pilot_on_synthetic_data
    pm.arch.PLATEAU_WINDOW, pm.arch.PLATEAU_REL_IMPROVEMENT = window, rel
    ref_train, ref_heldout, ref_stop, ref_checkpoints = _run_snapshot_pilot(pm, seed, tmp_path, monkeypatch)

    cfg = BuddyStage1Config(impl="pilot", heads="attn1", d_shared=16, content_pca_dim=12, lr=1e-3,
                            batch_size=64, temperature=0.1, max_epochs=30,
                            plateau_window=window, plateau_rel_improvement=rel)
    out = fit_buddy_stage1(cfg, store, seed, np.arange(120), pm, "cpu")

    assert out.train_embedding.dtype == np.float32 and out.heldout_embedding.dtype == np.float32
    assert np.array_equal(out.train_embedding, ref_train)
    assert np.array_equal(out.heldout_embedding, ref_heldout)
    assert out.info["stop_reason"] == ref_stop
    # The plateau monitor's recalls match the pilot's logged checkpoints (its own format).
    assert [f"epoch={c['epoch']} content_recall={c['content_recall']:.4f} "
            f"affect_recall={c['affect_recall']:.4f}" for c in out.info["trajectory"][1:]] == ref_checkpoints
    if rel == 10.0:   # every checkpoint counts as a plateau -> stops at the 2nd checkpoint
        assert out.info["epochs_run"] == 10 and "plateaued" in ref_stop
    assert out.train_labels is None and out.heldout_native is None


def test_pilot_port_monitor_subset_and_restores_module_state(pilot_on_synthetic_data, restore_torch_determinism):
    pm, store = pilot_on_synthetic_data
    before = {name: getattr(pm.arch, name) for name in pilot_constants(BuddyStage1Config())}
    cfg = BuddyStage1Config(impl="pilot", heads="mlp128", d_shared=8, content_pca_dim=10,
                            batch_size=32, max_epochs=10)
    out = fit_buddy_stage1(cfg, store, 3, np.arange(0, 120, 2), pm, "cpu")
    assert out.train_embedding.shape == (240, 8)
    assert out.heldout_embedding.shape == (120, 8)     # all held-out rows, not just the monitor
    assert out.info["epochs_run"] <= 10 and out.info["seconds"] >= 0.0
    assert {name: getattr(pm.arch, name) for name in before} == before
    assert (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled()) == restore_torch_determinism


def _stub_pilot(calls, learned_student=None):
    def build_buddy_graphs(img, txt, K, alpha, device, connect_components):
        calls.append("content")
        n = len(img)
        return None, None, csr_matrix(np.ones((n, n)) - np.eye(n))

    def build_single_modality_graph(name, nodes, pipeline, affect_pilot, device, expected_nodes):
        calls.append(name)
        assert len(nodes) == expected_nodes
        return csr_matrix(np.ones((expected_nodes, expected_nodes)) - np.eye(expected_nodes))

    arch = SimpleNamespace(**{name: 0 for name in pilot_constants(BuddyStage1Config())},
                           upper_triangle_edges=upper_triangle_edges, LearnedStudent=learned_student)
    return SimpleNamespace(
        pipeline=SimpleNamespace(K=20, ALPHA=0.5, build_buddy_graphs=build_buddy_graphs),
        heldout_pipeline=SimpleNamespace(K=20, ALPHA=0.5), affect_pilot=None,
        single_modality=SimpleNamespace(build_single_modality_graph=build_single_modality_graph),
        arch=arch, cca_audit=None,
    )


def _tiny_store(n_train=12, n_heldout=6, seed=0) -> H2HStore:
    rng = np.random.default_rng(seed)
    f = lambda r, c: rng.normal(size=(r, c))
    return H2HStore(
        train_paintings=_obj([f"t{i}" for i in range(n_train)]),
        heldout_paintings=_obj([f"h{i}" for i in range(n_heldout)]),
        train_img=f(n_train, 4).astype(np.float32), train_txt=f(n_train, 4).astype(np.float32),
        heldout_img=f(n_heldout, 4).astype(np.float32), heldout_txt=f(n_heldout, 4).astype(np.float32),
        train_content_raw=f(n_train, 8), heldout_content_raw=f(n_heldout, 8),
        train_affect28=rng.random((n_train, 28)), heldout_affect28=rng.random((n_heldout, 28)),
        train_percept_h=f(n_train, 2), heldout_percept_h=f(n_heldout, 2),
        train_emotion=_obj(["awe"] * n_train), heldout_emotion=_obj(["awe"] * n_heldout),
        train_genre=_obj(["landscape"] * n_train), heldout_genre=_obj([""] * n_heldout),
        train_patches=torch.zeros(n_train, 1), heldout_patches=torch.zeros(n_heldout, 1),
    )


def test_pilot_teacher_edges_cached_per_store():
    calls = []
    pilot = _stub_pilot(calls)
    store_a, store_b = _tiny_store(seed=0), _tiny_store(seed=1)
    first = h2h_buddy._pilot_teacher_edges(store_a, pilot, "cpu")
    again = h2h_buddy._pilot_teacher_edges(store_a, pilot, "cpu")
    assert calls == ["content", "train-affect-teacher"]
    assert all(x is y for x, y in zip(first, again))
    h2h_buddy._pilot_teacher_edges(store_b, pilot, "cpu")    # a different store is rebuilt
    assert calls == ["content", "train-affect-teacher"] * 2


def test_pilot_port_restores_constants_when_training_fails(restore_torch_determinism):
    def failing_student(heads):
        raise RuntimeError("boom")

    pilot = _stub_pilot([], learned_student=failing_student)
    with pytest.raises(RuntimeError, match="boom"):
        fit_buddy_stage1(BuddyStage1Config(content_pca_dim=4), _tiny_store(), 0, np.arange(6), pilot, "cpu")
    assert all(getattr(pilot.arch, name) == 0 for name in pilot_constants(BuddyStage1Config()))
    assert (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled()) == restore_torch_determinism


# ---------------------------------------------------------------- harness adapter

def test_harness_impl_runs_on_tiny_store_and_maps_heads(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    seen = []
    real_student = h2h_buddy.ParameterizedLearnedStudent

    def spy(**kwargs):
        seen.append(kwargs)
        return real_student(**kwargs)

    monkeypatch.setattr(h2h_buddy, "ParameterizedLearnedStudent", spy)
    store = _tiny_store(n_train=40, n_heldout=16)
    for heads in ("attn4", "attn", "mlp128"):
        cfg = BuddyStage1Config(impl="harness", heads=heads, num_heads=2, d_shared=8, content_pca_dim=6,
                                batch_size=16, max_epochs=5, teacher_graph_K=5)
        out = fit_buddy_stage1(cfg, store, 3, np.arange(0), None, "cpu")
        assert out.train_embedding.shape == (40, 8) and out.heldout_embedding.shape == (16, 8)
        assert out.train_embedding.dtype == np.float32 and out.heldout_embedding.dtype == np.float32
        assert np.isfinite(out.train_embedding).all() and np.isfinite(out.heldout_embedding).all()
        assert out.info["epochs_run"] == 5
    assert [s["heads"] for s in seen] == ["attn1", "attn1", "mlp128"]
    assert [s["num_heads"] for s in seen] == [2, 2, 2]
    assert [(s["content_dim"], s["affect_dim"]) for s in seen] == [(6, 28)] * 3


def test_fit_rejects_unknown_impl_and_harness_heads():
    store = _tiny_store()
    with pytest.raises(ValueError):
        fit_buddy_stage1(BuddyStage1Config(impl="nope"), store, 0, np.arange(6), None, "cpu")
    with pytest.raises(ValueError):
        fit_buddy_stage1(BuddyStage1Config(impl="harness", heads="attn8"), store, 0, np.arange(6), None, "cpu")
    with pytest.raises(ValueError):
        fit_buddy_stage1(BuddyStage1Config(impl="pilot", heads="attn"), store, 0, np.arange(6), None, "cpu")
