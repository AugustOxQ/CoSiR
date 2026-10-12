"""Ticket 02: the posterior boundary specified by rule M14 and the ticket."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def tmp_path():
    # Keep fixtures inside the worktree and retain them: this task permits no deletion.
    # rule gap: the first red/green runs used pytest's default /tmp fixture directory;
    # subsequent runs retain fixtures here to respect the task's write boundary.
    root = Path(__file__).resolve().parent / "results/smoke/unit"
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=root))


def test_new_grouping_loads_and_unknown_grouping_is_refused(tmp_path):
    from r8_bundle import load_posteriors

    selection = np.array([1, 3], dtype=np.int64)
    context = SimpleNamespace(groups=np.arange(5), selection=selection)
    posterior = np.full((5, 2), np.nan, dtype=np.float32)
    posterior[selection] = [[0.25, 0.75], [1.0, 0.0]]
    heads = tmp_path / "heads"
    heads.mkdir()
    np.savez(heads / "post_V100.npz", img=posterior, txt=posterior, rows=selection)
    (heads / "head_V100.json").write_text(json.dumps({"n_groups": 2}))
    loaded = load_posteriors(context, "V100", root=tmp_path)
    assert np.array_equal(loaded["txt"], posterior, equal_nan=True)
    assert np.array_equal(loaded["img"], posterior, equal_nan=True)
    with pytest.raises(AssertionError, match="unknown extra grouping"):
        load_posteriors(context, "V99", root=tmp_path)


def test_stage0_comparisons_use_exact_r1_and_absolute_float_tolerance():
    from run_r8_stage0 import Comparison

    c = Comparison()
    c.compare("r1", np.array([0.25 + 1e-14]), np.array([0.25]))
    c.compare("probability", np.array([0.25 + 1e-14]), np.array([0.25]))
    c.compare("too_far", np.array([0.25 + 2e-12]), np.array([0.25]))
    c.compare("masked", np.array([np.nan, 0.25]), np.array([np.nan, 0.25]), equal_nan=True)
    c.compare("invalid", np.array([np.nan]), np.array([0.25]), equal_nan=True)
    assert not c.keys["r1"]["pass"]
    assert c.keys["probability"]["pass"]
    assert not c.keys["too_far"]["pass"]
    assert c.keys["masked"]["pass"]
    assert not c.keys["invalid"]["pass"]
    assert len(c.result()["failures"]) == 3


def test_shared_head_path_matches_r7_fits_on_a_small_fixture():
    import r8_common as R
    from r8_recipe import fit_clip_heads

    rng = np.random.default_rng(8)
    st = np.arange(80, dtype=np.int64)
    selection = np.arange(80, 100, dtype=np.int64)
    local = st % 2
    membership = ((st // 2) % 2).astype(np.int8)
    halves = {"scorer_train": st, "half_of_local_row": membership}
    for h in (0, 1):
        halves[f"local_rows_half{h}"] = np.flatnonzero(membership == h)
        halves[f"global_rows_half{h}"] = st[halves[f"local_rows_half{h}"]]
    data = SimpleNamespace(img_features=rng.normal(size=(100, 8)).astype(np.float32),
                           txt_features=rng.normal(size=(100, 8)).astype(np.float32))
    post, cross, _ = fit_clip_heads(data, local, st, selection, halves, smoke=True,
                                  head_rows=20, check_rows=10)
    H = R.load_r7("r7_heads")
    _, old_post, _ = H.fit_standard(data, local, st, selection, 20, 10)
    old_cross, _ = H.fit_crossfit(data, local, st, halves, 20, 10)
    assert np.array_equal(post["txt"], old_post["txt"], equal_nan=True)
    for key in ("txt", "filled_by_half", "classes_half0", "classes_half1"):
        assert np.array_equal(cross[key], old_cross[key])


def test_resolution_one_recipe_matches_r7_on_a_small_fixture():
    import r8_common as R
    from r8_recipe import recipe_graph, recipe_at_resolution

    rng = np.random.default_rng(42)
    phi = R.unit(rng.normal(size=(64, 8)).astype(np.float32))
    g_st = np.repeat(np.arange(64, dtype=np.int64), 4)
    graph = recipe_graph(phi)
    ours, record = recipe_at_resolution(phi, g_st, graph, resolution=1.0, min_rows=4)
    G = R.load_r7("r7_groupings")
    rs = R.load_r7("r7_common").load("run_step1")
    old, _, _ = G.group_vectors(phi, g_st, np.zeros(64, dtype=np.int64), rs, graph=graph, min_rows=4)
    assert np.array_equal(ours, old)
    assert record["r"] == 1.0 and record["K"] == len(np.unique(old))
