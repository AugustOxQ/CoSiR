"""Tests of the guards in run_r7.py (final review B1; rule §7.2, §7.4, §7.7, §9, constitution C4 and C5).

No real data and no held row: every file is synthetic under `tmp_path`, and the role's root is `tmp_path`, so nothing is
written under `results/`. Each guard has one test that fails when the guard is broken, and a positive control where a
guard could pass for the wrong reason.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_r7 as X  # noqa: E402

R = X.R


def role_in(root):
    return R.Roles("real", R.DEV_SEED, R.TEST_SEEDS, False, Path(root))


def put(root, name, rec):
    return R.write_json_once(Path(root) / name, rec)


def pair_hashes(tag):
    return {p: f"{tag}{i}" * 8 for i, p in enumerate(X.PAIR_ORDER)}


# ---------------------------------------------------------------------------------------------- hash_problems (§7.2)
def test_hash_problems_clean():
    assert X.hash_problems(pair_hashes("a"), {56: pair_hashes("b")}, {43: pair_hashes("c")}) == []


def test_hash_problems_flags_an_h5_hash():
    new = {**pair_hashes("a"), "style__genre": X.H5[53][2]}
    problems = X.hash_problems(new, {}, {})
    assert len(problems) == 1 and "H5 seed 53" in problems[0] and "style__genre" in problems[0]


def test_hash_problems_flags_an_earlier_seed_hash():
    earlier = {43: {"emotion__style": "e" * 64, "emotion__genre": "f" * 64, "style__genre": "9" * 64}}
    new = {**pair_hashes("a"), "emotion__genre": "f" * 64}
    problems = X.hash_problems(new, {}, earlier)
    assert len(problems) == 1 and "earlier seed 43" in problems[0]


def test_hash_problems_flags_another_new_seed_and_a_duplicate_within_a_seed():
    new = pair_hashes("a")
    assert any("new seed 56" in p for p in X.hash_problems(new, {56: pair_hashes("a")}, {}))
    same = {**new, "emotion__genre": new["emotion__style"]}
    assert any("share an episode SHA-256" in p for p in X.hash_problems(same, {}, {}))


# ----------------------------------------------------------------------------------------- the rebuild refusal (§7.2)
def test_build_seed_refuses_to_rebuild_existing_episodes(tmp_path, monkeypatch):
    files = {k: tmp_path / f"{k}_seed55.{'json' if k == 'baselines' else 'npz'}" for k in X.EP_KINDS}
    files["episodes"].write_bytes(b"x")
    monkeypatch.setattr(X, "seed_files", lambda seed, smoke: files)
    monkeypatch.setattr(X, "run_baselines", lambda *a: pytest.fail("a seed must never be rebuilt"))
    with pytest.raises(AssertionError, match="never rebuilt"):
        X.build_seed(role_in(tmp_path), 55, {}, {}, check_only=False)
    assert files["episodes"].read_bytes() == b"x" and not (tmp_path / "build_seed55.json").exists()


# ----------------------------------------------------------------------------------------- the pending file (§7.4)
CEILING = {"sum4d": 100, "n": 4096, "point": 0.5, "ci95": [0.1, 0.9], "zone": "WEAK"}


def test_write_pending_is_write_once_and_refuses_a_changed_core_key(tmp_path):
    role = role_in(tmp_path)
    X.write_pending(role, "ceiling", CEILING)
    path = tmp_path / "ceiling_pending.json"
    first = path.read_text()
    X.write_pending(role, "ceiling", {**CEILING, "extra": 1})  # extras are not core keys
    assert path.read_text() == first
    with pytest.raises(AssertionError, match="pending changed"):
        X.write_pending(role, "ceiling", {**CEILING, "sum4d": 101})
    assert path.read_text() == first


# ----------------------------------------------------------------------------------------- the verdict refusal (§7.4)
def test_finalize_refuses_to_overwrite_a_verdict(tmp_path):
    role = role_in(tmp_path)
    put(tmp_path, "stage0_regression.json", {"pass": True, "smoke": False})
    put(tmp_path, "prep.json", {"smoke": False, "dropped": []})
    put(tmp_path, "ceiling_verdict.json", {**CEILING, "smoke": False})
    before = (tmp_path / "ceiling_verdict.json").read_text()
    with pytest.raises(AssertionError, match="refusing to overwrite verdict"):
        X.finalize(role, "ceiling")
    assert (tmp_path / "ceiling_verdict.json").read_text() == before


# ------------------------------------------------------------------------------- stage 3 after STOP and n = 0 (§7.7)
def ceiling_and_gate(tmp_path, zone, picks, alpha):
    put(tmp_path, "ceiling_verdict.json", {**CEILING, "zone": zone, "smoke": False})
    n = sum(p is not None for p in picks.values())
    put(tmp_path, "gate_verdict.json", {"picks": picks, "n": n, "alpha": alpha, "smoke": False})
    put(tmp_path, "sensitivity.json", {"smoke": False})


ONE_PICK = ({"csd": "R1_V1", "generic": None}, {"csd": 0.025})
NO_PICK = ({"csd": None, "generic": None}, {})


@pytest.mark.parametrize("stage", ("sensitivity", "build", "test"))
def test_stage3_refused_after_a_k1_stop(tmp_path, stage):
    ceiling_and_gate(tmp_path, "STOP", *ONE_PICK)
    with pytest.raises(AssertionError, match="K1 STOP"):
        X.stage3_ready(role_in(tmp_path), stage)


@pytest.mark.parametrize("stage", ("sensitivity", "build", "test"))
def test_stage3_refused_when_n_is_zero(tmp_path, stage):
    ceiling_and_gate(tmp_path, "CLEAR", *NO_PICK)
    with pytest.raises(AssertionError, match="n = 0"):
        X.stage3_ready(role_in(tmp_path), stage)


@pytest.mark.parametrize("zone", ("WEAK", "CLEAR"))
def test_stage3_ready_passes_with_a_pick_and_no_stop(tmp_path, zone):
    ceiling_and_gate(tmp_path, zone, *ONE_PICK)
    gate, got = X.stage3_ready(role_in(tmp_path), "build")
    assert got == zone and gate["n"] == 1


# ------------------------------------------------------------------- finalize compares the episode IDs (A1) and both ways (A3)
N = 6


def phase_arrays(with_ids=True):
    arr = {f"{name}__{m}": np.zeros(N) for name in ("B", "B0", "B1", "AFF") for m in X.METRICS}
    if with_ids:
        arr |= {"cl": np.arange(N, dtype=np.int64), "pair_index": np.repeat([0, 1, 2], 2).astype(np.int64),
                "parity": np.tile([0, 1], 3).astype(np.int64)}
    return arr


def test_required_keys_hold_the_episode_ids_on_every_stage():
    gate = {"candidates": {n: {"dropped": True} for n in X.S.CANDIDATES}}
    for stage, pending in (("ceiling", {}), ("gate", gate)):
        assert {"cl", "pair_index", "parity"} <= X.required_keys(stage, pending)


def test_phase_diffs_flags_a_phase_file_without_the_episode_ids(tmp_path):
    arr = phase_arrays()
    path = tmp_path / "phase.npz"
    np.savez(path, **phase_arrays(with_ids=False))
    diff = X.phase_diffs(arr, path, {"cl", "pair_index", "parity"}, "x")
    assert sorted(d.split(":")[0].split(".")[-1] for d in diff) == ["cl", "pair_index", "parity"]
    np.savez(path, **{**arr, "cl": arr["cl"][::-1].copy()})
    assert X.phase_diffs(arr, path, {"cl"}, "x") == ["phase.npz.cl"]
    np.savez(path, **arr)
    assert X.phase_diffs(arr, path, {"cl", "pair_index", "parity"}, "x") == []


GATE = {"candidates": {n: {"dropped": True} for n in X.S.CANDIDATES}, "references": {}, "picks": {}, "n": 0,
        "alpha": {}, "promising": False, "random_control_flag": False}


def gate_files(tmp_path, stats):
    root = tmp_path
    put(root, "ceiling_verdict.json", {**CEILING, "smoke": False})
    put(root, "prep.json", {"smoke": False, "dropped": []})
    put(root, "gate_pending.json", {**GATE, "smoke": False})
    (root / "rederive").mkdir()
    (root / "rederive" / "stats_gate.json").write_text(json.dumps(stats))
    (root / "seed42").mkdir()
    arr = phase_arrays()
    np.savez(root / "seed42" / "arrays.npz", **arr)
    np.savez(root / "rederive" / "phaseA_seed42.npz", **arr)
    return root


def test_finalize_gate_passes_on_equal_records_and_arrays(tmp_path):
    gate_files(tmp_path, GATE)
    X.finalize(role_in(tmp_path), "gate")
    assert (tmp_path / "gate_verdict.json").is_file()


def test_finalize_gate_refuses_a_key_only_the_rederivation_holds(tmp_path):
    stats = {**GATE, "candidates": {**GATE["candidates"], "R1_V9": {"dropped": True}}}
    gate_files(tmp_path, stats)
    with pytest.raises(SystemExit, match="disagreement"):
        X.finalize(role_in(tmp_path), "gate")
    assert not (tmp_path / "gate_verdict.json").exists()


def test_finalize_gate_refuses_a_phase_file_without_cl(tmp_path):
    gate_files(tmp_path, GATE)
    np.savez(tmp_path / "rederive" / "phaseA_seed42.npz", **{k: v for k, v in phase_arrays().items() if k != "cl"})
    with pytest.raises(SystemExit, match="disagreement"):
        X.finalize(role_in(tmp_path), "gate")


# ------------------------------------------------------------------------- the descriptive pass and corrected verdicts (m5)
def test_descriptive_record_refuses_a_corrected_verdict(tmp_path):
    import run_r7_descriptive as D
    put(tmp_path, "gate_verdict.json", {"smoke": False})
    assert D.record(tmp_path / "gate_verdict.json", False)["smoke"] is False
    put(tmp_path, "gate_verdict_fix1.json", {"smoke": False})
    with pytest.raises(AssertionError, match="corrected verdict exists"):
        D.record(tmp_path / "gate_verdict.json", False)
    put(tmp_path, "prep.json", {"smoke": False})  # other records are not verdicts
    assert D.record(tmp_path / "prep.json", False)["smoke"] is False


# ----------------------------------------------------------------------------------- the bank block layout (m4, L11)
def tiny_bank():
    """Two groupings a and b over 8 rows (rows 0-3 a=0, rows 4-7 a=1; b = row parity); one block of 2 episodes."""
    part = {"a": np.array([0, 0, 0, 0, 1, 1, 1, 1]), "b": np.array([0, 1, 0, 1, 0, 1, 0, 1])}
    ep = {"anchor": np.array([0, 1]), "pairs_a_img": np.array([[4, 5], [4, 5]]), "pairs_a_txt": np.array([[6, 7], [6, 7]]),
          "pairs_b_img": np.array([[0, 1], [0, 1]]), "pairs_b_txt": np.array([[2, 3], [2, 3]]),
          "candidates": np.array([[2, 0], [3, 1]])}  # candidate 0 shares a with the anchor, candidate 1 shares b
    return part, ep


def test_check_layout_accepts_a_block_and_refuses_a_broken_one():
    import r7_reader as Q
    part, ep = tiny_bank()
    Q.check_layout(ep, part, [("a", "b")], 2, 0)
    for field, bad in (("pairs_a_txt", np.array([[6, 3], [6, 7]])),      # a pair differs in a
                       ("pairs_a_img", np.array([[0, 5], [4, 5]])),      # a pair image shares the anchor's a-group
                       ("pairs_b_txt", np.array([[3, 1], [2, 3]])),      # b pair differs
                       ("candidates", np.array([[2, 0], [4, 1]]))):      # candidate 0 leaves the anchor's a-group
        with pytest.raises(AssertionError, match="block layout"):
            Q.check_layout({**ep, field: bad}, part, [("a", "b")], 2, 0)
