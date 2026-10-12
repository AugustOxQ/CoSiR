"""C13: worked examples and stored-file mutations, independent of r8 pipeline code."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from rederive import r8_rederive as RD


@pytest.fixture
def tmp_path():
    # The ticket forbids writes outside the worktree and any cleanup/deletion.
    base = Path(__file__).parent / "results/smoke/rederive_tests"
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=base))


def test_own_map_and_crossfit_use_paintings_and_opposite_half():
    # Both groups occur in both halves; second group's half-1 centre points up.
    phi = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], np.float32)
    labels = np.array([0, 1, 0, 1], np.int64)
    got = RD.placements(phi, labels, np.array([0, 0, 1, 1]),
                        np.array([[1, 0], [0, 1], [1, 1]], np.float32))
    assert got["selection"].tolist() == [[1, 0], [0, 1], [1, 0]]
    assert got["bank"].tolist() == [[1, 0], [0, 1], [1, 0], [0, 1]]
    opposed = RD.placements(phi[[0, 1, 3, 2]], labels, np.array([0, 0, 1, 1]), phi)
    assert opposed["bank"].argmax(1).tolist() == [1, 0, 1, 0]
    with pytest.raises(AssertionError, match="empty half"):
        RD.placements(phi, labels, np.array([0, 1, 0, 1]), phi)


def test_224_cells_strict_ties_and_threshold_equality():
    n = 6
    B = {c: {d: np.zeros((n, 13), np.float32) for d in RD.DIRS} for c in RD.CONDS}
    P = {c: np.tile([.4, .1, .1, .4], (n, 1)) for c in RD.CONDS}
    stack = {d: np.zeros((n, 4, 13), np.float32) for d in RD.DIRS}
    arrays, detail = RD.family(B, stack, P, [0., .25, .5, .75], np.arange(n) % 2, "XAFF")
    assert arrays["gate_a"].tolist() == [[1.] * n, [0.] * n, [0.] * n, [0.] * n]
    assert arrays["fused_cells"].tolist() == [0, 0]
    assert arrays["cf_cells"].tolist() == [0, 0]
    assert arrays["sigma"].tolist() == [0, 0]
    assert detail["rho_f"].shape == (224, 2)
    assert detail["rho_f"].dtype == np.int64
    assert not detail["rho_f"].any()
    assert all(not arrays[m].any() and not arrays["cf__" + m].any() for m in RD.METRICS)


def verdict_arrays():
    n = 12
    a = {"cl": np.arange(n) // 2, "pair_index": np.repeat(np.arange(3), 4), "parity": np.arange(n) % 2}
    for name in ("AFF", "B", "B0", "B1", "BX_NB100", "BX_NB67", "BX_ZNB100"):
        a[name + "__r1"] = np.zeros(n)
    for g in ("NB100", "NB67", "ZNB100"):
        for kind in RD.KINDS:
            name = kind + "_" + g
            a[name + "__r1"] = np.full(n, .25)
            a[name + "__gain"] = np.full(n, .25)
            a[name + "_cf__r1"] = np.zeros(n)
            a[name + "_cf__gain"] = np.zeros(n)
    return a


def test_pick_trigger_flag_and_interval_follow_rule():
    got = RD.statistics(verdict_arrays(), {"NB100", "NB67", "ZNB100"}, {})
    assert got["main"]["pick"] == "R1_NB100"
    assert got["main"]["random_control_flag"] is True
    assert got["main"]["promising"] is False
    assert got["screen"]["pick"] is None
    assert got["trigger"]["triggered"] is False
    assert got["trigger"]["strength"] is None
    rec = got["main"]["candidates"]["R1_NB100"]
    assert rec["SC3"] == {"sum4d": 12, "n": 12, "point": 25., "ci95": [25., 25.], "pass": True}
    assert rec["S1"]["pass"] is False  # Fixed 246 integer bar, even in a smoke.
    assert rec["bar_comparator"] == "B1"
    empty = RD.statistics({k: v for k, v in verdict_arrays().items() if not k.startswith(("R1_", "XAFF_", "BX_"))}, set(), {})
    assert empty["trigger"] == {"triggered": False, "strength": None, "reason": "no NB scorer was evaluated", "NB": {}}


def store_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    for folder in ("spaces", "heads", "groupings", "seed42"):
        (root / folder).mkdir(exist_ok=True)
    st, sel = np.arange(4, dtype=np.int64), np.arange(4, 20, dtype=np.int64)
    n = 12
    z = {"scorer_train": st, "g_st": st, "halves": np.array([0, 0, 1, 1]),
         "selection": sel, "groups": np.arange(20), "painting_image": st,
         "row_to_image": np.arange(20), "image_index": np.arange(20, dtype=np.int32),
         "anchor": np.full(n, 4), "candidates": np.tile(np.arange(5, 18), (n, 1)),
         "cl": np.arange(n) // 2, "pair_index": np.repeat(np.arange(3), 4), "parity": np.arange(n) % 2}
    for c in RD.CONDS:
        for d in RD.DIRS:
            z[f"cos__{c}__{d}"] = np.zeros((n, 13), np.float32)
            z[f"tu__{c}__{d}"] = np.zeros((n, 13), np.float32)
    for g in RD.A0:
        for m in ("img", "txt"):
            full = np.full((20, 2), np.nan, np.float32)
            full[sel] = [.5, .5]
            z[f"e2__{g}__{m}"] = full
        np.savez(root / "heads" / f"post_{g}.npz", img=full, txt=full, rows=sel)
    # B1 uses the frozen legacy R0 posterior, explicitly supplied by the fixture.
    np.savez(root / "heads" / "post_R0.npz", img=full, txt=full, rows=sel)
    np.savez(root / "fixture.npz", **z)
    labels = np.array([0, 1, 0, 1], np.int64)
    np.savez(root / "groupings/r8_groupings.npz", NB100=labels, scorer_train=st, g_st=st)
    (root / "groupings/r8_groupings.json").write_text(json.dumps({"groupings": {"NB100": {"K": 2}}, "dropped": {}}))
    phi = np.tile(np.eye(2, dtype=np.float32), (10, 1))
    np.savez(root / "spaces/NB100.npz", phi=phi, image_index=z["image_index"])
    img = np.full((20, 2), np.nan, np.float32)
    img[sel] = np.tile(np.eye(2, dtype=np.float32), (8, 1))
    np.savez(root / "heads/post_NB100.npz", img=img, txt=img, rows=sel)
    np.savez(root / "heads/xheads_NB100.npz", img=np.eye(2, dtype=np.float32)[labels],
             txt=np.eye(2, dtype=np.float32)[labels], rows=st,
             filled_by_half=1-z["halves"], classes_half0=np.arange(2), classes_half1=np.arange(2))
    for g, p in (("NB100", [.4, .1, .1, .4]), ("A0", [.6, .2, .2])):
        P = np.tile(p, (n, 1))
        s = np.sort(P, axis=1)
        np.savez(root / f"seed42/reader_{g}.npz", **{f"{key}__{c}": value for c in RD.CONDS
                 for key, value in (("P", P), ("pick", P.argmax(1)), ("m", s[:, -1] - s[:, -2]))})
    (root / "taus.json").write_text(json.dumps({"NB100": [0., 0., 0., 0.]}))
    return root / "fixture.npz"


def test_stored_derivation_agrees_and_hashes_every_input(tmp_path):
    fixture = store_fixture(tmp_path)
    derived = RD.derive_files(tmp_path, fixture=fixture, smoke=True)
    np.savez(tmp_path / "seed42/arrays_small.npz", **derived["arrays"])
    index = {"batches": [{"file": "arrays_small.npz", "sha256": RD.sha(tmp_path / "seed42/arrays_small.npz")}]}
    (tmp_path / "seed42/index.json").write_text(json.dumps(index))
    pending = tmp_path / "trigger_pending.json"
    pending.write_text(json.dumps({"inputs": {str(tmp_path / "taus.json"): RD.sha(tmp_path / "taus.json")},
                                   "expected": derived["statistics"]["trigger"]}))
    got = RD.check_files(tmp_path, pending, "trigger", fixture=fixture, smoke=True)
    assert got["status"] == "agree"
    assert str(pending) in got["input_sha256"]
    assert str(tmp_path / "heads/post_NB100.npz") in got["input_sha256"]
    assert all(Path(p).is_absolute() and len(v) == 64 for p, v in got["input_sha256"].items())


def test_one_anchor_change_is_named_even_with_refreshed_input_hash(tmp_path):
    fixture = store_fixture(tmp_path)
    derived = RD.derive_files(tmp_path, fixture=fixture, smoke=True)
    stored = derived["arrays"].copy()
    stored["R1_NB100__r1"] = stored["R1_NB100__r1"].copy()
    stored["R1_NB100__r1"][3] += .25
    np.savez(tmp_path / "seed42/arrays_small.npz", **stored)
    (tmp_path / "seed42/index.json").write_text(json.dumps({"batches": [{"file": "arrays_small.npz", "sha256": RD.sha(tmp_path / "seed42/arrays_small.npz")}]}))
    pending = tmp_path / "gate_pending.json"
    pending.write_text(json.dumps({"expected": derived["statistics"]}))
    got = RD.check_files(tmp_path, pending, "gate", fixture=fixture, smoke=True)
    assert got["status"] == "disagree"
    row = next(r for r in got["disagreements"] if r["where"] == "R1_NB100__r1")
    assert row["changed"] == 1 and row["first_index"] == [3]


@pytest.mark.parametrize("mutation,where", [("own", "M11 post_NB100 img"), ("bank", "M12 xheads_NB100 img"), ("tau", "tau NB100")])
def test_stored_own_map_bank_and_tau_changes_are_named(tmp_path, mutation, where):
    fixture = store_fixture(tmp_path)
    derived = RD.derive_files(tmp_path, fixture=fixture, smoke=True)
    np.savez(tmp_path / "seed42/arrays_small.npz", **derived["arrays"])
    (tmp_path / "seed42/index.json").write_text(json.dumps({"batches": [{"file": "arrays_small.npz", "sha256": RD.sha(tmp_path / "seed42/arrays_small.npz")}]}))
    pending = tmp_path / "trigger_pending.json"
    pending.write_text(json.dumps({"expected": derived["statistics"]["trigger"]}))
    if mutation == "tau":
        (tmp_path / "taus.json").write_text(json.dumps({"NB100": [0., .25, .5, .75]}))
    else:
        path = tmp_path / ("heads/post_NB100.npz" if mutation == "own" else "heads/xheads_NB100.npz")
        with np.load(path) as f:
            z = {k: f[k] for k in f.files}
        row = 4 if mutation == "own" else 0
        z["img"][row] = z["img"][row, ::-1]
        np.savez(path, **z)
    got = RD.check_files(tmp_path, pending, "trigger", fixture=fixture, smoke=True)
    assert got["status"] == "disagree"
    assert any(r["where"] == where for r in got["disagreements"])


def test_random_control_seeds_and_selection_painting_order():
    labels, got = RD.random_inputs(np.array([0, 1, 0, 1]), np.array([40, 10, 30, 20, 40]), 4)
    assert labels.tolist() == [0, 1, 1, 0]
    assert got["bank"].argmax(1).tolist() == [1, 1, 0, 0]
    assert got["selection"].argmax(1).tolist() == [1, 0, 0, 0, 1]


def test_stored_stage_evidence_checks_shape_index_finiteness_and_label_count(tmp_path):
    fixture = store_fixture(tmp_path)
    np.save(tmp_path / "embedding.npy", np.zeros((20, 256), np.float32))
    ix = np.arange(20, dtype=np.int64)
    np.save(tmp_path / "index.npy", ix)
    np.save(tmp_path / "labels.npy", np.array([0, 1, 0, 1]))
    np.save(tmp_path / "losses.npy", np.array([1., .5]))
    (tmp_path / "stageE_AL.json").write_text(json.dumps({"pass": True, "shape": True, "finite": True, "index": True}))
    (tmp_path / "stage1_fcmi.json").write_text(json.dumps({"pass": True, "label_count": True, "finite_loss": True}))
    evidence = {"encoders": {"AL": {"record": "stageE_AL.json", "embeddings": "embedding.npy", "image_index": "index.npy",
                "fixture": "fixture.npz", "index_sha256": RD.hashlib.sha256(ix.tobytes()).hexdigest(),
                "keys": {key: key for key in ("shape", "finite", "index")}}},
                "fcmi": {"record": "stage1_fcmi.json", "labels": "labels.npy", "losses": "losses.npy",
                         "keys": {key: key for key in ("label_count", "finite_loss")}}}
    (tmp_path / "stage_evidence.json").write_text(json.dumps(evidence))
    files = RD.Files()
    got = RD.stage_checks(files, tmp_path / "stage_evidence.json", tmp_path, True)
    assert got["AL"] == {"shape": True, "finite": True, "index": True}
    assert got["FCMI"] == {"label_count": True, "finite_loss": True, "n_labels": 2}
    assert not files.disagreements
    np.save(tmp_path / "losses.npy", np.array([1., np.nan]))
    files = RD.Files()
    RD.stage_checks(files, tmp_path / "stage_evidence.json", tmp_path, True)
    assert any(r["where"] == "SC2.finite_loss" for r in files.disagreements)


def test_sc4_integer_boundary_and_trigger_gain_tie():
    a = verdict_arrays()
    a["AFF__r1"][:] = .5
    for kind in RD.KINDS:
        a[f"{kind}_NB67__gain"][:] = .5
    got = RD.statistics(a, {"NB100", "NB67", "ZNB100"}, {})
    assert got["trigger"]["triggered"] is True
    assert got["trigger"]["strength"] == .67
    assert got["main"]["pick"] is None
    a = {"cl": np.arange(300), "pair_index": np.repeat(np.arange(3), 100), "parity": np.arange(300) % 2}
    for name in ("AFF", "B", "B0", "B1", "BX_NB100"):
        a[name + "__r1"] = np.full(300, .25) if name == "AFF" else np.zeros(300)
    for kind in RD.KINDS:
        nm = kind + "_NB100"
        s = np.full(300, .25)
        s[:49], s[200:] = 0., .5
        a[nm + "__r1"] = s
        a[nm + "__gain"] = np.full(300, .25)
        a[nm + "_cf__r1"] = a[nm + "_cf__gain"] = np.zeros(300)
    got = RD.statistics(a, {"NB100"}, {})
    rec = got["main"]["candidates"]["R1_NB100"]
    assert rec["SC4"]["emotion__style"]["sum4d"] == -49
    assert rec["pass"] is True and rec["S1"]["pass"] is True
    assert got["main"]["random_control_flag"] == "not evaluable"
    for kind in RD.KINDS:
        a[f"{kind}_NB100__r1"][49] = 0.
    got = RD.statistics(a, {"NB100"}, {})
    assert got["main"]["candidates"]["R1_NB100"]["SC4"]["emotion__style"]["sum4d"] == -50
    assert got["main"]["pick"] is None and got["main"]["promising"] is False


def test_c13_interval_tolerance_and_exact_integer_fields():
    files = RD.Files()
    RD.compare_json(files, "pending", {"ci95": [1., 2.], "sum4d": 1}, {"ci95": [1. + 5e-10, 2.], "sum4d": 1})
    assert not files.disagreements
    RD.compare_json(files, "pending", {"ci95": [1., 2.], "sum4d": 1}, {"ci95": [1. + 2e-9, 2.], "sum4d": 2})
    assert {r["where"] for r in files.disagreements} == {"pending.ci95", "pending.sum4d"}


def test_224_cell_fused_winner_counterpart_tie_and_opposite_parity_assembly():
    n, parity = 6, np.arange(6) % 2
    B = {c: {d: np.zeros((n, 13), np.float32) for d in RD.DIRS} for c in RD.CONDS}
    P = {"a": np.tile([1., 0., 0., 0.], (n, 1)), "b": np.tile([0., 0., 0., 1.], (n, 1))}
    stack = {d: np.zeros((n, 4, 13), np.float32) for d in RD.DIRS}
    for d in RD.DIRS:
        stack[d][:, 0, 0] = 1
        stack[d][:, 3, 1] = 1
    arrays, detail = RD.family(B, stack, P, [1.] * 4, parity, "XAFF")
    assert arrays["fused_cells"].tolist() == [1, 1]
    assert arrays["cf_cells"].tolist() == [0, 0]
    assert detail["rho_f"][1].tolist() == [12, 12]
    assert detail["gam_f"][1].tolist() == [12, 12]
    assert arrays["r1"].tolist() == arrays["gain"].tolist() == [1.] * n
    assert not arrays["cf__r1"].any()
    for d in RD.DIRS:
        stack[d][parity == 1] = 0
    arrays, detail = RD.family(B, stack, P, [1.] * 4, parity, "XAFF")
    assert arrays["fused_cells"].tolist() == [1, 0]
    assert not arrays["r1"].any()  # Each half is scored by the other half's cell.


def test_smoke_cli_writes_c13_contract_and_disagreement_exit(tmp_path):
    fixture = store_fixture(tmp_path)
    derived = RD.derive_files(tmp_path, fixture=fixture, smoke=True)
    batch = tmp_path / "seed42/arrays_small.npz"
    np.savez(batch, **derived["arrays"])
    index = tmp_path / "seed42/index.json"
    index.write_text(json.dumps({"batches": [{"file": batch.name, "sha256": RD.sha(batch)}]}))
    pending = tmp_path / "trigger_pending.json"
    pending.write_text(json.dumps({"smoke": True, "expected": derived["statistics"]["trigger"]}))
    env = {**os.environ, "R8_RESULTS": str(Path(__file__).parent / "results"),
           "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
    command = [sys.executable, str(Path(RD.__file__)), "--step", "trigger", "--root", str(tmp_path),
               "--pending", str(pending), "--fixture", str(fixture), "--smoke"]
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "rederive/trigger_rederive.json").read_text())
    assert report["status"] == "agree" and report["step"] == "trigger" and report["smoke"] is True
    assert str(pending) in report["input_sha256"] and len(report["rule_sha256"]) == 64
    stored = derived["arrays"].copy()
    stored["R1_NB100__r1"] = stored["R1_NB100__r1"].copy()
    stored["R1_NB100__r1"][0] += .25
    np.savez(batch, **stored)
    index.write_text(json.dumps({"batches": [{"file": batch.name, "sha256": RD.sha(batch)}]}))
    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    report = json.loads((tmp_path / "rederive/trigger_rederive.json").read_text())
    assert report["status"] == "disagree"
