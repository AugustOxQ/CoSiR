"""Tests of run_r9_stage0b.py (ticket 12; rule 3.2, 10 row 2, 12 row C12): the verdict logic on synthetic data.

Case selection from the three L1/L2 patterns, a differing pick, the 0.05 R@1-point tolerance (0.0501 fails, 0.0499
passes), an element changed in a CPU part, a script SHA that differs from its record, and the replay path override
(it reaches `r9_common.input_pins`, so a stage 0a pass cannot come from CLIP's pinned files). No feature, label or
episode is read; the full run is not run here (see the module docstring of run_r9_stage0b.py).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261202_r9_scope/test_r9_stage0b.py
"""
import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first, before anything that imports src)
import r9_backbone as BB  # noqa: E402
import r9_bundle as B  # noqa: E402
import r9_picks as P  # noqa: E402
import run_r9_stage0b as S0B  # noqa: E402

import numpy as np  # noqa: E402


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- case selection (item 2)

def test_case_1_when_l1_equals_the_pinned_checkpoint():
    assert S0B.select_case(True, None) == "case1"
    assert S0B.select_case(True, False) == "case1"          # L2 is irrelevant once L1 is bit-equal
    assert S0B.select_case(True, True) == "case1"


def test_case_2_when_l1_differs_from_the_pin_and_from_l2():
    assert S0B.select_case(False, False) == "case2"


def test_case_3_when_l1_equals_l2_but_not_the_pin():
    assert S0B.select_case(False, True) == "case3"


def test_no_l2_and_l1_not_equal_is_pending_not_a_pass():
    assert S0B.select_case(False, None) == "pending_l2"


def test_decide_per_case():
    ok = dict(item1=True, gates=True, regression=True, item3=True, item4=True, picks=True, independent=True)
    assert S0B.decide("case1", **ok)
    assert not S0B.decide("case1", **{**ok, "regression": False})     # case 1: the 131 items must be exact
    assert not S0B.decide("case1", **{**ok, "picks": False})          # the exact picks are gated in every case
    assert S0B.decide("case1", **{**ok, "item3": False, "independent": False})   # case 1: the 131 items cover them
    assert S0B.decide("case2", **{**ok, "regression": False})         # case 2: item 3 is the gate instead
    assert not S0B.decide("case2", **{**ok, "item3": False})
    assert not S0B.decide("case2", **{**ok, "picks": False})
    assert not S0B.decide("case2", **{**ok, "independent": False})    # rule 3.2 item 3: CPU-made parts exact
    for case in ("case3", "pending_l2", None):
        assert not S0B.decide(case, **ok)                             # stop; trace; the user decides
    for bad in ("item1", "gates", "item4"):
        assert not S0B.decide("case1", **{**ok, bad: False})
        assert not S0B.decide("case2", **{**ok, bad: False})


# ---------------------------------------------------------------- item 3: picks exact, 0.05 R@1 points

def want_numbers():
    return {"picks": {"aff_cells": [39, 119], "cf_cells": [149, 10], "B": {"0": [16.0, 8.0], "1": [8.0, 2.0]},
                      "tau": [0.1, 0.2, 0.3, 0.4], "rca_lambda": {"0": 0.5, "1": 0.5}},
            "r1": {"B": 18.341064453125, "AFF": 19.136555989583336},
            "points": {"P1": 6.174723307291666, "P3": 0.7954915364583334}}


def test_tolerance_is_0_05_r1_points():
    assert S0B.within_tolerance(18.341064453125 + 0.0499, 18.341064453125)
    assert not S0B.within_tolerance(18.341064453125 + 0.0501, 18.341064453125)
    assert not S0B.within_tolerance(18.341064453125 - 0.0501, 18.341064453125)    # both directions
    assert S0B.within_tolerance(18.341064453125, 18.341064453125)


def test_item3_passes_on_identical_numbers():
    res = S0B.item3_check(want_numbers(), want_numbers())
    assert res["passed"] and not res["failures"]


@pytest.mark.parametrize("delta,passes", [(0.0499, True), (0.0501, False), (-0.0501, False)])
def test_item3_r1_and_points_tolerance(delta, passes):
    for group, name in (("r1", "AFF"), ("points", "P3")):
        got = want_numbers()
        got[group][name] += delta
        res = S0B.item3_check(got, want_numbers())
        assert res["passed"] is passes, (group, delta)
        assert (not res["failures"]) is passes


@pytest.mark.parametrize("path", [("aff_cells", 1), ("cf_cells", 0), ("B", "0"), ("tau", 2), ("rca_lambda", "1")])
def test_item3_a_differing_pick_fails_even_by_a_tiny_amount(path):
    got = copy.deepcopy(want_numbers())
    key, sub = path
    if isinstance(got["picks"][key], list):
        got["picks"][key][sub] = got["picks"][key][sub] + 1e-12 if key == "tau" else got["picks"][key][sub] + 1
    else:
        v = got["picks"][key][sub]
        got["picks"][key][sub] = [v[0], v[1] * 2] if isinstance(v, list) else v + 1e-12
    res = S0B.item3_check(got, want_numbers())
    assert not res["passed"] and any(key in f for f in res["failures"])


def test_item3_a_missing_number_fails():
    got = want_numbers()
    del got["points"]["P1"]
    assert not S0B.item3_check(got, want_numbers())["passed"]


def test_count_changed_episodes():
    want = {"aff_fused": np.array([0.0, 0.25, 1.0, 0.5]), "cosine": np.array([0.0, 0.0, 1.0, 1.0])}
    got = {"aff_fused": np.array([0.0, 0.5, 1.0, 0.25]), "cosine": want["cosine"].copy()}
    res = S0B.count_changed(got, want)
    assert res["per_scorer"] == {"aff_fused": 2, "cosine": 0}
    assert res["any_a3_fed"] == 2 and res["n_episodes"] == 4
    with pytest.raises(AssertionError):
        S0B.count_changed({"aff_fused": np.zeros(3)}, {"aff_fused": np.zeros(4)})       # shapes must agree


# ---------------------------------------------------------------- item 1: element by element

def write_npz(path, **arrays):
    np.savez(path, **arrays)
    return path


def base_arrays():
    return {"image": np.arange(12, dtype=np.int64), "caption": np.arange(12, dtype=np.int64)[::-1].copy(),
            "post": np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4), "name": np.array("AIC")}


def test_item1_equal_arrays_pass_and_file_bytes_do_not_matter(tmp_path):
    a = write_npz(tmp_path / "a.npz", **base_arrays())
    b = write_npz(tmp_path / "b.npz", **base_arrays())          # another zip timestamp, same arrays
    assert S0B.compare_role("partitions", b, a) == []


@pytest.mark.parametrize("role", ["partitions", "graph", "bank_AIC", "n6_posteriors", "rb_heads_affect",
                                  "rb_bank_A0_half0"])
def test_item1_a_changed_array_element_fails(tmp_path, role):
    a = write_npz(tmp_path / "a.npz", **base_arrays())
    arrays = base_arrays()
    arrays["image"][7] += 1
    b = write_npz(tmp_path / "b.npz", **arrays)
    diff = S0B.compare_role(role, b, a)
    assert diff and "image" in diff[0]


def test_item1_one_float_ulp_fails(tmp_path):
    a = write_npz(tmp_path / "a.npz", **base_arrays())
    arrays = base_arrays()
    arrays["post"][1, 2] = np.nextafter(arrays["post"][1, 2], np.float32(2))
    b = write_npz(tmp_path / "b.npz", **arrays)
    assert S0B.compare_role("n6_posteriors", b, a)


def test_item1_dtype_shape_and_key_set_fail(tmp_path):
    a = write_npz(tmp_path / "a.npz", **base_arrays())
    for change in (lambda d: d.update(image=d["image"].astype(np.int32)),
                   lambda d: d.update(image=d["image"][:-1]),
                   lambda d: d.pop("caption"),
                   lambda d: d.update(extra=np.zeros(1))):
        arrays = base_arrays()
        change(arrays)
        b = write_npz(tmp_path / "b.npz", **arrays)
        assert S0B.compare_role("partitions", b, a), change


def test_item1_nan_in_the_same_place_is_equal(tmp_path):
    arrays = base_arrays()
    arrays["post"][0, 0] = np.nan
    a = write_npz(tmp_path / "a.npz", **arrays)
    b = write_npz(tmp_path / "b.npz", **arrays)
    assert S0B.compare_role("n6_posteriors", b, a) == []


def write_json(path, rec):
    Path(path).write_text(json.dumps(rec))
    return path


def test_item1_tau_is_exact(tmp_path):
    rec = {"taus": [3.8684538364530674e-05, 0.21702129490553143, 0.47973989883399526, 0.7502585816077211],
           "percentiles": [0, 25, 50, 75], "n_values": 24576, "config": "A0", "provenance": {"time": "x"}}
    pinned = write_json(tmp_path / "pinned.json", rec)
    same = write_json(tmp_path / "same.json", {**rec, "provenance": {"time": "other"}, "method": "other text"})
    assert S0B.compare_role("rc_tau", same, pinned) == []
    off = copy.deepcopy(rec)
    off["taus"][2] = np.nextafter(off["taus"][2], 1.0)
    assert S0B.compare_role("rc_tau", write_json(tmp_path / "off.json", off), pinned)


def baselines_record(**lam):
    scorers = {name: {"lambda_picks": {"0": 0.5, "1": 0.25}, "overall": {"timing": 1.0}} for name in P.LAMBDA_SCORERS}
    for name, v in lam.items():
        scorers[name]["lambda_picks"] = v
    return {"episodes_seed": 42, "fit_rows": 60000, "fit_rows_sha256": "ab" * 32,
            "episodes_sha256": {"emotion__style": "cd" * 32}, "scorers": scorers, "total_s": 1.0}


def test_item1_rca_and_pm_lambda_picks_are_exact(tmp_path):
    pinned = write_json(tmp_path / "p.json", baselines_record())
    same = baselines_record()
    same["total_s"] = 99.0
    same["scorers"]["rca"]["overall"] = {"timing": 5.0}                    # not a pick
    assert S0B.compare_role("baselines_seed42", write_json(tmp_path / "s.json", same), pinned) == []
    for name in ("rca", "kissme", "value_prototype"):
        off = baselines_record(**{name: {"0": 0.5, "1": 0.5}})
        assert S0B.compare_role("baselines_seed42", write_json(tmp_path / "o.json", off), pinned), name
    gone = baselines_record()
    del gone["scorers"]["tip"]
    assert S0B.compare_role("baselines_seed42", write_json(tmp_path / "g.json", gone), pinned)


def reader_json(**kw):
    rec = {"stage": "train", "smoke": False, "rule_sha256": "r" * 64, "config": "A0",
           "groupings": ["affect", "image", "caption"], "feature_names": ["a", "b"], "classes": {"0": [0, 1, 2]},
           "halves": {"0": {"cv_table": [{"C": 1.0, "mean_log_loss": 0.5, "fit_s": 1.3}], "chosen_C": 1.0,
                            "n_iter_refit": [54]}},
           "sklearn_version": "1.6.1", "numpy_version": "2.0", "runtime_s": 30.0,
           "provenance": {"script_sha256": "x", "finished_amsterdam": "t"},
           "halves_npz_sha256": "h1", "heads_npz_sha256": {"affect": "h2"}, "model_spec": {"x": "prose"}}
    rec.update(kw)
    return rec


def test_item1_reader_json_by_fields_numbers_equal_timings_and_provenance_excluded(tmp_path):
    pinned = write_json(tmp_path / "p.json", reader_json())
    other = reader_json(runtime_s=999.0, provenance={"script_sha256": "y"}, halves_npz_sha256="zz",
                        heads_npz_sha256={"affect": "z"}, model_spec={"x": "other prose"}, exploratory="a note")
    other["halves"]["0"]["cv_table"][0]["fit_s"] = 7.7                       # a timing
    assert S0B.compare_role("rb_reader_A0_json", write_json(tmp_path / "s.json", other), pinned) == []
    off = reader_json()
    off["halves"]["0"]["cv_table"][0]["mean_log_loss"] = 0.5000000001        # a number
    assert S0B.compare_role("rb_reader_A0_json", write_json(tmp_path / "o.json", off), pinned)
    off = reader_json()
    off["halves"]["0"]["chosen_C"] = 100.0
    assert S0B.compare_role("rb_reader_A0_json", write_json(tmp_path / "c.json", off), pinned)


def test_item1_reader_pickle_compares_fitted_models_element_by_element(tmp_path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    x = np.random.default_rng(0).normal(size=(40, 3))
    y = (x[:, 0] > 0).astype(int)

    def pk(c):
        return {"config": "A0", "halves": [{"scaler": StandardScaler().fit(x),
                                              "model": LogisticRegression(C=c).fit(x, y)}]}
    pa, pb, pc = (tmp_path / n for n in ("a.pkl", "b.pkl", "c.pkl"))
    pa.write_bytes(pickle.dumps(pk(1.0)))
    pb.write_bytes(pickle.dumps(pk(1.0)))
    pc.write_bytes(pickle.dumps(pk(100.0)))
    assert S0B.compare_role("rb_reader_A0", pb, pa) == []
    assert S0B.compare_role("rb_reader_A0", pc, pa)


def test_item1_covers_every_cpu_part_and_not_the_a3_fed_ones():
    roles = set(S0B.ITEM1_ROLES)
    assert {"partitions", "graph", "bank_AIC", "n6_posteriors", "step1_heads_style", "rb_reader_A0",
            "rb_reader_A0_json", "rc_tau", "baselines_seed42"} <= roles
    assert {f"rb_heads_{g}" for g in ("affect", "image", "caption", "csd", "rand")} <= roles
    assert {"rb_bank_A0_half0", "rb_bank_A0_half1"} <= roles
    assert not roles & set(BB.FROZEN_ROLES)                    # frozen roles are not re-made
    assert set(S0B.A3_FED_ROLES) == {"picks_seed42", "cells_seed42"}
    assert roles | set(S0B.A3_FED_ROLES) | {"A3_seed42", "picked"} == set(BB.RECIPE_ROLES)


def test_compare_item1_reports_a_missing_replay_file(tmp_path):
    a = write_npz(tmp_path / "a.npz", **base_arrays())
    res = S0B.compare_item1({"partitions": tmp_path / "missing.npz"}, {"partitions": a}, roles=("partitions",))
    assert not res["passed"] and not res["roles"]["partitions"]["equal"]


# ---------------------------------------------------------------- item 4: script SHA-256s

SCRIPTS = ("r9_recipe_groups.py", "r9_recipe_a3.py", "r9_recipe_reader.py", "r9_recipe_baselines.py",
           "r9_recipe_cells.py")


def make_replay(tmp_path, wrong=None):
    """A scripts folder and a replay folder whose records name them; `wrong` is the script whose record is stale."""
    scripts, replay = tmp_path / "scripts", tmp_path / "replay"
    (replay / "a3_L1").mkdir(parents=True)
    scripts.mkdir()
    for i, name in enumerate(SCRIPTS):
        body = f"# script {i}\n".encode()
        (scripts / name).write_bytes(body)
        rec = {"script": f"/elsewhere/checkout/{name}", "script_sha256": sha(b"stale" if name == wrong else body)}
        folder = replay / "a3_L1" if name == "r9_recipe_a3.py" else replay
        (folder / f"part{i}.bin.record.json").write_text(json.dumps(rec))
    return scripts, replay


def test_item4_matching_script_hashes_pass(tmp_path):
    scripts, replay = make_replay(tmp_path)
    res = S0B.check_scripts(replay, scripts)
    assert res["passed"] and not res["problems"]
    assert set(res["scripts"]) == set(SCRIPTS)
    assert res["scripts"]["r9_recipe_cells.py"]["sha256"] == sha(b"# script 4\n")


@pytest.mark.parametrize("name", SCRIPTS)
def test_item4_a_script_that_differs_from_its_record_fails(tmp_path, name):
    scripts, replay = make_replay(tmp_path, wrong=name)
    res = S0B.check_scripts(replay, scripts)
    assert not res["passed"] and any(name in p for p in res["problems"])


def test_item4_an_edited_script_after_the_run_fails(tmp_path):
    scripts, replay = make_replay(tmp_path)
    with open(scripts / "r9_recipe_reader.py", "ab") as f:
        f.write(b"# edited after the replay\n")
    assert not S0B.check_scripts(replay, scripts)["passed"]


def test_item4_a_recipe_script_without_any_record_fails(tmp_path):
    scripts, replay = make_replay(tmp_path)
    (replay / "part4.bin.record.json").unlink()                 # the cells script left no record
    res = S0B.check_scripts(replay, scripts)
    assert not res["passed"] and any("r9_recipe_cells.py" in p for p in res["problems"])


# ---------------------------------------------------------------- item 2: identity gates

def good_gates():
    return {"backbone": "clip_b32", "run": "L1", "config": {"passed": True},
            "inputs": {"passed": True, "pinned": True},
            "step1_loss": {"passed": True, "step1_loss": 20.68084716796875},
            "state_equal": {"pinned": {"state_equal": True}}}


def test_identity_gates():
    assert S0B.check_identity_gates(good_gates(), "L1") == []
    for edit in (lambda g: g["config"].update(passed=False), lambda g: g["inputs"].update(passed=False),
                 lambda g: g["inputs"].update(pinned=False), lambda g: g["step1_loss"].update(passed=None),
                 lambda g: g.update(run="L2"), lambda g: g.update(backbone="siglip2_so400m"),
                 lambda g: g.pop("config")):
        g = good_gates()
        edit(g)
        assert S0B.check_identity_gates(g, "L1"), edit


# ---------------------------------------------------------------- the replay path override (build 5 note)

def routed_files(tmp_path):
    files = {}
    for role in S0B.ROUTED_ROLES:
        p = tmp_path / "replay" / Path(BB.CLIP_PARTS[role]).name
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(f"replay {role}".encode())
        files[role] = p
    return files


def test_routed_rels_cover_the_stage_0a_reads_of_recipe_parts():
    rels = S0B.ROUTED_RELS
    assert B.A3_REL in rels and set(B.READER_RELS) <= set(rels)
    assert "20261031_pseudo_partitions/results/partitions.npz" in rels
    assert "20261117_reader_fix_csd/results/rc_tau.json" in rels
    assert B.BASELINES42_REL not in rels                 # PM fit rows are a frozen role: CLIP's file whatever the replay
    assert "20261108_new_method_quick_checks/results/n6_posteriors.npz" not in rels    # compared, not an input


def test_override_reaches_input_pins_and_parts(tmp_path):
    files = routed_files(tmp_path)
    pinned_before = R.input_pins("clip_b32")
    parts_before = dict(BB.parts("clip_b32"))
    with S0B.routed(files):
        pins = R.input_pins("clip_b32")
        for rel, role in S0B.ROUTED_RELS.items():
            path, got_sha = pins[rel]
            assert path == files[role] and path != pinned_before[rel][0]
            assert got_sha == sha(Path(path).read_bytes())            # the replay file's hash, not CLIP's pin
        assert BB.parts("clip_b32")["picked"] == files["picked"]
        assert BB.parts("clip_b32")["picks_seed42"] == files["picks_seed42"]
        assert BB.parts("clip_b32")["affect_km"] == parts_before["affect_km"]          # frozen roles stay CLIP's
        assert pins[B.BASELINES42_REL] == pinned_before[B.BASELINES42_REL]
        assert R.assert_input(B.A3_REL) == sha(files["A3_seed42"].read_bytes())     # a replay file with other bytes
    assert R.input_pins("clip_b32") == pinned_before                    # restored on exit
    assert BB.parts("clip_b32") == parts_before


def test_override_does_not_fall_back_to_the_pinned_file(tmp_path):
    files = routed_files(tmp_path)
    files["A3_seed42"].unlink()
    with S0B.routed(files):
        with pytest.raises(FileNotFoundError):
            R.assert_input(B.A3_REL)               # a read of CLIP's pinned path would have passed
    assert R.input_pins("clip_b32")[B.A3_REL][0] == BB.CLIP_PARTS["A3_seed42"]


def test_override_restores_after_an_exception(tmp_path):
    files = routed_files(tmp_path)
    before = R.input_pins("clip_b32")
    with pytest.raises(RuntimeError):
        with S0B.routed(files):
            raise RuntimeError("boom")
    assert R.input_pins("clip_b32") == before


def test_routed_refuses_a_path_that_is_clips_pinned_file(tmp_path):
    files = routed_files(tmp_path)
    files["A3_seed42"] = Path(BB.CLIP_PARTS["A3_seed42"])
    with pytest.raises(AssertionError):
        with S0B.routed(files):
            pass


def test_resolved_inputs_record_names_replay_paths_and_hashes(tmp_path):
    files = routed_files(tmp_path)
    rec = S0B.resolved_inputs(files)
    assert set(rec) == set(S0B.ROUTED_RELS)
    for rel, role in S0B.ROUTED_RELS.items():
        assert rec[rel]["path"] == str(files[role])
        assert rec[rel]["pinned_path"] == str(R.TEST / rel)
        assert rec[rel]["sha256"] == sha(files[role].read_bytes())


def test_routed_cells_come_from_the_replay_and_assert_taus_is_left_alone(tmp_path):
    import r9_score as S
    files = routed_files(tmp_path)
    cells = {"aff": [1, 2], "cf": [3, 4], "r1_fused": [5, 6], "r1_counterpart": [7, 8]}
    before_cells, before_taus = copy.deepcopy(S.CELLS), R.R3.assert_taus
    with S0B.routed(files, cells=S.cells_from_record(cells)):
        assert S.CELLS["aff"]["fused"] == (1, 2) and S.CELLS["r1"]["cf"] == (7, 8)    # regression_items reads S.CELLS
        assert R.R3.assert_taus is before_taus            # the replay's tau goes to score_seed as taus= (see score_chain)
    assert S.CELLS == before_cells


# ---------------------------------------------------------------- the output file

def test_the_record_is_never_overwritten(tmp_path):
    out = tmp_path / "stage0b.json"
    S0B.check_out(out)                                # absent: fine
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        S0B.check_out(out)


# ================================================================ fix round (review of ticket 12)
# The verdict wiring of `run()` and `score_chain()`, with the heavy steps replaced. Each test below fails for one
# mutant of the review's table (survivors M3) or one fixed finding.

from types import SimpleNamespace  # noqa: E402

import r9_recipe_cells as RC  # noqa: E402
import r9_score as SC  # noqa: E402
import run_r9_stage0a as RS  # noqa: E402


def touch_replay(folder, a3_run="L1", skip=()):
    """A replay folder with one small file per role run() needs (the contents do not matter: the comparisons are
    replaced)."""
    files = S0B.replay_files(folder, a3_run)
    for role in list(S0B.ITEM1_ROLES) + list(S0B.A3_ROLES) + list(S0B.A3_FED_ROLES):
        if role in skip:
            continue
        p = Path(files[role])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    return files


def good_chain(**kw):
    c = {"regression": {"passed": True}, "item3": {"passed": True, "picks_exact": {"aff_cells": True, "B": True}},
         "independent": {"passed": True, "problems": []}, "cpu_changed_episodes": 0,
         "changed_episodes": {"per_scorer": {}, "any_a3_fed": 0, "n_episodes": 4}, "runtime_s": 0.0}
    for k, v in kw.items():
        if k == "regression":
            c["regression"] = {"passed": v}
        elif k == "item3":
            c["item3"] = {**c["item3"], "passed": v}
        elif k == "picks":
            c["item3"] = {**c["item3"], "picks_exact": {"aff_cells": True, "B": v}}
        elif k == "independent":
            c["independent"] = {"passed": v, "problems": [] if v else ["heads_selection_bits_equal_stored: differs"]}
        elif k == "cpu":
            c["cpu_changed_episodes"] = v
    return c


def run_world(monkeypatch, tmp_path, *, case="case1", l2=False, chains=None, item1_ok=True, item1_l2_ok=True,
              gates_ok=True, scripts_ok=True, stale=(), stale_l2=(), pin_error=None, pin_problems=(), score=True,
              absent=(), absent_l2=(), a3_error=None, out="given"):
    replay = tmp_path / "replay"
    touch_replay(replay, skip=absent)
    l2_replay = None
    if l2:
        l2_replay = tmp_path / "replay_l2"
        touch_replay(l2_replay, "L2", skip=absent_l2)
    calls = []

    def is_l2(files):
        return l2 and Path(files["partitions"]).is_relative_to(tmp_path / "replay_l2")

    def check_pins():
        calls.append("check_pins")
        if pin_error:
            raise SystemExit(pin_error)
        return {"parts": {}}

    def compare_item1(files, *a, **k):
        calls.append("item1_L2" if is_l2(files) else "item1_L1")
        ok = item1_l2_ok if is_l2(files) else item1_ok
        return {"passed": ok, "roles": {"partitions": {"equal": ok, "differences": ["d"], "n_differences": 1}}}

    def a3_case(replay_, l2_replay_=None):
        calls.append("a3_case")
        if a3_error:
            raise RuntimeError(a3_error)
        return {"case": case, "identity_gates_passed": gates_ok, "identity_gate_problems": [], "gates": {}}

    def score_chain(files, refs):
        name = "L2" if is_l2(files) else "L1"
        calls.append(f"score_{name}")
        spec = (chains or {}).get(name, good_chain() if name == "L1" or case == "case2" else None)
        if spec is None:
            raise RuntimeError(f"chain {name} crashed")
        return spec

    monkeypatch.setattr(RS, "check_pins", check_pins)
    monkeypatch.setattr(RS, "load_references", lambda: {RS.REFIT_NAME: {"coef_sha256": {}}, RS.REGRESSION_NAME: {},
                                                        RS.COUNTS_NAME: {}})
    monkeypatch.setattr(S0B, "reference_pin_problems", lambda pins, **k: list(pin_problems))
    monkeypatch.setattr(S0B, "compare_item1", compare_item1)
    monkeypatch.setattr(S0B, "cells_upstream_problems",
                        lambda files: list(stale_l2 if is_l2(files) else stale))
    monkeypatch.setattr(S0B, "n6_record_problems", lambda path, refit: [])
    monkeypatch.setattr(S0B, "a3_case", a3_case)
    monkeypatch.setattr(S0B, "score_chain", score_chain)
    monkeypatch.setattr(S0B, "check_scripts", lambda replay_: {"passed": scripts_ok, "problems": [] if scripts_ok
                                                               else ["stale"], "scripts": {}})
    monkeypatch.setattr(S0B, "resolved_inputs", lambda files: {})
    monkeypatch.setattr(R, "RESULTS", tmp_path / "results")
    out_path = tmp_path / "out" / "stage0b.json" if out == "given" else None
    rc = S0B.run(replay, l2_replay, score=score, out=out_path)
    path = out_path or (tmp_path / "results" / (S0B.OUT_NAME if score and not absent and not absent_l2 else S0B.PARTIAL_NAME))
    rec = json.loads(path.read_text()) if path.exists() else None
    return rc, rec, calls


def test_run_case1_all_good_passes(monkeypatch, tmp_path):
    rc, rec, calls = run_world(monkeypatch, tmp_path)
    assert rc == 0 and rec["passed"] is True and rec["case"] == "case1"
    assert rec["reference_pins"]["passed"] is True


@pytest.mark.parametrize("what", ["item1_ok", "gates_ok", "scripts_ok", "regression", "picks", "stale"])
def test_run_case1_each_gate_is_in_the_verdict(monkeypatch, tmp_path, what):
    kw = {}
    chains = None
    if what in ("item1_ok", "gates_ok", "scripts_ok"):
        kw[what] = False                                           # item 1 left out of the verdict survived
    elif what == "regression":
        chains = {"L1": good_chain(regression=False)}
    elif what == "picks":
        chains = {"L1": good_chain(picks=False)}                   # L4: the exact picks are gated in case 1 too
    else:
        kw["stale"] = ["cells record: rc_tau was a, the replay file is now b"]    # a stale cells record
    rc, rec, _ = run_world(monkeypatch, tmp_path, chains=chains, **kw)
    assert rc == S0B.EXIT_FAIL and rec["passed"] is False
    if what == "stale":
        assert rec["item1"]["cells_upstream_current"]["equal"] is False and rec["item1"]["passed"] is False


def test_run_checks_clips_pinned_files_before_anything_is_routed_or_compared(monkeypatch, tmp_path):
    rc, rec, calls = run_world(monkeypatch, tmp_path)
    assert calls[0] == "check_pins"                                # M1: the reference side first
    assert calls.index("check_pins") < calls.index("item1_L1") < calls.index("score_L1")


@pytest.mark.parametrize("kw", [{"pin_error": "pins_clip.json differs from the files it names"},
                                {"pin_problems": ["graph: SHA-256 a differs from pins_clip.json's b"]}])
def test_run_with_a_bad_reference_pin_is_not_a_pass_and_scores_nothing(monkeypatch, tmp_path, kw):
    rc, rec, calls = run_world(monkeypatch, tmp_path, **kw)
    assert rc == S0B.EXIT_FAIL and rec["passed"] is False
    assert rec["reference_pins"]["passed"] is False and rec["reference_pins"]["problems"]
    assert not [c for c in calls if c.startswith("score_")]


def test_run_case2_needs_both_chains_to_pass(monkeypatch, tmp_path):
    rc, rec, _ = run_world(monkeypatch, tmp_path, case="case2", l2=True,
                           chains={"L1": good_chain(regression=False), "L2": good_chain(regression=False)})
    assert rc == 0 and rec["passed"] is True            # case 2: the 131 items need not be exact


def test_run_case2_with_only_l1_scored_is_not_a_pass(monkeypatch, tmp_path):
    rc, rec, _ = run_world(monkeypatch, tmp_path, case="case2", l2=True, chains={"L1": good_chain(), "L2": None})
    assert rec["passed"] is False and "L2" in rec["errors"]            # the mutant: L1 alone passed case 2
    rc, rec, calls = run_world(monkeypatch, tmp_path / "b", case="case2", l2=False)      # no L2 chain given
    assert rec["passed"] is False and "score_L2" not in calls


@pytest.mark.parametrize("bad", ["item3", "picks", "independent", "cpu", "item1"])
@pytest.mark.parametrize("which", ["L1", "L2"])
def test_run_case2_each_chain_must_pass_item3_and_the_cpu_made_parts_exact(monkeypatch, tmp_path, bad, which):
    chains = {"L1": good_chain(regression=False), "L2": good_chain(regression=False)}
    kw = {}
    if bad == "item1":
        kw["item1_ok" if which == "L1" else "item1_l2_ok"] = False       # M2: item 1 runs on the L2 chain too
    elif bad == "cpu":
        chains[which] = good_chain(regression=False, cpu=2)             # a changed PM or RCA episode
    else:
        chains[which] = good_chain(regression=False, **{bad: False})
    rc, rec, calls = run_world(monkeypatch, tmp_path, case="case2", l2=True, chains=chains, **kw)
    assert rec["passed"] is False and rc == S0B.EXIT_FAIL
    if bad == "item1" and which == "L2":
        assert "item1_L2" in calls and rec["item1_L2"]["passed"] is False


@pytest.mark.parametrize("case", ["case3", "pending_l2"])
def test_run_stops_in_case_3_and_pending_l2(monkeypatch, tmp_path, case):
    rc, rec, calls = run_world(monkeypatch, tmp_path, case=case)
    assert rec["passed"] is False and rec["case"] == case and not [c for c in calls if c.startswith("score_")]


def test_run_no_score_is_never_a_pass(monkeypatch, tmp_path):
    rc, rec, calls = run_world(monkeypatch, tmp_path, score=False, out=None)
    assert rc == S0B.EXIT_FAIL and rec["passed"] is False and rec["case"] == "case1" and "note" in rec


def test_run_with_an_unfinished_chain_still_writes_a_record_with_the_case(monkeypatch, tmp_path):
    # L3: 10 exited 3, so the cells step never ran; the case (from A3 alone) and the errors are what the trace needs
    for score in (False, True):
        rc, rec, calls = run_world(monkeypatch, tmp_path / str(score), case="case2", score=score, out=None,
                                   absent=("cells_seed42",))
        assert rc == S0B.EXIT_MISSING and rec["passed"] is False and rec["case"] == "case2"
        assert any("cells_seed42" in m for m in rec["missing_files"])
        assert not [c for c in calls if c.startswith("score_")]


def test_run_records_an_a3_case_that_cannot_be_computed(monkeypatch, tmp_path):
    rc, rec, _ = run_world(monkeypatch, tmp_path, a3_error="no checkpoint", out=None, absent=("A3_seed42",))
    assert rc == S0B.EXIT_MISSING and rec["case"] is None and "a3_case" in rec["errors"] and rec["passed"] is False


def test_the_partial_record_does_not_block_the_final_one(monkeypatch, tmp_path):
    rc, rec, _ = run_world(monkeypatch, tmp_path, out=None, absent=("cells_seed42",))
    assert (tmp_path / "results" / S0B.PARTIAL_NAME).exists() and not (tmp_path / "results" / S0B.OUT_NAME).exists()


# ---------------------------------------------------------------- cells record, picks file, A3 case

def test_cells_upstream_problems_names_a_stale_replay_file(tmp_path):
    files = {role: tmp_path / f"{role}.bin" for role in RC.UPSTREAM_ROLES}
    files["cells_seed42"] = tmp_path / "cells_seed42.json"
    for role in RC.UPSTREAM_ROLES:
        files[role].write_bytes(role.encode())
    up = {role: sha(role.encode()) for role in RC.UPSTREAM_ROLES}
    Path(str(files["cells_seed42"]) + ".record.json").write_text(json.dumps({"upstream_sha256": up}))
    assert S0B.cells_upstream_problems(files) == []
    files["A3_seed42"].write_bytes(b"retrained after the cells")
    assert any("A3_seed42" in p for p in S0B.cells_upstream_problems(files))
    files["A3_seed42"].unlink()
    assert any("A3_seed42" in p for p in S0B.cells_upstream_problems(files))      # a missing file is a problem
    Path(str(files["cells_seed42"]) + ".record.json").unlink()
    assert S0B.cells_upstream_problems(files)


def test_replay_files_uses_the_failed_picks_file_when_only_that_exists(tmp_path):
    (tmp_path / "picks_seed42_failed.json").write_text("{}")
    assert S0B.replay_files(tmp_path)["picks_seed42"] == tmp_path / "picks_seed42_failed.json"
    (tmp_path / "picks_seed42.json").write_text("{}")
    assert S0B.replay_files(tmp_path)["picks_seed42"] == tmp_path / "picks_seed42.json"


def a3_world(monkeypatch, tmp_path, *, d_pinned=(), d_l2=(), l2=True, gates_l2=None):
    import r9_recipe_a3 as A3
    replay = tmp_path / "replay"
    for run_, gates in (("L1", full_gates("L1")), *((("L2", gates_l2 or full_gates("L2")),) if l2 else ())):
        (replay / f"a3_{run_}").mkdir(parents=True)
        (replay / f"a3_{run_}" / "A3_seed42.pt").write_bytes(run_.encode())
        (replay / f"a3_{run_}" / "identity_gates.json").write_text(json.dumps(gates))

    def state_diff(a, b):
        return list(d_pinned) if Path(b) == Path(S0B.PINNED["A3_seed42"]) else list(d_l2)
    monkeypatch.setattr(A3, "state_diff", state_diff)
    return replay


def full_gates(run):
    import r9_recipe_a3 as A3
    g = good_gates()
    g["run"] = run
    g["config"] = {"passed": True, "config": A3.stored_config()}
    g["step1_loss"] = {"passed": True, "step1_loss": A3.PINNED["step1_loss"]}
    g["inputs"] = {"passed": True, "pinned": True,
                   "sha256": {k: A3.PINNED[k] for k in ("local_groups", "graph.npz", "bank_AIC.npz")}}
    return g


def test_a3_case_follows_the_state_diffs(monkeypatch, tmp_path):
    replay = a3_world(monkeypatch, tmp_path / "a", d_pinned=[], d_l2=["x"])
    assert S0B.a3_case(replay)["case"] == "case1"
    replay = a3_world(monkeypatch, tmp_path / "b", d_pinned=["w"], d_l2=["x"])         # the L1 difference decides
    assert S0B.a3_case(replay)["case"] == "case2"
    replay = a3_world(monkeypatch, tmp_path / "c", d_pinned=["w"], d_l2=[])
    assert S0B.a3_case(replay)["case"] == "case3"
    replay = a3_world(monkeypatch, tmp_path / "d", d_pinned=["w"], l2=False)
    assert S0B.a3_case(replay)["case"] == "pending_l2"


def test_a3_case_reads_the_l2_identity_gates_too(monkeypatch, tmp_path):
    bad = full_gates("L2")
    bad["config"]["passed"] = False
    replay = a3_world(monkeypatch, tmp_path, d_pinned=["w"], d_l2=["x"], gates_l2=bad)
    res = S0B.a3_case(replay)
    assert res["identity_gates_passed"] is False and any(p.startswith("L2:") for p in res["identity_gate_problems"])


def test_identity_gate_numbers_are_rechecked_not_trusted(monkeypatch, tmp_path):
    import r9_recipe_a3 as A3
    base = full_gates("L1")
    assert S0B.recheck_gate_numbers(base, A3.stored_config(), A3.PINNED) == []
    edits = {
        "step1": lambda g: g["step1_loss"].update(step1_loss=A3.PINNED["step1_loss"] * (1 + 2e-5)),
        "step1_missing": lambda g: g["step1_loss"].pop("step1_loss"),
        "config": lambda g: g["config"]["config"].update(lr=123.0),
        "config_tau": lambda g: g["config"]["config"].update(aspect_tau_fixed=True),
        "inputs": lambda g: g["inputs"]["sha256"].update(**{"graph.npz": "0" * 64}),
        "inputs_missing": lambda g: g["inputs"].pop("sha256"),
    }
    for name, edit in edits.items():
        g = copy.deepcopy(base)
        edit(g)
        assert S0B.recheck_gate_numbers(g, A3.stored_config(), A3.PINNED), name       # the flags still say passed
    g = copy.deepcopy(base)
    g["step1_loss"]["step1_loss"] = A3.PINNED["step1_loss"] * (1 + 0.9e-5)
    assert S0B.recheck_gate_numbers(g, A3.stored_config(), A3.PINNED) == []           # within 1e-5 relative
    replay = a3_world(monkeypatch, tmp_path, d_pinned=[], l2=False)                  # a3_case calls it
    gates = json.loads((replay / "a3_L1" / "identity_gates.json").read_text())
    gates["step1_loss"]["step1_loss"] *= 1.001
    (replay / "a3_L1" / "identity_gates.json").write_text(json.dumps(gates))
    assert S0B.a3_case(replay)["identity_gates_passed"] is False


# ---------------------------------------------------------------- M1, L1, L2 in item 1

def test_reference_pin_problems_hash_every_pinned_file_item1_compares(tmp_path):
    pinned, parts = {}, {}
    for role in ("graph", "bank_AIC", "rb_heads_rand", "cells_seed42"):
        p = tmp_path / f"{role}.bin"
        p.write_bytes(role.encode())
        pinned[role] = p
        parts[role] = {"sha256": sha(role.encode())}
    pins = {"parts": parts}
    roles = tuple(pinned)
    assert S0B.reference_pin_problems(pins, roles=roles, pinned=pinned) == []
    pinned["graph"].write_bytes(b"a replay output copied over the pinned path")
    problems = S0B.reference_pin_problems(pins, roles=roles, pinned=pinned)
    assert len(problems) == 1 and "graph" in problems[0]
    del parts["bank_AIC"]
    assert len(S0B.reference_pin_problems(pins, roles=roles, pinned=pinned)) == 2     # an unpinned role is a problem
    assert {"graph", "bank_AIC", "picks_seed42", "cells_seed42", "A3_seed42", "picked", "rc_tau",
            "baselines_seed42", "rb_reader_A0", "rb_reader_A0_json", "partitions", "n6_posteriors",
            "step1_heads_style"} <= set(S0B.REFERENCE_ROLES)
    assert {f"rb_heads_{g}" for g in ("affect", "image", "caption", "csd", "rand")} <= set(S0B.REFERENCE_ROLES)


@pytest.mark.parametrize("role", ["graph", "partitions", "rb_heads_rand", "n6_posteriors"])
def test_item1_refuses_a_replay_file_that_is_clips_pinned_file(tmp_path, role):
    pinned = write_npz(tmp_path / "pinned.npz", **base_arrays())
    link, copy_ = tmp_path / "link.npz", write_npz(tmp_path / "copy.npz", **base_arrays())
    link.symlink_to(pinned)
    res = S0B.compare_item1({role: link}, {role: pinned}, roles=(role,))
    assert not res["passed"] and "pinned" in res["roles"][role]["differences"][0]
    assert S0B.compare_item1({role: copy_}, {role: pinned}, roles=(role,))["passed"]
    assert not S0B.compare_item1({role: pinned}, {role: pinned}, roles=(role,))["passed"]


def test_n6_record_checks_the_affect_head_identity_and_the_coefficient_hashes(monkeypatch, tmp_path):
    import r9_heads as H
    coef = {"affect": {"img": "a", "txt": "b"}}
    path = tmp_path / "n6_posteriors.npz"
    rec_path = Path(str(path) + ".record.json")
    monkeypatch.setattr(H, "affect_identity", lambda prov, backbone="clip_b32": prov == {"arm": "L"})
    rec_path.write_text(json.dumps({"backbone": "clip_b32", "coef_sha256": coef, "affect_prov": {"arm": "L"}}))
    assert S0B.n6_record_problems(path, {"coef_sha256": coef}) == []
    rec_path.write_text(json.dumps({"backbone": "clip_b32", "coef_sha256": coef, "affect_prov": {"arm": "other"}}))
    assert S0B.n6_record_problems(path, {"coef_sha256": coef})
    rec_path.write_text(json.dumps({"backbone": "clip_b32", "coef_sha256": {"affect": {"img": "x"}},
                                    "affect_prov": {"arm": "L"}}))
    assert S0B.n6_record_problems(path, {"coef_sha256": coef})
    rec_path.write_text(json.dumps({"backbone": "clip_b32", "affect_prov": {"arm": "L"}}))
    assert S0B.n6_record_problems(path, {"coef_sha256": coef})                         # no hashes recorded
    rec_path.unlink()
    assert S0B.n6_record_problems(path, {"coef_sha256": coef})


def test_json_view_has_no_dead_branches():
    for role in ("picks_seed42", "cells_seed42"):
        with pytest.raises(KeyError):
            S0B.json_view(role, {})                  # these are item 3's (pick_numbers), not item 1's
    assert not set(S0B.ITEM1_ROLES) & {"picks_seed42", "cells_seed42"}


# ---------------------------------------------------------------- M2: the A3-independent items

def test_independent_items_are_the_cpu_made_ones():
    refs = RS.load_references()
    keys = list(refs[RS.REGRESSION_NAME]["items"])
    indep = [k for k in keys if S0B.is_a3_independent(k)]
    assert len(indep) == 79
    for k in indep:
        assert k.startswith(("heads_", "coef_sha256_", "bundle_coef_", "episodes_", "bundle_episodes_",
                             "per_anchor_seed42/", "seed42_arrays/cl", "seed42_arrays/pair_index",
                             "seed42_arrays/parity")), k
    for k in keys:
        if k.startswith(("aff_", "r1_", "B", "seed42_arrays/aff", "seed42_arrays/r1", "seed42_arrays/pick",
                         "seed42_arrays/margin", "seed42_arrays/P__")):
            assert not S0B.is_a3_independent(k), k
    assert {k for k in indep if k.startswith("per_anchor_seed42/")} == {k for k in keys if k.startswith("per_anchor_seed42/")}


def test_independent_problems_flag_a_changed_cpu_item_and_ignore_an_a3_fed_one():
    items = {k: {"equal": True, "got": 1, "want": 1} for k in
             ("heads_selection_bits_equal_stored", "aff_fused_r1", "per_anchor_seed42/rca__r1")}
    want = copy.deepcopy(items)
    assert S0B.independent_problems(items, want) == []
    items["aff_fused_r1"] = {"equal": False, "got": 2, "want": 1}
    assert S0B.independent_problems(items, want) == []
    items["per_anchor_seed42/rca__r1"] = {"equal": True, "got": 2, "want": 1}
    assert S0B.independent_problems(items, want)
    items["per_anchor_seed42/rca__r1"] = want["per_anchor_seed42/rca__r1"]
    del items["heads_selection_bits_equal_stored"]
    assert S0B.independent_problems(items, want)
    with pytest.raises(AssertionError):
        S0B.independent_problems({"aff_fused_r1": {}}, {"aff_fused_r1": {}})            # nothing to check: refuse


# ---------------------------------------------------------------- score_chain: what item 3 compares with what

STORED = ("aff_fused", "aff_cf", "r1_fused", "r1_cf", "cosine", "rca", "kissme")
CELLS_REC = {"aff": [39, 119], "cf": [149, 10], "r1_fused": [116, 119], "r1_counterpart": [58, 123],
             "sigma_star": [0.0, 0.0]}
TAUS4 = [0.1, 0.2, 0.3, 0.4]
INDEP_ITEM = "per_anchor_seed42/cosine__r1"
ITEMS = {"heads_selection_bits_equal_stored": {"equal": True, "got": True, "want": True},
         "aff_fused_r1": {"equal": True, "got": 1, "want": 1}, INDEP_ITEM: {"equal": True, "got": 1, "want": 1}}


def write_pick_files(folder, cells=None, taus=None, b=None, pinned_names=False):
    """The four pick files of a chain, under the replay's names (or CLIP's pinned names)."""
    folder.mkdir(parents=True, exist_ok=True)
    name = (lambda role, default: Path(BB.CLIP_PARTS[role]).name) if pinned_names else (lambda role, default: default)
    write_json(folder / name("cells_seed42", "cells_seed42.json"), cells or CELLS_REC)
    write_json(folder / name("picks_seed42", "picks_seed42.json"), {"B": b or {"0": [16.0, 8.0], "1": [8.0, 2.0]}, "B0": {"0": [1, 2]},
                                              "B1": {"1": [3, 4]}, "passed": True})
    write_json(folder / name("baselines_seed42", "baselines_seed42.json"),
               {"scorers": {n: {"lambda_picks": {"0": 0.5, "1": 0.25}} for n in P.LAMBDA_SCORERS}})
    write_json(folder / name("rc_tau", "rc_tau.json"), {"taus": taus or TAUS4})


def chain_world(monkeypatch, tmp_path, *, cells=None, taus=None, b=None, shift=None, items=None, per_anchor=None):
    """A replay chain (files in tmp_path/replay) and the pinned picks (tmp_path/pinned) with the heavy steps replaced.
    `shift`: {scorer: change of its mean R@1 in points}; `per_anchor`: {scorer: replacement array}."""
    replay, pinned = tmp_path / "replay", tmp_path / "pinned"
    write_pick_files(pinned, pinned_names=True)
    write_pick_files(replay, cells, taus, b)
    files = touch_replay_keep(replay)
    monkeypatch.setattr(S0B, "PINNED", {**S0B.PINNED, **{r: pinned / Path(S0B.PINNED[r]).name for r in
                                                         ("cells_seed42", "picks_seed42", "baselines_seed42", "rc_tau")}})
    calls = {}
    n = 4
    stored = {k: np.array([0.0, 1.0, 0.0, 1.0]) for k in STORED}
    targets = {"B": RS.EXTRA_TARGETS["B_mean_r1"], "B0": RS.EXTRA_TARGETS["B0_mean_r1"],
               "B1": RS.EXTRA_TARGETS["B1_mean_r1"], "aff_fused": RS.TARGETS["aff_fused_r1"],
               "aff_cf": RS.TARGETS["aff_cf_r1"]}
    scored = {k: {"r1": v.copy()} for k, v in stored.items()}
    for k, t in targets.items():
        scored[k] = {"r1": np.full(n, (t + (shift or {}).get(k, 0.0)) / 100)}
    for k, v in (per_anchor or {}).items():
        scored[k] = {"r1": v}

    def score_seed(bundle, picks, lambdas, readers, **kw):
        calls["score_seed"] = kw
        return scored
    monkeypatch.setattr(RS, "setup", lambda: SimpleNamespace(readers="readers", inputs={}, coef_sha256={}))
    monkeypatch.setattr(RS, "seed_bundle", lambda *a, **k: SimpleNamespace(episodes_sha256={"p": "s"}))
    monkeypatch.setattr(RS, "regression_items", lambda *a, **k: (copy.deepcopy(items or ITEMS), None))
    monkeypatch.setattr(RS, "compare_counts", lambda *a: [])
    monkeypatch.setattr(P, "check_seed42", lambda bundle: None)
    monkeypatch.setattr(P, "load_picks", lambda path, backbone: "picks")
    monkeypatch.setattr(P, "parse_lambda_picks", lambda rec: "lambdas")
    monkeypatch.setattr(SC, "score_seed", score_seed)
    import r9_stats as ST
    counts = {"checks": {"P1": {"point": 6.0}, "P3": {"point": 0.8}}}
    monkeypatch.setattr(ST, "pass_record", lambda *a, **k: copy.deepcopy(counts))
    monkeypatch.setattr(S0B, "stored_per_anchor_r1", lambda: stored)
    refs = {RS.REFIT_NAME: {}, RS.REGRESSION_NAME: {"items": copy.deepcopy(ITEMS)}, RS.COUNTS_NAME: copy.deepcopy(counts)}
    return files, refs, calls


def touch_replay_keep(replay):
    """touch_replay without overwriting the pick files written by write_pick_files."""
    files = S0B.replay_files(replay)
    for role in list(S0B.ITEM1_ROLES) + list(S0B.A3_ROLES) + list(S0B.A3_FED_ROLES):
        p = Path(files[role])
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b"x")
    return files


def test_score_chain_equal_to_the_pinned_numbers_passes(monkeypatch, tmp_path):
    files, refs, _ = chain_world(monkeypatch, tmp_path)
    res = S0B.score_chain(files, refs)
    assert res["item3"]["passed"], res["item3"]["failures"]
    assert res["regression"]["passed"] and res["independent"]["passed"] and res["cpu_changed_episodes"] == 0
    assert all(res["item3"]["picks_exact"].values()) and res["item3"]["picks_exact"]


@pytest.mark.parametrize("kw", [{"cells": {**CELLS_REC, "cf": [149, 11]}}, {"taus": [0.1, 0.2, 0.3, 0.4000001]},
                                {"b": {"0": [16.0, 8.0], "1": [8.0, 4.0]}}])
def test_score_chain_compares_the_chain_with_the_pinned_picks_not_with_itself(monkeypatch, tmp_path, kw):
    files, refs, _ = chain_world(monkeypatch, tmp_path, **kw)
    res = S0B.score_chain(files, refs)
    assert not res["item3"]["passed"] and not all(res["item3"]["picks_exact"].values())


@pytest.mark.parametrize("shift,passes", [(0.049, True), (0.051, False), (-0.051, False)])
def test_score_chain_r1_tolerance_against_the_stored_targets(monkeypatch, tmp_path, shift, passes):
    files, refs, _ = chain_world(monkeypatch, tmp_path, shift={"aff_fused": shift})
    assert S0B.score_chain(files, refs)["item3"]["passed"] is passes


def test_score_chain_passes_cells_and_taus_to_score_seed_explicitly(monkeypatch, tmp_path):
    cells = {"aff": [1, 2], "cf": [3, 4], "r1_fused": [5, 6], "r1_counterpart": [7, 8], "sigma_star": [0.0, 0.0]}
    files, refs, calls = chain_world(monkeypatch, tmp_path, cells=cells, taus=[0.5, 0.6, 0.7, 0.8])
    S0B.score_chain(files, refs)
    assert calls["score_seed"]["cells"] == SC.cells_from_record(cells)
    assert tuple(calls["score_seed"]["taus"]) == (0.5, 0.6, 0.7, 0.8)
    assert calls["score_seed"]["include_pm"] is True


def test_score_chain_an_a3_independent_item_that_differs_fails_independent(monkeypatch, tmp_path):
    bad = copy.deepcopy(ITEMS)
    bad[INDEP_ITEM] = {"equal": True, "got": 2, "want": 1}
    files, refs, _ = chain_world(monkeypatch, tmp_path, items=bad)
    res = S0B.score_chain(files, refs)
    assert not res["independent"]["passed"] and not res["regression"]["passed"]
    bad = copy.deepcopy(ITEMS)
    bad["aff_fused_r1"] = {"equal": True, "got": 2, "want": 1}               # A3-fed: case 2 allows it
    files, refs, _ = chain_world(monkeypatch, tmp_path / "b", items=bad)
    res = S0B.score_chain(files, refs)
    assert res["independent"]["passed"] and not res["regression"]["passed"]


def test_score_chain_counts_changed_cpu_episodes_apart_from_the_a3_fed_ones(monkeypatch, tmp_path):
    flipped = np.array([1.0, 1.0, 0.0, 1.0])
    files, refs, _ = chain_world(monkeypatch, tmp_path, per_anchor={"aff_fused": flipped})
    res = S0B.score_chain(files, refs)
    assert res["changed_episodes"]["per_scorer"]["aff_fused"] == 1 and res["cpu_changed_episodes"] == 0
    files, refs, _ = chain_world(monkeypatch, tmp_path / "b", per_anchor={"kissme": flipped})
    assert S0B.score_chain(files, refs)["cpu_changed_episodes"] == 1


# ---------------------------------------------------------------- the cells step reads the replay (spy)

def test_make_cells_reads_the_replay_through_every_step(monkeypatch, tmp_path):
    import r9_context as X
    import r9_recipe_groups as RG
    replay = tmp_path / "replay"
    files = touch_replay(replay)
    seen = {}

    def spy(name):
        seen[name] = {"rc_tau": BB.parts("clip_b32")["rc_tau"], "picks": BB.parts("clip_b32")["picks_seed42"],
                      "picked": BB.parts("clip_b32")["picked"], "partitions": BB.parts("clip_b32")["partitions"],
                      "reader": BB.parts("clip_b32")["rb_reader_A0"],
                      "a3": R.input_pins("clip_b32")[B.A3_REL][0]}

    monkeypatch.setattr(RC, "preflight", lambda r: None)
    monkeypatch.setattr(RC, "load_taus", lambda bb: spy("load_taus") or TAUS4)
    monkeypatch.setattr(RC, "setup", lambda bb: spy("setup") or SimpleNamespace(
        data=None, split=None, labels=None, heads=None, value_sets=None, readers=None, inputs={}))
    monkeypatch.setattr(X, "RowContext", lambda *a, **k: SimpleNamespace(episode_sha={"p": "s"}))
    monkeypatch.setattr(RC, "check_episode_sha", lambda sha: None)
    monkeypatch.setattr(RC, "cells_bundle", lambda ctx, bb: spy("cells_bundle") or "bundle")
    monkeypatch.setattr(RC, "compute_cells", lambda *a: spy("compute_cells") or ({}, {"b_mean_r1": P.MEAN_R1_TARGETS["B"]}))
    monkeypatch.setattr(RG, "read_shas", lambda bb: {})
    written = {}

    def write_cells(folder, rec, **kw):
        spy("write_cells")
        written.update(kw)
        return tmp_path / "cells_seed42.json"
    monkeypatch.setattr(RC, "write_cells", write_cells)
    (tmp_path / "cells_seed42.json").write_text("{}")
    assert S0B.make_cells(replay) == 0
    assert set(seen) == {"load_taus", "setup", "cells_bundle", "compute_cells", "write_cells"}
    for name, paths in seen.items():
        assert paths["rc_tau"] == files["rc_tau"] and paths["picks"] == files["picks_seed42"], name
        assert paths["picked"] == files["picked"] and paths["partitions"] == files["partitions"], name
        assert paths["reader"] == files["rb_reader_A0"] and paths["a3"] == files["A3_seed42"], name
    assert written["upstream_sha256"] == {role: sha(b"x") for role in RC.UPSTREAM_ROLES}     # the replay's bytes


def test_chain_item1_fails_on_any_of_its_three_parts(monkeypatch, tmp_path):
    files = {"n6_posteriors": tmp_path / "n6.npz"}
    refit = {"coef_sha256": {}}
    ok = {"passed": True, "roles": {}}
    for name, repl in (("compare_item1", {"passed": False, "roles": {}}), ("cells_upstream_problems", ["stale"]),
                       ("n6_record_problems", ["identity"])):
        monkeypatch.setattr(S0B, "compare_item1", lambda f, ok=ok: dict(ok))
        monkeypatch.setattr(S0B, "cells_upstream_problems", lambda f: [])
        monkeypatch.setattr(S0B, "n6_record_problems", lambda p, r: [])
        assert S0B.chain_item1(files, refit)["passed"] is True
        monkeypatch.setattr(S0B, name, (lambda f, r=repl: dict(r)) if name == "compare_item1"
                            else (lambda f, r=repl: list(r)) if name == "cells_upstream_problems"
                            else (lambda p, rf, r=repl: list(r)))
        res = S0B.chain_item1(files, refit)
        assert res["passed"] is False, name


def test_run_case2_with_an_incomplete_l2_chain_is_recorded_not_scored(monkeypatch, tmp_path):
    rc, rec, calls = run_world(monkeypatch, tmp_path, case="case2", l2=True, absent_l2=("cells_seed42",), out=None)
    assert rc == S0B.EXIT_MISSING and rec["passed"] is False
    assert any(m.startswith("L2 chain") and "cells_seed42" in m for m in rec["missing_files"])
    assert not [c for c in calls if c.startswith("score_")]
