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
    ok = dict(item1=True, gates=True, regression=True, item3=True, item4=True)
    assert S0B.decide("case1", **ok)
    assert not S0B.decide("case1", **{**ok, "regression": False})     # case 1: the 131 items must be exact
    assert S0B.decide("case2", **{**ok, "regression": False})         # case 2: item 3 is the gate instead
    assert not S0B.decide("case2", **{**ok, "item3": False})
    for case in ("case3", "pending_l2"):
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


def picks_record(**kw):
    rec = {"B": {"0": [16.0, 8.0], "1": [8.0, 2.0]}, "B0": {"0": [16.0, 16.0], "1": [16.0, 8.0]},
           "B1": {"0": [16.0, 16.0], "1": [16.0, 8.0]}, "mean_r1": {"B": 18.341064453125, "B0": 18.43, "B1": 18.8},
           "lambda_picks": {"rca": {"0": 0.5, "1": 0.5}}, "coef_sha256": {"affect": {"img": "aa"}},
           "episodes_sha256": {"p": "bb"}, "fit_rows_sha256": "cc", "passed": True, "time": "t1",
           "module_sha256": {"f": "1"}, "input_sha256": {"f": "2"}, "backbone": "clip_b32", "checks": {"x": 1}}
    rec.update(kw)
    return rec


def test_item1_a_differing_b_pick_fails_and_run_specific_fields_do_not(tmp_path):
    pinned = write_json(tmp_path / "p.json", picks_record())
    same = picks_record(time="t2", module_sha256={"f": "9"}, input_sha256={"g": "3"}, checks={"y": 2})
    assert S0B.compare_role("picks_seed42", write_json(tmp_path / "s.json", same), pinned) == []
    for kw in ({"B": {"0": [16.0, 8.0], "1": [8.0, 4.0]}}, {"B1": {"0": [16.0, 16.0], "1": [16.0, 16.0]}},
               {"mean_r1": {"B": 18.341064453126, "B0": 18.43, "B1": 18.8}}, {"passed": False}):
        assert S0B.compare_role("picks_seed42", write_json(tmp_path / "o.json", picks_record(**kw)), pinned), kw


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


def test_item1_cells_are_exact(tmp_path):
    cells = {"aff": [39, 119], "cf": [149, 10], "r1_fused": [116, 119], "r1_counterpart": [58, 123],
             "sigma_star": [0.0, 0.0], "switched_off": False, "gate_share": 0.5, "gate_open": 1, "gate_total": 2}
    pinned = write_json(tmp_path / "p.json", cells)
    assert S0B.compare_role("cells_seed42", write_json(tmp_path / "s.json", dict(cells)), pinned) == []
    assert S0B.compare_role("cells_seed42", write_json(tmp_path / "o.json", {**cells, "cf": [149, 11]}), pinned)


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


def test_routed_taus_and_cells_come_from_the_replay(tmp_path):
    import r9_score as S
    files = routed_files(tmp_path)
    write_json(files["rc_tau"], {"taus": [0.1, 0.2, 0.3, 0.4]})
    cells = {"aff": [1, 2], "cf": [3, 4], "r1_fused": [5, 6], "r1_counterpart": [7, 8], "sigma_star": [0.0, 0.0],
             "switched_off": False, "gate_share": 0.5, "gate_open": 1, "gate_total": 2}
    write_json(tmp_path / "replay" / "cells_seed42_replay.json", cells)
    before_cells, before_taus = copy.deepcopy(S.CELLS), R.R3.assert_taus
    with S0B.routed(files, cells=S0B.cells_for_score(cells)):
        assert tuple(R.R3.assert_taus()) == (0.1, 0.2, 0.3, 0.4)
        assert S.CELLS["aff"]["fused"] == (1, 2) and S.CELLS["r1"]["cf"] == (7, 8)
    assert S.CELLS == before_cells and R.R3.assert_taus is before_taus


# ---------------------------------------------------------------- the output file

def test_the_record_is_never_overwritten(tmp_path):
    out = tmp_path / "stage0b.json"
    S0B.check_out(out)                                # absent: fine
    out.write_text("{}")
    with pytest.raises(FileExistsError):
        S0B.check_out(out)
