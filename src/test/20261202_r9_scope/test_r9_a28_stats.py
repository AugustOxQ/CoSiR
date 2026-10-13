"""Ticket 21a: producer records through the real verdict consumer; synthetic counts only."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r9_stats as ST  # noqa: E402
import run_r9_apply_rule as AR  # noqa: E402

P = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
L1, L2 = ("P4", "P6"), ("P1", "P2", "P3", "P5", "P7")
BOUNDARY_COUNTS = dict(zip(P, (24, 30, 40, 61, 61, 124, 124)))


def scores(smoke=False):
    n = 192 if smoke else 12288
    cl = np.arange(n) % 12
    out = []
    for _ in range(3):
        s = {"cl": cl}
        for key in ("aff_fused", "aff_cf", "cosine", "rca", "B", "B0"):
            s[key] = {"r1": np.full(n, 0.5 if key == "aff_fused" else 0.25),
                      "gain": np.full(n, 0.25 if key == "aff_fused" else 0.0)}
        out.append(s)
    return out


def extra(seeds):
    return {"episodes_sha256": {s: {p: "a" * 64 for p in ST.R6C.PAIR_NAMES} for s in seeds},
            "runner_sha256": "b" * 64, "module_sha256": {"r9_stats.py": "c" * 64}}


def record(monkeypatch, counts, smoke=False, mod=ST):
    # Feed exact synthetic bootstrap counts into the real difference/record pipeline.
    # Bootstrap numerics and their point/CI cross-check already have tests in test_r9_stats.
    pending = iter(counts[nm] for nm in P)
    draws = np.linspace(-0.5, 0.5, 5000)

    def bootstrap(d, cl):
        return draws, np.arange(5000) < next(pending)

    monkeypatch.setattr(mod, "bootstrap_draws", bootstrap)
    monkeypatch.setattr(mod.C, "point_ci", lambda d, cl: mod.point_ci_from_draws(d, draws))
    seeds = [9001, 9002, 9003] if smoke else [55, 56, 57]
    return mod.pass_record_a28(scores(smoke), "smoke" if smoke else "a28", seeds, extra(seeds),
                              set_aside=[{"language": "Hindi", "pair": ST.R6C.PAIR_NAMES[0], "seed": seeds[0]}],
                              flagged_languages=["Urdu"], near_duplicates={"cos_099": 3, "cos_098": 7})


def load_record(tmp_path, rec, read="H8", smoke=False):
    path = tmp_path / f"held_pass_{read}.json"
    path.write_text(json.dumps(rec, allow_nan=False))
    return AR.load_pass(path.read_bytes(), path.name, read, smoke)


def test_a28_record_at_each_levels_boundaries(monkeypatch, tmp_path):
    rec = record(monkeypatch, BOUNDARY_COUNTS)
    loaded = load_record(tmp_path, rec)
    assert loaded["design"] == "two_level" and loaded["mode"] == "held"
    assert "secondary" not in loaded
    assert loaded["holm_order"] == {"level1": list(L1), "level2": list(L2)}
    for names, level, m in ((L1, 1, 2), (L2, 2, 5)):
        for k, nm in enumerate(names, 1):
            c = loaded["checks"][nm]
            assert (c["n"], c["level"], c["holm_k"]) == (BOUNDARY_COUNTS[nm], level, k)
            assert c["passes"] and c["own_count_passes"] and c["near_boundary"]
            assert c["level_two_sided"] == 1 - 0.05 / (m + 1 - k)
            assert c["ci_holm"] == pytest.approx(
                [100 * np.percentile(np.linspace(-0.5, 0.5, 5000), q)
                 for q in (2.5 / (m + 1 - k), 100 - 2.5 / (m + 1 - k))])
    assert loaded["set_aside"] == rec["set_aside"]
    assert loaded["flagged_languages"] == ["Urdu"]
    assert loaded["near_duplicates"] == {"cos_099": 3, "cos_098": 7}


def sensitivity(mod=ST, design="two_level"):
    # Unequal cluster sizes: sum M^2=14, N=6, SE^2=(2*14+100*6)/36.
    sig = {nm: {"sigma_a2": 2.0, "sigma_eps2": 100.0} for nm in P}
    return mod.sensitivity_record(sig, np.array([1, 2, 3]), 6, design)


@pytest.mark.parametrize("read", ["H8", "H9", "H10"])
@pytest.mark.parametrize("failure,verdict", [(None, "full claim"), ("P4", "NO-GO"), ("P1", "small claim")])
def test_round_trip_to_the_real_decision(monkeypatch, tmp_path, read, failure, verdict):
    counts = dict(BOUNDARY_COUNTS)
    if failure:
        counts[failure] += 1
    rec = record(monkeypatch, counts)
    loaded = load_record(tmp_path, rec, read)
    path = tmp_path / f"sensitivity_{read}.json"
    path.write_text(json.dumps(sensitivity(), allow_nan=False))
    sens, _ = AR.load_sensitivity(path, read)
    v = AR.decide(loaded, sens, read)
    assert v["verdict"] == verdict
    assert v["level2"]["tested"] is (failure != "P4")
    for nm in L2:
        if failure == "P4":
            assert v["checks"][nm]["passes"] is None
            assert v["checks"][nm]["reading"] is None
        elif failure == "P1" and nm != "P1":
            assert v["checks"][nm]["not_reached_after"] == "P1"
    if failure != "P4":
        assert v["n_languages"] == 26 and "Hindi" in v["claim"] and "Urdu" in v["claim"]


def test_margin_mismatch_is_the_consumers_exit_5(tmp_path):
    sens = sensitivity()
    sens["P1"]["x"] += 0.1
    path = tmp_path / "sensitivity_H8.json"
    path.write_text(json.dumps(sens))
    with pytest.raises(AR.Stop) as stopped:
        AR.load_sensitivity(path, "H8")
    assert stopped.value.code == 5 and "P1 x" in stopped.value.msg


@pytest.mark.parametrize("design", ["seven", "two_level"])
def test_sensitivity_uses_the_rules_z_and_unequal_cluster_counts(design):
    got = sensitivity(design=design)
    se = np.sqrt(628 / 36)
    assert set(got) == set(P)
    for nm in P:
        z = 3.532 if design == "seven" else 3.083 if nm in L1 else 3.417
        assert got[nm] == {"SE": se, "x": z * se, "x95": 2.80 * se}
        assert got[nm] == AR.detectable_margins({"sigma_a2": 2.0, "sigma_eps2": 100.0},
                                              np.array([1, 2, 3]), 6,
                                              "seven" if design == "seven" else "level1" if nm in L1 else "level2")


@pytest.mark.parametrize("smoke", [False, True])
def test_real_bootstrap_no_secondary_and_tie_orders(tmp_path, smoke):
    seeds = [9001, 9002, 9003] if smoke else [55, 56, 57]
    rec = ST.pass_record_a28(scores(smoke), "smoke" if smoke else "a28", seeds, extra(seeds),
                             set_aside=[], flagged_languages=[], near_duplicates={"cos_099": 0, "cos_098": 0})
    loaded = load_record(tmp_path, rec, smoke=smoke)
    assert loaded["seeds"] == seeds
    assert loaded["holm_order"] == {"level1": list(L1), "level2": list(L2)}
    assert "secondary" not in loaded
    assert all(c["n"] == 0 and c["point"] == 25.0 and c["passes"] for c in loaded["checks"].values())
    assert AR.decide(loaded, sensitivity(), "H8")["verdict"] == "full claim"


@pytest.mark.parametrize("read", ["H6", "H7"])
def test_seven_check_record_still_round_trips(tmp_path, read):
    seeds = [52, 53, 54]
    rec = ST.pass_record(scores(), "held", seeds, extra(seeds), secondary=False)
    loaded = load_record(tmp_path, rec, read)
    path = tmp_path / f"sensitivity_{read}.json"
    path.write_text(json.dumps(sensitivity(design="seven")))
    sens, _ = AR.load_sensitivity(path, read)
    assert AR.decide(loaded, sens, read)["verdict"] == "GO"
    assert loaded["holm_order"] == list(P) and "design" not in loaded


@pytest.mark.parametrize("mode,seeds", [("a28", [52, 53, 54]), ("a28", [9001, 9002, 9003]),
                                       ("smoke", [55, 56, 57]), ("held", [55, 56, 57])])
def test_a28_refuses_seeds_or_mode_of_another_path(mode, seeds):
    with pytest.raises((AssertionError, ValueError)):
        ST.pass_record_a28(scores(mode == "smoke"), mode, seeds, extra(seeds),
                           set_aside=[], flagged_languages=[], near_duplicates={"cos_099": 0, "cos_098": 0})


def test_sensitivity_uses_each_checks_own_variance_components():
    sig = {nm: {"sigma_a2": float(i), "sigma_eps2": float(10 * i)} for i, nm in enumerate(P, 1)}
    got = ST.sensitivity_record(sig, np.array([1, 2, 3]), 6, "two_level")
    for i, nm in enumerate(P, 1):
        assert got[nm]["SE"] == pytest.approx(np.sqrt(74 * i / 36))


# Mutate copies, never verdict sources in place. The checks use rule literals, so a coordinated error in the
# shared tables cannot pass just because the consumer shares that error.
MUTANTS = {
    "level split": ('("P4", "P6"), 2', '("P1", "P6"), 2', "record"),
    "level 1 m": ('("P4", "P6"), 2', '("P4", "P6"), 7', "record"),
    "level 2 m": ('("P1", "P2", "P3", "P5", "P7"), 5', '("P1", "P2", "P3", "P5", "P7"), 7', "record"),
    "level 1 z": ('"level1": 3.083', '"level1": 3.532', "sensitivity"),
    "level 2 z": ('"level2": 3.417', '"level2": 3.083', "sensitivity"),
}


def check_mutation_target(mod, which, monkeypatch):
    if which == "record":
        rec = record(monkeypatch, BOUNDARY_COUNTS, mod=mod)
        assert rec["holm_order"] == {"level1": list(L1), "level2": list(L2)}
        for names, level in ((L1, 1), (L2, 2)):
            for k, nm in enumerate(names, 1):
                c = rec["checks"][nm]
                assert (c["level"], c["holm_k"], c["passes"], c["near_boundary"]) == (level, k, True, True)
    else:
        got = sensitivity(mod)
        for nm in P:
            assert got[nm]["x"] == (3.083 if nm in L1 else 3.417) * np.sqrt(628 / 36)


@pytest.mark.parametrize("name", list(MUTANTS))
def test_each_table_mutant_fails_the_producer_check(tmp_path, monkeypatch, name):
    old, new, which = MUTANTS[name]
    check_mutation_target(ST, which, monkeypatch)
    src = Path(ST.__file__).read_text()
    assert src.count(old) == 1
    if name == "level split":
        # Swap P1 and P4 so the mutant still covers seven unique checks across two levels.
        src = src.replace('("P1", "P2", "P3", "P5", "P7"), 5',
                          '("P4", "P2", "P3", "P5", "P7"), 5')
    path = tmp_path / "r9_stats_mutant.py"
    path.write_text(src.replace(old, new))
    spec = importlib.util.spec_from_file_location("r9_stats_a28_mutant", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises((AssertionError, ValueError)):
        check_mutation_target(mod, which, monkeypatch)
