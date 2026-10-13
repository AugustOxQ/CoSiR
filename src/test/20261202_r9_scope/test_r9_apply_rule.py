"""Tests of run_r9_apply_rule.py (ticket 19): rule 4.1, 4.2, 7.4 (detectable margins), 7.6, 10 on hand-built files.

Every file is built here in tmp_path with the real names and the real counts' range (5,000 draws); nothing is read from
or written to results/. The Holm flags of the hand-built pass files come from an independent exact form (fractions:
p = (n + 1) / 5001 against 0.025 / (m + 1 - k)), and the n* tables are found here by search on that inequality, not
copied from the rule.
"""
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as CM  # noqa: E402  first, before anything that imports src
import r9_stats as ST  # noqa: E402
import run_r9_apply_rule as AR  # noqa: E402

P = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
L1 = ("P4", "P6")
L2 = ("P1", "P2", "P3", "P5", "P7")
QUANT = {"P1": "r1 AFF-COS", "P2": "r1 AFF-RCA", "P3": "r1 AFF-B", "P4": "r1 AFF-B0", "P5": "r1 AFF-CF",
         "P6": "gain AFF-CF", "P7": "gain AFF-RCA"}
SEVEN, TWO = ("H6", "H7"), ("H8", "H9", "H10")


# ---------------------------------------------------------------- independent reference (exact fractions)

def nstar(m: int, k: int) -> int:
    """Largest count n with p = (n + 1) / 5001 <= 0.025 / (m + 1 - k), found by search."""
    n = -1
    while Fraction(n + 2, 5001) <= Fraction(25, 1000) / (m + 1 - k):
        n += 1
    return n


def ref_holm(counts: dict, names: tuple, m: int) -> dict:
    """Order by count, ties in the order given; k-th passes if it and every earlier one pass at its own level."""
    order = sorted(names, key=lambda nm: (counts[nm], names.index(nm)))
    out, ok = {"order": order}, True
    for k, nm in enumerate(order, start=1):
        own = Fraction(counts[nm] + 1, 5001) <= Fraction(25, 1000) / (m + 1 - k)
        ok = ok and own
        out[nm] = {"k": k, "passes": ok, "own_count_passes": own, "near_boundary": abs(counts[nm] - nstar(m, k)) <= 1,
                   "level_two_sided": 1 - 0.05 / (m + 1 - k)}
    return out


def entry(nm, n, pt, h, level=None):
    e = {"quantity": QUANT[nm], "n": n, "point": pt, "ci95": [pt - 0.25, pt + 0.25], "holm_k": h["k"],
         "ci_holm": [pt - 0.375, pt + 0.375], "level_two_sided": h["level_two_sided"], "passes": h["passes"],
         "own_count_passes": h["own_count_passes"], "near_boundary": h["near_boundary"]}
    if level:
        e["level"] = level
    return e


def make_pass(read, counts, points=None, smoke=False, set_aside=(), flagged=()) -> dict:
    counts = dict(zip(P, counts)) if not isinstance(counts, dict) else counts
    points = points or {}
    seeds = [9001, 9002, 9003] if smoke else ([52, 53, 54] if read in SEVEN else [55, 56, 57])
    n_ep = 9 * (64 if smoke else 4096)
    pt = lambda nm: points.get(nm, 0.5)  # noqa: E731
    rec = {"rule_sha256": CM.RULE_SHA256, "mode": "smoke" if smoke else "held", "seeds": seeds, "n_episodes": n_ep,
           "n_clusters": 746, "episodes_sha256": {}, "runner_sha256": "b" * 64, "module_sha256": {"x.py": "c" * 64},
           "time": "2026-10-14 03:14:15"}
    if read in SEVEN:
        h = ref_holm(counts, P, 7)
        rec["checks"] = {nm: entry(nm, counts[nm], pt(nm), h[nm]) for nm in P}
        rec["holm_order"] = h["order"]
        rec["secondary"] = {"S1": {"n": 3}, "S2": {"n": 4}}      # may be present (pass_record writes it); never used
    else:
        h1, h2 = ref_holm(counts, L1, 2), ref_holm(counts, L2, 5)
        rec["checks"] = {nm: entry(nm, counts[nm], pt(nm), h1[nm], 1) for nm in L1}
        rec["checks"].update({nm: entry(nm, counts[nm], pt(nm), h2[nm], 2) for nm in L2})
        rec["checks"] = {nm: rec["checks"][nm] for nm in P}
        rec["holm_order"] = {"level1": h1["order"], "level2": h2["order"]}
        rec["design"] = "two_level"
        rec["set_aside"] = [dict(s) for s in set_aside]
        rec["flagged_languages"] = list(flagged)
    return rec


def make_sens(read, se=0.1) -> dict:
    rec = {}
    for nm in P:
        kind = AR.kind_of(read, nm)
        rec[nm] = {"SE": se, "x": AR.Z[kind] * se, "x95": 2.80 * se}
    rec.update({"module_sha256": {}, "time": "2026-10-14 03:00:00"})
    return rec


def stage(tmp_path, read, counts, points=None, smoke=False, sfx="", sens=True, **kw):
    d = tmp_path / ("smoke" if smoke else "results")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"held_pass_{read}{sfx}.json").write_text(json.dumps(make_pass(read, counts, points, smoke, **kw)))
    if sens:
        (d / f"sensitivity_{read}{sfx}.json").write_text(json.dumps(make_sens(read)))
    return d


ALL0 = dict.fromkeys(P, 0)


def run(tmp_path, read, counts, points=None, **kw):
    d = stage(tmp_path, read, counts, points, **{k: v for k, v in kw.items() if k in ("set_aside", "flagged")})
    return AR.apply(read, results=d)[1]


# ---------------------------------------------------------------- n* tables from the inequality

@pytest.mark.parametrize("m", [7, 2, 5])
def test_nstar_found_by_search_equals_stats_and_verdict(m, tmp_path):
    table = [nstar(m, k) for k in range(1, m + 1)]
    assert table == [ST.boundary(k, m) for k in range(1, m + 1)]
    assert table == sorted(table) and table[-1] == nstar(2, 2) == 124         # last rank is always 0.025 / 1
    # end to end: the k-th smallest count at n* passes, at n* + 1 it fails; the verdict records n*
    names = {7: P, 2: L1, 5: L2}[m]
    read = "H6" if m == 7 else "H8"
    for k in range(1, m + 1):
        base = dict(ALL0)
        # counts ascending by rank: ranks below k sit at their own n*, rank k at n* (pass) or n* + 1 (fail)
        for pos, nm in enumerate(names, start=1):
            base[nm] = table[pos - 1] if pos <= k else 124
        if m != 7:        # the other level's checks pass trivially at 0
            base.update({nm: 0 for nm in (L2 if m == 2 else L1)})
        v = run(tmp_path / f"a{m}_{k}", read, base)
        c = v["checks"][names[k - 1]]
        assert c["holm_k"] == k and c["n_star"] == table[k - 1] and c["passes"]
        base[names[k - 1]] = table[k - 1] + 1        # later names stay at 124 >= n* + 1, so the rank is kept
        if k == m:
            base[names[k - 1]] = 125
        v = run(tmp_path / f"b{m}_{k}", read, base)
        c = v["checks"][names[k - 1]]
        assert c["holm_k"] == k and not c["passes"] and not c["own_count_passes"]


# ---------------------------------------------------------------- H6, H7: Holm over seven

@pytest.mark.parametrize("read", SEVEN)
def test_seven_all_pass_is_go(read, tmp_path):
    v = run(tmp_path, read, ALL0)
    assert v["verdict"] == "GO" and v["kind"] is None and v["failed"] == []
    assert v["holm_order"] == list(P)                       # all counts tied: ties P1 to P7
    short = {"H6": "Qwen", "H7": "SigLIP 2"}[read]
    assert v["claim"].startswith(f"on ArtELingo's held paintings, AFF re-fitted by its fixed recipe on frozen {short} features beats cosine")
    assert "(or SigLIP 2)" not in v["claim"] and v["claim"].endswith("Not each pair, not another dataset, not a margin size.")
    assert "secondary" not in v and "S1" not in json.dumps(v["checks"])


def test_seven_one_fail_is_no_go_and_later_checks_not_reached(tmp_path):
    # sorted: P1 P2 P4 P6 (0, k=1..4), P3=41 (k=5, needs <= 40: fails), P5=50 (k=6, <= 61), P7=51 (k=7, <= 124)
    counts = dict(ALL0, P3=41, P5=50, P7=51)
    v = run(tmp_path, "H6", counts, points={"P3": 0.2, "P5": 0.4, "P7": -0.1})
    assert v["verdict"] == "NO-GO" and v["failed"] == ["P3", "P5", "P7"]
    assert v["checks"]["P3"]["reading_kind"] == "inconclusive" and v["checks"]["P3"]["not_reached_after"] is None
    for nm in ("P5", "P7"):
        assert v["checks"][nm]["own_count_passes"] and v["checks"][nm]["not_reached_after"] == "P3"
        assert v["checks"][nm]["reading"].startswith("not reached: the Holm procedure stopped at P3")
    assert v["checks"]["P7"]["reading_kind"] == "did not beat"
    assert v["kind"] == "did not beat"                      # one failed check reads "did not beat"
    assert v["claim"] is None


def test_all_inconclusive_is_inconclusive(tmp_path):
    v = run(tmp_path, "H7", dict(ALL0, P1=500), points={"P1": 0.3})
    assert v["verdict"] == "NO-GO" and v["kind"] == "inconclusive"
    assert v["checks"]["P1"]["holm_k"] == 7 and v["checks"]["P1"]["reading_kind"] == "inconclusive"


# ---------------------------------------------------------------- H8 to H10: two levels

@pytest.mark.parametrize("read", TWO)
def test_two_level_full_claim(read, tmp_path):
    v = run(tmp_path, read, ALL0)
    assert v["verdict"] == "full claim" and v["kind"] is None and v["failed"] == []
    assert v["holm_order"] == {"level1": ["P4", "P6"], "level2": ["P1", "P2", "P3", "P5", "P7"]}   # ties
    assert v["level2"]["tested"] is True
    assert "27 languages that entered the read" in v["claim"] and v["n_languages"] == 27
    assert v["backbone"] == AR.READS[read]["backbone"]


def test_level1_tie_order_p4_first_and_boundary_of_two(tmp_path):
    # both at 61: P4 is k=1 (needs <= 61) and P6 k=2 (<= 124); at 62 P4 would fail
    v = run(tmp_path, "H8", dict(ALL0, P4=61, P6=61))
    assert v["holm_order"]["level1"] == ["P4", "P6"] and v["verdict"] == "full claim"
    v = run(tmp_path / "b", "H8", dict(ALL0, P4=62, P6=62))
    assert v["verdict"] == "NO-GO" and not v["checks"]["P4"]["passes"]
    assert v["checks"]["P6"]["own_count_passes"] and v["checks"]["P6"]["not_reached_after"] == "P4"


def test_level1_fail_leaves_level2_untested(tmp_path):
    counts = dict(ALL0, P4=300, P1=900, P2=3, P3=0, P5=2000, P7=1)        # level-2 counts would mostly fail
    v = run(tmp_path, "H9", counts, points={"P4": 0.3, "P1": 0.9, "P2": -0.2})
    assert v["verdict"] == "NO-GO" and v["kind"] == "inconclusive" and v["failed"] == ["P4"]
    assert v["level1"]["passes"] is False and v["level2"]["tested"] is False
    for nm in L2:
        c = v["checks"][nm]
        assert c["passes"] is None and c["reading"] is None and c["reading_kind"] is None
        assert c["point"] == {"P1": 0.9, "P2": -0.2}.get(nm, 0.5) and len(c["ci95"]) == 2
        assert "holm_k" not in c and "n_star" not in c
    assert v["claim"] is None


def test_level2_fail_gives_small_claim_and_reads_with_level2_x(tmp_path):
    # all five level-2 counts at 25: k=1 needs <= 24 and fails; ranks 2 to 5 pass on their own counts
    counts = dict(ALL0, P1=25, P2=25, P3=25, P5=25, P7=25)
    v = run(tmp_path, "H10", counts, points={"P1": 0.4, "P2": 0.2, "P3": -0.3, "P5": 0.1, "P7": 0.1})
    assert v["verdict"] == "small claim" and v["kind"] is None
    assert v["level1"]["passes"] is True and v["level2"]["tested"] is True
    assert v["failed"] == list(L2)
    assert v["checks"]["P1"]["not_reached_after"] is None and v["checks"]["P1"]["reading_kind"] == "inconclusive"
    for nm in ("P2", "P3", "P5", "P7"):
        assert v["checks"][nm]["not_reached_after"] == "P1"
    assert v["checks"]["P3"]["reading_kind"] == "did not beat"
    assert v["checks"]["P1"]["x"] == pytest.approx(3.417 * 0.1)              # level-2 x
    assert v["checks"]["P4"]["passes"] and v["checks"]["P6"]["passes"]
    assert v["claim"].startswith("beats B′(A0) on R@1 with a positive condition gain over its matched control")


def test_level_two_ties_and_order(tmp_path):
    v = run(tmp_path, "H8", dict(ALL0, P1=5, P2=5, P3=5, P5=5, P7=5))
    assert v["holm_order"]["level2"] == ["P1", "P2", "P3", "P5", "P7"] and v["verdict"] == "full claim"
    v = run(tmp_path / "o", "H8", dict(ALL0, P1=40, P7=3))
    assert v["holm_order"]["level2"] == ["P2", "P3", "P5", "P7", "P1"]       # by count, ties in the given order
    assert v["checks"]["P1"]["holm_k"] == 5 and v["checks"]["P1"]["n_star"] == 124


def test_claim_names_set_aside_and_flagged(tmp_path):
    sa = [{"language": "Hindi", "pair": "emotion__style", "seed": 55},
          {"language": "Hindi", "pair": "style__genre", "seed": 56}, {"language": "Urdu", "pair": "emotion__genre", "seed": 57}]
    v = run(tmp_path, "H8", ALL0, set_aside=sa, flagged=["Arabic"])
    assert v["n_languages"] == 25 and "25 languages that entered the read" in v["claim"]
    assert "Hindi (emotion__style, seed 55)" in v["claim"] and "Urdu (emotion__genre, seed 57)" in v["claim"]
    assert "Arabic" in v["claim"] and v["set_aside"] == sa and v["flagged_languages"] == ["Arabic"]
    v = run(tmp_path / "s", "H8", dict(ALL0, P4=300), set_aside=sa[:1], flagged=["Arabic"])
    assert v["claim"] is None and v["set_aside"] == sa[:1]                    # named in the record even without a claim


# ---------------------------------------------------------------- boundary listing (rule 7.6)

def test_boundary_lists_counts_within_one_of_nstar(tmp_path):
    # seven: ranks 3 and 5 sit at n* + 1 (25) and n* (40), rank 6 at n* - 1 (60)
    counts = dict(ALL0, P3=25, P4=39, P5=40, P6=60, P7=100)
    v = run(tmp_path, "H6", counts)
    got = {b["check"]: b for b in v["boundary_report"]}
    exp = {}
    order = ref_holm(counts, P, 7)
    for nm in P:
        if abs(counts[nm] - nstar(7, order[nm]["k"])) <= 1:
            exp[nm] = (counts[nm], nstar(7, order[nm]["k"]))
    assert exp and {nm: (b["n"], b["n_star"]) for nm, b in got.items()} == exp
    for b in got.values():
        assert b["rederivation_n"] is None and "integer rule decides" in b["note"]


def test_boundary_untested_level2_not_listed_tested_listed(tmp_path):
    v = run(tmp_path, "H8", dict(ALL0, P4=61, P6=124, P1=24, P2=30, P3=40, P5=61, P7=124))
    assert {b["check"] for b in v["boundary_report"]} == set(P)
    assert all(b["n"] == b["n_star"] for b in v["boundary_report"])
    v = run(tmp_path / "n", "H8", dict(ALL0, P4=62, P6=124, P1=24))     # level 1 fails: level 2 untested
    assert {b["check"] for b in v["boundary_report"]} == {"P4", "P6"}


# ---------------------------------------------------------------- detectable margins: one x per read kind

def test_x_per_kind_on_hand_computed_numbers():
    # M = (3, 1), N = 4, sigma_a2 = 1, sigma_e2 = 2: SE^2 = (1 * (9 + 1) + 2 * 4) / 16 = 18 / 16
    se = math.sqrt(18 / 16)
    assert se == pytest.approx(1.0606601717798212, rel=1e-12)
    sig = {"sigma_a2": 1.0, "sigma_eps2": 2.0}
    import numpy as np
    M = np.array([3, 1])
    for kind, z in (("seven", 3.532), ("level1", 3.083), ("level2", 3.417)):
        m = AR.detectable_margins(sig, M, 4, kind)
        assert m["SE"] == pytest.approx(se, rel=1e-12)
        assert m["x"] == pytest.approx(z * 1.0606601717798212, rel=1e-12)
        assert m["x95"] == pytest.approx(2.80 * 1.0606601717798212, rel=1e-12)
    # the three z's are the strictest Holm level's z at one-sided alpha plus z at 0.8 power
    from statistics import NormalDist
    nd = NormalDist()
    for kind, m in (("seven", 7), ("level1", 2), ("level2", 5)):
        assert AR.Z[kind] == pytest.approx(nd.inv_cdf(1 - 0.025 / m) + nd.inv_cdf(0.8), abs=6e-4)


def test_failed_check_reading_uses_the_x_of_its_read_kind(tmp_path):
    se = 0.1
    v = run(tmp_path, "H6", dict(ALL0, P1=500), points={"P1": 0.3})
    assert v["checks"]["P1"]["x"] == pytest.approx(3.532 * se) and v["checks"]["P1"]["x95"] == pytest.approx(0.28)
    assert "x = 0.353 points" in v["checks"]["P1"]["c11_reading"]
    v = run(tmp_path / "l1", "H8", dict(ALL0, P4=500), points={"P4": 0.3})
    assert v["checks"]["P4"]["x"] == pytest.approx(3.083 * se) and "x = 0.308 points" in v["checks"]["P4"]["c11_reading"]
    v = run(tmp_path / "l2", "H8", dict(ALL0, P1=500), points={"P1": 0.3})
    assert v["checks"]["P1"]["x"] == pytest.approx(3.417 * se) and "x = 0.342 points" in v["checks"]["P1"]["c11_reading"]
    assert v["checks"]["P1"]["half_width_95"] == pytest.approx(0.25)


def test_wrong_x_in_sensitivity_file_stops(tmp_path):
    d = stage(tmp_path, "H8", ALL0)
    s = json.loads((d / "sensitivity_H8.json").read_text())
    s["P1"]["x"] = 3.083 * 0.1               # level-1 z on a level-2 check
    (d / "sensitivity_H8.json").write_text(json.dumps(s))
    with pytest.raises(AR.Stop) as e:
        AR.apply("H8", results=d)
    assert e.value.code == 5 and not (d / "held_verdict_H8.json").exists()


# ---------------------------------------------------------------- files, refusals, wordings

def test_verdict_is_never_overwritten(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    path, _ = AR.apply("H6", results=d)
    assert path.name == "held_verdict_H6.json"
    first = path.read_bytes()
    with pytest.raises(AR.Stop) as e:
        AR.apply("H6", results=d)
    assert e.value.code == 4 and path.read_bytes() == first


def test_pass_file_holm_flags_must_match_recomputation(tmp_path):
    d = stage(tmp_path, "H6", dict(ALL0, P3=500))
    r = json.loads((d / "held_pass_H6.json").read_text())
    r["checks"]["P3"]["passes"] = True
    (d / "held_pass_H6.json").write_text(json.dumps(r))
    with pytest.raises(AR.Stop) as e:
        AR.apply("H6", results=d)
    assert e.value.code == 5
    d = stage(tmp_path / "two", "H8", ALL0)
    r = json.loads((d / "held_pass_H8.json").read_text())
    r["checks"]["P1"]["holm_k"] = 1 + r["checks"]["P1"]["holm_k"]      # a level-2 rank that is not the recomputed one
    (d / "held_pass_H8.json").write_text(json.dumps(r))
    with pytest.raises(AR.Stop) as e:
        AR.apply("H8", results=d)
    assert e.value.code == 5


@pytest.mark.parametrize("edit,read", [
    (lambda r: r.update(rule_sha256="0" * 64), "H6"), (lambda r: r.update(mode="smoke"), "H6"),
    (lambda r: r.update(seeds=[55, 56, 57]), "H6"), (lambda r: r.update(seeds=[52, 53, 54]), "H8"),
    (lambda r: r["checks"].pop("P7"), "H6"), (lambda r: r["checks"]["P2"].update(n=5001), "H6"),
    (lambda r: r["checks"]["P4"].update(level=2), "H8"), (lambda r: r.update(n_episodes=100), "H8"),
    (lambda r: r.pop("set_aside"), "H8"), (lambda r: r["checks"]["P1"].update(point=float("nan")), "H6"),
])
def test_bad_pass_files_refuse(tmp_path, edit, read):
    d = stage(tmp_path, read, ALL0)
    p = d / f"held_pass_{read}.json"
    r = json.loads(p.read_text())
    edit(r)
    p.write_text(json.dumps(r))
    with pytest.raises(AR.Stop) as e:
        AR.apply(read, results=d)
    assert e.value.code == 4 and not (d / f"held_verdict_{read}.json").exists()


def test_missing_inputs_refuse(tmp_path):
    d = stage(tmp_path, "H6", ALL0, sens=False)
    with pytest.raises(AR.Stop) as e:
        AR.apply("H6", results=d)
    assert e.value.code == 4
    with pytest.raises(AR.Stop):
        AR.apply("H7", results=d)
    with pytest.raises(AR.Stop):
        AR.apply("H11", results=d)


def test_variants_fix1_and_reserve(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    stage(tmp_path, "H6", dict(ALL0, P1=500), sfx="_fix1", sens=False)       # the fix pass reuses the sensitivity file
    path, v = AR.apply("H6", variant="fix1", results=d)
    assert path.name == "held_verdict_H6_fix1.json" and v["verdict"] == "NO-GO" and v["variant"] == "fix1"
    assert v["sensitivity_file"] == "sensitivity_H6.json"
    with pytest.raises(AR.Stop):                                             # reserve needs the original verdict
        AR.apply("H6", variant="reserve", results=d)
    AR.apply("H6", results=d)
    stage(tmp_path, "H6", ALL0, sfx="_reserve")
    path, v = AR.apply("H6", variant="reserve", results=d)
    assert path.name == "held_verdict_H6_reserve.json" and v["sensitivity_file"] == "sensitivity_H6_reserve.json"


def test_smoke_mode_needs_smoke_files(tmp_path):
    d = stage(tmp_path, "H8", ALL0, smoke=True)
    path, v = AR.apply("H8", smoke=True, results=d)
    assert v["mode"] == "smoke" and v["seeds"] == [9001, 9002, 9003]
    d2 = stage(tmp_path / "r", "H8", ALL0, smoke=True)
    with pytest.raises(AR.Stop):                                             # a smoke pass file in real mode
        AR.apply("H8", results=d2)


def test_not_run_wordings(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    path, v = AR.apply("H7", not_run="switched_off", results=d)
    assert v["verdict"] is None and v["status"] == "not run: switched off" and v["claim"] is None
    path2, v2 = AR.apply("H9", not_run="dropped", results=d)
    assert v2["status"] == "dropped under the rule" and json.loads(path2.read_text())["status"] == v2["status"]
    d2 = stage(tmp_path / "w", "H6", ALL0)
    with pytest.raises(AR.Stop):                                             # a read that has a pass file was run
        AR.apply("H6", not_run="dropped", results=d2)
    with pytest.raises(AR.Stop):
        AR.apply("H6", not_run="failed", results=d)


def test_fixed_phrases_are_in_the_rule():
    AR.assert_rule()
    text = " ".join(CM.HERE.joinpath("DECISION_RULE.md").read_text(encoding="utf-8").split())
    r6 = " ".join((CM.TEST / CM.RULE_REL).read_text(encoding="utf-8").split())
    assert all(p in text for p in AR.RULE_PHRASES) and all(p in r6 for p in AR.R6_PHRASES)


def test_main_exit_codes_and_output(tmp_path, capsys, monkeypatch):
    d = stage(tmp_path, "H8", dict(ALL0, P4=500), points={"P4": 0.3})
    monkeypatch.setattr(AR, "RESULTS", d)
    assert AR.main(["--read", "H8"]) == 0
    out = capsys.readouterr().out
    assert "NO-GO" in out and "written" in out
    with pytest.raises(SystemExit) as e:
        AR.main(["--read", "H8"])
    assert e.value.code == 4


def test_point_exactly_zero_reads_did_not_beat(tmp_path):
    v = run(tmp_path, "H6", dict(ALL0, P1=500), points={"P1": 0.0})
    assert v["checks"]["P1"]["reading_kind"] == "did not beat" and "x" not in v["checks"]["P1"]
    assert v["checks"]["P1"]["reading"] == "AFF did not beat cosine on new paintings" and v["kind"] == "did not beat"
