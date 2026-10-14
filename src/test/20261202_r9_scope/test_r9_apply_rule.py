"""Tests of run_r9_apply_rule.py (ticket 19): rule 4.1, 4.2, 7.3, 7.4 (detectable margins), 7.6, 10 on hand-built files.

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
QUANT = {nm: ST.QUANTITIES[nm][2] for nm in ("P1", "P2", "P3", "P4", "P5", "P6", "P7")}      # the pass file's labels
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
    rec = {"rule_sha256": CM.RULE_SHA256, "read": read, "backbone": AR.READ_BACKBONE_IDS[read],
           "mode": "smoke" if smoke else "held", "seeds": seeds, "n_episodes": n_ep,
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
        rec["near_duplicates"] = {"cos_099": 3, "cos_098": 7}
    return rec


def make_sens(read, se=0.1) -> dict:
    rec = {}
    for nm in P:
        kind = AR.kind_of(read, nm)
        rec[nm] = {"SE": se, "x": AR.Z[kind] * se, "x95": 2.80 * se}
    rec.update({"module_sha256": {}, "time": "2026-10-14 03:00:00"})
    return rec


# Items as ticket 25's rd9_compare.py wrote them (a real run of its code on a small synthetic pass, 2026-10-14): a
# value entry has key, equal, ours, theirs; an array entry has key and equal. This is a shortened list of the 130 real
# entries; "compared" is len(items) here, as in the real file.
REAL_ITEMS = [
    {"key": "n_episodes", "equal": True, "ours": 40, "theirs": 40},
    {"key": "n_clusters", "equal": True, "ours": 10, "theirs": 10},
    {"key": "seeds.0", "equal": True, "ours": 52, "theirs": 52},
    {"key": "episodes_sha256.x", "equal": True, "ours": "a" * 64, "theirs": "a" * 64},
    {"key": "checks.P1.n", "equal": True, "ours": 30, "theirs": 30},
    {"key": "checks.P1.point", "equal": True, "ours": 10.0, "theirs": 10.0},
    {"key": "checks.P1.ci95.0", "equal": True, "ours": 2.5, "theirs": 2.5},
    {"key": "checks.P1.tested", "equal": True, "ours": True, "theirs": True},
    {"key": "checks.P1.passes", "equal": True, "ours": False, "theirs": False},
    {"key": "holm_order.0", "equal": True, "ours": "P6", "theirs": "P6"},
    {"key": "arrays.cl", "equal": True},
    {"key": "arrays.seed_index", "equal": True},
]


def write_agreement(d, read, sfx="", smoke=False, over=None):
    """The phase-2 agreement in ticket 25's names (rd9_compare.py per-read output), bound to the pass file as it is now."""
    raw = (d / f"held_pass_{read}{sfx}.json").read_bytes()
    checks = json.loads(raw)["checks"]
    items = [dict(x) for x in REAL_ITEMS]
    rec = {"agreement": True, "compared": len(items), "failed": 0, "items": items,
           "independent_identity": {"record_sha256": "1" * 64, "arrays_sha256": "2" * 64},
           "read": read, "backbone": "clip_b32", "phase": 2, "smoke": smoke,
           "pass_file": f"held_pass_{read}{sfx}.json", "held_pass_sha256": CM.sha256_bytes(raw),
           "rederived_counts": {nm: checks.get(nm, {}).get("n", 0) for nm in P}}
    rec.update(over or {})
    (d / f"rd9_{read}_agreement{sfx}.json").write_text(json.dumps(rec))


def write_smoke_record(d, name="smoke_record.json", **over):
    rec = {"passed": True, "module_sha256": CM.r9_module_shas()}
    rec.update(over)
    (d / name).write_text(json.dumps(rec))


def stage(tmp_path, read, counts, points=None, smoke=False, sfx="", sens=True, agree=True, **kw):
    d = tmp_path / ("smoke" if smoke else "results")
    d.mkdir(parents=True, exist_ok=True)
    (d / f"held_pass_{read}{sfx}.json").write_text(json.dumps(make_pass(read, counts, points, smoke, **kw)))
    if sens:
        (d / f"sensitivity_{read}{sfx}.json").write_text(json.dumps(make_sens(read)))
    if agree:
        write_agreement(d, read, sfx, smoke)
    if not smoke and not (d / "smoke_record.json").exists():
        write_smoke_record(d)
    return d


def edit_pass(d, read, fn, sfx=""):
    """Edit the pass file, then rewrite its agreement (so that only the edit is wrong)."""
    p = d / f"held_pass_{read}{sfx}.json"
    r = json.loads(p.read_text())
    fn(r)
    p.write_text(json.dumps(r))
    write_agreement(d, read, sfx)


def call(read, d, **kw):
    """AR.apply with the agreement files in d (as staged) and a ledger that is not the real one."""
    kw.setdefault("agreement_dir", d)
    kw.setdefault("ledger", d / "ledger.md")
    return AR.apply(read, results=d, **kw)


def ledger(d, rows):
    """A ledger with the header and the given {read: text} rows."""
    lines = ["| # | Date | Dataset | Backbone | Purpose | Episode | Script | Report |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r} | 2026-10-14 | x | y | z | (pending) | abc | {t} |" for r, t in rows.items()]
    (d / "ledger.md").write_text("\n".join(lines) + "\n")


ALL0 = dict.fromkeys(P, 0)


def run(tmp_path, read, counts, points=None, **kw):
    d = stage(tmp_path, read, counts, points, **{k: v for k, v in kw.items() if k in ("set_aside", "flagged")})
    return call(read, d)[1]


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
    assert v["claim"].startswith("AFF on SigLIP 2 so400m/14-384 beats B′(A0) on R@1 with a positive condition gain "
                                 "over its matched control")


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
        assert b["rederivation_n"] == b["n"] and "integer rule decides" in b["note"]


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
        call("H8", d)
    assert e.value.code == 5 and not (d / "held_verdict_H8.json").exists()


# ---------------------------------------------------------------- files, refusals, wordings

def test_verdict_is_never_overwritten(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    path, _ = call("H6", d)
    assert path.name == "held_verdict_H6.json"
    first = path.read_bytes()
    with pytest.raises(AR.Stop) as e:
        call("H6", d)
    assert e.value.code == 4 and path.read_bytes() == first


def test_pass_file_holm_flags_must_match_recomputation(tmp_path):
    d = stage(tmp_path, "H6", dict(ALL0, P3=500))
    edit_pass(d, "H6", lambda r: r["checks"]["P3"].update(passes=True))
    with pytest.raises(AR.Stop) as e:
        call("H6", d)
    assert e.value.code == 5
    d = stage(tmp_path / "two", "H8", ALL0)
    edit_pass(d, "H8", lambda r: r["checks"]["P1"].update(holm_k=1 + r["checks"]["P1"]["holm_k"]))   # not the recomputed rank
    with pytest.raises(AR.Stop) as e:
        call("H8", d)
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
    edit_pass(d, read, edit)
    with pytest.raises(AR.Stop) as e:
        call(read, d)
    assert e.value.code == 4 and not (d / f"held_verdict_{read}.json").exists()


def test_missing_inputs_refuse(tmp_path):
    d = stage(tmp_path, "H6", ALL0, sens=False)
    with pytest.raises(AR.Stop) as e:
        call("H6", d)
    assert e.value.code == 4
    with pytest.raises(AR.Stop):
        call("H7", d)
    with pytest.raises(AR.Stop):
        call("H11", d)


def refused(code, read, d, match=None, **kw):
    """call() must stop with `code`, say `match`, and leave no verdict file of its name behind."""
    before = sorted(p.name for p in d.glob("held_verdict_*.json"))
    with pytest.raises(AR.Stop) as e:
        call(read, d, **kw)
    assert e.value.code == code, e.value.msg
    if match is not None:
        assert match in e.value.msg, e.value.msg
    assert sorted(p.name for p in d.glob("held_verdict_*.json")) == before
    return e.value


def test_fix_pass_replaces_the_original_and_the_verdict_is_unsuffixed(tmp_path):
    # R6 rule 8.1: when held_pass_<Hn>_fix1.json exists it is the pass; two verdicts for one read are impossible
    d = stage(tmp_path, "H6", dict(ALL0, P1=500))                            # the original pass: NO-GO
    stage(tmp_path, "H6", ALL0, sfx="_fix1", sens=False)                     # the fix pass (GO) reuses the sensitivity
    refused(4, "H6", d, "fix pass needs no variant", variant="fix1")
    path, v = call("H6", d)
    assert path.name == "held_verdict_H6.json" and not (d / "held_verdict_H6_fix1.json").exists()
    assert v["verdict"] == "GO" and v["pass_file"] == "held_pass_H6_fix1.json" and v["fix_pass"] is True
    assert v["agreement_file"] == "rd9_H6_agreement_fix1.json" and v["sensitivity_file"] == "sensitivity_H6.json"
    refused(4, "H6", d, "never overwritten")                                 # one verdict per read


def test_fix_pass_needs_its_own_agreement_and_the_original_pass(tmp_path):
    d = stage(tmp_path, "H6", dict(ALL0, P1=500))
    stage(tmp_path, "H6", ALL0, sfx="_fix1", sens=False, agree=False)
    refused(4, "H6", d, "rd9_H6_agreement_fix1.json is missing")             # the original's agreement does not count
    write_agreement(d, "H6", "_fix1")
    (d / "held_pass_H6.json").rename(d / "held_pass_H6.moved")
    refused(4, "H6", d, "original held_pass_H6.json")
    (d / "held_pass_H6.moved").rename(d / "held_pass_H6.json")
    assert call("H6", d)[1]["verdict"] == "GO"


def test_fix_pass_uses_its_own_sensitivity_when_it_has_one(tmp_path):
    d = stage(tmp_path, "H6", dict(ALL0, P1=500))
    stage(tmp_path, "H6", ALL0, sfx="_fix1")
    assert call("H6", d)[1]["sensitivity_file"] == "sensitivity_H6_fix1.json"


def test_reserve_needs_the_original_verdict_and_its_own_files(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    stage(tmp_path, "H6", ALL0, sfx="_reserve")
    write_smoke_record(d, "smoke_record_reserve.json")
    refused(4, "H6", d, "original held_verdict_H6.json", variant="reserve")   # every other input is in place
    call("H6", d)
    path, v = call("H6", d, variant="reserve")
    assert path.name == "held_verdict_H6_reserve.json" and v["variant"] == "reserve"
    assert v["sensitivity_file"] == "sensitivity_H6_reserve.json" and v["pass_file"] == "held_pass_H6_reserve.json"
    assert v["agreement_file"] == "rd9_H6_agreement_reserve.json"
    assert v["smoke_record"]["name"] == "smoke_record_reserve.json"


def test_reserve_needs_its_own_agreement_and_smoke_record(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    stage(tmp_path, "H6", ALL0, sfx="_reserve", agree=False)
    call("H6", d)
    write_smoke_record(d, "smoke_record_reserve.json")
    refused(4, "H6", d, "rd9_H6_agreement_reserve.json is missing", variant="reserve")
    write_agreement(d, "H6", "_reserve")
    (d / "smoke_record_reserve.json").unlink()
    refused(4, "H6", d, "smoke_record_reserve.json is missing", variant="reserve")


def test_smoke_mode_needs_smoke_files(tmp_path):
    d = stage(tmp_path, "H8", ALL0, smoke=True)
    path, v = call("H8", d, smoke=True)
    assert v["mode"] == "smoke" and v["seeds"] == [9001, 9002, 9003]
    d2 = stage(tmp_path / "r", "H8", ALL0, smoke=True)
    with pytest.raises(AR.Stop):                                             # a smoke pass file in real mode
        call("H8", d2)


def test_not_run_wordings(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    ledger(d, {"H6": "not run: switched off", "H7": "not run: switched off", "H9": "dropped under the rule"})
    path, v = call("H7", d, not_run="switched_off")
    assert v["verdict"] is None and v["status"] == "not run: switched off" and v["claim"] is None
    assert path.name == "held_verdict_H7.json" and v["variant"] is None
    path2, v2 = call("H9", d, not_run="dropped")
    assert v2["status"] == "dropped under the rule" and json.loads(path2.read_text())["status"] == v2["status"]
    d2 = stage(tmp_path / "w", "H6", ALL0)
    ledger(d2, {"H6": "dropped under the rule"})
    refused(4, "H6", d2, "was run", not_run="dropped")                       # a read that has a pass file was run
    refused(4, "H6", d, "--not-run takes", not_run="failed")


def test_not_run_needs_the_ledger_row_with_the_same_status(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    refused(4, "H6", d, "ledger.md is missing", not_run="switched_off")
    ledger(d, {"H7": "not run: switched off"})
    refused(4, "H6", d, "row H6 is not in", not_run="switched_off")          # no row yet
    ledger(d, {"H6": "(pending)"})
    refused(4, "H6", d, "does not read exactly", not_run="switched_off")
    ledger(d, {"H6": "dropped under the rule"})
    refused(4, "H6", d, "does not read exactly", not_run="switched_off")     # the other wording
    refused(4, "H8", d, "row H8 is not in", not_run="dropped")
    ledger(d, {"H6": "not run: switched off"})
    refused(4, "H6", d, "does not read exactly", not_run="dropped")
    ledger(d, {"H6": "not run: switched off; dropped under the rule"})
    refused(4, "H6", d, "does not read exactly", not_run="dropped")         # both wordings: ambiguous
    refused(4, "H6", d, "does not read exactly", not_run="switched_off")
    ledger(d, {"H6": "not run: switched off"})
    assert call("H6", d, not_run="switched_off")[1]["status"] == "not run: switched off"


def test_not_run_ledger_row_is_matched_by_its_own_name(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    ledger(d, {"H60": "not run: switched off", "H6-R": "not run: switched off", "H7": "not run: switched off"})
    refused(4, "H6", d, "row H6 is not in", not_run="switched_off")


def test_not_run_ledger_status_must_be_a_whole_cell(tmp_path):
    """Rule 7.1: the wording is a cell of its own; a note that quotes it does not count (build 4 re-review)."""
    d = tmp_path / "results"
    d.mkdir()
    ledger(d, {"H6": "note: not run: switched off was proposed"})
    refused(4, "H6", d, "does not read exactly", not_run="switched_off")
    ledger(d, {"H6": "Not run: switched off"})
    refused(4, "H6", d, "does not read exactly", not_run="switched_off")
    ledger(d, {"H6": " not run: switched off "})          # padding is not part of the cell
    assert call("H6", d, not_run="switched_off")[1]["status"] == "not run: switched off"


def test_not_run_refuses_clip_switched_off(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    ledger(d, {"H8": "not run: switched off"})
    refused(4, "H8", d, "cannot be switched off", not_run="switched_off")
    ledger(d, {"H8": "dropped under the rule"})
    assert call("H8", d, not_run="dropped")[1]["status"] == "dropped under the rule"


@pytest.mark.parametrize("read,kind", [("H6", "switched_off"), ("H9", "dropped"), ("H7", "dropped")])
def test_not_run_refuses_after_the_read_started(tmp_path, read, kind):
    d = tmp_path / "results"
    d.mkdir()
    ledger(d, {read: NOT_RUN_TEXT[kind]})
    (d / f"held_started_{read}.json").write_text("{}")
    refused(4, read, d, f"held_started_{read}.json exists", not_run=kind)


NOT_RUN_TEXT = {"switched_off": "not run: switched off", "dropped": "dropped under the rule"}


def test_not_run_refuses_when_a_fix_pass_exists(tmp_path):
    d = stage(tmp_path, "H6", ALL0, sfx="_fix1")
    ledger(d, {"H6": "dropped under the rule"})
    refused(4, "H6", d, "was run", not_run="dropped")


@pytest.mark.parametrize("read,sibling", [("H9", "H6"), ("H10", "H7")])
def test_switched_off_new_backbone_rows_follow_the_artelingo_row(tmp_path, read, sibling):
    d = tmp_path / "results"
    d.mkdir()
    ledger(d, {read: "not run: switched off", sibling: "not run: switched off"})
    refused(4, read, d, f"only if {sibling} was", not_run="switched_off")    # no sibling verdict yet
    (d / f"held_verdict_{sibling}.json").write_text(json.dumps({"status": "run", "verdict": "GO"}))
    refused(4, read, d, f"only if {sibling} was", not_run="switched_off")    # the sibling ran
    (d / f"held_verdict_{sibling}.json").write_text(json.dumps({"status": "dropped under the rule"}))
    refused(4, read, d, f"only if {sibling} was", not_run="switched_off")    # dropped is not switched off
    (d / f"held_verdict_{sibling}.json").write_text(json.dumps({"status": "not run: switched off"}))
    assert call(read, d, not_run="switched_off")[1]["status"] == "not run: switched off"


def test_dropped_new_backbone_rows_need_no_sibling(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    ledger(d, {"H10": "dropped under the rule"})
    assert call("H10", d, not_run="dropped")[1]["status"] == "dropped under the rule"


def test_not_run_with_a_variant_is_refused(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    ledger(d, {"H6": "not run: switched off"})
    refused(4, "H6", d, "--not-run takes", not_run="switched_off", variant="reserve")
    assert not list(d.glob("held_verdict_*"))


def test_not_run_in_smoke_mode_needs_no_ledger(tmp_path):
    d = tmp_path / "smoke"
    d.mkdir()
    assert call("H6", d, smoke=True, not_run="switched_off")[1]["mode"] == "smoke"


def test_fixed_phrases_are_in_the_rule():
    AR.assert_rule()
    text = " ".join(CM.HERE.joinpath("DECISION_RULE.md").read_text(encoding="utf-8").split())
    r6 = " ".join((CM.TEST / CM.RULE_REL).read_text(encoding="utf-8").split())
    assert all(p in text for p in AR.RULE_PHRASES) and all(p in r6 for p in AR.R6_PHRASES)


def test_main_exit_codes_and_output(tmp_path, capsys, monkeypatch):
    d = stage(tmp_path, "H8", dict(ALL0, P4=500), points={"P4": 0.3})
    monkeypatch.setattr(AR, "RESULTS", d)
    monkeypatch.setattr(AR, "AGREEMENT_DIR", d)
    assert AR.main(["--read", "H8"]) == 0
    out = capsys.readouterr().out
    assert "NO-GO" in out and "written" in out
    with pytest.raises(SystemExit) as e:
        AR.main(["--read", "H8"])
    assert e.value.code == 4
    with pytest.raises(SystemExit):                      # there is no --variant fix1
        AR.main(["--read", "H8", "--variant", "fix1"])


def test_point_exactly_zero_reads_did_not_beat(tmp_path):
    v = run(tmp_path, "H6", dict(ALL0, P1=500), points={"P1": 0.0})
    assert v["checks"]["P1"]["reading_kind"] == "did not beat" and "x" not in v["checks"]["P1"]
    assert v["checks"]["P1"]["reading"] == "AFF did not beat cosine on new paintings" and v["kind"] == "did not beat"


# ---------------------------------------------------------------- the phase-2 agreement (rule 7.4, as R6 rule 8.3)

def test_no_verdict_without_the_agreement(tmp_path):
    d = stage(tmp_path, "H6", ALL0, agree=False)
    refused(4, "H6", d, "rd9_H6_agreement.json is missing")


@pytest.mark.parametrize("over,match", [
    ({"phase": 1}, "not a phase-2 record"), ({"phase": True}, "not a phase-2 record"),
    ({"smoke": True}, "agreement record is refused"), ({"smoke": "no"}, "smoke flag"),
    ({"agreement": False}, "agreement is not true"), ({"agreement": "yes"}, "agreement is not true"),
    ({"agreement": 1}, "agreement is not true"),
    ({"items": REAL_ITEMS[:-1] + [{"key": "arrays.seed_index", "equal": False}]}, "not equal"),
    ({"items": REAL_ITEMS[:-1] + [{"key": "arrays.seed_index", "equal": False}], "agreement": False}, "agreement is not true"),
    ({"items": REAL_ITEMS[:-1] + [{"key": "arrays.seed_index"}]}, "boolean equal"),
    ({"items": REAL_ITEMS[:-1] + [{"key": "arrays.seed_index", "equal": 1}]}, "boolean equal"),
    ({"items": None}, "items is not a list"), ({"items": "x"}, "items is not a list"),
    ({"failed": 1}, "failed is not 0"), ({"failed": False}, "failed is not 0"), ({"failed": None}, "failed is not 0"),
    ({"compared": 0, "items": []}, "compared"), ({"compared": len(REAL_ITEMS) + 1}, "compared"),
    ({"compared": True, "items": REAL_ITEMS[:1]}, "compared"),
    ({"all_agree": True, "agreement": None}, "agreement is not true"),
    ({"read": "H7"}, "its read is not H6"), ({"pass_file": "held_pass_H6_fix1.json"}, "its pass_file is not"),
    ({"held_pass_sha256": "0" * 64}, "checked another pass"), ({"held_pass_sha256": "xyz"}, "not a SHA-256"),
    ({"rederived_counts": {"P1": 0}}, "rederived_counts"), ({"rederived_counts": None}, "rederived_counts"),
])
def test_bad_agreement_refuses(tmp_path, over, match):
    d = stage(tmp_path, "H6", ALL0)
    write_agreement(d, "H6", over=over)
    refused(4, "H6", d, match)


def test_agreement_missing_keys_refuse(tmp_path):
    for key in ("phase", "smoke", "agreement", "compared", "failed", "items", "read", "pass_file",
                "held_pass_sha256", "rederived_counts"):
        d = stage(tmp_path / key, "H6", ALL0)
        f = d / "rd9_H6_agreement.json"
        r = json.loads(f.read_text())
        r.pop(key)
        f.write_text(json.dumps(r))
        refused(4, "H6", d)


def test_agreement_is_bound_to_the_pass_file_bytes(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    p = d / "held_pass_H6.json"
    p.write_text(p.read_text() + "\n")                    # the same record, other bytes
    refused(4, "H6", d, "checked another pass")


def test_smoke_agreement_is_refused_in_real_mode_and_real_in_smoke(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    write_agreement(d, "H6", smoke=True)
    refused(4, "H6", d, "smoke agreement record is refused in real mode")
    d = stage(tmp_path / "s", "H8", ALL0, smoke=True)
    write_agreement(d, "H8", smoke=False)
    refused(4, "H8", d, "real agreement record is refused in smoke mode", smoke=True)


def test_the_old_names_are_not_accepted_and_the_unread_keys_are_not_needed(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    f = d / "rd9_H6_agreement.json"
    r = json.loads(f.read_text())
    for key in ("independent_identity", "backbone"):          # written by ticket 25, not read here
        r.pop(key)
    f.write_text(json.dumps(r))
    assert call("H6", d)[1]["status"] == "run"
    d = tmp_path / "old"
    d = stage(d, "H6", ALL0)
    r = json.loads((d / "rd9_H6_agreement.json").read_text())
    for k in ("agreement", "compared", "failed", "items", "independent_identity", "backbone"):
        r.pop(k)
    r.update(all_agree=True, disagreements=[], n_quantities=129)
    (d / "rd9_H6_agreement.json").write_text(json.dumps(r))
    refused(4, "H6", d, "agreement is not true")


def test_rederived_counts_must_equal_the_pass_counts(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    rc = {nm: 0 for nm in P}
    rc["P5"] = 1
    write_agreement(d, "H6", over={"rederived_counts": rc})
    e = refused(5, "H6", d)
    assert "P5" in e.msg


def test_boundary_report_carries_the_rederivation_count(tmp_path):
    counts = dict(ALL0, P3=25, P4=39, P5=40, P6=60, P7=100)
    d = stage(tmp_path, "H6", counts)
    v = call("H6", d)[1]
    assert v["boundary_report"] and all(b["rederivation_n"] == counts[b["check"]] for b in v["boundary_report"])
    assert v["agreement_file"] == "rd9_H6_agreement.json"
    assert v["agreement"] == {"phase": 2, "smoke": False, "agreement": True, "read": "H6", "pass_file": "held_pass_H6.json",
                              "held_pass_sha256": CM.sha256_file(d / "held_pass_H6.json"), "compared": len(REAL_ITEMS)}
    assert v["agreement_sha256"] == CM.sha256_file(d / v["agreement_file"])


# ---------------------------------------------------------------- the smoke guard (rule 4.1 "6 passed", 6 item 5)

def test_no_verdict_without_a_passed_smoke_of_these_files(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    (d / "smoke_record.json").unlink()
    refused(4, "H6", d, "smoke_record.json is missing")
    write_smoke_record(d, passed=False)
    refused(4, "H6", d, "did not pass")
    write_smoke_record(d, passed="yes")
    refused(4, "H6", d, "did not pass")
    mods = CM.r9_module_shas()
    key = "src/test/20261202_r9_scope/run_r9_apply_rule.py"
    assert key in mods
    write_smoke_record(d, module_sha256=dict(mods, **{key: "0" * 64}))              # a changed file
    refused(4, "H6", d, "a new smoke is needed")
    write_smoke_record(d, module_sha256={k: v for k, v in mods.items() if k != key})   # a file the smoke lacks
    refused(4, "H6", d, "a new smoke is needed")
    write_smoke_record(d, module_sha256=dict(mods, extra_file_py="0" * 64))            # a file the code lacks
    refused(4, "H6", d, "a new smoke is needed")
    write_smoke_record(d, module_sha256={})
    refused(4, "H6", d, "a new smoke is needed")
    write_smoke_record(d, module_sha256=None)
    refused(4, "H6", d, "a new smoke is needed")
    write_smoke_record(d)
    v = call("H6", d)[1]
    assert v["smoke_record"] == {"name": "smoke_record.json", "sha256": CM.sha256_file(d / "smoke_record.json")}


def test_the_latest_smoke_record_counts(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    write_smoke_record(d, "smoke_record_crash1.json", passed=False)                 # crash1 outranks the first smoke
    refused(4, "H6", d, "smoke_record_crash1.json")
    write_smoke_record(d, "smoke_record_fix1.json", module_sha256={"x": "0" * 64})  # fix1 outranks crash1
    refused(4, "H6", d, "smoke_record_fix1.json")
    write_smoke_record(d, "smoke_record_fix1.json")
    assert call("H6", d)[1]["smoke_record"]["name"] == "smoke_record_fix1.json"


def test_smoke_mode_needs_no_smoke_record(tmp_path):
    d = stage(tmp_path, "H8", ALL0, smoke=True)
    assert not (d / "smoke_record.json").exists()
    assert call("H8", d, smoke=True)[1]["smoke_record"] is None


# ---------------------------------------------------------------- the set-aside list (rule 5.1 item 1, 6)

def sa(language, pair="emotion__style", seed=55):
    return {"language": language, "pair": pair, "seed": seed}


def test_set_aside_cap_is_two_per_pair_and_seed(tmp_path):
    d = stage(tmp_path / "ok", "H8", ALL0, set_aside=[sa("Hindi"), sa("Urdu"), sa("Hindi", seed=56), sa("Thai", seed=56),
                                                       sa("Igbo", "style__genre", 55)])
    v = call("H8", d)[1]
    assert v["n_languages"] == 23 and v["verdict"] == "full claim"
    d = stage(tmp_path / "three", "H8", ALL0, set_aside=[sa("Hindi"), sa("Urdu"), sa("Thai")])
    e = refused(5, "H8", d)
    assert "more than 2" in e.msg and "emotion__style" in e.msg
    d = stage(tmp_path / "late", "H9", ALL0, set_aside=[sa("Hindi", seed=57), sa("Urdu", seed=57), sa("Thai", seed=57)])
    refused(5, "H9", d, "more than 2")


def test_set_aside_duplicates_and_unknown_names_stop(tmp_path):
    d = stage(tmp_path / "dup", "H8", ALL0, set_aside=[sa("Hindi"), sa("Hindi")])
    refused(5, "H8", d, "duplicate")
    d = stage(tmp_path / "unk", "H8", ALL0, set_aside=[sa("Klingon")])
    refused(5, "H8", d, "Klingon")
    d = stage(tmp_path / "case", "H8", ALL0, set_aside=[sa("hindi")])             # 'arabic' and 'chinese' are lower-case
    refused(5, "H8", d, "hindi")
    d = stage(tmp_path / "lc", "H8", ALL0, set_aside=[sa("arabic"), sa("chinese")])
    assert call("H8", d)[1]["n_languages"] == 25


def test_the_language_list_is_the_rules_27(tmp_path):
    assert len(AR.LANGUAGES) == 27 and "arabic, chinese" in AR.LANGUAGE_LIST
    text = " ".join(CM.HERE.joinpath("DECISION_RULE.md").read_text(encoding="utf-8").split())
    assert AR.LANGUAGE_LIST in text


# ---------------------------------------------------------------- claims: backbone name and near-duplicates

@pytest.mark.parametrize("read,counts,want", [
    ("H8", ALL0, "full claim"), ("H9", ALL0, "full claim"), ("H10", ALL0, "full claim"),
    ("H8", dict(ALL0, P1=25, P2=25, P3=25, P5=25, P7=25), "small claim"),
    ("H9", dict(ALL0, P1=25, P2=25, P3=25, P5=25, P7=25), "small claim"),
    ("H10", dict(ALL0, P1=25, P2=25, P3=25, P5=25, P7=25), "small claim")])
def test_a28_claims_start_with_the_ledger_backbone_and_carry_the_near_duplicates(tmp_path, read, counts, want):
    d = stage(tmp_path, read, counts, set_aside=[sa("Hindi")], flagged=["Thai"])
    r = json.loads((d / f"held_pass_{read}.json").read_text())
    edit_pass(d, read, lambda r: r.update(near_duplicates={"cos_099": 11, "cos_098": 29}))
    v = call(read, d)[1]
    assert v["verdict"] == want
    assert v["claim"].startswith(f"AFF on {AR.READS[read]['backbone']}")
    assert "Near-duplicates (" in v["claim"] and "11 / 29 of 746" in v["claim"] and "Flagged languages: Thai" in v["claim"]
    assert "Hindi (emotion__style, seed 55)" in v["claim"] and "26 languages that entered the read" in v["claim"]
    assert v["claim"].endswith("Not native-language text, not any single language.")
    assert v["near_duplicates"] == {"cos_099": 11, "cos_098": 29}
    assert ("keeps the same sentence on that backbone." in v["claim"]) is (want == "full claim")


def test_the_ledger_names_are_the_backbone_names():
    names = {AR.READS[r]["backbone"] for r in AR.READS}
    assert names == {"CLIP ViT-B/32", "Qwen3-VL-Embedding-2B", "SigLIP 2 so400m/14-384"}


@pytest.mark.parametrize("nd", [None, {}, {"cos_099": 1}, {"cos_099": 5, "cos_098": 4}, {"cos_099": -1, "cos_098": 2},
                                {"cos_099": 1, "cos_098": 747}, {"cos_099": 1.0, "cos_098": 2},
                                {"cos_099": True, "cos_098": 2}, {"cos_099": 1, "cos_098": 2, "x": 3}, [1, 2]])
def test_bad_near_duplicates_refuse(tmp_path, nd):
    d = stage(tmp_path, "H8", ALL0)
    edit_pass(d, "H8", lambda r: r.update(near_duplicates=nd))
    refused(4, "H8", d, "near_duplicates")


def test_missing_near_duplicates_refuses(tmp_path):
    d = stage(tmp_path, "H8", ALL0)
    edit_pass(d, "H8", lambda r: r.pop("near_duplicates"))
    refused(4, "H8", d, "near_duplicates")


# ---------------------------------------------------------------- mutants that survived the first round

def test_x95_in_the_sensitivity_file_is_checked(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    s = json.loads((d / "sensitivity_H6.json").read_text())
    s["P3"]["x95"] = 2.0 * 0.1                              # not 2.80 * SE
    (d / "sensitivity_H6.json").write_text(json.dumps(s))
    e = refused(5, "H6", d)
    assert "2.80" in e.msg and "P3" in e.msg


def test_sensitivity_x_of_the_seven_kind_and_non_finite_values(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    s = json.loads((d / "sensitivity_H6.json").read_text())
    s["P2"]["x"] = 3.083 * 0.1
    (d / "sensitivity_H6.json").write_text(json.dumps(s))
    refused(5, "H6", d, "P2 x")
    s["P2"]["x"] = -1.0
    (d / "sensitivity_H6.json").write_text(json.dumps(s))
    refused(4, "H6", d, "P2 lacks")


def test_holm_order_of_seven_must_match_the_counts(tmp_path):
    d = stage(tmp_path, "H6", dict(ALL0, P3=5))             # recomputed: P1 P2 P4 P5 P6 P7 P3
    edit_pass(d, "H6", lambda r: r.update(holm_order=list(P)))     # a valid ordering of P1..P7, not by count
    e = refused(5, "H6", d)
    assert "holm_order" in e.msg


def test_holm_order_of_two_levels_must_match_the_counts(tmp_path):
    d = stage(tmp_path / "a", "H8", dict(ALL0, P4=5))       # level 1 recomputed: P6 P4
    edit_pass(d, "H8", lambda r: r["holm_order"].update(level1=["P4", "P6"]))
    assert "holm_order" in refused(5, "H8", d).msg
    d = stage(tmp_path / "b", "H8", dict(ALL0, P1=5))       # level 2 recomputed: P2 P3 P5 P7 P1
    edit_pass(d, "H8", lambda r: r["holm_order"].update(level2=["P1", "P2", "P3", "P5", "P7"]))
    assert "holm_order" in refused(5, "H8", d).msg


def test_near_boundary_flag_must_match_the_counts(tmp_path):
    d = stage(tmp_path, "H6", dict(ALL0, P3=41, P5=50, P7=51))      # P3 is rank 5 (n* 40)
    r = json.loads((d / "held_pass_H6.json").read_text())
    nm = next(n for n in P if r["checks"][n]["near_boundary"])
    edit_pass(d, "H6", lambda r: r["checks"][nm].update(near_boundary=not r["checks"][nm]["near_boundary"]))
    e = refused(5, "H6", d)
    assert "near_boundary" in e.msg
    d = stage(tmp_path / "two", "H9", ALL0)
    edit_pass(d, "H9", lambda r: r["checks"]["P1"].update(near_boundary=not r["checks"]["P1"]["near_boundary"]))
    assert "near_boundary" in refused(5, "H9", d).msg


@pytest.mark.parametrize("field,val", [("level_two_sided", 0.99), ("own_count_passes", False), ("passes", False)])
def test_every_holm_field_is_checked_on_both_designs(tmp_path, field, val):
    for read in ("H6", "H8"):
        d = stage(tmp_path / read, read, ALL0)
        edit_pass(d, read, lambda r: r["checks"]["P2"].update({field: val}))
        assert field in refused(5, read, d).msg


def test_untested_level2_checks_do_not_leak_into_the_boundary_list(tmp_path):
    # level 1 fails (P4 at 62 = n* + 1, P6 at 124 = n*); level 2 has counts at their n* (61 for k=4, 124 for k=5)
    counts = dict(ALL0, P4=62, P6=124, P2=61, P1=124)
    v = run(tmp_path, "H8", counts)
    assert v["level2"]["tested"] is False
    assert {b["check"] for b in v["boundary_report"]} == {"P4", "P6"}
    # the same level-2 counts are listed when level 1 passes
    v = run(tmp_path / "pass", "H8", dict(counts, P4=61, P6=124))
    assert {"P1", "P2"} <= {b["check"] for b in v["boundary_report"]}


def test_boundary_report_of_both_designs_lists_the_rank_and_pass(tmp_path):
    v = run(tmp_path, "H6", dict(ALL0, P3=41, P5=50, P7=51))
    b = {x["check"]: x for x in v["boundary_report"]}["P3"]
    assert (b["holm_k"], b["n"], b["n_star"], b["passes"]) == (5, 41, 40, False)


# ---------------------------------------------------------------- the overwrite guards, each on its own

def test_verdict_exists_guard_fires_before_any_other_check(tmp_path):
    d = tmp_path / "results"
    d.mkdir()
    (d / "held_verdict_H6.json").write_text("{}")                  # nothing else exists: no pass, no agreement, no smoke
    e = refused(4, "H6", d)
    assert "never overwritten" in e.msg and (d / "held_verdict_H6.json").read_text() == "{}"
    (d / "held_verdict_H6_reserve.json").write_text("{}")
    assert "never overwritten" in refused(4, "H6", d, variant="reserve").msg
    ledger(d, {"H7": "not run: switched off"})
    (d / "held_verdict_H7.json").write_text("{}")
    assert "never overwritten" in refused(4, "H7", d, not_run="switched_off").msg


def test_write_once_refuses_an_existing_file_and_leaves_no_temp_file(tmp_path):
    f = tmp_path / "held_verdict_H6.json"
    f.write_text("first")
    with pytest.raises(AR.Stop) as e:
        AR.write_once(f, {"a": 1})
    assert e.value.code == 4 and "never overwritten" in e.value.msg and f.read_text() == "first"
    assert [p.name for p in tmp_path.iterdir()] == ["held_verdict_H6.json"]
    g = tmp_path / "new.json"
    AR.write_once(g, {"a": 1})
    assert json.loads(g.read_text()) == {"a": 1} and sorted(p.name for p in tmp_path.iterdir()) == [
        "held_verdict_H6.json", "new.json"]


def test_write_once_wins_a_race_against_the_early_guard(tmp_path, monkeypatch):
    # the verdict appears after the early exists() check (another process): the atomic link still refuses
    d = stage(tmp_path, "H6", ALL0)
    real = AR.write_once

    def late(path, obj):
        path.write_text("other process")
        return real(path, obj)
    monkeypatch.setattr(AR, "write_once", late)
    refused_msg = pytest.raises(AR.Stop, call, "H6", d)
    assert refused_msg.value.code == 4 and (d / "held_verdict_H6.json").read_text() == "other process"


def test_reserve_does_not_fall_back_to_the_original_sensitivity(tmp_path):
    d = stage(tmp_path, "H6", ALL0)
    stage(tmp_path, "H6", ALL0, sfx="_reserve", sens=False)
    call("H6", d)
    write_smoke_record(d, "smoke_record_reserve.json")
    refused(4, "H6", d, "sensitivity_H6_reserve.json is missing", variant="reserve")


def test_smoke_agreement_is_read_from_the_smoke_folder_by_default(tmp_path):
    d = stage(tmp_path, "H8", ALL0, smoke=True)
    path, v = AR.apply("H8", smoke=True, results=d)           # no agreement_dir, no ledger
    assert v["agreement_file"] == "rd9_H8_agreement.json" and path.parent == d


# ---------------------------------------------------------------- final-review fixes (A1 L1, L2, L3, assert_rule)

def test_main_prints_the_rederivation_count_beside_the_runners(tmp_path, capsys, monkeypatch):
    # L1: rule 7.6 / R6 8.5: the user sees both counts (the step requires them equal, so one number shows twice).
    counts = dict(ALL0, P3=25, P4=39, P5=40, P6=60, P7=100)
    d = stage(tmp_path, "H6", counts)
    monkeypatch.setattr(AR, "RESULTS", d)
    monkeypatch.setattr(AR, "AGREEMENT_DIR", d)
    assert AR.main(["--read", "H6"]) == 0
    lines = [x for x in capsys.readouterr().out.splitlines() if "boundary (rule 7.6" in x]
    assert lines
    for line in lines:
        check = line.split("decides): ")[1].split()[0]
        assert f"{check} count {counts[check]}, re-derivation's count {counts[check]}, largest passing count" in line


@pytest.mark.parametrize("read", ["H6", "H8"])
@pytest.mark.parametrize("check", ["P1", "P4", "P7"])
def test_pass_quantity_label_is_checked(tmp_path, read, check):      # L2
    d = stage(tmp_path, read, ALL0)
    edit_pass(d, read, lambda r: r["checks"][check].update(quantity="R@1, AFF - B0" if check != "P4" else "R@1, AFF - B"))
    refused(4, read, d, "pass schema")


@pytest.mark.parametrize("read", ["H6", "H9"])
@pytest.mark.parametrize("field,value", [("read", "H7"), ("backbone", "clip_b32"), ("read", None), ("backbone", None)])
def test_pass_file_must_name_its_read_and_backbone(tmp_path, read, field, value):      # L3
    d = stage(tmp_path, read, ALL0)
    edit_pass(d, read, lambda r: r.pop(field) if value is None else r.update({field: value}))
    refused(4, read, d, "read and backbone")


def test_pass_file_of_another_read_with_the_same_backbone_is_refused(tmp_path):
    d = stage(tmp_path, "H9", ALL0)       # an H6 pass (also Qwen3-VL) under H9's name
    edit_pass(d, "H9", lambda r: r.update(read="H6"))
    refused(4, "H9", d, "read and backbone")


def alter_rule(tmp_path, monkeypatch, fault):
    if fault == "rule_sha":
        edited = tmp_path / "DECISION_RULE.md"
        edited.write_bytes(AR.RULE_PATH.read_bytes() + b"\nan edit\n")
        monkeypatch.setattr(AR, "RULE_PATH", edited)
        return 4, "not the committed rule's"
    if fault == "r6_sha":
        monkeypatch.setattr(CM, "R6_RULE_SHA256", "0" * 64)
        return 4, "of round 6"
    if fault == "phrase":
        monkeypatch.setattr(AR, "RULE_PHRASES", AR.RULE_PHRASES + ("a sentence the rule does not contain",))
        return 5, "fixed phrase"
    monkeypatch.setattr(AR, "R6_PHRASES", AR.R6_PHRASES + ("a sentence round 6 does not contain",))
    return 5, "fixed phrase"


@pytest.mark.parametrize("fault", ["rule_sha", "r6_sha", "phrase", "r6_phrase"])
def test_assert_rule_stops_a_read(tmp_path, monkeypatch, fault):
    d = stage(tmp_path, "H6", ALL0)
    code, match = alter_rule(tmp_path, monkeypatch, fault)
    refused(code, "H6", d, match)


@pytest.mark.parametrize("fault", ["rule_sha", "r6_sha", "phrase", "r6_phrase"])
def test_assert_rule_stops_a_not_run_verdict(tmp_path, monkeypatch, fault):
    d = tmp_path / "smoke"
    d.mkdir()
    code, match = alter_rule(tmp_path, monkeypatch, fault)
    with pytest.raises(AR.Stop) as e:
        call("H6", d, smoke=True, not_run="switched_off")
    assert e.value.code == code and match in e.value.msg
    assert not list(d.glob("held_verdict_*.json"))
