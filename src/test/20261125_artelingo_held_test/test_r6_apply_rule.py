"""Tests of run_r6_apply_rule.py (ticket 09): rule §4, §8.3, §8.5, §6.7, §9 on hand-built files of contracts §6, §8.

Every file is built here in tmp_path with the real names, the real seeds and the real counts' range (5,000 draws,
36,864 pooled episodes); nothing is read from or written to results/. The Holm flags of the hand-built pass files
come from an independent exact form (fractions: p = (n + 1) / 5001 against 0.025 / (m + 1 - k)), and the outcomes
each test expects are written out by hand.
"""
import hashlib
import importlib.util
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "run_r6_apply_rule.py"
RULE = HERE / "DECISION_RULE.md"
RULE_SHA = "7444a5e338838d837673b82b047c82b033eb1e2d1e0c3ed4e388dd8673070724"
P = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
S = ("S1", "S2")
NSTAR_P = (16, 19, 24, 30, 40, 61, 124)     # rule §8.5
NSTAR_S = (61, 124)
QUANT = {"P1": "r1 AFF-COS", "P2": "r1 AFF-RCA", "P3": "r1 AFF-B", "P4": "r1 AFF-B0", "P5": "r1 AFF-CF",
         "P6": "gain AFF-CF", "P7": "gain AFF-RCA", "S1": "r1 AFF-B1", "S2": "r1 AFF-R1 fused"}
CONTRACT_KEYS = {"verdict", "kind", "checks", "secondary", "boundary_report", "claim", "held_pass_sha256",
                 "agreement_sha256", "rule_sha256", "module_sha256", "time"}
DECIMAL = re.compile(r"\d*\.\d+")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common  # noqa: E402,F401  first, before anything that imports src (contracts section 1 amendment)
import r6_stats as ST  # noqa: E402

AR = load_module(SCRIPT, "run_r6_apply_rule")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- hand-built files (contracts §6, §8)

def ref_holm(counts: dict, names: tuple, m: int) -> dict:
    """Independent exact Holm: p = (n + 1) / 5001 <= 0.025 / (m + 1 - k), ties in the given order."""
    order = []
    for nm in names:                                   # insertion sort keeps ties in the given order
        i = len(order)
        while i > 0 and counts[order[i - 1]] > counts[nm]:
            i -= 1
        order.insert(i, nm)
    out, ok = {}, True
    for k, nm in enumerate(order, start=1):
        own = Fraction(counts[nm] + 1, 5001) <= Fraction(25, 1000) / (m + 1 - k)
        ok = ok and own
        nstar = (NSTAR_P if m == 7 else NSTAR_S)[k - 1]
        out[nm] = {"k": k, "passes": ok, "own_count_passes": own, "near_boundary": abs(counts[nm] - nstar) <= 1,
                   "level_two_sided": 1 - 0.05 / (m + 1 - k)}
    return {"order": order, **out}


def make_pass(counts, points=None, s_counts=(0, 0), s_points=(0.33, 0.21), smoke=False) -> dict:
    counts = dict(zip(P, counts)) if not isinstance(counts, dict) else counts
    points = points or {}
    hp = ref_holm(counts, P, 7)
    checks = {}
    for nm in P:
        pt = points.get(nm, 0.5)
        h = hp[nm]
        checks[nm] = {"quantity": QUANT[nm], "n": counts[nm], "point": pt, "ci95": [pt - 0.25, pt + 0.25],
                      "holm_k": h["k"], "ci_holm": [pt - 0.375, pt + 0.375], "level_two_sided": h["level_two_sided"],
                      "passes": h["passes"], "own_count_passes": h["own_count_passes"],
                      "near_boundary": h["near_boundary"]}
    sc = dict(zip(S, s_counts))
    hs = ref_holm(sc, S, 2)
    secondary = {nm: {"quantity": QUANT[nm], "n": sc[nm], "point": pt, "ci95": [pt - 0.25, pt + 0.25],
                      "holm_k": hs[nm]["k"], "ci_holm": [pt - 0.3, pt + 0.3],
                      "level_two_sided": hs[nm]["level_two_sided"], "near_boundary": hs[nm]["near_boundary"]}
                 for nm, pt in zip(S, s_points)}
    seeds = [9001, 9002, 9003] if smoke else [52, 53, 54]
    per_pair = 64 if smoke else 4096
    return {"rule_sha256": RULE_SHA, "mode": "smoke" if smoke else "held", "seeds": seeds,
            "n_episodes": 9 * per_pair, "n_clusters": 9876, "checks": checks, "holm_order": hp["order"],
            "secondary": secondary,
            "episodes_sha256": {str(s): {p: f"{s:064x}"[:60] + "abcd" for p in
                                         ("emotion__style", "emotion__genre", "style__genre")} for s in seeds},
            "runner_sha256": "b" * 64, "module_sha256": {"src/test/x/r6_stats.py": "c" * 64},
            "time": "2026-10-12 03:14:15"}


def make_sens() -> dict:
    rec = {nm: {"SE": 0.1, "x": 0.3532, "x95": 0.28} for nm in P}
    rec.update({nm: {"SE": 0.1, "x2": 0.3083, "x95": 0.28} for nm in S})
    rec.update({"module_sha256": {}, "time": "2026-10-12 03:00:00"})
    return rec


AGREEMENT_OF = {"held_pass.json": "rederive_agreement.json", "held_pass_fix1.json": "rederive_agreement_fix1.json",
                "held_pass_reserve.json": "rederive_agreement_reserve.json"}     # contracts §8 as amended


class Env:
    """A tmp results tree wired into a module (the real one or a mutated copy)."""

    def __init__(self, tmp_path, monkeypatch, mod=AR):
        self.mod = mod
        self.results = tmp_path / "results"
        self.smoke = self.results / "smoke"
        self.smoke.mkdir(parents=True)
        monkeypatch.setattr(mod, "RESULTS", self.results)
        monkeypatch.setattr(mod, "SMOKE", self.smoke)
        monkeypatch.setattr(mod, "RULE_PATH", RULE)

    def out(self, smoke=False, subdir=None) -> Path:
        d = (self.smoke if smoke else self.results) / (subdir or "")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write(self, name, obj, smoke=False, subdir=None) -> bytes:
        raw = (json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        (self.out(smoke, subdir) / name).write_bytes(raw)
        return raw

    def stage(self, rec, smoke=False, pass_file="held_pass.json", agreement=None, sens=True, agreement_flag=None,
              agreement_name=None, subdir=None) -> dict:
        """A pass file, its agreement record (the file of its pair unless named) and the sensitivity file."""
        raw = self.write(pass_file, rec, smoke, subdir)
        agr = {"phase": 2, "smoke": smoke if agreement_flag is None else agreement_flag, "all_agree": True,
               "held_pass_sha256": sha(raw), "pass_file": pass_file, "n_quantities": 61, "disagreements": [],
               "time": "2026-10-12 05:00:00"}
        agr.update(agreement or {})
        self.write(agreement_name or AGREEMENT_OF.get(pass_file, "rederive_agreement.json"), agr, smoke, subdir)
        if sens:
            name = "sensitivity_held_reserve.json" if pass_file == "held_pass_reserve.json" else "sensitivity_held.json"
            self.write(name, make_sens(), smoke, subdir)
        return agr

    def run(self, *argv):
        try:
            return self.mod.main(list(argv))
        except SystemExit as e:
            return e.code

    def verdict(self, smoke=False, name="held_verdict.json", subdir=None) -> dict:
        return json.loads((self.out(smoke, subdir) / name).read_text(encoding="utf-8"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    return Env(tmp_path, monkeypatch)


def go_rec(**kw):
    return make_pass([0] * 7, **kw)


# ---------------------------------------------------------------- the Holm recomputation and the fixed texts

@pytest.mark.parametrize("m,nstars,names", [(7, NSTAR_P, P), (2, NSTAR_S, S)])
def test_holm_ranks_match_the_exact_form_at_every_boundary(m, nstars, names):
    assert [ST.boundary(k, m) for k in range(1, m + 1)] == list(nstars)
    for k in range(1, m + 1):
        for n, want in ((nstars[k - 1], True), (nstars[k - 1] + 1, False)):
            counts = {nm: (0 if i < k - 1 else n if i == k - 1 else 5000) for i, nm in enumerate(names)}
            got = {e["name"]: e for e in AR.holm_ranks(counts, names, m)}
            ref = ref_holm(counts, names, m)
            assert got[names[k - 1]]["k"] == k and got[names[k - 1]]["own_count_passes"] is want
            assert got[names[k - 1]]["n_star"] == nstars[k - 1] and got[names[k - 1]]["near_boundary"] is True
            for nm in names:
                assert (got[nm]["passes"], got[nm]["own_count_passes"], got[nm]["k"], got[nm]["near_boundary"]) == \
                       (ref[nm]["passes"], ref[nm]["own_count_passes"], ref[nm]["k"], ref[nm]["near_boundary"])


def test_ties_keep_the_rule_order():
    assert [e["name"] for e in AR.holm_ranks({nm: 7 for nm in P}, P, 7)] == list(P)
    assert [e["name"] for e in AR.holm_ranks({"S1": 70, "S2": 70}, S, 2)] == ["S1", "S2"]


def test_the_step_uses_r6_common_and_r6_stats():
    assert AR.CM is r6_common and AR.ST is ST
    assert AR.RESULTS == r6_common.RESULTS and AR.SMOKE == r6_common.SMOKE


def test_the_claim_is_rule_section_4_verbatim():
    text = RULE.read_text(encoding="utf-8")
    head = "- **Claim licensed by a GO** (spec §6): "
    start = text.index(head) + len(head)
    claim = " ".join(text[start:text.index("\n\n", start)].split())
    assert claim == AR.CLAIM_GO
    assert sha(RULE.read_bytes()) == RULE_SHA == AR.CM.RULE_SHA256


# ---------------------------------------------------------------- GO

def test_go_with_both_secondary_checks_passing(env, capsys):
    env.stage(go_rec())
    assert env.run() == 0
    v = env.verdict()
    assert CONTRACT_KEYS <= set(v)
    assert (v["verdict"], v["kind"]) == ("GO", None)
    assert all(v["checks"][nm]["passes"] and v["checks"][nm]["reading"] == "passed"
               and v["checks"][nm]["not_reached_after"] is None for nm in P)
    assert v["secondary"]["tested"] is True
    assert [v["secondary"][nm]["passes"] for nm in S] == [True, True]
    assert v["claim"] == AR.CLAIM_GO + " AFF also beats B′(A1). AFF also beats R1."
    assert v["boundary_report"] == []
    raw = (env.results / "held_pass.json").read_bytes()
    assert v["held_pass_sha256"] == sha(raw)
    assert v["agreement_sha256"] == sha((env.results / "rederive_agreement.json").read_bytes())
    assert v["rule_sha256"] == RULE_SHA and any(k.endswith("run_r6_apply_rule.py") for k in v["module_sha256"])
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", v["time"])
    assert v["agreement"]["all_agree"] is True and v["pass_file"] == "held_pass.json"
    assert "verdict: GO" in capsys.readouterr().out


def test_go_with_one_secondary_pass_licenses_only_that_one(env):
    env.stage(go_rec(s_counts=(200, 10)))              # S2 first (10 <= 61), S1 second (200 > 124)
    assert env.run() == 0
    v = env.verdict()
    s1, s2 = v["secondary"]["S1"], v["secondary"]["S2"]
    assert v["secondary"]["holm_order"] == ["S2", "S1"]
    assert (s2["passes"], s2["license"]) == (True, "AFF also beats R1")
    assert (s1["passes"], s1["license"], s1["not_reached_after"]) == (False, None, None)
    assert s1["reading"].startswith("inconclusive at a detectable margin of x₂ = 0.308 points")
    assert s1["x2"] == 0.3083 and s1["x95"] == 0.28 and s1["half_width_95"] == pytest.approx(0.25)
    assert v["claim"] == AR.CLAIM_GO + " AFF also beats R1."


def test_go_secondary_tie_s1_first_and_s2_not_reached(env):
    env.stage(go_rec(s_counts=(70, 70)))               # tie: S1 at k = 1 (70 > 61) fails; S2 at k = 2 would pass
    assert env.run() == 0
    s = env.verdict()["secondary"]
    assert s["holm_order"] == ["S1", "S2"]
    assert (s["S1"]["passes"], s["S1"]["not_reached_after"]) == (False, None)
    assert (s["S2"]["passes"], s["S2"]["own_count_passes"], s["S2"]["not_reached_after"]) == (False, True, "S1")
    assert s["S2"]["reading"].startswith("not reached: the Holm procedure stopped at S1 (R@1 against B′(A1)); "
                                         "its own reading: inconclusive at a detectable margin of x₂ = ")
    assert env.verdict()["claim"] == AR.CLAIM_GO


def test_go_with_no_secondary_pass(env):
    env.stage(go_rec(s_counts=(3000, 3100), s_points=(-0.1, 0.0)))
    assert env.run() == 0
    v = env.verdict()
    assert v["verdict"] == "GO" and v["claim"] == AR.CLAIM_GO
    assert v["secondary"]["S1"]["reading"] == "AFF did not beat B′(A1) on new paintings"
    assert v["secondary"]["S2"]["reading"] == "AFF did not beat R1 on new paintings"
    assert [v["secondary"][nm]["license"] for nm in S] == [None, None]


# ---------------------------------------------------------------- NO-GO

def test_first_rank_failure_leaves_the_rest_not_reached(env, capsys):
    env.stage(make_pass([17] * 7))                     # tie: P1 at k = 1 (17 > 16); P2..P7 own counts would pass
    assert env.run() == 0
    v = env.verdict()
    assert (v["verdict"], v["kind"], v["claim"]) == ("NO-GO", "inconclusive", None)
    c = v["checks"]
    assert (c["P1"]["passes"], c["P1"]["own_count_passes"], c["P1"]["not_reached_after"]) == (False, False, None)
    assert c["P1"]["reading"].startswith("inconclusive at a detectable margin of x = 0.353 points (x95 = 0.280; "
                                         "realised half-width 0.250 at 95%, 0.375 at its Holm level)")
    for nm in P[1:]:
        assert (c[nm]["passes"], c[nm]["own_count_passes"], c[nm]["not_reached_after"]) == (False, True, "P1")
        assert c[nm]["reading"].startswith("not reached: the Holm procedure stopped at P1 (R@1 against cosine); "
                                           "its own reading: inconclusive at a detectable margin of x = ")
    assert v["secondary"]["tested"] is False
    out = capsys.readouterr().out
    assert "verdict: NO-GO (inconclusive)" in out and "P2: failed, inconclusive, not reached after P1" in out


def test_middle_failure(env):
    counts = {"P1": 0, "P2": 0, "P3": 5, "P4": 31, "P5": 35, "P6": 50, "P7": 200}
    env.stage(make_pass(counts))
    assert env.run() == 0
    c = env.verdict()["checks"]
    assert [c[nm]["passes"] for nm in P] == [True, True, True, False, False, False, False]
    assert [c[nm]["reading"] for nm in ("P1", "P2", "P3")] == ["passed"] * 3
    assert (c["P4"]["holm_k"], c["P4"]["not_reached_after"]) == (4, None)        # 31 > 30
    assert c["P5"]["not_reached_after"] == "P4" and c["P6"]["not_reached_after"] == "P4"
    assert c["P5"]["reading"].startswith("not reached: the Holm procedure stopped at P4 (R@1 against B′(A0))")
    assert (c["P7"]["own_count_passes"], c["P7"]["not_reached_after"]) == (False, None)   # 200 > 124, own failure
    assert env.verdict()["kind"] == "inconclusive"


def test_no_go_all_inconclusive(env):
    counts = {"P1": 0, "P2": 130, "P3": 0, "P4": 0, "P5": 0, "P6": 0, "P7": 140}
    env.stage(make_pass(counts, points={"P2": 0.05, "P7": 1e-9}))
    assert env.run() == 0
    v = env.verdict()
    assert (v["verdict"], v["kind"], v["failed"]) == ("NO-GO", "inconclusive", ["P2", "P7"])
    assert all(v["checks"][nm]["reading_kind"] == "inconclusive" for nm in ("P2", "P7"))
    assert all(v["checks"][nm]["reading"] == "passed" for nm in ("P1", "P3", "P4", "P5", "P6"))
    assert v["claim"] is None


@pytest.mark.parametrize("point", [-0.05, 0.0])
def test_no_go_with_one_did_not_beat(env, point):
    counts = {"P1": 0, "P2": 0, "P3": 0, "P4": 2600, "P5": 0, "P6": 0, "P7": 140}
    env.stage(make_pass(counts, points={"P4": point}))
    assert env.run() == 0
    v = env.verdict()
    assert (v["verdict"], v["kind"]) == ("NO-GO", "did not beat")
    assert v["checks"]["P4"]["reading"] == "AFF did not beat B′(A0) on new paintings"
    assert v["checks"]["P7"]["reading_kind"] == "inconclusive"
    assert v["checks"]["P1"]["reading"] == "passed" and v["claim"] is None


def test_did_not_beat_names_of_the_gain_checks(env):
    counts = {"P1": 0, "P2": 0, "P3": 0, "P4": 0, "P5": 0, "P6": 2500, "P7": 2600}
    env.stage(make_pass(counts, points={"P6": -0.2, "P7": -0.3}))
    assert env.run() == 0
    c = env.verdict()["checks"]
    assert c["P6"]["reading"] == "AFF did not beat the condition-free scorers on condition gain on new paintings"
    assert c["P7"]["reading"] == "AFF did not beat RCA on condition gain on new paintings"


def check_secondary_untested(env, capsys):
    env.stage(make_pass([17] * 7, s_counts=(0, 0)))
    assert env.run() == 0
    v = env.verdict()
    assert v["verdict"] == "NO-GO" and v["claim"] is None
    s = v["secondary"]
    assert s["tested"] is False and set(s) == {"tested", "S1", "S2"}
    for nm in S:
        assert s[nm]["passes"] is None and s[nm]["reading"] is None and "license" not in s[nm]
        assert set(s[nm]) == {"quantity", "label", "point", "ci95", "passes", "reading"}
    assert "S1, S2: not tested (NO-GO)" in capsys.readouterr().out


def test_secondary_never_tested_after_a_no_go_even_when_its_counts_would_pass(env, capsys):
    check_secondary_untested(env, capsys)


# ---------------------------------------------------------------- boundaries (§8.5)

def test_boundary_report_after_a_go_lists_checks_and_secondary(env):
    counts = {"P1": 0, "P2": 0, "P3": 0, "P4": 0, "P5": 0, "P6": 60, "P7": 124}    # k = 6 n* 61; k = 7 n* 124
    env.stage(make_pass(counts, s_counts=(123, 0)))                                  # S1 at k = 2, n* 124
    assert env.run() == 0
    v = env.verdict()
    assert v["verdict"] == "GO" and v["secondary"]["S1"]["passes"] is True
    rep = {(b["check"], b["holm_k"], b["n"], b["n_star"], b["passes"]) for b in v["boundary_report"]}
    assert rep == {("P6", 6, 60, 61, True), ("P7", 7, 124, 124, True), ("S1", 2, 123, 124, True)}
    assert all(b["rederivation_n"] == b["n"] for b in v["boundary_report"])


def test_boundary_one_count_above_fails_and_is_reported_secondary_not_after_no_go(env):
    counts = {"P1": 0, "P2": 0, "P3": 0, "P4": 0, "P5": 0, "P6": 59, "P7": 125}    # 59: two below 61, not near
    env.stage(make_pass(counts, s_counts=(61, 61)))                                  # S1 near at k = 1, untested
    assert env.run() == 0
    v = env.verdict()
    assert v["verdict"] == "NO-GO" and v["checks"]["P7"]["near_boundary"] is True
    assert [(b["check"], b["passes"]) for b in v["boundary_report"]] == [("P7", False)]


# ---------------------------------------------------------------- modes: smoke, fix1, reserve

def test_smoke_mode_writes_only_under_smoke_and_prints_no_number(env, capsys):
    env.stage(make_pass([0] * 7, smoke=True), smoke=True)
    assert env.run("--smoke") == 0
    assert env.verdict(smoke=True)["mode"] == "smoke"
    assert not (env.results / "held_verdict.json").exists()
    out = capsys.readouterr().out
    assert not DECIMAL.search(out) and "verdict" not in out.replace("held_verdict.json", "")


def test_a_repeated_smoke_runs_in_its_own_subfolder(env, capsys):
    (env.smoke / "held_verdict.json").write_text("{}\n")             # the first smoke's verdict stays
    env.stage(go_rec(smoke=True), smoke=True, subdir="fix1")
    assert env.run("--smoke") == 4                                   # the first folder still refuses
    capsys.readouterr()
    assert env.run("--smoke", "--smoke-subdir", "fix1") == 0
    assert env.verdict(smoke=True, subdir="fix1")["verdict"] == "GO"
    assert (env.smoke / "held_verdict.json").read_text() == "{}\n"
    assert not DECIMAL.search(capsys.readouterr().out)


@pytest.mark.parametrize("argv", [("--smoke-subdir", "fix1"), ("--smoke", "--smoke-subdir", "../x"),
                                  ("--smoke", "--smoke-subdir", "a/b"), ("--smoke", "--smoke-subdir", "")])
def test_the_smoke_subfolder_is_a_plain_name_under_smoke(env, argv):
    assert env.run(*argv) == 4
    assert not list(env.results.rglob("held_verdict*.json"))


def test_fix1_pair_decides_when_its_pass_exists(env):
    env.stage(make_pass([17] * 7))                                   # the first pair (a NO-GO), kept
    env.stage(go_rec(), pass_file="held_pass_fix1.json", sens=False)  # the corrected pass and its own agreement
    assert env.run() == 0
    v = env.verdict()
    assert v["verdict"] == "GO" and (v["pass_file"], v["agreement_file"]) == ("held_pass_fix1.json",
                                                                              "rederive_agreement_fix1.json")
    assert v["held_pass_sha256"] == sha((env.results / "held_pass_fix1.json").read_bytes())
    assert v["agreement_sha256"] == sha((env.results / "rederive_agreement_fix1.json").read_bytes())
    assert v["sensitivity_file"] == "sensitivity_held.json"


def test_a_fix1_pass_reuses_sensitivity_held_and_never_looks_for_a_fix1_copy(env):
    env.stage(make_pass([17] * 7))
    env.stage(make_pass([17] * 7), pass_file="held_pass_fix1.json", sens=False)
    sens = make_sens()
    sens["P1"]["x"] = 0.5
    env.write("sensitivity_held_fix1.json", sens)                    # not a contract file: ignored
    assert env.run() == 0
    v = env.verdict()
    assert v["sensitivity_file"] == "sensitivity_held.json" and v["checks"]["P1"]["x"] == 0.3532


def test_reserve_writes_its_own_verdict_from_its_own_sensitivity_and_keeps_the_original(env):
    (env.results / "held_verdict.json").write_text('{"verdict": "NO-GO"}\n')
    env.write("sensitivity_held.json", make_sens())
    env.stage(make_pass([17] * 7), pass_file="held_pass_reserve.json")
    sens = make_sens()
    sens["P1"]["x"] = 0.5
    env.write("sensitivity_held_reserve.json", sens)
    assert env.run("--reserve") == 0
    v = env.verdict(name="held_verdict_reserve.json")
    assert v["verdict"] == "NO-GO" and v["checks"]["P1"]["x"] == 0.5
    assert (v["pass_file"], v["agreement_file"], v["sensitivity_file"]) == (
        "held_pass_reserve.json", "rederive_agreement_reserve.json", "sensitivity_held_reserve.json")
    assert (env.results / "held_verdict.json").read_text() == '{"verdict": "NO-GO"}\n'


def test_write_once_never_replaces(tmp_path):
    p = tmp_path / "held_verdict.json"
    AR.write_once(p, {"a": 1})
    with pytest.raises(AR.Stop) as e:
        AR.write_once(p, {"a": 2})
    assert e.value.code == 4 and json.loads(p.read_text()) == {"a": 1}
    assert sorted(x.name for x in tmp_path.iterdir()) == ["held_verdict.json"]


# ---------------------------------------------------------------- refusals (exit 4, nothing written)

def _verdict_exists(e):
    e.stage(go_rec())
    (e.results / "held_verdict.json").write_text("{}\n")


def _agreement_missing(e):
    e.stage(go_rec())
    (e.results / "rederive_agreement.json").unlink()


def _all_agree_false(e):
    e.stage(go_rec(), agreement={"all_agree": False, "note": '"all_agree": true'})


def _all_agree_string(e):
    e.stage(go_rec(), agreement={"all_agree": "true"})


def _disagreements(e):
    e.stage(go_rec(), agreement={"disagreements": [{"quantity": "P3 n", "runner": 3, "rederive": 4}]})


def _phase_1(e):
    e.stage(go_rec(), agreement={"phase": 1})


def _sha_unbound(e):                                   # the pass changed after the agreement checked it
    e.stage(go_rec())
    p = e.results / "held_pass.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    rec["checks"]["P1"]["point"] = 0.75                # still a consistent pass file, only not the one checked
    p.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")


def _sha_other_pass(e):
    e.stage(go_rec(), agreement={"held_pass_sha256": sha(b"another pass")})


def _smoke_record_real_mode(e):
    e.stage(go_rec(), agreement_flag=True)


def _real_record_smoke_mode(e):
    e.stage(make_pass([0] * 7, smoke=True), smoke=True, agreement_flag=False)


def _smoke_flag_missing(e):
    e.stage(go_rec(), agreement={"smoke": None})


def _rule_sha(e):
    rec = go_rec()
    rec["rule_sha256"] = "2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925"
    e.stage(rec)


def _mode_regression(e):
    rec = go_rec()
    rec["mode"] = "regression"
    e.stage(rec)


def _held_pass_in_smoke_mode(e):
    e.stage(go_rec(), smoke=True)                      # a pass of mode "held" under results/smoke/


def _seeds(e):
    rec = go_rec()
    rec["seeds"] = [52, 53, 55]
    e.stage(rec)


def _agreement_names_another_pass(e):
    e.stage(go_rec(), pass_file="seed42_pass_counts.json", agreement_name="rederive_agreement.json")


def _first_agreement_names_reserve_pass(e):
    e.stage(go_rec(), pass_file="held_pass_reserve.json", agreement_name="rederive_agreement.json")


def _pass_file_missing(e):
    e.stage(go_rec())
    (e.results / "held_pass.json").unlink()


def _fix1_without_original(e):
    e.stage(go_rec(), pass_file="held_pass_fix1.json")


def _fix1_with_first_agreement_only(e):              # the first agreement is not accepted for a fix-1 pass
    e.stage(make_pass([17] * 7))
    e.write("held_pass_fix1.json", go_rec())


def _fix1_agreement_names_first_pass(e):
    first = e.stage(make_pass([17] * 7))
    e.write("held_pass_fix1.json", go_rec())
    e.write("rederive_agreement_fix1.json", first)     # names held_pass.json, with its SHA-256


def _sensitivity_missing(e):
    e.stage(go_rec(), sens=False)


def _sensitivity_incomplete(e):
    e.stage(go_rec())
    sens = make_sens()
    del sens["P3"]["x"]
    e.write("sensitivity_held.json", sens)


def _sensitivity_without_se(e):
    e.stage(go_rec())
    sens = make_sens()
    del sens["S2"]["SE"]
    e.write("sensitivity_held.json", sens)


def _reserve_sensitivity_missing(e):                 # a reserve read needs its own file, no fallback
    (e.results / "held_verdict.json").write_text("{}\n")
    e.stage(go_rec(), pass_file="held_pass_reserve.json", sens=False)
    e.write("sensitivity_held.json", make_sens())


def _malformed_count(e):
    rec = go_rec()
    rec["checks"]["P2"]["n"] = 0.0
    e.stage(rec)


def _missing_check(e):
    rec = go_rec()
    del rec["checks"]["P7"]
    e.stage(rec)


def _not_json(e):
    e.stage(go_rec())
    (e.results / "rederive_agreement.json").write_text("phase: 2\nall_agree: true\n")


def _nan_point(e):
    rec = go_rec()
    rec["checks"]["P1"]["point"] = float("nan")        # json.dumps writes NaN; the step's parser refuses it
    e.stage(rec)


def _rule_file_changed(e):
    e.stage(go_rec())
    copy = e.results.parent / "DECISION_RULE.md"
    copy.write_bytes(RULE.read_bytes() + b"\n")
    e.mod.RULE_PATH = copy                             # monkeypatch restores it (set in Env)


# name -> (setup, argv, the fragment of the one clause that must refuse it)
REFUSALS = {
    "verdict_exists": (_verdict_exists, (), "held_verdict.json exists; a verdict is never overwritten (rule"),
    "agreement_missing": (_agreement_missing, (), "rederive_agreement.json is missing"),
    "all_agree_false": (_all_agree_false, (), "all_agree is not true"),
    "all_agree_string": (_all_agree_string, (), "all_agree is not true"),
    "disagreements": (_disagreements, (), "disagreements list is not empty"),
    "phase_1": (_phase_1, (), "not a phase-2 record"),
    "sha_unbound": (_sha_unbound, (), "checked another pass"),
    "sha_other_pass": (_sha_other_pass, (), "checked another pass"),
    "smoke_record_real_mode": (_smoke_record_real_mode, (), "a smoke agreement record is refused in real mode"),
    "real_record_smoke_mode": (_real_record_smoke_mode, ("--smoke",),
                               "a real agreement record is refused in smoke mode"),
    "smoke_flag_missing": (_smoke_flag_missing, (), "smoke flag is not a boolean"),
    "rule_sha": (_rule_sha, (), "rule_sha256 is not this rule's"),
    "mode_regression": (_mode_regression, (), "its mode is not 'held'"),
    "held_pass_in_smoke_mode": (_held_pass_in_smoke_mode, ("--smoke",), "its mode is not 'smoke'"),
    "seeds": (_seeds, (), "its seeds are not [52, 53, 54]"),
    "agreement_names_another_pass": (_agreement_names_another_pass, (),
                                     "rederive_agreement.json: its pass_file is not held_pass.json"),
    "first_agreement_names_reserve_pass": (_first_agreement_names_reserve_pass, (),
                                           "rederive_agreement.json: its pass_file is not held_pass.json"),
    "pass_file_missing": (_pass_file_missing, (), "held_pass.json, named by rederive_agreement.json, is missing"),
    "fix1_without_original": (_fix1_without_original, (),
                              "held_pass_fix1.json exists, but the original held_pass.json it corrects is missing"),
    "fix1_with_first_agreement_only": (_fix1_with_first_agreement_only, (),
                                       "rederive_agreement_fix1.json is missing"),
    "fix1_agreement_names_first_pass": (_fix1_agreement_names_first_pass, (),
                                        "rederive_agreement_fix1.json: its pass_file is not held_pass_fix1.json"),
    "sensitivity_missing": (_sensitivity_missing, (), "sensitivity_held.json is missing"),
    "sensitivity_incomplete": (_sensitivity_incomplete, (), "P3 lacks a finite non-negative SE, x, x95"),
    "sensitivity_without_se": (_sensitivity_without_se, (), "S2 lacks a finite non-negative SE, x2, x95"),
    "reserve_sensitivity_missing": (_reserve_sensitivity_missing, ("--reserve",),
                                    "sensitivity_held_reserve.json is missing"),
    "malformed_count": (_malformed_count, (), "P2 does not have the fields and types of contracts"),
    "missing_check": (_missing_check, (), "checks are not exactly P1 to P7"),
    "not_json": (_not_json, (), "rederive_agreement.json is not valid JSON"),
    "nan_point": (_nan_point, (), "held_pass.json is not valid JSON"),
    "rule_file_changed": (_rule_file_changed, (), "DECISION_RULE.md: its SHA-256 is not the committed rule's"),
}


def expect_refused(e: Env, name: str, capsys) -> None:
    setup, argv, fragment = REFUSALS[name]
    setup(e)
    smoke = "--smoke" in argv
    before = {p: p.read_bytes() for p in e.out(smoke).iterdir() if p.is_file()}
    code = e.run(*argv)
    out = capsys.readouterr().out
    assert code == 4, f"{name}: exit {code}"
    assert f"REFUSED: " in out and fragment in out, f"{name}: refused by another clause: {out}"
    after = {p: p.read_bytes() for p in e.out(smoke).iterdir() if p.is_file()}
    assert after == before, f"{name}: files changed"


@pytest.mark.parametrize("name", sorted(REFUSALS))
def test_refusal(env, capsys, name):
    expect_refused(env, name, capsys)


def test_refusals_on_the_reserve_path(env, capsys):
    env.stage(go_rec(), pass_file="held_pass_reserve.json")
    assert env.run("--reserve") == 4                                  # the original verdict is missing
    (env.results / "held_verdict.json").write_text("{}\n")
    (env.results / "held_verdict_reserve.json").write_text("{}\n")
    assert env.run("--reserve") == 4                                  # the reserve verdict exists
    (env.results / "held_verdict_reserve.json").unlink()
    env.stage(go_rec(), pass_file="held_pass.json", agreement_name="rederive_agreement_reserve.json")
    assert env.run("--reserve") == 4                                  # a reserve record must name the reserve pass
    assert not (env.results / "held_verdict_reserve.json").exists()
    with pytest.raises(SystemExit):
        AR.main(["--smoke", "--reserve"])


# ---------------------------------------------------------------- contradictions (exit 5, nothing written)

def _flip(rec, nm, field):
    rec["checks"][nm][field] = not rec["checks"][nm][field]


@pytest.mark.parametrize("mutate", [
    lambda r: _flip(r, "P4", "passes"),
    lambda r: _flip(r, "P5", "own_count_passes"),
    lambda r: _flip(r, "P2", "near_boundary"),
    lambda r: r["checks"]["P3"].update(holm_k=4),
    lambda r: r.update(holm_order=["P2", "P1", "P3", "P4", "P5", "P6", "P7"]),
    lambda r: r["secondary"]["S1"].update(holm_k=2),
    lambda r: r["secondary"]["S2"].update(near_boundary=True),
    lambda r: r["checks"]["P1"].update(level_two_sided=0.95),
], ids=["passes", "own_count", "near_boundary", "holm_k", "holm_order", "s_holm_k", "s_near", "level"])
def test_a_pass_file_that_contradicts_the_integer_holm_stops(env, capsys, mutate):
    rec = make_pass({"P1": 0, "P2": 0, "P3": 5, "P4": 31, "P5": 35, "P6": 50, "P7": 200})
    mutate(rec)
    env.stage(rec)
    assert env.run() == 5
    assert "STOP" in capsys.readouterr().out
    assert not (env.results / "held_verdict.json").exists()


# ---------------------------------------------------------------- mutations on copies (never in place)
# Each guard line of the script carries "# GUARD: <tag>". A mutant copy in tmp_path replaces that line's condition;
# the refusal test (or the gatekeeping test) run against the mutant must then fail.

GUARDS = {
    "sha-binding": ("False", ("sha_unbound", "sha_other_pass")),
    "smoke-flag": ("False", ("smoke_record_real_mode", "real_record_smoke_mode")),
    "verdict-exists": ("False", ("verdict_exists",)),
    "agreement-missing": ("False", ("agreement_missing",)),
    "phase-2": ("False", ("phase_1",)),
    "all-agree": ("False", ("all_agree_false", "all_agree_string")),
    "pass-rule-sha": ("False", ("rule_sha",)),
    "pass-mode": ("False", ("mode_regression", "held_pass_in_smoke_mode")),
    "pass-file-name": ("False", ("agreement_names_another_pass", "fix1_agreement_names_first_pass")),
    "fix1-pair": ("False", ("fix1_with_first_agreement_only",)),
}


def load_mutant(tmp_path: Path, tag: str, cond: str):
    src = SCRIPT.read_text(encoding="utf-8")
    pat = re.compile(rf"^(\s*)if .+:  # GUARD: {re.escape(tag)}$", re.M)
    assert len(pat.findall(src)) == 1, tag
    mdir = tmp_path / f"mutant_{tag}"
    mdir.mkdir()
    (mdir / SCRIPT.name).write_text(pat.sub(lambda m: f"{m.group(1)}if {cond}:  # GUARD: {tag}", src),
                                    encoding="utf-8")
    return load_module(mdir / SCRIPT.name, f"mutant_{tag.replace('-', '_')}")


def test_every_guard_of_the_script_has_a_mutation_test():
    tags = re.findall(r"# GUARD: (\S+)$", SCRIPT.read_text(encoding="utf-8"), re.M)
    assert sorted(tags) == sorted(set(GUARDS) | {"gatekeeping"})


@pytest.mark.parametrize("tag", sorted(GUARDS))
def test_dropping_a_guard_makes_its_refusal_test_fail(tmp_path, monkeypatch, capsys, tag):
    cond, scenarios = GUARDS[tag]
    mutant = load_mutant(tmp_path, tag, cond)
    for i, name in enumerate(scenarios):
        e = Env(tmp_path / f"run{i}", monkeypatch, mod=mutant)
        with pytest.raises((AssertionError, OSError)):       # not refused by its clause: a verdict, or a crash
            expect_refused(e, name, capsys)


def test_dropping_the_gatekeeping_makes_its_test_fail(tmp_path, monkeypatch, capsys):
    mutant = load_mutant(tmp_path, "gatekeeping", "True")
    with pytest.raises(AssertionError):
        check_secondary_untested(Env(tmp_path / "run", monkeypatch, mod=mutant), capsys)


