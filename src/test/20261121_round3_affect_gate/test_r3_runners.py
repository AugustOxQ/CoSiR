"""Unit tests of the pure pieces of round 3's runners (rule DECISION_RULE.md §6.2, §6.5 to §6.7, §8 boundaries, §10):
the seed guards of the build and of both test phases, the build's crash rule, codes_provenance and episode-hash checks,
the build-record check, the §6.7 reading, the GO decision, the boundary stop, the rule application's refusals
(no agreement record, a sensitivity override outside smoke mode) and the descriptive phase's refusal without a verdict.
Synthetic files only (tmp_path); no episode seed is built or scored. Run from /project/CoSiR:
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest src/test/20261121_round3_affect_gate/test_r3_runners.py -q \
        -p no:cacheprovider
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r3_apply_rule as AR  # noqa: E402
import r3_common as R3  # noqa: E402
import r3_stats as RS  # noqa: E402
import run_r3_build as BLD  # noqa: E402
import run_r3_test as RT  # noqa: E402

from src.eval.aspect_episodes import AspectEpisodes, episodes_sha256  # noqa: E402

CHECKS8 = RS.GO_CHECKS + ("secondary",)


# ---------------------------------------------------------------- seed guards

def test_build_seed_guard():
    assert BLD.check_seeds([49, 50, 51], False) == (49, 50, 51)
    assert BLD.check_seeds([50], False) == (50,)
    assert BLD.check_seeds([9002, 9003], True) == (9002, 9003)
    for seeds, smoke in (([42], False), ([52], False), ([48, 49], False), ([9001], False), ([49], True),
                         ([9004], True), ([], False), ([49, 49], False), ([51, 49], False)):
        with pytest.raises(SystemExit):
            BLD.check_seeds(seeds, smoke)


def test_test_phases_need_exactly_the_three_seeds_in_order():
    assert BLD.check_seeds([49, 50, 51], False, full=True) == (49, 50, 51)
    assert BLD.check_seeds([9001, 9002, 9003], True, full=True) == (9001, 9002, 9003)
    for seeds, smoke in (([49, 50], False), ([49, 50, 51, 52], False), ([9001, 9002, 9003], False),
                         ([49, 50, 51], True), ([9001, 9002], True), ([51, 50, 49], False)):
        with pytest.raises(SystemExit):
            BLD.check_seeds(seeds, smoke, full=True)


def _forbid(monkeypatch, *targets):
    def boom(*a, **k):
        raise AssertionError("must not be called before the seed guard refuses")
    for mod, name in targets:
        monkeypatch.setattr(mod, name, boom)


def test_go_phase_refuses_other_seeds_before_reading_anything(monkeypatch, tmp_path):
    monkeypatch.setattr(R3, "RES", tmp_path)
    _forbid(monkeypatch, (RT.RB, "build_bundle"), (RT.BLD, "verify_build_record"), (RT.R3, "assert_taus"))
    for seeds, smoke in (((49, 50), False), ((48, 49, 50), False), ((9001, 9002, 9003), False), ((49, 50, 51), True),
                         ((9001, 9002), True), ((42,), False), ((9001, 9002, 9004), True)):
        with pytest.raises(SystemExit, match="seed"):
            RT.go_phase(seeds, smoke)
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------- descriptive phase refuses without a verdict

def test_descriptive_refuses_without_verdict(monkeypatch, tmp_path):
    monkeypatch.setattr(R3, "RES", tmp_path)
    _forbid(monkeypatch, (RT.RB, "load_bundle_cache"), (RT.BLD, "verify_build_record"))
    with pytest.raises(SystemExit, match="test_verdict"):
        RT.descriptive_phase((9001, 9002, 9003), True)
    with pytest.raises(SystemExit, match="test_verdict"):
        RT.descriptive_phase((49, 50, 51), False)


def test_descriptive_refuses_a_verdict_of_another_rule_or_mode(monkeypatch, tmp_path):
    monkeypatch.setattr(R3, "RES", tmp_path)
    _forbid(monkeypatch, (RT.RB, "load_bundle_cache"), (RT.BLD, "verify_build_record"))
    sm = tmp_path / "smoke"
    sm.mkdir()
    (sm / "test_verdict.json").write_text(json.dumps({"rule_sha256": "0" * 64, "smoke": True, "verdict": "GO"}))
    with pytest.raises(SystemExit, match="rule"):
        RT.descriptive_phase((9001, 9002, 9003), True)
    (sm / "test_verdict.json").write_text(json.dumps({"rule_sha256": R3.RULE_SHA, "smoke": False, "verdict": "GO"}))
    with pytest.raises(SystemExit, match="smoke"):
        RT.descriptive_phase((9001, 9002, 9003), True)


def test_go_npz_keys_are_only_those_of_rule_6_4():
    keys = set(RT.go_npz_keys())
    for who in ("aff_fused", "aff_cf", "r1_fused", "B", "Bp", "cosine", "rca"):
        for m in ("r1", "gain", "other", "swap", "strict"):
            assert f"{who}__{m}" in keys
    assert not any(k.startswith("r1_cf") for k in keys)
    assert not any("bar" in k or "gate" in k or "pick" in k or "share" in k for k in keys)
    assert {"cl", "pair_index", "parity", "aff_fused_cells", "aff_cf_cells", "r1_fused_cells", "sigma", "meta"} <= keys


# ---------------------------------------------------------------- the build: hashes, codes_provenance, crash rule

def _write_fake_seed(ab_res, seed, smoke, n, rng, parts=("episodes", "per_anchor", "baselines"), same_as=None):
    """A stand-in for run_baselines.py's three outputs (random episodes; no scorer)."""
    d = ab_res / ("smoke" if smoke else "")
    d.mkdir(parents=True, exist_ok=True)
    saved, shas = {}, {}
    for p in BLD.PAIR_ORDER:
        a, b = p.split("__")
        if same_as is not None:
            arrs = same_as[p]
        else:
            arrs = [rng.integers(0, 10 ** 6, size=n), rng.integers(0, 10 ** 6, size=(n, 13))] + \
                   [rng.integers(0, 10 ** 6, size=(n, 4)) for _ in range(4)]
        ep = AspectEpisodes(a, b, *[np.asarray(x, np.int64) for x in arrs])
        shas[p] = episodes_sha256(ep)
        for f, v in zip(BLD.EP_FIELDS, arrs):
            saved[f"{p}__{f}"] = np.asarray(v, np.int64)
    if "episodes" in parts:
        np.savez(d / f"episodes_seed{seed}.npz", pair_order=np.array(BLD.PAIR_ORDER), **saved)
    if "per_anchor" in parts:
        np.savez(d / f"per_anchor_seed{seed}.npz", anchor_group=np.zeros(3 * n, np.int64))
    if "baselines" in parts:
        (d / f"baselines_seed{seed}.json").write_text(json.dumps(
            {"episodes_seed": seed, "n_per_pair": n, "pair_order": list(BLD.PAIR_ORDER), "episodes_sha256": shas}))
    return shas, {p: [saved[f"{p}__{f}"] for f in BLD.EP_FIELDS] for p in BLD.PAIR_ORDER}


class FakeRunner:
    """Plays run_baselines.py: each call follows the next script entry ('ok', 'crash_partial', 'crash_none',
    'ok_other_episodes', 'crash_after_baselines')."""

    def __init__(self, ab_res, script, n, seed_rng=0):
        self.ab_res, self.script, self.n, self.calls = ab_res, list(script), n, 0
        self.rng = np.random.default_rng(seed_rng)
        self.first = None

    def __call__(self, seed, smoke, log_file):
        what = self.script[self.calls]
        self.calls += 1
        Path(log_file).write_text(f"fake build {what}\n")
        if what == "crash_none":
            return 1
        if what == "crash_partial":
            _, self.first = _write_fake_seed(self.ab_res, seed, smoke, self.n, np.random.default_rng(seed),
                                             parts=("episodes",))
            return 1
        if what == "crash_after_baselines":
            _write_fake_seed(self.ab_res, seed, smoke, self.n, np.random.default_rng(seed))
            return 1
        rng = np.random.default_rng(seed) if what == "ok" else self.rng
        _write_fake_seed(self.ab_res, seed, smoke, self.n, rng)
        return 0


def _earlier(rng):
    return {s: {p: format(int(rng.integers(0, 2 ** 62)), "064x") for p in BLD.PAIR_ORDER} for s in R3.EARLIER_SEEDS}


def test_hash_check():
    new = {"emotion__style": "a", "emotion__genre": "b", "style__genre": "c"}
    ok, probs = BLD.hash_check(new, {}, {42: {"emotion__style": "x", "emotion__genre": "y", "style__genre": "z"}})
    assert ok and not probs
    assert not BLD.hash_check({**new, "style__genre": "a"}, {}, {})[0]
    assert not BLD.hash_check(new, {50: {"emotion__style": "q", "emotion__genre": "c", "style__genre": "r"}}, {})[0]
    assert not BLD.hash_check(new, {}, {47: {"emotion__style": "x", "emotion__genre": "y", "style__genre": "b"}})[0]


def test_build_clean_smoke(tmp_path):
    ab, rec = tmp_path / "ab", tmp_path / "rec"
    run = FakeRunner(ab, ["ok"], 8)
    r = BLD.build_seed(9002, True, runner=run, ab_res=ab, rec_dir=rec, earlier=_earlier(np.random.default_rng(1)),
                       n_per_pair=8)
    assert r["passed"] is True and run.calls == 1 and r["built_here"] is True
    assert r["codes_provenance_before"] is None
    rec_json = json.loads((rec / "build_seed9002.json").read_text())
    assert rec_json["passed"] is True and rec_json["provenance"]["rule_sha256"] == R3.RULE_SHA
    assert set(rec_json["sha256"]) == {"episodes", "per_anchor", "baselines"}
    BLD.verify_build_record(9002, True, ab_res=ab, rec_dir=rec)


def test_build_crash_rule_repeats_once_and_requires_equal_episodes(tmp_path):
    ab, rec = tmp_path / "ab", tmp_path / "rec"
    run = FakeRunner(ab, ["crash_partial", "ok"], 8)
    r = BLD.build_seed(9002, True, runner=run, ab_res=ab, rec_dir=rec, earlier=_earlier(np.random.default_rng(1)),
                       n_per_pair=8)
    assert run.calls == 2 and r["passed"] is True
    a1, a2 = r["attempts"]
    assert a1["exit_status"] == 1 and a2["exit_status"] == 0
    assert a1["episodes_sha256"] == r["sha256"]["episodes"]          # the same deterministic episodes
    assert a1["deleted_partial_outputs"] == ["episodes"]


def test_build_crash_rule_second_failure_stops(tmp_path):
    ab, rec = tmp_path / "ab", tmp_path / "rec"
    run = FakeRunner(ab, ["crash_none", "crash_partial"], 8)
    with pytest.raises(SystemExit, match="second"):
        BLD.build_seed(9002, True, runner=run, ab_res=ab, rec_dir=rec, earlier=_earlier(np.random.default_rng(1)),
                       n_per_pair=8)
    assert run.calls == 2
    assert json.loads((rec / "build_seed9002.json").read_text())["passed"] is False


def test_build_crash_rule_unequal_episodes_stop(tmp_path):
    ab, rec = tmp_path / "ab", tmp_path / "rec"
    run = FakeRunner(ab, ["crash_partial", "ok_other_episodes"], 8)
    with pytest.raises(SystemExit, match="episode"):
        BLD.build_seed(9002, True, runner=run, ab_res=ab, rec_dir=rec, earlier=_earlier(np.random.default_rng(1)),
                       n_per_pair=8)
    assert json.loads((rec / "build_seed9002.json").read_text())["passed"] is False


def test_build_crash_after_baselines_is_not_repeated(tmp_path):
    ab, rec = tmp_path / "ab", tmp_path / "rec"
    run = FakeRunner(ab, ["crash_after_baselines", "ok"], 8)
    with pytest.raises(SystemExit):
        BLD.build_seed(9002, True, runner=run, ab_res=ab, rec_dir=rec, earlier=_earlier(np.random.default_rng(1)),
                       n_per_pair=8)
    assert run.calls == 1


def test_build_never_rebuilds_a_completed_seed(tmp_path):
    ab, rec = tmp_path / "ab", tmp_path / "rec"
    _write_fake_seed(ab, 50, False, 8, np.random.default_rng(5))
    run = FakeRunner(ab, ["ok"], 8)
    with pytest.raises(SystemExit, match="never rebuilt"):
        BLD.build_seed(50, False, runner=run, ab_res=ab, rec_dir=rec, earlier=_earlier(np.random.default_rng(1)),
                       n_per_pair=8, codes_sha=lambda: BLD.CODES_SHA)
    assert run.calls == 0


def test_build_refuses_partial_outputs_of_an_earlier_run(tmp_path):
    ab, rec = tmp_path / "ab", tmp_path / "rec"
    _write_fake_seed(ab, 50, False, 8, np.random.default_rng(5), parts=("episodes",))
    run = FakeRunner(ab, ["ok"], 8)
    with pytest.raises(SystemExit, match="partial"):
        BLD.build_seed(50, False, runner=run, ab_res=ab, rec_dir=rec, earlier=_earlier(np.random.default_rng(1)),
                       n_per_pair=8, codes_sha=lambda: BLD.CODES_SHA)
    assert run.calls == 0


def test_build_smoke_records_an_existing_smoke_seed_without_rebuilding(tmp_path):
    ab, rec = tmp_path / "ab", tmp_path / "rec"
    _write_fake_seed(ab, 9001, True, 8, np.random.default_rng(5))
    run = FakeRunner(ab, ["ok"], 8)
    r = BLD.build_seed(9001, True, runner=run, ab_res=ab, rec_dir=rec, earlier=_earlier(np.random.default_rng(1)),
                       n_per_pair=8)
    assert run.calls == 0 and r["passed"] is True and r["built_here"] is False


def test_build_codes_provenance_must_stay(tmp_path):
    ab, rec = tmp_path / "ab", tmp_path / "rec"
    seq = iter([BLD.CODES_SHA, "f" * 64])                 # before: as D15; after: changed
    run = FakeRunner(ab, ["ok"], 8)
    with pytest.raises(SystemExit, match="codes_provenance"):
        BLD.build_seed(49, False, runner=run, ab_res=ab, rec_dir=rec, earlier=_earlier(np.random.default_rng(1)),
                       n_per_pair=8, codes_sha=lambda: next(seq))
    assert json.loads((rec / "build_seed49.json").read_text())["passed"] is False
    run2 = FakeRunner(tmp_path / "ab2", ["ok"], 8)
    with pytest.raises(SystemExit, match="codes_provenance"):          # changed before the build: nothing runs
        BLD.build_seed(50, False, runner=run2, ab_res=tmp_path / "ab2", rec_dir=tmp_path / "rec2",
                       earlier=_earlier(np.random.default_rng(1)), n_per_pair=8, codes_sha=lambda: "e" * 64)
    assert run2.calls == 0


def test_build_hash_match_with_an_earlier_seed_stops(tmp_path):
    ab, rec = tmp_path / "ab", tmp_path / "rec"
    shas, _ = _write_fake_seed(tmp_path / "probe", 9002, True, 8, np.random.default_rng(9002))
    earlier = _earlier(np.random.default_rng(1))
    earlier[45]["style__genre"] = shas["emotion__genre"]            # the fake build of 9002 reproduces these
    run = FakeRunner(ab, ["ok"], 8)
    with pytest.raises(SystemExit, match="hash"):
        BLD.build_seed(9002, True, runner=run, ab_res=ab, rec_dir=rec, earlier=earlier, n_per_pair=8)
    assert json.loads((rec / "build_seed9002.json").read_text())["passed"] is False


def test_build_hash_match_with_another_new_seed_stops(tmp_path):
    ab, rec = tmp_path / "ab", tmp_path / "rec"
    earlier = _earlier(np.random.default_rng(1))
    BLD.build_seed(9002, True, runner=FakeRunner(ab, ["ok"], 8), ab_res=ab, rec_dir=rec, earlier=earlier, n_per_pair=8)
    _, arrs = _write_fake_seed(tmp_path / "probe", 9002, True, 8, np.random.default_rng(9002))

    def copy_runner(seed, smoke, log_file):                       # 9003 gets 9002's episodes
        Path(log_file).write_text("fake\n")
        _write_fake_seed(ab, seed, smoke, 8, None, same_as=arrs)
        return 0
    with pytest.raises(SystemExit, match="hash"):
        BLD.build_seed(9003, True, runner=copy_runner, ab_res=ab, rec_dir=rec, earlier=earlier, n_per_pair=8)


def test_build_refuses_when_another_new_seed_failed(tmp_path):
    ab, rec = tmp_path / "ab", tmp_path / "rec"
    with pytest.raises(SystemExit):
        BLD.build_seed(9002, True, runner=FakeRunner(ab, ["crash_none", "crash_none"], 8), ab_res=ab, rec_dir=rec,
                       earlier=_earlier(np.random.default_rng(1)), n_per_pair=8)
    run = FakeRunner(ab, ["ok"], 8)
    with pytest.raises(SystemExit, match="did not pass"):
        BLD.build_seed(9003, True, runner=run, ab_res=ab, rec_dir=rec, earlier=_earlier(np.random.default_rng(1)),
                       n_per_pair=8)
    assert run.calls == 0                                          # refused before building


def test_verify_build_record_detects_a_changed_file(tmp_path):
    ab, rec = tmp_path / "ab", tmp_path / "rec"
    BLD.build_seed(9002, True, runner=FakeRunner(ab, ["ok"], 8), ab_res=ab, rec_dir=rec,
                   earlier=_earlier(np.random.default_rng(1)), n_per_pair=8)
    BLD.verify_build_record(9002, True, ab_res=ab, rec_dir=rec)
    p = ab / "smoke" / "per_anchor_seed9002.npz"
    np.savez(p, anchor_group=np.ones(24, np.int64))
    with pytest.raises(SystemExit, match="SHA-256"):
        BLD.verify_build_record(9002, True, ab_res=ab, rec_dir=rec)
    with pytest.raises(SystemExit, match="build record"):
        BLD.verify_build_record(9003, True, ab_res=ab, rec_dir=rec)


def test_cross_check_of_the_three_records():
    rng = np.random.default_rng(3)

    def rec(seed):
        return {"seed": seed, "episode_pair_sha256": {p: format(int(rng.integers(0, 2 ** 62)), "064x")
                                                      for p in BLD.PAIR_ORDER}}
    recs = {s: rec(s) for s in (49, 50, 51)}
    earlier = _earlier(rng)
    BLD.cross_check(recs, earlier=earlier)
    recs[51]["episode_pair_sha256"]["style__genre"] = recs[49]["episode_pair_sha256"]["emotion__style"]
    with pytest.raises(SystemExit, match="hash"):
        BLD.cross_check(recs, earlier=earlier)


# ---------------------------------------------------------------- rule application: §6.7 reading, GO, boundaries

def _check(point, lo, hi):
    return {"point": point, "ci95": [lo, hi], "pass": bool(lo > 0)}


def _pooled(over=None, sec=(0.2, 0.05, 0.35)):
    checks = {k: _check(1.0, 0.5, 1.5) for k in RS.GO_CHECKS}
    for k, v in (over or {}).items():
        checks[k] = _check(*v)
    return {"checks": checks, "go": all(c["pass"] for c in checks.values()), "secondary": _check(*sec)}


def _sens(x=0.25):
    return {"checks": {k: {"SE": x / 2.8, "half_width": 1.96 * x / 2.8, "x": x, "seed42_half_width": 0.3}
                       for k in CHECKS8}}


def test_reading_of_a_failed_check_rule_6_7():
    r = AR.read_check(_check(0.12, -0.03, 0.27), 0.18, "cosine")
    assert r["kind"] == "inconclusive" and r["x"] == 0.18 and r["realised_half_width"] == pytest.approx(0.15)
    assert "inconclusive at a detectable margin of x" in r["reading"] and "not evidence that AFF fails" in r["reading"]
    for pt in (0.0, -0.4):
        r = AR.read_check(_check(pt, -0.5, 0.3), 0.18, "cosine")
        assert r["kind"] == "not_beaten" and r["reading"] == "AFF did not beat cosine on fresh episodes"
    assert AR.read_check(_check(0.3, 1e-9, 0.6), 0.18, "cosine")["kind"] == "pass"
    assert AR.NAMES == {"r1_vs_cosine": "cosine", "r1_vs_rca": "RCA", "r1_vs_B": "B", "r1_vs_Bprime": "B′(A0)",
                        "r1_vs_counterpart": "the matched counterpart",
                        "gain_statistic": "the condition-free comparators on condition gain",
                        "gain_vs_rca": "RCA on condition gain"}


def test_go_iff_all_seven_lower_bounds_above_zero():
    d = AR.decide(_pooled(), _sens())
    assert d["verdict"] == "GO" and d["failed"] == []
    d = AR.decide(_pooled({"r1_vs_B": (0.4, 0.0, 0.8)}), _sens())             # lower bound exactly 0 fails
    assert d["verdict"] == "NO-GO" and d["failed"] == ["r1_vs_B"]
    assert d["checks"]["r1_vs_B"]["kind"] == "inconclusive"
    d = AR.decide(_pooled({"gain_vs_rca": (-0.1, -0.4, 0.2), "r1_vs_cosine": (0.1, -0.1, 0.3)}), _sens(0.3))
    assert d["verdict"] == "NO-GO" and d["failed"] == ["r1_vs_cosine", "gain_vs_rca"]
    assert d["checks"]["gain_vs_rca"]["reading"] == "AFF did not beat RCA on condition gain on fresh episodes"
    assert d["checks"]["r1_vs_cosine"]["x"] == 0.3


def test_secondary_never_changes_go():
    d = AR.decide(_pooled(sec=(-0.2, -0.5, 0.1)), _sens())
    assert d["verdict"] == "GO" and d["secondary"]["pass"] is False
    assert d["secondary"]["reading"] == "AFF did not beat R1 on fresh episodes"
    d = AR.decide(_pooled({"r1_vs_rca": (-1.0, -2.0, 0.0)}, sec=(0.3, 0.1, 0.5)), _sens())
    assert d["verdict"] == "NO-GO" and d["secondary"]["pass"] is True


def test_decide_refuses_an_inconsistent_pass_flag():
    p = _pooled()
    p["checks"]["r1_vs_B"]["pass"] = False                                    # lower bound 0.5 says pass
    with pytest.raises(SystemExit):
        AR.decide(p, _sens())


def test_boundary_hits():
    assert AR.boundary_hits(_pooled()) == []
    hits = AR.boundary_hits(_pooled({"r1_vs_rca": (0.2, 5e-13, 0.4)}, sec=(0.1, -1e-12, 0.2)))
    assert [h["check"] for h in hits] == ["r1_vs_rca", "secondary"]
    assert hits[0]["lower_bound"] == 5e-13 and hits[0]["threshold"] == 0.0
    assert AR.boundary_hits(_pooled({"r1_vs_rca": (0.2, 1.1e-12, 0.4)})) == []


def _setup_apply(tmp_path, monkeypatch, smoke, pooled, agreement=None, sens=True):
    monkeypatch.setattr(R3, "RES", tmp_path / "res")
    out = R3.res_dir(smoke)
    files = {}
    for s in (R3.SMOKE_SEEDS if smoke else R3.TEST_SEEDS):
        p = out / f"go_seed{s}.npz"
        np.savez(p, cl=np.arange(4))
        files[p.name] = R3.sha_file(p)
    rec = {**pooled, "seeds": list(R3.SMOKE_SEEDS if smoke else R3.TEST_SEEDS), "smoke": smoke, "files_sha256": files,
           "provenance": {"rule_sha256": R3.RULE_SHA, "smoke": smoke}}
    (out / "go_pooled.json").write_text(json.dumps(rec))
    sp = tmp_path / "standin_sensitivity.json"
    sp.write_text(json.dumps({**_sens(), "stand_in": True, "provenance": {"rule_sha256": R3.RULE_SHA, "smoke": True}}))
    if sens:
        real = tmp_path / "res" / "sensitivity.json"
        real.write_text(json.dumps({**_sens(), "provenance": {"rule_sha256": R3.RULE_SHA, "smoke": False}}))
    ag = tmp_path / "phase2_agreement.json"
    monkeypatch.setattr(AR, "AGREEMENT", ag)
    if agreement is not None:
        ag.write_text(json.dumps(agreement))
    return out, sp


def test_apply_rule_non_smoke_refuses_without_agreement(tmp_path, monkeypatch):
    out, _ = _setup_apply(tmp_path, monkeypatch, False, _pooled())
    with pytest.raises(SystemExit, match="agreement"):
        AR.main([])
    for bad in ({"all_agree": False, "rule_sha256": R3.RULE_SHA}, {"all_agree": True, "rule_sha256": "0" * 64},
                {"all_agree": "true", "rule_sha256": R3.RULE_SHA}):
        (tmp_path / "phase2_agreement.json").write_text(json.dumps(bad))
        with pytest.raises(SystemExit, match="agreement"):
            AR.main([])
    assert not (out / "test_verdict.json").exists()
    (tmp_path / "phase2_agreement.json").write_text(json.dumps({"all_agree": True, "rule_sha256": R3.RULE_SHA}))
    AR.main([])
    v = json.loads((out / "test_verdict.json").read_text())
    assert v["verdict"] == "GO" and v["rule_sha256"] == R3.RULE_SHA and v["smoke"] is False
    assert (out / "test_verdict.txt").exists()
    with pytest.raises(SystemExit, match="refusing to overwrite"):                    # never overwritten
        AR.main([])


def test_apply_rule_sensitivity_override_only_in_smoke(tmp_path, monkeypatch):
    _, sp = _setup_apply(tmp_path, monkeypatch, False, _pooled(),
                         agreement={"all_agree": True, "rule_sha256": R3.RULE_SHA})
    with pytest.raises(SystemExit, match="sensitivity"):
        AR.main(["--sensitivity", str(sp)])


def test_apply_rule_smoke_reads_the_real_sensitivity_by_default(tmp_path, monkeypatch):
    out, sp = _setup_apply(tmp_path, monkeypatch, True, _pooled({"r1_vs_B": (0.4, -0.1, 0.9)}), sens=False)
    with pytest.raises(SystemExit, match="sensitivity"):                 # the real file does not exist yet
        AR.main(["--smoke"])
    AR.main(["--smoke", "--sensitivity", str(sp)])                       # no agreement record needed in smoke mode
    v = json.loads((out / "test_verdict.json").read_text())
    assert v["verdict"] == "NO-GO" and v["checks"]["r1_vs_B"]["kind"] == "inconclusive"


def test_apply_rule_refuses_a_go_file_that_changed(tmp_path, monkeypatch):
    out, sp = _setup_apply(tmp_path, monkeypatch, True, _pooled())
    np.savez(out / "go_seed9002.npz", cl=np.arange(5))
    with pytest.raises(SystemExit, match="SHA-256"):
        AR.main(["--smoke", "--sensitivity", str(sp)])


def test_apply_rule_boundary_stops_with_exit_3(tmp_path, monkeypatch):
    out, sp = _setup_apply(tmp_path, monkeypatch, True, _pooled({"gain_vs_rca": (0.0, 0.0, 0.0)}))
    with pytest.raises(SystemExit) as e:
        AR.main(["--smoke", "--sensitivity", str(sp)])
    assert e.value.code == 3
    assert (out / "verdict_boundary.json").exists() and not (out / "test_verdict.json").exists()
    b = json.loads((out / "verdict_boundary.json").read_text())
    assert [h["check"] for h in b["hits"]] == ["gain_vs_rca"]
    AR.main(["--smoke", "--sensitivity", str(sp), "--boundary-reported"])          # after the user was told
    v = json.loads((out / "test_verdict.json").read_text())
    assert v["verdict"] == "NO-GO" and [h["check"] for h in v["boundary"]] == ["gain_vs_rca"]


def test_apply_rule_boundary_flag_without_a_boundary_is_refused(tmp_path, monkeypatch):
    _, sp = _setup_apply(tmp_path, monkeypatch, True, _pooled())
    with pytest.raises(SystemExit, match="boundary"):
        AR.main(["--smoke", "--sensitivity", str(sp), "--boundary-reported"])
