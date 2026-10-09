"""Real-data test of the descriptive pass (ticket 13) on seed 42, selection rows: the regression's outputs stand in for
held arrays. Nothing of a held row is read; F/results is never written; prints no metric.

One module fixture runs run_r6_held's own setup (inputs, rows, heads refit and checked, PM fits, readers), builds the
seed-42 selection bundle with its seed_bundle, writes picks_seed42.json into tmp_path with run_r6_picks.run, scores
the bundle with score_bundle(..., include_pm=True) and writes the stand-ins in tmp_path: held_arrays.npz in contracts
section 7's layout (r6_descriptive.arrays_from_scored), the seed's episode file (r6_episodes.build_seed on selection
rows, save_episodes) and the pass record (r6_stats.pass_record, mode "regression", the seed-42 counts file's
content). Then the descriptive core runs on seed 42 alone (check_pass, seed_inputs with its rescoring, describe), and
its numbers are compared, exactly, with every seed-42 value an earlier round recorded:

  - round 3 (R3 rule section 5 item 3; r3_common.AFF_BRAINSTORM, the values round 3's regression matched in
    results/regression_check.json item 3): AFF fused and CF R@1; bar comparator B'(A0); bar margin; margin, gain
    statistic and either change against the counterpart; per-pair bar margins; AFF - R1 in fused R@1 and in bar
    margin; AFF's tau_0 gate open counts 9,941 (a) and 3,627 (b);
  - round 3 (R3 rule section 5 item 2; r3_common.RC_NUMBERS; regression_check.json item 2): R1 fused and counterpart
    R@1, bar comparator "counterpart", R1's margin and gain statistic (two of R1's seven checks);
  - round 1 (R1 = round 1's R-c, R3 rule section 5 item 2; 20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.json,
    an R3 rule D15 input): R1 against B and against B'(A0) (r1 and gain), the either change against its
    counterpart, R1's per-pair bar margins (r1 and gain), pick accuracy (R3 rule D14: 51.261393229166664) with its
    interval, both-correct share, per pair and condition, pick shares;
  - R3 rule D7 (r3_common.REDUNDANCY_42; regression_check.json "redundancy_D7"): the six redundancy values and
    "affect least redundant in both directions";
  - D6: R1's tau_0 gate open on all 12,288 episodes in both conditions (tau_0 is seed 42's smallest margin);
  - round 4 (20261122_round4_aff_vetoes/results/dev_seed42.json "beside_aff", rule section 6 item 3 and 4): B, B0 and
    B1 mean R@1, AFF - B1;
  - AB (20261030_aspect_baselines/results/baselines_seed42.json, an R3 rule D15 input): cosine, RCA and the nine PM
    scorers' R@1, gain and swap with intervals, overall and per pair.

About 8 to 12 minutes on CPU:

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_descriptive_seed42.py
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_bundle as B  # noqa: E402
import r6_descriptive as D  # noqa: E402
import r6_episodes as E  # noqa: E402
import r6_picks as P  # noqa: E402
import r6_stats as ST  # noqa: E402
import run_r6_held as RH  # noqa: E402
import run_r6_picks as RP  # noqa: E402

import numpy as np  # noqa: E402

R3 = R.R3
AFF, RC = R3.AFF_BRAINSTORM, R3.RC_NUMBERS
ROUND1_REL = "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.json"
DEV42_REL = "20261122_round4_aff_vetoes/results/dev_seed42.json"
AB42_REL = "20261030_aspect_baselines/results/baselines_seed42.json"
N = 3 * R.N_PER_PAIR


def _json(rel):
    R.assert_input(rel)                                  # SHA-256 asserted (rule section 6 item 1)
    return json.loads(Path(R.INPUT_PATHS[rel]).read_text())


def ci3(pc) -> list:
    return [pc["point"], *pc["ci95"]]


def pc2(pc) -> dict:
    return {"point": pc["point"], "ci95": list(pc["ci95"])}


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    env = RH.setup()
    bundle = RH.seed_bundle(env, "selection", R.DEV_SEED, R.N_PER_PAIR)
    tmp = tmp_path_factory.mktemp("stand_in")
    assert RP.run(bundle, tmp / P.PICKS_NAME) == 0
    picks = P.load_picks(tmp / P.PICKS_NAME)
    lambdas = P.frozen_lambdas()
    scored = RH.score_bundle(env, bundle, picks, lambdas, include_pm=True)
    np.savez(tmp / "held_arrays.npz", **D.arrays_from_scored([scored], [R.DEV_SEED]))
    eps = E.build_seed(env.labels, env.split.groups, env.split.selection, env.index, env.value_sets, R.DEV_SEED,
                       R.N_PER_PAIR)
    E.save_episodes(tmp / "held_episodes_seed42.npz", eps)
    extra = {"episodes_sha256": {R.DEV_SEED: dict(bundle.episodes_sha256)}, "runner_sha256": RH.runner_sha256(),
             "module_sha256": R.r6_module_shas()}
    prec = ST.pass_record([scored], "regression", [R.DEV_SEED], extra)
    core = D.load_core(tmp / "held_arrays.npz", [R.DEV_SEED], R.N_PER_PAIR)
    agree = D.check_pass(core, prec)
    episodes = E.load_episodes(tmp / "held_episodes_seed42.npz")
    d = D.seed_inputs(bundle, picks, lambdas, env.readers, core[R.DEV_SEED], episodes, env.split.groups)
    rec = json.loads(json.dumps(D.describe([d])))        # through JSON, as descriptive.json stores it
    return SimpleNamespace(rec=rec, prec=prec, agree=agree, d=d, scored=scored, episodes=episodes, bundle=bundle)


def test_stand_in_binds_to_its_pass_record(real):
    assert real.agree == {c: True for c in ST.CHECKS + ST.SECONDARY}
    assert dict(real.episodes.sha) == dict(real.bundle.episodes_sha256) == E.identity_targets()[42]["episodes_sha256"]
    assert real.rec["seeds"] == [42] and real.rec["n_episodes"] == N and real.rec["n_clusters"] == 4602
    for c in ST.CHECKS + ST.SECONDARY:
        want = (real.prec["checks"] if c in ST.CHECKS else real.prec["secondary"])[c]
        assert ci3(real.rec["checks_by_scope"]["pooled"][c]) == [want["point"], *want["ci95"]], c
        tw = real.rec["two_way_bootstrap"]["quantities"][c]
        assert ci3({"point": tw["point"], "ci95": tw["ci95_anchor"]}) == [want["point"], *want["ci95"]], c


def test_aff_equals_round3(real):
    """r3_common.AFF_BRAINSTORM (R3 rule section 5 item 3; round 3's regression_check.json item 3)."""
    rec = real.rec
    rows = rec["rows"]
    assert rows["aff_fused"]["pooled"]["r1"]["point"] == AFF["fused_r1"] == 19.136555989583336
    assert rows["aff_cf"]["pooled"]["r1"]["point"] == AFF["cf_r1"]
    bar = rec["bar_margin"]["AFF"]["pooled"]
    assert bar["comparator"] == AFF["comparator"] == "B_prime"
    assert ci3(bar["r1"]) == list(AFF["bar"])
    assert {p: v["point"] for p, v in bar["per_pair_r1"].items()} == AFF["per_pair_bar"]
    vs_cf = rec["bar_margin"]["AFF_vs_counterpart"]["pooled"]
    assert ci3(vs_cf["r1"]) == list(AFF["margin"])
    assert ci3(vs_cf["gain"]) == list(AFF["gain_statistic"])
    assert vs_cf["either"]["point"] == AFF["either"]
    assert ci3(rows["r1_fused"]["pooled"]["aff_minus"]["r1"]) == list(AFF["aff_minus_r1_fused"])
    assert ci3(rec["bar_margin"]["AFF_minus_R1_bar_margin_pooled"]) == list(AFF["aff_minus_r1_bar"])
    assert rec["gate_open_shares"]["AFF"]["pooled"]["tau_0"]["open_count"] == AFF["open_tau0_counts"]
    assert rec["gate_open_shares"]["AFF"]["per_seed"]["42"] == rec["gate_open_shares"]["AFF"]["pooled"]
    # the same numbers through the per-scope check rows (the bar comparator B'(A0) is B0's row)
    assert ci3(rec["checks_by_scope"]["pooled"]["P4"]) == list(AFF["bar"])
    assert ci3(rec["checks_by_scope"]["pooled"]["P5"]) == list(AFF["margin"])
    assert ci3(rec["checks_by_scope"]["pooled"]["P6"]) == list(AFF["gain_statistic"])
    assert ci3(rec["checks_by_scope"]["pooled"]["S2"]) == list(AFF["aff_minus_r1_fused"])
    for p, v in AFF["per_pair_bar"].items():
        assert rows["B0"]["per_pair"][p]["aff_minus"]["r1"]["point"] == v
        assert rec["checks_by_scope"]["per_pair"][p]["P4"]["point"] == v


def test_r1_equals_round3_and_round1(real):
    """R1's checks: r3_common.RC_NUMBERS (R3 rule section 5 item 2) and round 1's cand_Rc_Rb_expected_A0.json."""
    rec, r1j = real.rec, _json(ROUND1_REL)
    rows = rec["rows"]
    assert rows["r1_fused"]["pooled"]["r1"]["point"] == RC["fused_r1"] == r1j["r1_means"]["fused"]
    assert rows["r1_cf"]["pooled"]["r1"]["point"] == RC["cf_r1"] == r1j["r1_means"]["counterpart"]
    bar = rec["bar_margin"]["R1"]["pooled"]
    assert bar["comparator"] == RC["comparator"] == r1j["bar"]["comparator"] == "counterpart"
    assert ci3(bar["r1"]) == list(RC["bar"])
    assert bar["comparator_mean_r1"] == r1j["bar"]["comparator_mean_r1"]
    for p in R.PAIR_NAMES:
        assert pc2(bar["per_pair_r1"][p]) == pc2(r1j["bar"]["per_pair_r1"][p]), p
        assert pc2(bar["per_pair_gain"][p]) == pc2(r1j["bar"]["per_pair_gain"][p]), p
    chk = rec["r1_checks"]["pooled"]["checks"]
    assert ci3(chk["r1_vs_counterpart"]) == list(RC["bar"]) == ci3(r1j["margin"]["r1"])
    assert ci3(chk["gain_statistic"]) == list(RC["gain_statistic"]) == ci3(r1j["gain_statistic"])
    assert ci3(chk["r1_vs_B"]) == ci3(r1j["fused_vs_B"]["r1"])
    assert ci3(chk["r1_vs_Bprime"]) == ci3(r1j["fused_vs_Bprime"]["r1"])
    assert rec["r1_checks"]["per_seed"]["42"]["checks"] == chk
    assert rec["r1_checks"]["pooled"]["bar"] == bar
    # the R1 counterpart's either change, from the rows (round 1's margin.either)
    r1_minus_cf_either = np.asarray(real.d.pa["r1_fused"]["r1"]) + np.asarray(real.d.pa["r1_fused"]["other"]) \
        - np.asarray(real.d.pa["r1_cf"]["r1"]) - np.asarray(real.d.pa["r1_cf"]["other"])
    assert ci3(R.C.point_ci(r1_minus_cf_either, real.d.cl)) == ci3(r1j["margin"]["either"])
    # D6: on seed 42 tau_0 is the smallest of R1's margins, so R1's tau_0 gate is open everywhere
    assert rec["gate_open_shares"]["R1"]["pooled"]["tau_0"]["open_count"] == {"a": N, "b": N}


def test_pick_accuracy_equals_round1(real):
    """R3 rule D14 (R1 on seed 42: 51.261393229166664) and round 1's record of it."""
    r1j = _json(ROUND1_REL)
    pa = real.rec["pick_accuracy"]["pooled"]
    assert pa["pick_accuracy"]["correct_share"]["point"] == 51.261393229166664
    assert pc2(pa["pick_accuracy"]["correct_share"]) == pc2(r1j["pick_accuracy"]["correct_share"])
    for k in ("both_correct_share", "per_pair_condition", "chance"):
        assert pa["pick_accuracy"][k] == r1j["pick_accuracy"][k], k
    assert pa["pick_share"] == r1j["pick_share"]
    assert real.rec["pick_accuracy"]["told_mapping"] == {"emotion": "affect", "style": "image", "genre": "image"}


def test_redundancy_equals_round3(real):
    """R3 rule D7's six seed-42 values (r3_common.REDUNDANCY_42), on the frozen B."""
    red = real.rec["redundancy_D7"]["42"]
    assert red["redundancy"] == R3.REDUNDANCY_42
    assert red["affect_least_redundant_both_directions"] is True


def test_comparators_equal_round4(real):
    """B, B0, B1 mean R@1 and AFF - B1 (rule section 6 items 3 and 4; round 4's dev_seed42.json beside_aff)."""
    bes = _json(DEV42_REL)["beside_aff"]
    rows = real.rec["rows"]
    assert rows["B"]["pooled"]["r1"]["point"] == bes["B_mean_r1"]
    assert rows["B0"]["pooled"]["r1"]["point"] == bes["Bprime_A0_mean_r1"]
    assert rows["B1"]["pooled"]["r1"]["point"] == bes["Bprime_A1_mean_r1"]
    assert pc2(real.rec["aff_minus_b1"]["pooled"]) == pc2(bes["AFF_minus_Bprime_A1"])
    assert ci3(real.rec["checks_by_scope"]["pooled"]["S1"]) == ci3(bes["AFF_minus_Bprime_A1"])


def test_ab_scorers_equal_baselines_seed42(real):
    """Cosine, RCA and the nine PM scorers: R@1, gain and swap with their intervals, overall and per pair, equal
    run_baselines.py's own summaries of seed 42 (baselines_seed42.json, summarize over anchor paintings)."""
    ab = _json(AB42_REL)["scorers"]
    rows = real.rec["rows"]
    for s in ("cosine", "rca") + B.PM_NAMES:
        for m in D.ROW_METRICS:
            assert pc2(rows[s]["pooled"][m]) == pc2(ab[s]["overall"][m]), (s, m)
            for p in R.PAIR_NAMES:
                assert pc2(rows[s]["per_pair"][p][m]) == pc2(ab[s]["per_pair"][p][m]), (s, m, p)


def test_rows_two_way_and_reuse_shapes(real):
    rec = real.rec
    assert rec["scorers"] == list(D.SCORERS) and rec["external"] == {}
    for name in D.SCORERS:
        assert rec["rows"][name]["pooled"] == rec["rows"][name]["per_seed"]["42"], name
    tw = rec["two_way_bootstrap"]
    assert tw["n_anchor_clusters"] == 4602 and tw["n_episodes"] == N
    for c, q in tw["quantities"].items():
        for side in ("", "_targets"):
            lo, hi = q[f"ci95_two_way{side}"]
            assert lo <= q["point"] <= hi and q[f"half_width_ratio{side}"] > 0.8, (c, side)
    ir = rec["item_reuse"]["pooled"]
    assert ir["anchors"]["slots"] == N and ir["candidates"]["slots"] == 13 * N and ir["members"]["slots"] == 30 * N
    assert ir == rec["item_reuse"]["per_seed"]["42"]
    assert 0 < ir["candidates"]["painting_reuse_pct"] < 100 and ir["candidates"]["distinct_paintings"] <= 6451
