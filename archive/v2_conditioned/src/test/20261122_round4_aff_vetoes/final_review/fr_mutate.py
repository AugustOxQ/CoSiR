"""Final review: mutation checks of the round-4 guards on scratch copies (never the repo's files).

Each mutation copies the round-4 modules, the three synthetic test files and DECISION_RULE.md into a fresh folder
final_review/mut/<id>/ (r4_common's ROOT patched to the repo root, nothing else changed), applies one exact string
replacement (asserted to occur exactly once), runs pytest on the synthetic tests, records the outcome and deletes the
folder. 'killed' = at least one test failed or errored. Usage: python fr_mutate.py [ids...] (default all; 'base' runs
the unmutated copy). Writes out/mutations/<id>.log and out/mutations.json (merged).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SRC = Path("/project/CoSiR/src/test/20261122_round4_aff_vetoes")
FR = SRC / "final_review"
MUT = FR / "mut"
LOGS = FR / "out/mutations"
FILES = ("r4_common.py", "r4_fusion.py", "r4_stats.py", "r4_bundle.py", "run_r4_seed42.py", "test_r4_fusion.py",
         "test_r4_runners.py", "test_r4_common.py", "DECISION_RULE.md")
TESTS = ("test_r4_common.py", "test_r4_fusion.py", "test_r4_runners.py")
PY = "/root/miniconda3/envs/CoSiR/bin/python"

M = {
    # ---- the section 5 order guard (run_r4_seed42.Guard)
    "G1_require_noop": ("run_r4_seed42.py", "        if not self.released:\n            raise GuardError(f\"{what}:",
                        "        if False:\n            raise GuardError(f\"{what}:", "Guard.require never refuses"),
    "G2_items_out_of_order": ("run_r4_seed42.py", "        if k != len(self.passed) + 1:", "        if False:",
                              "Guard.item_passed accepts any order"),
    "G3_release_without_record": ("run_r4_seed42.py", "        self._record_passed(reg_path)\n        self.released = True",
                                  "        self.released = True", "release() does not need the written record"),
    "G4_gates_before_items_1_4": ("run_r4_seed42.py", "        if self.passed[:4] != [1, 2, 3, 4]:", "        if False:",
                                  "item 5 gates allowed before items 1 to 4"),
    "G5_item_order_swapped": ("run_r4_seed42.py", "ITEMS = (item1, item2, item3, item4, item5)",
                              "ITEMS = (item1, item2, item3, item5, item4)", "items 4 and 5 swapped in the runner"),
    # ---- gate algebra (r4_fusion) and item 5
    "A1_v2_factor_inverted": ("r4_fusion.py", "(np.asarray(pick_a1[c]) == AFFECT)", "(np.asarray(pick_a1[c]) != AFFECT)",
                              "V2's factor 1[pi_A1 != affect]"),
    "A2_v24_drops_keep": ("r4_fusion.py", "    if name in (\"V4\", \"V24\"):\n        g = apply_keep(g, keep)",
                          "    if name in (\"V4\",):\n        g = apply_keep(g, keep)", "V24 without the abstention factor"),
    "A3_v24_drops_pick": ("r4_fusion.py", "    if name in (\"V2\", \"V24\"):\n        g = apply_affect_pick(g, pick_a1)",
                          "    if name in (\"V2\",):\n        g = apply_affect_pick(g, pick_a1)", "V24 without the A1-pick factor"),
    "A4_abstain_le": ("r4_fusion.py", "< float(v75)).astype", "<= float(v75)).astype", "a_v = 1[v <= v75]"),
    "A5_no_subset_assert": ("r4_fusion.py", "            if np.any(g[t][c] > np.asarray(g_aff[t][c])):",
                            "            if False:", "no 'closed where AFF is closed' assertion"),
    "A6_item5_v2_check_not_independent": ("run_r4_seed42.py", "a * (np.asarray(st[\"pick_stored\"][c]) == 0))",
                                          "a * (np.asarray(pick[c]) == 0))", "item 5 checks V2 against its own picks"),
    "A7_item5_v4_check_not_independent": ("run_r4_seed42.py", "G[\"V4\"][t][c], a * below)",
                                          "G[\"V4\"][t][c], a * keep)", "item 5 checks V4 against its own keep"),
    "A8_develop_on_R1_gates": ("run_r4_seed42.py", "    b, T, g_aff = st[\"b\"], st[\"rd\"][\"T\"], st[\"g_aff\"]",
                               "    b, T, g_aff = st[\"b\"], st[\"rd\"][\"T\"], st[\"g_r1\"]",
                               "development step builds the candidates on R1's gates (item 5 verified AFF's)"),
    "A9_develop_keep_ones": ("run_r4_seed42.py", "keep=st[\"keep\"] if USES_V[k] else None)",
                             "keep=np.ones_like(st[\"keep\"]) if USES_V[k] else None)",
                             "development step passes a_v = 1 (item 5 verified the real a_v)"),
    "A10_develop_other_v75": ("run_r4_seed42.py", "keep=st[\"keep\"] if USES_V[k] else None)",
                              "keep=R4F.abstain(b.v, 0.9 * R4.V75) if USES_V[k] else None)",
                              "development step builds a_v with another threshold than item 5 verified"),
    "A11_develop_A0_picks": ("run_r4_seed42.py", "pick_a1=st[\"ra1\"][\"pick\"] if R4.READS_CSD[k] else None,",
                             "pick_a1=st[\"rd\"][\"pick\"] if R4.READS_CSD[k] else None,",
                             "development step passes the A0 picks as the A1 picks"),
    # ---- the candidate counterpart
    "C1_counterpart_on_aff_gates": ("r4_fusion.py",
                                    "    return RF3.run_family(bundle, T, gates_candidate(name, g_aff, pick_a1=pick_a1, keep=keep), fused_only=fused_only)",
                                    "    gates_candidate(name, g_aff, pick_a1=pick_a1, keep=keep)\n    return RF3.run_family(bundle, T, g_aff, fused_only=fused_only)",
                                    "run_candidate runs AFF's gates"),
    "C2_develop_aff_family_as_candidate": ("run_r4_seed42.py",
                                           "recs[k] = guard.call(R4S.dev_record, k, fam[k], st[\"fam_aff\"],",
                                           "recs[k] = guard.call(R4S.dev_record, k, st[\"fam_aff\"], st[\"fam_aff\"],",
                                           "dev record of each candidate built from AFF's family"),
    # ---- integer carry and its tie band (r4_stats.carry, r4_common)
    "K1_band_strict": ("r4_stats.py", "if M - records[n][\"delta_int\"] <= TIE_BAND]", "if M - records[n][\"delta_int\"] < TIE_BAND]",
                       "tie band exclusive"),
    "K2_delta_ge0": ("r4_stats.py", "records[n][\"d10\"][\"clears\"] and records[n][\"delta_int\"] > 0]",
                     "records[n][\"d10\"][\"clears\"] and records[n][\"delta_int\"] >= 0]", "E admits Delta_k = 0"),
    "K3_band_25": ("r4_common.py", "TIE_BAND_UNITS = 24", "TIE_BAND_UNITS = 25", "tie band 25"),
    "K4_last_tied": ("r4_stats.py", "\"carried\": tied[0], \"boundaries\": bnd}", "\"carried\": tied[-1], \"boundaries\": bnd}",
                     "carry takes the last tied member"),
    "K5_E_ignores_d10": ("r4_stats.py", "E = [n for n in order if records[n][\"d10\"][\"clears\"] and records[n][\"delta_int\"] > 0]",
                         "E = [n for n in order if records[n][\"delta_int\"] > 0]", "E ignores the development bar"),
    "K6_delta_from_float_mean": ("r4_stats.py",
                                 "        di = int((F.as_int4(fu[\"r1\"], \"candidate r1\").astype(np.int64)\n                  - F.as_int4(aff_fam[\"fused\"][\"r1\"], \"AFF r1\").astype(np.int64)).sum())",
                                 "        di = int(np.floor(4 * n * (np.mean(fu[\"r1\"]) - np.mean(aff_fam[\"fused\"][\"r1\"]))))",
                                 "Delta_k from float means, floored"),
    "K7_delta_vs_cf": ("r4_stats.py", "- F.as_int4(aff_fam[\"fused\"][\"r1\"], \"AFF r1\").astype(np.int64)).sum())",
                       "- F.as_int4(fam[\"cf\"][\"r1\"], \"AFF r1\").astype(np.int64)).sum())",
                       "Delta_k against the candidate's counterpart instead of AFF"),
    # ---- the bar comparator and D10 (r4_stats)
    "B1_v2_order_A0_first": ("r4_stats.py", "        return [(\"Bprime_A1\", pBp1), (\"Bprime_A0\", pBp0), (\"counterpart\", cf), (\"B\", pB)]",
                             "        return [(\"Bprime_A0\", pBp0), (\"Bprime_A1\", pBp1), (\"counterpart\", cf), (\"B\", pB)]",
                             "V2/V24 tie order B'(A0) before B'(A1)"),
    "B2_ties_to_last": ("r4_stats.py", "        if means[i] > means[best]:", "        if means[i] >= means[best]:",
                        "bar comparator ties to the last"),
    "B3_v2_without_A1": ("r4_stats.py", "    if name in (\"V4\", \"AFF\", \"R1\", \"IMGABST\"):",
                         "    if name in (\"V4\", \"AFF\", \"R1\", \"IMGABST\", \"V2\", \"V24\"):",
                         "V2/V24 compared without B'(A1)"),
    "D1_c1_strict": ("r4_stats.py", "c1, c2, c3 = bool(bar[\"point\"] >= C.BAR_TARGET)", "c1, c2, c3 = bool(bar[\"point\"] > C.BAR_TARGET)",
                     "D10 clause 1 strict"),
    "D2_c2_nonstrict": ("r4_stats.py", "bool(bar[\"ci95\"][0] > 0), bool(gain[\"ci95\"][0] > 0)",
                        "bool(bar[\"ci95\"][0] >= 0), bool(gain[\"ci95\"][0] > 0)", "D10 clause 2 >= 0"),
    "D3_c3_on_bar": ("r4_stats.py", "bool(bar[\"ci95\"][0] > 0), bool(gain[\"ci95\"][0] > 0)",
                     "bool(bar[\"ci95\"][0] > 0), bool(bar[\"ci95\"][0] > 0)", "D10 clause 3 reads the bar margin"),
    "D4_c1_threshold_04": ("r4_stats.py", "(bar[\"point\"] >= C.BAR_TARGET)", "(bar[\"point\"] >= 0.4)", "bar 0.4"),
    # ---- the boundary stop
    "S1_eps_zero": ("r4_stats.py", "BOUNDARY_EPS = 1e-12", "BOUNDARY_EPS = 0.0", "no 1e-12 band"),
    "S2_no_delta0_flag": ("r4_stats.py", "        if di == 0:", "        if False:", "Delta_k = 0 not flagged"),
    "S3_no_gap24_flag": ("r4_stats.py", "if M - records[n][\"delta_int\"] == TIE_BAND]", "if M - records[n][\"delta_int\"] == TIE_BAND + 1]",
                         "tie gap 24 not flagged"),
    "S4_no_stop": ("run_r4_seed42.py", "    if bnd and not dry:", "    if False:", "the runner does not stop at a boundary"),
    "S5_resume_no_sha": ("run_r4_seed42.py", "    if R4.sha_file(P[\"bnd\"]) != sha:", "    if False:",
                         "resume accepts any SHA-256"),
    "S6_resume_no_change_check": ("run_r4_seed42.py", "        if R4.sha_file(P[k]) != bnd[key]:", "        if False:",
                                  "resume accepts changed files"),
    # ---- the seed guard
    "Q1_seed_guard_not_set": ("r4_common.py", "R3.TEST_SEEDS = (52, 53, 54)   #", "R3.TEST_SEEDS = R3.TEST_SEEDS   #",
                              "round 3's seed guard not redirected"),
}


def run_one(mid):
    d = MUT / mid
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in FILES:
        shutil.copy2(SRC / f, d / f)
    p = d / "r4_common.py"
    s = p.read_text()
    assert s.count("ROOT = HERE.parents[2]") == 1
    p.write_text(s.replace("ROOT = HERE.parents[2]", "ROOT = Path(\"/project/CoSiR\")"))
    desc = "unmutated copy"
    if mid != "base":
        f, old, new, desc = M[mid]
        q = d / f
        s = q.read_text()
        n = s.count(old)
        if n != 1:
            shutil.rmtree(d)
            return {"id": mid, "error": f"pattern found {n} times in {f}"}
        q.write_text(s.replace(old, new))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x" if mid != "base" else "-q", "-p", "no:cacheprovider", *TESTS],
                       cwd=d, env=env, capture_output=True, text=True, timeout=1800)
    out = r.stdout + r.stderr
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"{mid}.log").write_text(out)
    tail = [ln for ln in out.splitlines() if ln.strip()][-1:] if out else [""]
    failed = [ln.split(" ")[1] if ln.startswith("FAILED") else ln for ln in out.splitlines()
              if ln.startswith("FAILED") or ln.startswith("ERROR")]
    shutil.rmtree(d)
    return {"id": mid, "desc": desc, "returncode": r.returncode, "killed": r.returncode != 0, "summary": tail[0],
            "first_failures": failed[:3], "seconds": int(time.time() - t0)}


def main():
    ids = sys.argv[1:] or ["base", *M]
    path = FR / "out/mutations.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        for r in ex.map(run_one, ids):
            res[r["id"]] = r
            print(json.dumps(r), flush=True)
            path.write_text(json.dumps(res, indent=1))
    if MUT.exists() and not any(MUT.iterdir()):
        MUT.rmdir()


if __name__ == "__main__":
    main()
