"""Final review of round 5: mutation tests of the decision-critical guards, on scratch copies only.

Each mutation copies this round's modules, runners, tests, rule, run log, cache/ and results/ into
<scratch>/fr5/mut/<id>/20261123_idea3_goemotions/, patches only r5_common's ROOT to /project/CoSiR, applies one textual
mutation (the original text must occur exactly once), runs the relevant list-A test files with pytest -x, records
killed (a test fails) or survived (all pass), and deletes the copy. The real files are never touched.
Usage: fr5_mutate.py [ID ...]  (default: all; "BASE" runs the unmutated copy). Writes out/mutations/<id>.log and
out/mutations.json (merged).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SRC = Path("/project/CoSiR/src/test/20261123_idea3_goemotions")
SCR = Path("/tmp/claude-0/-project-CoSiR/90512fb4-cc01-42fc-b721-23f4fb0c79c9/scratchpad/fr5/mut")
OUT = SRC / "final_review/out"
LOGS = OUT / "mutations"
LOGS.mkdir(parents=True, exist_ok=True)
PY = "/root/miniconda3/envs/CoSiR/bin/python"
ALL = ["test_r5_common.py", "test_r5_goemo.py", "test_r5_placement.py", "test_r5_bundle.py", "test_r5_fusion.py",
       "test_r5_stats.py", "test_r5_diag.py", "test_r5_runner.py"]
GUARD = ["test_r5_common.py", "test_r5_bundle.py", "test_r5_fusion.py", "test_r5_diag.py", "test_r5_runner.py"]
STATS = ["test_r5_stats.py", "test_r5_runner.py"]
FUS = ["test_r5_fusion.py", "test_r5_runner.py"]
BUN = ["test_r5_bundle.py", "test_r5_runner.py"]
RUN = ["test_r5_runner.py"]

M = {
    # D11 guard
    "G1": ("r5_guard.py", '        if not _STATE["released"]:', "        if False:", GUARD,
           "require() never refuses a ge placement before release"),
    "G2": ("r5_guard.py", "    if not ok:\n        raise GuardError(\"release:", "    if False:\n        raise GuardError(\"release:",
           GUARD, "release() accepts any regression record with this rule's SHA"),
    "G3": ("r5_guard.py", "        if fp in _GE_FPS:\n            raise GuardError(f\"{what}: a clip",
           "        if False:\n            raise GuardError(f\"{what}: a clip", GUARD,
           "a clip placement carrying the GE array is accepted"),
    "G4": ("r5_guard.py", '    return _read_record(path, "carry.json", "require_carry")', "    return {}", GUARD,
           "require_carry() accepts a missing carry.json (diagnostics before the carry)"),
    "G5": ("run_r5_seed42.py", "        if k != len(self.done) + 1:", "        if False:", RUN,
           "items may be marked out of order"),
    "G6": ("run_r5_seed42.py", "ITEMS = (item1, item2, item3, item4)", "ITEMS = (item1, item2, item4, item3)", RUN,
           "items 3 and 4 swapped"),
    # D10
    "D1": ("r5_stats.py", 'c1 = bool(bar["point"] >= C.BAR_TARGET)', 'c1 = bool(bar["point"] > C.BAR_TARGET)', STATS,
           "clause 1 strict"),
    "D2": ("r5_stats.py", 'c2 = bool(bar["ci95"][0] > 0)', 'c2 = bool(bar["ci95"][0] >= 0)', STATS,
           "clause 2 non-strict"),
    "D3": ("r5_stats.py", 'c3 = bool(gain["ci95"][0] > 0)', 'c3 = bool(bar["ci95"][0] > 0)', STATS,
           "clause 3 reads the bar margin's lower bound"),
    "D4": ("r5_stats.py", 'c1 = bool(bar["point"] >= C.BAR_TARGET)', 'c1 = bool(bar["point"] >= 0.4)', STATS,
           "bar threshold 0.4"),
    "D5": ("r5_stats.py", "BOUNDARY_EPS = 1e-12", "BOUNDARY_EPS = 0.0", STATS, "boundary band 0"),
    # Delta_k
    "K1": ("r5_stats.py", "    return int((a - b).sum())", "    return int((b - a).sum())", STATS,
           "Delta_k sign reversed"),
    "K2": ("r5_stats.py", 'a = F.as_int4(fam["fused"]["r1"]', 'a = F.as_int4(fam["cf"]["r1"]', STATS,
           "Delta_k from the counterpart"),
    "K3": ("r5_stats.py", "        if di == 0:", "        if False:", STATS, "Delta_k = 0 not flagged"),
    # carry band
    "C1": ("r5_stats.py", 'tied = [n for n in E if M - records[n]["delta_int"] <= TIE_BAND]',
           'tied = [n for n in E if M - records[n]["delta_int"] < TIE_BAND]', STATS, "tie band exclusive"),
    "C2": ("r5_stats.py", 'records[n]["delta_int"] > 0]', 'records[n]["delta_int"] >= 0]', STATS,
           "Delta_k = 0 admitted to E"),
    "C3": ("r5_stats.py", '"carried": tied[0]', '"carried": tied[-1]', STATS, "last tied carried"),
    "C4": ("r5_stats.py", 'E = [n for n in order if k10[n]["clauses"]["clears"] and records[n]["delta_int"] > 0]',
           'E = [n for n in order if records[n]["delta_int"] > 0]', STATS, "E ignores D10"),
    "C5": ("r5_common.py", "TIE_BAND_UNITS = 24 ", "TIE_BAND_UNITS = 25 ", STATS + ["test_r5_common.py"], "band 25"),
    "C6": ("r5_stats.py", '== TIE_BAND]', '== TIE_BAND + 1]', STATS, "gap 24 not flagged"),
    # G-T / G-TF wiring
    "W1": ("r5_fusion.py", 'READER_INPUTS = {"G-T": ("bundle", "ext"),', 'READER_INPUTS = {"G-T": ("ext", "ext"),', FUS,
           "G-T's reader reads F_G"),
    "W2": ("r5_fusion.py", 'READER_INPUTS = {"G-T": ("bundle", "ext"),', 'READER_INPUTS = {"G-T": ("bundle", "bundle"),',
           FUS, "G-T's term on the bundle's stack (CLIP)"),
    "W3": ("r5_fusion.py", '"G-TF": ("ext", "ext")}', '"G-TF": ("bundle", "ext")}', FUS, "G-TF's reader reads F"),
    "W4": ("r5_fusion.py", "        return tau_prime(m, bundle)", "        return TAUS", FUS,
           "G-TF uses AFF's tau on seed 42"),
    "W5": ("r5_fusion.py", "    fam = RF3.run_family(bundle, cand[\"T\"], gates)",
           "    fam = RF3.run_family(bundle, cand[\"T\"], expected_d6(\"G-T\", bundle, ext)[\"gates\"])", FUS,
           "the family (fused and counterpart) runs from AFF's gates"),
    "W6": ("r5_fusion.py", "    fam = RF3.run_family(bundle, cand[\"T\"], gates)",
           "    fam = RF3.run_family(bundle, RF3.reader(SimpleNamespace(F=bundle.F, stack=bundle.stack), "
           "readers=_readers(bundle))[\"T\"], gates)", FUS, "the family runs from AFF's term (CLIP stack)"),
    "W7": ("r5_fusion.py", "    if bad:\n        raise AssertionError(f\"rule D7", "    if False:\n        raise AssertionError(f\"rule D7",
           FUS, "D7's gate check never fires"),
    # D5 positive check and no-assign
    "P1": ("r5_bundle.py", '"affect_slice_differs_from_bundle_on_an_episode": any(',
           '"affect_slice_differs_from_bundle_on_an_episode": True or any(', BUN, "differs-check vacuous"),
    "P2": ("r5_bundle.py", '"affect_slice_equals_independent_einsum": all(',
           '"affect_slice_equals_independent_einsum": True or all(', BUN, "einsum check vacuous"),
    "P3": ("r5_bundle.py", '"F_columns_0_to_5_differ_from_bundle_on_an_episode": any(',
           '"F_columns_0_to_5_differ_from_bundle_on_an_episode": True or any(', BUN, "F-columns check vacuous"),
    "P4": ("run_r5_seed42.py", '    if not (pc.get("all_pass") is True and all(v is True for v in pc.values())):',
           "    if False:", RUN, "the runner ignores the positive check"),
    "P5": ("r5_bundle.py",
           '    return {"affect": {"img": post["affect"]["img"], "txt": Q}, "image": post["image"], "caption": post["caption"]}',
           '    post["affect"]["txt"] = Q\n    return {"affect": {"img": post["affect"]["img"], "txt": Q}, "image": post["image"], "caption": post["caption"]}',
           BUN, "post_q assigns Q into bundle.post (D5 no-assign)"),
    # item-1 / item-4 pins
    "N1": ("run_r5_seed42.py", '    return {"missing": sorted(want - got), "extra": sorted(got - want)}',
           '    return {"missing": [], "extra": []}', RUN, "pinned names never differ"),
    "N2": ("run_r5_seed42.py", "            if not dry and k in PINNED_NAMES:", "            if False:", RUN,
           "the real run adds no pin row"),
    # agreement-line gate
    "A1": ("run_r5_seed42.py", 'AGREEMENT_FORBIDDEN = ("not", "fail", "disagree", "pending")',
           'AGREEMENT_FORBIDDEN = ("not", "fail", "disagree")', RUN, "a 'pending' line opens the gate"),
    # deferred-minor probes (expected survivors per the ledger)
    "T1": ("r5_stats.py", 'for k, p in (("fused", "fpick"), ("cf", "cpick"))},\n           "Bprime_G_minus',
           'for k, p in (("fused", "cpick"), ("cf", "fpick"))},\n           "Bprime_G_minus', STATS,
           "dev_record's cell_text swaps fused and counterpart picks"),
    "T2": ("r5_diag.py", "    return {k: {x: {m: diff_cand[k][x][m] - diff_aff[k][x][m] for m in METRICS3}",
           "    return {k: {x: {m: diff_aff[k][x][m] - diff_cand[k][x][m] for m in METRICS3}",
           ["test_r5_diag.py", "test_r5_runner.py"], "diagnostic (d) minus_aff sign reversed"),
    "G7": ("r5_guard.py", '    if fp in _GE_FPS:\n        raise GuardError("the bundle\'s affect caption posterior is the GE placement',
           '    if False:\n        raise GuardError("the bundle\'s affect caption posterior is the GE placement', GUARD,
           "clip_from_bundle accepts the GE array (mint-time fingerprint check removed)"),
    "G8": ("r5_guard.py", "    if not any(_cached_txt_equals(hit, Q) for hit in R5.RB3._HEADS.values()):",
           "    if False:", GUARD, "clip_from_bundle skips the by-value check against round 3's cached heads"),
    "T3": ("r5_diag.py", 'out["minus_aff"] = summarize(minus(diff, aff["diff"]), cl, pair_index)',
           'out["minus_aff"] = summarize(minus(aff["diff"], diff), cl, pair_index)',
           ["test_r5_diag.py", "test_r5_runner.py"], "sharper_term passes AFF and the candidate in swapped order"),
    "A2": ("run_r5_seed42.py", "        if m is None or m.group(1) != carry_sha:", "        if m is None:", RUN,
           "any carry SHA opens the gate"),
}


def prepare(mid):
    d = SCR / mid / "20261123_idea3_goemotions"
    if d.parent.exists():
        shutil.rmtree(d.parent)
    d.mkdir(parents=True)
    for p in SRC.iterdir():
        if p.is_file() and (p.suffix in (".py", ".md") or p.name == ".gitignore"):
            shutil.copy2(p, d / p.name)
    shutil.copytree(SRC / "cache", d / "cache")
    shutil.copytree(SRC / "results", d / "results")
    rc = d / "r5_common.py"
    t = rc.read_text()
    assert t.count("ROOT = HERE.parents[2]") == 1
    rc.write_text(t.replace("ROOT = HERE.parents[2]", 'ROOT = Path("/project/CoSiR")'))
    return d


def run(mid):
    d = prepare(mid)
    if mid == "BASE":
        tests, desc = ALL, "unmutated copy"
    else:
        f, old, new, tests, desc = M[mid]
        p = d / f
        t = p.read_text()
        n = t.count(old)
        if n != 1:
            shutil.rmtree(d.parent)
            return {"id": mid, "error": f"original text found {n} times in {f}", "desc": desc}
        p.write_text(t.replace(old, new))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    cp = subprocess.run([PY, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider", *tests], cwd=d, env=env,
                        capture_output=True, text=True)
    (LOGS / f"{mid}.log").write_text(cp.stdout[-20000:] + "\n--- stderr ---\n" + cp.stderr[-5000:])
    first = ""
    for ln in cp.stdout.splitlines():
        if ln.startswith("FAILED") or ln.startswith("ERROR"):
            first = ln[:220]
            break
    shutil.rmtree(d.parent)
    status = "baseline pass" if (mid == "BASE" and cp.returncode == 0) else (
        "survived" if cp.returncode == 0 else "killed")
    return {"id": mid, "desc": desc, "status": status, "returncode": cp.returncode, "first_failure": first,
            "tests": tests, "seconds": round(time.time() - t0), "summary": cp.stdout.strip().splitlines()[-1:]}


if __name__ == "__main__":
    ids = sys.argv[1:] or list(M)
    if ids == ["MERGE"]:
        allres = {p.stem: json.loads(p.read_text()) for p in sorted(LOGS.glob("*.json"))}
        (OUT / "mutations.json").write_text(json.dumps(allres, indent=1))
        sys.exit(0)
    for mid in ids:
        r = run(mid)
        (LOGS / f"{mid}.json").write_text(json.dumps(r, indent=1))
        print(json.dumps({k: r.get(k) for k in ("id", "status", "seconds", "first_failure", "error")}), flush=True)
