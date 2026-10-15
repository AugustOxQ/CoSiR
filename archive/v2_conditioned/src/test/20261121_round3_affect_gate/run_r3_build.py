"""Round 3: build the test seeds and check them (DECISION_RULE.md §6.2, D15; smoke seeds for §10).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261121_round3_affect_gate/run_r3_build.py --seeds 49 50 51
    ... --seeds 9002 9003 --smoke        smoke seeds (64 episodes per pair), records in results/smoke/

Per seed, in the order given (49, 50, 51 non-smoke; a subset of 9001, 9002, 9003 with --smoke):
  - non-smoke: codes_provenance.json must have its D15 SHA-256 before the build;
  - src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed <s> [--smoke] (never --overwrite) as a
    subprocess, stdout and stderr to results/build_seed{s}.log (smoke: results/smoke/), which this script never prints
    or reads (it holds run_baselines' scorer tables); the exit status must be 0;
  - non-smoke: codes_provenance.json must still have its D15 SHA-256 after the build;
  - crash rule (§6.2): a build that exits non-zero before baselines_seed{s}.json exists has the SHA-256 of any
    episodes_seed{s}.npz it left (and that file's per-pair hashes, if readable) recorded, its partial outputs deleted,
    and runs once more; the episode SHA-256s of both runs must be equal when both exist; a second failure stops;
  - the per-pair episode SHA-256s recomputed from episodes_seed{s}.npz equal those of baselines_seed{s}.json; the 3 of
    the seed differ from each other, from those of the other new seeds (this run's and earlier records) and from every
    per-pair SHA-256 of baselines_seed{42,43,45,47,48}.json (their D15 SHA-256s asserted first). On a match: stop;
  - results/build_seed{s}.json: the SHA-256s of episodes_seed{s}.npz, per_anchor_seed{s}.npz, baselines_seed{s}.json,
    the per-pair episode hashes, codes_provenance before and after, every attempt's exit status, the time. Every
    later step asserts these SHA-256s (verify_build_record).
A seed whose baselines_seed{s}.json exists is never rebuilt (refused); in smoke mode a complete existing smoke seed
(e.g. 9001, built for Task 1) is recorded as it is, not rebuilt. Prints only file names, hashes and PASS/FAIL.
Run seeds one after another (one invocation): run_baselines.py rewrites codes_provenance.json during each build.
After the hash check passes, the seed ledger (docs/superpowers/episode_seed_ledger.md) gets its rows (main session).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r3_common as R3  # noqa: E402

RUN_BASELINES = "20261030_aspect_baselines/run_baselines.py"
CODES_PROV = "20261030_aspect_baselines/results/codes_provenance.json"
CODES_SHA = R3.INPUTS[CODES_PROV]
EARLIER = {s: f"20261030_aspect_baselines/results/baselines_seed{s}.json" for s in R3.EARLIER_SEEDS}
PAIR_ORDER = ("emotion__style", "emotion__genre", "style__genre")
EP_FIELDS = ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")   # run_baselines
N_PER_PAIR = {False: 4096, True: 64}
KINDS = ("episodes", "per_anchor", "baselines")


def say(msg):
    print(f"[run_r3_build] {msg}", flush=True)


# ---------------------------------------------------------------- seeds and paths

def check_seeds(seeds, smoke, full=False) -> tuple:
    """Non-smoke: a subset of the test seeds (49, 50, 51); smoke: of the smoke seeds (9001, 9002, 9003); unique, in
    the rule's order. full=True (the test phases): exactly the three seeds."""
    allowed = tuple(R3.SMOKE_SEEDS if smoke else R3.TEST_SEEDS)
    try:
        seeds = tuple(int(s) for s in seeds)
    except (TypeError, ValueError):
        raise SystemExit(f"seeds must be integers: {seeds}")
    mode = "smoke" if smoke else "non-smoke"
    if not seeds:
        raise SystemExit("no seed given")
    if len(set(seeds)) != len(seeds):
        raise SystemExit(f"a seed is repeated: {seeds}")
    bad = [s for s in seeds if s not in allowed]
    if bad:
        raise SystemExit(f"{mode} seeds must be among {allowed}; refused seed(s) {bad}")
    if list(seeds) != sorted(seeds):
        raise SystemExit(f"seeds must be in the rule's order {allowed}: got {seeds}")
    if full and seeds != allowed:
        raise SystemExit(f"this phase pools exactly the {mode} seeds {allowed}; got seed(s) {seeds}")
    return seeds


def seed_files(seed, smoke, ab_res=None) -> dict:
    d = Path(ab_res or (R3.AB / "results")) / ("smoke" if smoke else "")
    return {"episodes": d / f"episodes_seed{seed}.npz", "per_anchor": d / f"per_anchor_seed{seed}.npz",
            "baselines": d / f"baselines_seed{seed}.json"}


def record_path(seed, smoke, rec_dir=None) -> Path:
    return Path(rec_dir or R3.res_dir(smoke)) / f"build_seed{seed}.json"


# ---------------------------------------------------------------- hashes

def episode_pair_hashes(npz_path) -> dict:
    """Per-pair SHA-256 of an episodes_seed{s}.npz, as run_baselines.py and run_gonogo.EvalContext compute it."""
    from src.eval.aspect_episodes import AspectEpisodes, episodes_sha256
    if tuple(f.name for f in fields(AspectEpisodes)[2:]) != EP_FIELDS:
        raise AssertionError("AspectEpisodes fields differ from run_baselines.py's FIELDS")
    with np.load(npz_path) as z:
        if tuple(str(p) for p in z["pair_order"]) != PAIR_ORDER:
            raise AssertionError(f"{Path(npz_path).name}: pair order differs from {PAIR_ORDER}")
        out = {}
        for p in PAIR_ORDER:
            a, b = p.split("__")
            out[p] = episodes_sha256(AspectEpisodes(a, b, *(z[f"{p}__{k}"].astype(np.int64) for k in EP_FIELDS)))
    return out


def recorded_pair_hashes(baselines_path) -> tuple:
    """(per-pair SHA-256s, episodes_seed, n_per_pair) of a baselines_seed{s}.json (nothing else of it is read)."""
    rec = json.loads(Path(baselines_path).read_text())
    h = rec["episodes_sha256"]
    if tuple(h) != PAIR_ORDER or tuple(rec["pair_order"]) != PAIR_ORDER:
        raise AssertionError(f"{Path(baselines_path).name}: pair order differs from {PAIR_ORDER}")
    return {p: str(h[p]) for p in PAIR_ORDER}, int(rec["episodes_seed"]), int(rec["n_per_pair"])


def earlier_pair_hashes() -> dict:
    """{seed: {pair: SHA-256}} of baselines_seed{42,43,45,47,48}.json, each file's D15 SHA-256 asserted first."""
    R3.assert_inputs(list(EARLIER.values()))
    return {s: recorded_pair_hashes(R3.input_path(n))[0] for s, n in EARLIER.items()}


def hash_check(new, others_new, earlier) -> tuple:
    """new: {pair: sha} of one new seed; others_new and earlier: {seed: {pair: sha}}. -> (ok, problems)."""
    problems = []
    vals = list(new.values())
    if len(set(vals)) != len(vals):
        problems.append("two aspect pairs of this seed share an episode SHA-256")
    for label, group in (("new", others_new), ("earlier", earlier)):
        for s, hs in group.items():
            for p, h in hs.items():
                for q, v in new.items():
                    if v == h:
                        problems.append(f"{q} equals {label} seed {s} {p}")
    return not problems, problems


def cross_check(records, earlier=None):
    """The test phases: the per-pair episode hashes of all build records differ from each other and from the earlier
    seeds' (rule §6.2), whether or not the seeds were built in one run. Raises SystemExit on a match."""
    earlier = earlier_pair_hashes() if earlier is None else earlier
    for s, rec in records.items():
        others = {t: r["episode_pair_sha256"] for t, r in records.items() if t != s}
        good, problems = hash_check(rec["episode_pair_sha256"], others, earlier)
        if not good:
            raise SystemExit(f"episode hash check failed for seed {s}: {problems}; stop and report to the user")


def codes_provenance_sha() -> str:
    """Fresh SHA-256 of codes_provenance.json (not r3_common's per-process cache: run_baselines rewrites the file)."""
    return R3.sha_file(R3.input_path(CODES_PROV))


# ---------------------------------------------------------------- one build

def run_baselines(seed, smoke, log_file) -> int:
    cmd = [sys.executable, str(R3.TEST / RUN_BASELINES), "--episodes-seed", str(int(seed))] + (["--smoke"] if smoke else [])
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
           "PYTHONDONTWRITEBYTECODE": "1"}
    with open(log_file, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=str(R3.ROOT), env=env).returncode


def _stop(rec_path, record, smoke, msg):
    record["passed"] = False
    record["stop"] = msg
    R3.write_json_once(rec_path, record, smoke)
    say(f"seed {record['seed']}: FAIL ({msg})")
    raise SystemExit(f"seed {record['seed']}: {msg}; stop and report to the user (rule §6.2)")


def build_seed(seed, smoke, runner=run_baselines, ab_res=None, rec_dir=None, earlier=None, others_new=None,
               n_per_pair=None, codes_sha=codes_provenance_sha) -> dict:
    """Build (or, smoke only, record an existing) seed and check it; writes build_seed{s}.json. -> the record.
    runner, ab_res, rec_dir, earlier, n_per_pair, codes_sha: injected by the unit tests."""
    seed, smoke = int(seed), bool(smoke)
    check_seeds([seed], smoke)
    files = seed_files(seed, smoke, ab_res)
    rec_dir = Path(rec_dir or R3.res_dir(smoke))
    rec_dir.mkdir(parents=True, exist_ok=True)
    rec_path = record_path(seed, smoke, rec_dir)
    R3.refuse_existing([rec_path], smoke)
    n_expected = N_PER_PAIR[smoke] if n_per_pair is None else int(n_per_pair)
    earlier = earlier_pair_hashes() if earlier is None else earlier
    others = dict(others_new or {})                # the other new seeds: this run's and earlier records
    for t in (R3.SMOKE_SEEDS if smoke else R3.TEST_SEEDS):
        p = record_path(t, smoke, rec_dir)
        if t != seed and t not in others and p.exists():
            other = json.loads(p.read_text())
            if other.get("passed") is not True or "episode_pair_sha256" not in other:
                raise SystemExit(f"seed {t}'s build record {p.name} did not pass; the test stopped there (rule §6.2): "
                                 f"report to the user before building seed {seed}")
            others[t] = other["episode_pair_sha256"]
    record = {"seed": seed, "smoke": smoke, "rule_sha256": R3.RULE_SHA, "attempts": [], "built_here": True,
              "codes_provenance_before": None, "codes_provenance_after": None,
              "files": {k: str(p) for k, p in files.items()}}
    present = [k for k in KINDS if files[k].exists()]
    if files["baselines"].exists():
        if not smoke:
            raise SystemExit(f"seed {seed}: {files['baselines'].name} exists; a completed seed is never rebuilt "
                             f"(rule §6.2)")
        if present != list(KINDS):
            raise SystemExit(f"smoke seed {seed}: partial outputs {present}; stop and report")
        record["built_here"] = False
        record["note"] = "smoke seed built earlier with run_baselines.py --smoke; recorded as it is, not rebuilt"
        say(f"smoke seed {seed}: complete outputs exist; recorded, not rebuilt")
    elif present:
        raise SystemExit(f"seed {seed}: partial outputs {present} of an earlier run exist; stop and report to the user "
                         f"(the crash rule repeats a build only within one run)")
    else:
        for attempt in (1, 2):
            log_file = rec_dir / (f"build_seed{seed}.log" if attempt == 1 else f"build_seed{seed}_attempt{attempt}.log")
            R3.refuse_existing([log_file], smoke)
            att = {"attempt": attempt, "log": log_file.name}
            record["attempts"].append(att)
            if not smoke:
                att["codes_provenance_before"] = codes_sha()
                record["codes_provenance_before"] = record["codes_provenance_before"] or att["codes_provenance_before"]
                if att["codes_provenance_before"] != CODES_SHA:
                    _stop(rec_path, record, smoke, "codes_provenance.json differs from its D15 SHA-256 before the build")
            say(f"seed {seed}: build attempt {attempt} -> {log_file.name}")
            t0 = time.time()
            att["exit_status"] = int(runner(seed, smoke, log_file))
            att["seconds"] = round(time.time() - t0, 1)
            if not smoke:
                att["codes_provenance_after"] = codes_sha()
                record["codes_provenance_after"] = att["codes_provenance_after"]
                if att["codes_provenance_after"] != CODES_SHA:
                    _stop(rec_path, record, smoke, "codes_provenance.json changed during the build")
            if att["exit_status"] == 0:
                break
            if files["baselines"].exists():
                _stop(rec_path, record, smoke, f"the build exited {att['exit_status']} after writing "
                                               f"{files['baselines'].name} (not a crash before it; not repeated)")
            if files["episodes"].exists():
                att["episodes_sha256"] = R3.sha_file(files["episodes"])
                try:
                    att["episode_pair_sha256"] = episode_pair_hashes(files["episodes"])
                except Exception as e:  # noqa: BLE001  unreadable partial file: recorded
                    att["episode_pair_sha256_error"] = type(e).__name__
            if attempt == 2:
                _stop(rec_path, record, smoke, "the second build attempt failed too")
            att["deleted_partial_outputs"] = [k for k in KINDS if files[k].exists()]
            for k in att["deleted_partial_outputs"]:
                files[k].unlink()
            say(f"seed {seed}: attempt 1 exited {att['exit_status']} before {files['baselines'].name}; partial outputs "
                f"{att['deleted_partial_outputs']} deleted; repeating once")
        missing = [k for k in KINDS if not files[k].exists()]
        if missing:
            _stop(rec_path, record, smoke, f"the build exited 0 but {missing} are missing")
        first = record["attempts"][0]
        if len(record["attempts"]) == 2 and "episodes_sha256" in first:
            if first["episodes_sha256"] != R3.sha_file(files["episodes"]):
                _stop(rec_path, record, smoke, "the two runs' episode SHA-256s differ (crash rule)")
            record["crash_rule_episodes_equal"] = True

    record["sha256"] = {k: R3.sha_file(files[k]) for k in KINDS}
    recorded, rec_seed, rec_n = recorded_pair_hashes(files["baselines"])
    record["episode_pair_sha256"] = recorded
    checks = {"baselines_record_is_this_seed": rec_seed == seed, "n_per_pair": rec_n == n_expected}
    try:
        checks["episode_pair_hashes_match_baselines_json"] = episode_pair_hashes(files["episodes"]) == recorded
    except Exception as e:  # noqa: BLE001
        checks["episode_pair_hashes_match_baselines_json"] = False
        record["episode_pair_hashes_error"] = type(e).__name__
    good, problems = hash_check(recorded, others, earlier)
    checks["episode_hashes_distinct"] = good
    record["hash_check"] = {"pass": good, "problems": problems, "compared_new_seeds": sorted(others),
                            "compared_earlier_seeds": sorted(earlier)}
    record["checks"] = checks
    for k in KINDS:
        say(f"seed {seed}: {files[k].name} sha256 {record['sha256'][k]}")
    for p, h in recorded.items():
        say(f"seed {seed}: episodes {p} sha256 {h}")
    for k, v in checks.items():
        say(f"seed {seed}: CHECK {k} {'PASS' if v else 'FAIL'}")
    if not all(checks.values()):
        failed = [k for k, v in checks.items() if not v]
        _stop(rec_path, record, smoke, f"build checks failed: {failed}" +
              (f" (episode hash check: {problems})" if not good else ""))
    record["passed"] = True
    record["time_amsterdam"] = R3.now_ams()
    R3.write_json_once(rec_path, record, smoke)
    say(f"seed {seed}: BUILD PASS -> {rec_path.name}")
    return record


def verify_build_record(seed, smoke, ab_res=None, rec_dir=None) -> dict:
    """Every later step (D15): the seed's build record passed under this rule, and the three files still have the
    SHA-256s it recorded."""
    p = record_path(seed, smoke, rec_dir)
    if not p.exists():
        raise SystemExit(f"seed {seed}: no build record {p.name}; run run_r3_build.py first")
    rec = json.loads(p.read_text())
    if rec.get("provenance", {}).get("rule_sha256") != R3.RULE_SHA:
        raise SystemExit(f"{p.name}: written under another rule")
    if rec.get("seed") != int(seed) or rec.get("smoke") is not bool(smoke) or rec.get("passed") is not True:
        raise SystemExit(f"{p.name}: another seed or mode, or the build did not pass")
    for k, f in seed_files(seed, smoke, ab_res).items():
        if not f.exists() or R3.sha_file(f) != rec["sha256"][k]:
            raise SystemExit(f"seed {seed}: {f.name} is missing or its SHA-256 differs from {p.name}")
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    seeds = check_seeds(args.seeds, args.smoke)
    R3.assert_rule()
    R3.assert_inputs([RUN_BASELINES])
    if not args.smoke:
        R3.assert_inputs([CODES_PROV])
    earlier = earlier_pair_hashes()
    done = {}
    for s in seeds:
        rec = build_seed(s, args.smoke, earlier=earlier, others_new=done)
        done[s] = rec["episode_pair_sha256"]
    say(f"BUILD PASS: seeds {list(seeds)}{' (smoke)' if args.smoke else ''}")


if __name__ == "__main__":
    main()
