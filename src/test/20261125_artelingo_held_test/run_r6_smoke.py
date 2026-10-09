"""Round 6: the end-to-end smoke (DECISION_RULE.md of this folder: section 6 item 7, section 8 item 1, section 9; spec
D21; contracts sections 7 to 9; ticket 15). CPU orchestration in stages; the GPU jobs run between them, launched by the
main session (this script only writes their inputs, prints their commands and checks their outputs).

    cd /project/CoSiR && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/run_r6_smoke.py \
        --stage check|gpu-inputs|list-input|finish [--after-crash | --fix 1 | --reserve] [GPU output folders]

Kinds (one smoke folder and one record each; the held runner and the apply and descriptive steps read the record):
  (no flag)      results/smoke/          results/smoke_record.json          the smoke before the read
  --after-crash  results/smoke/crash1/   results/smoke_record_crash1.json   code corrected after a crash (rule 8.1)
  --fix 1        results/smoke/fix1/     results/smoke_record_fix1.json     the corrected runner of a --fix 1 pass
  --reserve      results/smoke/reserve/  results/smoke_record_reserve.json  the corrected code of the reserve read
The record is written once, outside results/smoke/, never replaced or deleted (a passed one is refused at the start;
to smoke again, the run chat moves the record and the smoke folder aside); its copy in this folder is committed.

Stages, in this order:
  check       step 1 alone.
  gpu-inputs  step 1; step 2, run_r6_held.run_smoke (smoke seeds 9001 to 9003, 64 episodes per pair, selection rows,
              results/smoke[/<kind>]/); step 3, the GPU job inputs of round 1 under <smoke>/gpu/ (r6_gpu_inputs'
              writers): a verbaliser job per smoke seed (all 192 episodes; run with the four wordings, so that the
              smoke seed's own DTS tuning and every seed's chosen wording are covered), the reranker job of seed 9001,
              the FT feature job of every member row; then it prints the GPU commands and stops.
  list-input  step 1; the round-1 verbaliser outputs checked (every key of every seed and wording exactly once, of the
              listed jobs, by the current GPU script bytes); the one listing job of round 2: every distinct phrase of
              those answers and the three aspect names, at K 8 and 16. Prints the GPU command and stops.
  finish      step 1; step 4: the GPU outputs checked (keys, counts, script bytes); the DTS stages sanity, tune, chosen
              and stop (run_r6_dts.py) on smoke seed 9001 into <smoke>/dts/ (the stop's clock start: --dts-clock-start,
              default the gpu-inputs stage's start, agent default); <smoke>/external_sources.json with all four entries
              (a family whose outputs were not given, or whose DTS sanity or stop did not pass at smoke scale, is
              {"missing": reason}, and the record says so); the smoke agreement record ("smoke": true, bound to the
              smoke pass's bytes); run_r6_apply_rule --smoke; run_r6_descriptive --smoke, in which every external
              row whose family's outputs were given (and passed their check) must be present. Step 5, the wiring
              mutation: a copy of r6_score.py in <smoke>/mutation/code/ with AFF's gated term passed where CF
              expects G_cf (`G = gated` for `G = cf_terms(gated)`), and the smoke scoring (run_r6_held.run_smoke)
              run on it in a subprocess (results in <smoke>/mutation/results/): the per-cell condition-free assertion
              must fire (non-zero exit, its message in the log). Step 6, the leak check of every log under
              <smoke>/logs/ (the steps' captured output and this script's own): no decimal number, no R@1 value, no
              metric key with a
              value. Step 7, the smoke record.
Step 1 (every stage): refused (exit 4) unless the earlier stage outputs exist and are current (rule section 6 item 7):
  refit_check.json, picks_seed42.json, regression_seed42.json (passed, written by the current bytes of every r6 module
  their runners ran), sensitivity_seed42.json (from the current regression file, current bytes), and the seed-42 DTS
  records dts_sanity.json (passed), dts_tune.json, dts_seed42.json, dts_stop.json (built, no stop, bound to the chosen
  record's bytes), each by the current bytes of what run_r6_dts.py ran and of dts_settings.json, their GPU outputs by
  the current GPU script bytes. Every stage that must rerun is listed, in order (a stale refit, picks or regression
  also reruns what follows it). A DTS stop that stops is not rerun: the user decides (rule section 9). The later stages
  also refuse unless the smoke pass of step 2 was made by today's bytes of every r6 file.

The record (results/<record>.json and its copy here): "passed" (true only when every step passed, the mutation fired
and the leak check found nothing), the kind, the SHA-256 of the held runner, of this script, module_sha256 =
r6_common.r6_module_shas() (every r6_*.py and run_r6_*.py here, dts_settings.json, the r6 scripts), asserted equal to
the smoke pass's, the time, each step's pass and log, the mutation's result, the leak check, the external rows.

Exit codes: 0 passed (a stage done); 3 a step failed (finish: the record is written with "passed": false); 4 refused.
Prints step names, pass or FAIL, file paths and SHA-256s; never a metric, never a decimal number.
"""
import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_dts as DT  # noqa: E402
import r6_episodes as E  # noqa: E402
import r6_external as XT  # noqa: E402
import r6_gpu_common as G  # noqa: E402
import r6_gpu_inputs as I  # noqa: E402
import r6_gpu_listing as L  # noqa: E402
import r6_stats as ST  # noqa: E402
import run_r6_apply_rule as AR  # noqa: E402
import run_r6_descriptive as RD  # noqa: E402
import run_r6_dts as RDTS  # noqa: E402
import run_r6_held as RH  # noqa: E402

import numpy as np  # noqa: E402

EXIT_FAIL = 3
EXIT_REFUSE = 4
STAGES = ("check", "gpu-inputs", "list-input", "finish")
KINDS = {"smoke": SimpleNamespace(kind="smoke", subdir=None, record="smoke_record.json", flag=[]),
         "crash1": SimpleNamespace(kind="crash1", subdir="crash1", record="smoke_record_crash1.json",
                                   flag=["--after-crash"]),
         "fix1": SimpleNamespace(kind="fix1", subdir="fix1", record="smoke_record_fix1.json", flag=["--fix", "1"]),
         "reserve": SimpleNamespace(kind="reserve", subdir="reserve", record="smoke_record_reserve.json",
                                    flag=["--reserve"])}
# the consumers' names (asserted in the tests): run_r6_held, run_r6_apply_rule, run_r6_descriptive
assert (RH.SMOKE_RECORDS == {"held": KINDS["smoke"].record, "fix1": KINDS["fix1"].record,
                             "reserve": KINDS["reserve"].record} and RH.SMOKE_RECORD_CRASH == KINDS["crash1"].record
        and AR.SMOKE_RECORDS == RD.SMOKE_RECORDS == R.SMOKE_RECORDS == {k: v.record for k, v in KINDS.items()}
        and RH.DTS_STOP_NAME == RDTS.STOP)
DTS_SEED = R.SMOKE_SEEDS[0]                    # the smoke seed whose DTS stages run (and the reranker's seed)
SEED42_CHAIN = (("refit", RH.REFIT_NAME, "run_r6_refit.py", "run_r6_refit.py"),
                ("picks", RH.PICKS_NAME, "run_r6_picks.py", "run_r6_picks.py"),
                ("regression", RH.REGRESSION_NAME, "run_r6_held.py", "run_r6_held.py --mode regression"),
                ("sensitivity", RH.SENS42_NAME, "run_r6_sensitivity.py", "run_r6_sensitivity.py"))
DTS42 = (RDTS.SANITY, RDTS.TUNE, RDTS.chosen_name(R.DEV_SEED), RDTS.STOP)
DTS42_COMMAND = ("run_r6_dts.py stages sanity, tune, chosen and stop on seed 42 (cached GPU outputs only if their "
                 "scripts are unchanged)")
WIRING_FROM = "    G = cf_terms(gated)\n"
WIRING_TO = "    G = gated  # the smoke's wiring mutation (run_r6_smoke.py): AFF's gated term where CF expects G_cf\n"
WIRING_MESSAGE = "must be identical under both conditions (condition-free)"
WIRING_CELL = "the CF term of cell"
DRIVER = '''"""The smoke scoring of run_r6_smoke.py's wiring mutation: run_r6_held.run_smoke with the r6_score.py
of this folder (a copy) in place of the real one. argv: the round-6 folder, the results folder."""
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
F, RESULTS = Path(sys.argv[1]), Path(sys.argv[2])
sys.path[:0] = [str(CODE), str(F)]
import r6_common  # noqa: E402,F401  (the round-6 folder's; this folder holds only r6_score.py)
import r6_score  # noqa: E402
import run_r6_held as RH  # noqa: E402

assert Path(r6_score.__file__).resolve().parent == CODE and RH.S is r6_score, "the copy of r6_score.py is not in use"
print(f"r6_score.py from the copy in {CODE}", flush=True)
sys.exit(RH.run_smoke(results=RESULTS, here=F))
'''
LEAKS = (("decimal number", re.compile(r"\d*\.\d+")),
         ("R@1 value", re.compile(r"R@1\W{0,3}[-+]?\d")),
         ("metric key with a value",
          re.compile(r"\b(r1|r1_point|gain|gain_point|other|swap|strict|point|ci95|ci_holm|mean_r1|hits|score|"
                     r"score_int|margin|n_j)\b[\"']?\s*[:=]\s*[-+\[(]?\d")))
SUBPROCESS_ENV = {"PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
                  "CUDA_VISIBLE_DEVICES": ""}


class Refused(Exception):
    """A precondition is not met (exit 4); nothing is run."""


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- paths, output, capture

def paths_of(kind, results, folder=None) -> SimpleNamespace:
    k = KINDS[kind]
    results = Path(results)
    smoke = results / "smoke" / (k.subdir or "")
    return SimpleNamespace(kind=k, results=results, folder=Path(folder or HERE), smoke=smoke, gpu=smoke / "gpu",
                           logs=smoke / "logs", dts=smoke / "dts", record=results / k.record,
                           state=smoke / "gpu" / "smoke_state.json",
                           prefix="smoke" + (f"_{k.subdir}" if k.subdir else ""))


class Say:
    """Prints a line and appends it to this stage's own log under <smoke>/logs/ (scanned by the leak check)."""

    def __init__(self, log):
        self.log = Path(log)

    def __call__(self, msg=""):
        print(msg, flush=True)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")


@contextlib.contextmanager
def captured(log):
    """Everything a step prints, Python-level and file-descriptor-level (stdout and stderr), appended to ``log``."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    with open(log, "ab", buffering=0) as raw, open(log, "a", encoding="utf-8", buffering=1) as text:
        os.dup2(raw.fileno(), 1)
        os.dup2(raw.fileno(), 2)
        try:
            with contextlib.redirect_stdout(text), contextlib.redirect_stderr(text):
                yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def run_step(log, fn) -> int:
    """fn() inside captured(log); -> its exit code (an exception: its traceback in the log, code 1)."""
    with captured(log):
        try:
            code = fn()
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
        except Exception:                                    # noqa: BLE001  (the step failed; the log says why)
            traceback.print_exc()
            code = 1
    return int(code or 0)


def rel(path) -> str:
    """A path as recorded: relative to the main checkout when under it."""
    p = Path(path).resolve()
    return p.relative_to(R.MAIN).as_posix() if p.is_relative_to(R.MAIN) else str(p)


def next_log(paths, name) -> Path:
    paths.logs.mkdir(parents=True, exist_ok=True)
    k = len(list(paths.logs.glob("[0-9][0-9]_*.log")))
    return paths.logs / f"{k + 1:02d}_{name}.log"


@contextlib.contextmanager
def patched(module, **values):
    """Module attributes set for the duration (run_r6_apply_rule's RESULTS and SMOKE when the results folder is not
    the default one, as in the tests)."""
    old = {k: getattr(module, k) for k in values}
    try:
        for k, v in values.items():
            setattr(module, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


# ---------------------------------------------------------------- step 1: the earlier stages are current

def _scripts_stale(fp, here) -> list:
    """Names of the r6 files a GPU fingerprint lists whose current SHA-256 differs (files outside this folder, e.g.
    the reranker's src module, are not r6 files and are skipped)."""
    out = []
    for name, sha in ((fp or {}).get("scripts_sha256") or {}).items():
        f = Path(here) / name
        if f.is_file() and R.sha256_file(f) != sha:
            out.append(name)
    return out


def _chain_problem(results, here, name, file, runner) -> str | None:
    path = Path(results) / file
    if not path.is_file():
        return f"{file} does not exist"
    try:
        rec = _read_json(path)
    except (OSError, ValueError) as e:
        return f"{file} cannot be read ({type(e).__name__})"
    if name != "sensitivity" and rec.get("passed") is not True:
        return f"{file} did not pass"
    if name == "sensitivity":
        reg = Path(results) / RH.REGRESSION_NAME
        if not reg.is_file() or rec.get("regression_sha256") != R.sha256_file(reg):
            return f"{file} was not written from the current {RH.REGRESSION_NAME}"
        bad = [c for c in ST.CHECKS + ST.SECONDARY
               if not all(RH._is_num((rec.get(c) or {}).get(k)) for k in RH.SIGMA_KEYS)]
        if bad:
            return f"{file} holds no sigma parts for {bad}"
    stale = RH.stale_modules(rec, runner, here)
    if stale:
        return f"{file} was written by other bytes of {stale}"
    return None


def _dts_problem(results, here) -> tuple:
    """-> (problem or None, stops: bool). stops: the seed-42 stop record stops the work (not a rerun). The stop record
    is checked by the held runner's own run_r6_held.dts_stop_problem; the other three records here."""
    why, fired = RH.dts_stop_problem(results, here)
    if fired:
        return why, True
    recs = {}
    for name in DTS42:
        path = Path(results) / name
        if not path.is_file():
            return f"{name} does not exist", False
        try:
            recs[name] = _read_json(path)
        except (OSError, ValueError) as e:
            return f"{name} cannot be read ({type(e).__name__})", False
        if recs[name].get("seed") != R.DEV_SEED:
            return f"{name} is not a seed-42 record", False
    if why is not None:
        return why, False
    stop, chosen = recs[RDTS.STOP], RDTS.chosen_name(R.DEV_SEED)
    if recs[RDTS.SANITY].get("passed") is not True:
        return f"{RDTS.SANITY} did not pass", False
    if (stop.get("input_sha256") or {}).get(chosen) != R.sha256_file(Path(results) / chosen):
        return f"{RDTS.STOP} was not evaluated on the current {chosen}", False
    settings_key = RH._key("dts_settings.json", here)
    settings_sha = R.sha256_file(Path(here) / "dts_settings.json")
    for name, rec in recs.items():
        stale = RH.stale_modules(rec, "run_r6_dts.py", here)
        if (rec.get("module_sha256") or {}).get(settings_key) != settings_sha:
            stale.append(settings_key)
        if stale:
            return f"{name} was written by other bytes of {stale}", False
        for fps in (rec.get("gpu_fingerprints") or {}).values():
            for folder, fp in (fps or {}).items():
                gone = _scripts_stale(fp, here)
                if gone:
                    return f"{name}: its GPU output {folder} was made by other bytes of {gone}", False
    return None, False


def stage_problems(results, here=None) -> SimpleNamespace:
    """Step 1 (rule section 6 item 7): -> namespace(rerun [(stage, command, reason)] in order, stops (a reason or
    None), sha256 {file: sha} of the records read)."""
    here = Path(here or HERE)
    rerun, first = [], None
    for i, (name, file, runner, command) in enumerate(SEED42_CHAIN):
        why = _chain_problem(results, here, name, file, runner)
        if why is not None and first is None:
            first = name
        if why is not None or first is not None:
            rerun.append((name, command, why or f"it follows {first}, which reruns"))
    why, stops = _dts_problem(results, here)
    if why is not None and not stops:
        rerun.append(("DTS seed 42", DTS42_COMMAND, why))
    files = [f for _, f, _, _ in SEED42_CHAIN] + list(DTS42)
    sha = {f: R.sha256_file(Path(results) / f) for f in files if (Path(results) / f).is_file()}
    return SimpleNamespace(rerun=rerun, stops=why if stops else None, sha256=sha)


def require_stages(results, here, say) -> dict:
    """Step 1, refusing: every stage to rerun, in order, in the refusal; -> the records' SHA-256s."""
    st = stage_problems(results, here)
    if st.stops:
        raise Refused(st.stops)
    if st.rerun:
        raise Refused("these stages must rerun before the smoke (rule section 6 item 7), in this order:\n"
                      + "\n".join(f"  {name}: {command} ({why})" for name, command, why in st.rerun))
    say("step 1: refit, picks, regression, sensitivity and the seed-42 DTS records exist and are current")
    return st.sha256


def require_smoke_current(paths, here, say) -> dict:
    """The later stages: the smoke pass of step 2 and the stage state were made by today's bytes of every r6 file."""
    here = Path(here or HERE)
    pass_path, cur = paths.smoke / "held_pass.json", R.r6_module_shas(here)
    _require(pass_path.is_file() and paths.state.is_file(),
             f"{pass_path} or {paths.state} is missing: run --stage gpu-inputs first", Refused)
    rec, state = _read_json(pass_path), _read_json(paths.state)
    _require(rec.get("mode") == "smoke" and rec.get("module_sha256") == cur and state.get("module_sha256") == cur,
             f"the r6 files changed since the held smoke of {paths.smoke}: move that folder aside and smoke again from "
             f"--stage gpu-inputs (rule section 6 item 7)", Refused)
    say(f"the smoke pass {pass_path} was made by the current bytes of every r6 file")
    return state


# ---------------------------------------------------------------- stage gpu-inputs (steps 2 and 3)

def load_data(env):
    """(data, annotations): the env's data when given (one load for the whole stage)."""
    from src.data.artelingo import ANNOTATIONS_PATH, load_artelingo
    data = env.data if env is not None and getattr(env, "data", None) is not None else load_artelingo()
    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        annotations = json.load(f)
    return data, annotations


def _kept(job, want) -> bool:
    """A job folder an earlier attempt wrote is kept when its record holds the same episodes (never overwritten)."""
    rec = _read_json(Path(job) / I.RECORD)
    _require(all(rec.get(k) == v for k, v in want.items()), f"{job} exists and holds other inputs: move it aside")
    return True


def write_gpu_jobs(paths, eps, data, annotations, staging, image_root) -> dict:
    """Round 1's job folders (r6_gpu_inputs' writers) -> {"verbalise": {seed: name}, "rerank": name, "ft": name}."""
    paths.gpu.mkdir(parents=True, exist_ok=True)
    jobs = {"verbalise": {}, "rerank": None, "ft": None}
    files = {s: paths.smoke / RH.episodes_name(s) for s in R.SMOKE_SEEDS}
    for s in R.SMOKE_SEEDS:
        name = f"{paths.prefix}_verbalise_{s}"
        out, want = paths.gpu / name, {"seed": s, "n_episodes": eps[s].n,
                                       "episodes_sha256_in_pair_order": [eps[s].sha[p] for p in R.PAIR_NAMES]}
        if not (out.exists() and _kept(out, want)):
            I.write_job(eps[s], data.sample_ids, data.paintings, annotations, out, None, G.sha256_file(files[s]),
                        staging, image_root)
        jobs["verbalise"][str(s)] = name
        print(f"verbaliser job {out}: seed {s}, {eps[s].n} episodes", flush=True)
    name = f"{paths.prefix}_rerank_{DTS_SEED}"
    out = paths.gpu / name
    want = {"seed": DTS_SEED, "n_episodes": eps[DTS_SEED].n,
            "episodes_sha256_in_pair_order": [eps[DTS_SEED].sha[p] for p in R.PAIR_NAMES]}
    if not (out.exists() and _kept(out, want)):
        I.write_rerank_job(eps[DTS_SEED], data.sample_ids, data.paintings, annotations, out, None,
                           G.sha256_file(files[DTS_SEED]), staging, image_root)
    jobs["rerank"] = name
    print(f"reranker job {out}: seed {DTS_SEED}", flush=True)
    rows = I.episode_rows([eps[s] for s in R.SMOKE_SEEDS])
    _require(np.array_equal(rows, XT.member_union(paths.smoke, R.SMOKE_SEEDS)),
             "the FT rows are not the member rows the descriptive pass joins")
    name = f"{paths.prefix}_ft"
    out = paths.gpu / name
    shas = [G.sha256_file(files[s]) for s in R.SMOKE_SEEDS]
    if not (out.exists() and _kept(out, {"episodes_files_sha256": shas})):
        I.write_ft_job(rows, data.sample_ids, data.paintings, annotations, out, staging, image_root,
                       {"episodes_files_sha256": shas})
    jobs["ft"] = name
    print(f"FT feature job {out}: {len(rows)} rows", flush=True)
    return jobs


def flag_text(paths) -> str:
    return " ".join(paths.kind.flag)


def print_round1(paths, jobs, staging, say):
    ver = list(jobs["verbalise"].values())
    ft = XT.FT_CKPTS
    say("GPU round 1 (main session; on a DAS6 node through the cluster CLI, or the local GPU under its lock):")
    say(f"  ship each job folder of {paths.gpu} with its images:")
    for name in ver + [jobs["rerank"]]:
        say(f"    /usr/bin/python3 scripts/das6_sync_r6.py --node <node> --job-dir {paths.gpu / name} --images --run")
    say(f"    /usr/bin/python3 scripts/das6_sync_r6.py --node <node> --job-dir {paths.gpu / jobs['ft']} --images "
        f"--ckpt LB_lr3e-5={R.MAIN / ft['LB']['path']} --ckpt LoRA_lr1e-4={R.MAIN / ft['LoRA']['path']} --run")
    say("  launch:")
    for name in ver:
        say(f"    cluster launch --node <node> -- bash scripts/run_r6_verbalise.sh {name} "
            f"--wordings {','.join(DT.WORDINGS)}")
    say(f"    cluster launch --node <node> -- bash scripts/run_r6_rerank.sh {jobs['rerank']}")
    say(f"    cluster launch --node <node> -- bash scripts/run_r6_ftfeat.sh {jobs['ft']}")
    say(f"  (locally: R6_JOB_ROOT={paths.gpu} R6_IMAGE_DIR={staging} R6_OUT=<folder> flock -o -w <seconds> "
        f"/tmp/gpu0.lock bash scripts/run_r6_<job>.sh <job> ...)")
    say("  then pull each tag (cluster pull --tag <tag>) and run:")
    say(f"    run_r6_smoke.py --stage list-input {flag_text(paths)} --verbalise-out <outputs/r6_verbalise/<job> of "
        f"each verbaliser job, every shard>")


def stage_gpu_inputs(paths, here, env, staging, image_root, say) -> int:
    _require(not paths.record.exists(), f"{paths.record} exists: a smoke record is never replaced; move it and "
                                        f"{paths.smoke} aside to smoke again", Refused)
    _require(not (paths.smoke / "held_verdict.json").exists(),
             f"{paths.smoke} holds a smoke verdict: move it (and {paths.record.name}, if any) aside to smoke again",
             Refused)
    seed42 = require_stages(paths.results, here, say)
    started = R.amsterdam_now()
    box = {}

    def setup():
        box["env"] = env if env is not None else RH.setup()
        return 0

    log = next_log(paths, "setup")
    code = run_step(log, setup)
    say(f"setup (data, split, heads, PM fits, readers): {'pass' if code == 0 else 'FAIL'}; log {log}")
    if code:
        return EXIT_FAIL
    log = next_log(paths, "held_smoke")
    code = run_step(log, lambda: RH.run_smoke(results=paths.results, subdir=paths.kind.subdir, here=here,
                                              env=box["env"]))
    say(f"step 2, held smoke (run_r6_held.run_smoke, seeds {R.SMOKE_SEEDS[0]} to {R.SMOKE_SEEDS[-1]}): "
        f"{'pass' if code == 0 else f'FAIL (exit {code})'}; log {log}")
    if code:
        return EXIT_FAIL
    eps = {s: E.load_episodes(paths.smoke / RH.episodes_name(s)) for s in R.SMOKE_SEEDS}
    box["jobs"] = None

    def jobs():
        data, ann = load_data(box["env"])
        box["jobs"] = write_gpu_jobs(paths, eps, data, ann, staging, image_root)
        return 0

    log = next_log(paths, "gpu_inputs")
    code = run_step(log, jobs)
    say(f"step 3, GPU job inputs under {paths.gpu}: {'pass' if code == 0 else 'FAIL'}; log {log}")
    if code:
        return EXIT_FAIL
    state = {"kind": paths.kind.kind, "started": started, "jobs": box["jobs"], "image_staging": str(staging),
             "module_sha256": R.r6_module_shas(here), "seed42_records_sha256": seed42,
             "smoke_pass_sha256": R.sha256_file(paths.smoke / "held_pass.json"), "time": R.amsterdam_now()}
    _write_json(paths.state, state)
    say(f"state {paths.state}")
    print_round1(paths, box["jobs"], staging, say)
    return 0


# ---------------------------------------------------------------- GPU outputs (stages list-input and finish)

def _job_key(job, files) -> tuple:
    return tuple(G.sha256_file(Path(job) / f) for f in files)


def check_verbaliser(paths, state, outs, here) -> dict:
    """The round-1 verbaliser outputs: each folder an output of one smoke job (fingerprint input SHA-256s) by the
    current GPU script bytes; per seed every (seed, episode, condition, wording) key of the four wordings exactly once
    after merging (r6_dts). -> {seed: namespace(dirs, job, answers)}."""
    _, settings_sha = DT.load_settings()
    files = ("rows_manifest.npz", "verbalise_input.npz")
    jobs = {int(s): paths.gpu / name for s, name in state["jobs"]["verbalise"].items()}
    keys = {_job_key(j, files): s for s, j in jobs.items()}
    by_seed = {s: [] for s in jobs}
    for d in outs:
        fp = DT._fingerprint(d, "r6_gpu_verbalise", settings_sha)
        ins = fp.get("inputs_sha256") or {}
        s = keys.get(tuple(ins.get(f) for f in files))
        _require(s is not None, f"{d}: not an output of a verbaliser job of this smoke")
        stale = _scripts_stale(fp, here)
        _require(not stale and "r6_gpu_verbalise.py" in (fp.get("scripts_sha256") or {}),
                 f"{d}: made by other bytes of {stale or ['r6_gpu_verbalise.py']} (rule section 6 item 7)")
        by_seed[s].append(Path(d))
    out = {}
    for s, j in jobs.items():
        _require(by_seed[s], f"no verbaliser output of seed {s} was given")
        eps = E.load_episodes(paths.smoke / RH.episodes_name(s))
        merged = DT.merge_verbaliser(by_seed[s], DT.WORDINGS, settings_sha, seeds={s})
        DT.check_verbaliser_jobs(merged, [j], eps)
        for w in DT.WORDINGS:
            DT.select_answers(merged.answers, s, w, np.arange(eps.n), eps.n)
        _require(len(merged.answers) == eps.n * len(DT.CONDITIONS) * len(DT.WORDINGS),
                 f"seed {s}: {len(merged.answers)} answers, not one per episode, condition and wording")
        out[s] = SimpleNamespace(dirs=by_seed[s], job=j, answers=merged.answers, n=eps.n, files=dict(merged.files))
        print(f"verbaliser outputs of seed {s}: every key present once ({len(by_seed[s])} folder(s))", flush=True)
    return out


def listing_items(ver) -> list:
    """Round 2's listing items: the three aspect names and every distinct phrase of the four wordings of the three
    smoke seeds, at K 8 and 16 (one job, so that no (phrase, K) is answered twice)."""
    phrases = list(DT.TARGET_NAME.values())
    for s, v in sorted(ver.items()):
        for w in DT.WORDINGS:
            raw = DT.select_answers(v.answers, s, w, np.arange(v.n), v.n)
            phrases += [DT.phrase_of(a) for c in DT.CONDITIONS for a in raw[c]]
    return DT.listing_items(phrases, DT.KS)


def write_listing_job(job, items, ver, here):
    """listing_input.jsonl and job_record.json (run_r6_dts.stage_list_input's format), or kept when an earlier attempt
    wrote the same items."""
    settings, settings_sha = DT.load_settings()
    job = Path(job)
    if job.exists():
        _require(L.load_listing_input(job / RDTS.LISTING_INPUT, settings) == items,
                 f"{job} exists and holds other items: move it aside")
        return
    partial = job.with_name(job.name + ".partial")
    _require(not partial.exists(), f"{partial} exists (an interrupted write); move it aside")
    partial.mkdir(parents=True)
    with open(partial / RDTS.LISTING_INPUT, "w", encoding="utf-8") as f:
        for p, K in items:
            f.write(json.dumps({"phrase": p, "K": int(K)}, ensure_ascii=True) + "\n")
    _require(L.load_listing_input(partial / RDTS.LISTING_INPUT, settings) == items,
             "the listing input does not read back as written")
    rec = {"stage": "list-input", "for": "smoke", "seeds": list(R.SMOKE_SEEDS), "wordings": list(DT.WORDINGS),
           "K": list(DT.KS), "n_items": len(items), "rule_sha256": R.RULE_SHA256, "settings_sha256": settings_sha,
           "listing_input_sha256": G.sha256_file(partial / RDTS.LISTING_INPUT),
           "input_sha256": {"verbaliser": {str(s): v.files for s, v in ver.items()}},
           "module_sha256": R.r6_module_shas(here), "time": R.amsterdam_now()}
    G.write_json(partial / RDTS.JOB_RECORD, rec)
    os.replace(partial, job)


def stage_list_input(paths, here, outs, say) -> int:
    require_stages(paths.results, here, say)
    state = require_smoke_current(paths, here, say)
    box = {}

    def work():
        ver = check_verbaliser(paths, state, outs, here)
        items = listing_items(ver)
        name = f"{paths.prefix}_listing"
        write_listing_job(paths.gpu / name, items, ver, here)
        box["name"] = name
        print(f"listing job {paths.gpu / name}: {len(items)} items", flush=True)
        return 0

    log = next_log(paths, "list_input")
    code = run_step(log, work)
    say(f"round-1 verbaliser outputs and the listing job input: {'pass' if code == 0 else 'FAIL'}; log {log}")
    if code:
        return EXIT_FAIL
    st = dict(state, listing_job=box["name"])
    _write_json(paths.state, st)
    say("GPU round 2 (main session):")
    say(f"    /usr/bin/python3 scripts/das6_sync_r6.py --node <node> --job-dir {paths.gpu / box['name']} --run")
    say(f"    cluster launch --node <node> -- bash scripts/run_r6_listing.sh {box['name']}")
    say("  then pull it and run:")
    say(f"    run_r6_smoke.py --stage finish {flag_text(paths)} --verbalise-out <each, as before> --listing-out "
        f"<outputs/r6_listing/{box['name']}> --rerank-out <outputs/r6_rerank/{state['jobs']['rerank']}, every shard> "
        f"--ft-out <outputs/r6_ft/{state['jobs']['ft']}>")
    return 0


def check_listings(paths, state, outs, here) -> list:
    settings, settings_sha = DT.load_settings()
    _require("listing_job" in state, "the listing job of round 2 was not written (--stage list-input)")
    items = L.load_listing_input(paths.gpu / state["listing_job"] / RDTS.LISTING_INPUT, settings)
    for d in outs:
        fp = DT._fingerprint(d, "r6_gpu_listing", settings_sha)
        stale = _scripts_stale(fp, here)
        _require(not stale and "r6_gpu_listing.py" in (fp.get("scripts_sha256") or {}),
                 f"{d}: made by other bytes of {stale or ['r6_gpu_listing.py']} (rule section 6 item 7)")
    merged = DT.merge_listings(outs, settings_sha)
    missing = [it for it in items if tuple(it) not in merged.answers]
    _require(not missing, f"{len(missing)} listing items have no answer in the given listing outputs")
    print(f"listing outputs: every one of the {len(items)} items present ({len(outs)} folder(s))", flush=True)
    return [Path(d) for d in outs]


def check_rerank(paths, state, outs, here) -> dict:
    entry = {"job": str(paths.gpu / state["jobs"]["rerank"]), "out": [str(d) for d in outs]}
    m = XT.load_mllm(entry, paths.smoke, True)
    n = E.load_episodes(paths.smoke / RH.episodes_name(DTS_SEED)).n
    _require(set(m.merged) == set(range(n)), f"the reranker scores hold {len(m.merged)} of the {n} episodes")
    for d, fp in m.fingerprints.items():
        stale = _scripts_stale(fp, here)
        _require(not stale and "r6_gpu_rerank.py" in (fp.get("scripts_sha256") or {}),
                 f"{d}: made by other bytes of {stale or ['r6_gpu_rerank.py']} (rule section 6 item 7)")
    print(f"reranker outputs: every episode of seed {DTS_SEED} present once ({len(outs)} folder(s))", flush=True)
    return entry


def check_ft(paths, state, outs, here) -> dict:
    entry = {"job": str(paths.gpu / state["jobs"]["ft"]), "out": [str(d) for d in outs]}
    f = XT.load_ft(entry, paths.smoke, True)
    missing = {v: x for v, x in f.variants.items() if isinstance(x, str)}
    _require(not missing, f"FT variants without features: {sorted(missing)}")
    for v, x in f.variants.items():
        stale = _scripts_stale(x.fingerprint, here)
        _require(not stale and "r6_gpu_ft_features.py" in (x.fingerprint.get("scripts_sha256") or {}),
                 f"{x.folder}: made by other bytes of {stale or ['r6_gpu_ft_features.py']} (rule section 6 item 7)")
    print(f"FT outputs: {', '.join(f.variants)} on every member row ({len(f.rows)} rows)", flush=True)
    return entry


# ---------------------------------------------------------------- stage finish (steps 4 to 7)

def dts_stages(paths, ver, listing_outs, clock_start, say) -> tuple:
    """run_r6_dts.py's sanity, tune, chosen and stop on smoke seed 9001 into <smoke>/dts/ (a stage whose record an
    earlier attempt wrote is kept). -> (passed: bool, missing reason or None, {stage: {...}})."""
    v = ver[DTS_SEED]
    eps_path = paths.smoke / RH.episodes_name(DTS_SEED)
    common = ["--seed", str(DTS_SEED), "--episodes", str(eps_path), "--out", str(paths.dts)]
    va = [x for d in v.dirs for x in ("--verbalise-out", str(d))] + ["--verbalise-job", str(v.job)]
    la = [x for d in listing_outs for x in ("--listing-out", str(d))]
    plan = (("sanity", RDTS.SANITY, ["--stage", "sanity", *la, *common]),
            ("tune", RDTS.TUNE, ["--stage", "tune", *va, *la, *common]),
            ("chosen", RDTS.chosen_name(DTS_SEED), ["--stage", "chosen", *va, *la, *common]),
            ("stop", RDTS.STOP, ["--stage", "stop", "--seed", str(DTS_SEED), "--out", str(paths.dts),
                                 "--clock-start", clock_start]))
    steps = {}
    for name, out, argv in plan:
        if (paths.dts / out).is_file():
            steps[name] = {"passed": True, "kept": True}
            say(f"DTS {name} (seed {DTS_SEED}): kept {paths.dts / out}")
            continue
        log = next_log(paths, f"dts_{name}")
        code = run_step(log, lambda argv=argv: RDTS.main(argv))
        steps[name] = {"passed": code == 0, "exit": code, "log": rel(log)}
        say(f"DTS {name} (seed {DTS_SEED}): {'pass' if code == 0 else f'exit {code}'}; log {log}")
        if code == RDTS.EXIT_FAIL and name in ("sanity", "stop"):          # a smoke-scale outcome, not a wiring fault
            why = {"sanity": "the smoke seed's DTS-N sanity did not pass at 64 episodes per pair",
                   "stop": "the smoke seed's DTS stop record stops (rule section 7 items 5 to 7)"}[name]
            steps[name]["passed"] = True
            return True, why, steps
        if code:
            return False, f"DTS {name} failed (exit {code})", steps
    return True, None, steps


def write_agreement(paths) -> dict:
    """The smoke's stand-in for the phase-2 record (rule section 6 item 7): "smoke": true, bound to the smoke pass's
    bytes. Written once; an earlier attempt's file must bind the same pass."""
    pass_sha = R.sha256_file(paths.smoke / "held_pass.json")
    path = paths.smoke / "rederive_agreement.json"
    if path.exists():
        old = _read_json(path)
        _require(old.get("smoke") is True and old.get("held_pass_sha256") == pass_sha,
                 f"{path} binds another pass: move {paths.smoke} aside and smoke again")
        return old
    rec = {"phase": 2, "smoke": True, "all_agree": True, "held_pass_sha256": pass_sha, "pass_file": "held_pass.json",
           "n_quantities": len(ST.CHECKS) + len(ST.SECONDARY), "disagreements": [], "time": R.amsterdam_now(),
           "what": "the smoke chain's stand-in for the phase-2 agreement (run_r6_smoke.py): bound to the smoke pass's "
                   "bytes; no re-derivation ran"}
    _write_json(path, rec)
    return rec


def wiring_mutation(paths, here, control=False, timeout=3600) -> dict:
    """Step 5: the smoke scoring (run_r6_held.run_smoke in a subprocess) on a copy of r6_score.py in which AFF's gated
    term is passed where CF expects G_cf; the per-cell condition-free assertion must fire. ``control``: the same run
    on an unchanged copy, which must pass. Never mutates in place; every file stays under <smoke>/mutation*/."""
    here = Path(here or HERE)
    base = paths.smoke / ("mutation_control" if control else "mutation")
    d, k = base, 1
    while d.exists():
        k += 1
        d = base.with_name(f"{base.name}_{k}")
    code_dir, res = d / "code", d / "results"
    code_dir.mkdir(parents=True)
    res.mkdir()
    src = (here / "r6_score.py").read_text(encoding="utf-8")
    _require(src.count(WIRING_FROM) == 1, f"r6_score.py does not hold the line {WIRING_FROM.strip()!r} exactly once")
    (code_dir / "r6_score.py").write_text(src if control else src.replace(WIRING_FROM, WIRING_TO), encoding="utf-8")
    (code_dir / "smoke_scoring.py").write_text(DRIVER, encoding="utf-8")
    for _, f, _, _ in SEED42_CHAIN:
        shutil.copy2(paths.results / f, res / f)
    log = next_log(paths, "wiring_control" if control else "wiring_mutation")
    with open(log, "ab") as f:
        p = subprocess.run([sys.executable, str(code_dir / "smoke_scoring.py"), str(here), str(res)], cwd=R.MAIN,
                           env={**os.environ, **SUBPROCESS_ENV}, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    text = log.read_text(encoding="utf-8", errors="replace")
    found = WIRING_MESSAGE in text and WIRING_CELL in text
    out = {"exit": p.returncode, "message_found": found, "log": rel(log), "folder": rel(d),
           "mutated_line": None if control else WIRING_TO.strip()}
    if control:
        out["passed"] = p.returncode == 0 and "smoke held pass written" in text and not found
    else:
        out["fired"] = out["passed"] = p.returncode != 0 and found
    return out


def leak_hits(text) -> list:
    """[(line number, kind)] of the lines that show a decimal number, an R@1 value or a metric key with a value."""
    out = []
    for i, line in enumerate(str(text).splitlines(), 1):
        for kind, pat in LEAKS:
            if pat.search(line):
                out.append((i, kind))
    return out


def leak_check(files) -> dict:
    """Step 6 over ``files``; the hits name the file, line and kind only (never the text)."""
    hits = []
    for f in files:
        for line, kind in leak_hits(Path(f).read_text(encoding="utf-8", errors="replace")):
            hits.append({"file": rel(f), "line": line, "kind": kind})
    return {"passed": not hits, "files": [rel(f) for f in files], "hits": hits}


def external_rows(path) -> dict:
    rec = _read_json(path)
    return {name: ({"missing": e["missing"]} if "missing" in e else "present")
            for name, e in (rec.get("external") or {}).items()}


def stage_finish(paths, here, env, outs, clock_start, say) -> int:
    here = Path(here or HERE)
    _require(not paths.record.exists(), f"{paths.record} exists: a smoke record is never replaced", Refused)
    _require(not (paths.smoke / "held_verdict.json").exists(),
             f"{paths.smoke} holds a smoke verdict already: move it aside and smoke again from --stage gpu-inputs",
             Refused)
    seed42 = require_stages(paths.results, here, say)
    state = require_smoke_current(paths, here, say)
    steps, external, box = {}, {}, {}

    # step 4: the GPU outputs (keys, counts, script bytes)
    def gpu_check(name, fn, given):
        if not given:
            say(f"GPU outputs {name}: none given; marked missing")
            return None, f"no {name} output folder was given to the smoke (job not run)"
        log, res = next_log(paths, f"gpu_outputs_{name}"), {}

        def work():
            res["v"] = fn()
            return 0
        code = run_step(log, work)
        steps[f"gpu_outputs_{name}"] = {"passed": code == 0, "log": rel(log)}
        say(f"GPU outputs {name}: {'pass' if code == 0 else 'FAIL'}; log {log}")
        return (res["v"], None) if code == 0 else (None, f"the {name} outputs failed their check (log {rel(log)})")

    ver, ver_why = gpu_check("verbaliser", lambda: check_verbaliser(paths, state, outs["verbalise"], here),
                             outs["verbalise"])
    lis, lis_why = gpu_check("listing", lambda: check_listings(paths, state, outs["listing"], here), outs["listing"])
    mllm, mllm_why = gpu_check("reranker", lambda: check_rerank(paths, state, outs["rerank"], here), outs["rerank"])
    ft, ft_why = gpu_check("FT", lambda: check_ft(paths, state, outs["ft"], here), outs["ft"])
    external["mllm"] = mllm if mllm is not None else {"missing": mllm_why}
    external["ft"] = ft if ft is not None else {"missing": ft_why}
    lp = XT.resolve(XT.FT_CKPTS["LP"]["path"])
    external["ft_lp"] = {} if lp.is_file() else {"missing": f"the LP checkpoint {rel(lp)} is not on this machine"}
    if ver is not None and lis is not None:
        ok, why, dsteps = dts_stages(paths, ver, lis, clock_start or state["started"][:16], say)
        steps["dts_stages"] = {"passed": ok, "stages": dsteps, "missing": why}
        external["dts"] = ({"record": str(paths.dts / RDTS.chosen_name(DTS_SEED)),
                            "verbalise_job": [str(ver[s].job) for s in sorted(ver)],
                            "verbalise_out": [str(d) for s in sorted(ver) for d in ver[s].dirs],
                            "listing_out": [str(d) for d in lis]} if ok and why is None else {"missing": why})
    else:
        external["dts"] = {"missing": "; ".join(x for x in (ver_why, lis_why) if x)}
    sources = {k: external[k] for k in XT.SOURCE_KEYS}
    _write_json(paths.smoke / XT.SOURCES_NAME, sources)
    say(f"external sources {paths.smoke / XT.SOURCES_NAME}: "
        + ", ".join(f"{k} {'missing' if 'missing' in v else 'given'}" for k, v in sources.items()))

    # step 4: agreement, rule application, descriptive pass
    log = next_log(paths, "agreement")
    code = run_step(log, lambda: write_agreement(paths) and 0)                # 0 once written (a dict)
    steps["agreement"] = {"passed": code == 0, "log": rel(log)}
    say(f"smoke agreement record {paths.smoke / 'rederive_agreement.json'}: {'pass' if code == 0 else 'FAIL'}")
    sub = ["--smoke-subdir", paths.kind.subdir] if paths.kind.subdir else []
    log = next_log(paths, "apply_rule")
    with patched(AR, RESULTS=paths.results, SMOKE=paths.results / "smoke"):
        code = run_step(log, lambda: AR.main(["--smoke", *sub]))
    steps["apply_rule"] = {"passed": code == 0, "exit": code, "log": rel(log)}
    say(f"run_r6_apply_rule --smoke: {'pass' if code == 0 else f'FAIL (exit {code})'}; log {log}")
    if code == 0:
        def describe():
            box["env"] = env if env is not None else RH.setup()
            return RD.run(smoke=True, out=paths.smoke, env=box["env"], records=paths.results)
        log = next_log(paths, "descriptive")
        code = run_step(log, describe)
        steps["descriptive"] = {"passed": code == 0, "exit": code, "log": rel(log)}
        say(f"run_r6_descriptive --smoke: {'pass' if code == 0 else f'FAIL (exit {code})'}; log {log}")
    else:
        steps["descriptive"] = {"passed": False, "skipped": "the rule application failed"}
    rows = external_rows(paths.smoke / "descriptive.json") if steps["descriptive"]["passed"] else {}
    if rows:                    # a row whose family's outputs were given and passed their check must be present
        bad = sorted(n for n, v in rows.items() if v != "present" and "missing" not in sources[XT.FAMILY[n]])
        steps["external_rows"] = {"passed": not bad, "missing_although_given": bad}
        say(f"external rows: {', '.join(n for n, v in rows.items() if v == 'present') or 'none'} present"
            + (f"; FAIL, missing although given: {', '.join(bad)}" if bad else ""))

    # step 5: the wiring mutation
    say("step 5, the wiring mutation (a subprocess: setup, then the smoke scoring on the mutated copy)")
    try:
        mut = wiring_mutation(paths, here)
    except Exception as e:                                   # noqa: BLE001
        mut = {"fired": False, "passed": False, "error": type(e).__name__}
    say(f"wiring mutation: {'the condition-free assertion fired' if mut.get('fired') else 'FAIL (did not fire)'}; "
        f"log {mut.get('log')}")

    # step 6: the leak check (every log of this smoke, this stage's own included)
    leak = leak_check(sorted(paths.logs.glob("*.log")))
    say(f"step 6, leak check of {len(leak['files'])} logs: {'pass' if leak['passed'] else 'FAIL'}"
        + "".join(f"; {h['file']} line {h['line']} ({h['kind']})" for h in leak["hits"][:10]))

    # step 7: the record
    cur = R.r6_module_shas(here)
    pass_rec = _read_json(paths.smoke / "held_pass.json")
    same = pass_rec.get("module_sha256") == cur
    passed = bool(all(s["passed"] for s in steps.values()) and mut.get("fired") is True
                  and leak["passed"] and same)
    outputs = {n: R.sha256_file(paths.smoke / n) for n in ("held_pass.json", "held_arrays.npz", "held_verdict.json",
                                                           "rederive_agreement.json", "descriptive.json",
                                                           XT.SOURCES_NAME) if (paths.smoke / n).is_file()}
    settings_key = RH._key("dts_settings.json", here)
    rec = {"passed": passed, "kind": paths.kind.kind, "flag": flag_text(paths) or None,
           "smoke_folder": rel(paths.smoke), "rule_sha256": R.RULE_SHA256,
           "runner_sha256": R.sha256_file(here / "run_r6_held.py"),
           "smoke_runner_sha256": R.sha256_file(here / "run_r6_smoke.py"),
           "dts_settings_sha256": cur.get(settings_key),
           "scripts_sha256": {k: v for k, v in cur.items() if k.startswith("scripts/")},
           "module_sha256": cur,
           "covers": "r6_common.r6_module_shas(): every r6_*.py and run_r6_*.py of this folder, dts_settings.json, "
                     "scripts/run_r6_*.sh and scripts/das6_sync_r6.py",
           "same_bytes_as_the_smoke_pass": same, "started": state.get("started"), "time": R.amsterdam_now(),
           "seed42_records_sha256": seed42, "steps": steps, "mutation": mut, "leak_check": leak,
           "external_sources": sources, "external_rows": rows, "outputs_sha256": outputs}
    data = (json.dumps(rec, indent=1) + "\n").encode()
    with open(paths.record, "xb") as f:                    # written once, never replaced
        f.write(data)
    (paths.folder / paths.kind.record).write_bytes(data)
    say(f"step 7, smoke record {paths.record} {R.sha256_file(paths.record)} and its copy "
        f"{paths.folder / paths.kind.record}")
    missing = sorted(k for k, v in sources.items() if "missing" in v)
    if missing:
        say(f"external families marked missing (see the record): {', '.join(missing)}")
    say(f"smoke ({paths.kind.kind}): {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else EXIT_FAIL


# ---------------------------------------------------------------- main

def kind_of(after_crash=False, fix=None, reserve=False) -> str:
    _require(fix in (None, 1) and sum((bool(after_crash), fix is not None, bool(reserve))) <= 1,
             "at most one of --after-crash, --fix 1, --reserve", Refused)
    return "crash1" if after_crash else "fix1" if fix else "reserve" if reserve else "smoke"


def run(stage, kind="smoke", outs=None, clock_start=None, results=None, folder=None, here=None, env=None,
        staging=None, image_root=None) -> int:
    """One stage (module docstring). results, folder (where the record's copy goes), here (the folder whose r6 files
    count), env (run_r6_held.setup()'s namespace), staging and image_root (r6_gpu_inputs) may be given by a test."""
    here = Path(here or HERE)
    paths = paths_of(kind, results or R.RESULTS, folder)
    outs = {k: [Path(d) for d in (outs or {}).get(k) or []] for k in ("verbalise", "listing", "rerank", "ft")}
    say = Say(paths.logs / f"run_r6_smoke_{stage}.log")
    say(f"run_r6_smoke --stage {stage} ({paths.kind.kind}) {R.amsterdam_now()}: smoke folder {paths.smoke}")
    try:
        if stage == "check":
            require_stages(paths.results, here, say)
            return 0
        if stage == "gpu-inputs":
            return stage_gpu_inputs(paths, here, env, Path(staging or I.IMAGE_STAGING), image_root or I.WIKIART, say)
        if stage == "list-input":
            return stage_list_input(paths, here, outs["verbalise"], say)
        return stage_finish(paths, here, env, outs, clock_start, say)
    except Refused as e:
        say(f"refused: {e}")
        return EXIT_REFUSE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    once = ap.add_mutually_exclusive_group()
    once.add_argument("--after-crash", action="store_true", help="the smoke of code corrected after a crash")
    once.add_argument("--fix", type=int, choices=(1,), help="the smoke of the corrected runner of --fix 1")
    once.add_argument("--reserve", action="store_true", help="the smoke of the reserve read's corrected code")
    ap.add_argument("--verbalise-out", type=Path, action="append", default=[], help="a verbaliser output folder")
    ap.add_argument("--listing-out", type=Path, action="append", default=[], help="a listing output folder")
    ap.add_argument("--rerank-out", type=Path, action="append", default=[], help="a reranker output folder")
    ap.add_argument("--ft-out", type=Path, action="append", default=[], help="an FT feature output folder")
    ap.add_argument("--dts-clock-start", default=None,
                    help="finish: the smoke DTS stop's clock start 'YYYY-MM-DD HH:MM' (default the gpu-inputs start)")
    args = ap.parse_args(argv)
    try:
        kind = kind_of(args.after_crash, args.fix, args.reserve)
    except Refused as e:
        print(f"refused: {e}")
        return EXIT_REFUSE
    outs = {"verbalise": args.verbalise_out, "listing": args.listing_out, "rerank": args.rerank_out,
            "ft": args.ft_out}
    return run(args.stage, kind, outs, args.dts_clock_start)


if __name__ == "__main__":
    sys.exit(main())
