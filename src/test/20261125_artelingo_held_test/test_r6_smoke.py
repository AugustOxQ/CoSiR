"""Tests of the smoke chain's guards and records (run_r6_smoke.py, ticket 15; rule section 6 item 7, section 8 item 1):
step 1's refusals (a missing stage output, a stage run by other module bytes, a DTS stop), the later stages' refusal
of a smoke made by other bytes, the leak check, the record names each consumer reads, the module-SHA refusal of the
held runner, the apply step and the descriptive pass against the smoke record (each with a tmp copy of the r6 files
in which one file changes), and the held_arrays round trip from the held runner's writer (first read, --fix 1,
--reserve) to the descriptive pass's reader. No data is loaded; the end to end on real selection rows is
test_r6_smoke_e2e.py.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python -m pytest -q -p no:cacheprovider \
        src/test/20261125_artelingo_held_test/test_r6_smoke.py
"""
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_descriptive as D  # noqa: E402
import run_r6_descriptive as RD  # noqa: E402
import run_r6_dts as RDTS  # noqa: E402
import run_r6_held as RH  # noqa: E402
import run_r6_smoke as SM  # noqa: E402
import test_r6_apply_rule as TA  # noqa: E402  (the apply step's results-tree harness)
import test_r6_held as TH  # noqa: E402  (stand-in records, ledger, stubs of the held runner)

import numpy as np  # noqa: E402

DECIMAL = re.compile(r"\d*\.\d+")
CHAIN = [name for name, _, _, _ in SM.SEED42_CHAIN]


# ---------------------------------------------------------------- helpers

def r6_copy(tmp_path) -> Path:
    """A copy of the files r6_module_shas() hashes, at their paths relative to a tmp checkout root."""
    root = HERE.parents[2]
    here = tmp_path / "checkout" / HERE.relative_to(root)
    here.mkdir(parents=True)
    (here.parents[2] / "scripts").mkdir()
    for f in sorted(HERE.glob("r6_*.py")) + sorted(HERE.glob("run_r6_*.py")) + [HERE / "dts_settings.json"]:
        shutil.copy2(f, here / f.name)
    for f in sorted((root / "scripts").glob("run_r6_*.sh")) + [root / "scripts/das6_sync_r6.py"]:
        shutil.copy2(f, here.parents[2] / "scripts" / f.name)
    assert R.r6_module_shas(here) == R.r6_module_shas()
    return here


def touch(path):
    with open(path, "a") as f:
        f.write("# changed after the stage ran\n")


def write_json(path, rec):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=1))


def gpu_fp(job, *names, here=HERE):
    return {"job": job, "scripts_sha256": {n: R.sha256_file(Path(here) / n) for n in names}}


def write_dts42(res, shas=None, here=HERE, stop=False):
    """The seed-42 DTS records as run_r6_dts.py writes the fields step 1 reads: the current module SHA-256s, the GPU
    outputs' fingerprints, a passed sanity, a stop record (built, no stop) bound to the chosen record's bytes."""
    shas = dict(shas or R.r6_module_shas(here))
    base = {"seed": R.DEV_SEED, "module_sha256": shas, "time": "2026-10-10 08:00:00"}
    lis = {"list_1": gpu_fp("r6_gpu_listing", "r6_gpu_listing.py", "r6_gpu_common.py", here=here)}
    ver = {"verb_1": gpu_fp("r6_gpu_verbalise", "r6_gpu_verbalise.py", "r6_gpu_common.py", here=here)}
    write_json(res / RDTS.SANITY, {**base, "stage": "sanity", "passed": True, "gpu_fingerprints": {"listing": lis}})
    write_json(res / RDTS.TUNE, {**base, "stage": "tune", "gpu_fingerprints": {"verbaliser": ver, "listing": lis}})
    chosen = res / RDTS.chosen_name(R.DEV_SEED)
    write_json(chosen, {**base, "stage": "chosen", "gpu_fingerprints": {"verbaliser": ver, "listing": lis}})
    write_json(res / RDTS.STOP, {**base, "stage": "stop", "stop": stop, "built": True,
                                 "input_sha256": {chosen.name: R.sha256_file(chosen)}})
    return res


def records(tmp_path, here=HERE, name="results"):
    shas = R.r6_module_shas(here)
    res = TH.write_records(tmp_path / name, shas=shas, smoke_names=())
    return write_dts42(res, shas, here)


def check(res, capsys, here=HERE):
    code = SM.run("check", results=res, here=here)
    return code, capsys.readouterr().out


def rerun_list(out) -> list:
    """The stage names of a step-1 refusal, in order."""
    return re.findall(r"^  ([A-Za-z0-9 -]+): ", out, re.M)


# ---------------------------------------------------------------- kinds, record names, consumers

def test_kinds_name_the_folders_and_records_each_consumer_reads(tmp_path):
    res = tmp_path / "results"
    want = {"smoke": ("smoke", "smoke_record.json"), "crash1": ("smoke/crash1", "smoke_record_crash1.json"),
            "fix1": ("smoke/fix1", "smoke_record_fix1.json"), "reserve": ("smoke/reserve", "smoke_record_reserve.json")}
    for kind, (folder, record) in want.items():
        p = SM.paths_of(kind, res)
        assert p.smoke == res / folder and p.record == res / record and p.kind.record == record
        sub = p.kind.subdir
        assert (R.SMOKE / (sub or "")) == RD.out_dir(True, sub) == SM.paths_of(kind, R.RESULTS).smoke
        assert AR_out(sub) == SM.paths_of(kind, R.RESULTS).smoke                 # the apply step's smoke folder
    assert [SM.kind_of(after_crash=True), SM.kind_of(fix=1), SM.kind_of(reserve=True), SM.kind_of()] == \
        ["crash1", "fix1", "reserve", "smoke"]
    with pytest.raises(SystemExit):
        SM.main(["--stage", "check", "--fix", "1", "--reserve"])
    # the latest record each consumer reads
    res.mkdir()
    for name in ("smoke_record.json", "smoke_record_crash1.json", "smoke_record_fix1.json",
                 "smoke_record_reserve.json"):
        (res / name).write_text("{}")
    assert RH.latest_smoke_record(res, "held").name == "smoke_record_fix1.json"
    assert RH.latest_smoke_record(res, "fix1").name == "smoke_record_fix1.json"
    assert RH.latest_smoke_record(res, "reserve").name == "smoke_record_reserve.json"
    for mod in (TA.AR, RD):
        assert mod.latest_smoke_record(res, False).name == "smoke_record_fix1.json"
        assert mod.latest_smoke_record(res, True).name == "smoke_record_reserve.json"
    (res / "smoke_record_fix1.json").rename(tmp_path / "kept_fix1.json")
    assert RH.latest_smoke_record(res, "held", after_crash=True).name == "smoke_record_crash1.json"
    for mod in (TA.AR, RD):
        assert mod.latest_smoke_record(res, False).name == "smoke_record_crash1.json"
    (res / "smoke_record_crash1.json").rename(tmp_path / "kept_crash1.json")
    assert RH.latest_smoke_record(res, "held", after_crash=True).name == "smoke_record.json"
    for mod in (TA.AR, RD):
        assert mod.latest_smoke_record(res, False).name == "smoke_record.json"


def AR_out(sub):
    return TA.AR.out_dir(True, sub)


def test_the_record_covers_every_r6_file():
    root = HERE.parents[2]
    files = (sorted(HERE.glob("r6_*.py")) + sorted(HERE.glob("run_r6_*.py")) + [HERE / "dts_settings.json"]
             + sorted((root / "scripts").glob("run_r6_*.sh")) + [root / "scripts/das6_sync_r6.py"])
    keys = {f.relative_to(root).as_posix() for f in files}
    assert set(R.r6_module_shas()) == keys and len(keys) >= 30
    assert {"src/test/20261125_artelingo_held_test/run_r6_smoke.py", "src/test/20261125_artelingo_held_test/"
            "run_r6_held.py", "src/test/20261125_artelingo_held_test/dts_settings.json"} <= keys
    assert not any("test_r6_" in k for k in keys)


# ---------------------------------------------------------------- step 1

def test_step1_passes_when_every_stage_is_current(tmp_path, capsys):
    res = records(tmp_path)
    code, out = check(res, capsys)
    assert code == 0, out
    assert "step 1: refit, picks, regression, sensitivity and the seed-42 DTS records exist and are current" in out
    assert not DECIMAL.search(out.replace(str(tmp_path), "<tmp>"))


@pytest.mark.parametrize("missing", [f for _, f, _, _ in SM.SEED42_CHAIN] + list(SM.DTS42))
def test_step1_refuses_a_missing_stage_output_and_lists_what_must_rerun(tmp_path, capsys, missing):
    res = records(tmp_path)
    (res / missing).rename(tmp_path / "moved.json")
    code, out = check(res, capsys)
    assert code == SM.EXIT_REFUSE and "refused: these stages must rerun before the smoke" in out, out
    names = rerun_list(out)
    files = [f for _, f, _, _ in SM.SEED42_CHAIN]
    if missing in files:
        k = files.index(missing)
        assert names == CHAIN[k:], names                       # the stage and every later one, in order
        assert f"{missing} does not exist" in out
        assert k >= 2 or f"it follows {CHAIN[k]}" in out          # (the sensitivity file names its own reason)
    else:
        assert names == ["DTS seed 42"] and f"{missing} does not exist" in out


def test_step1_refuses_stages_run_by_other_module_bytes(tmp_path, capsys):
    """A tmp copy of the r6 files stands in for this folder; the records were written by its bytes. One file changed
    in the copy: every stage whose runner ran it (through imports) must rerun, with the stages after it."""
    here = r6_copy(tmp_path)
    res = records(tmp_path, here)
    assert check(res, capsys, here)[0] == 0
    touch(here / "r6_heads.py")
    code, out = check(res, capsys, here)
    first = next(n for n, _, runner, _ in SM.SEED42_CHAIN if "r6_heads.py" in RH.ran_modules(runner, HERE))
    want = CHAIN[CHAIN.index(first):] + (["DTS seed 42"] if "r6_heads.py" in RH.ran_modules("run_r6_dts.py", HERE)
                                         else [])
    assert code == SM.EXIT_REFUSE and rerun_list(out) == want, out
    key = (here / "r6_heads.py").relative_to(here.parents[2]).as_posix()
    assert f"was written by other bytes of ['{key}']" in out

    here2 = r6_copy(tmp_path / "b")
    res2 = records(tmp_path / "b", here2)
    touch(here2 / "r6_dts.py")
    code, out = check(res2, capsys, here2)
    want = [n for n, _, runner, _ in SM.SEED42_CHAIN if "r6_dts.py" in RH.ran_modules(runner, HERE)]
    assert code == SM.EXIT_REFUSE and rerun_list(out) == (CHAIN[CHAIN.index(want[0]):] if want else []) + [
        "DTS seed 42"], out


def test_step1_refuses_dts_records_of_other_settings_or_gpu_script_bytes(tmp_path, capsys):
    here = r6_copy(tmp_path)
    res = records(tmp_path, here)
    (here / "dts_settings.json").write_text((here / "dts_settings.json").read_text() + "\n")
    code, out = check(res, capsys, here)
    assert code == SM.EXIT_REFUSE and rerun_list(out) == ["DTS seed 42"] and "dts_settings.json" in out, out
    here2 = r6_copy(tmp_path / "b")
    res2 = records(tmp_path / "b", here2)
    gpu_only = "r6_gpu_verbalise.py"
    assert gpu_only not in RH.ran_modules("run_r6_dts.py", HERE)             # only its fingerprint names it
    touch(here2 / gpu_only)
    code, out = check(res2, capsys, here2)
    assert code == SM.EXIT_REFUSE and rerun_list(out) == ["DTS seed 42"], out
    assert f"made by other bytes of ['{gpu_only}']" in out


def test_step1_refuses_a_sensitivity_file_of_another_regression_and_failed_records(tmp_path, capsys):
    res = records(tmp_path)
    TH.edit(res / RH.REGRESSION_NAME, time="2026-10-11 09:00:00")           # rewritten: other bytes
    code, out = check(res, capsys)
    assert code == SM.EXIT_REFUSE and rerun_list(out) == ["sensitivity"], out
    assert "was not written from the current regression_seed42.json" in out
    res = records(tmp_path, name="r2")
    TH.edit(res / RH.PICKS_NAME, passed=False)
    code, out = check(res, capsys)
    assert rerun_list(out) == CHAIN[1:] and "picks_seed42.json did not pass" in out, out
    res = records(tmp_path, name="r3")
    TH.edit(res / RDTS.SANITY, passed=False)
    code, out = check(res, capsys)
    assert rerun_list(out) == ["DTS seed 42"] and "dts_sanity.json did not pass" in out, out
    res = records(tmp_path, name="r4")
    TH.edit(res / RDTS.chosen_name(R.DEV_SEED), time="2026-10-11 09:00:00")   # the stop saw other chosen bytes
    code, out = check(res, capsys)
    assert rerun_list(out) == ["DTS seed 42"] and "was not evaluated on the current dts_seed42.json" in out, out


def test_step1_a_dts_stop_goes_to_the_user(tmp_path, capsys):
    res = write_dts42(records(tmp_path), stop=True)
    code, out = check(res, capsys)
    assert code == SM.EXIT_REFUSE and "the user decides" in out and "must rerun" not in out, out


def test_later_stages_refuse_a_smoke_made_by_other_bytes(tmp_path, capsys):
    res = records(tmp_path)
    paths = SM.paths_of("smoke", res)
    code = SM.run("list-input", results=res, outs={"verbalise": [tmp_path / "v"]})
    out = capsys.readouterr().out
    assert code == SM.EXIT_REFUSE and "run --stage gpu-inputs first" in out, out
    shas = R.r6_module_shas()
    write_json(paths.smoke / "held_pass.json", {"mode": "smoke", "module_sha256": shas})
    write_json(paths.state, {"module_sha256": {**shas, sorted(shas)[0]: "0" * 64}, "started": "x"})
    for stage in ("list-input", "finish"):
        code = SM.run(stage, results=res)
        out = capsys.readouterr().out
        assert code == SM.EXIT_REFUSE and "the r6 files changed since the held smoke" in out, (stage, out)
    write_json(paths.record, {"passed": True})
    assert SM.run("finish", results=res) == SM.EXIT_REFUSE
    assert "a smoke record is never replaced" in capsys.readouterr().out
    assert SM.run("gpu-inputs", results=res) == SM.EXIT_REFUSE              # before any data is loaded
    assert "a smoke record is never replaced" in capsys.readouterr().out


# ---------------------------------------------------------------- step 6, the leak check

CLEAN = """run_r6_smoke --stage finish (smoke) 2026-10-10 21:04:11: smoke folder /x/results/smoke
[21:04:12] [run_r6_held] data and split [6s]
smoke held pass written 4f1c2d0e5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d
dts tune: pass, chose W2 K8; record /x/results/smoke/dts/dts_tune.json
[run_r6_descriptive] seed 9001: external rows DTS, DTS-CF, DTS-N, FT-LP, FT-LB, FT-LoRA, MLLM; missing none
cluster launch --node <node> -- bash scripts/run_r6_ftfeat.sh smoke_ft --ckpt LB_lr3e-5=/x/best_params.pt
"""


def test_the_leak_check_catches_a_planted_decimal_and_metric_values(tmp_path):
    log = tmp_path / "clean.log"
    log.write_text(CLEAN)
    assert SM.leak_hits(CLEAN) == [] and SM.leak_check([log])["passed"] is True
    for planted, kind in (("18.3", "decimal number"), (".5", "decimal number"), ("AFF R@1 19", "R@1 value"),
                          ('"r1": 0', "metric key with a value"), ("gain = -3", "metric key with a value"),
                          ("point: [1, 2]", "metric key with a value")):
        copy = tmp_path / f"copy_{len(list(tmp_path.iterdir()))}.log"
        copy.write_text(CLEAN + f"P3 margin {planted} here\n" + "last line\n")
        rec = SM.leak_check([log, copy])
        assert rec["passed"] is False, planted
        assert {"file": SM.rel(copy), "line": CLEAN.count("\n") + 1, "kind": kind} in rec["hits"], (planted, rec)
        assert planted not in json.dumps(rec)                                 # the hit names no value


# ---------------------------------------------------------------- the smoke record against today's bytes

def test_the_held_runner_refuses_when_an_r6_file_changes_after_the_smoke(tmp_path, monkeypatch, capsys):
    """run_r6_held.py --mode held asserts the latest smoke record's module_sha256 (rule section 8 item 1): a tmp copy of
    the r6 files, the record written from it as run_r6_smoke.py writes module_sha256; one changed file is refused."""
    here = r6_copy(tmp_path)
    res = TH.write_records(tmp_path / "results", shas=R.r6_module_shas(here), smoke_names=())
    write_json(res / "smoke_record.json", {"passed": True, "kind": "smoke", "module_sha256": R.r6_module_shas(here),
                                           "time": "2026-10-10 21:00:00"})
    led = TH.write_ledger(tmp_path / "held_ledger.md", TH.ledger_line())
    TH.install(monkeypatch, stop_at={"inputs"})

    def held():
        return RH.run_held(results=res, ledger=led, here=here, folder=tmp_path / "folder")
    with pytest.raises(TH.Reached):                                       # the copy unchanged: past the smoke guard
        held()
    capsys.readouterr()
    for name in ("r6_episodes.py", "run_r6_smoke.py", "dts_settings.json"):
        touch(here / name)
        key = (here / name).relative_to(here.parents[2]).as_posix()
        assert held() == RH.EXIT_REFUSE
        out = capsys.readouterr().out
        assert f"smoke_record.json: the SHA-256s of ['{key}'] differ from the smoke's" in out, out
        write_json(res / "smoke_record.json", {"passed": True, "module_sha256": R.r6_module_shas(here)})
    touch(here.parents[2] / "scripts/run_r6_rerank.sh")
    assert held() == RH.EXIT_REFUSE and "scripts/run_r6_rerank.sh" in capsys.readouterr().out


def test_the_apply_step_refuses_when_an_r6_file_changes_after_the_smoke(tmp_path, monkeypatch, capsys):
    here = r6_copy(tmp_path)
    monkeypatch.setattr(TA.AR, "SHA_HERE", here)
    e = TA.Env(tmp_path / "a", monkeypatch)                    # smoke records of today's bytes (= the copy's)
    e.stage(TA.go_rec())
    assert e.run() == 0 and e.verdict()["smoke_record"]["name"] == "smoke_record.json"
    capsys.readouterr()
    touch(here / "r6_stats.py")
    e = TA.Env(tmp_path / "b", monkeypatch)
    e.stage(TA.go_rec())
    assert e.run() == 4
    key = (here / "r6_stats.py").relative_to(here.parents[2]).as_posix()
    assert f"smoke_record.json: the SHA-256s of ['{key}'] differ from the smoke's" in capsys.readouterr().out
    assert not (e.results / "held_verdict.json").exists()
    e.stage(TA.go_rec(smoke=True), smoke=True)                           # smoke mode reads no record
    assert e.run("--smoke") == 0


def test_the_descriptive_pass_refuses_when_an_r6_file_changes_after_the_smoke(tmp_path):
    here, out = r6_copy(tmp_path), tmp_path / "results"
    write_json(out / "smoke_record.json", {"passed": True, "module_sha256": R.r6_module_shas(here)})
    write_json(out / "smoke_record_reserve.json", {"passed": True, "module_sha256": R.r6_module_shas(here)})
    assert RD.smoke_guard(out, False, False, here=here)["file"] == "smoke_record.json"
    assert RD.smoke_guard(out, False, True, here=here)["file"] == "smoke_record_reserve.json"
    touch(here / "r6_external.py")
    for reserve, name in ((False, "smoke_record.json"), (True, "smoke_record_reserve.json")):
        with pytest.raises(RD.Refused, match=f"{name}: the SHA-256s of .*r6_external.py.* differ from the smoke's"):
            RD.smoke_guard(out, False, reserve, here=here)
    assert RD.smoke_guard(out, True, False, here=here) is None


# ---------------------------------------------------------------- held_arrays: T08's writer, T13's reader

def test_held_arrays_round_trip_in_the_three_namings(tmp_path, monkeypatch, capsys):
    """The held runner's own writes (stubbed bundles and scores of the real shapes: 3 seeds x 12,288 episodes) as the
    first read, the --fix 1 pass and the --reserve read: held_arrays.npz, held_arrays_fix1.npz and
    held_arrays_reserve.npz are each accepted by r6_descriptive.load_core (keys, dtypes, seed_index as the seed's
    position) and reproduce their pass file (check_pass), byte for byte the layout of arrays_from_scored."""
    c = SimpleNamespace(res=TH.write_records(tmp_path / "results"), folder=tmp_path / "folder", tmp=tmp_path,
                        ledger=TH.write_ledger(tmp_path / "held_ledger.md", TH.ledger_line()))
    ns = TH.install(monkeypatch)
    assert TH.held(c) == 0
    TH.write_ledger(c.ledger, TH.ledger_line(episodes=TH.all_hashes()))
    TH.write_records(c.res, smoke_names=("smoke_record.json", "smoke_record_fix1.json"))
    assert TH.held(c, fix=1) == 0
    (c.res / "held_verdict.json").write_text("{}")
    TH.write_ledger(c.ledger, TH.ledger_line(report="abc", episodes=TH.all_hashes()), TH.ledger_line("H5-R"))
    TH.write_records(c.res, smoke_names=("smoke_record.json", "smoke_record_reserve.json"))
    assert TH.held(c, reserve=True) == 0
    capsys.readouterr()
    seeds = list(R.HELD_SEEDS)
    want = D.arrays_from_scored([ns.scored[s] for s in seeds], seeds)
    for kind in ("held", "fix1", "reserve"):
        n = RH.read_names(kind)
        assert n.arrays == {"held": "held_arrays.npz", "fix1": "held_arrays_fix1.npz",
                            "reserve": "held_arrays_reserve.npz"}[kind]
        with np.load(c.res / n.arrays) as z:
            got = {k: z[k] for k in z.files}
        assert set(got) == set(want) == set(D.array_keys())
        for k in want:
            assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k]), (kind, k)
        assert np.array_equal(got["seed_index"], np.repeat(np.arange(3, dtype=np.int64), 3 * R.N_PER_PAIR))
        core = D.load_core(c.res / n.arrays, seeds, R.N_PER_PAIR)
        assert sorted(core) == seeds
        pr = json.loads((c.res / n.pass_).read_text())
        assert pr["seeds"] == seeds and pr["kind"] == kind
        D.check_pass(core, pr)                                                # raises on any difference
        assert pr["outputs"][n.arrays] == R.sha256_file(c.res / n.arrays)


# ---------------------------------------------------------------- step 5, the wiring mutation's text

def test_the_wiring_mutation_replaces_the_one_cf_line_and_the_copy_compiles(tmp_path):
    src = (HERE / "r6_score.py").read_text()
    assert src.count(SM.WIRING_FROM) == 1 and "def cf_terms(gated)" in src
    mutated = src.replace(SM.WIRING_FROM, SM.WIRING_TO)
    path = tmp_path / "r6_score_wired.py"
    path.write_text(mutated)
    compile(mutated, str(path), "exec")
    spec = importlib.util.spec_from_file_location("r6_score_wired", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import inspect
    body = inspect.getsource(mod.frozen_cells)
    assert "G = gated" in body and "G = cf_terms(gated)" not in body and "# guard:cf_cell" in body
    assert compile(SM.DRIVER, "smoke_scoring.py", "exec")
