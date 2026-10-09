"""Final review A, check 4: rule section 8.1 and 9 refusals of the held runner end to end through run_r6_held.run_held
(the function main() calls; only its paths are pointed at a temp folder), my own scenarios. A run that passes every
refusal stops at head_guard (a stand-in env whose head check failed), so "reached heads" means "not refused before the
read". Picks and lambdas are stubbed (not this area); inputs, recorded hashes, ledger, started files, smoke records and
seed-42 records are real code on files in the temp folder.

    python final_review/fr_a_guards.py <tmp dir>      -> fr_a_guards.json beside this file, one summary line
"""
import contextlib
import hashlib
import io
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
import r6_common as R  # noqa: E402
import run_r6_held as RH  # noqa: E402

TMP = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fr_a_guards")
OUT = Path(__file__).with_name("fr_a_guards.json")
RH.P.load_picks = lambda path: {}
RH.P.frozen_lambdas = lambda: {}
RUNNER = R.sha256_file(F / "run_r6_held.py")
HEAD = ["# Held-out ledger", "", "| # | Date | Dataset and split | Purpose | Episode / data SHA-256 | Script SHA-256 | Report |",
        "|---|---|---|---|---|---|---|"]
SIG = {c: {"sigma_a2": 4.0, "sigma_eps2": 1500.0} for c in ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "S1", "S2")}
COEF = {"affect": {"img": "a" * 64}}


def h(*p):
    return hashlib.sha256("|".join(map(str, p)).encode()).hexdigest()


def nine(salt=""):
    return {str(s): {p: h(s, p, salt) for p in R.PAIR_NAMES} for s in R.HELD_SEEDS}


def row(rid="H5", eps="(pending)", scripts=None, report="(pending)"):
    sc = ", ".join(scripts if scripts is not None else [RUNNER])
    return f"| {rid} | 2026-10-12 | ArtELingo held rows, seeds 52 to 54 | AFF held read | {eps} | {sc} | {report} |"


def records(res, shas=None, smoke=("smoke_record.json",)):
    shas = dict(shas or R.r6_module_shas())
    res.mkdir(parents=True, exist_ok=True)
    meta = {"module_sha256": shas, "time": "2026-10-12 09:00:00"}
    (res / "refit_check.json").write_text(json.dumps({"passed": True, "coef_sha256": COEF, **meta}))
    (res / "picks_seed42.json").write_text(json.dumps({"passed": True, "coef_sha256": COEF, **meta}))
    (res / "regression_seed42.json").write_text(json.dumps({"passed": True, **meta}))
    (res / "sensitivity_seed42.json").write_text(json.dumps(
        {**SIG, "regression_sha256": R.sha256_file(res / "regression_seed42.json"), **meta}))
    for n in smoke:
        (res / n).write_text(json.dumps({"passed": True, "module_sha256": shas}))


def started(res, folder, *attempts, name="held_started.json", copy=True):
    att = [{"attempt": i + 1, "time": "t", "flags": {"after_crash": False, "fix": None, "reserve": False,
                                                     "smoke": False, **fl}, "episodes_sha256": hs}
           for i, (fl, hs) in enumerate(attempts)]
    raw = json.dumps({"rule_sha256": R.RULE_SHA256, "attempts": att}).encode()
    (res / name).write_bytes(raw)
    if copy:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(raw)


def run_case(name, prep, kw=None, ledger_rows=None):
    d = TMP / name
    if d.exists():
        shutil.rmtree(d)
    res, folder = d / "results", d / "folder"
    records(res)
    folder.mkdir(parents=True)
    rows = ledger_rows if ledger_rows is not None else [row()]
    prep(SimpleNamespace(res=res, folder=folder, d=d))
    led = d / "held_ledger.md"
    if not led.exists():
        led.write_text("\n".join(HEAD + rows) + "\n")
    env = SimpleNamespace(head_check={"passed": False}, coef_sha256=COEF)
    before = {p.relative_to(d).as_posix(): p.read_bytes() for p in d.rglob("*") if p.is_file()}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = RH.run_held(results=res, ledger=led, here=F, folder=folder, env=env, **(kw or {}))
    after = {p.relative_to(d).as_posix(): p.read_bytes() for p in d.rglob("*") if p.is_file()}
    changed = sorted(k for k in after if before.get(k) != after[k])
    msg = buf.getvalue().strip().splitlines()[-1] if buf.getvalue().strip() else ""
    return {"code": code, "msg": msg[:600], "reached_heads": "refit heads' selection posteriors" in msg,
            "files_changed": changed}


def stale(res, fname, key=None):
    p = res / fname
    r = json.loads(p.read_text())
    k = key or "src/test/20261125_artelingo_held_test/r6_stats.py"
    r["module_sha256"][k] = "0" * 64
    p.write_text(json.dumps(r))


def main():
    if TMP.exists():
        shutil.rmtree(TMP)
    TMP.mkdir(parents=True)
    nop = (lambda c: None)
    one = ({}, nine())
    C = {}   # name -> (prep, kwargs, ledger rows, expect: "heads" or a substring of the refusal)
    C["control_first_read"] = (nop, {}, None, "heads")
    C["verdict_exists"] = (lambda c: (c.res / "held_verdict.json").write_text("{}"), {}, None, "held_verdict.json exists")
    C["pass_exists"] = (lambda c: (c.res / "held_pass.json").write_text("{}"), {}, None, "held_pass.json exists")
    C["pass_exists_after_crash"] = (lambda c: ((c.res / "held_pass.json").write_text("{}"),
                                               started(c.res, c.folder, one)), {"after_crash": True}, None,
                                    "held_pass.json exists")
    C["started_exists_no_flag"] = (lambda c: started(c.res, c.folder, one), {}, None, "held_started.json exists")
    C["after_crash_one_attempt"] = (lambda c: started(c.res, c.folder, one), {"after_crash": True}, None, "heads")
    C["after_crash_partial_first_attempt"] = (lambda c: started(c.res, c.folder, ({}, {"52": nine()["52"]})),
                                              {"after_crash": True}, None, "heads")
    C["after_crash_two_attempts"] = (lambda c: started(c.res, c.folder, one, ({"after_crash": True}, nine())),
                                     {"after_crash": True}, None, "already records 2 attempts")
    C["after_crash_without_started"] = (nop, {"after_crash": True}, None, "does not exist")
    C["after_crash_copy_differs"] = (lambda c: (started(c.res, c.folder, one),
                                                (c.folder / "held_started.json").write_text("{}")),
                                     {"after_crash": True}, None, "missing or differs")
    C["after_crash_copy_missing"] = (lambda c: started(c.res, c.folder, one, copy=False), {"after_crash": True}, None,
                                     "missing or differs")
    C["h5_missing"] = (nop, {}, [row("H4")], "ledger row H5 is missing")
    C["h5_twice"] = (nop, {}, [row(), row()], "appears 2 times")
    C["h5_other_sha"] = (nop, {}, [row(scripts=["e" * 64])], "latest script SHA-256 is not")
    C["h5_this_then_other"] = (nop, {}, [row(scripts=[RUNNER, "e" * 64])], "latest script SHA-256 is not")
    C["h5_other_then_this"] = (nop, {}, [row(scripts=["e" * 64, RUNNER])], "heads")
    C["h5_sha_uppercase"] = (nop, {}, [row(scripts=[RUNNER.upper()])], "latest script SHA-256 is not")
    C["h5_report_filled"] = (nop, {}, [row(report="[report](x.md) " + "f" * 64)], "report cell")
    C["h5_report_pending_no_parens"] = (nop, {}, [row(report="pending")], "report cell")
    C["h5_episode_cell_filled_first_read"] = (nop, {}, [row(eps=", ".join(h for s in nine().values()
                                                                          for h in s.values()))], "episode cell")
    C["copy_exists_first_read"] = (lambda c: (c.folder / "held_started.json").write_text("{}"), {}, None,
                                   "committed copy")
    C["smoke_record_missing"] = (lambda c: (c.res / "smoke_record.json").unlink(), {}, None, "does not exist")
    C["smoke_record_failed"] = (lambda c: (c.res / "smoke_record.json").write_text(
        json.dumps({"passed": False, "module_sha256": R.r6_module_shas()})), {}, None, "did not pass")
    C["smoke_record_stale_sha"] = (lambda c: stale(c.res, "smoke_record.json"), {}, None, "a new smoke is needed")
    C["smoke_record_lacks_a_module"] = (lambda c: (lambda r: (r["module_sha256"].pop(
        "src/test/20261125_artelingo_held_test/r6_stats.py"), (c.res / "smoke_record.json").write_text(json.dumps(r))))(
        json.loads((c.res / "smoke_record.json").read_text())), {}, None, "a new smoke is needed")
    C["smoke_record_extra_module"] = (lambda c: (lambda r: (r["module_sha256"].update({"x/r6_old.py": "1" * 64}),
                                                            (c.res / "smoke_record.json").write_text(json.dumps(r))))(
        json.loads((c.res / "smoke_record.json").read_text())), {}, None, "a new smoke is needed")
    C["crash1_record_used_for_after_crash"] = (
        lambda c: (started(c.res, c.folder, one), (c.res / "smoke_record_crash1.json").write_text(
            json.dumps({"passed": True, "module_sha256": {"x": "0" * 64}}))), {"after_crash": True}, None,
        "smoke_record_crash1.json")
    C["fix1_record_counts_for_first_read_when_present"] = (
        lambda c: (c.res / "smoke_record_fix1.json").write_text(json.dumps({"passed": True, "module_sha256": {}})),
        {}, None, "smoke_record_fix1.json")
    C["regression_stale"] = (lambda c: (stale(c.res, "regression_seed42.json"), (lambda r: (
        r.update(regression_sha256=R.sha256_file(c.res / "regression_seed42.json")),
        (c.res / "sensitivity_seed42.json").write_text(json.dumps(r))))(
        json.loads((c.res / "sensitivity_seed42.json").read_text()))), {}, None, "rerun the regression")
    C["regression_failed"] = (lambda c: (lambda r: (r.update(passed=False), (c.res / "regression_seed42.json")
                                                     .write_text(json.dumps(r))))(
        json.loads((c.res / "regression_seed42.json").read_text())), {}, None, "did not pass")
    C["sens42_not_from_current_regression"] = (lambda c: (lambda r: (r.update(regression_sha256="0" * 64), (
        c.res / "sensitivity_seed42.json").write_text(json.dumps(r))))(
        json.loads((c.res / "sensitivity_seed42.json").read_text())), {}, None, "was not written from the current")
    C["refit_stale"] = (lambda c: stale(c.res, "refit_check.json", "src/test/20261125_artelingo_held_test/r6_heads.py"),
                        {}, None, "rerun run_r6_refit.py")
    # --fix 1
    fixrow = [row(eps=", ".join(x for s in nine().values() for x in s.values()))]

    def fix_ready(c, extra=()):
        started(c.res, c.folder, one)
        (c.res / "held_pass.json").write_text("{}")
        (c.res / "smoke_record_fix1.json").write_text(json.dumps({"passed": True, "module_sha256": R.r6_module_shas()}))
        for f in extra:
            f(c)
    C["fix1_ready"] = (fix_ready, {"fix": 1}, fixrow, "heads")
    C["fix1_without_pass"] = (lambda c: fix_ready(c, [lambda c: (c.res / "held_pass.json").unlink()]), {"fix": 1},
                              fixrow, "which does not exist")
    C["fix1_pass_fix1_exists"] = (lambda c: fix_ready(c, [lambda c: (c.res / "held_pass_fix1.json").write_text("{}")]),
                                  {"fix": 1}, fixrow, "held_pass_fix1.json exists")
    C["fix1_without_its_smoke"] = (lambda c: fix_ready(c, [lambda c: (c.res / "smoke_record_fix1.json").unlink()]),
                                   {"fix": 1}, fixrow, "smoke_record_fix1.json is missing")
    C["fix1_verdict_exists"] = (lambda c: fix_ready(c, [lambda c: (c.res / "held_verdict.json").write_text("{}")]),
                                {"fix": 1}, fixrow, "held_verdict.json exists")
    C["fix1_h5_episode_cell_pending"] = (fix_ready, {"fix": 1}, [row()], "nine episode SHA-256s")
    C["fix1_h5_episode_cell_other"] = (fix_ready, {"fix": 1},
                                       [row(eps=", ".join(x for s in nine("o").values() for x in s.values()))],
                                       "nine episode SHA-256s")
    C["fix1_second"] = (lambda c: (fix_ready(c), started(c.res, c.folder, one, ({"fix": 1}, nine()))), {"fix": 1},
                        fixrow, "already records a --fix 1")
    C["fix1_stale_smoke_fix1"] = (lambda c: fix_ready(c, [lambda c: stale(c.res, "smoke_record_fix1.json")]),
                                  {"fix": 1}, fixrow, "a new smoke is needed")
    # --reserve
    rrow = [row(eps="x"), row("H5-R")]

    def res_ready(c, extra=()):
        started(c.res, c.folder, one)
        (c.res / "held_verdict.json").write_text("{}")
        (c.res / "smoke_record_reserve.json").write_text(json.dumps({"passed": True,
                                                                      "module_sha256": R.r6_module_shas()}))
        for f in extra:
            f(c)
    C["reserve_ready"] = (res_ready, {"reserve": True}, rrow, "heads")
    C["reserve_without_h5r"] = (res_ready, {"reserve": True}, [row(eps="x")], "ledger row H5-R is missing")
    C["reserve_without_verdict"] = (lambda c: res_ready(c, [lambda c: (c.res / "held_verdict.json").unlink()]),
                                    {"reserve": True}, rrow, "held_verdict.json exists, and it does not")
    C["reserve_started_exists"] = (lambda c: res_ready(c, [lambda c: started(c.res, c.folder, one,
                                                                             name="held_started_reserve.json")]),
                                   {"reserve": True}, rrow, "one reserve read only")
    C["reserve_verdict_exists"] = (lambda c: res_ready(c, [lambda c: (c.res / "held_verdict_reserve.json")
                                                           .write_text("{}")]), {"reserve": True}, rrow,
                                   "held_verdict_reserve.json exists")
    C["reserve_pass_exists"] = (lambda c: res_ready(c, [lambda c: (c.res / "held_pass_reserve.json").write_text("{}")]),
                                {"reserve": True}, rrow, "held_pass_reserve.json exists")
    C["reserve_without_its_smoke"] = (lambda c: res_ready(c, [lambda c: (c.res / "smoke_record_reserve.json")
                                                              .unlink()]), {"reserve": True}, rrow, "does not exist")
    C["reserve_partial_hashes"] = (lambda c: res_ready(c, [lambda c: started(c.res, c.folder,
                                                                             ({}, {"52": nine()["52"]}))]),
                                   {"reserve": True}, rrow, "nine held episode")
    C["reserve_copy_exists"] = (lambda c: res_ready(c, [lambda c: (c.folder / "held_started_reserve.json")
                                                        .write_text("{}")]), {"reserve": True}, rrow, "committed copy")
    C["reserve_h5r_report_filled"] = (res_ready, {"reserve": True}, [row(eps="x"), row("H5-R", report="done")],
                                      "report cell")
    C["flags_two_at_once"] = (nop, {"after_crash": True, "reserve": True}, None, "at most one of")
    C["fix_2"] = (nop, {"fix": 2}, None, "at most one of")
    out = {}
    for name, (prep, kw, rows, expect) in C.items():
        r = run_case(name, prep, kw, rows)
        ok = (r["reached_heads"] if expect == "heads" else (r["code"] == 4 and expect in r["msg"]
                                                             and not r["reached_heads"]))
        ok = ok and r["code"] == 4 and (r["files_changed"] == [] if expect != "heads" else True)
        out[name] = {**r, "expect": expect, "ok": ok}
    OUT.write_text(json.dumps(out, indent=1))
    bad = {k: (v["code"], v["msg"][:90], v["files_changed"]) for k, v in out.items() if not v["ok"]}
    print(f"guards: {len(out)} scenarios, {len(bad)} not as expected")
    for k, v in bad.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
