"""Scoped re-review (r6 fix wave and T15 fix round): mutations of the new guards, each on a copy in the scratch dir
(never in place). A pytest plugin loads the copy under the module's own name (HERE pinned to this round folder) before
the test file imports it; the targeted tests must fail (pytest exit 1 = caught). Pattern of fr_c_mutate.py.

Usage: rr_mutate.py <scratch dir> [ids...]   -> <scratch>/mut/<id>/pytest.log and final_review/rr_mutate.json
"""
import json
import os
import subprocess
import sys
from pathlib import Path

F = Path(__file__).resolve().parents[1]
OUT = Path(sys.argv[1]) / "mut"
OUT.mkdir(parents=True, exist_ok=True)
PY = sys.executable
HERE_LINE = "HERE = Path(__file__).resolve().parent\n"

MUTS = [  # (id, module, old, new, test file, -k)
    # fix wave item 1 (value_sets.json)
    ("vs_file_guard_off", "run_r6_held", "_refuse_unless(why is None, str(why))  # guard:value_sets_file", "pass",
     "test_r6_held.py", "value_sets"),
    ("vs_equal_sha_only", "run_r6_held", 'R.sha256_bytes(raw) == checked["sha256"] and got == want',
     'R.sha256_bytes(raw) == checked["sha256"]', "test_r6_held.py", "value_sets"),
    ("vs_equal_content_only", "run_r6_held", 'R.sha256_bytes(raw) == checked["sha256"] and got == want',
     "got == want", "test_r6_held.py", "value_sets"),
    ("vs_problem_no_sha", "run_r6_held", "    if want != R.sha256_file(path):", "    if False:",
     "test_r6_held.py", "value_sets"),
    ("vs_regression_not_written", "run_r6_held",
     "    R.write_value_sets(value_sets, env.value_sets, env.data)", "    pass  #",
     "test_r6_regression.py", ""),
    ("vs_regression_sha_missing", "run_r6_held", '"value_sets_sha256": R.sha256_file(value_sets)}',
     '"value_sets_sha256": None}', "test_r6_regression.py", ""),
    ("vs_smoke_chain_off", "run_r6_smoke", "why = RH.value_sets_problem(results)  # guard:value_sets", "why = None",
     "test_r6_smoke.py", "value_sets"),
    ("vs_attempt_not_recorded", "run_r6_held", '"dts_stop": dts_stop, "value_sets": value_sets, "git": '
     'git_provenance(here),', '"dts_stop": dts_stop, "git": git_provenance(here),', "test_r6_held.py", "value_sets"),
    # fix wave item 3 (time-box)
    ("tb_off_by_one_early", "run_r6_held", "today <= READ_DEADLINE", "today < READ_DEADLINE", "test_r6_held.py",
     "time_box"),
    ("tb_one_day_late", "run_r6_held", "today <= READ_DEADLINE",
     "today.toordinal() <= READ_DEADLINE.toordinal() + 1", "test_r6_held.py", "time_box"),
    ("tb_boxes_flags", "run_r6_held", 'if kind == "held" and not after_crash:          # the time-box',
     "if True:          # the time-box", "test_r6_held.py", "time_box"),
    ("tb_clock_utc", "run_r6_held", 'return datetime.now(ZoneInfo("Europe/Amsterdam")).date()',
     'return datetime.now(ZoneInfo("UTC")).date()', "test_r6_held.py", "time_box"),
    # fix wave item 4 (git provenance)
    ("git_not_recorded", "run_r6_held", '"git": git_provenance(here),', "", "test_r6_held.py", "git"),
    # fix wave item 5 (settings copy)
    ("sc_call_removed", "run_r6_dts", "    settings_copy_guard(args)                       # seed 42", "    pass  #",
     "test_r6_dts.py", "settings"),
    ("sc_exists_only", "run_r6_dts",
     "copy.is_file() and copy.read_bytes() == Path(G.SETTINGS_PATH).read_bytes()", "copy.is_file()",
     "test_r6_dts.py", "settings"),
    # T15 fix round
    ("fb_no_fingerprints", "run_r6_dts",
     '            and fb.get("gpu_fingerprints") == built_fingerprints(sanity, chosen))', "            )",
     "test_r6_dts.py", "first_build or budget or stop"),
    ("fb_no_settings_sha", "run_r6_dts",
     '            and fb.get("settings_sha256") == chosen.get("settings_sha256")\n', "",
     "test_r6_dts.py", "first_build or budget or stop"),
    ("fb_no_setting", "run_r6_dts", '            and fb.get("setting") == chosen.get("setting")\n', "",
     "test_r6_dts.py", "first_build or budget or stop"),
    ("fb_written_out_of_budget", "run_r6_dts", '    if not b["within_budget"]:\n        return None',
     "    if False:\n        return None", "test_r6_dts.py", "first_build or budget or stop"),
    ("fb_never_used", "run_r6_dts",
     'built_time = fb["time"] if fb else max((sanity["time"], chosen["time"]), key=D.parse_amsterdam)',
     'built_time = max((sanity["time"], chosen["time"]), key=D.parse_amsterdam)', "test_r6_dts.py",
     "first_build or budget or stop"),
    ("stop_chosen_time_only", "run_r6_dts",
     'built_time = fb["time"] if fb else max((sanity["time"], chosen["time"]), key=D.parse_amsterdam)',
     'built_time = fb["time"] if fb else chosen["time"]', "test_r6_dts.py", "first_build or budget or stop"),
    ("clock_constant_moved", "r6_common", 'DTS_CLOCK_START = "2026-10-09 12:29"', 'DTS_CLOCK_START = "2026-10-09 13:29"',
     "test_r6_dts.py", "clock"),
    ("held_clock_check_off", "run_r6_held", "    if clock != R.DTS_CLOCK_START:", "    if False:",
     "test_r6_held.py", "dts"),
    ("held_inputs_binding_off", "run_r6_held", "    if DTS_CHOSEN_NAME not in inputs or other:", "    if False:",
     "test_r6_held.py or test_r6_smoke.py", "dts"),
    ("smoke_not_given_passes", "run_r6_smoke",
     'steps[f"given_{fam}"] = {"passed": False, "missing": what}  # guard:not_given',
     'steps[f"given_{fam}"] = {"passed": True, "missing": what}', "test_r6_smoke.py", "given or waive"),
    ("smoke_no_retry", "run_r6_smoke", "    for seed in R.SMOKE_SEEDS:\n        code, seeds[seed]",
     "    for seed in R.SMOKE_SEEDS[:1]:\n        code, seeds[seed]", "test_r6_smoke.py", "retr"),
    ("smoke_mutation_any_error", "run_r6_smoke",
     'out["fired"] = out["passed"] = p.returncode != 0 and found  # guard:mutation_fired',
     'out["fired"] = out["passed"] = p.returncode != 0', "test_r6_smoke.py", "mutation"),

    # second pass: the full test files, and the stop's "later of" clause in record_first_built
    ("tb_off_by_one_early_full", "run_r6_held", "today <= READ_DEADLINE", "today < READ_DEADLINE", "test_r6_held.py",
     ""),
    ("tb_clock_utc_full", "run_r6_held", 'return datetime.now(ZoneInfo("Europe/Amsterdam")).date()',
     'return datetime.now(ZoneInfo("UTC")).date()', "test_r6_held.py", ""),
    ("stop_chosen_time_only_full", "run_r6_dts",
     'built_time = fb["time"] if fb else max((sanity["time"], chosen["time"]), key=D.parse_amsterdam)',
     'built_time = fb["time"] if fb else chosen["time"]', "test_r6_dts.py", ""),
    ("fb_chosen_time_only_full", "run_r6_dts",
     '    built = max((sanity["time"], chosen["time"]), key=D.parse_amsterdam)\n',
     '    built = chosen["time"]\n', "test_r6_dts.py", ""),
]


def plugin(mod, path):
    return (f"import importlib.util, sys\nsys.path.insert(0, {str(F)!r})\nimport r6_common\n"
            f"spec = importlib.util.spec_from_file_location({mod!r}, {str(path)!r})\n"
            f"m = importlib.util.module_from_spec(spec)\nsys.modules[{mod!r}] = m\nspec.loader.exec_module(m)\n")


def plugin_common(path):
    """r6_common itself: loaded from the copy before anything imports it."""
    return (f"import importlib.util, sys\nsys.path.insert(0, {str(F)!r})\n"
            f"spec = importlib.util.spec_from_file_location('r6_common', {str(path)!r})\n"
            f"m = importlib.util.module_from_spec(spec)\nsys.modules['r6_common'] = m\nspec.loader.exec_module(m)\n")


def run_one(mid, mod, old, new, tests, k):
    """Run the targeted tests on a copy of ``mod`` with ``old`` replaced by ``new``; -> {test: (exit, failed ids)}."""
    src = (F / f"{mod}.py").read_text()
    if src.count(old) != 1:
        return {"error": f"pattern found {src.count(old)} times"}
    d = OUT / mid
    d.mkdir(parents=True, exist_ok=True)
    copy = d / f"{mod}.py"
    text = src.replace(old, new).replace(HERE_LINE, f"HERE = Path({str(F)!r})\n")
    if mod == "run_r6_held":     # the copy hashes the original runner, so the ledger rows the tests write still match
        pin = "    return R.sha256_file(Path(__file__).resolve())\n"
        assert text.count(pin) == 1
        text = text.replace(pin, "    return R.sha256_file(HERE / 'run_r6_held.py')\n")
    copy.write_text(text)
    (d / f"rr_plugin_{mid}.py").write_text(plugin_common(copy) if mod == "r6_common" else plugin(mod, copy))
    env = dict(os.environ, PYTHONPATH=str(d), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               CUDA_VISIBLE_DEVICES="")
    out = {}
    for test in tests.split(" or "):
        log = d / f"pytest_{test}.log"
        argv = [PY, "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider", "-p", f"rr_plugin_{mid}", "--basetemp",
                str(d / f"pt_{test}"), str(F / test)] + (["-k", k] if k else [])
        with open(log, "w") as fh:
            p = subprocess.run(argv, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(d), timeout=3000)
        text = log.read_text().splitlines()
        failed = sorted({ln.split("::", 1)[1].split(" ")[0] for ln in text
                         if (ln.startswith("FAILED ") or ln.startswith("ERROR ")) and "::" in ln})
        tail = [ln for ln in text if (" passed" in ln or " failed" in ln)][-1:]
        out[test] = {"exit": p.returncode, "failed": failed, "tail": tail}
    return out


res = {}
only = sys.argv[2:] or [m[0] for m in MUTS]
controls = {}
for mid, mod, old, new, tests, k in MUTS:
    if mid not in only:
        continue
    key = (mod, tests, k)
    if key not in controls:          # the same copy with nothing changed: failures it shows are copy artefacts
        controls[key] = run_one(f"control_{mod}_{abs(hash(key)) % 10**6}", mod, HERE_LINE, HERE_LINE, tests, k)
    ctl = controls[key]
    got = run_one(mid, mod, old, new, tests, k)
    if "error" in got:
        res[mid] = got
        print(mid, got, flush=True)
        continue
    new_fail = {t: sorted(set(v["failed"]) - set(ctl[t]["failed"])) for t, v in got.items()}
    caught = any(new_fail.values())
    res[mid] = {"caught": caught, "new_failures": new_fail,
                "control_failures": {t: v["failed"] for t, v in ctl.items()},
                "tails": {t: v["tail"] for t, v in got.items()}}
    print(mid, "CAUGHT" if caught else "NOT CAUGHT", {t: len(v) for t, v in new_fail.items()},
          "control fails", {t: len(v["failed"]) for t, v in ctl.items()},
          [x for v in new_fail.values() for x in v][:3], flush=True)
prev = F / "final_review" / "rr_mutate.json"
allres = json.loads(prev.read_text()) if prev.exists() and sys.argv[2:] else {}
allres.update(res)
prev.write_text(json.dumps(allres, indent=1))
