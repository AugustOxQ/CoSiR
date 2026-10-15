"""Scoped re-review (r6 fix wave, T15 fix round): the DTS budget decision on crafted seed-42 stage records.

Mine, from rule section 7 items 5 to 7 and the agent default of run_handoff_items.md: built time = the later of the
sanity and chosen records' times, unless dts_first_built.json exists and the current chosen record has its setting,
settings SHA-256 and the sanity and chosen GPU output fingerprints (then the first build's time); within the budget
iff built time <= clock start + 24 h of elapsed time (clock start 2026-10-09 12:29 Amsterdam); stop iff not within or
hits > 9,406. dts_first_built.json is written by the chosen stage iff seed 42, absent, and its build is within budget.

The code's: run_r6_dts.record_first_built (as stage_chosen calls it after writing the chosen record), then
run_r6_dts.main(["--stage", "stop", ...]) (the real entry point), then run_r6_held.dts_stop_problem (the held guard).

Usage: python rr_dts_budget.py <out_dir>   -> <out_dir>/rr_dts_budget.json
"""
import hashlib
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
import r6_common as R  # noqa: E402
import run_r6_dts as RD  # noqa: E402
import run_r6_held as RH  # noqa: E402

OUT = Path(sys.argv[1])
AMS = ZoneInfo("Europe/Amsterdam")
START = "2026-10-09 12:29"
N = 3 * 4096
SETTINGS_SHA = hashlib.sha256((F / "dts_settings.json").read_bytes()).hexdigest()
FP = {"verbaliser": {"v1": {"job": "r6_gpu_verbalise", "inputs_sha256": {"a": "1" * 64}}},
      "listing": {"l1": {"job": "r6_gpu_listing", "inputs_sha256": {"b": "2" * 64}}}}


def t(s):
    fmt = "%Y-%m-%d %H:%M:%S" if s.count(":") == 2 else "%Y-%m-%d %H:%M"
    return datetime.strptime(s, fmt).replace(tzinfo=AMS)


def within(built):
    return t(built).astimezone(timezone.utc) <= t(START).astimezone(timezone.utc) + timedelta(hours=24)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wjson(p, rec):
    Path(p).write_text(json.dumps(rec, indent=1))


def world(d, sanity_time, chosen_time, hits=100, setting=("W1", 8), fp_s=FP, fp_c=FP, settings_sha=SETTINGS_SHA):
    d.mkdir(parents=True)
    mods = R.r6_module_shas()
    r1 = np.zeros(N, dtype=np.float64)
    r1[:hits] = 0.25
    np.savez(d / RD.per_anchor_name(42), dts__r1=r1)
    base = {"seed": 42, "module_sha256": mods, "settings_sha256": settings_sha}
    wjson(d / RD.SANITY, {**base, "passed": True, "time": sanity_time, "gpu_fingerprints": fp_s})
    wjson(d / RD.TUNE, {**base, "chosen": {"wording_id": setting[0], "K": setting[1]}, "time": "2026-10-09 20:00:00"})
    chosen = {**base, "n_episodes": N, "setting": f"{setting[0]} K{setting[1]}", "time": chosen_time,
              "per_anchor_file": RD.per_anchor_name(42), "per_anchor_sha256": sha(d / RD.per_anchor_name(42)),
              "gpu_fingerprints": fp_c}
    wjson(d / RD.chosen_name(42), chosen)
    return chosen


def mine(sanity, chosen, fb, hits):
    built = max(sanity["time"], chosen["time"], key=t)
    src = "this run"
    if fb is not None and (fb["setting"], fb["settings_sha256"], fb["gpu_fingerprints"]) == (
            chosen["setting"], chosen["settings_sha256"],
            {"sanity": sanity.get("gpu_fingerprints"), "chosen": chosen.get("gpu_fingerprints")}):
        built, src = fb["time"], "first_built"
    ok = within(built)
    return {"built_time": built, "within": ok, "stop": (not ok) or hits > 9406, "source": src}


def stop(d):
    code = RD.main(["--stage", "stop", "--out", str(d), "--clock-start", START])
    rec = json.loads((d / RD.STOP).read_text()) if (d / RD.STOP).is_file() else None
    return code, rec


def case_fresh(d, sanity_time, chosen_time, hits=100):
    chosen = world(d, sanity_time, chosen_time, hits)
    first = RD.record_first_built(d, 42, chosen)
    fb = json.loads((d / RD.FIRST_BUILT).read_text()) if (d / RD.FIRST_BUILT).is_file() else None
    want_first = "written" if within(max(sanity_time, chosen_time, key=t)) else None
    sanity = json.loads((d / RD.SANITY).read_text())
    code, rec = stop(d)
    m = mine(sanity, chosen, fb, hits)
    return {"first_built": first, "first_built_want": want_first, "exit": code, "code": _code(rec), "mine": m,
            "agree": (first == want_first and _agree(code, rec, m)), "held_guard": _held(d, rec)}


def case_rerun(d, change):
    """First build at 2026-10-10 09:00:00 (written), then a rerun at 2026-10-12 09:00:00 with ``change``."""
    chosen = world(d, "2026-10-09 15:00:00", "2026-10-10 09:00:00")
    assert RD.record_first_built(d, 42, chosen) == "written"
    fb = json.loads((d / RD.FIRST_BUILT).read_text())
    setting, fp_s, fp_c, ssha = ("W1", 8), FP, FP, SETTINGS_SHA
    if change == "setting":
        setting = ("W3", 16)
    elif change == "settings_sha":                 # the first build ran another settings file: its record says so
        fb["settings_sha256"] = "0" * 64
    elif change == "sanity_outputs":
        fp_s = {"verbaliser": {}, "listing": {"l2": {"job": "r6_gpu_listing"}}}
    elif change == "chosen_outputs":
        fp_c = {"verbaliser": {"v2": {"job": "r6_gpu_verbalise"}}, "listing": FP["listing"]}
    moved = d.parent / (d.name + "_rerun")
    chosen = world(moved, "2026-10-12 08:00:00", "2026-10-12 09:00:00", setting=setting, fp_s=fp_s, fp_c=fp_c,
                   settings_sha=ssha)
    wjson(moved / RD.FIRST_BUILT, fb) if change == "settings_sha" else shutil.copy2(d / RD.FIRST_BUILT, moved)
    sanity = json.loads((moved / RD.SANITY).read_text())
    code, rec = stop(moved)
    m = mine(sanity, chosen, fb, 100)
    return {"exit": code, "code": _code(rec), "mine": m, "agree": _agree(code, rec, m), "held_guard": _held(moved, rec)}


def _code(rec):
    return None if rec is None else {k: rec.get(k) for k in ("built", "stop", "reason", "budget_time_source")} | {
        "built_time": rec["budget"]["built_time"]}


def _agree(code, rec, m):
    if rec is None:
        return False
    src = "first_built" if rec["budget_time_source"] == RD.FIRST_BUILT else "this run"
    return (rec["built"] == m["within"] and rec["stop"] == m["stop"] and src == m["source"]
            and rec["budget"]["built_time"] == m["built_time"] and code == (3 if m["stop"] else 0))


def _held(d, rec):
    why, fired = RH.dts_stop_problem(d, F)
    return {"problem": why, "fired": fired, "accepted": why is None}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    w = OUT / "worlds"
    res = {
        "fresh_1229_00": case_fresh(w / "a", "2026-10-09 15:00:00", "2026-10-10 12:29:00"),
        "fresh_1229_01": case_fresh(w / "b", "2026-10-09 15:00:00", "2026-10-10 12:29:01"),
        "fresh_sanity_later_1229_01": case_fresh(w / "c", "2026-10-10 12:29:01", "2026-10-10 12:00:00"),
        "fresh_in_time_above_aff": case_fresh(w / "d", "2026-10-09 15:00:00", "2026-10-10 09:00:00", hits=9407),
        "fresh_in_time_at_aff": case_fresh(w / "e", "2026-10-09 15:00:00", "2026-10-10 09:00:00", hits=9406),
    }
    for change in ("unchanged", "setting", "settings_sha", "sanity_outputs", "chosen_outputs"):
        res[f"rerun_{change}"] = case_rerun(w / f"r_{change}", change)
    # the clock start: another start for seed 42 is refused through main, nothing written
    d = w / "clock"
    world(d, "2026-10-09 15:00:00", "2026-10-10 09:00:00")
    code = RD.main(["--stage", "stop", "--out", str(d), "--clock-start", "2026-10-09 18:00"])
    res["clock_other"] = {"exit": code, "stop_written": (d / RD.STOP).exists(), "agree": code == RD.EXIT_REFUSED
                          and not (d / RD.STOP).exists()}
    res["constants"] = {"R.DTS_CLOCK_START": R.DTS_CLOCK_START, "RD.DTS_CLOCK_START": RD.DTS_CLOCK_START}
    (OUT / "rr_dts_budget.json").write_text(json.dumps(res, indent=1, default=str))
    for k, v in res.items():
        if k != "constants":
            print(k, "agree" if v["agree"] else "DISAGREE", v.get("exit"),
                  (v.get("held_guard") or {}).get("accepted"))
