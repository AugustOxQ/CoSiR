"""fr_c: rule section 7 items 5 to 7 (built, the stop at 9,406 hits, the 24-hour budget from 2026-10-09 12:29) on
crafted seed-42 stage records, through run_r6_dts.main and run_r6_held.dts_stop_problem. Reviewer C."""
import json, sys
from pathlib import Path
F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
import r6_common as R  # noqa: E402
import run_r6_dts as RD  # noqa: E402
import run_r6_held as RH  # noqa: E402
import r6_gpu_common as G  # noqa: E402
import numpy as np  # noqa: E402
OUT = Path(sys.argv[1])
N = 12288
SSHA = G.sha256_file(F / "dts_settings.json")


def world(name, hits, sanity_time="2026-10-09 23:00:00", chosen_time="2026-10-10 09:00:00", sanity_ok=True,
          chosen=True, n=N):
    res = OUT / "stop" / name
    res.mkdir(parents=True)
    r1 = np.zeros(N)
    r1[: hits // 4] = 1.0
    if hits % 4:
        r1[hits // 4] = (hits % 4) / 4
    np.savez(res / "dts_seed42_per_anchor.npz", dts__r1=r1)
    base = {"seed": 42, "module_sha256": R.r6_module_shas(), "settings_sha256": SSHA}
    G.write_json(res / "dts_sanity.json", {**base, "passed": sanity_ok, "time": sanity_time})
    G.write_json(res / "dts_tune.json", {**base, "chosen": {"wording_id": "W3", "K": 16}, "time": sanity_time})
    if chosen:
        G.write_json(res / "dts_seed42.json", {**base, "n_episodes": n, "setting": "W3 K16", "time": chosen_time,
                                               "per_anchor_file": "dts_seed42_per_anchor.npz",
                                               "per_anchor_sha256": G.sha256_file(res / "dts_seed42_per_anchor.npz")})
    return res


def run(res):
    code = RD.main(["--stage", "stop", "--out", str(res), "--clock-start", "2026-10-09 12:29"])
    rec = json.loads((res / "dts_stop.json").read_text()) if (res / "dts_stop.json").exists() else None
    why, fired = RH.dts_stop_problem(res, F)
    return {"exit": code, "hits": rec and rec.get("hits"), "built": rec and rec["built"], "stop": rec and rec["stop"],
            "reason": rec and rec["reason"], "held_runner_ok": why is None, "held_fired": fired}


out = {
    "9406_no_stop": run(world("a", 9406)),
    "9407_stop": run(world("b", 9407)),
    "9405_no_stop": run(world("c", 9405)),
    "built_at_deadline_12:29:00": run(world("d", 9000, chosen_time="2026-10-10 12:29:00")),
    "built_after_deadline_12:29:01": run(world("e", 9000, chosen_time="2026-10-10 12:29:01")),
    "not_built_now_within_budget": run(world("f", 9000, chosen=False)),
    "sanity_failed_now": run(world("g", 9000, sanity_ok=False)),
    "rerun_after_code_change_at_2026-10-10_15:00": run(world("h", 9000, sanity_time="2026-10-10 14:00:00",
                                                             chosen_time="2026-10-10 15:00:00")),
}
print(json.dumps(out))
(F / "final_review" / "fr_c_stop.json").write_text(json.dumps(out, indent=1))
