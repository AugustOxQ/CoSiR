"""The seed-42 sensitivity input of round 6 (DECISION_RULE.md section 6 item 5, section 8 item 2; contracts section 7
and its amendment of 05:45; ticket 07): seed 42's per-episode differences of P1 to P7, S1 and S2 (from the regression,
results/seed42_per_episode.npz) split into sigma_a^2 and sigma_eps^2 as R3 rule section 6.1 does
(r3_stats.sensitivity's parts, through r6_stats.sigma_split), written to results/sensitivity_seed42.json:

    {"P1".."P7", "S1", "S2": {"quantity", "sigma_a2", "sigma_eps2"} (R@1 points squared), "seed": 42, "N": 12288,
     "n_paintings", "per_episode_sha256", "regression_sha256", "module_sha256", "input_sha256", "coef_sha256" (the
     regression's, equal to refit_check.json's), "time"}

The held runner (ticket 08) reads the sigma parts to write sensitivity_held.json (rule section 8 item 2).

Order guard: it refuses (exit 4) unless results/regression_seed42.json exists with "passed": true, was run by the
current bytes of every r6 module run_r6_held.py ran, and records the SHA-256 of the per-episode file it reads.

    cd /project/CoSiR && PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/envs/CoSiR/bin/python \
        src/test/20261125_artelingo_held_test/run_r6_sensitivity.py

Prints pass or refusal and the file path, never a sigma. Exit 0: written.

Guards carry a `# guard:<name>` marker; test_r6_regression.py deletes each on a copy and shows that its scenario then
goes through.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_stats as ST  # noqa: E402
import run_r6_held as RH  # noqa: E402

import numpy as np  # noqa: E402

OUT_NAME = "sensitivity_seed42.json"
CHECKS = ST.CHECKS + ST.SECONDARY
N_SEED42 = len(R.PAIRS) * R.N_PER_PAIR


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def guard(results, here=None) -> dict:
    """The regression passed, with the current bytes of the modules it ran, and recorded the per-episode file's
    SHA-256 (matching the file on disk). Raises RH.Refused; -> {"regression": record, "per_episode_sha256": sha}."""
    reg_path, npz = results / RH.REGRESSION_NAME, results / RH.PER_EPISODE_NAME
    RH._refuse_unless(reg_path.is_file() and npz.is_file(),
                      f"{reg_path.name} or {npz.name} is missing in {results}: run run_r6_held.py --mode regression "
                      f"first (rule section 6 item 4)")  # guard:regression_missing
    reg = json.loads(reg_path.read_text())
    RH._refuse_unless(reg.get("passed") is True, f"{reg_path} did not pass")  # guard:regression_passed
    stale = RH.stale_modules(reg, "run_r6_held.py", here)
    RH._refuse_unless(not stale, f"{reg_path} was written by other bytes of {stale}: rerun the regression (rule "
                                 f"section 6 item 7)")  # guard:regression_stale
    sha = R.sha256_file(npz)
    RH._refuse_unless((reg.get("outputs") or {}).get(npz.name) == sha,
                      f"{npz} is not the file the regression wrote")  # guard:per_episode_sha
    return {"regression": reg, "per_episode_sha256": sha, "regression_sha256": R.sha256_file(reg_path)}


def sensitivity_record(z) -> dict:
    """{check: {"quantity", "sigma_a2", "sigma_eps2"}} from the per-episode npz (fractions in), plus N and the
    painting count."""
    cl = np.asarray(z["cl"])
    _require(cl.shape == (N_SEED42,), f"cl has shape {cl.shape}, not ({N_SEED42},)")
    out = {}
    for c in CHECKS:
        d = np.asarray(z[f"diff__{c}"], np.float64)
        _require(d.shape == cl.shape, f"diff__{c}: shape {d.shape} differs from cl's")
        out[c] = {"quantity": ST.QUANTITIES[c][2], **ST.sigma_split(d, cl)}
    out["seed"] = R.DEV_SEED
    out["N"] = int(cl.size)
    out["n_paintings"] = int(np.unique(cl).size)
    return out


def run(results=None, here=None) -> int:
    results = Path(results or R.RESULTS)
    try:
        g = guard(results, here)
    except RH.Refused as e:
        print(f"refused: {e}")
        return RH.EXIT_REFUSE
    with np.load(results / RH.PER_EPISODE_NAME) as z:
        rec = sensitivity_record(z)
    rec.update({"per_episode_sha256": g["per_episode_sha256"], "regression_sha256": g["regression_sha256"],
                "module_sha256": R.r6_module_shas(), "input_sha256": g["regression"].get("input_sha256"),
                "coef_sha256": g["regression"].get("coef_sha256"), "time": R.amsterdam_now()})
    out = results / OUT_NAME
    RH.write_json(out, rec)
    print(f"sensitivity seed 42: written ({len(CHECKS)} checks); record {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.parse_args(argv)
    return run()


if __name__ == "__main__":
    sys.exit(main())
