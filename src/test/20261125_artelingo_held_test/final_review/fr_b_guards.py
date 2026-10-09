"""Final review B: guards of my area fired through the real entry points (cheap cases; nothing written to F/results).

  1. rule 6.1 input SHA-256: run_r6_refit.main() with one expected SHA changed in memory -> SystemExit before any data
  2. order guard: run_r6_held.run_regression(results=<empty temp dir>) -> exit 4 before inputs or data
  3. sensitivity order guard: run_r6_sensitivity.run(results=<empty temp dir>) -> exit 4
  4. picks target guard: r6_picks.load_picks on a copy of picks_seed42.json with B's mean changed by one ulp -> raises
  5. stale records: run_r6_held.order_guard with refit_check.json's run_r6_refit.py SHA changed -> Refused
    argv[1]: temp dir
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

F = Path("/project/CoSiR-r6-fr/src/test/20261125_artelingo_held_test")
sys.path.insert(0, str(F))
import r6_common as R  # noqa: E402
import r6_picks as P  # noqa: E402
import run_r6_held as RH  # noqa: E402
import run_r6_refit as RR  # noqa: E402
import run_r6_sensitivity as RS  # noqa: E402

TMP = Path(sys.argv[1])
out = {}

# 1
rel = "20261116_grouping_step1_style/results/step1_heads_style.npz"
saved = R.INPUT_SHA256[rel]
R.INPUT_SHA256[rel] = "0" * 64
R._SHA_CACHE.clear()
try:
    RR.main(["--out", str(TMP / "never.json")])
    out["input_sha"] = "NOT CAUGHT"
except SystemExit as e:
    out["input_sha"] = f"caught: {str(e)[:60]}"
finally:
    R.INPUT_SHA256[rel] = saved
out["input_sha_no_output"] = not (TMP / "never.json").exists()

# 2, 3
empty = TMP / "empty_results"
empty.mkdir(exist_ok=True)
out["regression_order_guard_exit"] = RH.run_regression(results=empty)
out["sensitivity_guard_exit"] = RS.run(results=empty)

# 4
cp = TMP / "picks_copy.json"
rec = json.loads((R.RESULTS / "picks_seed42.json").read_text())
rec["mean_r1"]["B"] = float(np.nextafter(rec["mean_r1"]["B"], 100))
cp.write_text(json.dumps(rec))
try:
    P.load_picks(cp)
    out["picks_target"] = "NOT CAUGHT"
except AssertionError as e:
    out["picks_target"] = "caught"

# 5
d = TMP / "stale"
d.mkdir(exist_ok=True)
for n in ("refit_check.json", "picks_seed42.json"):
    shutil.copy(R.RESULTS / n, d / n)
rc = json.loads((d / "refit_check.json").read_text())
key = [k for k in rc["module_sha256"] if k.endswith("r6_heads.py")][0]
rc["module_sha256"][key] = "f" * 64
(d / "refit_check.json").write_text(json.dumps(rc))
try:
    RH.order_guard(d)
    out["stale_refit"] = "NOT CAUGHT"
except RH.Refused as e:
    out["stale_refit"] = "caught"
try:
    RH.order_guard(R.RESULTS)
    out["order_guard_current_records"] = "passes"
except RH.Refused as e:
    out["order_guard_current_records"] = f"refused: {e}"
(TMP / "guards.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out))
