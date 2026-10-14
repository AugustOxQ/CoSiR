"""Final review B2, item 5: run rederive/rd9_compare.py --phase1 against the real CLIP dry-run pins file
(t13_clip_dryrun/pins_clip_b32.json, the schema build_pins writes for every backbone) with t25's CLIP phase-1 output.
Run 1: unchanged code. Run 2: resolve_pin patched to resolve src/test-relative keys (to see the next failure).
Writes nothing outside the scratchpad."""
import json, runpy, sys, traceback
from pathlib import Path
RDIR = Path("/project/CoSiR-wt/r9-fr-b2/src/test/20261202_r9_scope/rederive")
O = Path("/project/CoSiR-wt/r9-t25/src/test/20261202_r9_scope/rederive/rd9_out")
S = Path("/tmp/claude-0/-project-CoSiR/18057a5a-4e60-48cd-83b0-ff14c8a319ae/scratchpad/fr_b2")
PINS = "/project/CoSiR/src/test/20261202_r9_scope/results/t13_clip_dryrun/pins_clip_b32.json"
sys.path.insert(0, str(RDIR)); sys.path.insert(0, "/project/CoSiR-wt/r9-fr-b2")
import rd9_inputs
res = {}
def attempt(name, out):
    sys.argv = ["rd9_compare.py", "--phase1", "--backbone", "clip_b32", "--pins", PINS, "--derived",
                str(O / "rd9_clip_b32_phase1.json"), "--derived-arrays", str(O / "rd9_clip_b32_seed42_arrays.npz"),
                "--out", str(S / out)]
    try:
        runpy.run_path(str(RDIR / "rd9_compare.py"), run_name="__main__")
        rec = json.loads((S / out).read_text())
        res[name] = {"ran": True, "agreement": rec.get("agreement"), "compared": rec.get("compared"),
                     "failed_keys": [x["key"] for x in rec.get("items", []) if not x["equal"]][:40]}
    except BaseException as e:
        res[name] = {"ran": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                     "where": traceback.format_exc().strip().splitlines()[-3][:200]}
attempt("unchanged", "p1_unchanged.json")
orig = rd9_inputs.resolve_pin
def patched(rel):
    p = Path(rel)
    if not p.is_absolute() and not str(p).startswith("src/") and (rd9_inputs.T / p).exists():
        return rd9_inputs.T / p
    return orig(rel)
rd9_inputs.resolve_pin = patched
attempt("resolve_pin_patched", "p1_patched.json")
orig_entries = rd9_inputs.pin_entries
def entries_without_inputs(record):
    return orig_entries({k: v for k, v in record.items() if k != "input_sha256"})
rd9_inputs.pin_entries = entries_without_inputs
attempt("input_sha256_skipped", "p1_skip_inputs.json")
def entries_parts_only(record):
    return orig_entries({"parts": record["parts"]})
rd9_inputs.pin_entries = entries_parts_only
attempt("parts_only", "p1_parts_only.json")
# Run 5: compare_phase1 called directly with the dry-run pins' seed42 record and CLIP's pinned JSON parts
import numpy as np
sys.argv = ["x"]
C = runpy.run_path(str(RDIR / "rd9_compare.py"), run_name="not_main")
pins = json.loads(Path(PINS).read_text())
own = json.loads((O / "rd9_clip_b32_phase1.json").read_text())
with np.load(O / "rd9_clip_b32_seed42_arrays.npz") as z:
    arrays = {k: z[k] for k in z.files}
part = lambda r: json.loads(Path(pins["parts"][r]["path"]).read_text())
c = C["Comparison"]()
try:
    C["compare_phase1"](c, own, arrays, part("cells_seed42"), part("picks_seed42"), part("baselines_seed42"),
                        part("rc_tau"), pins["seed42"])
    rec = c.record()
    res["direct_seed42_from_pins"] = {"ran": True, "agreement": rec["agreement"], "compared": rec["compared"],
                                      "failed_keys": [x["key"] for x in rec["items"] if not x["equal"]][:40]}
except BaseException as e:
    res["direct_seed42_from_pins"] = {"ran": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                                      "items_before": len(c.items), "failed_before": [x["key"] for x in c.items if not x["equal"]][:20]}
res["pins_seed42_keys"] = sorted(pins["seed42"])
res["own_array_keys_sample"] = sorted(arrays)[:8]
print(json.dumps(res, indent=1))
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
