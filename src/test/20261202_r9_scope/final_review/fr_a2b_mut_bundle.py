"""Final review A2b, item 1: the five r9_bundle guard deletions that survived A2's sweep with its full set (117, 118,
119, 122, 123), rerun against test_r9_context.py, which A2's TESTS map left out for r9_bundle.py.

Mutates ONLY a scratch copy of the round folder (argument 1: that copy's round folder), with A2's own mutate() and
plan (fr_a2_mut.py, fr_a2_mut_plan.json of r9-fr-a at 4458634; r9_bundle.py is byte-identical at 8db7ceb, checked
below). Reverts each edit at once and checks the SHA-256. Writes fr_a2b_mut_bundle.json beside this file.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SNAP = HERE.parent
A2 = Path("/project/CoSiR-wt/r9-fr-a/src/test/20261202_r9_scope/final_review")
COPY = Path(sys.argv[1]).resolve()
assert COPY != SNAP and "scratchpad" in str(COPY), COPY
PY = "/root/miniconda3/envs/CoSiR/bin/python"
TEST = {"fit_rows_sha": "test_fit_pm_asserts_the_unsorted_draw_sha", "pm_raw": "test_fit_pm_refuses_masked_features",
        "readers": "test_readers_guard", "cl_groups": "test_cl_groups_guard", "scores": "test_scores_guard"}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


spec = importlib.util.spec_from_file_location("fr_a2_mut", A2 / "fr_a2_mut.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)  # defines functions only; F there points to r9-fr-a, never used here
plan = {x["id"]: x for x in json.loads((A2 / "fr_a2_mut_plan.json").read_text())}
path = COPY / "r9_bundle.py"
assert sha(path) == sha(SNAP / "r9_bundle.py"), "copy differs from the snapshot"
env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "PYTHONDONTWRITEBYTECODE": "1",
       "CUDA_VISIBLE_DEVICES": ""}
out = {}
for i in (117, 118, 119, 122, 123):
    x = plan[i]
    assert x["file"] == "r9_bundle.py"
    orig, h0 = path.read_bytes(), sha(path)
    try:
        path.write_text(M.mutate(orig.decode(), x))
        t = time.time()
        p = subprocess.run([PY, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "test_r9_context.py",
                            "-k", TEST[x["guard"]]], cwd=COPY, env=env, capture_output=True, text=True, timeout=200)
        lines = p.stdout.strip().splitlines()
        out[str(i)] = {"guard": x["guard"], "lineno": x["lineno"], "test": TEST[x["guard"]], "rc": p.returncode,
                       "caught": p.returncode == 1,
                       "first_failure": next((ln for ln in lines if ln.startswith(("FAILED", "ERROR"))), None),
                       "summary": lines[-1] if lines else "", "seconds": round(time.time() - t)}
    finally:
        path.write_bytes(orig)
        assert sha(path) == h0
    print(i, out[str(i)]["guard"], out[str(i)]["caught"], out[str(i)]["summary"], flush=True)
(HERE / "fr_a2b_mut_bundle.json").write_text(json.dumps(out, indent=1) + "\n")
