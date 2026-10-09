"""fr_c: mutation checks of area C's rule-named guards. Each mutation edits one module on a copy in the scratch dir
(never in place); a pytest plugin loads the copy under the module's own name before the test file imports it; the
suite must fail. Usage: fr_c_mutate.py <scratch dir>. Reviewer C."""
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
    ("stop_ge", "r6_dts", '"dts_above_aff": bool(hits > aff_hits)', '"dts_above_aff": bool(hits >= aff_hits)',
     "test_r6_dts.py", "stop"),
    ("budget_plus_1h", "r6_dts", "deadline = start.astimezone(timezone.utc) + timedelta(hours=hours)",
     "deadline = start.astimezone(timezone.utc) + timedelta(hours=hours + 1)", "test_r6_dts.py", "budget"),
    ("sanity_ge0", "run_r6_dts", '"passed": bool(r.gain_int4 > 0)', '"passed": bool(r.gain_int4 >= 0)',
     "test_r6_dts.py", "sanity or stages"),
    ("names_swapped", "r6_dts", "for j, c in enumerate(CONDITIONS)}", "for j, c in enumerate(CONDITIONS[::-1])}",
     "test_r6_dts.py", "name"),
    ("frozen_parity", "r6_dts", "apply = parity == 1 - h", "apply = parity == h", "test_r6_dts.py",
     "frozen or held"),
    ("tie_last", "r6_dts", "if scores[s] > scores[best]:", "if scores[s] >= scores[best]:", "test_r6_dts.py", "tie"),
    ("no_failed_fallback", "r6_dts", "f[c][d][m[c]] = zero[c][d][m[c]]  # guard:failed_rows", "pass",
     "test_r6_dts.py", "inf or failed or fallback"),
    ("first_line_no_strip", "r6_dts", "    s = answer.strip()\n    return s.splitlines()[0] if s else \"\"",
     "    s = answer\n    return s.splitlines()[0] if s.splitlines() else \"\"", "test_r6_dts.py", "phrase"),
    ("unpermute_inverse", "r6_external", "np.put_along_axis(out, perms, shown, axis=-1)",
     "out[...] = np.take_along_axis(shown, perms, axis=-1)", "test_r6_external.py", "mllm or rerank or unpermute"),
    ("two_way_same_stream", "r6_descriptive", "rng_c = np.random.default_rng(seed), np.random.default_rng(list(cand_seed))",
     "rng_c = np.random.default_rng(seed), np.random.default_rng(seed)", "test_r6_descriptive.py", "two_way"),
]


def plugin(mod, path):
    return (f"import importlib.util, sys\nsys.path.insert(0, {str(F)!r})\nimport r6_common\n"
            f"spec = importlib.util.spec_from_file_location({mod!r}, {str(path)!r})\n"
            f"m = importlib.util.module_from_spec(spec)\nsys.modules[{mod!r}] = m\nspec.loader.exec_module(m)\n")


res = {}
only = sys.argv[2:] or [m[0] for m in MUTS]
for mid, mod, old, new, test, k in MUTS:
    if mid not in only:
        continue
    src = (F / f"{mod}.py").read_text()
    if src.count(old) != 1:
        res[mid] = {"error": f"pattern found {src.count(old)} times"}
        continue
    d = OUT / mid
    d.mkdir(parents=True, exist_ok=True)
    copy = d / f"{mod}.py"
    copy.write_text(src.replace(old, new).replace(HERE_LINE, f"HERE = Path({str(F)!r})\n"))
    (d / f"frc_plugin_{mid}.py").write_text(plugin(mod, copy))
    env = dict(os.environ, PYTHONPATH=str(d), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
               CUDA_VISIBLE_DEVICES="")
    log = d / "pytest.log"
    with open(log, "w") as fh:
        p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", f"frc_plugin_{mid}",
                            "--basetemp", str(d / "pt"), str(F / test), "-k", k], stdout=fh, stderr=subprocess.STDOUT,
                           env=env, cwd=str(d), timeout=3000)
    tail = log.read_text().strip().splitlines()[-1:] if log.read_text().strip() else []
    res[mid] = {"exit": p.returncode, "caught": p.returncode == 1, "tail": tail}
    print(mid, res[mid], flush=True)
prev = F / "final_review" / "fr_c_mutate.json"
allres = json.loads(prev.read_text()) if prev.exists() else {}
allres.update(res)
prev.write_text(json.dumps(allres, indent=1))
