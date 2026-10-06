"""Scoped re-review: re-run the four mutations the fix wave targets (M01, M05, M07, M10) on scratch copies under
rereview/mut/ (never the originals), and the baseline on the originals in place. Reports which tests fail."""
import shutil
import subprocess
from pathlib import Path

RR = Path(__file__).resolve().parent
RD = RR.parent.parent
MUT = RR / "m1" / "m2" / "mut"   # depth chosen so common.py's HERE.parents[2] is rereview/, which holds no module
FILES = ("common.py", "rc_core.py", "rb_features.py", "test_reader_fix.py", "test_rb.py", "DECISION_RULE.md")
MUTATIONS = [
    ("M01 assert_rule guard deleted", "common.py",
     "    got = sha_file(RULE)\n    if got != RULE_SHA:", "    got = RULE_SHA\n    if got != RULE_SHA:"),
    ("M01b assert_rule body is a no-op", "common.py",
     "    got = sha_file(RULE)\n    if got != RULE_SHA:", "    got = sha_file(RULE)\n    if False:"),
    ("M05 per-pair bar against B always", "common.py",
     '"per_pair_r1": {p: point_ci(v[pair_index == i], cl[pair_index == i])',
     '"per_pair_r1": {p: point_ci((np.asarray(pn["r1"]) - np.asarray(pB["r1"]))[pair_index == i], cl[pair_index == i])'),
    ("M07 cf_version takes condition a only", "common.py",
     "(0.5 * (t[\"a\"][d].astype(np.float64) + t[\"b\"][d].astype(np.float64))).astype(np.float32)",
     "(t[\"a\"][d].astype(np.float64)).astype(np.float32)"),
    ("M10 assemble apply = parity == half", "rc_core.py",
     "        apply = parity != half\n        k, u, a = cells[picks[half]]", "        apply = parity == half\n        k, u, a = cells[picks[half]]"),
]
ENV = {"PYTHONPATH": "/project/CoSiR", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8",
       "PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"}


def run(folder):
    r = subprocess.run(["/root/miniconda3/envs/CoSiR/bin/python", "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider",
                        str(folder / "test_reader_fix.py"), str(folder / "test_rb.py")],
                       capture_output=True, text=True, cwd="/project/CoSiR", env=ENV)
    lines = r.stdout.strip().splitlines()
    failed = [ln.split("::")[-1].split(" ")[0] for ln in lines if ln.startswith("FAILED")]
    summ = [ln for ln in lines if " passed" in ln or " failed" in ln or " error" in ln]
    return r.returncode, (summ[-1] if summ else r.stdout[-400:] + r.stderr[-400:]), failed


def where():
    code = ("import sys; sys.path.insert(0, %r); import test_reader_fix as T; print(T.C.__file__, T.K.__file__)" % str(MUT))
    r = subprocess.run(["/root/miniconda3/envs/CoSiR/bin/python", "-c", code], capture_output=True, text=True,
                       cwd="/project/CoSiR", env=ENV)
    return r.stdout.strip() or r.stderr[-300:]


def setup():
    if MUT.exists():
        shutil.rmtree(RR / "m1")
    MUT.mkdir(parents=True)
    for f in FILES:
        shutil.copy(RD / f, MUT / f)


print("originals in place:", run(RD))
setup()
print("unmutated copy:", run(MUT))
for name, f, old, new in MUTATIONS:
    setup()
    src = (MUT / f).read_text()
    assert src.count(old) == 1, f"{name}: pattern found {src.count(old)} times"
    (MUT / f).write_text(src.replace(old, new))
    code, summary, failed = run(MUT)
    print("   imported from:", where())
    print(f"{name:42s} -> {'CAUGHT' if code != 0 else 'SURVIVED'} [{summary}] failed={failed}")
shutil.rmtree(RR / "m1")
