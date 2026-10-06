"""Final review: mutation check of the unit tests. Each mutation is applied to a scratch copy of the code in
final_review/mut/ (never the originals); the copied tests run against it. A mutation that leaves every test passing is a
guard the tests do not hold."""
import shutil
import subprocess
from pathlib import Path

FR = Path(__file__).resolve().parent
RD = FR.parent
MUT = FR / "mut"
FILES = ("common.py", "rc_core.py", "rb_features.py", "test_reader_fix.py", "test_rb.py", "DECISION_RULE.md")

MUTATIONS = [
    ("M01 assert_rule guard deleted", "common.py",
     "    got = sha_file(RULE)\n    if got != RULE_SHA:", "    got = RULE_SHA\n    if got != RULE_SHA:"),
    ("M02 bar comparator ties to the later", "common.py", "        if means[i] > means[best]:", "        if means[i] >= means[best]:"),
    ("M03 clause 1 strict", "common.py", 'c1 = bool(bar_r1["point"] >= BAR_TARGET)', 'c1 = bool(bar_r1["point"] > BAR_TARGET)'),
    ("M04 clause 3 reads the bar interval", "common.py", 'c3 = bool(gain_stat["ci95"][0] > 0)', 'c3 = bool(bar_r1["ci95"][0] > 0)'),
    ("M05 per-pair bar against B always", "common.py",
     '"per_pair_r1": {p: point_ci(v[pair_index == i], cl[pair_index == i])',
     '"per_pair_r1": {p: point_ci((np.asarray(pn["r1"]) - np.asarray(pB["r1"]))[pair_index == i], cl[pair_index == i])'),
    ("M06 sigma with ddof 0", "common.py", "vs = np.var(np.asarray(sup, dtype=np.float64), axis=1, ddof=1)",
     "vs = np.var(np.asarray(sup, dtype=np.float64), axis=1, ddof=0)"),
    ("M07 cf_version takes condition a only", "common.py",
     "(0.5 * (t[\"a\"][d].astype(np.float64) + t[\"b\"][d].astype(np.float64))).astype(np.float32)",
     "(t[\"a\"][d].astype(np.float64)).astype(np.float32)"),
    ("M08 gate strict", "rc_core.py", "(np.asarray(margins[c]) >= t)", "(np.asarray(margins[c]) > t)"),
    ("M09 tau from condition a only", "rc_core.py",
     'm = np.concatenate([np.asarray(margins["a"], np.float64), np.asarray(margins["b"], np.float64)])',
     'm = np.concatenate([np.asarray(margins["a"], np.float64)] * 2)'),
    ("M10 R-c assembly scores the tuning half itself (in-sample)", "rc_core.py",
     "        apply = parity != half\n        k, u, a = cells[picks[half]]", "        apply = parity == half\n        k, u, a = cells[picks[half]]"),
    ("M11 R-c control picks the largest sigma", "rc_core.py",
     "sigma = max(control_sums(), key=lambda s: float(r1[s][tune].mean()))",
     "sigma = max(control_sums()[::-1], key=lambda s: float(r1[s][tune].mean()))"),
    ("M12 R-c fused pick by R@1 only", "rc_core.py",
     "crit = lambda i: min(float(fr1[i][tune].mean()) - r_ctrl, float(fg[i][tune].mean()))",
     "crit = lambda i: float(fr1[i][tune].mean()) - r_ctrl"),
    ("M13 G_cf as gbar * mean z(T)", "rc_core.py",
     'm = (0.5 * (gated["a"][d].numpy().astype(np.float64) + gated["b"][d].numpy().astype(np.float64))).astype(np.float32)',
     'm = (0.25 * (gated["a"][d].numpy().astype(np.float64) + gated["b"][d].numpy().astype(np.float64)) * 2).astype(np.float32) * 0.999'),
    ("M14 R-b labels in sorted order", "rb_features.py", "    parts = list(parts)\n    ya =", "    parts = sorted(parts)\n    ya ="),
    ("M15 R-b condition b not swapped", "rb_features.py",
     '"b": episode_features(post, parts, ep.pairs_b_img, ep.pairs_b_txt, ep.pairs_a_img, ep.pairs_a_txt)}',
     '"b": episode_features(post, parts, ep.pairs_a_img, ep.pairs_a_txt, ep.pairs_b_img, ep.pairs_b_txt)}'),
    ("M16 R-b uses half-reader 0 only", "rb_features.py", "    return stack.mean(axis=0)", "    return stack[0]"),
    ("M17 R-b C ties to the larger", "rb_features.py", "        if mean_losses[i] < mean_losses[best]:",
     "        if mean_losses[i] <= mean_losses[best]:"),
    ("M18 R-b match share on contrasts", "rb_features.py",
     "match = (a_img.argmax(axis=-1) == a_txt.argmax(axis=-1)).mean(axis=1)",
     "match = (pi[con_img].argmax(axis=-1) == pt[con_txt].argmax(axis=-1)).mean(axis=1)"),
    ("M19 R-b pick ties to the last", "rb_features.py",
     "return P.argmax(axis=1).astype(np.int64), top[:, -1] - top[:, -2]",
     "return (P.shape[1] - 1 - P[:, ::-1].argmax(axis=1)).astype(np.int64), top[:, -1] - top[:, -2]"),
]


def setup():
    if MUT.exists():
        shutil.rmtree(MUT)
    MUT.mkdir()
    for f in FILES:
        shutil.copy(RD / f, MUT / f)


def run():
    r = subprocess.run(["/root/miniconda3/envs/CoSiR/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        str(MUT / "test_reader_fix.py"), str(MUT / "test_rb.py")],
                       capture_output=True, text=True, cwd="/project/CoSiR",
                       env={"PYTHONPATH": "/project/CoSiR", "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "8",
                            "PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"})
    last = [ln for ln in r.stdout.strip().splitlines() if "passed" in ln or "failed" in ln or "error" in ln]
    return r.returncode, (last[-1] if last else r.stdout[-300:] + r.stderr[-300:])


setup()
print("baseline:", run())
for name, f, old, new in MUTATIONS:
    setup()
    src = (MUT / f).read_text()
    assert src.count(old) == 1, f"{name}: pattern found {src.count(old)} times"
    (MUT / f).write_text(src.replace(old, new))
    code, summary = run()
    print(f"{name:62s} -> {'CAUGHT' if code != 0 else 'SURVIVED (all tests pass)'}  [{summary}]")
shutil.rmtree(MUT)
