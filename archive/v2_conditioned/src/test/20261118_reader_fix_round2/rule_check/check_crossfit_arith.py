"""Read-only, round-1 data only: recompute round-1 R-c's 224 cells with rc_core, check that the stored arrays are
reproduced, and compare rc_core.select_fused's float criterion with the exact integer criterion (every per-episode
R@1 and gain is a multiple of 0.25). Also B's interval and B'(A1) at full precision. Computes no round-2 number."""
import sys, time
from pathlib import Path
import numpy as np
R1 = Path("/project/CoSiR/src/test/20261117_reader_fix_csd")
sys.path.insert(0, str(R1))
import common as C
import rc_core as K
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor
from src.eval.aspect_nested import _zdict
t0 = time.time()
b = C.load_bundle(smoke=False)
print(f"bundle [{time.time()-t0:.0f}s]")
print("B r1", C.point_ci(np.asarray(b.pB["r1"], np.float64), b.cl), "full", 100*float(np.mean(b.pB["r1"])))
for a in ("A0", "A1"):
    print("B'", a, 100*float(np.mean(b.pBp[a]["r1"])))
z = np.load(R1 / "results/cand_Rc_Rb_expected_A0.npz")
T = {c: {d: z[f"T__{c}__{d}"] for d in DIRECTIONS} for c in CONDITIONS}
margins = {c: z[f"margin__{c}"] for c in CONDITIONS}
taus, n = K.thresholds(margins)
zB, zT = _zdict(b.B), _zdict(T)
g = K.gates(margins, taus)
gated = {k: K.gated_terms(zT, g[k]) for k in g}
G = {k: K.g_cf(gated[k]) for k in g}
cells = K.rc_cells()
t1 = time.time()
fr1, fg, cr1 = K.cell_statistics(zB, gated, G, cells)
print(f"224 cells (fused + counterpart) [{time.time()-t1:.1f}s]")
par = b.ctx.parity
ctrl = K.control_choice(zB, par)
fp = K.select_fused(fr1, fg, ctrl, par, range(224))
cp = K.select_cf(cr1, par, range(224))
print("float picks fused", {h: cells[i] for h, i in fp.items()}, "cf", {h: cells[i] for h, i in cp.items()}, "ctrl", ctrl)
# exact integer criterion
r1c = {s: None for s in [0]}
from src.eval.aspect_nested import _combine, control_sums
ctrl_r1 = {s: np.asarray(per_anchor(_combine(zB, zB, zB, s, 0.0))["r1"], np.float64) for s in control_sums()}
for h in (0, 1):
    tune = par == h
    assert np.all((4 * fr1[:, tune]) == np.round(4 * fr1[:, tune])) and np.all((4 * fg[:, tune]) == np.round(4 * fg[:, tune]))
    I_r1 = np.round(4 * fr1[:, tune].sum(axis=1)).astype(np.int64)
    I_g = np.round(4 * fg[:, tune].sum(axis=1)).astype(np.int64)
    sig = max(control_sums(), key=lambda s: int(np.round(4 * ctrl_r1[s][tune].sum())))
    I_c = int(np.round(4 * ctrl_r1[sig][tune].sum()))
    crit_int = np.minimum(I_r1 - I_c, I_g)
    best_int = int(np.argmax(crit_int))           # first max
    crit_float = np.array([min(float(fr1[i][tune].mean()) - ctrl[h][1], float(fg[i][tune].mean())) for i in range(224)])
    order = np.argsort(-crit_int, kind="stable")
    print(f"half {h}: sigma int {sig} float {ctrl[h][0]}; best exact cell {best_int} {cells[best_int]} vs float pick {fp[h]} {cells[fp[h]]}; "
          f"top exact criteria (x4 sums) {crit_int[order[:5]].tolist()} cells {order[:5].tolist()}; "
          f"#cells tied at the max {int((crit_int == crit_int.max()).sum())}")
    # float vs exact consistency over all pairs of cells
    incons = 0
    for i in range(224):
        for j in range(i + 1, 224):
            if (crit_int[i] == crit_int[j]) != (crit_float[i] == crit_float[j]) or (crit_int[i] > crit_int[j]) != (crit_float[i] > crit_float[j]):
                incons += 1
    print(f"   cell pairs whose float order differs from the exact order: {incons}")
    cI = np.round(4 * cr1[:, tune].sum(axis=1)).astype(np.int64)
    print(f"   counterpart: exact best {int(np.argmax(cI))} vs float {cp[h]}; tied at max {int((cI == cI.max()).sum())}")
fused = K.assemble(zB, gated, cells, fp, par); cfs = K.assemble(zB, G, cells, cp, par)
pn, pc = per_anchor(fused), per_anchor(cfs)
ok = all(np.array_equal(pn[m], z[f"fused__{m}"]) and np.array_equal(pc[m], z[f"cf__{m}"]) for m in ("r1", "gain", "other", "swap", "strict"))
print("stored round-1 R-c per-anchor arrays reproduced:", ok)
print(f"done [{time.time()-t0:.0f}s]")
