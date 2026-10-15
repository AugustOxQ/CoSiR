"""Read-only, round-1 data only: examples of round-1 R-c cells whose min-margin criterion is mathematically tied
but whose float criteria (rc_core.select_fused's expression) differ, or vice versa."""
import sys
from pathlib import Path
import numpy as np
R1 = Path("/project/CoSiR/src/test/20261117_reader_fix_csd")
sys.path.insert(0, str(R1))
import common as C
import rc_core as K
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS
from src.eval.aspect_nested import _zdict
b = C.load_bundle(smoke=False)
z = np.load(R1 / "results/cand_Rc_Rb_expected_A0.npz")
T = {c: {d: z[f"T__{c}__{d}"] for d in DIRECTIONS} for c in CONDITIONS}
margins = {c: z[f"margin__{c}"] for c in CONDITIONS}
taus, _ = K.thresholds(margins)
zB, zT = _zdict(b.B), _zdict(T)
g = K.gates(margins, taus)
gated = {k: K.gated_terms(zT, g[k]) for k in g}
G = {k: K.g_cf(gated[k]) for k in g}
cells = K.rc_cells()
fr1, fg, cr1 = K.cell_statistics(zB, gated, G, cells)
par = b.ctx.parity
ctrl = K.control_choice(zB, par)
for h in (0, 1):
    tune = par == h
    I_r1 = np.round(4 * fr1[:, tune].sum(1)).astype(np.int64); I_g = np.round(4 * fg[:, tune].sum(1)).astype(np.int64)
    I_c = int(round(4 * ctrl[h][1] * tune.sum()))
    ci = np.minimum(I_r1 - I_c, I_g)
    cf = np.array([min(float(fr1[i][tune].mean()) - ctrl[h][1], float(fg[i][tune].mean())) for i in range(224)])
    shown = 0
    for i in range(224):
        for j in range(i + 1, 224):
            if (ci[i] == ci[j]) != (cf[i] == cf[j]) and shown < 3:
                print(f"half {h}: cells {i} {cells[i]} and {j} {cells[j]}: exact criterion x4n {ci[i]} vs {ci[j]} "
                      f"(R@1-limited {I_r1[i]-I_c <= I_g[i]} / {I_r1[j]-I_c <= I_g[j]}); float {cf[i]!r} vs {cf[j]!r}")
                shown += 1
