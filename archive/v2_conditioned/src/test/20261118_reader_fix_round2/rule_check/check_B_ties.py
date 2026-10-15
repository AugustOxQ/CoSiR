"""Read-only: properties of the stored condition-free score B that the top-k set K depends on (dtype, condition-
freeness, ties at the k_top boundary, B vs z(B) order), and the magnitude of fused scores. Computes no round-2
candidate number."""
import sys, time
from pathlib import Path
import numpy as np
R1 = Path("/project/CoSiR/src/test/20261117_reader_fix_csd")
sys.path.insert(0, str(R1))
import common as C
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS
from src.eval.aspect_nested import _zdict
t0 = time.time()
b = C.load_bundle(smoke=False)
print(f"bundle loaded [{time.time()-t0:.0f}s]")
B = b.B
zB = _zdict(B)
for d in DIRECTIONS:
    Ba, Bb = np.asarray(B["a"][d]), np.asarray(B["b"][d])
    print(d, "B dtype", Ba.dtype, Ba.shape, "B[a]==B[b] exactly:", np.array_equal(Ba, Bb))
    zb = zB["a"][d].numpy()
    print("   z(B) dtype", zb.dtype, "max|z(B)|", float(np.abs(zb).max()))
    s = -np.sort(-Ba.astype(np.float64), axis=1)
    for k in (1, 2, 3, 5):
        tie_boundary = np.mean(s[:, k - 1] == s[:, k])
        print(f"   k={k}: share of rows with B tied across the k-th/(k+1)-th boundary {tie_boundary:.6f} ({int(tie_boundary*len(s))} rows)")
    anytie = np.mean((np.diff(np.sort(Ba, axis=1), axis=1) == 0).any(axis=1))
    print(f"   rows with any tie among 13 B scores: {anytie:.6f}")
    oB = np.argsort(-Ba, axis=1, kind="stable")
    oZ = np.argsort(-zb, axis=1, kind="stable")
    for k in (2, 3, 5):
        KB = np.sort(oB[:, :k], axis=1); KZ = np.sort(oZ[:, :k], axis=1)
        print(f"   k={k}: rows where top-k set from B differs from top-k set from z(B): {int((KB != KZ).any(axis=1).sum())}")
    # z(B) ties not present in B
    zt = np.mean((np.diff(np.sort(zb, axis=1), axis=1) == 0).any(axis=1))
    print(f"   rows with any tie among 13 z(B) scores: {zt:.6f}")
    # where does p_A / p_B sit in B's order (share of rows where target column is in top-k)
    for k in (2, 3, 5):
        inK0 = (oB[:, :k] == 0).any(axis=1).mean(); inK1 = (oB[:, :k] == 1).any(axis=1).mean()
        print(f"   k={k}: share of rows with p_A in K {inK0:.4f}, p_B in K {inK1:.4f}")
z = np.load(R1 / "results/cand_Rc_Rb_expected_A0.npz")
T = {c: {d: z[f"T__{c}__{d}"] for d in DIRECTIONS} for c in CONDITIONS}
zT = _zdict(T)
print("max|z(T)|", max(float(np.abs(zT[c][d].numpy()).max()) for c in CONDITIONS for d in DIRECTIONS))
print("parity counts", np.bincount(b.ctx.parity), "n clusters", len(np.unique(b.cl)), "per pair", np.bincount(b.ctx.pair_index))
m = {c: z[f"margin__{c}"] for c in CONDITIONS}
print("margin dtype", m["a"].dtype, "min", min(m[c].min() for c in m), "taus", z["extra__taus"].tolist())
# number of margin values exactly equal to a tau (boundary sensitivity of the >= gate)
for t in z["extra__taus"]:
    print("   values == tau", t, int(sum((m[c] == t).sum() for c in m)))
print(f"done [{time.time()-t0:.0f}s]")
