"""Rule check (read only, seed 42): AFF assembled at the recorded cells (fused 39/119, counterpart 149/10) through the
rule's float32 path; per aspect pair, the means of B', the counterpart and B, to see whether a per-pair comparator
(one reading of D12's "episodes considered") would change §5 item 3's per-pair bar margins. Prints only.
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261121_round3_affect_gate/rule_check/rc3_perpair_comparator.py
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T = HERE.parents[1]
sys.path.insert(0, str(T / "20261118_reader_fix_round2"))
import r2_common as R  # noqa: E402
import r2_fusion as F  # noqa: E402

C, K = R.C, R.K
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, per_anchor  # noqa: E402
from src.eval.aspect_nested import _zdict  # noqa: E402

z = np.load(T / "20261120_r1_levers_brainstorm/cache/bs_cache.npz")
cl, pi, par = z["anchor_group"], z["pair_index"], z["parity"]
Bd = {c: {d: z[f"B__{d}"] for d in DIRECTIONS} for c in CONDITIONS}
stack = {d: z[f"stack4__{d}"][:, :3] for d in DIRECTIONS}
P = {c: z[f"P_R1__{c}"] for c in CONDITIONS}
pB = {m: z[f"pB__{m}"] for m in METRICS}
pBp = {m: z[f"pBpA0__{m}"] for m in METRICS}
taus = json.loads((T / "20261117_reader_fix_csd/results/rc_tau.json").read_text())["taus"]
Tt = C.expected_term(stack, P)
m = {c: C.top_two_margin(P[c]) for c in CONDITIONS}
pk = {c: np.asarray(P[c]).argmax(1) for c in CONDITIONS}
zB, zT = _zdict(Bd), _zdict(Tt)
info = F.rank_info(Bd)
g = {t: {c: (gg[c] * (pk[c] == 0)).astype(np.float32) for c in CONDITIONS} for t, gg in K.gates(m, taus).items()}
gated = {t: K.gated_terms(zT, g[t]) for t in g}
G = {t: K.g_cf(gated[t]) for t in g}
pn = per_anchor(F.assemble(zB, info, gated, {0: 39, 1: 119}, par))
pc = per_anchor(F.assemble(zB, info, G, {0: 149, 1: 10}, par))
for i, p in enumerate(C.POOLED_ORDER):
    mk = pi == i
    means = {k: 100 * float(np.mean(v["r1"][mk])) for k, v in (("B_prime", pBp), ("counterpart", pc), ("B", pB))}
    best = max(means, key=lambda k: (means[k], -["B_prime", "counterpart", "B"].index(k)))
    pooled = 100 * float(np.mean(pn["r1"][mk] - pBp["r1"][mk]))
    own = 100 * float(np.mean(pn["r1"][mk] - {"B_prime": pBp, "counterpart": pc, "B": pB}[best]["r1"][mk]))
    print(p, {k: round(v, 4) for k, v in means.items()}, "per-pair comparator:", best,
          f"bar vs pooled comparator (B') {pooled:.6f}; vs per-pair comparator {own:.6f}")
