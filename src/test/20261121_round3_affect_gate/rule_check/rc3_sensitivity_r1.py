"""Rule check (read only, seed 42): the rule's §6.1 projection applied to R1 (= round-1 R-c) fused minus counterpart,
to verify the prior's "detectable margin of about 0.2". Prints only; writes nothing.
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261121_round3_affect_gate/rule_check/rc3_sensitivity_r1.py
"""
from pathlib import Path

import numpy as np

T = Path(__file__).resolve().parents[2]
z = np.load(T / "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.npz")
cl = np.load(T / "20261120_r1_levers_brainstorm/cache/bs_cache.npz")["anchor_group"]
x = 100 * (z["fused__r1"].astype(np.float64) - z["cf__r1"].astype(np.float64))
_, idx = np.unique(cl, return_inverse=True)
P, n = idx.max() + 1, len(x)
m = np.bincount(idx, minlength=P).astype(np.float64)
s = np.bincount(idx, weights=x, minlength=P)
mean_p = s / m
grand = x.mean()
ss_within = ((x - mean_p[idx]) ** 2).sum()
ss_between = (m * (mean_p - grand) ** 2).sum()
ms_w = ss_within / (n - P)
ms_b = ss_between / (P - 1)
n0 = (n - (m ** 2).sum() / n) / (P - 1)
sa2 = max(0.0, (ms_b - ms_w) / n0)
se2 = (sa2 * (9 * (m ** 2).sum() - 6 * n) + ms_w * 3 * n) / (3 * n) ** 2
se = np.sqrt(se2)
print(f"n={n} P={P} sum m^2={int((m**2).sum())} n0={n0:.4f} sigma_eps^2={ms_w:.4f} sigma_a^2={sa2:.4f}")
print(f"point {grand:.4f}; projected SE {se:.4f}; half-width {1.96*se:.4f}; detectable x = 2.80 SE = {2.8*se:.4f}")
