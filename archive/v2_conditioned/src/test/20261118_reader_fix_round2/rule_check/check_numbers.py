"""Read-only: verify round-1 numbers the draft rule quotes against round 1's stored results."""
import json
from pathlib import Path
import numpy as np
R = Path("/project/CoSiR/src/test/20261117_reader_fix_csd/results")
def J(n): return json.loads((R / n).read_text())
rc = J("cand_Rc_Rb_expected_A0.json")
print("Rc r1_means", rc["r1_means"])
print("Rc margin r1", rc["margin"]["r1"])
print("Rc bar", rc["bar"]["comparator"], rc["bar"]["r1"])
print("Rc gain_statistic", rc["gain_statistic"])
print("Rc n_episodes, n_clusters", rc["n_episodes"], rc["n_clusters"])
print("Rc crossfit", json.dumps(rc["crossfit"], indent=0))
print("Rc taus", rc["taus"])
print("Rc pick_accuracy", rc["pick_accuracy"]["correct_share"])
print("Rc tau0_sanity", rc.get("tau0_sanity"))
print("Rc keys", list(rc.keys()))
tau = J("rc_tau.json"); print("rc_tau", tau["taus"], tau["n_values"], tau["parent"])
for n in ("cand_Rb_expected_A0.json", "cand_Rb_expected_A1.json", "cand_Rb_argmax_A0.json"):
    r = J(n)
    print(n, "means", {k: round(v, 6) for k, v in r["r1_means"].items()}, "margin", r["margin"]["r1"], "bar", r["bar"]["comparator"], r["bar"]["r1"], "gain", r["gain_statistic"], "pick", r["pick_accuracy"]["correct_share"]["point"])
z = np.load(R / "cand_Rc_Rb_expected_A0.npz")
print("Rc npz keys", sorted(z.files))
for k in sorted(z.files):
    print("  ", k, z[k].dtype, z[k].shape)
print((R / "rule_application.txt").read_text())
