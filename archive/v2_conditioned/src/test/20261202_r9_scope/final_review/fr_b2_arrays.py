"""Final review B2, check 3: the CLIP dry run of stage 2 (t13_clip_dryrun) against the stored arrays (own code, no
r9_* import). 96 array SHA-256s vs round 3's seed42_arrays.npz and AB's per_anchor_seed42.npz (C-contiguous bytes,
dtype, shape); gate share from aff_gate__a/b; regression_seed42.json split; seed-42 means and P-points vs round 6."""
import hashlib, json
from pathlib import Path
import numpy as np
T = Path("/project/CoSiR/src/test")
DRY = T / "20261202_r9_scope/results/t13_clip_dryrun"
reg_p = T / "20261125_artelingo_held_test/results/regression_seed42.json"
pc_p = T / "20261125_artelingo_held_test/results/seed42_pass_counts.json"
def sha_b(b): return hashlib.sha256(b).hexdigest()
out = {}
reg = json.loads(reg_p.read_text())
out["regression_sha256"] = sha_b(reg_p.read_bytes())
items = reg["items"]
n_r3 = [k for k in items if k.startswith("seed42_arrays/")]
n_ab = [k for k in items if k.startswith("per_anchor_seed42/")]
out["regression_split"] = {"n_items": len(items), "n_failed": sum(not v.get("equal") for v in items.values()),
                           "seed42_arrays": len(n_r3), "per_anchor_seed42": len(n_ab),
                           "other": len(items) - len(n_r3) - len(n_ab),
                           "record_n_items": reg["n_items"], "record_n_failed": reg["n_failed"]}
seed = json.loads((DRY / "stage2_seed42_clip_b32.json").read_text())
arrays = seed["arrays"]
srcs = {"seed42_arrays": np.load(T / "20261121_round3_affect_gate/results/seed42_arrays.npz"),
        "per_anchor_seed42": np.load(T / "20261030_aspect_baselines/results/per_anchor_seed42.npz")}
bad, n_ok = [], 0
for key, rec in arrays.items():
    src, name = key.split("/", 1)
    a = srcs[src][name]
    mine = {"sha256": sha_b(np.ascontiguousarray(a).tobytes()), "dtype": str(a.dtype), "shape": list(a.shape)}
    if mine != {k: rec[k] for k in ("sha256", "dtype", "shape")}:
        bad.append({"key": key, "mine": mine, "record": rec})
    else:
        n_ok += 1
out["arrays"] = {"n_record": len(arrays), "n_equal": n_ok, "mismatch": bad,
                 "record_keys_equal_regression_array_items": sorted(arrays) == sorted(n_r3 + n_ab),
                 "regression_array_items_missing_from_record": sorted(set(n_r3 + n_ab) - set(arrays)),
                 "npz_keys_not_in_record": {s: sorted(set(f"{s}/{k}" for k in z.files) - set(arrays))
                                            for s, z in srcs.items()}}
# gate share, own code: tau index of cell c is c // 56 (224 cells = 4 taus x 56); tune half h, episodes of parity 1-h
z = srcs["seed42_arrays"]
cells = [int(c) for c in z["aff_fused_cells"]]
parity = z["parity"]
n_open = n_tot = 0
for h, c in enumerate(cells):
    for g in ("a", "b"):
        gate = z[f"aff_gate__{g}"][c // 56][parity == 1 - h]
        assert set(np.unique(gate).tolist()) <= {0, 1}
        n_open += int(gate.sum()); n_tot += gate.size
dry2 = json.loads((DRY / "stage2_clip_b32.json").read_text())
out["gate_share"] = {"cells": cells, "gate_shape": list(z["aff_gate__a"].shape), "open": n_open, "total": n_tot,
                     "share": n_open / n_tot, "record_stage2": dry2["gate_share"],
                     "aff_cf_cells": [int(c) for c in z["aff_cf_cells"]], "aff_sigma": z["aff_sigma"].tolist(),
                     "r1_cells": [int(c) for c in z["r1_fused_cells"]], "r1_cf_cells": [int(c) for c in z["r1_cf_cells"]],
                     "r1_sigma": z["r1_sigma"].tolist(), "taus": z["taus"].tolist()}
# the other parity's gate share with tau of the same half, to show the parity matters
alt_open = alt_tot = 0
for h, c in enumerate(cells):
    for g in ("a", "b"):
        gate = z[f"aff_gate__{g}"][c // 56][parity == h]
        alt_open += int(gate.sum()); alt_tot += gate.size
out["gate_share"]["same_parity_share_would_be"] = alt_open / alt_tot
# seed-42 means: AFF and CF from round 3 arrays, B/B0/B1 from round 6's picks file
picks6 = json.loads((T / "20261125_artelingo_held_test/results/picks_seed42.json").read_text())
mine_means = {"AFF": float(np.mean(z["aff_fused__r1"]) * 100), "CF": float(np.mean(z["aff_cf__r1"]) * 100),
              **{k: picks6["mean_r1"][k] for k in ("B", "B0", "B1")}}
out["mean_r1"] = {"mine": mine_means, "record": seed["mean_r1"],
                  "equal": all(mine_means[k] == seed["mean_r1"][k] for k in mine_means)}
# P-points and n_j against round 6's seed42_pass_counts.json
pc6 = json.loads(pc_p.read_text())
out["pass_counts_sha256"] = sha_b(pc_p.read_bytes())
RUN = {"rule_sha256", "runner_sha256", "module_sha256", "time", "input_sha256"}
def jd(a, b, p=""):
    if type(a) is not type(b): return [f"{p}: type"]
    if isinstance(a, dict):
        o = [f"{p}/{k}: one side" for k in set(a) ^ set(b)]
        return o + [d for k in a if k in b for d in jd(a[k], b[k], f"{p}/{k}")]
    if isinstance(a, list):
        return [f"{p}: len"] if len(a) != len(b) else [d for i, (x, y) in enumerate(zip(a, b)) for d in jd(x, y, f"{p}[{i}]")]
    return [] if a == b else [f"{p}: {a!r} != {b!r}"]
got = {k: v for k, v in seed["pass_counts"].items() if k not in RUN}
want = {k: v for k, v in pc6.items() if k not in RUN}
out["pass_counts"] = {"diff": jd(got, want), "fields_compared": sorted(want)}
out["checks_vs_pc6"] = [f"{p}: {seed['checks'][p]} vs point {pc6['checks'][p]['point']} n {pc6['checks'][p]['n']}"
                        for p in seed["checks"] if seed["checks"][p]["point"] != pc6["checks"][p]["point"]
                        or seed["checks"][p]["n_j"] != pc6["checks"][p]["n"]]
Path(__file__).with_suffix(".json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: (v if len(str(v)) < 700 else str(v)[:700]) for k, v in out.items()}, indent=1))
