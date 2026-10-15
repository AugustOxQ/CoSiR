"""Final review A2, item 3: the CLIP H6 smoke (run into the scratchpad) - its pass record against r9_stats on its
own saved arrays, by my own reassembly. Usage: python fr_a2_smoke_check.py <smoke folder>. Writes
fr_a2_smoke_check.json beside this file."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import r9_stats as ST  # noqa: E402
import run_r9_held as RH  # noqa: E402

d = Path(sys.argv[1])
rec = json.loads((d / "held_pass_H6.json").read_text())
with np.load(d / "held_arrays_H6.npz") as z:
    arr = {k: z[k] for k in z.files}
per_seed = []
for i, _ in enumerate(rec["seeds"]):
    m = arr["seed_index"] == i
    x = {nm: {met: arr[f"{nm}__{met}"][m] for met in RH.METRICS} for nm in RH.READ_SCORERS}
    x["cl"], x["pair_index"] = arr["cl"][m], arr["pair_index"][m]
    per_seed.append(x)
extra = {k: rec[k] for k in ("episodes_sha256", "runner_sha256", "module_sha256")}
mine = ST.pass_record(per_seed, "smoke", rec["seeds"], extra, secondary=False)
same = {k: json.dumps(rec.get(k), sort_keys=True) == json.dumps(json.loads(json.dumps(mine.get(k), default=float)),
                                                                  sort_keys=True)
        for k in sorted(set(mine))}
started = json.loads((d / "held_started_H6.json").read_text())
copy = d / "folder" / "held_started_H6.json"
out = {"pass_keys_equal": same, "all_equal": all(same.values()), "seeds": rec["seeds"], "mode": rec["mode"],
       "n_anchors": int(len(arr["cl"])), "array_keys_with_share_or_rate": [k for k in arr if "share" in k or "rate" in k],
       "array_keys_secondary": [k for k in arr if k.startswith(("B1__", "r1_", "bilinear", "diag"))],
       "started_attempts": len(started["attempts"]), "started_copy_equal": copy.read_bytes() == (d / "held_started_H6.json").read_bytes(),
       "checks_n": {c: rec["checks"][c]["n"] for c in rec["checks"]}}
(HERE / "fr_a2_smoke_check.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items() if k != "pass_keys_equal"}))
print([k for k, v in same.items() if not v])
