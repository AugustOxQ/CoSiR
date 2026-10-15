"""Final review A2b, item 2: the CLIP H8 smoke of build 9 (run by A2 from r9-fr-a at 4458634 into its scratchpad).

Its pass record against r9_stats.pass_record_a28 recomputed on its own saved arrays, by my own reassembly (seed_index
split, my scorer dicts). Plus file-boundary checks: output SHAs, episode files' hashes, started copy, cluster count,
sensitivity counts, the disclosure fields against the pinned source files. Read-only on the smoke folder.
Usage: python fr_a2b_smoke_h8.py <smoke folder>. Writes fr_a2b_smoke_h8.json beside this file.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import r9_common as R  # noqa: E402
import r9_stats as ST  # noqa: E402
import r9_episodes as E  # noqa: E402
import r9_a28_data as AD  # noqa: E402
import run_r9_held as RH  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


d = Path(sys.argv[1])
ppath = d / "held_pass_H8.json"
rec = json.loads(ppath.read_text())
with np.load(d / "held_arrays_H8.npz") as z:
    arr = {k: z[k] for k in z.files}
per_seed = []
for i, _ in enumerate(rec["seeds"]):
    m = arr["seed_index"] == i
    x = {nm: {met: arr[f"{nm}__{met}"][m] for met in RH.METRICS} for nm in RH.READ_SCORERS}
    x["cl"], x["pair_index"] = arr["cl"][m], arr["pair_index"][m]
    per_seed.append(x)
extra = {k: rec[k] for k in ("episodes_sha256", "runner_sha256", "module_sha256")}
mine = ST.pass_record_a28(per_seed, "smoke", rec["seeds"], extra, set_aside=rec["set_aside"],
                          flagged_languages=rec["flagged_languages"], near_duplicates=rec["near_duplicates"])


def canon(v):
    return json.dumps(json.loads(json.dumps(v, default=float)), sort_keys=True)


same = {k: canon(rec.get(k)) == canon(mine.get(k)) for k in sorted(set(mine))}
# Episode files: hashes recomputed by load_episodes and compared with the pass record.
eps = {}
for s in rec["seeds"]:
    ep = E.load_episodes(d / RH.episodes_name("H8", s))
    m = arr["seed_index"] == rec["seeds"].index(s)
    eps[str(s)] = {"sha_equal": ep.sha == rec["episodes_sha256"][str(s)], "n": int(ep.n),
                   "pair_index_equal": bool(np.array_equal(ep.pair_index, arr["pair_index"][m]))}
sens = json.loads((d / "sensitivity_H8.json").read_text())
started = d / "held_started_H8.json"
copy = d / "folder" / "held_started_H8.json"
st = json.loads(started.read_text())
files = json.loads(AD.PINS_PATH.read_text())["files"]
flags_src = json.loads(Path(files["results/language_flags_a28.json"]["path"]).read_text())["flagged"]
near_src = json.loads(Path(rec["near_duplicates_source"]["file"]).read_text())
n_ep = len(rec["seeds"]) * len(R.PAIRS) * R.N_SMOKE
out = {
    "pass_sha256": sha(ppath),
    "pass_keys_equal": same, "all_equal": all(same.values()),
    "not_recomputed_keys": sorted(set(rec) - set(mine)),
    "mode": rec["mode"], "design": rec["design"], "seeds": rec["seeds"], "read": rec["read"],
    "backbone": rec["backbone"], "kind": rec["kind"],
    "n_anchors": int(len(arr["cl"])), "n_expected": n_ep, "n_episodes_field": rec["n_episodes"],
    "n_clusters_field": rec["n_clusters"], "n_clusters_mine": int(np.unique(arr["cl"]).size),
    "sens_N": sens.get("N"), "sens_n_paintings": sens.get("n_paintings"),
    "sens_episodes_sha_equal": sens.get("episodes_sha256") == rec["episodes_sha256"],
    "outputs_sha_equal": {n: sha(d / n) == h for n, h in rec["outputs"].items()},
    "episodes": eps,
    "started_attempts": len(st["attempts"]), "started_copy_equal": copy.read_bytes() == started.read_bytes(),
    "started_attempt_equal_pass": {k: st["attempts"][-1].get(k) == rec.get(k) for k in
                                   ("mode", "read", "backbone", "runner_sha256", "module_sha256", "coef_sha256",
                                    "pins_sha256", "input_sha256")},
    "flagged_equal_pinned_source": sorted(flags_src) == sorted(rec["flagged_languages"]),
    "near_equal_source": {k: int(near_src[k]) == rec["near_duplicates"][k] for k in ("cos_099", "cos_098")},
    "near_source_sha_equal": sha(rec["near_duplicates_source"]["file"]) == rec["near_duplicates_source"]["sha256"],
    "array_keys_with_share_or_rate": [k for k in arr if "share" in k or "rate" in k],
    "array_keys_secondary": [k for k in arr if k.startswith(("B1__", "r1_", "pm", "PM", "bilinear", "diag"))],
    "checks_n": {c: rec["checks"][c]["n"] for c in rec["checks"]},
}
(HERE / "fr_a2b_smoke_h8.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps({k: v for k, v in out.items() if k not in ("pass_keys_equal",)}, default=str)[:3000])
print("differ:", [k for k, v in same.items() if not v])
