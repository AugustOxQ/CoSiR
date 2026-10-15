"""Final review B2, check 1: stage 0b item 1 to 3 with own comparer (no r9_* import).
Every file of the replay folder against CLIP's pinned file, in full: npz (keys both ways, dtype, shape, elements,
NaN == NaN in place), json (whole file, exact types; every differing path listed), pkl (bytes), A3 checkpoint
(every tensor: dtype, shape, torch.equal; every non-tensor field)."""
import hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
MAIN = Path("/project/CoSiR")
SNAP = Path("/project/CoSiR-wt/r9-fr-b2")
REPLAY = Path("/project/CoSiR-wt/r9-replay/src/test/20261202_r9_scope/results/replay/clip_b32")
pins = json.loads((SNAP / "src/test/20261202_r9_scope/pins_clip.json").read_text())
def pinned(role):
    rel = pins["parts"][role]["path"]
    base = SNAP if rel.startswith("src/test/20261202_r9_scope/") else MAIN
    return base / rel
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arr_diff(a, b):
    if a.dtype != b.dtype: return f"dtype {a.dtype} != {b.dtype}"
    if a.shape != b.shape: return f"shape {a.shape} != {b.shape}"
    if a.dtype.kind == "f":
        same = (a == b) | (np.isnan(a) & np.isnan(b))
    elif a.dtype.kind == "c":
        same = (a == b)
    else:
        same = (a == b)
    n = int((~same).sum())
    return None if n == 0 else f"{n} of {a.size} differ"
def jdiff(a, b, p=""):
    if type(a) is not type(b): return [f"{p}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, dict):
        out = [f"{p}/{k}: only replay" for k in a if k not in b] + [f"{p}/{k}: only pinned" for k in b if k not in a]
        if list(a) != list(b) and not out: out.append(f"{p}: key order differs")
        for k in a:
            if k in b: out += jdiff(a[k], b[k], f"{p}/{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b): return [f"{p}: len {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in jdiff(x, y, f"{p}[{i}]")]
    if isinstance(a, float) and a != a and b != b: return []
    return [] if a == b else [f"{p}: {str(a)[:80]} != {str(b)[:80]}"]
res = {}
NPZ = ["partitions", "graph", "bank_AIC", "n6_posteriors", "step1_heads_style", "rb_heads_affect", "rb_heads_image",
       "rb_heads_caption", "rb_heads_csd", "rb_heads_rand", "rb_bank_A0_half0", "rb_bank_A0_half1"]
for role in NPZ:
    rp, pp = REPLAY / pinned(role).name, pinned(role)
    with np.load(rp, allow_pickle=False) as a, np.load(pp, allow_pickle=False) as b:
        keys_a, keys_b = sorted(a.files), sorted(b.files)
        diffs = {k: arr_diff(a[k], b[k]) for k in keys_b if k in keys_a}
        nan_keys = [k for k in keys_b if b[k].dtype.kind == "f" and np.isnan(b[k]).any()]
        res[role] = {"keys": len(keys_b), "keys_equal": keys_a == keys_b,
                     "differ": {k: v for k, v in diffs.items() if v}, "keys_with_nan": nan_keys,
                     "file_sha_equal": sha(rp) == sha(pp)}
for role in ["rb_reader_A0_json", "rc_tau", "baselines_seed42", "picks_seed42", "cells_seed42"]:
    rp, pp = REPLAY / pinned(role).name, pinned(role)
    if role == "cells_seed42": rp = REPLAY / "cells_seed42.json"
    d = jdiff(json.loads(rp.read_text()), json.loads(pp.read_text()))
    res[role] = {"n_diff": len(d), "diff_paths": d[:60], "file_sha_equal": sha(rp) == sha(pp)}
res["rb_reader_A0"] = {"file_sha_equal": sha(REPLAY / "rb_reader_A0.pkl") == sha(pinned("rb_reader_A0"))}
for role in ["picked"]:
    rp, pp = REPLAY / "a3_L1" / pinned(role).name, pinned(role)
    d = jdiff(json.loads(rp.read_text()), json.loads(pp.read_text()))
    res[role] = {"n_diff": len(d), "diff_paths": d[:40], "file_sha_equal": sha(rp) == sha(pp)}
import torch
a = torch.load(REPLAY / "a3_L1/A3_seed42.pt", map_location="cpu", weights_only=False)
b = torch.load(pinned("A3_seed42"), map_location="cpu", weights_only=False)
def tdiff(x, y, p=""):
    if type(x) is not type(y): return [f"{p}: type {type(x).__name__} != {type(y).__name__}"]
    if torch.is_tensor(x):
        if x.dtype != y.dtype or x.shape != y.shape: return [f"{p}: {x.dtype}{tuple(x.shape)} != {y.dtype}{tuple(y.shape)}"]
        return [] if torch.equal(x, y) else [f"{p}: values differ"]
    if isinstance(x, dict):
        out = [f"{p}/{k}: only replay" for k in x if k not in y] + [f"{p}/{k}: only pinned" for k in y if k not in x]
        for k in x:
            if k in y: out += tdiff(x[k], y[k], f"{p}/{k}")
        return out
    if isinstance(x, (list, tuple)):
        if len(x) != len(y): return [f"{p}: len"]
        return [d for i, (u, v) in enumerate(zip(x, y)) for d in tdiff(u, v, f"{p}[{i}]")]
    if isinstance(x, float) and x != x and y != y: return []
    return [] if x == y else [f"{p}: {str(x)[:60]} != {str(y)[:60]}"]
td = tdiff(a, b)
n_t = sum(1 for _ in (a.get("model_state_dict") or a.get("state_dict") or {}).values()) if isinstance(a, dict) else None
res["A3_seed42"] = {"top_keys": sorted(a) if isinstance(a, dict) else str(type(a)), "n_state_tensors": n_t,
                    "differences": td[:40]}
Path(__file__).with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n")
for k, v in res.items():
    print(k, {kk: (vv if not isinstance(vv, (list, dict)) or len(str(vv)) < 300 else str(vv)[:300]) for kk, vv in v.items()})
