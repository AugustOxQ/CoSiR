"""Final review B: run the held runner's regression mode through its own main() (the function the CLI calls; results go
to the snapshot's F/results), capturing what it computed in memory so that my own code (fr_b_check.py, a separate
process) can check it: the refit heads' selection posteriors (all five heads), the coefficient SHA-256s, the seed-42
bundle's arrays and every per-anchor array score_seed returned. Nothing is compared here.

    argv[1]: output npz (temp dir)
"""
import json
import sys
from pathlib import Path

F = Path("/project/CoSiR-r6-fr/src/test/20261125_artelingo_held_test")
sys.path.insert(0, str(F))
import run_r6_held as RH  # noqa: E402

import numpy as np  # noqa: E402

CAP = {}
_setup, _seed_bundle, _score = RH.setup, RH.seed_bundle, RH.S.score_seed


def setup():
    env = _setup()
    CAP["env"] = env
    return env


def seed_bundle(*a, **k):
    b = _seed_bundle(*a, **k)
    CAP["bundle"] = b
    return b


def score_seed(*a, **k):
    out = _score(*a, **k)
    CAP.setdefault("scored", []).append((k.get("include_pm"), out))
    return out


RH.setup, RH.seed_bundle, RH.S.score_seed = setup, seed_bundle, score_seed


def flat(prefix, x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            flat(f"{prefix}__{k}" if prefix else str(k), v, out)
    elif isinstance(x, (list, tuple)) and x and isinstance(x[0], dict):
        for i, v in enumerate(x):
            flat(f"{prefix}__{i}", v, out)
    else:
        out[prefix] = np.asarray(x)


def main():
    code = RH.main(["--mode", "regression"])
    env, b = CAP["env"], CAP["bundle"]
    arrs = {}
    sel = np.asarray(env.split.selection)
    post = RH.H.predict(env.heads, env.data, sel)
    for h in post:
        for m in ("img", "txt"):
            arrs[f"selpost__{h}__{m}"] = post[h][m][sel]
    arrs["selection"] = sel
    for k in ("cos", "t_n1u", "t6u_B", "t6u_B0", "t6u_B1", "stack", "F"):
        flat(f"bundle__{k}", getattr(b, k), arrs)
    for k in ("cl", "parity", "pair_index", "anchor"):
        arrs[f"bundle__{k}"] = np.asarray(getattr(b, k))
    for f in ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt"):
        arrs[f"ep__{f}"] = np.asarray(getattr(b.pooled, f))
    assert len(CAP["scored"]) == 1 and CAP["scored"][0][0] is True
    sc = CAP["scored"][0][1]
    for k, v in sc.items():
        if k in ("cl", "pair_index"):
            arrs[f"scored__{k}"] = np.asarray(v)
        else:
            flat(f"scored__{k}", v, arrs)
    meta = {"exit": code, "coef_sha256": env.coef_sha256, "episodes_sha256": b.episodes_sha256,
            "head_check": {k: v["equal"] for k, v in env.head_check["items"].items()},
            "value_sets": {k: [int(x) for x in v] for k, v in env.value_sets.items()}}
    np.savez(sys.argv[1], **arrs, meta=np.array(json.dumps(meta)))
    print(f"capture: exit {code}, {len(arrs)} arrays")


if __name__ == "__main__":
    main()
