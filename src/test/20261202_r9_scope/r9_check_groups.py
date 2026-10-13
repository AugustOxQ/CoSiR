"""Compare a recipe-groups output directory with a backbone's stored parts, array by array (stage 0b, ticket 07).

    python r9_check_groups.py --backbone clip_b32 --dir results/replay/clip_b32

Checks every file of the directory that exists (partitions, graph, bank_AIC, and the two posterior files with the
keys r9_check_heads compares): same key set, dtype, shape and values. Exit 0 only if every present file matches.
"""
import argparse
from pathlib import Path

import r9_common as R  # noqa: E402  (first)
from r9_backbone import BACKBONES, parts

import numpy as np
from scipy.sparse import load_npz

ROLES = ("partitions", "graph", "bank_AIC", "n6_posteriors", "step1_heads_style")


def _arrays(path):
    if Path(path).name == "graph.npz":
        g = load_npz(path).tocsr()
        return {"data": g.data, "indices": g.indices, "indptr": g.indptr, "shape": np.asarray(g.shape)}
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def compare(got_path, want_path, label):
    got, want = _arrays(got_path), _arrays(want_path)
    ok = set(got) == set(want)
    print(f"{label}/keys: {'pass' if ok else 'FAIL'}", flush=True)
    for key in want:
        same = (key in got and got[key].dtype == want[key].dtype and got[key].shape == want[key].shape
                and np.array_equal(got[key], want[key]))
        ok &= bool(same)
        print(f"{label}/{key}: {'pass' if same else 'FAIL'}", flush=True)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", choices=BACKBONES, required=True)
    ap.add_argument("--dir", type=Path, required=True)
    args = ap.parse_args(argv)
    stored = parts(args.backbone)
    passed, seen = True, 0
    for role in ROLES:
        got = args.dir / stored[role].name
        if got.exists():
            seen += 1
            passed &= compare(got, stored[role], role)
    assert seen, f"no recipe output in {args.dir}"
    print(f"{args.backbone}: {seen} files, {'PASS' if passed else 'FAIL'}", flush=True)
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
