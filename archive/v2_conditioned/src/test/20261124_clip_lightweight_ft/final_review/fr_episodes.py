"""Final review: independent re-derivation of the episode numbers of eval.json (own scoring, metrics, placement and
comparison code; only load_artelingo / artelingo_splits are shared, for the feature rows and the painting groups).

Run (CPU): CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python fr_episodes.py
Writes fr_episodes.json (gitignored) next to this file and prints the comparison summary.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from src.data.artelingo import load_artelingo  # noqa: E402
from src.data.artelingo_splits import artelingo_splits  # noqa: E402

E1 = ROOT / "src/test/20261030_aspect_baselines/results"
R3 = ROOT / "src/test/20261121_round3_affect_gate/results"
R4 = ROOT / "src/test/20261122_round4_aff_vetoes/results"
JOBS = ROOT / "res/cluster_jobs"
PAIRS = [("emotion", "style"), ("emotion", "genre"), ("style", "genre")]
FIELDS = ["anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt"]
SELECTED = {"LP": ("20261007-070950-65bb2f4", "LP_lr3e-4"), "LB": ("20261007-071151-65bb2f4", "LB_lr3e-5"),
            "LoRA": ("20261007-071251-65bb2f4", "LoRA_lr1e-4")}
CACHE_REF = ("20261007-071151-65bb2f4", "LB_lr3e-5")   # selected LB run's features_epoch0.npz


def npz_path(tag, name, f):
    return JOBS / tag / "code/outputs/clipft" / name / f


def my_sha(a, b, arrays):
    h = hashlib.sha256(f"{a}|{b}".encode())
    for k in FIELDS:
        h.update(np.ascontiguousarray(arrays[k], dtype=np.int64).tobytes())
    return h.hexdigest()


def unit32(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def strict_first(s, col):
    """1 where column col is strictly above all others and the row is finite."""
    t = s[:, col]
    o = np.delete(s, col, axis=1)
    return ((o < t[:, None]).all(1) & np.isfinite(s).all(1)).astype(np.float64)


def cf_metrics(img, txt, anchor, cand, dtype=np.float32):
    """Condition-free cosine: per-anchor r1, other, gain, either and per-side first-place rates (p_A, p_B)."""
    I = unit32(img).astype(dtype)
    T = unit32(txt).astype(dtype)
    fa, fb = [], []
    for q, c in ((I[anchor], T[cand]), (T[anchor], I[cand])):         # i2t, t2i
        s = (c * q[:, None, :]).sum(-1, dtype=dtype)
        fa.append(strict_first(s, 0))
        fb.append(strict_first(s, 1))
    side_a = 0.5 * (fa[0] + fa[1])      # p_A first, averaged over directions
    side_b = 0.5 * (fb[0] + fb[1])
    # condition a target col 0, condition b target col 1; the same scores under both conditions
    r1 = 0.5 * (side_a + side_b)
    other = 0.5 * (side_b + side_a)
    return {"r1": r1, "other": other, "gain": r1 - other, "either": r1 + other, "side_a": side_a, "side_b": side_b}


def boot(values, clusters, n=5000, seed=42, chunk=250):
    """Painting-cluster percentile bootstrap (the documented procedure), in percentage points."""
    _, inv = np.unique(np.asarray(clusters), return_inverse=True)
    k = inv.max() + 1
    s = np.bincount(inv, weights=np.asarray(values, float), minlength=k)
    c = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    out = []
    done = 0
    while done < n:
        m = min(chunk, n - done)
        d = rng.integers(0, k, size=(m, k))
        out.append(s[d].sum(1) / c[d].sum(1))
        done += m
    b = np.concatenate(out)
    return {"point": 100 * float(np.mean(values)), "ci95": [100 * float(np.percentile(b, 2.5)),
                                                             100 * float(np.percentile(b, 97.5))], "n_clusters": int(k)}


def main():
    data = load_artelingo()
    sp = artelingo_splits(data)
    groups = np.asarray(sp.groups)
    n = len(groups)
    sel = np.asarray(sp.selection)
    selmask = np.zeros(n, bool)
    selmask[sel] = True

    def place(path):
        z = np.load(path)
        rows, img, txt = z["rows"], z["img"], z["txt"]
        assert len(np.unique(rows)) == len(rows)
        fi = np.full((n, 512), np.nan, np.float32)
        ft = np.full((n, 512), np.nan, np.float32)
        k = selmask[rows]
        fi[rows[k]] = img[k]
        ft[rows[k]] = txt[k]
        assert np.isfinite(fi[sel]).all() and np.isfinite(ft[sel]).all()
        return fi, ft

    feats = {"plain": (np.where(selmask[:, None], data.img_features, np.nan).astype(np.float32),
                       np.where(selmask[:, None], data.txt_features, np.nan).astype(np.float32))}
    for v, (tag, name) in SELECTED.items():
        feats[f"ft:{v}"] = place(npz_path(tag, name, "features.npz"))
    feats["ft:CLIPcache"] = place(npz_path(*CACHE_REF, "features_epoch0.npz"))

    per_seed = {}
    flips64 = {}
    for s in (42, 49, 50, 51):
        z = np.load(E1 / f"episodes_seed{s}.npz")
        base = json.loads((E1 / f"baselines_seed{s}.json").read_text())
        anchor, cand, pidx = [], [], []
        for i, (a, b) in enumerate(PAIRS):
            arr = {k: z[f"{a}__{b}__{k}"] for k in FIELDS}
            assert my_sha(a, b, arr) == base["episodes_sha256"][f"{a}__{b}"], (s, a, b)
            rows_all = np.concatenate([arr[k].ravel() for k in FIELDS])
            assert selmask[rows_all].all(), "episode row outside selection"
            anchor.append(arr["anchor"].astype(np.int64))
            cand.append(arr["candidates"].astype(np.int64))
            pidx.append(np.full(len(arr["anchor"]), i))
        anchor, cand, pidx = np.concatenate(anchor), np.concatenate(cand), np.concatenate(pidx)
        cl = groups[anchor]
        sc = {}
        for name, (fi, ft) in feats.items():
            sc[name] = cf_metrics(fi, ft, anchor, cand)
            m64 = cf_metrics(fi, ft, anchor, cand, dtype=np.float64)
            flips64[f"{s}:{name}"] = int((m64["r1"] != sc[name]["r1"]).sum())
        # stored per-anchor arrays
        if s == 42:
            src = {"AFF": (R4 / "seed42_arrays.npz", "aff_fused"), "B": (R4 / "seed42_arrays.npz", "B"),
                   "Bp0": (R4 / "seed42_arrays.npz", "Bp0"), "Bp1": (R4 / "seed42_arrays.npz", "Bp1")}
        else:
            src = {"AFF": (R3 / f"go_seed{s}.npz", "aff_fused"), "B": (R3 / f"go_seed{s}.npz", "B"),
                   "Bp0": (R3 / f"go_seed{s}.npz", "Bp")}
        for name, (p, pre) in src.items():
            st = np.load(p)
            assert np.array_equal(st["cl"], cl) and np.array_equal(st["pair_index"], pidx), (s, name)
            # stored cosine next to it must equal my plain cosine
            assert np.array_equal(np.asarray(st["cosine__r1"], float), sc["plain"]["r1"]), (s, name, "cosine")
            r1 = np.asarray(st[f"{pre}__r1"], float)
            oth = np.asarray(st[f"{pre}__other"], float)
            g = np.asarray(st[f"{pre}__gain"], float)
            assert np.allclose(g, r1 - oth, atol=1e-12), (s, name, "gain identity")
            sc[name] = {"r1": r1, "other": oth, "gain": g, "either": r1 + oth}
        # plain also equals the round-1 stored per-anchor cosine
        pa = np.load(E1 / f"per_anchor_seed{s}.npz")
        for m in ("r1", "other", "gain"):
            assert np.array_equal(np.asarray(pa[f"cosine__{m}"], float), sc["plain"][m]), (s, m)
        per_seed[s] = {"cl": cl, "pi": pidx, "sc": sc}

    METR = ("r1", "either", "gain", "other")
    COMP = {"AFF_minus_ft": ("AFF", "ft"), "ft_minus_plain": ("ft", "plain"), "ft_minus_Bp0": ("ft", "Bp0")}
    VARS = ("LP", "LB", "LoRA", "CLIPcache")

    def scope(seeds, pair=None):
        cl = np.concatenate([per_seed[s]["cl"] for s in seeds])
        pi = np.concatenate([per_seed[s]["pi"] for s in seeds])
        mk = np.ones(len(cl), bool) if pair is None else pi == pair
        names = [k for k in per_seed[seeds[0]]["sc"] if all(k in per_seed[s]["sc"] for s in seeds)]
        val = {(k, m): np.concatenate([per_seed[s]["sc"][k][m] for s in seeds])[mk] for k in names for m in METR}
        out = {"n_episodes": int(mk.sum()), "scorers": {k: {m: boot(val[k, m], cl[mk]) for m in METR} for k in names},
               "comparisons": {}}
        for v in VARS:
            pick = (lambda t, v=v: f"ft:{v}" if t == "ft" else t)
            out["comparisons"][v] = {c: {m: boot(val[pick(a), m] - val[pick(b), m], cl[mk]) for m in METR}
                                     for c, (a, b) in COMP.items()}
        # per-side first-place rates for condition-free scorers (points only)
        out["side_rates"] = {k: {"A": 100 * float(np.concatenate([per_seed[s]["sc"][k]["side_a"] for s in seeds])[mk].mean()),
                                 "B": 100 * float(np.concatenate([per_seed[s]["sc"][k]["side_b"] for s in seeds])[mk].mean())}
                             for k in names if "side_a" in per_seed[seeds[0]]["sc"][k]}
        return out

    rep = {"seeds": {str(s): scope((s,)) for s in (42, 49, 50, 51)}, "pooled": scope((49, 50, 51)),
           "pairs": {str(i): scope((49, 50, 51), i) for i in range(3)},
           "pairs_seed42": {str(i): scope((42,), i) for i in range(3)},
           "pairs_per_seed": {f"{s}:{i}": scope((s,), i) for s in (49, 50, 51) for i in range(3)},
           "float64_r1_flips": flips64}
    (HERE / "fr_episodes.json").write_text(json.dumps(rep, indent=1))

    # compare against eval.json
    ev = json.loads((HERE.parent / "results/eval.json").read_text())
    n_cmp, worst, bad = 0, 0.0, []

    def walk(a, b, path):
        nonlocal n_cmp, worst
        if isinstance(a, dict):
            for k in a:
                if k in ("side_rates",):
                    continue
                if k not in b:
                    bad.append(f"missing in mine: {path}/{k}")
                    continue
                walk(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list):
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        elif isinstance(a, (int, float)) and not isinstance(a, bool):
            n_cmp += 1
            d = abs(float(a) - float(b))
            worst = max(worst, d)
            if d > 1e-9:
                bad.append(f"{path}: eval {a} mine {b}")

    for key in ("seeds", "pooled", "pairs", "pairs_seed42"):
        walk(ev[key], rep[key], key)
    print(f"compared {n_cmp} numeric leaves of eval.json; max |diff| {worst:.3e}; mismatches {len(bad)}")
    for b in bad[:20]:
        print("  ", b)
    print("float64 r1 flips (anchors whose r1 changes when scoring in float64):",
          {k: v for k, v in flips64.items() if v})


if __name__ == "__main__":
    main()
