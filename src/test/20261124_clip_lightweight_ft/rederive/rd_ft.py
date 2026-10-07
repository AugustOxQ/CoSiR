"""Independent recompute of the lightweight CLIP fine-tuning comparator (spec 2026-10-07 §4 selection, §5 evaluation).

Written without reading ft_eval.py, its tests, or results/eval.json / eval.log. Own code for: selection from
metrics.json, best-epoch verification (val retrieval recomputed from features.npz), feature placement, cosine scoring,
per-anchor R@1 / other / either, means and per-pair points. Uses src.eval.aspect_metrics.cluster_bootstrap for the
intervals (as the brief prescribes) and src.eval.aspect_episodes for the episode container and SHA-256.

Run (CPU only):
  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261124_clip_lightweight_ft/rederive/rd_ft.py \
    [--prep-cache <scratch>/rd_prep.npz]
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from src.eval.aspect_episodes import AspectEpisodes, concat_episodes, episodes_sha256  # noqa: E402
from src.eval.aspect_metrics import cluster_bootstrap  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNS_TXT = ROOT / ".superpowers/sdd/2026-10-07-clip-lightweight-ft/runs.txt"
JOBS = ROOT / "res/cluster_jobs"
E1 = ROOT / "src/test/20261030_aspect_baselines/results"
R3 = ROOT / "src/test/20261121_round3_affect_gate/results"
R4 = ROOT / "src/test/20261122_round4_aff_vetoes/results"
PAIRS = (("emotion", "style"), ("emotion", "genre"), ("style", "genre"))
PAIR_NAMES = [f"{a}__{b}" for a, b in PAIRS]
FIELDS = ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")
SEEDS = (42, 49, 50, 51)
POOL = (49, 50, 51)
VARIANTS = ("LP", "LB", "LoRA")
N_BOOT, BOOT_SEED, CHUNK = 5000, 42, 250
EXPECTED_ROWS = 308_723


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def now() -> str:
    return subprocess.run(["date", "+%F %H:%M"], capture_output=True, text=True,
                          env={"TZ": "Europe/Amsterdam"}).stdout.strip()


# ---------------------------------------------------------------- data (cached in scratch to save reloads)
def prepare(cache: Path | None) -> dict:
    if cache is not None and cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    from src.data.artelingo import load_artelingo
    from src.data.artelingo_splits import artelingo_splits
    data = load_artelingo()
    sp = artelingo_splits(data)
    n = len(data.sample_ids)
    assert n == EXPECTED_ROWS, n
    _, painting_code = np.unique(np.asarray(data.paintings), return_inverse=True)
    sel, val = np.sort(np.asarray(sp.selection, np.int64)), np.sort(np.asarray(sp.val, np.int64))
    out = {"n_rows": np.asarray(n), "selection": sel, "val": val, "held": np.asarray(sp.held, np.int64),
           "groups": np.asarray(sp.groups, np.int64), "painting_code": painting_code.astype(np.int64),
           "frozen_img_sel": data.img_features[sel].astype(np.float32),
           "frozen_txt_sel": data.txt_features[sel].astype(np.float32),
           "frozen_img_val": data.img_features[val].astype(np.float32),
           "frozen_txt_val": data.txt_features[val].astype(np.float32)}
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **out)
    return out


def full_array(n: int, rows: np.ndarray, values: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """(n, d) float32, NaN everywhere except rows in ``keep`` (taken from ``values`` aligned with ``rows``)."""
    out = np.full((n, values.shape[1]), np.nan, dtype=np.float32)
    pos = np.searchsorted(rows, keep)
    assert (rows[pos] == keep).all(), "a kept row is missing from the features file"
    out[keep] = values[pos]
    mask = np.zeros(n, bool)
    mask[keep] = True
    assert np.isnan(out[~mask]).all() and np.isfinite(out[mask]).all()
    return out


# ---------------------------------------------------------------- selection (spec §4)
def read_runs() -> list[dict]:
    runs = []
    for line in RUNS_TXT.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        node, variant, lr, ok, tag = parts[:5]
        runs.append({"node": node, "variant": variant, "lr_str": lr, "lr": float(lr), "ok": ok == "True", "tag": tag})
    return runs


def run_dir(run: dict) -> Path:
    d = JOBS / run["tag"] / "code/outputs/clipft" / f"{run['variant']}_lr{run['lr_str']}"
    assert d.is_dir() and "_smoke" not in d.name, d
    return d


def exact_selection(e: dict) -> Fraction:
    return (Fraction(e["i2t_correct"], e["n_images"]) + Fraction(e["t2i_correct"], e["n_captions"])) / 2


def select(runs: list[dict]) -> dict:
    out = {}
    for v in VARIANTS:
        vr = [r for r in runs if r["variant"] == v]
        assert len(vr) == 3 and all(r["ok"] for r in vr), v
        cands, per_run, epoch0 = [], [], []
        for r in vr:
            d = run_dir(r)
            m = json.loads((d / "metrics.json").read_text())
            rec = json.loads((d / "run_record.json").read_text())
            assert m["variant"] == v and abs(m["lr"] - r["lr"]) < 1e-15 and not m["smoke"], d
            assert rec["variant"] == v and abs(rec["lr"] - r["lr"]) < 1e-15 and not rec["smoke"], d
            eps = m["epochs"]
            assert [e["epoch"] for e in eps] == list(range(len(eps))), d
            for e in eps:
                fx = exact_selection(e)
                assert abs(float(fx) - e["selection"]) < 1e-15, (d, e["epoch"])
                assert abs(e["i2t_correct"] / e["n_images"] - e["i2t_r1"]) < 1e-15
                assert abs(e["t2i_correct"] / e["n_captions"] - e["t2i_r1"]) < 1e-15
                if e["epoch"] >= 1:
                    cands.append((-fx, r["lr"], e["epoch"], r, e))
            within = [e for e in eps if e["epoch"] >= 1]
            best_ge1 = min(within, key=lambda e: (-exact_selection(e), e["epoch"]))
            best_all = min(eps, key=lambda e: (-exact_selection(e), e["epoch"]))
            epoch0.append(eps[0]["selection"])
            per_run.append({"lr": r["lr"], "tag": r["tag"], "dir": str(d.relative_to(ROOT)), "n_epochs": len(eps),
                            "selection_by_epoch": [e["selection"] for e in eps],
                            "best_epoch_ge1": best_ge1["epoch"], "best_selection_ge1": best_ge1["selection"],
                            "best_epoch_all": best_all["epoch"],
                            "run_record_best_epoch": rec["best_epoch"],
                            "run_record_best_selection": rec["best_selection"],
                            "run_record_agrees": rec["best_epoch"] == best_ge1["epoch"]
                            and abs(rec["best_selection"] - best_ge1["selection"]) < 1e-15})
        cands.sort(key=lambda t: (t[0], t[1], t[2]))
        _, lr, ep, r, e = cands[0]
        ties = [c for c in cands if c[0] == cands[0][0]]
        # the same choice with the stored floats
        fl = min(((-c[4]["selection"], c[1], c[2]) for c in cands))
        assert (fl[1], fl[2]) == (lr, ep), "float and exact orderings disagree"
        assert len(set(epoch0)) == 1, f"{v}: epoch-0 selection differs between learning rates {epoch0}"
        out[v] = {"lr": lr, "epoch": ep, "selection": e["selection"], "i2t_r1": e["i2t_r1"], "t2i_r1": e["t2i_r1"],
                  "i2t_correct": e["i2t_correct"], "t2i_correct": e["t2i_correct"],
                  "n_tied_at_max": len(ties), "tag": r["tag"], "dir": str(run_dir(r).relative_to(ROOT)),
                  "epoch0_selection": epoch0[0], "epoch0_i2t_r1": None, "epoch0_t2i_r1": None, "runs": per_run}
        m0 = json.loads((run_dir(r) / "metrics.json").read_text())["epochs"][0]
        out[v]["epoch0_i2t_r1"], out[v]["epoch0_t2i_r1"] = m0["i2t_r1"], m0["t2i_r1"]
    return out


# ---------------------------------------------------------------- val retrieval from a features file (best-epoch check)
def unit32(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def val_retrieval(img_val: np.ndarray, txt_val: np.ndarray, paint_val: np.ndarray) -> dict:
    """Image->caption over all val captions (a painting's image = its first val row's image) and caption->image over
    the val paintings' images; argmax ties to the lower index."""
    uniq, first = np.unique(paint_val, return_index=True)
    rep_img = unit32(img_val[first])
    t = unit32(txt_val)
    cap_paint = np.searchsorted(uniq, paint_val)
    i2t = 0
    for s in range(0, len(uniq), 1024):
        sim = rep_img[s:s + 1024] @ t.T
        top = sim.argmax(axis=1)
        i2t += int((cap_paint[top] == np.arange(s, s + len(top))).sum())
    t2i = 0
    for s in range(0, len(t), 4096):
        sim = t[s:s + 4096] @ rep_img.T
        t2i += int((sim.argmax(axis=1) == cap_paint[s:s + 4096]).sum())
    # how far apart are the image features of rows of one painting?
    rep_full = img_val[first][cap_paint]
    spread = float(np.abs(img_val - rep_full).max())
    return {"i2t_correct": i2t, "t2i_correct": t2i, "n_images": len(uniq), "n_captions": len(t),
            "max_abs_within_painting_img_diff": spread}


def check_features(sel_info: dict, prep: dict) -> dict:
    out = {}
    val, selection = prep["val"], prep["selection"]
    want_rows = np.union1d(val, selection)
    paint_val = prep["painting_code"][val]
    for v, s in sel_info.items():
        d = ROOT / s["dir"]
        m = json.loads((d / "metrics.json").read_text())["epochs"]
        res = {}
        for fname, epoch in (("features.npz", s["epoch"]), ("features_epoch0.npz", 0)):
            z = np.load(d / fname, allow_pickle=False)
            rows, img, txt = z["rows"], z["img"], z["txt"]
            assert rows.dtype == np.int64 and img.dtype == np.float32 and txt.dtype == np.float32
            assert np.array_equal(rows, want_rows), f"{d}/{fname}: rows != val U selection (sorted)"
            pos = np.searchsorted(rows, val)
            vr = val_retrieval(img[pos], txt[pos], paint_val)
            per_epoch = [{"epoch": e["epoch"], "d_i2t": vr["i2t_correct"] - e["i2t_correct"],
                          "d_t2i": vr["t2i_correct"] - e["t2i_correct"]} for e in m]
            nearest = min(per_epoch, key=lambda x: (abs(x["d_i2t"]) + abs(x["d_t2i"]), x["epoch"]))
            res[fname] = {"claimed_epoch": epoch, "recomputed": vr,
                          "metrics_counts_at_claimed_epoch": {"i2t_correct": m[epoch]["i2t_correct"],
                                                              "t2i_correct": m[epoch]["t2i_correct"]},
                          "count_diff_at_claimed_epoch": [per_epoch[epoch]["d_i2t"], per_epoch[epoch]["d_t2i"]],
                          "nearest_epoch_by_counts": nearest["epoch"],
                          "count_diffs_by_epoch": [[x["d_i2t"], x["d_t2i"]] for x in per_epoch],
                          "sha256": sha256(d / fname)}
        if v == "LP":
            z0 = np.load(d / "features_epoch0.npz", allow_pickle=False)
            pos = np.searchsorted(z0["rows"], selection)
            res["LP_epoch0_equals_frozen_selection"] = {
                "img_max_abs": float(np.abs(z0["img"][pos] - prep["frozen_img_sel"]).max()),
                "txt_max_abs": float(np.abs(z0["txt"][pos] - prep["frozen_txt_sel"]).max())}
        out[v] = res
    return out


# ---------------------------------------------------------------- episodes, scoring, per-anchor metrics
def load_episodes(seed: int, in_sel: np.ndarray) -> AspectEpisodes:
    z = np.load(E1 / f"episodes_seed{seed}.npz", allow_pickle=False)
    base = json.loads((E1 / f"baselines_seed{seed}.json").read_text())
    assert list(z["pair_order"]) == PAIR_NAMES and base["pair_order"] == PAIR_NAMES
    parts = []
    for (a, b), name in zip(PAIRS, PAIR_NAMES):
        ep = AspectEpisodes(a, b, *(z[f"{name}__{k}"].astype(np.int64) for k in FIELDS))
        assert episodes_sha256(ep) == base["episodes_sha256"][name], (seed, name)
        assert in_sel[ep.rows()].all(), (seed, name)
        assert len(ep.anchor) == 4096
        parts.append(ep)
    return concat_episodes(parts)


def cos_scores(img: np.ndarray, txt: np.ndarray, ep: AspectEpisodes, dtype) -> dict:
    """Cosine per ranking row: i2t ranks the 13 candidates' captions for the anchor image, t2i their images for the
    anchor caption. Unit-normalisation in ``dtype``; NaN rows stay NaN."""
    def unit(x):
        x = np.asarray(x, dtype=dtype)
        return x / np.linalg.norm(x, axis=-1, keepdims=True)
    out = {}
    for d, (qf, cf) in (("i2t", (img, txt)), ("t2i", (txt, img))):
        q = unit(qf[ep.anchor])                      # (n, 512)
        c = unit(cf[ep.candidates])                  # (n, 13, 512)
        out[d] = np.matmul(c, q[:, :, None])[:, :, 0]
    return out


def hit(s: np.ndarray, col: int) -> np.ndarray:
    """1 where column ``col`` is strictly above every other column (ties miss); non-finite rows miss."""
    s = np.asarray(s, dtype=np.float64)
    others = np.concatenate([s[:, :col], s[:, col + 1:]], axis=1)
    ok = np.isfinite(s).all(axis=1)
    return ((s[:, [col]] > others).all(axis=1) & ok).astype(np.float64)


def anchor_metrics(scores_a: dict, scores_b: dict) -> dict:
    """Per-anchor R@1, other-aspect rate and either rate, averaged over i2t and t2i; condition a targets column 0,
    condition b column 1."""
    r1 = np.zeros(len(scores_a["i2t"]))
    other = np.zeros_like(r1)
    for d in ("i2t", "t2i"):
        r1 += 0.5 * 0.5 * (hit(scores_a[d], 0) + hit(scores_b[d], 1))
        other += 0.5 * 0.5 * (hit(scores_b[d], 0) + hit(scores_a[d], 1))
    return {"r1": r1, "other": other, "either": r1 + other}


def cosine_metrics(img, txt, ep, dtype=np.float32) -> dict:
    s = cos_scores(img, txt, ep, dtype)
    return anchor_metrics(s, s)


def stored(z, key: str) -> dict:
    r1, other = np.asarray(z[f"{key}__r1"], np.float64), np.asarray(z[f"{key}__other"], np.float64)
    return {"r1": r1, "other": other, "either": r1 + other}


def boot(values, clusters) -> dict:
    b = cluster_bootstrap(values, clusters, n_boot=N_BOOT, seed=BOOT_SEED, chunk=CHUNK)
    return {"point": 100 * b["point"], "ci95": [100 * c for c in b["ci95"]], "n_clusters": b["n_clusters"]}


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep-cache", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=HERE / "rd_ft.json")
    args = ap.parse_args()
    t0 = time.time()
    prep = prepare(args.prep_cache)
    n = int(prep["n_rows"])
    selection, groups = prep["selection"], prep["groups"]
    in_sel = np.zeros(n, bool)
    in_sel[selection] = True
    print(f"[{now()}] data ready ({time.time() - t0:.0f}s)", flush=True)

    runs = read_runs()
    sel = select(runs)
    for v, s in sel.items():
        print(f"  {v}: lr {s['lr']:g} epoch {s['epoch']} selection {s['selection']:.6f} "
              f"(epoch 0 {s['epoch0_selection']:.6f}; ties at max {s['n_tied_at_max']})", flush=True)
    feat_checks = check_features(sel, prep)
    for v, c in feat_checks.items():
        print(f"  {v}: features.npz diff@claimed {c['features.npz']['count_diff_at_claimed_epoch']} nearest "
              f"{c['features.npz']['nearest_epoch_by_counts']}; epoch0 diff "
              f"{c['features_epoch0.npz']['count_diff_at_claimed_epoch']}", flush=True)

    # full feature-row arrays, NaN outside selection
    feats = {"plain": (full_array(n, selection, prep["frozen_img_sel"], selection),
                       full_array(n, selection, prep["frozen_txt_sel"], selection))}
    for v, s in sel.items():
        z = np.load(ROOT / s["dir"] / "features.npz", allow_pickle=False)
        feats[v] = (full_array(n, z["rows"], z["img"], selection), full_array(n, z["rows"], z["txt"], selection))
    z0 = np.load(ROOT / sel["LB"]["dir"] / "features_epoch0.npz", allow_pickle=False)
    feats["LB_epoch0"] = (full_array(n, z0["rows"], z0["img"], selection),
                          full_array(n, z0["rows"], z0["txt"], selection))
    ft_names = ["LP", "LB", "LoRA", "LB_epoch0"]

    per_seed, clusters, pair_idx, checks = {}, {}, {}, {"plain_equals_baselines": {}, "float64_flips": {},
                                                          "stored_alignment": {}}
    for seed in SEEDS:
        ep = load_episodes(seed, in_sel)
        cl = groups[ep.anchor]
        pi = np.repeat(np.arange(3), 4096)
        m = {k: cosine_metrics(*feats[k], ep) for k in ["plain"] + ft_names}
        base = np.load(E1 / f"per_anchor_seed{seed}.npz", allow_pickle=False)
        eq = {k: bool(np.array_equal(m["plain"][k], base[f"cosine__{k}"])) for k in ("r1", "other")}
        eq["anchor_group"] = bool(np.array_equal(base["anchor_group"], cl))
        assert all(eq.values()), (seed, eq)
        checks["plain_equals_baselines"][seed] = eq
        checks["float64_flips"][seed] = {k: int((cosine_metrics(*feats[k], ep, np.float64)["r1"] != m[k]["r1"]).sum())
                                         for k in ["plain"] + ft_names}
        if seed == 42:
            z = np.load(R4 / "seed42_arrays.npz", allow_pickle=False)
            names = {"AFF": "aff_fused", "B": "B", "Bp_A0": "Bp0", "Bp_A1": "Bp1"}
            z3 = np.load(R3 / "seed42_arrays.npz", allow_pickle=False)
            checks["stored_alignment"]["r3_seed42_aff_equals_r4"] = bool(
                np.array_equal(z3["aff_fused__r1"], z["aff_fused__r1"]))
        else:
            z = np.load(R3 / f"go_seed{seed}.npz", allow_pickle=False)
            names = {"AFF": "aff_fused", "B": "B", "Bp_A0": "Bp"}
        al = {"cl": bool(np.array_equal(z["cl"], cl)), "pair_index": bool(np.array_equal(z["pair_index"], pi)),
              "cosine_r1": bool(np.array_equal(z["cosine__r1"], m["plain"]["r1"]))}
        assert all(al.values()), (seed, al)
        checks["stored_alignment"][seed] = al
        for k, key in names.items():
            m[k] = stored(z, key)
        per_seed[seed], clusters[seed], pair_idx[seed] = m, cl, pi
        print(f"[{now()}] seed {seed} scored", flush=True)

    pooled = {k: {mm: np.concatenate([per_seed[s][k][mm] for s in POOL]) for mm in ("r1", "other", "either")}
              for k in per_seed[49]}
    pooled_cl = np.concatenate([clusters[s] for s in POOL])
    pooled_pi = np.concatenate([pair_idx[s] for s in POOL])
    blocks = {**{str(s): (per_seed[s], clusters[s], pair_idx[s]) for s in SEEDS},
              "pooled_49_51": (pooled, pooled_cl, pooled_pi)}

    means = {b: {k: {mm: 100 * float(v[mm].mean()) for mm in ("r1", "other", "either")} for k, v in m.items()}
             for b, (m, _, _) in blocks.items()}
    diffs = {}
    for v in ft_names:
        diffs[v] = {}
        for label, (x, y) in (("AFF_minus_ft", ("AFF", v)), ("ft_minus_plain", (v, "plain")),
                              ("ft_minus_BpA0", (v, "Bp_A0"))):
            diffs[v][label] = {b: {mm: boot(m[x][mm] - m[y][mm], cl) for mm in ("r1", "either")}
                               for b, (m, cl, _) in blocks.items()}
        print(f"[{now()}] {v} differences done", flush=True)
    per_pair = {name: {k: {mm: 100 * float(v[mm][pooled_pi == i].mean()) for mm in ("r1", "either")}
                       for k, v in pooled.items()} for i, name in enumerate(PAIR_NAMES)}

    paint_cl = np.concatenate([prep["painting_code"][load_episodes(s, in_sel).anchor] for s in POOL])
    result = {
        "what": "independent recompute of the CLIP fine-tuning comparator (spec 2026-10-07 §4, §5); pp units",
        "written": now(),
        "scorers": {"plain": "frozen cached CLIP features, own cosine", "LP/LB/LoRA": "selected run features.npz",
                    "LB_epoch0": "untrained cache-path reference (selected LB run's features_epoch0.npz)",
                    "AFF": "aff_fused", "B": "B", "Bp_A0": "Bp (round 3) / Bp0 (round 4 seed 42)",
                    "Bp_A1": "Bp1 (round 4, seed 42 only)"},
        "sources": {"seeds_49_51": "src/test/20261121_round3_affect_gate/results/go_seed{s}.npz",
                    "seed_42": "src/test/20261122_round4_aff_vetoes/results/seed42_arrays.npz"},
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "chunk": CHUNK,
                      "clusters": "artelingo_splits groups of the anchor row (painting / identical-image groups), "
                                  "pooled = seeds 49, 50, 51 concatenated in that order",
                      "pooled_n_clusters_groups": int(len(np.unique(pooled_cl))),
                      "pooled_n_clusters_painting_strings": int(len(np.unique(paint_cl)))},
        "selection": sel,
        "feature_checks": feat_checks,
        "checks": checks,
        "means": means,
        "differences": diffs,
        "per_pair_pooled_49_51": per_pair,
        "runtime_s": round(time.time() - t0, 1),
    }
    args.out.write_text(json.dumps(result, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"[{now()}] wrote {args.out} sha256 {sha256(args.out)}", flush=True)


if __name__ == "__main__":
    main()
