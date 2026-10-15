"""E1: Tier-1 baselines on ArtELingo selection aspect episodes (CVPR plan Task 9).

Run from the repo root (CPU only; codes are encoded once on CPU and cached):
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python src/test/20261030_aspect_baselines/run_baselines.py --episodes-seed 42
Only selection rows are finite in every evaluation array. PCA basis and PairScaler use training rows (no labels).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")           # CPU only: never touch the shared GPU
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
torch.set_num_threads(8)

from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.eval.aspect_episodes import (PaintingValueIndex, build_aspect_episodes, concat_episodes,  # noqa: E402
                                      episodes_sha256, validate_aspect_episodes)
from src.eval.aspect_metrics import METRICS, per_anchor, summarize  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, agreement_term, cosine_scores, crossfit_lambda  # noqa: E402
from src.eval.pair_metric_baselines import (bilinear_agreement_term, diag_agreement_term, fit_pair_scaler,  # noqa: E402
                                            fit_pca_basis, kissme_term, pair_probe_term, rca_term,
                                            tip_adapter_term, value_prototype_term, wang_term, xing_term)

PAIRS = (("emotion", "style", "genre"), ("emotion", "genre", "style"), ("style", "genre", "emotion"))
GO_CANDIDATES = ("diag", "diag_relu", "bilinear", "kissme", "rca", "xing", "wang", "probe", "tip")
FIELDS = ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")
T0 = perf_counter()


def log(*a):
    print(f"[{perf_counter() - T0:7.1f}s]", *a, flush=True)


def load_ra():
    spec = importlib.util.spec_from_file_location("ra", ROOT / "src/test/20261018_affect_factor_learning/run_affect.py")
    ra = importlib.util.module_from_spec(spec)
    sys.modules["ra"] = ra
    spec.loader.exec_module(ra)
    return ra


R3_SOURCE = "stage (d) cache img_codes / txt_codes (original R3, trained on all train rows)"


def get_codes(ra, data, cache):
    """SE, C0, R3 codes (scorer-train + selection rows finite). Encoded once on CPU, cached to results/codes_*.npz.
    Returns (codes, provenance); the checkpoint is re-hashed on every call, also on a cache hit."""
    ra.grid.DEVICE = "cpu"                                   # model_codes reads grid.DEVICE at call time
    prep_record = json.loads((ra.CACHE / "affect_prepare.json").read_text())
    codes, prov = {}, {}
    for name in ("SE", "C0", "R3"):
        path = HERE / "results" / f"codes_{name}.npz"
        if name == "R3":
            prov[name] = {"source": R3_SOURCE}
        else:
            ckpt = ra.model_path(name)
            prov[name] = {"checkpoint": str(ckpt.relative_to(ROOT)), "sha256": ra.grid.sha256_file(ckpt)}
        if path.exists():
            z = np.load(path)
            codes[name] = (z["img"], z["txt"])
        else:
            ic, tc, info = ra.model_codes(name, data, cache, prep_record)
            if name != "R3":
                assert info["sha256"] == prov[name]["sha256"], name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, img=ic, txt=tc)
            codes[name] = (ic, tc)
            log(f"encoded {name} codes on CPU: {info}")
    return codes, prov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=None, help="episodes per pair (default 4096; 64 with --smoke)")
    ap.add_argument("--overwrite", action="store_true", help="replace existing outputs in results/")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n = args.n or (64 if args.smoke else 4096)
    seed = args.episodes_seed
    out_dir = HERE / "results" / ("smoke" if args.smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / f"{k}_seed{seed}.{e}" for k, e in (("episodes", "npz"), ("per_anchor", "npz"), ("baselines", "json"))]
    clash = [t.name for t in targets if t.exists()]
    if clash and not args.smoke and not args.overwrite:
        sys.exit(f"refusing to overwrite {clash} in {out_dir}; pass --overwrite to replace them")

    ra = load_ra()
    data = ra.load_artelingo()
    cache, *_ = ra.grid.load_grid()
    sp = artelingo_splits(data)
    labels = artelingo_aspect_labels(data)
    groups = sp.groups
    selection, scorer_train = np.asarray(sp.selection), np.asarray(sp.scorer_train)
    n_rows = len(groups)
    in_sel = np.zeros(n_rows, dtype=bool)
    in_sel[selection] = True

    # ---- features and codes masked to selection rows
    img = ra.sel.masked(data.img_features, selection)
    txt = ra.sel.masked(data.txt_features, selection)
    assert np.isnan(img[~in_sel]).all() and np.isnan(txt[~in_sel]).all()
    assert np.isfinite(img[in_sel]).all() and np.isfinite(txt[in_sel]).all()
    raw_codes, code_prov = get_codes(ra, data, cache)
    if not args.smoke:
        (HERE / "results" / "codes_provenance.json").write_text(json.dumps(code_prov, indent=1))
    codes = {}
    for k, (ic, tc) in raw_codes.items():
        ic, tc = ra.sel.masked(ic, selection), ra.sel.masked(tc, selection)
        assert np.isnan(ic[~in_sel]).all() and np.isnan(tc[~in_sel]).all(), k
        assert np.isfinite(ic[in_sel]).all() and np.isfinite(tc[in_sel]).all(), k
        codes[k] = (ic, tc)
    log("loaded data, splits, features, codes (selection rows only)")

    # ---- basis and scaler: the same 60,000 scorer-train rows, no labels
    fit_rows = np.random.default_rng(0).choice(scorer_train, 60000, replace=False)
    assert not in_sel[fit_rows].any()
    fit_sha = hashlib.sha256(np.ascontiguousarray(fit_rows, dtype=np.int64).tobytes()).hexdigest()
    fi, ft = (np.asarray(x[fit_rows], np.float32) for x in (data.img_features, data.txt_features))
    fi, ft = (x / np.linalg.norm(x, axis=1, keepdims=True) for x in (fi, ft))   # unit rows, like EvalInputs
    basis = fit_pca_basis(fi, ft)
    scaler = fit_pair_scaler(fi, ft)
    log("fitted PCA basis and PairScaler on 60,000 scorer-train rows")

    # ---- episodes
    index = PaintingValueIndex(labels, groups)
    parts, shas, saved = [], {}, {}
    for a, b, third in PAIRS:
        ep = build_aspect_episodes(labels, groups, selection, a, b, n, seed, third=third, index=index)
        validate_aspect_episodes(ep, labels, groups, index, third=third)
        assert in_sel[ep.rows()].all(), f"{a}__{b}: an episode row is not a selection row"
        shas[f"{a}__{b}"] = episodes_sha256(ep)
        for f in FIELDS:
            saved[f"{a}__{b}__{f}"] = getattr(ep, f)
        parts.append(ep)
        log(f"episodes {a} x {b} (third {third}): {len(ep.anchor)} validated, sha {shas[f'{a}__{b}'][:12]}")
    np.savez(out_dir / f"episodes_seed{seed}.npz", pair_order=np.array([f"{a}__{b}" for a, b, _ in PAIRS]), **saved)
    pooled = concat_episodes(parts)
    n_total = len(pooled.anchor)
    pair_index = np.repeat(np.arange(len(PAIRS)), n)
    anchor_group = groups[pooled.anchor]
    parity = np.arange(n_total) % 2

    # ---- scorers
    base = EvalInputs(img, txt)
    code_inputs = {k: EvalInputs(img, txt, *codes[k]) for k in codes}
    cos = cosine_scores(base, pooled)
    terms = {
        "diag": lambda: diag_agreement_term(base, pooled, relu=False),
        "diag_relu": lambda: diag_agreement_term(base, pooled, relu=True),
        "bilinear": lambda: bilinear_agreement_term(base, pooled, basis),
        "kissme": lambda: kissme_term(base, pooled, basis),
        "rca": lambda: rca_term(base, pooled, basis),
        "xing": lambda: xing_term(base, pooled, basis),
        "wang": lambda: wang_term(base, pooled),
        "probe": lambda: pair_probe_term(base, pooled, scaler),
        "tip": lambda: tip_adapter_term(base, pooled),
        "value_prototype": lambda: value_prototype_term(base, pooled),
        "SE": lambda: agreement_term(code_inputs["SE"], pooled),
        "C0": lambda: agreement_term(code_inputs["C0"], pooled),
        "R3": lambda: agreement_term(code_inputs["R3"], pooled),
        "SE_uniform": lambda: agreement_term(code_inputs["SE"], pooled, uniform=True),
    }
    scores, picks, timings = {"cosine": cos}, {"cosine": None}, {"cosine": {"term_s": 0.0, "crossfit_s": 0.0}}
    for name, fn in terms.items():
        t1 = perf_counter()
        term = fn()
        t2 = perf_counter()
        scores[name], p = crossfit_lambda(cos, term, parity)
        picks[name] = {str(k): (v if np.isfinite(v) else "inf") for k, v in p.items()}
        timings[name] = {"term_s": t2 - t1, "crossfit_s": perf_counter() - t2}
        log(f"{name}: term {t2 - t1:.1f}s crossfit {perf_counter() - t2:.1f}s picks {picks[name]}")
    order = ["cosine", *terms]

    # ---- metrics
    pa = {s: per_anchor(scores[s]) for s in order}
    assert (pa["cosine"]["gain"] == 0).all(), "cosine condition gain must be exactly 0"
    for s in order:
        for m in METRICS:
            assert np.isfinite(pa[s][m]).all(), f"{s}/{m}: non-finite per-anchor value"
    summ, per_pair = {}, {}
    for s in order:
        summ[s] = summarize(pa[s], anchor_group)
        per_pair[s] = {f"{a}__{b}": summarize({m: pa[s][m][pair_index == i] for m in METRICS},
                                              anchor_group[pair_index == i]) for i, (a, b, _) in enumerate(PAIRS)}
    np.savez(out_dir / f"per_anchor_seed{seed}.npz", anchor_group=anchor_group, pair_index=pair_index,
             **{f"{s}__{m}": pa[s][m] for s in order for m in METRICS})

    ranking = sorted(((0.5 * (summ[s]["r1"]["point"] + summ[s]["gain"]["point"]), s) for s in GO_CANDIDATES),
                     reverse=True)
    go = [{"scorer": s, "mean_r1_gain": v, "r1": summ[s]["r1"]["point"], "gain": summ[s]["gain"]["point"]}
          for v, s in ranking]
    record = {"episodes_seed": seed, "n_per_pair": n, "pair_order": [f"{a}__{b}" for a, b, _ in PAIRS],
              "episodes_sha256": shas, "codes_provenance": code_prov, "fit_rows_sha256": fit_sha, "fit_rows": 60000,
              "scorers": {s: {"overall": summ[s], "per_pair": per_pair[s], "lambda_picks": picks[s]} for s in order},
              "go_bar_ranking": go, "timings_s": timings, "total_s": perf_counter() - T0}
    (out_dir / f"baselines_seed{seed}.json").write_text(json.dumps(record, indent=1))

    # ---- table
    def cell(x):
        return f"{x['point']:6.2f} [{x['ci95'][0]:5.2f},{x['ci95'][1]:5.2f}]"
    print(f"\n{'scorer':16s} {'R@1':22s} {'gain':22s} {'other':>6s} {'swap':>6s}  lambda picks")
    for s in order:
        print(f"{s:16s} {cell(summ[s]['r1'])} {cell(summ[s]['gain'])} {summ[s]['other']['point']:6.2f} "
              f"{summ[s]['swap']['point']:6.2f}  {picks[s]}")
    print("\nGO-bar candidate ranking (mean of R@1 and gain):")
    for g in go:
        print(f"  {g['scorer']:10s} {g['mean_r1_gain']:6.2f}  (R@1 {g['r1']:.2f}, gain {g['gain']:.2f})")
    log("done")


if __name__ == "__main__":
    main()
