"""Round 8 step 3: the paired-image reference P for the told route (descriptive; CPU only; writes only here).

A = round 7's told scorer as it ran (K1 for T; the told R0 scorer for R0), reproduced from saved artifacts.
P = the same scorer with every caption's posterior replaced by the image-head posterior of the caption's own painting
    (row r's txt posterior := row r's img posterior), in the told scorer and its matched counterpart alike (both are
    built from the same posterior stack inside F.told). Nothing else changes: episodes, seed 42, 224 cells and their
    cross-fit selection (re-run on the P term), intervals. The told route fits no reader; no component is trained on
    caption posteriors, so there is nothing to refit except the per-run cell selection, which F.told redoes.

Usage: r8_caption_ref.py {T,R0}  -> results_<part>.json in this folder. Round 7's results/ is read, never written.
"""
import os
import sys
import json
import time
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
R7 = "/project/CoSiR/src/test/20261126_design_l_style_source"
sys.path.insert(0, R7)

import numpy as np

import r7_common as R
import r7_bundle as B
import r7_fusion as F
import r7_gate as S
import run_r7_descriptive as D

HERE = Path(__file__).resolve().parent
METRICS = D.METRICS
TOLD_MAP = {"T": {"emotion": "affect", "style": "T", "genre": "image"},
            "R0": {"emotion": "affect", "style": "R0", "genre": "image"}}
PARTS = {"T": B.A0 + ("T",), "R0": B.A1}


def paired_image(post):
    """Replace the txt side by the img side, row by row (NaN outside the selection stays NaN)."""
    img = post["img"]
    return {"img": img.copy(), "txt": img.copy()}


def episode_ids(bundle):
    return {"cl": np.asarray(bundle.cl), "pair_index": np.asarray(bundle.pair_index), "parity": np.asarray(bundle.parity)}


def summarize(scored, ids):
    s = D.summary(scored["fused"], scored["cf"], ids)
    m = ids["pair_index"] == 2
    s["k1_style_x_genre"] = S.ceiling_stats(scored["fused"]["r1"][m], scored["cf"]["r1"][m], ids["cl"][m])
    s["cells"] = {k: scored[k].tolist() for k in ("fused_cells", "cf_cells")}
    s["sigma"] = scored["sigma"] if not isinstance(scored["sigma"], dict) else {str(k): v for k, v in scored["sigma"].items()}
    return s


def main():
    part = sys.argv[1]
    assert part in TOLD_MAP, "part must be T or R0"
    R.assert_rule()
    R.assert_threads()
    role = R.roles("real")
    t0 = time.time()
    bundle = B.build_bundle(role.dev, role.smoke, extra=("T",), root=role.root)
    parts, told_map = PARTS[part], TOLD_MAP[part]
    ids = episode_ids(bundle)
    post = bundle.post[part]
    rows = np.asarray(bundle.ctx.selection)

    # The image head posterior is a function of the painting (rows of one painting share it).
    groups = np.asarray(bundle.ctx.groups)[rows]
    order = np.argsort(groups, kind="stable")
    g, img = groups[order], post["img"][rows][order]
    first = np.r_[True, g[1:] != g[:-1]]
    rep = np.maximum.accumulate(np.where(first, np.arange(len(g)), 0))
    dmax = float(np.abs(img - img[rep]).max())
    img_const_within_painting = dmax < 1e-5       # float32 batching noise only; the argmax never differs
    assert (img.argmax(1) == img[rep].argmax(1)).all() and img_const_within_painting, "img posterior varies within painting"

    out = {"part": part, "units": "R@1 points; accuracy in percent", "seed": role.dev, "n_episodes": int(len(ids["cl"])),
           "img_post_constant_within_painting": img_const_within_painting, "img_post_max_abs_diff_within_painting": dmax,
           "n_selection_rows": int(len(rows))}

    # ---- A: reproduce round 7 from the saved artifacts
    A = F.told(bundle, parts, told_map)
    if part == "T":
        saved = D.npz(Path(R7) / "results/seed42/arrays_ceiling.npz")
        name = "told_T"
        chk = {}
        for k, v in A["arrays"].items():
            assert k in saved, f"{k}: not in saved arrays"
            chk[k] = bool(np.array_equal(np.asarray(v), saved[k]))
        out["reproduction"] = {"arrays_compared": len(chk), "all_exact": all(chk.values()), "mismatch": [k for k, v in chk.items() if not v]}
        sd = {m: saved[f"{name}__{m}"] for m in METRICS}
        out["reproduction"]["ceiling_verdict"] = {}
        ver = json.loads((Path(R7) / "results/ceiling_verdict.json").read_text())
        m = ids["pair_index"] == 2
        k1 = S.ceiling_stats(A["fused"]["r1"][m], A["cf"]["r1"][m], ids["cl"][m])
        out["reproduction"]["ceiling_verdict"] = {"saved": {k: ver[k] for k in ("sum4d", "n", "point", "ci95", "zone")},
                                                  "rerun": k1, "equal": all(k1[k] == ver[k] for k in ("sum4d", "n", "point", "ci95", "zone"))}
        desc = json.loads((Path(R7) / "results/descriptive_ceiling.json").read_text())["told_T"]
        sa = D.summary(A["fused"], A["cf"], ids)
        def eq(x, y):
            return x == y
        out["reproduction"]["descriptive_told_T_margins_equal"] = {
            scope: eq(sa["per_pair"][scope]["margin"], desc["per_pair"][scope]["margin"]) for scope in sa["per_pair"]}
        out["reproduction"]["descriptive_told_T_margins_equal"]["pooled"] = eq(sa["pooled"]["margin"], desc["pooled"]["margin"])
    else:
        ref = np.load(Path("/project/CoSiR/src/test/20261119_no_caption_csd_spike/results/sp_cand_told_A1_224.npz"))
        out["reproduction"] = {"vs_spike_told_A1_fused_r1_exact": bool(np.array_equal(A["fused"]["r1"], ref["fused_r1"])),
                               "vs_spike_told_A1_margin_r1_exact": bool(np.array_equal(
                                   A["fused"]["r1"] - A["cf"]["r1"], ref["margin_r1"]))}
    out["A"] = summarize(A, ids)

    # ---- P: captions take the posterior of their own painting's image
    P = F.told(bundle, parts, told_map, posteriors_override={part: paired_image(post)})
    out["P"] = summarize(P, ids)
    out["P_cells_frozen_from_A"] = None
    fr = None
    # sensitivity: P scored with A's cells (no re-selection); gates are all open in the told route
    stack, _ = B.stack_and_features(
        type(bundle)(**{**vars(bundle), "post": {**bundle.post, part: paired_image(post)}}), parts)
    pick = F.C.told_index(parts, told_map, np.asarray(bundle.pair_index))
    Pm = {c: np.eye(len(parts))[pick[c]] for c in F.CONDITIONS}
    term = F.C.expected_term(stack, Pm)
    fr = F.score_frozen(bundle, term, A["gates"], A["fused_cells"], A["cf_cells"])
    out["P_cells_frozen_from_A"] = summarize({"fused": fr["fused"], "cf": fr["cf"], "fused_cells": A["fused_cells"],
                                              "cf_cells": A["cf_cells"], "sigma": [0.0, 0.0]}, ids)
    out["P_cells_frozen_from_A"].pop("sigma")

    # ---- placement of the caption items
    ep = bundle.ctx.pooled
    ep_rows = np.unique(np.concatenate([np.asarray(ep.anchor).ravel(), np.asarray(ep.candidates).ravel()]))
    pl = {"n_episode_rows": int(len(ep_rows))}
    if part == "T":
        lab = np.asarray(R.load("run_told_oracle").artelingo_aspect_labels(bundle.ctx.data)["style"])
        assert lab.shape == (len(bundle.ctx.groups),) and (lab[rows] >= 0).all()
        for scope, r in (("selection_rows", rows), ("episode_rows", ep_rows)):
            y = lab[r]
            pl[scope] = {
                "caption_head_own_txt_post": 100 * float((post["txt"][r].argmax(1) == y).mean()),
                "P_caption_post_is_image_post": 100 * float((post["img"][r].argmax(1) == y).mean()),
                "image_head_img_post": 100 * float((post["img"][r].argmax(1) == y).mean()),
                "chance_majority_class": 100 * float(np.bincount(y).max() / len(y)),
                "caption_head_agrees_with_image_head": 100 * float((post["txt"][r].argmax(1) == post["img"][r].argmax(1)).mean()),
            }
        head = json.loads((Path(R7) / "results/heads/head_T.json").read_text())
        pl["head_check_rows_accuracy"] = head["standard"]["heldout_accuracy"]
    else:
        pl["note"] = "R0 has no true style code; agreement of the caption head with the image head"
        for scope, r in (("selection_rows", rows), ("episode_rows", ep_rows)):
            pl[scope] = {"caption_head_agrees_with_image_head": 100 * float((post["txt"][r].argmax(1) == post["img"][r].argmax(1)).mean())}
    out["placement"] = pl
    out["elapsed_s"] = time.time() - t0
    path = HERE / f"results_{part}.json"
    path.write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print("wrote", path)


if __name__ == "__main__":
    main()
