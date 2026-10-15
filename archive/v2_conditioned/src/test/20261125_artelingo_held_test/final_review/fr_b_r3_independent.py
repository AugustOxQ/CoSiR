"""Final review B: independent re-derivations on seed 42 from round 3's / round 4's frozen code, no r6 module imported.

  1. round 3's r3_bundle.build_bundle(42, False) (EvalContext, stored n6 posteriors, its own affect refit):
     B and B'(A0) cross-fitted with crossfit_condition_free (R3 rule D10, D11); B'(A1) = crossfit_condition_free over
     A1 = (affect, image, caption, csd) with step 1's STORED csd posteriors (round 4's D2, D4); mean R@1 x 100.
  2. AFF, CF, R1 fused, R1 counterpart through r3_fusion.score_frozen with round 3's cells (on round 3's B).
  3. The held split with my own checks (metadata only: painting names, leakage groups, index arrays).
  4. The development value sets with my own eligible-value count, and the episodes of seeds 42, 9001, 9002, 9003
     built by the ORIGINAL build_aspect_episodes on selection rows (the restriction is a no-op iff V == ok lists),
     their per-pair episodes_sha256 against AB's records.
Saves arrays to argv[1] (npz) and a JSON summary to argv[2].
"""
import json
import sys
from pathlib import Path

import numpy as np

MAIN = Path("/project/CoSiR")
TEST = MAIN / "src/test"
R3D = TEST / "20261121_round3_affect_gate"
AB = TEST / "20261030_aspect_baselines"
sys.path.insert(0, str(R3D))
sys.path.insert(0, str(MAIN))
import r3_bundle as RB  # noqa: E402
import r3_common as R3  # noqa: E402
import r3_fusion as RF  # noqa: E402

from src.eval.aspect_metrics import per_anchor  # noqa: E402
from src.eval.aspect_quick_checks import crossfit_condition_free, uniform_probe_scores  # noqa: E402
from src.eval.aspect_episodes import (PaintingValueIndex, build_aspect_episodes, eligible_values,  # noqa: E402
                                      episodes_sha256)
from src.data.artelingo_splits import artelingo_aspect_labels, artelingo_splits  # noqa: E402
from src.data.splits import grouped_split  # noqa: E402

PAIRS = (("emotion", "style", "genre"), ("emotion", "genre", "style"), ("style", "genre", "emotion"))


def main():
    out, summ = {}, {}
    b = RB.build_bundle(42, False)
    ctx = b.ctx
    summ["B_mean"] = 100 * float(np.mean(b.pB["r1"]))
    summ["B0_mean"] = 100 * float(np.mean(b.pBp["r1"]))
    # B'(A1): stored csd posteriors (step1_heads_style.npz, CLIP heads), placed on selection rows
    with np.load(TEST / "20261116_grouping_step1_style/results/step1_heads_style.npz") as z:
        assert np.array_equal(z["selection"], ctx.selection)
        csd = {}
        for m in ("img", "txt"):
            full = np.full((len(ctx.groups), z[f"style_csd__{m}"].shape[1]), np.nan, np.float32)
            full[ctx.selection] = z[f"style_csd__{m}"]
            csd[m] = full
    post_a1 = {**b.post, "csd": csd}
    B1 = crossfit_condition_free(ctx.cos, b.t_n1u, uniform_probe_scores(post_a1, ctx.pooled,
                                                                         ("affect", "image", "caption", "csd")),
                                 ctx.parity)
    pB1 = per_anchor(B1[0])
    summ["B1_mean"] = 100 * float(np.mean(pB1["r1"]))
    summ["B1_picks"] = {str(k): list(map(float, v)) for k, v in B1[1].items()}
    for name, src in (("B", b.B), ("B0", b.Bp)):
        _, p = crossfit_condition_free(ctx.cos, b.t_n1u,
                                       (RB.modules().n6.n6_terms(RB.modules().n6.load_posteriors(
                                           R3.C.INPUT_FILES["n6_posteriors"], ctx), ctx.pooled)[2] if name == "B"
                                        else uniform_probe_scores(b.post, ctx.pooled, ("affect", "image", "caption"))),
                                       ctx.parity)
        summ[f"{name}_picks"] = {str(k): list(map(float, v)) for k, v in p.items()}
    for name, pa in (("B", b.pB), ("B0", b.pBp), ("B1", pB1)):
        for m, v in pa.items():
            out[f"{name}__{m}"] = np.asarray(v)
    # bundle arrays for comparison with r6's
    for k in ("cos", "t_n1u"):
        for c, dd in getattr(b, k).items():
            for d, v in dd.items():
                out[f"bundle__{k}__{c}__{d}"] = np.asarray(v)
    for d, v in b.stack.items():
        out[f"bundle__stack__{d}"] = np.asarray(v)
    for c, v in b.F.items():
        out[f"bundle__F__{c}"] = np.asarray(v)
    out["cl"], out["pair_index"], out["anchor"] = np.asarray(b.cl), np.asarray(b.pair_index), np.asarray(b.anchor)
    for h in ("affect", "image", "caption"):              # round 3's D1 affect refit and the stored E2 posteriors
        for m in ("img", "txt"):
            out[f"r3post__{h}__{m}"] = np.asarray(b.post[h][m][ctx.selection])
    # AFF / CF / R1 through round 3's frozen line
    rd = RF.reader(b, readers=b.readers)
    taus = R3.assert_taus()
    g = {"aff": RF.gates_aff(rd["m"], rd["pick"], taus), "r1": RF.gates_r1(rd["m"], taus)}
    for who, cells in (("aff", R3.AFF_CELLS), ("r1", R3.RC_CELLS)):
        r = RF.score_frozen(b, rd["T"], g[who], cells["fused"], cells["cf"])
        for part in ("fused", "cf"):
            for m, v in r[part].items():
                out[f"{who}_{part}__{m}"] = np.asarray(v)
    # 3. the held split, my own checks (index arrays and painting names only)
    data = ctx.data
    sp = artelingo_splits(data)
    gs = grouped_split(sp.groups, seed=42)
    held, train, val, st, sel = (np.asarray(x) for x in (sp.held, gs.train, sp.val, sp.scorer_train, sp.selection))
    paint = np.asarray(data.paintings, dtype=object)
    s = {}
    s["sizes"] = {k: int(len(v)) for k, v in dict(held=held, train=train, val=val, scorer_train=st, selection=sel).items()}
    s["held_is_grouped_split_held"] = bool(np.array_equal(np.sort(gs.held), held))
    s["held_20pct"] = len(held) / len(paint)
    for name, other in (("train", train), ("val", val), ("scorer_train", st), ("selection", sel)):
        s[f"rows_disjoint_{name}"] = bool(np.intersect1d(held, other).size == 0)
        s[f"leak_groups_disjoint_{name}"] = bool(np.intersect1d(sp.groups[held], sp.groups[other]).size == 0)
        s[f"paintings_disjoint_{name}"] = bool(not (set(paint[held].tolist()) & set(paint[other].tolist())))
    s["partition"] = bool(np.array_equal(np.sort(np.concatenate([train, val, held])), np.arange(len(paint))))
    with np.load(TEST / "20261013_stage_d_selection/cache/prepare.npz") as z:
        s["prepare_equal"] = {k: bool(np.array_equal(v, z[k])) for k, v in
                              dict(groups=sp.groups, split_train=gs.train, scorer_train=st, selection=sel).items()}
    with np.load(TEST / "20261014_stage_d_final/cache/held_codes.npz") as z:
        s["held_codes_equal"] = bool(np.array_equal(z["held_rows"], held))
    labels = artelingo_aspect_labels(data)
    known = (labels["emotion"] >= 0) & (labels["style"] >= 0) & (labels["genre"] >= 0)
    s["held_rows_all_three_known"] = int(known[held].sum())
    summ["split"] = s
    # 4. development values (own count of distinct paintings per value on the selection pool) and episodes
    groups = sp.groups
    pool = sel[known[sel]]
    dev = {}
    for x in ("emotion", "style", "genre"):
        lab = labels[x][pool]
        vals = sorted(int(v) for v in np.unique(lab) if len(np.unique(groups[pool][lab == v])) >= 30)
        dev[x] = vals
        assert vals == [int(v) for v in eligible_values(labels[x], groups, pool, 30)]
    summ["dev_values"] = dev
    summ["dev_counts"] = {k: len(v) for k, v in dev.items()}
    index = PaintingValueIndex(labels, groups)
    eps = {}
    for seed, n, rel in [(42, 4096, "results/baselines_seed42.json")] + \
            [(s_, 64, f"results/smoke/baselines_seed{s_}.json") for s_ in (9001, 9002, 9003)]:
        rec = json.loads((AB / rel).read_text())
        for a, bb, t in PAIRS:
            # my own restriction: V must equal the pool's eligible lists, so the original builder is the restricted one
            pool_p = sel[(labels[a][sel] >= 0) & (labels[bb][sel] >= 0) & (labels[t][sel] >= 0)]
            ok_a = [int(v) for v in eligible_values(labels[a], groups, pool_p, 30)]
            ok_b = [int(v) for v in eligible_values(labels[bb], groups, pool_p, 30)]
            assert ok_a == dev[a] and ok_b == dev[bb], (seed, a, bb)
            ep = build_aspect_episodes(labels, groups, sel, a, bb, n, seed, third=t, index=index)
            h = episodes_sha256(ep)
            eps[f"{seed}__{a}__{bb}"] = {"got": h, "want": rec["episodes_sha256"][f"{a}__{bb}"],
                                          "equal": h == rec["episodes_sha256"][f"{a}__{bb}"]}
            if seed == 42:
                for f in ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt"):
                    out.setdefault(f"ep__{f}", []).append(getattr(ep, f))
    for f in [k for k in out if k.startswith("ep__")]:
        out[f] = np.concatenate(out[f])
    summ["episodes"] = eps
    np.savez(sys.argv[1], **out)
    Path(sys.argv[2]).write_text(json.dumps(summ, indent=1))
    print("r3 independent: B %s B0 %s B1 %s; episodes equal %d/%d; split ok %s" % (
        summ["B_mean"] == 18.341064453125, summ["B0_mean"] == 18.436686197916664, summ["B1_mean"] == 18.804931640625,
        sum(v["equal"] for v in eps.values()), len(eps),
        all(v for k, v in s.items() if isinstance(v, bool)) and all(s["prepare_equal"].values())))


if __name__ == "__main__":
    main()
