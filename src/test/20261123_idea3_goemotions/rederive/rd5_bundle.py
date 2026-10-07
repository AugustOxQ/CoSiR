"""Round 5 re-derivation: one seed's bundle from the allowed loaders and frozen components only (rule §8), in round
3's §4 item 1 call sequence; the grouping stack and the 18 features come from rd5_core (own code)."""
import json
from types import SimpleNamespace

import numpy as np

import rd5_core as core
from rd5_paths import P, SHA, sha_file, sha_array
from rd5_placement import global_labels

E2_PARTS = ("affect", "image", "caption")       # E2's k-means groupings (affect-km, image, caption) for B


def told_head_matches(prov) -> dict:
    """fit_one_head's record against told_oracle.json arm L's head (accuracies stored rounded to 2 decimals)."""
    stored = json.loads(P["told_json"].read_text())["arms"]["L"]["head"]
    checks = {"n_classes": prov["n_classes"] == stored["n_classes"] == 41,
              "draw_rows_sha256": prov["draw_rows_sha256"] == stored["draw_rows_sha256"],
              "acc_img": round(prov["heldout_accuracy"]["img"], 2) == stored["heldout_accuracy"]["img"] == 9.81,
              "acc_txt": round(prov["heldout_accuracy"]["txt"], 2) == stored["heldout_accuracy"]["txt"] == 35.72,
              "check_majority_share": round(prov["check_majority_share"], 2) == stored["check_majority_share"] == 6.41,
              "uniform": prov["uniform"] == stored["uniform"] == 2.4390243902439024}
    return checks


def build(seed, A, log=print):
    """A: the dict of allowed functions (rd5_paths.allowed())."""
    if sha_file(P["a3_ckpt"]) != SHA["a3_ckpt"]:
        raise SystemExit("A3 checkpoint SHA-256 differs")
    ctx = A["EvalContext"](seed, False)
    ep = ctx.pooled
    st = np.asarray(A["artelingo_splits"](ctx.data).scorer_train)
    if not (np.all(np.diff(st) > 0) and len(st) == 183_694):
        raise AssertionError("scorer_train must be ascending with 183,694 rows")
    log(f"seed {seed}: ctx with {ctx.n} episodes")
    inp = A["model_inputs"](ctx, "A3", st, False)[0]
    t_n1u = A["centered_term"](inp, ep, uniform=True)
    del inp
    post_E2 = A["load_posteriors"](P["n6_posteriors"], ctx)
    B = A["crossfit_condition_free"](ctx.cos, t_n1u, A["uniform_probe_scores"](post_E2, ep, E2_PARTS), ctx.parity)[0]
    part_L = np.load(P["told_npz"])["partition_L"]
    lab = global_labels(part_L, st, len(ctx.groups))
    heads, prov = A["fit_one_head"](ctx, lab, st, 60_000)
    head_checks = told_head_matches(prov)
    if not all(head_checks.values()):
        raise SystemExit(f"affect head differs from told_oracle.json arm L: {head_checks}")
    log("affect heads fitted (fit_one_head), identity with told_oracle.json asserted")
    post = {"affect": heads, "image": post_E2["image"], "caption": post_E2["caption"]}
    for h in core.A0:
        for m in ("img", "txt"):
            x = post[h][m]
            if not (np.isfinite(x[ctx.selection]).all() and np.isnan(x[~ctx.in_sel]).all()):
                raise AssertionError(f"{h}/{m}: finite on selection and NaN elsewhere required")
    Bp0 = A["crossfit_condition_free"](ctx.cos, t_n1u, A["uniform_probe_scores"](post, ep, core.A0), ctx.parity)[0]
    stack = core.grouping_stack(post, ep, core.A0)
    F = core.reader_features(post, ep, core.A0)
    if not np.array_equal(F["b"][:, 2], -F["a"][:, 2]):
        raise AssertionError("Delta^b != -Delta^a")
    halves = A["load_readers"]("A0", False)[0]["halves"]
    pa_base = np.load(P["per_anchor_seed42"] if seed == 42 else
                      P["per_anchor_seed42"].parent / f"per_anchor_seed{seed}.npz")
    cos_pa = core.per_anchor(ctx.cos)
    ext_ok = (np.array_equal(pa_base["anchor_group"], ctx.anchor_group)
              and np.array_equal(pa_base["pair_index"], ctx.pair_index)
              and all(np.array_equal(cos_pa[m], pa_base[f"cosine__{m}"]) for m in core.METRICS))
    if not ext_ok:
        raise AssertionError("per_anchor_seed file misaligned with the episodes, or cosine differs")
    rca_pa = {m: pa_base[f"rca__{m}"].astype(np.float64) for m in core.METRICS}
    return SimpleNamespace(seed=seed, ctx=ctx, ep=ep, scorer_train=st, t_n1u=t_n1u, post_E2=post_E2, B=B, post=post,
                           Bp0=Bp0, stack=stack, F=F, halves=halves, cl=ctx.anchor_group, pair_index=ctx.pair_index,
                           parity=ctx.parity, cos_pa=cos_pa, rca_pa=rca_pa, pB=core.per_anchor(B),
                           pBp0=core.per_anchor(Bp0), head_prov=prov, head_checks=head_checks, heads=heads,
                           lab=lab, part_L=part_L)


def b_prime(bundle, post_Q, A) -> tuple:
    """B'_Q = crossfit_condition_free(cos, T_N1u, uniform_probe_scores(post_Q, ep, A0), parity)[0], per anchor."""
    ctx = bundle.ctx
    s = A["crossfit_condition_free"](ctx.cos, bundle.t_n1u, A["uniform_probe_scores"](post_Q, bundle.ep, core.A0),
                                     ctx.parity)[0]
    if not core.condition_free(s):
        raise AssertionError("B'_Q is not condition-free")
    pa = core.per_anchor(s)
    if not (pa["gain"] == 0).all():
        raise AssertionError("B'_Q gain must be 0")
    return s, pa


def scores_equal(x, y) -> bool:
    return all(np.array_equal(np.asarray(x[c][d]), np.asarray(y[c][d]), equal_nan=True)
               for c in core.CONDITIONS for d in core.DIRECTIONS)


def pa_equal(x, y, keys=core.METRICS) -> bool:
    return all(np.array_equal(np.asarray(x[m]), np.asarray(y[m])) for m in keys)


def sha_pa(pa) -> str:
    return sha_array(np.stack([np.asarray(pa[m], np.float64) for m in core.METRICS]))
