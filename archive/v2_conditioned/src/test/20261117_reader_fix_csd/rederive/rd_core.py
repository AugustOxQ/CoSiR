"""Independent re-derivation core (controller's re-derivation agent; decides nothing by itself).

Written from DECISION_RULE.md (SHA-256 613d8c9d...) alone. It does NOT import or read the implementation files of the
parent folder (common.py, run_ra.py, run_rc.py, rc_core.py, rb_*.py, test_*.py). It uses only the project base library
(src.eval.*) and earlier, reviewed step code (run_sweep.setup, run_told_oracle.fit_one_head, run_checks.model_inputs,
run_gonogo.EvalContext). Every rule quantity (agreements, sigma, Delta, picks, reader terms, counterparts, comparators,
bar margin, gain statistic, clauses, pick accuracy, pick shares, R-b features) is computed here with our own code.
CPU only.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/project/CoSiR")
HERE = Path(__file__).resolve().parent
RFX = HERE.parent
RES = RFX / "results"
OUT = HERE / "out"
RULE = RFX / "DECISION_RULE.md"
RULE_SHA = "613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c"

sys.path.insert(0, str(ROOT / "src/test/20261112_community_sweep"))
import run_sweep as rsw  # noqa: E402  (imports run_told_oracle, diagnose_*, run_checks, run_n6, run_gonogo, repo root)

rto, df, rc, rg, n6, dc = rsw.rto, rsw.df, rsw.rc, rsw.rg, rsw.n6, rsw.dc

from src.data.artelingo_splits import artelingo_splits  # noqa: E402
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS, cluster_bootstrap, per_anchor  # noqa: E402
from src.eval.aspect_nested import crossfit_nested  # noqa: E402
from src.eval.aspect_quick_checks import (aspect_deltas, centered_term, crossfit_condition_free,  # noqa: E402
                                          probe_dots, uniform_probe_scores)

N6_POST = ROOT / "src/test/20261108_new_method_quick_checks/results/n6_posteriors.npz"
N6_POST_SHA = "2ad75026e5869c461fed156ffcdcf207c66f04a59dae514358522fbbaa9df2e0"
HEADS = ROOT / "src/test/20261116_grouping_step1_style/results/step1_heads_style.npz"
HEADS_SHA = "898a37017d82d3e130b20155e90f69e51f82f30892e04316dcb56d7aaaf8df8b"
STEP1_EVAL = ROOT / "src/test/20261116_grouping_step1_style/results/step1_eval_style.npz"
STEP1_EVAL_SHA = "8d10a0fbbd34212c73849239faed9d0a07372e68732dc452bf2fd57f7409c68c"
TOLD_NPZ = ROOT / "src/test/20261111_community_told_oracle/results/per_anchor_told_oracle.npz"
TOLD_NPZ_SHA = "27e8101161607da2d6936aa5a8c84f8c296629fe08e37a68c8418c9356af1366"
TOLD_JSON = ROOT / "src/test/20261111_community_told_oracle/results/told_oracle.json"
EPISODES42_SHA = "12af979432ff1a20c88b614ab9e672f203eeeed9c2465bff305e72d01e6c0986"

GROUPINGS = ("affect", "image", "caption", "csd", "rand")
CONFIGS = {"A0": ("affect", "image", "caption"),
           "A1": ("affect", "image", "caption", "csd"),
           "AR": ("affect", "image", "caption", "rand")}
TOLD = {"A0": {"emotion": "affect", "style": "image", "genre": "image"},
        "A1": {"emotion": "affect", "style": "csd", "genre": "image"},
        "AR": {"emotion": "affect", "style": "rand", "genre": "image"}}
PAIRS = (("emotion", "style"), ("emotion", "genre"), ("style", "genre"))
PAIR_NAMES = tuple(f"{a}__{b}" for a, b in PAIRS)
E2_PARTS = ("affect", "image", "caption")          # B's T_6u: E2's k-means-64 heads (affect-km, image, caption)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_inputs():
    for p, s in ((RULE, RULE_SHA), (N6_POST, N6_POST_SHA), (HEADS, HEADS_SHA), (STEP1_EVAL, STEP1_EVAL_SHA),
                 (TOLD_NPZ, TOLD_NPZ_SHA)):
        got = sha_file(p)
        if got != s:
            raise SystemExit(f"{p}: SHA-256 {got} != {s}")


def point_ci(values, clusters):
    """Percent point estimate and 95% painting-bootstrap interval (5,000 resamples, seed 42)."""
    r = cluster_bootstrap(np.asarray(values, dtype=np.float64), clusters, n_boot=5000, seed=42)
    return {"point": 100 * r["point"], "ci95": [100 * r["ci95"][0], 100 * r["ci95"][1]]}


# ---------------------------------------------------------------- context and posteriors

def seed42_setup():
    """run_sweep.setup(): seed-42 context (episodes SHA-checked against baselines_seed42.json), B asserted = C2."""
    S = rsw.setup()
    if sha_file(rg.E1 / "episodes_seed42.npz") != EPISODES42_SHA:
        raise SystemExit("episodes_seed42.npz SHA-256 differs from the rule")
    return S


def load_full(z, key, ctx):
    a = np.asarray(z[key], dtype=np.float32)
    full = np.full((len(ctx.groups), a.shape[1]), np.nan, dtype=np.float32)
    full[ctx.selection] = a
    return full


def e2_posteriors(ctx):
    """E2's stored k-means-64 heads (affect-km, image, caption) from n6_posteriors.npz, NaN outside selection."""
    z = np.load(N6_POST)
    if not np.array_equal(z["selection"], ctx.selection):
        raise AssertionError("n6_posteriors selection differs")
    return {h: {m: load_full(z, f"{h}__{m}", ctx) for m in ("img", "txt")} for h in E2_PARTS}


def standard_posteriors(ctx, scorer_train, partition_L):
    """D2: affect refit with fit_one_head on partition_L; image/caption from n6_posteriors; csd/rand CLIP heads."""
    stored_head = json.loads(TOLD_JSON.read_text())["arms"]["L"]["head"]
    post_aff, prov = rto.fit_one_head(ctx, rto.global_labels(partition_L, scorer_train, len(ctx.groups)),
                                      scorer_train, 60_000)
    prov_r = json.loads(json.dumps(prov))
    same = (prov_r["n_classes"] == stored_head["n_classes"]
            and prov_r["draw_rows_sha256"] == stored_head["draw_rows_sha256"]
            and all(round(prov_r["heldout_accuracy"][m], 2) == stored_head["heldout_accuracy"][m] for m in ("img", "txt")))
    e2 = e2_posteriors(ctx)
    z = np.load(HEADS)
    if not np.array_equal(z["selection"], ctx.selection):
        raise AssertionError("step1 heads selection differs")
    post = {"affect": post_aff, "image": e2["image"], "caption": e2["caption"],
            "csd": {"img": load_full(z, "style_csd__img", ctx), "txt": load_full(z, "style_csd__txt", ctx)},
            "rand": {"img": load_full(z, "style_rand__img", ctx), "txt": load_full(z, "style_rand__txt", ctx)}}
    return post, e2, {"affect_head": prov_r, "affect_head_matches_told_oracle_L": bool(same)}


def t_n1u_term(ctx, scorer_train):
    inp, _, _ = rc.model_inputs(ctx, "A3", scorer_train, False)
    t = centered_term(inp, ctx.pooled, uniform=True)
    del inp
    return t


def rebuild_B(ctx, t_n1u, post_e2):
    """D8: crossfit_condition_free(cos, T_N1u, T_6u over E2's affect-km, image, caption)."""
    t6u = uniform_probe_scores(post_e2, ctx.pooled, E2_PARTS)
    B, picks = crossfit_condition_free(ctx.cos, t_n1u, t6u, ctx.parity)
    return B, picks


def rebuild_Bprime(ctx, t_n1u, post, config):
    """D9: crossfit_condition_free(cos, T_N1u, uniform_probe_scores over the configuration's groupings)."""
    t6u = uniform_probe_scores(post, ctx.pooled, CONFIGS[config])
    Bp, picks = crossfit_condition_free(ctx.cos, t_n1u, t6u, ctx.parity)
    return Bp, picks


# ---------------------------------------------------------------- agreements, sigma, Delta (our own code)

def pair_agreements(post, ep, h, cond):
    """(E,4) support-pair and (E,4) contrast-pair agreements a_h(i,t) = p_h(i).p_h(t), float64."""
    si, st, ci, ct, _ = ep.condition(cond)
    pi = post[h]["img"].astype(np.float64)
    pt = post[h]["txt"].astype(np.float64)
    sup = (pi[si] * pt[st]).sum(axis=-1)
    con = (pi[ci] * pt[ct]).sum(axis=-1)
    return sup, con


def pair_agreements32(post, ep, h, cond):
    """The same in float32 (einsum on the float32 posteriors, as aspect_deltas computes them)."""
    si, st, ci, ct, _ = ep.condition(cond)
    pi, pt = post[h]["img"], post[h]["txt"]
    return np.einsum("nsc,nsc->ns", pi[si], pt[st]), np.einsum("nsc,nsc->ns", pi[ci], pt[ct])


def sigma_h(post, ep, groupings=GROUPINGS, f32=False):
    """Rule 4.1: sigma_h = sqrt(mean_e (s2_S + s2_C)/4), ddof 1, from condition a's sets."""
    out = {}
    for h in groupings:
        sup, con = (pair_agreements32 if f32 else pair_agreements)(post, ep, h, "a")
        sup, con = np.asarray(sup, np.float64), np.asarray(con, np.float64)
        v = (sup.var(axis=1, ddof=1) + con.var(axis=1, ddof=1)) / 4.0
        out[h] = float(np.sqrt(v.mean()))
    return out


def deltas(post, ep, groupings):
    """{cond: (E,H)} Delta via the base library's aspect_deltas (float32), and our float64 version."""
    d32 = {c: aspect_deltas(post, ep, c, groupings) for c in CONDITIONS}
    d64 = {}
    for c in CONDITIONS:
        cols = []
        for h in groupings:
            sup, con = pair_agreements(post, ep, h, c)
            cols.append(sup.mean(axis=1) - con.mean(axis=1))
        d64[c] = np.stack(cols, axis=1)
    return d32, d64


def first_argmax(x):
    """Row-wise arg max, ties to the first column (np.argmax keeps the first maximum)."""
    return np.asarray(x).argmax(axis=1)


def top_two_margin(x):
    s = -np.sort(-np.asarray(x, dtype=np.float64), axis=1)
    return s[:, 0] - s[:, 1]


# ---------------------------------------------------------------- terms

def score_stack(post, ep, groupings):
    """{dir: (E,H,K)} s_h = p_h(query).p_h(candidate), float32 (probe_dots)."""
    dots = probe_dots(post, ep, groupings)
    return {d: np.stack([dots[h][d] for h in groupings], axis=1) for d in DIRECTIONS}


def hard_term(stack, picks):
    rows = np.arange(len(picks["a"]))
    return {c: {d: np.ascontiguousarray(stack[d][rows, picks[c]]).astype(np.float32) for d in DIRECTIONS}
            for c in CONDITIONS}


def expected_term(stack, P):
    """R-b expected: T^c = sum_h P^c(h) s_h."""
    return {c: {d: np.einsum("nh,nhk->nk", np.asarray(P[c], np.float64), stack[d].astype(np.float64)).astype(np.float32)
                for d in DIRECTIONS} for c in CONDITIONS}


def cf_term(T):
    """D7: T_cf = (T^a + T^b)/2, float64 average cast to float32, identical under both conditions."""
    m = {d: (0.5 * (np.asarray(T["a"][d], np.float64) + np.asarray(T["b"][d], np.float64))).astype(np.float32)
         for d in DIRECTIONS}
    return {c: {d: m[d].copy() for d in DIRECTIONS} for c in CONDITIONS}


# ---------------------------------------------------------------- evaluation of one reader term

def fuse(B, T, parity):
    nested, control, picks = crossfit_nested(B, B, T, parity)
    cf, cf_picks = crossfit_condition_free(B, B, cf_term(T), parity)
    return nested, control, picks, cf, cf_picks


def bar_comparator(pBp, pc, pB):
    """D10: the largest mean R@1 among (B', counterpart, B); ties to the earliest in that order."""
    order = (("B_prime", pBp), ("counterpart", pc), ("B", pB))
    means = {n: float(np.mean(np.asarray(p["r1"], np.float64))) for n, p in order}
    best = order[0]
    for n, p in order[1:]:
        if means[n] > means[best[0]]:
            best = (n, p)
    return best[0], best[1], means


def either(p):
    return np.asarray(p["r1"], np.float64) + np.asarray(p["other"], np.float64)


def diff3(pa, pb, cl):
    return {"r1": point_ci(np.asarray(pa["r1"], np.float64) - np.asarray(pb["r1"], np.float64), cl),
            "gain": point_ci(np.asarray(pa["gain"], np.float64) - np.asarray(pb["gain"], np.float64), cl),
            "either": point_ci(either(pa) - either(pb), cl)}


def evaluate(pn, pc, pB, pBp, ctx):
    """Rule D10 to D12 on per-anchor arrays: margin, bar comparator, bar margin, gain statistic, clauses."""
    cl, pi = ctx.anchor_group, ctx.pair_index
    comp_name, comp, means = bar_comparator(pBp, pc, pB)
    bar_v = np.asarray(pn["r1"], np.float64) - np.asarray(comp["r1"], np.float64)
    gain_v = np.asarray(pn["gain"], np.float64) - np.asarray(pc["gain"], np.float64)
    bar = point_ci(bar_v, cl)
    gain = point_ci(gain_v, cl)
    out = {"r1_means": {"fused": 100 * float(np.mean(pn["r1"])), "counterpart": 100 * means["counterpart"],
                        "B": 100 * means["B"], "B_prime": 100 * means["B_prime"]},
           "margin": diff3(pn, pc, cl), "fused_vs_B": diff3(pn, pB, cl), "fused_vs_Bprime": diff3(pn, pBp, cl),
           "counterpart_vs_B": diff3(pc, pB, cl),
           "bar": {"comparator": comp_name, "r1": bar,
                   "per_pair_r1": {p: point_ci(bar_v[pi == i], cl[pi == i]) for i, p in enumerate(PAIR_NAMES)}},
           "gain_statistic": gain,
           "clears_bar": {"c1": bool(bar["point"] >= 0.5), "c2": bool(bar["ci95"][0] > 0),
                          "c3": bool(gain["ci95"][0] > 0)}}
    out["clears_bar"]["clears"] = all(out["clears_bar"][k] for k in ("c1", "c2", "c3"))
    out["per_pair"] = {}
    for i, p in enumerate(PAIR_NAMES):
        m = pi == i
        sub = lambda x: {k: np.asarray(v)[m] for k, v in x.items()}  # noqa: E731
        out["per_pair"][p] = {"margin": diff3(sub(pn), sub(pc), cl[m]),
                              "fused_vs_B_r1": point_ci(np.asarray(pn["r1"])[m] - np.asarray(pB["r1"])[m], cl[m]),
                              "fused_vs_Bprime_r1": point_ci(np.asarray(pn["r1"])[m] - np.asarray(pBp["r1"])[m], cl[m])}
    return out, bar_v


def told_index(config, ctx):
    g = CONFIGS[config]
    asp = {"a": np.array([PAIRS[i][0] for i in ctx.pair_index]), "b": np.array([PAIRS[i][1] for i in ctx.pair_index])}
    return {c: np.array([g.index(TOLD[config][a]) for a in asp[c]]) for c in CONDITIONS}


def pick_accuracy(picks, config, ctx):
    """D13: per episode the mean over conditions of 1[pick = told], with its interval; per pair and condition."""
    ti = told_index(config, ctx)
    corr = {c: (np.asarray(picks[c]) == ti[c]) for c in CONDITIONS}
    v = 0.5 * (corr["a"].astype(float) + corr["b"].astype(float))
    return {"correct_share": point_ci(v, ctx.anchor_group),
            "per_pair_condition": {p: {c: 100 * float(corr[c][ctx.pair_index == i].mean()) for c in CONDITIONS}
                                   for i, p in enumerate(PAIR_NAMES)},
            "both_correct_share": 100 * float((corr["a"] & corr["b"]).mean())}


def pick_shares(picks, config, ctx):
    g = CONFIGS[config]
    allp = np.concatenate([picks["a"], picks["b"]])
    return {"overall": {h: 100 * float(np.mean(allp == j)) for j, h in enumerate(g)},
            "per_condition": {c: {h: 100 * float(np.mean(picks[c] == j)) for j, h in enumerate(g)} for c in CONDITIONS},
            "per_pair_condition": {p: {c: {h: 100 * float(np.mean(picks[c][ctx.pair_index == i] == j))
                                           for j, h in enumerate(g)} for c in CONDITIONS}
                                   for i, p in enumerate(PAIR_NAMES)}}


# ---------------------------------------------------------------- R-b features (rule 4.2 item 4), for Phase 2

def rb_features(post, ep, groupings, cond):
    """(E, 6H) in configuration order: S, C, Delta, sd_S (ddof 1), sd_C (ddof 1), share of the 4 support pairs whose
    image and caption arg-max groups coincide. Condition b swaps supports and contrasts (ep.condition does)."""
    si, st, _, _, _ = ep.condition(cond)
    cols = []
    for h in groupings:
        sup, con = pair_agreements(post, ep, h, cond)
        ai = np.argmax(post[h]["img"][si], axis=-1)
        at = np.argmax(post[h]["txt"][st], axis=-1)
        cols += [sup.mean(1), con.mean(1), sup.mean(1) - con.mean(1), sup.std(1, ddof=1), con.std(1, ddof=1),
                 (ai == at).mean(1)]
    return np.stack(cols, axis=1)


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, default=float))


# ---------------------------------------------------------------- R-b and R-c helpers (Phase 2)

def rb_features_f32(post, ep, groupings, cond):
    """Variant in aspect_deltas' precision: agreements as float32 einsum and S, C, Delta as float32 means (exactly as
    aspect_deltas computes them); the two standard deviations in float64 from the float32 agreements."""
    si, st, _, _, _ = ep.condition(cond)
    cols = []
    for h in groupings:
        sup, con = pair_agreements32(post, ep, h, cond)
        S, C = sup.mean(1), con.mean(1)
        ai = np.argmax(post[h]["img"][si], axis=-1)
        at = np.argmax(post[h]["txt"][st], axis=-1)
        cols += [np.asarray(x, np.float64) for x in (S, C, S - C, sup.astype(np.float64).std(1, ddof=1),
                                                     con.astype(np.float64).std(1, ddof=1), (ai == at).mean(1))]
    return np.stack(cols, axis=1)


def load_readers(config):
    """The stored half-readers (sklearn StandardScaler + LogisticRegression), loaded as data."""
    import pickle
    with open(RES / f"rb_reader_{config}.pkl", "rb") as f:
        r = pickle.load(f)
    if tuple(r["groupings"]) != CONFIGS[config]:
        raise AssertionError("reader groupings differ")
    return r


def reader_proba(r, X):
    """P = mean over the two half-readers of predict_proba(scaler.transform(X))."""
    ps = [h["model"].predict_proba(h["scaler"].transform(X)) for h in r["halves"]]
    return 0.5 * (ps[0] + ps[1]), ps


def prepare_seed42():
    """Seed-42 context, standard posteriors, T_N1u, B and the three B' (all checked against step 1)."""
    from types import SimpleNamespace
    check_inputs()
    S = seed42_setup()
    ctx = S.ctx
    z1 = np.load(STEP1_EVAL)
    post, e2, prov = standard_posteriors(ctx, S.scorer_train, S.partition_L)
    t_n1u = t_n1u_term(ctx, S.scorer_train)
    B, _ = rebuild_B(ctx, t_n1u, e2)
    pB = per_anchor(B)
    checks = {"affect_head_matches_told_oracle_L": prov["affect_head_matches_told_oracle_L"],
              "B_equals_step1_B": all(np.array_equal(np.asarray(pB[m]), z1[f"B__{m}"]) for m in METRICS)}
    pBp = {}
    for cfg in CONFIGS:
        Bp, _ = rebuild_Bprime(ctx, t_n1u, post, cfg)
        pBp[cfg] = per_anchor(Bp)
        checks[f"Bprime_{cfg}_equals_step1"] = all(np.array_equal(np.asarray(pBp[cfg][m]), z1[f"{cfg}__Bprime__{m}"])
                                                   for m in METRICS)
    if not all(checks.values()):
        raise SystemExit(f"prepare_seed42 checks failed: {checks}")
    return SimpleNamespace(S=S, ctx=ctx, ep=ctx.pooled, cl=ctx.anchor_group, z1=z1, post=post, e2=e2, t_n1u=t_n1u,
                           B=B, pB=pB, pBp=pBp, checks=checks)


def full_eval(T, picks, cfg, P):
    """Fuse T on B, counterpart, rule D10-D13 measures and step-1 references for one candidate (P: prepare_seed42)."""
    ctx, cl, z1 = P.ctx, P.cl, P.z1
    nested, control, Tp, cf, cfp = fuse(P.B, T, ctx.parity)
    pn, pc = per_anchor(nested), per_anchor(cf)
    ev, bar_v = eval_arrays(pn, pc, picks, cfg, P)
    ev["crossfit"] = {"T_picks": Tp, "cf_picks": cfp,
                      "control_ranks_as_B": all(np.array_equal(np.asarray(per_anchor(control)[m]), np.asarray(P.pB[m]))
                                                for m in METRICS)}
    return ev, pn, pc, bar_v


def eval_arrays(pn, pc, picks, cfg, P):
    ctx, cl, z1 = P.ctx, P.cl, P.z1
    ev, bar_v = evaluate(pn, pc, P.pB, P.pBp[cfg], ctx)
    ev["pick_accuracy"] = pick_accuracy(picks, cfg, ctx)
    ev["pick_share"] = pick_shares(picks, cfg, ctx)
    ev["picks_count"] = {c: np.bincount(picks[c], minlength=len(CONFIGS[cfg])).tolist() for c in CONDITIONS}
    s1n = {m: z1[f"{cfg}__reader__fused__{m}"] for m in METRICS}
    s1c = {m: z1[f"{cfg}__reader__cf__{m}"] for m in METRICS}
    s1_name, s1_comp, _ = bar_comparator(P.pBp[cfg], s1c, P.pB)
    s1_bar_v = s1n["r1"] - s1_comp["r1"]
    ev["vs_step1_argmax"] = {
        "step1_bar_comparator": s1_name, "step1_bar_r1": point_ci(s1_bar_v, cl),
        "step1_margin_r1": point_ci(s1n["r1"] - s1c["r1"], cl), "step1_fused_r1": 100 * float(np.mean(s1n["r1"])),
        "margin_r1": point_ci((pn["r1"] - pc["r1"]) - (s1n["r1"] - s1c["r1"]), cl),
        "bar_margin_r1": point_ci(bar_v - s1_bar_v, cl), "fused_r1": point_ci(pn["r1"] - s1n["r1"], cl)}
    return ev, bar_v
