"""Independent inputs for R7 §9, L11/L14; adapted from rd3_bundle (copied).

EvalContext, N1u and condition-free cross-fit are the permitted frozen pieces.
Features, probabilities, terms and all per-anchor metrics are our own code.
Legacy regression reads only frozen A0/A1 objects, never a new grouping.
"""
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import r7_common as R
from rederive import rd7_core as K

# R3 D15, retained when copying its independent re-derivation.
LEGACY_SHA = {
    "20261116_grouping_step1_style/results/step1_heads_style.npz":
        "898a37017d82d3e130b20155e90f69e51f82f30892e04316dcb56d7aaaf8df8b",
    "20261108_new_method_quick_checks/results/n6_posteriors.npz":
        "2ad75026e5869c461fed156ffcdcf207c66f04a59dae514358522fbbaa9df2e0",
    "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt":
        "dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2",
    "20261117_reader_fix_csd/results/rb_reader_A0.pkl":
        "6387b469662446734e650571c88a152651da3d66bcade28fb6cdd7828f58d34c",
    "20261117_reader_fix_csd/results/rc_tau.json":
        "e10cf52b2e81ea5b5f7243363d7ddf226440c3a95512cbc7e0b9ed18c5e702bf",
}


def checked_path(path):
    path = Path(path)
    for key, want in {**LEGACY_SHA, **R.INPUTS}.items():
        if path.resolve() == R.path_of(key).resolve():
            assert R.sha_file(path) == want, f"input SHA differs: {path}"
            break
    return path


def checked_npz(path):
    with np.load(checked_path(path), allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def legacy_reader(config):
    assert config in ("A0", "A1"), "legacy regression uses A0 or A1"
    with open(checked_path(R.R1D / f"results/rb_reader_{config}.pkl"), "rb") as f:
        return pickle.load(f)


def context(seed, smoke=False):
    assert seed in ((9001, 9002, 9003) if smoke else (42, 55, 56, 57)), "seed role"
    ctx = R.load("run_gonogo").EvalContext(seed, smoke)
    assert ctx.parity.shape == (ctx.n,) and np.array_equal(ctx.parity, np.arange(ctx.n) % 2), "per-seed parity"
    rc = R.load("run_checks")
    qc = R.load("src.eval.aspect_quick_checks")
    from src.data.artelingo_splits import artelingo_splits
    st = np.asarray(artelingo_splits(ctx.data).scorer_train)
    checked_path(R.TEST / "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt")
    tu = qc.centered_term(rc.model_inputs(ctx, "A3", st, False)[0], ctx.pooled, uniform=True)
    n6 = R.load("run_n6")
    path = checked_path(R.TEST / "20261108_new_method_quick_checks/results/n6_posteriors.npz")
    e2 = n6.load_posteriors(path, ctx)
    B = qc.crossfit_condition_free(ctx.cos, tu, qc.uniform_probe_scores(e2, ctx.pooled, K.E2_PARTS), ctx.parity)[0]
    return SimpleNamespace(ctx=ctx, B=B, t_n1u=tu, post={}, parity=ctx.parity,
                           cl=ctx.anchor_group, pair_index=ctx.pair_index, e2=e2)


def validate_post(post, ctx):
    for h, pm in post.items():
        assert set(pm) == {"img", "txt"}, f"posterior keys: {h}"
        assert pm["img"].shape == pm["txt"].shape, f"posterior shape: {h}"
        for m, p in pm.items():
            assert p.dtype == np.float32 and p.shape[0] == len(ctx.groups) and p.shape[1] >= 2, f"posterior rows: {h}/{m}"
            assert np.isfinite(p[ctx.selection]).all() and np.isnan(p[~ctx.in_sel]).all(), f"selection mask: {h}/{m}"
            assert np.allclose(p[ctx.selection].sum(1), 1, rtol=0, atol=1e-6), f"posterior mass: {h}/{m}"


def build_legacy(seed, smoke=False):
    """Frozen R3 re-derivation head cache supplies D2 posteriors, not scores."""
    b = context(seed, smoke)
    cache = R.R3D / "rederive/out/rd3_bundle_seed42.npz"
    record = json.loads(cache.with_suffix(".json").read_text())
    assert R.sha_file(cache) == record["cache_sha256"], "independent frozen head cache SHA"
    z = checked_npz(cache)
    assert np.array_equal(z["selection"], b.ctx.selection), "frozen head selection rows"
    for h in K.A0:
        b.post[h] = {}
        for m in ("img", "txt"):
            full = np.full((len(b.ctx.groups), z[f"post__{h}__{m}"].shape[1]), np.nan, np.float32)
            full[b.ctx.selection] = z[f"post__{h}__{m}"]
            b.post[h][m] = full
    csd = checked_npz(R.TEST / "20261116_grouping_step1_style/results/step1_heads_style.npz")
    assert np.array_equal(csd["selection"], b.ctx.selection), "CSD selection rows"
    b.post["csd"] = {}
    for m in ("img", "txt"):
        local = csd[f"style_csd__{m}"]
        full = np.full((len(b.ctx.groups), local.shape[1]), np.nan, np.float32)
        full[b.ctx.selection] = local
        b.post["csd"][m] = full
    validate_post(b.post, b.ctx)
    return b


def condition_free(b, parts):
    qc = R.load("src.eval.aspect_quick_checks")
    scores = qc.crossfit_condition_free(b.ctx.cos, b.t_n1u,
             qc.uniform_probe_scores(b.post, b.ctx.pooled, parts), b.parity)[0]
    assert all(np.array_equal(scores["a"][d], scores["b"][d]) for d in K.DIRS), "condition-free comparator"
    return K.metrics(scores)


def read(b, parts, pk, stored=None):
    stack = K.grouping_scores(b.post, b.ctx.pooled, parts)
    feats = K.features_both(b.post, parts, b.ctx.pooled)
    assert all(f.shape == (b.ctx.n, 6 * len(parts)) and np.isfinite(f).all() for f in feats.values()), "reader features"
    assert all(np.array_equal(feats["b"][:, 6 * j + 2], -feats["a"][:, 6 * j + 2])
               for j in range(len(parts))), "reader swapped deltas"
    P, _ = K.reader_probs(pk, feats, parts)
    pm = {c: K.picks_margins(P[c]) for c in K.COND}
    out = {"P": P, "m": {c: pm[c][1] for c in K.COND}, "pick": {c: pm[c][0] for c in K.COND},
           "T": K.weighted_term(stack, P)}
    if stored is not None:
        z = checked_npz(stored)
        assert set(z) == {f"{name}__{c}" for c in K.COND for name in ("P", "m", "pick")}, "stored reader keys"
        for c in K.COND:
            for name in ("P", "m", "pick"):
                got, want = out[name][c], z[f"{name}__{c}"]
                assert got.shape == want.shape and got.dtype == want.dtype, f"stored reader shape/dtype: {name}/{c}"
                if name == "pick":
                    assert np.array_equal(got, want), f"stored reader picks: {c}"
                else:
                    assert np.allclose(got, want, rtol=0, atol=1e-12), f"stored reader {name}: {c}"
            prob = z[f"P__{c}"]
            assert np.isfinite(prob).all() and np.all((prob >= 0) & (prob <= 1)), f"stored reader probability range: {c}"
            assert np.allclose(prob.sum(1), 1, rtol=0, atol=1e-12), f"stored reader probability mass: {c}"
            pi, margin = K.picks_margins(prob)
            assert np.array_equal(pi, z[f"pick__{c}"]), f"stored reader positional picks: {c}"
            assert np.array_equal(margin, z[f"m__{c}"]), f"stored reader top-two margins: {c}"
            out["P"][c], out["pick"][c], out["m"][c] = prob, pi, margin
        out["T"] = K.weighted_term(stack, out["P"])
    return out


def baselines(b, csd="csd"):
    return {"B": K.metrics(b.B), "B0": condition_free(b, K.A0),
            "B1": condition_free(b, (*K.A0, csd))}


def score(b, reader, taus, kind, allowed=(0, 3)):
    from rederive import rd7_family as F
    assert kind in ("R1", "XAFF", "AFF"), "scorer kind"
    if kind == "R1":
        gates = F.gates_r1(reader["m"], taus)
    else:
        gates = F.gates_x_aff(reader["m"], reader["pick"], taus, (0,) if kind == "AFF" else allowed)
    rec, fused, cf, _ = F.run_family(b.B, reader["T"], gates, taus, b.parity)
    assert np.all(cf["gain"] == 0), "counterpart gain"
    return {"record": rec, "fused": fused, "cf": cf,
            **{f"gate_{c}": np.stack([g[c] for g in gates]).astype(np.float32) for c in K.COND}}


def told(b, style):
    """§5: emotion→affect, style→fourth grouping, genre→image; all gates open."""
    parts = (*K.A0, style)
    assert set(np.unique(b.pair_index)) == {0, 1, 2}, "told pair indices"
    pi = {"a": np.array([0, 0, 3])[b.pair_index], "b": np.array([3, 1, 1])[b.pair_index]}
    P = {c: np.eye(4, dtype=np.float64)[pi[c]] for c in K.COND}
    reader = {"P": P, "pick": pi, "m": {c: np.ones(b.ctx.n, np.float64) for c in K.COND},
              "T": K.weighted_term(K.grouping_scores(b.post, b.ctx.pooled, parts), P)}
    return score(b, reader, (1., 1., 1., 1.), "R1")


def load_post(b, root, name):
    z = checked_npz(Path(root) / f"heads/post_{name}.npz")
    assert set(z) == {"img", "txt", "rows"}, f"head keys: {name}"
    assert z["rows"].dtype == np.int64 and z["rows"].ndim == 1, f"head row dtype: {name}"
    assert np.array_equal(z["rows"], b.ctx.selection), f"head selection rows: {name}"
    post = {m: z[m] for m in ("img", "txt")}
    validate_post({name: post}, b.ctx)
    b.post[name] = post
    return post


def build(seed, root, smoke=False):
    """R7 fitted objects; no import, cache or array from the R7 scorer pipeline."""
    b = context(seed, smoke)
    for h in K.A0:
        load_post(b, root, h)
    return b


def fitted_reader(root, name):
    with open(Path(root) / f"readers/reader_{name}.pkl", "rb") as f:
        pk = pickle.load(f)
    assert isinstance(pk, dict) and {"halves", "groupings", "feature_names"} <= set(pk), f"reader keys: {name}"
    return pk


def aff_taus():
    z = json.loads(checked_path(R.R1D / "results/rc_tau.json").read_text())
    assert tuple(z["taus"]) == K.RULE_TAUS, "frozen AFF thresholds"
    return z["taus"]
