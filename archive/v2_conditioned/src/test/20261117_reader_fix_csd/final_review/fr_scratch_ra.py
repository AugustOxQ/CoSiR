"""Final review: R-a on A0 and A1 from scratch (sigma, scaled-Delta picks, terms, library cross-fits, bar)."""
import json, sys, importlib.util
from pathlib import Path
import numpy as np
sys.path.insert(0, "/project/CoSiR")
from src.eval.aspect_metrics import per_anchor, cluster_bootstrap
from src.eval.aspect_nested import crossfit_nested
from src.eval.aspect_quick_checks import crossfit_condition_free
T = Path("/project/CoSiR/src/test"); RES = Path(__file__).resolve().parent.parent / "results"
spec = importlib.util.spec_from_file_location("run_step1", T / "20261116_grouping_step1_style/run_step1.py")
rs1 = importlib.util.module_from_spec(spec); sys.modules["run_step1"] = rs1; spec.loader.exec_module(rs1)
S = rs1.rsw.setup(); ctx = S.ctx; ep = ctx.pooled; cl = ctx.anchor_group; par = ctx.parity; NR = len(ctx.groups); sel = ctx.selection
def full(a):
    f = np.full((NR, a.shape[1]), np.nan, np.float32); f[sel] = a; return f
zn6 = np.load(T / "20261108_new_method_quick_checks/results/n6_posteriors.npz"); zst = np.load(T / "20261116_grouping_step1_style/results/step1_heads_style.npz")
post = {"image": {m: full(zn6[f"image__{m}"]) for m in ("img", "txt")}, "caption": {m: full(zn6[f"caption__{m}"]) for m in ("img", "txt")},
        "csd": {m: full(zst[f"style_csd__{m}"]) for m in ("img", "txt")}, "rand": {m: full(zst[f"style_rand__{m}"]) for m in ("img", "txt")}}
pl = np.load(T / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz")["partition_L"]
post["affect"], _ = rs1.rto.fit_one_head(ctx, rs1.rto.global_labels(np.asarray(pl, np.int64), S.scorer_train, NR), S.scorer_train, rs1.n6.HEAD_ROWS)
G = ("affect", "image", "caption", "csd", "rand")
def ag(h, i, t, dt): return np.einsum("nsc,nsc->ns", post[h]["img"][i].astype(dt), post[h]["txt"][t].astype(dt))
sig = {}
for dt in (np.float32, np.float64):
    sig[dt.__name__] = {h: float(np.sqrt(np.mean((np.var(ag(h, ep.pairs_a_img, ep.pairs_a_txt, dt).astype(np.float64), 1, ddof=1)
                                            + np.var(ag(h, ep.pairs_b_img, ep.pairs_b_txt, dt).astype(np.float64), 1, ddof=1)) / 4))) for h in G}
stored_sig = json.loads((RES / "ra_sigma.json").read_text())["sigma"]
out = {"sigma_rel_diff_f32": {h: abs(sig["float32"][h] / stored_sig[h] - 1) for h in G},
       "sigma_rel_diff_f64": {h: abs(sig["float64"][h] / stored_sig[h] - 1) for h in G}}
z1 = np.load(T / "20261116_grouping_step1_style/results/step1_eval_style.npz")
for cfg, parts in (("A0", ("affect", "image", "caption")), ("A1", ("affect", "image", "caption", "csd"))):
    D = np.stack([ag(h, ep.pairs_a_img, ep.pairs_a_txt, np.float32).mean(1) - ag(h, ep.pairs_b_img, ep.pairs_b_txt, np.float32).mean(1) for h in parts], 1)
    sc = D.astype(np.float64) / np.array([stored_sig[h] for h in parts])
    picks = {"a": sc.argmax(1), "b": (-sc).argmax(1)}
    st = {d: np.stack([np.einsum("nc,nkc->nk", post[h]["img" if d == "i2t" else "txt"][ep.anchor], post[h]["txt" if d == "i2t" else "img"][ep.candidates]) for h in parts], 1) for d in ("i2t", "t2i")}
    r = np.arange(len(cl))
    Tm = {c: {d: st[d][r, picks[c]].astype(np.float32) for d in st} for c in "ab"}
    cf = {c: {d: (0.5 * (Tm["a"][d].astype(np.float64) + Tm["b"][d].astype(np.float64))).astype(np.float32) for d in st} for c in "ab"}
    pn = per_anchor(crossfit_nested(S.B, S.B, Tm, par)[0]); pc = per_anchor(crossfit_condition_free(S.B, S.B, cf, par)[0])
    pBp = {m: z1[f"{cfg}__Bprime__{m}"] for m in ("r1",)}
    means = [pBp["r1"].mean(), pc["r1"].mean(), S.pB["r1"].mean()]; k = int(np.argmax(means)); comp = [pBp["r1"], pc["r1"], S.pB["r1"]][k]
    bar = cluster_bootstrap(pn["r1"] - comp, cl); g = cluster_bootstrap(pn["gain"] - pc["gain"], cl)
    zz = np.load(RES / f"cand_Ra_{cfg}.npz")
    out[f"Ra_{cfg}"] = {"comparator": ["B_prime", "counterpart", "B"][k], "bar": [100 * bar["point"]] + [100 * x for x in bar["ci95"]],
                        "gain": [100 * g["point"]] + [100 * x for x in g["ci95"]],
                        "picks_equal": bool(all(np.array_equal(picks[c], zz[f"pick__{c}"]) for c in "ab")),
                        "fused_cf_equal": bool(np.array_equal(pn["r1"], zz["fused__r1"]) and np.array_equal(pc["r1"], zz["cf__r1"])),
                        "picks_change_with_f64_sigma": int(sum(((D.astype(np.float64) / np.array([sig['float64'][h] for h in parts])) * s).argmax(1).__ne__(picks[c]).sum() for c, s in (("a", 1), ("b", -1))))}
print(json.dumps(out, indent=1))
(Path(__file__).resolve().parent / "out/fr_scratch_ra.json").write_text(json.dumps(out, indent=1))
