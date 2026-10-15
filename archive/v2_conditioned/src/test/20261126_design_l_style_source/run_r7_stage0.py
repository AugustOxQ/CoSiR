"""Round 7 stage 0: the regression checks G1 to G4 (DECISION_RULE.md §4, §9 order and read-only, §10 inputs; spec D22).

Order (§9). assert_rule and assert_inputs (fresh) -> load every reference (R3 seed42_arrays, the spike's told and
R1 files, round 1's rb_reader_A1.pkl/.json, rb_heads_*, rb_bank_A1_half*, R6 seed42_per_episode) -> build the real
heads xheads/post/head_{affect,image,caption,R0} and the R0 reader (kept as the round's own files, write-once; a
file already there is read, not rebuilt) -> seed-42 bundle -> G1 to G4 -> assert_inputs again (fresh).

G1  AFF through score_aff and through score_x_aff with allowed {affect}: cells, sigma*, gates and all per-anchor arrays
    equal R3's aff_fused__* / aff_cf__*; pooled 19.137, +0.700 over B0, +0.332 over B1.
G2  told A1 (one-hot, 224 cells) equals fused_r1 and margin_r1 of sp_cand_told_A1_224.npz.
G3  cross-fitted heads vs rb_heads_* (1e-12), banks vs rb_bank_A1_half* (exact), chosen C per half (exact), reader
    predict_proba on the seed-42 A1 features vs rb_reader_A1.pkl (1e-12); R1[R0] with tau from this reader's seed-42
    margins vs sp_cand_R1_A1_224.npz fused_r1 and margin_r1 (exact). Round 1 names the fourth head csd, ours R0:
    arrays are compared in column order, names never across the two.
G4  B, B0 equal R3 D10, D11 (18.341064453125, 18.436686197916664); AFF - B1 per episode equals R6 diff__S1.

A check that fails is recorded (compared keys, max abs differences) and the run exits 1 after writing
stage0_regression.json; nothing is improvised (§8). --chain-smoke runs the same wiring on seed 9001 under
RESULTS/smoke/chain with synthetic-size heads and banks and records shapes and keys only. --checks G1,G3 runs a subset
(the record is then stage0_regression_<checks>.json). Writes only under RESULTS (or the smoke root).
"""
import r7_common as R
import r7_bundle as Bd
import r7_fusion as F
import r7_heads as H
import r7_reader as RD

import argparse
import json
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np

A0, A1 = Bd.A0, Bd.A1
BASE = A0
HEAD_NAMES = ("affect", "image", "caption", "R0")
R1_NAME = {"affect": "affect", "image": "image", "caption": "caption", "R0": "csd"}   # round 1's file names
METRICS = ("r1", "gain", "other", "swap", "strict")
TOL = 1e-12
D10, D11 = 18.341064453125, 18.436686197916664
REFS = {
    "r3": "20261121_round3_affect_gate/results/seed42_arrays.npz",
    "told": "20261119_no_caption_csd_spike/results/sp_cand_told_A1_224.npz",
    "r1": "20261119_no_caption_csd_spike/results/sp_cand_R1_A1_224.npz",
    "r6": "20261125_artelingo_held_test/results/seed42_per_episode.npz",
}
CHECKS = ("G1", "G2", "G3", "G4")


# ---------------------------------------------------------------- comparisons

class Cmp:
    """Collects the compared keys with their max abs differences; ``fails`` lists every mismatch. In smoke mode
    (no reference values) only shapes and dtypes are recorded."""

    def __init__(self, smoke):
        self.smoke, self.keys, self.fails = smoke, {}, []

    def _diff(self, key, a, b, tol):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            self.keys[key] = {"max_abs_diff": None, "shapes": [list(a.shape), list(b.shape)]}
            self.fails.append(f"{key}: shapes {a.shape} vs {b.shape}")
            return
        if a.dtype.kind in "US" or b.dtype.kind in "US":
            ok, d = bool(np.array_equal(a, b)), 0.0
        else:
            d = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.0
            ok = bool(np.array_equal(a, b)) if tol == 0 else d <= tol
        self.keys[key] = {"max_abs_diff": d, "rule": "exact" if tol == 0 else f"<= {tol}"}
        if not ok:
            self.fails.append(f"{key}: max abs diff {d} ({'exact' if tol == 0 else tol})")

    def compare(self, key, ours, ref, tol=0):
        """tol 0: numpy.array_equal (integers, per-anchor R@1); else max abs difference <= tol."""
        if self.smoke:
            a = np.asarray(ours)
            self.keys[key] = {"shape": list(a.shape), "dtype": str(a.dtype)}
            return
        self._diff(key, ours, ref, tol)

    def value(self, key, ok, detail):
        """A stated number or rule (not a reference array): recorded, and a failure if not ok (real runs only)."""
        self.keys[key] = detail
        if not self.smoke and not ok:
            self.fails.append(f"{key}: {detail}")

    def result(self, **extra):
        return {"pass": not self.fails, "compared": self.keys, "failures": self.fails, **extra}


def pooled(x):
    return 100 * float(np.mean(x))


# ---------------------------------------------------------------- references (read before any round-7 write)

def load_refs():
    refs = {}
    for k, key in REFS.items():
        path = R.path_of(key)
        assert R.sha_file(path) == R.INPUTS[key], f"{key}: SHA-256 differs"
        with np.load(path, allow_pickle=False) as z:
            refs[k] = {f: z[f] for f in z.files}
    key = "20261117_reader_fix_csd/results/rb_reader_A1.pkl"
    assert R.sha_file(R.path_of(key)) == R.INPUTS[key], f"{key}: SHA-256 differs"
    with open(R.path_of(key), "rb") as f:
        refs["reader_pkl"] = pickle.load(f)
    res = R.R1D / "results"
    refs["reader_json"] = json.loads((res / "rb_reader_A1.json").read_text())
    for n in HEAD_NAMES:
        with np.load(res / f"rb_heads_{R1_NAME[n]}.npz", allow_pickle=False) as z:
            refs[f"heads_{n}"] = {f: z[f] for f in z.files}
    for j in (0, 1):
        with np.load(res / f"rb_bank_A1_half{j}.npz", allow_pickle=False) as z:
            refs[f"bank_{j}"] = {f: z[f] for f in z.files}
    refs["files_sha256"] = {str(p.relative_to(R.TEST)): R.sha_file(p) for p in sorted(res.glob("rb_heads_*.npz"))
                            if p.stem.split("_")[-1] in R1_NAME.values()}
    refs["files_sha256"].update({f"rb_bank_A1_half{j}.npz": R.sha_file(res / f"rb_bank_A1_half{j}.npz") for j in (0, 1)})
    refs["files_sha256"]["rb_reader_A1.json"] = R.sha_file(res / "rb_reader_A1.json")
    return refs


# ---------------------------------------------------------------- build the kept heads and the R0 reader

def build_kept(root, smoke):
    """Heads affect, image, caption, R0 and reader R0 under root (an existing real file is read, not rebuilt)."""
    rb = R.load("rb_build")
    data, sp, st, groups = rb.splits()
    halves = H.load_halves(st, groups)
    labels = H.load_groupings(BASE + ("csd",), st, halves["g_st"], root)
    labels["R0"] = labels.pop("csd")                  # round 1's csd labels, the frozen R0 (build_heads asserts it)
    gfile = root / "groupings" / "r7_groupings.npz"
    if gfile.is_file():
        with np.load(gfile, allow_pickle=False) as z:
            assert np.array_equal(z["R0"], labels["R0"]), "groupings file: R0 differs from round 1's csd labels"
    built = {}
    for name in HEAD_NAMES:
        exists = (root / "heads" / f"head_{name}.json").is_file() and not smoke
        if not exists:
            rec = H.build_heads(name, data, sp, labels[name], halves, root, smoke)
            assert not rec["dropped"], f"head {name} dropped: {rec.get('reason')}"
        built[name] = "read" if exists else "built"
    if (root / "readers" / "reader_R0.json").is_file() and not smoke:
        built["reader_R0"] = "read"
    else:
        rec = RD.build_reader("R0", labels, halves, root, smoke)
        assert not rec["dropped"], f"reader R0 dropped: {rec.get('reason')}"
        built["reader_R0"] = "built"
    return built, halves


# ---------------------------------------------------------------- the checks

def g1(ctx):
    b, refs, c = ctx["bundle"], ctx["refs"], Cmp(ctx["smoke"])
    r3 = refs["r3"] if refs else None
    pk = b.readers
    routes = {"score_aff": F.score_aff(b),
              "score_x_aff{affect}": F.score_x_aff(b, A0, pk, F.TAUS, allowed={A0.index("affect")}, name="AFF")}
    for tag, out in routes.items():
        g = lambda k: r3[k] if r3 else None
        c.compare(f"{tag}/fused_cells", out["fused_cells"], g("aff_fused_cells"))
        c.compare(f"{tag}/cf_cells", out["cf_cells"], g("aff_cf_cells"))
        c.compare(f"{tag}/sigma", out["arrays"]["AFF__sigma"], g("aff_sigma"))
        c.compare(f"{tag}/gate_a", out["gate_a"], g("aff_gate__a"))
        c.compare(f"{tag}/gate_b", out["gate_b"], g("aff_gate__b"))
        for m in METRICS:
            c.compare(f"{tag}/aff_fused__{m}", out["fused"][m], g(f"aff_fused__{m}"))
            c.compare(f"{tag}/aff_cf__{m}", out["cf"][m], g(f"aff_cf__{m}"))
    a, b2 = routes["score_aff"], routes["score_x_aff{affect}"]
    for k in ("fused_cells", "cf_cells", "gate_a", "gate_b"):
        assert np.array_equal(a[k], b2[k]), f"the two G1 routes differ in {k}"
    for m in METRICS:
        assert np.array_equal(a["fused"][m], b2["fused"][m]) and np.array_equal(a["cf"][m], b2["cf"][m]), \
            f"the two G1 routes differ in {m}"
    c.compare("taus", np.array(F.TAUS), r3["taus"] if r3 else None)
    if r3 is not None:
        c.value("fused_cells_39_119", a["fused_cells"].tolist() == [39, 119], a["fused_cells"].tolist())
        c.value("cf_cells_149_10", a["cf_cells"].tolist() == [149, 10], a["cf_cells"].tolist())
        c.value("sigma_zero", a["arrays"]["AFF__sigma"].tolist() == [0., 0.], a["arrays"]["AFF__sigma"].tolist())
        c.value("pooled_19.137", round(pooled(a["fused"]["r1"]), 3) == 19.137, pooled(a["fused"]["r1"]))
        c.value("over_B0_0.700", round(pooled(a["fused"]["r1"]) - pooled(b.pBp["r1"]), 3) == 0.700,
                pooled(a["fused"]["r1"]) - pooled(b.pBp["r1"]))
        b1 = Bd.condition_free(b, A1)[0]
        c.value("over_B1_0.332", round(pooled(a["fused"]["r1"]) - pooled(b1["r1"]), 3) == 0.332,
                pooled(a["fused"]["r1"]) - pooled(b1["r1"]))
    ctx["aff"] = a
    return c.result()


def g2(ctx):
    b, refs, c = ctx["bundle"], ctx["refs"], Cmp(ctx["smoke"])
    out = F.told(b, A1, {"emotion": "affect", "style": "R0", "genre": "image"})
    t = refs["told"] if refs else None
    g = lambda k: t[k] if t else None
    c.compare("anchor_group", b.cl, g("anchor_group"))
    c.compare("pair_index", b.pair_index, g("pair_index"))
    margin = out["fused"]["r1"] - out["cf"]["r1"]
    c.compare("fused_r1", out["fused"]["r1"], g("fused_r1"))
    c.compare("margin_r1", margin, g("margin_r1"))
    if t:
        sg = b.pair_index == 2
        c.value("style_x_genre_21.82", round(pooled(out["fused"]["r1"][sg]), 2) == 21.82, pooled(out["fused"]["r1"][sg]))
        c.value("style_x_genre_margin_0.134", round(pooled(margin[sg]), 3) == 0.134, pooled(margin[sg]))
    return c.result()


def g3(ctx):
    b, refs, c, root = ctx["bundle"], ctx["refs"], Cmp(ctx["smoke"]), ctx["root"]
    g = lambda *k: _dig(refs, *k) if refs else None
    # (a) cross-fitted heads
    for n in HEAD_NAMES:
        with np.load(root / "heads" / f"xheads_{n}.npz", allow_pickle=False) as z:
            ours = {f: z[f] for f in z.files}
        assert {"img", "txt", "filled_by_half", "classes_half0", "classes_half1"} <= set(ours), f"xheads_{n}: keys"
        for k in ("img", "txt"):
            c.compare(f"heads/{n}/{k}", ours[k], g(f"heads_{n}", k), tol=TOL)
        for k in ("filled_by_half", "classes_half0", "classes_half1"):
            c.compare(f"heads/{n}/{k}", ours[k], g(f"heads_{n}", k))
    # (b) banks
    for j in (0, 1):
        with np.load(root / "readers" / f"bank_R0_half{j}.npz", allow_pickle=False) as z:
            ours = {f: z[f] for f in z.files}
        for k in RD.EPISODE_FIELDS + ("block_pairs", "block_size", "seed", "half"):
            c.compare(f"banks/half{j}/{k}", ours[k], g(f"bank_{j}", k))
    # (c) chosen C per half
    rec = json.loads((root / "readers" / "reader_R0.json").read_text())
    for j in (0, 1):
        c.compare(f"reader/chosen_C_half{j}", np.array([rec["halves"][str(j)]["chosen_C"]]),
                  np.array([g("reader_json")["halves"][str(j)]["chosen_C"]]) if refs else None)
    # (d) predict_proba on the seed-42 A1 features, columns in order (names csd/R0 never compared)
    with open(root / "readers" / "reader_R0.pkl", "rb") as f:
        pk = pickle.load(f)
    assert len(pk["feature_names"]) == 24 and len(pk["halves"]) == 2, "reader_R0.pkl: layout"
    rbe = R.load("rb_eval")
    _, feats = Bd.stack_and_features(b, A1)
    for cond in F.CONDITIONS:
        mine = rbe.half_reader_probs(pk, feats[cond], len(A1))
        theirs = rbe.half_reader_probs(g("reader_pkl"), feats[cond], len(A1)) if refs else [None, None]
        for j in (0, 1):
            c.compare(f"reader/predict_proba/{cond}/half{j}", mine[j], theirs[j], tol=TOL)
    # (e) R1[R0] with tau from this reader's seed-42 margins
    rd = F.reader(b, A1, pk)
    taus, n_m = F.thresholds(rd["m"])
    out = F.score_r1(b, A1, pk, taus, name="R1_R0")
    r1 = refs["r1"] if refs else None
    h = lambda k: r1[k] if r1 else None
    c.compare("R1_R0/anchor_group", b.cl, h("anchor_group"))
    c.compare("R1_R0/pair_index", b.pair_index, h("pair_index"))
    c.compare("R1_R0/fused_r1", out["fused"]["r1"], h("fused_r1"))
    c.compare("R1_R0/margin_r1", out["fused"]["r1"] - out["cf"]["r1"], h("margin_r1"))
    return c.result(taus=[float(t) for t in taus], n_margins=int(n_m), fused_cells=out["fused_cells"].tolist(),
                    cf_cells=out["cf_cells"].tolist())


def g4(ctx):
    b, refs, c = ctx["bundle"], ctx["refs"], Cmp(ctx["smoke"])
    aff = ctx.get("aff") or F.score_aff(b)
    b1, picks = Bd.condition_free(b, A1)
    assert set(picks) >= {0, 1} or len(picks) == 2, "B1 picks"
    d = aff["fused"]["r1"] - b1["r1"]
    r6 = refs["r6"] if refs else None
    g = lambda k: r6[k] if r6 else None
    c.compare("cl", b.cl, g("cl"))
    c.compare("pair_index", b.pair_index, g("pair_index"))
    c.compare("AFF_minus_B1_vs_diff__S1", d, g("diff__S1"))
    c.compare("B_pooled", np.array([pooled(b.pB["r1"])]), np.array([D10]), tol=1e-9)
    c.compare("B0_pooled", np.array([pooled(b.pBp["r1"])]), np.array([D11]), tol=1e-9)
    return c.result(B=pooled(b.pB["r1"]), B0=pooled(b.pBp["r1"]), B1=pooled(b1["r1"]),
                    B1_picks={str(k): v for k, v in dict(picks).items()} if isinstance(picks, dict) else str(picks))


def _dig(refs, *k):
    x = refs
    for key in k:
        x = x[key]
    return x


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain-smoke", action="store_true", help="seed 9001, smoke sizes, shapes and keys only")
    p.add_argument("--checks", default=",".join(CHECKS), help="comma list of G1..G4 (default all)")
    args = p.parse_args()
    checks = [x for x in args.checks.split(",") if x]
    assert checks and set(checks) <= set(CHECKS), f"--checks must be among {CHECKS}"
    smoke = args.chain_smoke
    roles = R.roles("chain" if smoke else "real")
    root, seed = roles.root, roles.dev
    R.assert_rule()
    R.assert_threads()
    rec = {"stage": "stage0", "smoke": smoke, "seed": seed, "results_root": str(root), "checks_requested": checks}
    rec["sha256_start"] = R.assert_inputs(fresh=True)
    refs = None if smoke else load_refs()           # every reference is read before any round-7 write
    if refs:
        rec["reference_files_sha256"] = refs["files_sha256"]
    out = {}
    ctx = {"smoke": smoke, "refs": refs, "root": root}
    try:
        rec["built"], _ = build_kept(root, smoke)
        ctx["bundle"] = Bd.build_bundle(seed, smoke, extra=("R0",), root=root)
    except Exception as e:                           # nothing can be checked without them
        out = {k: {"pass": False, "error": f"build: {type(e).__name__}: {e}"} for k in checks}
        traceback.print_exc()
    if not out:
        for name, fn in (("G1", g1), ("G2", g2), ("G3", g3), ("G4", g4)):
            if name not in checks:
                continue
            try:
                out[name] = fn(ctx)
            except Exception as e:
                traceback.print_exc()
                out[name] = {"pass": False, "error": f"{type(e).__name__}: {e}"}
            print(f"{name}: {'pass' if out[name]['pass'] else 'FAIL'}"
                  + ("" if out[name]["pass"] else f" {out[name].get('failures') or out[name].get('error')}"))
    rec["checks"] = out
    try:                                             # §9: the SHA-256s again at the end of stage 0
        rec["sha256_end"] = R.assert_inputs(fresh=True)
        rec["sha256_unchanged"] = rec["sha256_end"] == rec["sha256_start"]
    except SystemExit as e:
        rec["sha256_end"], rec["sha256_unchanged"] = str(e), False
    rec["pass"] = bool(all(v["pass"] for v in out.values()) and rec["sha256_unchanged"]
                       and set(out) == set(checks))
    name = "stage0_regression.json" if set(checks) == set(CHECKS) else f"stage0_regression_{'_'.join(checks)}.json"
    path = root / name
    if path.exists() and not R.in_smoke(path):
        path = R.fix_path(path)
    R.write_json_once(path, rec)
    print(f"stage 0 {'passed' if rec['pass'] else 'FAILED'}; record {path}")
    sys.exit(0 if rec["pass"] else 1)


if __name__ == "__main__":
    main()
