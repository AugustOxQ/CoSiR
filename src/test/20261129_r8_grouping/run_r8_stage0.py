"""SC1a..e (rule §4): CPU regression through the r8 bundle, reader and shared helpers.

Full run: seed 42, R7's stored heads, rebuilt reader/text heads/CSD recipe, all
comparisons and fresh §10 hashes at both ends in results/stage0_regression.json.
--smoke: seed 9001, small banks/heads/recipe, shapes and wiring only; never a verdict.
It writes only under results/smoke/. Earlier-round writers are never called.
"""
import argparse
import json
import pickle
import time
import traceback
from pathlib import Path

import numpy as np
import r8_common as R
import r8_bundle as B
import r8_reader as Q
import r8_recipe as H
import r8_scoring_io as IO

CHECKS = ("SC1a", "SC1b", "SC1c", "SC1d", "SC1e")
METRICS = ("r1", "gain", "other", "strict", "swap")
TOL = 1e-12
R7_ROOT = R.MAIN / "src/test/20261126_design_l_style_source/results"


class Comparison:
    """Record every max difference and failure without stopping subsequent comparisons."""

    def __init__(self, smoke=False):
        self.smoke, self.keys, self.fails = smoke, {}, []

    def compare(self, key, ours, reference, *, exact=None, equal_nan=False):
        a = np.asarray(ours)
        if self.smoke:
            self.keys[key] = {"shape": list(a.shape), "dtype": str(a.dtype)}
            return
        b = np.asarray(reference)
        if exact is None:
            exact = (a.dtype.kind in "biuUS" or key.rsplit("/", 1)[-1] == "r1" or
                     key.endswith(tuple(f"__{m}" for m in METRICS)))
        ok, diff = False, None
        if a.shape == b.shape and a.dtype == b.dtype:
            if a.dtype.kind in "US":
                ok = bool(np.array_equal(a, b))
                diff = 0.0 if ok else None
            else:
                valid = np.isfinite(a) & np.isfinite(b)
                same_mask = bool(np.array_equal(np.isnan(a), np.isnan(b))) if equal_nan else True
                finite = bool((valid | (np.isnan(a) & np.isnan(b))).all()) if equal_nan else bool(valid.all())
                if same_mask and finite:
                    diff = float(np.max(np.abs(a[valid].astype(np.float64) - b[valid].astype(np.float64)))) if valid.any() else 0.0
                    ok = bool(np.array_equal(a, b, equal_nan=equal_nan)) if exact else diff <= TOL
        entry = {"pass": ok, "max_abs_diff": diff, "rule": "exact" if exact else f"<= {TOL}"}
        if a.shape != b.shape or a.dtype != b.dtype:
            entry.update(shapes=[list(a.shape), list(b.shape)], dtypes=[str(a.dtype), str(b.dtype)])
        self.keys[key] = entry
        if not ok:
            self.fails.append(f"{key}: {entry}")

    def result(self, **extra):
        return {"pass": not self.fails, "compared": self.keys, "failures": self.fails, **extra}


def read_npz(path, required):
    with np.load(path, allow_pickle=False) as z:
        assert set(required) <= set(z.files), f"{path}: missing keys {set(required) - set(z.files)}"
        return {k: z[k] for k in required}


def read_r0_labels(context):
    path = R7_ROOT / "groupings/r7_groupings.npz"
    a = read_npz(path, ("scorer_train", "g_st", "R0"))
    assert np.array_equal(a["scorer_train"], context.scorer_train), f"{path}: scorer-train differs"
    assert np.array_equal(a["g_st"], context.g_st), f"{path}: painting codes differ"
    labels = a["R0"]
    assert labels.dtype == np.int64 and labels.shape == context.scorer_train.shape, f"{path}: R0 shape/dtype"
    assert np.array_equal(np.unique(labels), np.arange(17)), f"{path}: R0 classes"
    assert np.array_equal(labels[context.first_rows][context.g_st], labels), f"{path}: R0 painting alignment"
    return labels


def standard_post(context, name):
    path = R7_ROOT / "heads" / f"post_{name}.npz"
    a = read_npz(path, ("rows", "img", "txt"))
    assert a["rows"].dtype == np.int64 and np.array_equal(a["rows"], context.selection), f"{path}: selection differs"
    post = {k: a[k] for k in ("img", "txt")}
    B._posterior_ok(post, context, name)
    for side in post:
        x = post[side][context.selection]
        assert (x >= 0).all() and np.allclose(x.sum(1), 1, rtol=0, atol=1e-5), f"{path}/{side}: probability sums"
    return post


def load_refs():
    """Required rule-§4 array families only; all files have already passed §10 hashes."""
    with np.load(R7_ROOT / "seed42/arrays.npz", allow_pickle=False) as z:
        prefixes = ("AFF__", "AFF_cf__", "R1_R0__", "R1_R0_cf__", "XAFF_R0__", "XAFF_R0_cf__",
                    "B__", "B0__", "B1__", "BX_R0__")
        keys = [k for k in z.files if k.startswith(prefixes) or k in ("cl", "pair_index", "parity")]
        arrays = {k: z[k] for k in keys}
    for prefix in prefixes:
        assert {f"{prefix}{m}" for m in METRICS} <= set(arrays), f"R7: missing {prefix} metrics"
    for key, a in arrays.items():
        if key.endswith(tuple(f"__{m}" for m in METRICS)):
            assert a.shape == (12_288,) and a.dtype == np.float64 and np.isfinite(a).all(), f"R7/{key}: metrics"
    for key in ("cl", "pair_index", "parity"):
        assert arrays[key].shape == (12_288,) and arrays[key].dtype == np.int64, f"R7/{key}: episode IDs"
    r3keys = ("aff_fused_cells", "aff_cf_cells", "aff_sigma", "aff_gate__a", "aff_gate__b") + tuple(
        f"aff_{who}__{m}" for who in ("fused", "cf") for m in METRICS)
    r3 = read_npz(R.MAIN / "src/test/20261121_round3_affect_gate/results/seed42_arrays.npz", r3keys)
    return {"arrays": arrays, "r3": r3,
            "reader": read_npz(R7_ROOT / "seed42/reader_R0.npz", tuple(f"{k}__{c}" for k in ("P", "m", "pick") for c in ("a", "b"))),
            "reader_record": json.loads((R7_ROOT / "readers/reader_R0.json").read_text()),
            "taus": np.asarray(json.loads((R7_ROOT / "taus.json").read_text())["R0"], dtype=np.float64)}


def compare_family(c, ours, refs, prefixes):
    keys = {k for k in refs if k.startswith(tuple(prefixes))}
    assert keys and keys <= set(ours), f"missing output array keys: {keys - set(ours)}"
    assert keys == {k for k in ours if k.startswith(tuple(prefixes))}, "array family keys differ"
    for key in sorted(keys):
        c.compare(key, ours[key], refs[key])


def sc1a(state):
    bundle, smoke, refs = state["bundle"], state["smoke"], state["refs"]
    c = Comparison(smoke)
    # The A0 build sequence stays unchanged in the copied bundle. Verify its heads,
    # then explicitly use R7's stored standard posteriors as the SC1 scoring input.
    for name in B.A0:
        post = standard_post(bundle.ctx, name)
        for side in ("img", "txt"):
            c.compare(f"input/{name}/{side}", bundle.post[name][side], post[side], equal_nan=True)
        bundle.post[name] = post
    bundle.stack, bundle.F = B.stack_and_features(bundle, B.A0)
    out = state["fusion"].score_aff(bundle)
    r3 = refs["r3"] if refs else {}
    for name, key in (("fused_cells", "aff_fused_cells"), ("cf_cells", "aff_cf_cells"),
                      ("gate_a", "aff_gate__a"), ("gate_b", "aff_gate__b")):
        c.compare(f"R3/{key}", out[name], r3.get(key))
    c.compare("R3/aff_sigma", out["arrays"]["AFF__sigma"], r3.get("aff_sigma"))
    for who in ("fused", "cf"):
        for m in METRICS:
            key = f"aff_{who}__{m}"
            c.compare(f"R3/{key}", out[who][m], r3.get(key))
    if not smoke:
        c.compare("fixed/fused_cells", out["fused_cells"], np.array([39, 119], dtype=np.int64))
        c.compare("fixed/cf_cells", out["cf_cells"], np.array([149, 10], dtype=np.int64))
        c.compare("fixed/sigma", out["arrays"]["AFF__sigma"], np.zeros(2, dtype=np.float64))
        compare_family(c, out["arrays"], refs["arrays"], ("AFF__", "AFF_cf__"))
    else:
        for key, a in out["arrays"].items():
            c.compare(key, a, None)
    return c.result()


def sc1b(state):
    bundle, smoke, refs, root = (state[k] for k in ("bundle", "smoke", "refs", "root"))
    c = Comparison(smoke)
    labels, halves, post = IO.reader_inputs("R0", R7_ROOT)
    rec = Q.build_reader("R0", labels, halves, root=root, smoke=smoke,
                         bank_n=32 if smoke else Q.BANK_N, post=post)
    assert not rec["dropped"], f"R0 reader dropped: {rec.get('reason')}"
    fields = Q.EPISODE_FIELDS + ("block_pairs", "block_size", "seed", "half")
    for half in (0, 1):
        a = read_npz(root / f"readers/bank_R0_half{half}.npz", fields)
        b = read_npz(R7_ROOT / f"readers/bank_R0_half{half}.npz", fields) if refs else {}
        for key in fields:
            c.compare(f"bank/half{half}/{key}", a[key], b.get(key))
        c.compare(f"reader/chosen_C_half{half}", np.array(rec["halves"][str(half)]["chosen_C"], dtype=np.float64),
                  np.array(refs["reader_record"]["halves"][str(half)]["chosen_C"], dtype=np.float64) if refs else None,
                  exact=True)
    with (root / "readers/reader_R0.pkl").open("rb") as f:
        reader = pickle.load(f)
    fusion = state["fusion"]
    rd = fusion.reader(bundle, B.A1, reader)
    for kind in ("P", "m", "pick"):
        for cond in ("a", "b"):
            key = f"{kind}__{cond}"
            c.compare(f"reader/{key}", rd[kind][cond], refs["reader"].get(key) if refs else None)
    taus, n_margins = fusion.thresholds(rd["m"])
    assert n_margins == 2 * bundle.n, "threshold margin count"
    c.compare("tau_R0", np.asarray(taus, dtype=np.float64), refs["taus"] if refs else None)
    for score in (fusion.score_r1, fusion.score_x_aff):
        out = score(bundle, B.A1, reader, taus)
        prefixes = ("R1_R0__", "R1_R0_cf__") if score is fusion.score_r1 else ("XAFF_R0__", "XAFF_R0_cf__")
        if refs:
            compare_family(c, out["arrays"], refs["arrays"], prefixes)
        else:
            for key, a in out["arrays"].items():
                c.compare(key, a, None)
    return c.result(taus=taus, n_margins=n_margins)


def sc1c(state):
    bundle, refs = state["bundle"], state["refs"]
    c = Comparison(state["smoke"])
    bx, _ = B.condition_free(bundle, B.A1)
    arrays = {k: getattr(bundle, k) for k in ("cl", "pair_index", "parity")}
    for name, pa in (("B", bundle.pB), ("B0", bundle.pBp), ("B1", bx), ("BX_R0", bx)):
        arrays.update({f"{name}__{m}": pa[m] for m in METRICS})
    for key, a in arrays.items():
        c.compare(key, a, refs["arrays"].get(key) if refs else None)
    return c.result()


def sc1d(state):
    context = R.ctx()
    c = Comparison(state["smoke"])
    post, cross, record = H.fit_clip_heads(context.data, state["r0_labels"], context.scorer_train,
                                         context.selection, context.halves_dict, smoke=state["smoke"],
                                         head_rows=5000 if state["smoke"] else None,
                                         check_rows=128 if state["smoke"] else None)
    ref_post = standard_post(context, "R0") if not state["smoke"] else {}
    ref_cross = read_npz(R7_ROOT / "heads/xheads_R0.npz", ("txt", "filled_by_half", "classes_half0", "classes_half1")) if not state["smoke"] else {}
    c.compare("post_R0/txt", post["txt"], ref_post.get("txt"), equal_nan=True)
    for key in ("txt", "filled_by_half", "classes_half0", "classes_half1"):
        c.compare(f"xheads_R0/{key}", cross[key], ref_cross.get(key))
    return c.result(head_rows=record["head_rows"], check_rows=record["check_rows"])


def sc1e(state):
    context = R.ctx()
    c = Comparison(state["smoke"])
    source = np.load("/data/SSD2/pre_extract/artelingo/style_csd_vitl/embeddings.npy", mmap_mode="r", allow_pickle=False)
    assert source.shape == (61_402, 768) and source.dtype == np.float32, "CSD source: shape/dtype"
    phi = R.unit(np.asarray(source[R.painting_image()], dtype=np.float32))
    g = context.g_st
    if state["smoke"]:
        phi = phi[:256]
        g = g[g < len(phi)]
    graph = H.recipe_graph(phi)
    labels, record = H.recipe_at_resolution(phi, g, graph, resolution=1.0)
    c.compare("R0/labels", labels, state["r0_labels"] if not state["smoke"] else None)
    if not state["smoke"]:
        c.compare("R0/n_groups", np.array(record["K"], dtype=np.int64), np.array(17, dtype=np.int64))
    return c.result(recipe=record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    root = R.SMOKE / "stage0" if args.smoke else R.RESULTS
    path = root / "stage0_regression.json"
    # Never reuse real rebuilds: every full invocation refits, using a fresh fix root if needed.
    if not args.smoke and path.exists():
        path = R.fix_path(path)
        root = R.RESULTS / path.stem
    seed = R.SMOKE_SEEDS[0] if args.smoke else R.DEV_SEED
    started = time.monotonic()
    rec = {"stage": "stage0", "smoke": args.smoke, "seed": seed, "results_root": str(root),
           "checks_requested": list(CHECKS), "checks": {}}
    # rule gap: wiring smoke hashes only inputs it uses; the full SC1 hashes all §10 inputs.
    keys = tuple(k for k in R.INPUTS if k.startswith(("20261126_design_l_style_source/", "20261121_round3_",
                 "20261031_pseudo_", "20261116_grouping_step1_", "20261117_reader_fix_")) or
                 k.startswith("/data/SSD2/pre_extract/artelingo/style_csd_vitl/") or
                 k == "/data/SSD2/pre_extract/artelingo/features/metadata.json") if args.smoke else None
    state = {"smoke": args.smoke, "root": root}
    try:
        rec["rule_sha256_start"] = R.assert_rule()
        R.assert_threads()
        rec["sha256_start"] = R.assert_inputs(keys=keys, fresh=True)
        state["refs"] = None if args.smoke else load_refs()
        state["r0_labels"] = read_r0_labels(R.ctx())
        state["fusion"] = R.load_r7("r7_fusion")
        state["bundle"] = B.build_bundle(seed, args.smoke, extra=("R0",), root=R7_ROOT)
        for name, fn in zip(CHECKS, (sc1a, sc1b, sc1c, sc1d, sc1e)):
            try:
                rec["checks"][name] = fn(state)
            except (Exception, SystemExit) as e:
                rec["checks"][name] = {"pass": False, "error": f"{type(e).__name__}: {e}"}
                traceback.print_exc()
            print(f"{name}: {'wired' if args.smoke else 'pass'}" if rec["checks"][name]["pass"] else f"{name}: FAIL", flush=True)
    except (Exception, SystemExit) as e:
        rec["checks"] = {name: {"pass": False, "error": f"build: {type(e).__name__}: {e}"} for name in CHECKS}
        traceback.print_exc()
    try:
        rec["rule_sha256_end"] = R.assert_rule()
        rec["sha256_end"] = R.assert_inputs(keys=keys, fresh=True)
        rec["sha256_unchanged"] = (rec["sha256_end"] == rec.get("sha256_start") and
                                   rec["rule_sha256_start"] == rec["rule_sha256_end"])
    except (Exception, SystemExit) as e:
        rec["sha256_end"], rec["sha256_unchanged"] = str(e), False
    ok = set(rec["checks"]) == set(CHECKS) and all(x["pass"] for x in rec["checks"].values()) and rec["sha256_unchanged"]
    rec.update({"pass": bool(ok and not args.smoke), "wiring_pass": bool(ok) if args.smoke else None,
                "wall_s": round(time.monotonic() - started, 3)})
    R.write_json_once(path, rec)
    print(f"stage 0 {'wired' if args.smoke and ok else 'passed' if ok else 'FAILED'}; record {path}")
    if not ok:
        raise RuntimeError(f"SC1 failed; record written to {path}")


if __name__ == "__main__":
    main()
