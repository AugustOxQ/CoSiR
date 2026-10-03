"""H3 learnability scorer (PREREGISTRATION.md §7), CPU.

Fresh label episodes on scorer-train rows (fit vs A3), seed-42 selection episodes (ceiling, matched-k). No val, held or
seed 43/44/45 episodes. LAB checkpoints are scored for H3 only.

Real run:  python score_h3.py          -> results/h3.json (overwrites an existing one only if it recorded needs_seed43)
Smoke run: python score_h3.py --smoke  -> results/smoke/ (A1's smoke checkpoint stands in for A3 and C0)
Exit code 3: a LAB fit is inconclusive and its seed-43 checkpoint does not exist (needs_seed43; h3_reading is null).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from common import C0_CKPT, FRESH_LABEL_SEED, LAB_PARTS, LABEL_RUNS, N_FRESH, ROOT, folders, local_rows, rg  # noqa: E402

from src.data.artelingo_splits import artelingo_splits  # noqa: E402
from src.eval.aspect_episodes import (PaintingValueIndex, build_aspect_episodes, concat_episodes,  # noqa: E402
                                      episodes_sha256, validate_aspect_episodes)
from src.eval.aspect_metrics import METRICS, compare, per_anchor  # noqa: E402
from src.eval.aspect_nested import crossfit_nested, fit_reading, h3_reading  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, agreement_term  # noqa: E402
from src.train.train_factors import encode_rows, load_factor_checkpoint  # noqa: E402

PAIRS = (("emotion", "style", "genre"), ("emotion", "genre", "style"), ("style", "genre", "emotion"))
SEED = 42
RUNS_SCORED = LABEL_RUNS + ("MK3",)


def either_point(arr, cl):
    v = compare({"e": arr}, {"e": np.zeros_like(arr)}, cl, "e")
    return {"point": v["point"], "ci95": v["ci95"]}


def ckpt_path(run, seed, smoke):
    """Smoke: A1's smoke checkpoint stands in for A3/C0; this folder's smoke checkpoints for the rest."""
    if run in ("A3", "C0"):
        return rg.checkpoint_path("A1", SEED, True) if smoke else (rg.checkpoint_path("A3", SEED, False)
                                                                     if run == "A3" else C0_CKPT)
    return folders(smoke)["ckpt"] / f"{run}_seed{seed}.pt"


def failed_path(run, seed, smoke):
    return folders(smoke)["res"] / f"failed_{run}_seed{seed}.json"


def run_state(run, seed, smoke):
    """ok (checkpoint, no failed record), failed (failed record, no checkpoint); anything else is an error."""
    ck, fp = ckpt_path(run, seed, smoke), failed_path(run, seed, smoke)
    if fp.exists():
        assert not ck.exists(), f"{run} seed {seed}: both {fp.name} and a checkpoint exist"
        return "failed"
    if ck.exists():
        return "ok"
    raise FileNotFoundError(f"{run} seed {seed}: neither {ck} nor {fp.name}; not trained yet or crashed without a record")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke
    f = folders(smoke)
    res = f["res"]
    out_json = res / "h3.json"
    if out_json.exists() and not smoke:
        prev = json.loads(out_json.read_text())
        if not prev.get("needs_seed43"):
            raise SystemExit(f"{out_json} exists without needs_seed43; refusing to overwrite a real H3 result")
    res.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    pilot = json.loads((res / "pilot_seed42.json").read_text())
    a3_path = ckpt_path("A3", SEED, smoke)
    a3_sha = rg.sha_file(a3_path)
    assert pilot["A3"]["provenance"]["sha256"] == a3_sha, "pilot A3 SHA-256 != the A3 checkpoint loaded here"
    g_star = float(pilot["A3"]["g_star"])
    state = {run: run_state(run, SEED, smoke) for run in RUNS_SCORED}

    ctx = rg.EvalContext(SEED, smoke)
    data = ctx.data
    st, local_groups = local_rows(data, artelingo_splits(data))
    n = len(st)

    # ---- fresh label episodes on scorer-train rows
    z = np.load(res / "partitions_LAB.npz")
    lab = {k: z[k] for k in z.files if k in LAB_PARTS}
    groups = z["local_groups"]
    assert np.array_equal(groups, local_groups), "partitions_LAB local_groups != scorer-train local groups"
    assert all(len(v) == n for v in lab.values())
    n_fresh = 256 if smoke else N_FRESH
    index = PaintingValueIndex(lab, groups)
    parts, fresh_prov = [], {}
    for i, (a, b, third) in enumerate(PAIRS):
        ep = build_aspect_episodes(lab, groups, np.arange(n), a, b, n_fresh, FRESH_LABEL_SEED + i, third=third,
                                   min_paintings=30, index=index)
        validate_aspect_episodes(ep, lab, groups, index, third=third)
        assert len(ep.anchor) == n_fresh, f"{a}__{b}: {len(ep.anchor)} episodes, expected {n_fresh}"
        fresh_prov[f"{a}__{b}"] = {"sha256": episodes_sha256(ep), "seed": FRESH_LABEL_SEED + i, "third": third,
                                   "n": int(len(ep.anchor))}
        parts.append(ep)
    fresh = concat_episodes(parts)
    fclusters = groups[fresh.anchor]

    def term_fresh(ckpt):
        model, _ = load_factor_checkpoint(ckpt, device="cpu")
        ic, tc = encode_rows(model, data.img_features, data.txt_features, rows=st, device="cpu")
        assert np.isfinite(ic).all() and np.isfinite(tc).all(), f"{ckpt}: non-finite codes"
        return per_anchor(agreement_term(EvalInputs(data.img_features[st], data.txt_features[st], ic, tc), fresh))

    def term_transfer(ckpt):
        ic, tc = ctx.encode(ckpt)
        return per_anchor(agreement_term(EvalInputs(ctx.img, ctx.txt, ic, tc), ctx.pooled)), (ic, tc)

    pa_a3 = term_fresh(a3_path)
    pa_c0 = term_fresh(ckpt_path("C0", SEED, smoke))
    arrays = {f"fresh__A3__{m}": np.asarray(pa_a3[m]) for m in METRICS}
    result = {"provenance": {"smoke": smoke, "n_scorer_train_rows": int(n), "a3_checkpoint_sha256": a3_sha,
                             "seed42_episodes_sha256": ctx.shas, "n_seed42_episodes": int(ctx.n),
                             "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
              "fresh": {"episodes": fresh_prov, "n_total": int(len(fresh.anchor)),
                        "n_clusters": int(len(np.unique(fclusters)))}}

    # ---- fit of each LAB run against A3 on the fresh label episodes
    fit, fits_resolved, needs_seed43, runs = {}, {}, [], {}
    for run in LABEL_RUNS:
        if state[run] == "failed":
            fit[run] = fits_resolved[run] = "no_fit"
            runs[run] = {"status": "failed", "failed_record": str(failed_path(run, SEED, smoke).relative_to(ROOT)),
                         "fit": {"reading": "no_fit"}, "resolved": "no_fit"}
            continue
        ck42 = ckpt_path(run, SEED, smoke)
        pa = term_fresh(ck42)
        for m in METRICS:
            arrays[f"fresh__{run}__{m}"] = np.asarray(pa[m])
        r = compare(pa, pa_a3, fclusters, "gain")
        reading = fit_reading(r)
        entry = {"vs_A3_gain": r, "reading": reading, "seed43": None}
        if reading == "inconclusive":
            ck43 = ckpt_path(run, 43, smoke)
            if failed_path(run, 43, smoke).exists():
                assert not ck43.exists(), f"{run} seed 43: both failed record and checkpoint"
                reading = "no_fit"
                entry["seed43"] = {"status": "failed", "reading": "no_fit"}
            elif ck43.exists():
                pa43 = term_fresh(ck43)
                avg = {m: 0.5 * (np.asarray(pa[m]) + np.asarray(pa43[m])) for m in METRICS}
                r2 = compare(avg, pa_a3, fclusters, "gain")
                reading = "fits" if r2["ci95"][0] > 0 else "no_fit"
                entry["seed43"] = {"checkpoint_sha256": rg.sha_file(ck43), "vs_A3_gain_two_seed_mean": r2,
                                   "reading": reading}
            else:
                needs_seed43.append(run)
        entry["resolved"] = reading
        fits_resolved[run] = reading
        entry["X_minus_C0"] = {m: compare(pa, pa_c0, fclusters, m) for m in ("r1", "gain")}
        hist = json.loads((res / f"history_{run}_seed{SEED}.json").read_text())
        last = np.asarray(hist["history"]["aspect_loss"][-10:], dtype=np.float64)
        const = 3.258 if hist["config"]["lambda_swap"] > 0 else 2.565
        entry["loss_vs_constant"] = {"last10_mean": float(last.mean()), "constant": const,
                                     "pct_below_constant": float(100 * (const - last.mean()) / const)}
        taus = hist["history"]["tau"]
        entry["tau"] = {"first": float(taus[0]), "last": float(taus[-1])}
        runs[run] = entry
        fit[run] = reading

    # ---- seed-42 transfer (term-only, nested) for each LAB run and MK3
    cl = ctx.anchor_group
    ta_pa, _ = term_transfer(a3_path)
    for run in RUNS_SCORED:
        if state[run] == "failed":
            continue
        pt, (ic, tc) = term_transfer(ckpt_path(run, SEED, smoke))
        inp = EvalInputs(ctx.img, ctx.txt, ic, tc)
        tu = agreement_term(inp, ctx.pooled, uniform=True)
        ta = agreement_term(inp, ctx.pooled)
        nested, _control, picks = crossfit_nested(ctx.cos, tu, ta, ctx.parity)
        pn = per_anchor(nested)
        for m in METRICS:
            arrays[f"transfer__{run}__termonly__{m}"] = np.asarray(pt[m])
            arrays[f"transfer__{run}__nested__{m}"] = np.asarray(pn[m])
        runs.setdefault(run, {})["transfer_seed42"] = {
            "term_only": ctx.summary(pt),
            "term_only_either": either_point(pt["r1"] + pt["other"], cl),
            "nested": ctx.summary(pn), "nested_gain_point": ctx.summary(pn)["gain"]["point"], "picks": picks}

    fitting = [r for r in LABEL_RUNS if fit[r] == "fits"]
    nested_gains = {r: runs[r]["transfer_seed42"]["nested_gain_point"] for r in fitting}
    best_run = max(nested_gains, key=nested_gains.get) if fitting else None
    best_gain = nested_gains[best_run] if best_run else None
    if state["MK3"] == "failed":
        mk3 = None
        matched_k = {"status": "failed", "failed_record": str(failed_path("MK3", SEED, smoke).relative_to(ROOT)), "granularity_lever": False}
    else:
        mk3 = compare(per_anchor_of(arrays, "MK3"), ta_pa, cl, "gain")
        matched_k = {"MK3_minus_A3_term_only_gain": mk3, "granularity_lever": bool(mk3["ci95"][0] > 0)}
    h3 = None if needs_seed43 else h3_reading(fit, best_gain, g_star)

    result.update(runs=runs, fit=fit, best_fitting_run=best_run, best_nested_gain=best_gain, g_star=g_star,
                  h3_reading=h3, matched_k=matched_k, needs_seed43=needs_seed43)
    rg.assert_finite_tree(result)
    out_json.write_text(json.dumps(result, indent=1))
    np.savez_compressed(res / "per_anchor_h3.npz", **arrays)

    lines = [f"H3 diagnostic{' (SMOKE)' if smoke else ''}: {result['fresh']['n_total']} fresh label episodes, "
             f"{ctx.n} seed-42 episodes", f"{'run':4s} {'fit vs A3 (gain pp)':>28s} {'reading':>12s} "
             f"{'nested gain':>11s} {'loss %<const':>12s} {'tau first/last':>16s}"]
    for run in LABEL_RUNS:
        e = runs[run]
        if e.get("status") == "failed":
            lines.append(f"{run:4s} FAILED (counts as no_fit)")
            continue
        c = e["vs_A3_gain"]
        lines.append(f"{run:4s} {c['point']:8.2f} [{c['ci95'][0]:6.2f},{c['ci95'][1]:6.2f}] {e['resolved']:>12s} "
                     f"{e['transfer_seed42']['nested_gain_point']:11.2f} {e['loss_vs_constant']['pct_below_constant']:12.2f} "
                     f"{e['tau']['first']:8.4f}/{e['tau']['last']:.4f}")
    lines.append(f"best fitting run {best_run}, best nested gain {best_gain}, g*={g_star:.3f}, h3_reading={h3}, "
                 f"needs_seed43={needs_seed43}")
    if mk3 is None:
        lines.append("matched-k: MK3 FAILED, granularity_lever=False")
    else:
        lines.append(f"matched-k MK3 - A3 term-only gain {mk3['point']:.2f} [{mk3['ci95'][0]:.2f}, {mk3['ci95'][1]:.2f}], "
                     f"granularity_lever={matched_k['granularity_lever']}")
    text = "\n".join(lines)
    (res / "h3.txt").write_text(text + "\n")
    print(text)
    print(f"runtime {time.time() - t0:.0f}s")
    if needs_seed43:
        print(f"needs seed-43 checkpoint for {needs_seed43}; exit 3")
        sys.exit(3)


def per_anchor_of(arrays, run):
    return {m: arrays[f"transfer__{run}__termonly__{m}"] for m in METRICS}


if __name__ == "__main__":
    main()
