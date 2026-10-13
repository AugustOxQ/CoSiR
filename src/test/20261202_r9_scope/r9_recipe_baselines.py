"""Recipe IV (ticket 10; rule §2, §3.2 items 1 and 3, §11; constitution C1): RCA and PM fits, their lambda picks, and the
picks of B, B'(A0) and B'(A1), for one backbone.

    python r9_recipe_baselines.py --backbone B [--out DIR] [--partitions-dir DIR] [--a3-dir DIR]

What it runs, on seed 42 (selection rows, 4,096 episodes per pair), in the order of the original scripts:
1. The heads of the backbone, refit by `r9_heads.fit_heads_r9` (CLIP: the pinned inputs, then `check_selection` against
   the stored posteriors; another backbone: ticket 07's partitions, then the coefficient hashes against the ones
   `n6_posteriors.npz.record.json` holds). The PCA basis and pair scaler: `r9_bundle.fit_pm`, on the frozen 60,000-row
   draw (`fit_rows_sha256` 0a0d3b77..., asserted against CLIP's `baselines_seed42.json`, for every backbone).
2. The seed-42 episodes (`RowContext`; they need labels, groups and rows, no features) and the terms of one light bundle
   (`build_baseline_bundle`): cosine, T_N1u (A3 codes), the T_6u of B, B0 and B1, and RCA and the nine PM terms
   (`r9_bundle.pm_term_scores`, run_baselines.py's calls). It is `build_bundle_r6` without the reader features, so it
   does not need ticket 09's readers. The episode hashes are asserted equal to CLIP's `baselines_seed42.json`.
3. RCA's and each PM scorer's lambda by `crossfit_lambda(cos, term, parity)` (run_baselines.py), written as
   `baselines_seed42.json`, the keys the chain reads: `fit_rows`, `fit_rows_sha256`, `lambda_picks`,
   `episodes_sha256`, in `run_baselines.py`'s layout (`scorers[name].lambda_picks`, the string "inf" for an infinite
   lambda, tune-half keys "0" and "1"). Only RCA and the nine PM scorers are written: SE, C0, R3 and SE_uniform of
   run_baselines.py need CLIP's factor codes and no part of the chain reads them.
4. B, B0, B1 by `crossfit_condition_free` (`r9_picks.crossfit_seed42`, run_r6_picks.py), written as `picks_seed42.json`.

Checks, each recorded in `picks_seed42.json` (a failure writes `picks_seed42_failed.json` only, exit 3, nothing else):
  frozen_equals_crossfit  the frozen assembly with these picks reproduces the cross-fit bit for bit
  lambda_convention       `crossfit_lambda` rerun gives the lambdas just written, read back through the chain's parser
  CLIP only:   mean_r1_equal_target (18.341064453125, 18.436686197916664, 18.804931640625, exact) and
               lambda_equals_pinned (the ten lambda picks of CLIP's `baselines_seed42.json`, key for key)
A new backbone's targets are its own; its `picks_seed42.json` has no `mean_r1_equal_target` and `targets: null`.

Inputs. `--backbone clip_b32` without the two options uses round 6's pinned files (every input SHA-256 asserted).
Any other case reads the backbone's parts (`r9_backbone.parts`: `results/recipe/<B>/`) and records their SHA-256s
without a pin: stage 2's `pins_<B>.json` does not exist yet and lists these two outputs itself. `--partitions-dir`
(holds `partitions.npz` and `n6_posteriors.npz.record.json`) and `--a3-dir` (holds `A3_seed42.pt` and `picked.json`)
point the CLIP replay of stage 0b at its own parts; CLIP's targets then still apply.

Outputs in `--out` (default `results/recipe/<B>/`): `baselines_seed42.json` and `picks_seed42.json`, each with
`<file>.record.json` (SHA-256 of the feature-store files and every upstream part, of this script and the modules it
imports, library versions, the keys made and the keys copied). Nothing existing is overwritten.
CPU only, 8 threads. CLIP takes about 10 minutes (heads 2 min, bundle and PM terms 5 min, picks 1 min).
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first: MAIN and src)
import r9_bundle as B  # noqa: E402
import r9_context as X  # noqa: E402
import r9_heads as H  # noqa: E402
import r9_picks as P  # noqa: E402
from r9_backbone import BACKBONES, parts  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_quick_checks import _require_condition_free, centered_term, uniform_probe_scores  # noqa: E402
from src.eval.aspect_scorers import EvalInputs, crossfit_lambda  # noqa: E402

CLIP = "clip_b32"
EXIT_DIFF = 3
FILES = ("baselines_seed42.json", "picks_seed42.json")
FAILED = "picks_seed42_failed.json"
CODE_MODULES = ("r9_recipe_baselines", "r9_bundle", "r9_picks", "r9_context", "r9_heads", "r9_episodes", "r9_data",
                "r9_common", "r9_score", "src.eval.aspect_quick_checks", "src.eval.aspect_scorers",
                "src.eval.aspect_nested", "src.eval.pair_metric_baselines")
BASELINES_MADE = ["scorers", "lambda_picks", "convention"]
BASELINES_COPIED = ["episodes_seed", "n_per_pair", "pair_order", "episodes_sha256", "fit_rows_sha256", "fit_rows"]
T0 = time.monotonic()


def log(*a):
    print(f"[{time.monotonic() - T0:7.1f}s]", *a, flush=True)


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


# ---------------------------------------------------------------- the light bundle

def build_baseline_bundle(ctx, pm, a3) -> SimpleNamespace:
    """The seed's terms that RCA, the PM scorers, B, B0 and B1 need: `build_bundle_r6` without stack, F and readers.
    ``a3`` = EvalInputs with A3's codes of ``ctx.rows`` (NaN elsewhere). Same calls, same order, same bits."""
    t0 = time.time()
    B.check_context(ctx)
    H.require_threads()
    n, ep = int(ctx.n), ctx.pooled
    _require(isinstance(a3, EvalInputs) and a3.img_codes is not None and a3.txt_codes is not None
             and a3.img_codes.shape[0] == a3.txt_codes.shape[0] == R.N_ROWS
             and bool(np.isfinite(a3.img_codes[ctx.rows]).all() and np.isfinite(a3.txt_codes[ctx.rows]).all()),
             "A3 inputs: codes of every row of the context are required")
    t_n1u = centered_term(a3, ep, uniform=True)
    post_e2 = {part: ctx.post[B.E2_FROM[part]] for part in B.E2_PARTS}
    t6u_B = uniform_probe_scores(post_e2, ep, B.E2_PARTS)
    post = {h: ctx.post[h] for h in B.A0}
    for h in B.A0:
        for side in ("img", "txt"):
            _require(bool(np.isfinite(post[h][side][ctx.rows]).all()), f"{h}/{side}: non-finite posteriors on rows")
    t6u_B0 = uniform_probe_scores(post, ep, B.A0)
    t6u_B1 = uniform_probe_scores({**post, "csd": ctx.post["csd"]}, ep, B.A1)
    pm_terms = B.pm_term_scores(ctx, pm)
    b = SimpleNamespace(
        n=n, parity=np.asarray(ctx.parity), cos=ctx.cos, t_n1u=t_n1u, t6u_B=t6u_B, t6u_B0=t6u_B0, t6u_B1=t6u_B1,
        pm_terms=pm_terms, episodes_sha256=dict(ctx.episode_sha), coef_sha256=ctx.coef_sha256,
        fit_rows_sha256=pm.fit_rows_sha256, mode=ctx.mode, seed=int(ctx.seed), smoke=bool(ctx.smoke))
    for k in ("cos", "t_n1u", "t6u_B", "t6u_B0", "t6u_B1"):
        B.check_scores(getattr(b, k), n, k, np.float32)
        _require_condition_free(getattr(b, k), k)
    _require(tuple(pm_terms) == B.PM_TERM_NAMES, f"pm_terms must be {B.PM_TERM_NAMES}")
    log(f"light bundle done [{time.time() - t0:.0f}s]")
    return b


# ---------------------------------------------------------------- lambda picks and the baselines record

def crossfit_lambdas(bundle) -> dict:
    """{scorer: {0: lam, 1: lam}} for RCA and the nine PM scorers: run_baselines.py's `crossfit_lambda(cos, term,
    parity)` (the pick of tune half h was tuned on parity h and scores parity 1 - h)."""
    P.check_seed42(bundle)
    return {name: P.as_lambda_pick(crossfit_lambda(bundle.cos, bundle.pm_terms[name], bundle.parity)[1], name)
            for name in P.LAMBDA_SCORERS}


def lambda_json(lam) -> dict:
    """run_baselines.py's format: tune-half keys "0" and "1", the string "inf" for an infinite lambda."""
    return {str(h): ("inf" if np.isinf(v) else float(v)) for h, v in sorted(lam.items())}


def clip_baselines() -> dict:
    """CLIP's baselines_seed42.json (SHA-256 asserted): the frozen episode hashes, fit-row hash and lambdas."""
    R.assert_input(B.BASELINES42_REL, CLIP)
    return json.loads(Path(R.input_pins(CLIP)[B.BASELINES42_REL][0]).read_text())


def baselines_record(bundle, lams) -> dict:
    """baselines_seed42.json's content. The seed-42 episodes and the 60,000 fit rows are CLIP's: both hashes are
    asserted against CLIP's file here, whatever the backbone."""
    clip = clip_baselines()
    _require(dict(bundle.episodes_sha256) == clip["episodes_sha256"],
             f"episodes_sha256 {dict(bundle.episodes_sha256)} differs from CLIP's baselines_seed42.json "
             f"{clip['episodes_sha256']}")
    _require(bundle.fit_rows_sha256 == clip["fit_rows_sha256"] and clip["fit_rows"] == B.FIT_ROWS,
             f"fit_rows_sha256 {bundle.fit_rows_sha256} differs from CLIP's baselines_seed42.json "
             f"{clip['fit_rows_sha256']}")
    _require(tuple(lams) == P.LAMBDA_SCORERS, f"lambdas for {tuple(lams)}, not {P.LAMBDA_SCORERS}")
    picks = {name: lambda_json(lams[name]) for name in P.LAMBDA_SCORERS}
    return {"episodes_seed": R.DEV_SEED, "n_per_pair": R.N_PER_PAIR, "pair_order": list(R.PAIR_NAMES),
            "episodes_sha256": dict(bundle.episodes_sha256), "fit_rows_sha256": bundle.fit_rows_sha256,
            "fit_rows": B.FIT_ROWS, "lambda_picks": picks,
            "convention": "the pick of tune half h was tuned on the episodes of parity h and scores parity 1 - h",
            "scorers": {name: {"lambda_picks": picks[name]} for name in P.LAMBDA_SCORERS}}


# ---------------------------------------------------------------- records, refusal, the outputs

def _paths(out):
    out = Path(out)
    return [out / name for f in FILES for name in (f, f + ".record.json")] + [out / FAILED]


def preflight(out, backbone):
    assert backbone in BACKBONES, f"unknown backbone: {backbone}"
    present = [str(p) for p in _paths(out) if p.exists()]
    if present:
        raise FileExistsError(f"refusing to overwrite {present[0]}. There is no resume: to rerun in {out}, remove "
                              f"these files first: {' '.join(present)}")


def code_record() -> dict:
    import importlib
    import scipy
    import sklearn
    import torch
    modules = {}
    for name in CODE_MODULES:
        mod = sys.modules[__name__] if name == "r9_recipe_baselines" else importlib.import_module(name)
        modules[name] = {"file": str(Path(mod.__file__).resolve()), "sha256": R.sha256_file(mod.__file__)}
    return {"modules": modules, "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                                             "sklearn": sklearn.__version__, "torch": torch.__version__}}


def _write_json(path, content):
    with Path(path).open("x") as file:
        json.dump(content, file, indent=1)
        file.write("\n")


def _write_with_record(out, name, content, backbone, input_sha256, made, copied):
    written = set(content)
    _require(written == set(made) | set(copied), f"{name}: keys made+copied differ from the keys written")
    _write_json(Path(out) / name, content)
    rec = {"backbone": backbone, "input_sha256": dict(input_sha256), "script": str(Path(__file__)),
           "script_sha256": R.sha256_file(__file__), "code": code_record(), "keys_made": list(made),
           "keys_copied": list(copied)}
    _write_json(Path(out) / (name + ".record.json"), rec)


def finish(bundle, out, backbone, input_sha256) -> int:
    """Lambda picks, cross-fit picks, the checks; write the two files (exit 0) or only picks_seed42_failed.json
    (exit 3). ``input_sha256`` goes into both records."""
    out = Path(out)
    preflight(out, backbone)
    P.check_seed42(bundle)
    lams = crossfit_lambdas(bundle)
    base = baselines_record(bundle, lams)
    parsed = P.parse_lambda_picks(json.loads(json.dumps(base)))             # what the chain will read back
    picks, scores = P.crossfit_seed42(bundle)
    checks = {"frozen_equals_crossfit": P.frozen_equals_crossfit(bundle, picks, scores),
              "lambda_convention": P.lambda_convention(bundle, parsed)}
    if backbone == CLIP:
        pinned = P.frozen_lambdas(CLIP)
        checks["lambda_equals_pinned"] = {name: parsed[name] == pinned[name] for name in P.LAMBDA_SCORERS}
    rec = P.picks_record(picks, checks)
    if backbone != CLIP:                                                     # the targets are CLIP's
        del rec["checks"]["mean_r1_equal_target"]
        rec["targets"] = None
        rec["passed"] = all(bool(v) for chk in rec["checks"].values() for v in chk.values())
    rec.update({"backbone": backbone, "lambda_picks": base["lambda_picks"],
                "episodes_sha256": dict(bundle.episodes_sha256), "coef_sha256": bundle.coef_sha256,
                "fit_rows_sha256": bundle.fit_rows_sha256, "input_sha256": dict(input_sha256)})
    out.mkdir(parents=True, exist_ok=True)
    for chk, items in rec["checks"].items():
        for name, ok in items.items():
            print(f"{chk} {name}: {'pass' if ok else 'FAIL'}", flush=True)
    if not rec["passed"]:
        _write_json(out / FAILED, rec)
        print(f"picks seed 42: FAILED; record {out / FAILED}; stage stop (rule §3.2 item 4): exit {EXIT_DIFF}")
        return EXIT_DIFF
    picks_made = ["B", "B0", "B1", "mean_r1", "checks", "passed", "rule_sha256", "module_sha256", "time", "backbone",
                  "lambda_picks", "coef_sha256", "input_sha256"]
    picks_copied = [k for k in rec if k not in picks_made]                   # convention, targets, the hashes of CLIP
    _write_with_record(out, FILES[0], base, backbone, input_sha256, BASELINES_MADE, BASELINES_COPIED)
    _write_with_record(out, FILES[1], rec, backbone, input_sha256, picks_made, picks_copied)
    print(f"picks seed 42: PASSED; written {out / FILES[0]} and {out / FILES[1]}")
    return 0


# ---------------------------------------------------------------- the inputs of one run

def a3_inputs(ctx, backbone, a3_dir=None) -> EvalInputs:
    """A3's codes on ctx.rows. CLIP without ``a3_dir``: the pinned checkpoint (`r9_bundle.model_inputs`). Otherwise
    the checkpoint and picked.json of ``a3_dir`` (default: the backbone's parts), tied by SHA-256, no pin."""
    if backbone == CLIP and a3_dir is None:
        return B.model_inputs(ctx, backbone)
    ckpt, picked_path = ((Path(a3_dir) / "A3_seed42.pt", Path(a3_dir) / "picked.json") if a3_dir is not None
                         else (parts(backbone)["A3_seed42"], parts(backbone)["picked"]))
    picked = json.loads(Path(picked_path).read_text())
    _require(picked["run"] == "A3" and picked["checkpoint_sha256"] == R.sha256_file(ckpt)
             and picked.get("backbone", backbone) == backbone and picked.get("smoke", False) is False,
             f"{picked_path} does not describe {ckpt} (run A3, its SHA-256, this backbone)")
    ic, tc = ctx.encode(ckpt)
    return EvalInputs(ctx.img, ctx.txt, ic, tc)


def head_fit(rows, backbone, partitions_dir):
    """-> (heads, the SHA-256s of the head inputs, check record). CLIP, pinned: fit_heads_r9 and check_selection."""
    split = rows.split
    if backbone == CLIP and partitions_dir is None:
        heads = H.fit_heads_r9(rows.data, split.groups, split.scorer_train, backbone=backbone)
        check = H.check_selection(heads, rows.data, split.selection, backbone)
        _require(check["passed"], "the refit heads differ from the stored posteriors (rule §6.2)", SystemExit)
        return heads, dict(heads["affect"]["fit_inputs"]["input_sha256"]), {"selection_check": "passed"}
    pins = H.recipe_head_pins(backbone, partitions_dir)
    heads = H.fit_heads_r9(rows.data, split.groups, split.scorer_train, backbone=backbone, input_pins=pins)
    root = Path(partitions_dir) if partitions_dir is not None else Path(parts(backbone)["partitions"]).parent
    record = root / (Path(H.N6_REL).name + ".record.json")
    stored = json.loads(record.read_text())
    _require(stored["backbone"] == backbone and stored["coef_sha256"] == H.coef_sha256(heads),
             f"the refit heads differ from the ones {record} holds (coefficient hashes)", SystemExit)
    shas = dict(heads["affect"]["fit_inputs"]["input_sha256"])
    shas[str(record)] = R.sha256_file(record)
    return heads, shas, {"coef_sha256_equals_record": str(record)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backbone", choices=BACKBONES, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default results/recipe/<backbone>/")
    ap.add_argument("--partitions-dir", type=Path, default=None, help="folder of partitions.npz (CLIP replay)")
    ap.add_argument("--a3-dir", type=Path, default=None, help="folder of A3_seed42.pt and picked.json (CLIP replay)")
    args = ap.parse_args(argv)
    out = args.out or R.RESULTS / "recipe" / args.backbone
    H.require_threads()
    preflight(out, args.backbone)
    pinned = args.backbone == CLIP and args.partitions_dir is None and args.a3_dir is None
    rows = R.load_rows(args.backbone)
    log(f"{args.backbone}: data and split loaded")
    heads, head_shas, _ = head_fit(rows, args.backbone, args.partitions_dir)
    log("heads fitted and checked")
    pm = B.fit_pm(rows.data, rows.split.scorer_train, args.backbone)
    log("PM fits done (fit rows SHA-256 asserted)")
    ctx = X.RowContext("selection", R.DEV_SEED, rows.data, rows.split, rows.labels, heads, rows.value_sets,
                       R.N_PER_PAIR)
    a3 = a3_inputs(ctx, args.backbone, args.a3_dir)
    bundle = build_baseline_bundle(ctx, pm, a3)
    from r9_recipe_groups import read_shas
    inputs = {**read_shas(args.backbone), **head_shas}
    if pinned:
        inputs.update(R.assert_inputs(args.backbone))
    else:
        a3_files = ([args.a3_dir / "A3_seed42.pt", args.a3_dir / "picked.json"] if args.a3_dir is not None
                    else [parts(args.backbone)["A3_seed42"], parts(args.backbone)["picked"]])
        inputs.update({str(p): R.sha256_file(p) for p in a3_files})
        inputs[str(R.input_pins(CLIP)[B.BASELINES42_REL][0])] = R.assert_input(B.BASELINES42_REL, CLIP)
        for rel in (R.PREPARE_REL, R.HELD_CODES_REL):
            inputs[str(R.INPUT_PATHS[rel])] = R.assert_input(rel)
    code = finish(bundle, out, args.backbone, inputs)
    log("done")
    return code


if __name__ == "__main__":
    sys.exit(main())
