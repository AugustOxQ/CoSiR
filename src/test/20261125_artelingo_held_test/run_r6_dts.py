"""Round 6 describe-then-score on seed 42, CPU stages (DECISION_RULE.md of this folder: section 7 items 4 to 7, section
6 item 6; contracts section 9; ticket 11). Selection rows only; the GPU jobs (r6_gpu_verbalise.py, r6_gpu_listing.py)
run between the stages, launched by the main session. Each stage reads the GPU outputs joined by (seed, episode_index,
condition, wording) and (phrase, K), every needed key present exactly once after merging the given folders (r6_dts).

Order (rule section 7 item 4):
  1. --stage list-input --for sanity --job-out <job>        the listing job of DTS-N: the three aspect names x K 8, 16
     (listing job)
  2. --stage sanity --listing-out <dir>...                  DTS-N at K 8 and 16 on all 12,288 episodes; its pooled
     condition gain (cross-fitted fused) must be above 0 at each K, else exit 3 (the pipeline is debugged)
     (verbaliser job on the tuning subset: r6_gpu_inputs.py --first-per-pair 1024, wordings W1 to W4)
  3. --stage list-input --for tune --verbalise-out <dir>... --verbalise-job <job>... --job-out <job>
     (listing job)
  4. --stage tune --verbalise-out ... --verbalise-job ... --listing-out ...
     the 8 settings W1..W4 x K 8, 16 on the first 1,024 episodes of each pair, scored by (R@1 + gain) / 2 of the
     cross-fitted fused scores on those 3,072 (exact integers); ties to the first in W1 K8, W1 K16, W2 K8, ...
     (verbaliser job on all 12,288 episodes with the chosen wording)
  5. --stage list-input --for chosen ... --job-out <job>    the chosen wording's phrases and the names at the chosen K
     (listing job)
  6. --stage chosen --verbalise-out ... --verbalise-job ... --listing-out ...
     the chosen setting on all 12,288: its lambda picks per half, DTS-CF's and DTS-N's at the chosen K, the
     parsing-failure counts and every setting's score -> results/dts_seed42.json (+ dts_seed42_per_anchor.npz)
  7. --stage stop --clock-start "2026-10-09 12:29"
     hits = sum of int64 as_int4(R@1) of DTS's cross-fitted fused scores over the 12,288; the build stops iff hits
     > 9,406 (AFF's); DTS must have been built (sanity passed, chosen run) within 24 hours of the clock start (the first
     DTS commit, Amsterdam time, given here, never parsed from the log) -> results/dts_stop.json. Exit 0: no stop;
     exit 3: stop (the user decides).

Held phrases (after the held runner wrote results/held_episodes_seed<s>.npz and the verbaliser ran on them, rule section
8 item 3; no metric): --stage list-input --for held --seed 52|53|54 --episodes <that file> --verbalise-out ...
--verbalise-job ... --job-out <job> lists the chosen wording's phrases and the names at the chosen K (dts_seed42.json).

Common arguments: --episodes <the seed's episodes file, r6_episodes.save_episodes> (its per-pair SHA-256s asserted
equal to the recorded ones), --seed (42; the smoke seeds 9001 to 9003 at 64 per pair, tuning subset 16 per pair, are
for tests and the smoke chain), --out (default results/, smoke results/smoke/dts/), --embeddings (the value-embedding
cache, default <out>/dts_value_embeddings.npz). Outputs are never overwritten. Every output records module_sha256,
the settings SHA-256, the input file SHA-256s, the GPU outputs' fingerprints and the time. Prints pass or fail and
file paths only, never a metric.

    cd /project/CoSiR && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/run_r6_dts.py --stage <stage> ...
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_dts as D  # noqa: E402
import r6_episodes as E  # noqa: E402
import r6_gpu_common as G  # noqa: E402

import numpy as np  # noqa: E402

STAGES = ("list-input", "sanity", "tune", "chosen", "stop")
EXIT_FAIL = 3
EXIT_REFUSED = 2
ADMITTED = {R.DEV_SEED: R.N_PER_PAIR, **{s: R.N_SMOKE for s in R.SMOKE_SEEDS}}
LIST_FOR = ("sanity", "tune", "chosen", "held")
SANITY, TUNE, STOP = "dts_sanity.json", "dts_tune.json", "dts_stop.json"
EMBEDDINGS = "dts_value_embeddings.npz"
LISTING_INPUT, JOB_RECORD = "listing_input.jsonl", "job_record.json"
# the files whose bytes the DTS stages ran; the stop refuses stage records made by other bytes
DTS_FILES = tuple(f"src/test/20261125_artelingo_held_test/{f}" for f in
                  ("r6_dts.py", "run_r6_dts.py", "r6_common.py", "r6_episodes.py", "r6_gpu_common.py",
                   "r6_gpu_inputs.py", "dts_settings.json"))


class Refused(Exception):
    pass


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def chosen_name(seed) -> str:
    return f"dts_seed{int(seed)}.json"


def per_anchor_name(seed) -> str:
    return f"dts_seed{int(seed)}_per_anchor.npz"


def tune_per_pair(seed) -> int:
    return D.TUNE_PER_PAIR if int(seed) == R.DEV_SEED else R.N_SMOKE // 4


def default_out(seed) -> Path:
    return R.SMOKE / "dts" if int(seed) in R.SMOKE_SEEDS else R.RESULTS


# ---------------------------------------------------------------- inputs

def load_episodes_checked(seed, path):
    """The seed's episodes from save_episodes' file, each pair's SHA-256 equal to the recorded one (AB's
    baselines_seed42.json, or its smoke files)."""
    _require(path is not None, "--episodes is required for this stage", Refused)
    eps = E.load_episodes(path)
    want = E.identity_targets()[int(seed)]
    _require(eps.seed == int(seed) and eps.n == len(R.PAIRS) * want["n_per_pair"]
             and eps.sha == want["episodes_sha256"],
             f"{path}: not the recorded episodes of seed {seed} ({want['n_per_pair']} per pair)")  # guard:episodes_identity
    return eps


def load_held_episodes(seed, path):
    """A held seed's episodes from the held runner's file (each pair's hash checked by load_episodes): the listing
    input of the held phrases reads them only to join the verbaliser outputs; no feature or label is loaded."""
    _require(path is not None, "--episodes is required for this stage", Refused)
    eps = E.load_episodes(path)
    _require(eps.seed == int(seed) and eps.n == len(R.PAIRS) * R.N_PER_PAIR,
             f"{path}: not {len(R.PAIRS)} x {R.N_PER_PAIR} episodes of held seed {seed}")
    return eps


def load_context(seed, episodes_path) -> SimpleNamespace:
    """Selection-row context of the seed: episodes (checked), CLIP features NaN outside the selection rows,
    cosine_scores, parity, pair index, anchor groups."""
    from src.data.artelingo import load_artelingo
    from src.eval.aspect_scorers import EvalInputs, cosine_scores
    eps = load_episodes_checked(seed, episodes_path)
    data = load_artelingo()
    split = R.load_split(data)
    rows = split.selection
    inside = np.zeros(R.N_ROWS, dtype=bool)
    inside[rows] = True
    feats = {}
    for m in ("img", "txt"):
        x = np.full(getattr(data, f"{m}_features").shape, np.nan, dtype=np.float32)
        x[rows] = getattr(data, f"{m}_features")[rows]
        _require(bool(np.isnan(x[~inside]).all() and np.isfinite(x[inside]).all()), f"{m}: masking failed")
        feats[m] = x
    del data
    member = np.asarray(eps.pooled.rows())
    _require(bool(inside[member].all()), "an episode member row lies outside the selection rows")
    cos = cosine_scores(EvalInputs(feats["img"], feats["txt"]), eps.pooled)
    anchor = np.asarray(eps.pooled.anchor, dtype=np.int64)
    return SimpleNamespace(mode="selection", seed=int(seed), n=int(eps.n), n_total=int(eps.n), eps=eps,
                           pooled=eps.pooled, parity=np.asarray(eps.parity), pair_index=np.asarray(eps.pair_index),
                           img=feats["img"], txt=feats["txt"], cos=cos, anchor_group=split.groups[anchor],
                           episodes_file_sha256=G.sha256_file(episodes_path))


def gpu_inputs(args, settings_sha, eps, wordings, need_verbaliser=True):
    """(merged verbaliser outputs, job SHA-256s, merged listings) of the folders given (None where not needed)."""
    ver, jobs = None, {}
    if need_verbaliser:
        _require(args.verbalise_out and args.verbalise_job, "--verbalise-out and --verbalise-job are required",
                 Refused)
        ver = D.merge_verbaliser(args.verbalise_out, wordings, settings_sha, seeds={int(args.seed)})
        jobs = D.check_verbaliser_jobs(ver, args.verbalise_job, eps)
    return ver, jobs


def listings_of(args, settings_sha):
    _require(args.listing_out, "--listing-out is required", Refused)
    return D.merge_listings(args.listing_out, settings_sha)


# ---------------------------------------------------------------- records

def read_record(out, name, seed):
    path = Path(out) / name
    if not path.is_file():
        return None
    rec = json.loads(path.read_text())
    _require(rec.get("seed") == int(seed), f"{path}: a record of seed {rec.get('seed')}, not {seed}")
    return rec


def base_record(args, stage, settings_sha, inputs, extra=None) -> dict:
    rec = {"stage": stage, "seed": int(args.seed), "rule_sha256": R.RULE_SHA256, "settings_sha256": settings_sha,
           "module_sha256": R.r6_module_shas(), "input_sha256": inputs, "time": R.amsterdam_now()}
    rec.update(extra or {})
    return rec


def write_new(path, obj):
    """Write JSON to a path that must not exist (outputs are never overwritten)."""
    path = Path(path)
    _require(not path.exists(), f"{path} exists; outputs are never overwritten (move it aside to "
                                f"rerun)", Refused)  # guard:no_overwrite
    G.write_json(path, obj)
    return path


def input_shas(ctx_or_eps_sha, ver, jobs, lis, emb_sha=None) -> dict:
    out = {"episodes_file": ctx_or_eps_sha}
    if ver is not None:
        out["verbaliser"] = ver.files
        out["verbaliser_jobs"] = jobs
    if lis is not None:
        out["listings"] = lis.files
    if emb_sha is not None:
        out["value_embeddings"] = emb_sha
    return out


def fingerprints(ver, lis) -> dict:
    out = {}
    if ver is not None:
        out["verbaliser"] = {Path(d).name: fp for d, fp in ver.fingerprints.items()}
    if lis is not None:
        out["listing"] = {Path(d).name: fp for d, fp in lis.fingerprints.items()}
    return out


def failures_json(fail) -> dict:
    return {c: dict(fail[c]) for c in D.CONDITIONS}


def require_sanity(args):
    rec = read_record(args.out, SANITY, args.seed)
    _require(rec is not None and rec["passed"] is True,
             f"{Path(args.out) / SANITY}: DTS-N's sanity has not passed; it runs first (rule section 7 item "
             f"4)", Refused)  # guard:sanity_first
    return rec


# ---------------------------------------------------------------- stages

def stage_list_input(args, settings, settings_sha) -> int:
    """The listing job folder: listing_input.jsonl ({"phrase", "K"} per distinct pair) and job_record.json."""
    import r6_gpu_listing as L
    _require(args.job_out is not None and args.for_stage is not None, "--for and --job-out are required", Refused)
    job = Path(args.job_out)
    _require(not job.exists(), f"{job} exists; job folders are never overwritten", Refused)
    ver, jobs, eps_sha = None, {}, None
    names = list(D.TARGET_NAME.values())
    if args.for_stage == "sanity":
        items = D.listing_items(names, D.KS)
        wordings, Ks = [], list(D.KS)
    else:
        if args.for_stage == "held":
            eps = load_held_episodes(args.seed, args.episodes)
        else:
            eps = load_episodes_checked(args.seed, args.episodes)
        eps_sha = G.sha256_file(args.episodes)
        if args.for_stage == "held":
            chosen = read_record(args.out, chosen_name(R.DEV_SEED), R.DEV_SEED)
            _require(chosen is not None, f"{Path(args.out) / chosen_name(R.DEV_SEED)} is missing", Refused)
            pos = np.arange(eps.n, dtype=np.int64)
            wordings, Ks, extra = [chosen["wording_id"]], [int(chosen["K"])], names
        elif args.for_stage == "tune":
            import r6_gpu_inputs as I
            pos = I.select_episodes(eps, tune_per_pair(args.seed))
            wordings, Ks, extra = list(D.WORDINGS), list(D.KS), []
        else:
            tune = read_record(args.out, TUNE, args.seed)
            _require(tune is not None, f"{Path(args.out) / TUNE} is missing; tune first", Refused)
            pos = np.arange(eps.n, dtype=np.int64)
            wordings, Ks, extra = [tune["chosen"]["wording_id"]], [int(tune["chosen"]["K"])], names
        ver, jobs = gpu_inputs(args, settings_sha, eps, wordings)
        phrases = list(extra)
        for w in wordings:
            raw = D.select_answers(ver.answers, int(args.seed), w, pos, eps.n)
            phrases += [D.phrase_of(a) for c in D.CONDITIONS for a in raw[c]]
        items = D.listing_items(phrases, Ks)
    partial = job.with_name(job.name + ".partial")
    _require(not partial.exists(), f"{partial} exists (an interrupted write); remove it first", Refused)
    partial.mkdir(parents=True)
    with open(partial / LISTING_INPUT, "w", encoding="utf-8") as f:
        for p, K in items:
            f.write(json.dumps({"phrase": p, "K": int(K)}, ensure_ascii=True) + "\n")
    back = L.load_listing_input(partial / LISTING_INPUT, settings)
    _require(back == items, "the listing input does not read back as written")
    rec = base_record(args, "list-input", settings_sha, input_shas(eps_sha, ver, jobs, None),
                      {"for": args.for_stage, "wordings": wordings, "K": Ks, "n_items": len(items),
                       "listing_input_sha256": G.sha256_file(partial / LISTING_INPUT),
                       "gpu_fingerprints": fingerprints(ver, None)})
    G.write_json(partial / JOB_RECORD, rec)
    os.replace(partial, job)
    print(f"dts list-input ({args.for_stage}): pass; listing job folder {job}", flush=True)
    return 0


def stage_sanity(args, ctx, settings, settings_sha) -> int:
    out = Path(args.out) / SANITY
    _require(not out.exists(), f"{out} exists; outputs are never overwritten", Refused)  # guard:no_overwrite
    lis = listings_of(args, settings_sha)
    emb = D.ValueEmbedder(args.embeddings)
    per_K = {}
    for K in D.KS:
        r = D.run_names(ctx, K, lis.answers, emb)
        per_K[str(K)] = {"passed": bool(r.gain_int4 > 0), "gain_int4_sum": r.gain_int4,
                         "gain_point": 100 * float(r.s.pa["gain"].mean()), "r1_point": 100 * float(r.s.pa["r1"].mean()),
                         "picks": D.picks_to_json(r.s.picks), "failures": failures_json(r.fail),
                         "below_K": r.below_K}
    passed = all(v["passed"] for v in per_K.values())
    emb_sha = emb.save()
    rec = base_record(args, "sanity", settings_sha, input_shas(ctx.episodes_file_sha256, None, {}, lis, emb_sha),
                      {"passed": passed, "per_K": per_K, "n_episodes": ctx.n, "embedder": emb.meta,
                       "gpu_fingerprints": fingerprints(None, lis)})
    write_new(out, rec)
    print(f"dts sanity (DTS-N, K 8 and 16): {'pass' if passed else 'FAIL'}; record {out}", flush=True)
    if not passed:
        print("rule section 7 item 4: the pipeline is debugged within the budget; exit 3", flush=True)
        return EXIT_FAIL  # guard:sanity_exit
    return 0


def stage_tune(args, ctx, settings, settings_sha) -> int:
    import r6_gpu_inputs as I
    out = Path(args.out) / TUNE
    _require(not out.exists(), f"{out} exists; outputs are never overwritten", Refused)  # guard:no_overwrite
    require_sanity(args)
    pos = I.select_episodes(ctx.eps, tune_per_pair(args.seed))
    sub = D.subset_context(ctx, pos)
    sub.n_total = ctx.n
    _require(np.array_equal(sub.parity, np.arange(sub.n) % 2), "the tuning subset's parity is not alternating")
    ver, jobs = gpu_inputs(args, settings_sha, ctx.eps, D.WORDINGS)
    lis = listings_of(args, settings_sha)
    emb = D.ValueEmbedder(args.embeddings)
    scores, per = {}, {}
    for w, K in D.SETTINGS_ORDER:
        r = D.run_setting(sub, ver.answers, w, K, lis.answers, emb)
        scores[(w, K)] = r.score
        per[D.setting_name(w, K)] = {"wording_id": w, "K": K, "score_int": r.score,
                                     "score": 100 * r.score / (8 * sub.n),
                                     "r1_point": 100 * float(r.s.pa["r1"].mean()),
                                     "gain_point": 100 * float(r.s.pa["gain"].mean()),
                                     "picks": D.picks_to_json(r.s.picks), "failures": failures_json(r.fail),
                                     "below_K": r.below_K}
    w, K = D.choose_setting(scores)
    emb_sha = emb.save()
    rec = base_record(args, "tune", settings_sha, input_shas(ctx.episodes_file_sha256, ver, jobs, lis, emb_sha),
                      {"settings": per, "order": [D.setting_name(*s) for s in D.SETTINGS_ORDER],
                       "chosen": {"setting": D.setting_name(w, K), "wording_id": w, "K": K},
                       "n_episodes": sub.n, "per_pair": tune_per_pair(args.seed),
                       "positions_sha256": R.sha256_bytes(np.ascontiguousarray(pos, dtype=np.int64).tobytes()),
                       "score_definition": "score_int = sum 4 R@1 + sum 4 gain (int64); score = 100 score_int / (8 n)",
                       "embedder": emb.meta, "gpu_fingerprints": fingerprints(ver, lis)})
    write_new(out, rec)
    print(f"dts tune: pass, chose {D.setting_name(w, K)}; record {out}", flush=True)
    return 0


def stage_chosen(args, ctx, settings, settings_sha) -> int:
    out = Path(args.out) / chosen_name(args.seed)
    npz = Path(args.out) / per_anchor_name(args.seed)
    for p in (out, npz):
        _require(not p.exists(), f"{p} exists; outputs are never overwritten", Refused)  # guard:no_overwrite
    sanity = require_sanity(args)
    tune = read_record(args.out, TUNE, args.seed)
    _require(tune is not None, f"{Path(args.out) / TUNE} is missing; tune first", Refused)
    w, K = tune["chosen"]["wording_id"], int(tune["chosen"]["K"])
    _require((w, K) in D.SETTINGS_ORDER, f"chosen setting {tune['chosen']}")
    ver, jobs = gpu_inputs(args, settings_sha, ctx.eps, [w])
    lis = listings_of(args, settings_sha)
    emb = D.ValueEmbedder(args.embeddings)
    r = D.run_setting(ctx, ver.answers, w, K, lis.answers, emb)
    cf = D.score_cf(ctx.cos, r.term.T, ctx.parity)
    rn = D.run_names(ctx, K, lis.answers, emb)
    for s, term in ((r.s, r.term.T), (rn.s, rn.term.T)):
        fz = D.frozen_fused(ctx.cos, term, s.picks, ctx.parity)
        _require(all(np.array_equal(fz[c][d], s.scores[c][d]) for c in D.CONDITIONS for d in D.DIRECTIONS),
                 "frozen picks do not reproduce the cross-fitted scores")  # guard:frozen_convention
    _require(D.picks_to_json(rn.s.picks) == sanity["per_K"][str(K)]["picks"],
             "DTS-N's picks differ from the sanity stage's at the chosen K")
    arrays = {"pair_index": np.asarray(ctx.pair_index, dtype=np.int64),
              "parity": np.asarray(ctx.parity, dtype=np.int64),
              "anchor_group": np.asarray(ctx.anchor_group, dtype=np.int64)}
    for name, s in (("dts", r.s), ("dts_cf", cf), ("dts_n", rn.s)):
        for m in D.METRICS:
            arrays[f"{name}__{m}"] = np.asarray(s.pa[m], dtype=np.float64)
    npz.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(npz.stem + ".partial.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, npz)
    emb_sha = emb.save()
    summary = {name: {"r1_point": 100 * float(s.pa["r1"].mean()), "gain_point": 100 * float(s.pa["gain"].mean())}
               for name, s in (("dts", r.s), ("dts_cf", cf), ("dts_n", rn.s))}
    rec = base_record(args, "chosen", settings_sha, input_shas(ctx.episodes_file_sha256, ver, jobs, lis, emb_sha),
                      {"setting": D.setting_name(w, K), "wording_id": w,
                       "wording": settings["verbaliser"]["wordings"][w], "K": K,
                       "model": {"id": settings["model"]["id"], "snapshot": settings["model"]["snapshot"]},
                       "picks": {"dts": D.picks_to_json(r.s.picks), "dts_cf": D.picks_to_json(cf.picks),
                                 "dts_n": D.picks_to_json(rn.s.picks)},
                       "pick_convention": "key h = tuned on parity half h, scores parity 1 - h",
                       "failures": {"dts": failures_json(r.fail), "dts_n": failures_json(rn.fail)},
                       "below_K": {"dts": r.below_K, "dts_n": rn.below_K}, "n_values": r.term.n_values,
                       "settings_scores": tune["settings"], "sanity": {k: {"passed": v["passed"], "picks": v["picks"]}
                                                                     for k, v in sanity["per_K"].items()},
                       "n_episodes": ctx.n, "summary": summary, "per_anchor_file": npz.name,
                       "per_anchor_sha256": G.sha256_file(npz), "embedder": emb.meta,
                       "gpu_fingerprints": fingerprints(ver, lis)})
    write_new(out, rec)
    print(f"dts chosen ({D.setting_name(w, K)}): pass; record {out}, per-anchor arrays {npz}", flush=True)
    return 0


def dts_shas(rec) -> dict:
    return {f: rec["module_sha256"].get(f) for f in DTS_FILES}


def stage_stop(args) -> int:
    """Rule section 7 items 5 to 7."""
    _require(args.clock_start is not None, "--clock-start is required (the first DTS commit, Amsterdam time)", Refused)
    out = Path(args.out) / STOP
    _require(not out.exists(), f"{out} exists; outputs are never overwritten", Refused)  # guard:no_overwrite
    start = str(args.clock_start)
    D.parse_amsterdam(start)
    recs = {name: read_record(args.out, name, args.seed) for name in (SANITY, TUNE, chosen_name(args.seed))}
    sanity, chosen = recs[SANITY], recs[chosen_name(args.seed)]
    n_total = len(R.PAIRS) * ADMITTED[int(args.seed)]
    built = bool(sanity is not None and sanity["passed"] and chosen is not None and recs[TUNE] is not None
                 and chosen["n_episodes"] == n_total)
    now = R.amsterdam_now()
    if not built:
        b = D.budget(start, now)
        missing = [k for k, v in recs.items() if v is None] + (["sanity passed"] if sanity and not sanity["passed"]
                                                                else [])
        _require(not b["within_budget"], f"DTS is not built yet ({', '.join(missing)} missing); the budget ends "
                                         f"{b['deadline']}", Refused)
        rec = {"stage": "stop", "seed": int(args.seed), "built": False, "stop": True,
               "reason": "not built within the 24-hour budget", "budget": b, "missing": missing,
               "rule_sha256": R.RULE_SHA256, "settings_sha256": G.sha256_file(D.G.SETTINGS_PATH),
               "module_sha256": R.r6_module_shas(),
               "input_sha256": {k: G.sha256_file(Path(args.out) / k) for k, v in recs.items() if v is not None},
               "time": now}
        write_new(out, rec)
        print(f"dts stop: STOP (not built within the budget); record {out}", flush=True)
        return EXIT_FAIL
    _require(chosen["setting"] == D.setting_name(recs[TUNE]["chosen"]["wording_id"], recs[TUNE]["chosen"]["K"]),
             "the chosen stage ran another setting than the tuning chose")
    now_shas = {f: R.r6_module_shas().get(f) for f in DTS_FILES}
    changed = [k for k, v in recs.items() if dts_shas(v) != now_shas]
    _require(not changed, f"stage records made by other DTS code bytes: {changed}; rerun them (rule section 6 item "
                          f"7)", Refused)  # guard:same_modules
    npz = Path(args.out) / chosen["per_anchor_file"]
    _require(G.sha256_file(npz) == chosen["per_anchor_sha256"], f"{npz} is not the chosen stage's file")
    with np.load(npz, allow_pickle=False) as z:
        r1 = z["dts__r1"]
    dec = D.stop_decision(r1, n_total)
    b = D.budget(start, max((sanity["time"], chosen["time"]), key=D.parse_amsterdam))
    stop = bool(dec["dts_above_aff"] or not b["within_budget"])
    reason = ("DTS's hit count is above AFF's" if dec["dts_above_aff"] else
              "not built within the 24-hour budget" if not b["within_budget"] else None)
    rec = {"stage": "stop", "seed": int(args.seed), "built": True, "setting": chosen["setting"], **dec,
           "budget": b, "stop": stop, "reason": reason, "rule_sha256": R.RULE_SHA256,
           "settings_sha256": chosen["settings_sha256"], "module_sha256": R.r6_module_shas(),
           "input_sha256": {name: G.sha256_file(Path(args.out) / name) for name in recs} | {npz.name:
                                                                                        G.sha256_file(npz)},
           "time": now}
    write_new(out, rec)
    print(f"dts stop: {'STOP (' + reason + ')' if stop else 'pass (no stop)'}; record {out}", flush=True)
    return EXIT_FAIL if stop else 0


# ---------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Round 6 describe-then-score, seed-42 CPU stages (rule section 7); "
                                             "prints no metric")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--seed", type=int, default=R.DEV_SEED, choices=sorted(ADMITTED) + list(R.HELD_SEEDS),
                    help="42 (default); smoke seeds 9001 to 9003; held seeds 52 to 54 only with list-input --for held")
    ap.add_argument("--episodes", type=Path, help="the seed's episodes file (r6_episodes.save_episodes)")
    ap.add_argument("--verbalise-out", type=Path, action="append", default=[], help="verbaliser output folder")
    ap.add_argument("--verbalise-job", type=Path, action="append", default=[], help="verbaliser job folder")
    ap.add_argument("--listing-out", type=Path, action="append", default=[], help="listing output folder")
    ap.add_argument("--embeddings", type=Path, help="value-embedding cache (default <out>/dts_value_embeddings.npz)")
    ap.add_argument("--out", type=Path, help="stage records folder (default results/, smoke results/smoke/dts/)")
    ap.add_argument("--for", dest="for_stage", choices=LIST_FOR, help="list-input: which stage (held: a held seed's "
                    "phrases of the chosen setting)")
    ap.add_argument("--job-out", type=Path, help="list-input: the listing job folder to create")
    ap.add_argument("--clock-start", help="stop: the first DTS commit, 'YYYY-MM-DD HH:MM' Amsterdam time")
    args = ap.parse_args(argv)
    held = args.seed in R.HELD_SEEDS
    if held != (args.stage == "list-input" and args.for_stage == "held"):
        ap.error("held seeds 52 to 54 go with --stage list-input --for held only, and --for held with a held seed")
    args.out = Path(args.out or default_out(args.seed))
    args.embeddings = Path(args.embeddings or args.out / EMBEDDINGS)
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        if args.stage == "stop":
            return stage_stop(args)
        settings, settings_sha = D.load_settings()
        if args.stage == "list-input":
            return stage_list_input(args, settings, settings_sha)
        if args.stage in ("tune", "chosen"):
            require_sanity(args)
        ctx = load_context(args.seed, args.episodes)
        return {"sanity": stage_sanity, "tune": stage_tune, "chosen": stage_chosen}[args.stage](
            args, ctx, settings, settings_sha)
    except Refused as e:
        print(f"dts {args.stage}: refused: {e}", flush=True)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
