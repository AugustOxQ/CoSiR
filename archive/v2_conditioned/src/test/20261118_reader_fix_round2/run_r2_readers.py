"""Round 2's reader stream (binding rule: DECISION_RULE.md of this folder, §4.3, §4.4, §4.8, §7): the seed-42
probabilities P^c(h) of R2 and R3, for the fusion stream (run_r2_fusion.py). Every setting is fixed by the rule; this
script chooses none. CPU only; reads no held row and no evaluation label.

  r2 --config X   §4.3: round 1's two half-readers of X re-standardised with mu42, sigma42 of the seed-42 features and
                  corrected by the EM class prior pi_hat. The code check (§4.3 e) runs first and stops on failure.
                  -> results/probs_R2_<X>.npz (P__a, P__b = adapted P') and .json (mu42, sigma42, pi_hat, EM record,
                     code check, shift report of §4.3 f)
  r3 --config X   §4.4: replacement draws, impure banks of purity 1..4, their features, the purity-4 checks, D(k) and
                  k*, written to results/r3_k_<X>.json BEFORE any half-reader is trained; then round 1's recipe at k*
                  -> results/r3_reader_<X>.pkl (frozen half-readers for the test seeds), results/probs_R3_<X>.npz
                     (P__a, P__b) and .json. If k* = 4, R3 is R1: only probs_R3_<X>.json (r3_is_r1: true) is written,
                     with no npz and no training (§4.4 h).
  r3 --config X --check-only
                  the purity-4 checks of §4.4 d and f on the real banks and the real seed-42 bundle, nothing else
                  (no impure bank, no D(k) for k < 4, no training) -> results/smoke/r3_check_<X>.json

X in A0, A1 (A1 = the ablation of §4.8: round 1's A1 readers and banks, pi_train = 1/4, 24 features).

Output contract (the fusion stream reads it): P__a, P__b float64 (n_episodes, H), columns in configuration order
common.CONFIGS[X], rows in the bundle's episode order; the JSON holds reader, config, smoke, n_episodes, groupings,
npz_sha256 and, for R3, r3_is_r1.

--smoke: common.load_bundle(smoke=True) (600 seed-42 episodes). R2 uses the real (D14, hash-checked) half-readers.
R3 uses the real D14 banks; the draws are made for the full N of each half (so the hashes of order, img and cap are the
real run's), and features, D(k) and training use the first 1,000 episodes of each bank block. The purity-4 feature
check compares those rows of round 1's half{j}__X exactly; the purity-4 SMD check is skipped (another seed-42 sample).
Outputs go to results/smoke/ (overwritable). Smoke numbers are not results.

Run (from /project/CoSiR):
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261118_reader_fix_round2/run_r2_readers.py {r2,r3} --config A0 [--smoke]
Non-smoke outputs are never overwritten.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r2_common as R  # noqa: E402
import r2_readers as RR  # noqa: E402

C, rb, rbe, rf = R.C, R.rb, R.rbe, R.rf
CONDITIONS = RR.CONDITIONS
MODULES = ["DECISION_RULE.md", "common.py", "rc_core.py", "rb_build.py", "rb_eval.py", "rb_features.py"]

for _m, _name in ((C, "common.py"), (rb, "rb_build.py"), (rbe, "rb_eval.py"), (rf, "rb_features.py")):
    if Path(_m.__file__).resolve() != R.r1_path(_name).resolve():
        raise ImportError(f"{_name} resolved to {_m.__file__}, not round 1's")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def code_sha():
    return {p.name: R.sha_file(p) for p in (HERE / "r2_common.py", HERE / "r2_readers.py", Path(__file__).resolve())}


def runtime_info():
    import sklearn
    return {"argv": sys.argv, "pid": os.getpid(), "sklearn_version": sklearn.__version__, "numpy_version": np.__version__,
            "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}}


def reader_files(config):
    return [f"results/rb_reader_{config}.{e}" for e in ("pkl", "json", "npz")]


def r3_files(config):
    parts = C.CONFIGS[config]
    return (["results/rb_halves.npz", "results/rb_halves.json"]
            + [f"results/rb_heads_{h}.{e}" for h in parts for e in ("npz", "json")]
            + [f"results/rb_bank_{config}_half{j}.{e}" for j in (0, 1) for e in ("npz", "json")]
            + [f"results/rb_reader_{config}.npz", f"results/rb_reader_{config}.json", f"results/rb_diag_{config}.json"])


def save_npz(path, P):
    """P__a, P__b float64 (n_episodes, H); returns the file's SHA-256."""
    np.savez(path, P__a=np.asarray(P["a"], dtype=np.float64), P__b=np.asarray(P["b"], dtype=np.float64))
    return R.sha_file(path)


def check_probs(P, n, H):
    for c in CONDITIONS:
        if P[c].shape != (n, H) or P[c].dtype != np.float64 or not np.isfinite(P[c]).all():
            raise AssertionError(f"P^{c}: shape {P[c].shape}, dtype {P[c].dtype}; expected finite float64 ({n}, {H})")
        if not np.allclose(P[c].sum(axis=1), 1.0, rtol=0, atol=1e-12):
            raise AssertionError(f"P^{c} rows do not sum to 1")
    return True


def pick_share(P, parts):
    """Percentage of the (episode, condition) values whose pick (ties to the first grouping) is each grouping."""
    picks = np.concatenate([rf.picks_and_margins(P[c])[0] for c in CONDITIONS])
    return {h: 100 * float(np.mean(picks == i)) for i, h in enumerate(parts)}


def top_distribution(P):
    return {"both_conditions": rf.distribution(np.concatenate([P[c].max(axis=1) for c in CONDITIONS])),
            **{c: rf.distribution(P[c].max(axis=1)) for c in CONDITIONS}}


def seed42(config, smoke):
    parts = C.CONFIGS[config]
    if parts != rb.CONFIGS[config]:
        raise AssertionError("configuration groupings differ between common.py and rb_build.py")
    bundle = C.load_bundle(smoke=smoke)
    F42, feat_checks = rbe.seed42_features(bundle, parts)
    return bundle, F42, feat_checks


def r1_against_round1_rc(P1, bundle):
    """A0: R1's picks and margins equal round-1 R-c's stored pick__{a,b} and margin__{a,b} (D14), the rows of the
    bundle's episodes. Confirms that the R1 of R2's code check is round 1's reader."""
    z = np.load(R.r1_path("results/cand_Rc_Rb_expected_A0.npz"))
    idx = bundle.subset_index if bundle.smoke else slice(None)
    out = {}
    for c in CONDITIONS:
        pick, margin = rf.picks_and_margins(P1[c])
        out[c] = bool(np.array_equal(pick, z[f"pick__{c}"][idx].astype(np.int64))
                      and np.array_equal(margin, z[f"margin__{c}"][idx]))
    return out


# =============================================================== R2 (§4.3)

def stage_r2(args):
    config, smoke = args.config, args.smoke
    parts = C.CONFIGS[config]
    H = len(parts)
    R.assert_rule()
    names = MODULES + reader_files(config) + (["results/cand_Rc_Rb_expected_A0.npz"] if config == "A0" else [])
    inputs = R.assert_inputs(names)
    out = R.res_dir(smoke)
    p = {"npz": out / f"probs_R2_{config}.npz", "json": out / f"probs_R2_{config}.json"}
    R.refuse_existing(p.values(), smoke)
    t0 = time.time()
    pk, rrec, rz = rb.load_readers(config, False)                 # the real D14 half-readers, also in smoke
    fnames = rf.feature_names(parts)
    if pk["feature_names"] != fnames or len(pk["halves"]) != 2:
        raise AssertionError("the readers were trained on another feature layout")
    halves = pk["halves"]
    bundle, F42, feat_checks = seed42(config, smoke)
    n = len(F42["a"])
    log(f"seed-42 features: {n} episodes x {len(fnames)} features per condition [{time.time() - t0:.0f}s]")

    # R1 exactly as rb_eval.stage_config computes it
    P1 = {c: rf.average_probs(rbe.half_reader_probs(pk, F42[c], H)) for c in CONDITIONS}
    check_probs(P1, n, H)
    r1_rc = None
    if config == "A0":
        r1_rc = r1_against_round1_rc(P1, bundle)
        if not all(r1_rc.values()) and not smoke:
            raise SystemExit(f"R1's picks or margins differ from round-1 R-c's stored arrays: {r1_rc}")

    # a. re-standardisation statistics
    mu, sd, zero = RR.restandardise_stats(F42["a"], F42["b"])
    if zero:
        log(f"WARNING: sigma42 is exactly 0 for features {[fnames[i] for i in zero]}; replaced by 1 (§4.3 a)")
    # e. code check, before any R2 probability
    code = RR.code_check(P1, RR.reader_probs(halves, F42, RR.scaler_stats(halves), H))
    log(f"code check passed: max |diff| a {code['a']['max_abs_diff']:.3g}, b {code['b']['max_abs_diff']:.3g}; "
        "picks identical")
    # a. probabilities with mu42, sigma42 for both half-readers; b. EM over the 24,576 values (a then b)
    P = RR.reader_probs(halves, F42, [(mu, sd), (mu, sd)], H)
    check_probs(P, n, H)
    pi_train = np.full(H, 1.0 / H)
    pi_hat, em = RR.em_prior(np.vstack([P["a"], P["b"]]), pi_train)
    log(f"EM: pi_hat {np.round(pi_hat, 4).tolist()} after {em['n_iter']} iterations "
        f"(criterion met {em['converged_by_criterion']}, cap reached {em['cap_reached']})")
    Pp = {c: RR.adapt(P[c], pi_hat, pi_train) for c in CONDITIONS}
    check_probs(Pp, n, H)

    # f. shift report (diagnostic)
    shift = {"i_per_half_reader": {}}
    for j, h in enumerate(halves):
        m, s = np.asarray(h["scaler"].mean_), np.asarray(h["scaler"].scale_)
        shift["i_per_half_reader"][str(j)] = {"mu42_minus_mean_over_scale": dict(zip(fnames, ((mu - m) / s).tolist())),
                                              "sigma42_over_scale": dict(zip(fnames, (sd / s).tolist()))}
    shift["ii_pi_hat"] = pi_hat.tolist()
    shift["ii_em_iterations"] = em["n_iter"]
    oof = np.vstack([rz["half0__oof_proba"], rz["half1__oof_proba"]])
    shift["iii_top_probability"] = {
        "R2_adapted_seed42": top_distribution(Pp),
        "R2_restandardised_before_EM_seed42": top_distribution(P),
        "R1_seed42": top_distribution(P1),
        "round1_bank_oof_pooled_halves": rf.distribution(oof.max(axis=1)),
        "note": "round 1's values: R1 on seed 42 mean 0.694, bank out-of-fold mean 0.781"}
    pick1 = {c: rf.picks_and_margins(P1[c])[0] for c in CONDITIONS}
    pick2 = {c: rf.picks_and_margins(Pp[c])[0] for c in CONDITIONS}
    diff = np.concatenate([pick2[c] != pick1[c] for c in CONDITIONS])
    shift["iv_pick_differs_from_R1"] = {"share_both_conditions": float(diff.mean()),
                                        **{c: float(np.mean(pick2[c] != pick1[c])) for c in CONDITIONS},
                                        "n_values": int(len(diff))}
    shift["pick_share_percent"] = {"R1": pick_share(P1, parts), "R2": pick_share(Pp, parts)}

    sha = save_npz(p["npz"], Pp)
    rec = {"reader": "R2", "config": config, "groupings": list(parts), "smoke": bool(smoke), "n_episodes": int(n),
           "n_values": int(2 * n), "npz_sha256": sha, "npz_keys": ["P__a", "P__b"],
           "rows": "P__a, P__b: the adapted P'^c(h) (§4.3 b), float64 (n_episodes, H), columns in configuration order, "
                   "rows in the bundle's episode order",
           "feature_names": fnames, "mu42": mu.tolist(), "sigma42": sd.tolist(),
           "sigma42_zero_replaced_by_1": [fnames[i] for i in zero],
           "pi_train": pi_train.tolist(), "pi_hat": pi_hat.tolist(), "em": em,
           "code_check": code, "r1_picks_margins_equal_round1_Rc": r1_rc,
           "shift_report": shift, "half_readers_chosen_C": {j: rrec["halves"][j]["chosen_C"] for j in ("0", "1")},
           "reader_pkl_sha256": rrec["provenance"]["pkl_sha256"], "feature_checks": feat_checks,
           "bundle_checks": bundle.checks, "inputs_sha256": inputs, "code_sha256": code_sha(), "run": runtime_info(),
           "runtime_s": round(time.time() - t0, 1)}
    R.write_json_once(p["json"], rec, smoke)
    log(f"R2/{config}{' [SMOKE]' if smoke else ''}: wrote {p['npz'].name} and {p['json'].name}; top probability mean "
        f"R1 {shift['iii_top_probability']['R1_seed42']['both_conditions']['mean']:.3f}, R2 "
        f"{shift['iii_top_probability']['R2_adapted_seed42']['both_conditions']['mean']:.3f}; picks differing from R1 "
        f"{100 * shift['iv_pick_differs_from_R1']['share_both_conditions']:.1f}% [{time.time() - t0:.0f}s]")


# =============================================================== R3 (§4.4)

def cross_fitted_posteriors(parts, Hh):
    post = {}
    for h in parts:
        hp, _ = rb.load_heads(h, False, Hh["sha256"])
        if not np.array_equal(hp["filled_by_half"], 1 - Hh["half_of_local_row"]):
            raise AssertionError(f"rb_heads_{h}: rows are not scored by the other half's heads")
        post[h] = {"img": hp["img"], "txt": hp["txt"]}
    return post


def r3_banks(config, smoke, purities, post, Hh, rz):
    """Per half j: the draws (full N), the impure banks' features for each purity (all episodes, or the smoke subset),
    the purity-4 checks against round 1's half{j}__X / half{j}__y (exact; a failure stops), and the SHA-256s."""
    parts = C.CONFIGS[config]
    paint = Hh["painting_of_local_row"]
    half_of = Hh["half_of_local_row"]
    out = {}
    for j in (0, 1):
        t = time.time()
        ep, blocks, n_block, _ = rb.load_bank(config, j, False, Hh["sha256"])
        N = len(ep.anchor)
        Rw = Hh[f"local_rows_half{j}"]
        if (N != RR.BANK_N[config] or n_block != RR.BLOCK_N or blocks != rb.BLOCKS[config]
                or len(Rw) != RR.HALF_ROWS[j] or not (np.diff(Rw) > 0).all()):
            raise AssertionError(f"half {j}: bank or half rows differ from the rule (§4.4 a)")
        if not (half_of[ep.rows()] == j).all():
            raise AssertionError(f"bank half {j}: a row outside half {j}, its posteriors would not be cross-fitted")
        order, img, cap, dinfo = RR.replacement_draws(ep.anchor, Rw, paint, RR.R3_SEEDS[j])
        if not ((half_of[img] == j).all() and (half_of[cap] == j).all()):
            raise AssertionError(f"half {j}: a replacement row outside half {j}")
        pa = paint[ep.anchor][:, None, None]
        if not ((paint[img] != pa).all() and (paint[cap] != pa).all() and (paint[cap] != paint[img]).all()):
            raise AssertionError(f"half {j}: a replacement pair breaks the painting constraints")
        full = {"anchor": ep.anchor, "candidates": ep.candidates,
                **{k: getattr(ep, k) for kk in RR.PAIR_KEYS for k in kk}}
        dinfo["share_slots_reusing_an_episode_painting"] = RR.episode_painting_reuse(full, img, cap, paint)
        idx = (RR.subset_index(len(blocks), n_block, RR.SMOKE_PER_BLOCK) if smoke
               else np.arange(N, dtype=np.int64))
        base = {k: full[k][idx] for kk in RR.PAIR_KEYS for k in kk}
        ya, yb = rf.bank_labels(blocks, n_block, parts)
        X, sha = {}, {"order": RR.sha_array(order), "img": RR.sha_array(img), "cap": RR.sha_array(cap)}
        for k in purities:
            X[k], y, epi = RR.bank_matrix(post, parts, RR.impure_pairs(base, order[idx], img[idx], cap[idx], k),
                                          ya[idx], yb[idx])
            sha[f"features_purity{k}"] = RR.sha_array(X[k])
        rows = RR.stacked_rows(idx, N)
        RR.require_equal(f"half{j}__X (purity-4 features)", X[4], rz[f"half{j}__X"][rows])
        RR.require_equal(f"half{j}__y (labels)", y, rz[f"half{j}__y"][rows])
        out[j] = {"X": X, "y": y, "episode": epi, "n_episodes": int(len(idx)), "n_bank": int(N), "sha256": sha,
                  "draws": dinfo, "blocks": [f"{a}__{b}" for a, b in blocks]}
        log(f"half {j}: draws ({dinfo['img_entries_redrawn']} image / {dinfo['cap_entries_redrawn']} caption entries "
            f"redrawn), features of purities {list(purities)} on {len(idx)} episodes; purity-4 features and labels "
            f"equal round 1's exactly [{time.time() - t:.0f}s]")
    return out


def stage_r3(args):
    config, smoke, check_only = args.config, args.smoke, args.check_only
    if check_only and smoke:
        raise SystemExit("--check-only runs on the real banks and the real seed-42 bundle; do not combine with --smoke")
    parts = C.CONFIGS[config]
    Hn = len(parts)
    R.assert_rule()
    inputs = R.assert_inputs(MODULES + r3_files(config))
    out = R.res_dir(smoke or check_only)
    if check_only:
        p = {"check": out / f"r3_check_{config}.json"}
    else:
        p = {"k": out / f"r3_k_{config}.json", "npz": out / f"probs_R3_{config}.npz",
             "json": out / f"probs_R3_{config}.json", "pkl": out / f"r3_reader_{config}.pkl"}
    R.refuse_existing(p.values(), smoke or check_only)
    t0 = time.time()
    bundle, F42, feat_checks = seed42(config, smoke)
    x42 = np.vstack([F42["a"], F42["b"]])
    fnames = rf.feature_names(parts)
    log(f"seed-42 features: {len(x42)} rows [{time.time() - t0:.0f}s]")
    Hh = rb.load_halves(False)
    post = cross_fitted_posteriors(parts, Hh)
    rz = np.load(R.r1_path(f"results/rb_reader_{config}.npz"))
    diag = json.loads(R.r1_path(f"results/rb_diag_{config}.json").read_text())
    purities = (4,) if check_only else RR.PURITIES
    banks = r3_banks(config, smoke, purities, post, Hh, rz)
    del post

    # f. SMD per feature, D(k), the purity-4 SMD check, k*
    bank_by_k = {k: np.vstack([banks[0]["X"][k], banks[1]["X"][k]]) for k in purities}
    tab = RR.d_table(x42, bank_by_k)
    sr = diag["c_shift_report"]
    if smoke:
        smd_check = "skipped (smoke: a 600-episode seed-42 sample and a bank subset)"
    else:
        if (sr["n_seed42"], sr["n_bank"]) != (len(x42), len(bank_by_k[4])):
            raise SystemExit(f"R3 purity-4 SMD check: row counts {(len(x42), len(bank_by_k[4]))} differ from round 1's "
                             f"{(sr['n_seed42'], sr['n_bank'])}")
        RR.require_smd_equal(tab[4][0], fnames, sr["smd"])
        smd_check = True
    checks = {f"features_purity4_equal_round1_half{j}__X": True for j in (0, 1)}
    checks.update({f"labels_equal_round1_half{j}__y": True for j in (0, 1)})
    checks["smd_purity4_equal_round1_c_shift_report"] = smd_check
    D = {k: tab[k][1] for k in purities}
    common_rec = {
        "reader": "R3", "config": config, "groupings": list(parts), "smoke": bool(smoke), "feature_names": fnames,
        "n_seed42_rows": int(len(x42)), "n_bank_rows": {k: int(len(bank_by_k[k])) for k in purities},
        "subset": ("first %d episodes of each bank block (smoke); draws made for the full N" % RR.SMOKE_PER_BLOCK
                   if smoke else "all bank episodes"),
        "smd": {k: dict(zip(fnames, tab[k][0].tolist())) for k in purities}, "D": D,
        "purity4_checks": checks,
        "sha256": {f"half{j}": banks[j]["sha256"] for j in (0, 1)},
        "sha256_definition": "order, img, cap: (N, 2, 4) int64, C order, full N; features_purity{k}: the stacked "
                             "(2 n_episodes, F) float64 feature matrix of the half (rows 0..n-1 condition a, n..2n-1 "
                             "condition b), C order",
        "draws": {f"half{j}": banks[j]["draws"] for j in (0, 1)},
        "mean_abs_delta": {"seed42": RR.mean_abs_delta(x42, parts),
                           "bank": {k: RR.mean_abs_delta(bank_by_k[k], parts) for k in purities}},
        "feature_checks_seed42": feat_checks, "inputs_sha256": inputs, "code_sha256": code_sha(),
        "run": runtime_info()}
    if check_only:
        rec = {**common_rec, "stage": "purity-4 checks only (--check-only; §4.4 d and f), real banks and real seed-42 "
                                      "bundle", "check_only": True, "runtime_s": round(time.time() - t0, 1)}
        R.write_json_once(p["check"], rec, True)
        log(f"R3/{config} purity-4 checks passed on the real banks: features and labels equal round 1's exactly "
            f"(both halves); SMDs equal rb_diag_{config}.json exactly ({smd_check}); D(4) = {D[4]!r} -> {p['check']}")
        return
    k_star = RR.choose_k(D)
    rec_k = {**common_rec, "stage": "D(k) and k*, written before any R3 half-reader is trained (§4.4 f)",
             "k_star": int(k_star), "tie_rule": "smallest D(k) at full precision; exact ties to the larger k",
             "runtime_s": round(time.time() - t0, 1)}
    R.write_json_once(p["k"], rec_k, smoke)
    k_sha = R.sha_file(p["k"])
    log("D(k): " + ", ".join(f"k={k} {D[k]:.6f}" for k in purities) + f" -> k* = {k_star} (written {p['k'].name})")

    base_rec = {"reader": "R3", "config": config, "groupings": list(parts), "smoke": bool(smoke),
                "n_episodes": int(len(F42["a"])), "k_star": int(k_star), "D": D, "r3_k_json_sha256": k_sha,
                "inputs_sha256": inputs, "code_sha256": code_sha(), "run": runtime_info()}
    if k_star == 4:                                             # §4.4 h: R3 is R1, no retraining, no npz
        rec = {**base_rec, "r3_is_r1": True, "npz_sha256": None,
               "note": "k* = 4: R3 is R1 (round 1's half-readers, no retraining); not evaluated a second time, has "
                       "R1's development numbers and cannot be carried (DECISION_RULE.md §4.4 h)",
               "runtime_s": round(time.time() - t0, 1)}
        R.write_json_once(p["json"], rec, smoke)
        log(f"R3/{config}: k* = 4, R3 is R1; wrote {p['json'].name} only")
        return

    # g. training at k*, round 1's recipe unchanged
    for j in (0, 1):                                            # free the purities not trained
        banks[j]["X"] = {k_star: banks[j]["X"][k_star]}
    del bank_by_k
    halves, recs, oofs = [], {}, []
    for j in (0, 1):
        t = time.time()
        d = banks[j]
        X, y, epi = d["X"][k_star], d["y"], d["episode"]
        cnt = np.bincount(y, minlength=Hn)
        if (cnt != cnt[0]).any():
            raise AssertionError(f"half {j}: classes are not balanced ({cnt})")
        scaler, model, rec_j, oof = rb.fit_half_reader(X, y, epi, d["n_episodes"], Hn)
        halves.append({"scaler": scaler, "model": model, "C": rec_j["chosen_C"]})
        recs[str(j)] = {**rec_j, "convergence_warnings_cv_total": {str(r["C"]): int(sum(r["convergence_warnings"]))
                                                                   for r in rec_j["cv_table"]},
                        "cv_mean_log_loss": {str(r["C"]): r["mean_log_loss"] for r in rec_j["cv_table"]}}
        oofs.append(oof)
        log(f"half {j}: chosen C {rec_j['chosen_C']}, out-of-fold accuracy {rec_j['oof_accuracy_at_chosen_C']:.2f}%, "
            f"refit warnings {rec_j['refit_convergence_warnings']} [{time.time() - t:.0f}s]")
    pk3 = {"halves": halves}
    P = {c: rf.average_probs(rbe.half_reader_probs(pk3, F42[c], Hn)) for c in CONDITIONS}
    check_probs(P, len(F42["a"]), Hn)
    import sklearn
    with open(p["pkl"], "wb") as f:
        pickle.dump({"reader": "R3", "config": config, "groupings": list(parts), "feature_names": fnames,
                     "k_star": int(k_star), "halves": halves, "sklearn_version": sklearn.__version__,
                     "numpy_version": np.__version__, "rule_sha256": R.RULE_SHA, "smoke": bool(smoke),
                     "use": "P^c(h) = mean over the two halves of model.predict_proba(scaler.transform(x))"},
                    f, protocol=pickle.HIGHEST_PROTOCOL)
    pkl_sha = R.sha_file(p["pkl"])
    sha = save_npz(p["npz"], P)
    oof_all = np.vstack(oofs)
    rec = {**base_rec, "r3_is_r1": False, "npz_sha256": sha, "npz_keys": ["P__a", "P__b"], "pkl_sha256": pkl_sha,
           "rows": "P__a, P__b: P^c(h) = mean over R3's two half-readers, float64 (n_episodes, H), columns in "
                   "configuration order, rows in the bundle's episode order",
           "half_readers": {j: {"chosen_C": r["chosen_C"], "oof_accuracy_at_chosen_C": r["oof_accuracy_at_chosen_C"],
                                "refit_convergence_warnings": r["refit_convergence_warnings"],
                                "convergence_warnings_cv_total": r["convergence_warnings_cv_total"],
                                "cv_mean_log_loss": r["cv_mean_log_loss"]} for j, r in recs.items()},
           "half_readers_full_record": recs,
           "top_probability": {"bank_oof_at_chosen_C_pooled_halves": rf.distribution(oof_all.max(axis=1)),
                               "seed42": top_distribution(P)},
           "pick_share_percent_seed42": pick_share(P, parts),
           "half_reader_pick_agreement_seed42_percent": {
               c: 100 * float(np.mean(halves[0]["model"].predict_proba(halves[0]["scaler"].transform(F42[c])).argmax(1)
                                      == halves[1]["model"].predict_proba(halves[1]["scaler"].transform(F42[c])).argmax(1)))
               for c in CONDITIONS},
           "bank_episodes_per_half": {str(j): banks[j]["n_episodes"] for j in (0, 1)},
           "runtime_s": round(time.time() - t0, 1)}
    R.write_json_once(p["json"], rec, smoke)
    log(f"R3/{config}{' [SMOKE]' if smoke else ''}: k* = {k_star}; chosen C {[h['C'] for h in halves]}; top probability "
        f"mean seed 42 {rec['top_probability']['seed42']['both_conditions']['mean']:.3f}, bank OOF "
        f"{rec['top_probability']['bank_oof_at_chosen_C_pooled_halves']['mean']:.3f}; wrote {p['npz'].name}, "
        f"{p['json'].name}, {p['pkl'].name} [{time.time() - t0:.0f}s]")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reader", choices=("r2", "r3"))
    ap.add_argument("--config", required=True, choices=("A0", "A1"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="r3 only: the purity-4 checks on the real banks")
    args = ap.parse_args()
    R.assert_rule()
    if args.check_only and args.reader != "r3":
        raise SystemExit("--check-only applies to r3 only")
    {"r2": stage_r2, "r3": stage_r3}[args.reader](args)


if __name__ == "__main__":
    main()
