"""R2, the adapted learned reader (rule §4.3), our own code. Usage: rd2_r2.py A0 | A1 (A1 = the ablation of §4.8).

(a) mu42 = X.mean(axis=0), sigma42 = X.std(axis=0) of X = vstack(F_a, F_b), the 24,576 seed-42 feature rows; a zero
    sigma entry -> 1 (reported). Half-reader j: model_j.predict_proba((x - mu42) / sigma42); P_n = mean of the two.
(e) code check first: with each half's own scaler statistics and no EM, R1's P within 1e-12 and identical picks.
(b) EM prior correction (Saerens et al. 2002): pi_train = 1/H, pi^(0) = 1/H, stop at the first s with
    max|pi^(s+1) - pi^(s)| < 1e-10 or s + 1 = 10,000; pi_hat = pi^(s+1).
(c) P' = P pi_hat / pi_train renormalised; picks, top-two margins and tau_0..3 of P' (thresholds are not candidate
    numbers). (f) shift report. No T, no cell and no R@1 is computed here.
Writes out/rd2_r2_<config>.{json,npz}."""
import sys
import time

import numpy as np

import rd2_core as K
from rd2_core import COND, RC

TOL, CAP = 1e-10, 10_000


def em(P, H):
    """P: (N, H) float64 averaged half-reader probabilities. Returns pi_hat, s_stop, n_updates, converged, trace."""
    pi_train = np.full(H, 1.0 / H)
    pi = np.full(H, 1.0 / H)
    trace = []
    s = 0
    while True:
        w = P * pi[None, :] / pi_train[None, :]
        Pp = w / w.sum(axis=1, keepdims=True)
        pi_new = Pp.mean(axis=0)
        delta = float(np.max(np.abs(pi_new - pi)))
        if s < 5 or s % 100 == 0:
            trace.append({"s": s, "max_abs_change": delta, "pi": pi_new.tolist()})
        if delta < TOL:
            return pi_new, s, s + 1, True, trace, delta
        if s + 1 == CAP:
            return pi_new, s, s + 1, False, trace, delta
        pi = pi_new
        s += 1


def adapt(P, pi_hat, H):
    pi_train = np.full(H, 1.0 / H)
    w = P * pi_hat[None, :] / pi_train[None, :]
    return w / w.sum(axis=1, keepdims=True)


def distribution(x):
    x = np.asarray(x, np.float64)
    return {"n": int(len(x)), "mean": float(x.mean()),
            "deciles": {str(q): float(v) for q, v in zip(range(10, 100, 10), np.percentile(x, range(10, 100, 10)))}}


def main(config):
    t0 = time.time()
    RC.assert_rule()
    RC.assert_inputs([f"results/rb_reader_{config}.pkl", f"results/rb_reader_{config}.json",
                      f"results/rb_diag_{config}.json"])
    cache = K.load_cache()
    F = cache["F"][config]
    parts = K.CONFIGS[config]
    H = len(parts)
    pk = K.load_reader_pickle(config)
    names = K.feature_names(parts)
    # R1 (frozen half-readers with their scalers)
    r1_half = {c: [K.half_probs_scaled(h["model"], h["scaler"].transform(F[c])) for h in pk["halves"]] for c in COND}
    P1 = {c: K.mean_two(*r1_half[c]) for c in COND}
    pick1 = {c: K.picks_margins(P1[c])[0] for c in COND}
    # (a) re-standardisation statistics
    X = np.vstack([F["a"], F["b"]]).astype(np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    zero = [names[i] for i in np.flatnonzero(sd == 0)]
    sd_used = np.where(sd == 0, 1.0, sd)
    # precision cross-check of mu: exact (fsum) mean
    import math
    mu_fsum = np.array([math.fsum(X[:, i]) / X.shape[0] for i in range(X.shape[1])])
    # (e) code check: own-statistics standardisation, no EM, reproduces R1
    cc_half = {c: [K.half_probs_scaled(h["model"], (F[c] - h["scaler"].mean_) / h["scaler"].scale_)
                   for h in pk["halves"]] for c in COND}
    Pcc = {c: K.mean_two(*cc_half[c]) for c in COND}
    code_check = {"max_abs_diff_vs_R1": float(max(np.abs(Pcc[c] - P1[c]).max() for c in COND)),
                  "picks_identical": all(np.array_equal(K.picks_margins(Pcc[c])[0], pick1[c]) for c in COND),
                  "bitwise_equal": all(np.array_equal(Pcc[c], P1[c]) for c in COND)}
    code_check["passed"] = bool(code_check["max_abs_diff_vs_R1"] <= 1e-12 and code_check["picks_identical"])
    if not code_check["passed"]:
        raise SystemExit(f"R2 code check failed: {code_check}")
    K.log(f"{config} code check passed {code_check}", t0)
    # R2 probabilities before EM
    r2_half = {c: [K.half_probs_scaled(h["model"], (F[c] - mu) / sd_used) for h in pk["halves"]] for c in COND}
    P = {c: K.mean_two(*r2_half[c]) for c in COND}
    Pall = np.vstack([P["a"], P["b"]])
    pi_hat, s_stop, n_upd, conv, trace, last_delta = em(Pall, H)
    K.log(f"{config} EM: pi_hat {pi_hat.tolist()} s_stop {s_stop} updates {n_upd} converged {conv}", t0)
    Pa = {c: adapt(P[c], pi_hat, H) for c in COND}
    pm = {c: K.picks_margins(Pa[c]) for c in COND}
    picks = {c: pm[c][0] for c in COND}
    margins = {c: pm[c][1] for c in COND}
    taus = K.thresholds(margins["a"], margins["b"])
    # shift report (diagnostic)
    shift = {"per_half": {}}
    for j, h in enumerate(pk["halves"]):
        sc = h["scaler"]
        shift["per_half"][str(j)] = {"mean_shift_in_bank_sd": dict(zip(names, ((mu - sc.mean_) / sc.scale_).tolist())),
                                     "sd_ratio": dict(zip(names, (sd / sc.scale_).tolist()))}
    top_r2 = np.concatenate([Pa[c].max(axis=1) for c in COND])
    top_r1 = np.concatenate([P1[c].max(axis=1) for c in COND])
    top_r2_noem = np.concatenate([P[c].max(axis=1) for c in COND])
    differ = np.concatenate([picks[c] != pick1[c] for c in COND])
    allp = np.concatenate([picks[c] for c in COND])
    allp1 = np.concatenate([pick1[c] for c in COND])
    shift.update({
        "top_probability": {"R2_seed42": distribution(top_r2), "R2_before_EM_seed42": distribution(top_r2_noem),
                            "R1_seed42": distribution(top_r1)},
        "pick_differs_from_R1_share_pct": 100 * float(differ.mean()),
        "pick_differs_from_R1_share_pct_per_condition": {c: 100 * float(np.mean(picks[c] != pick1[c])) for c in COND},
        "pick_share_pct": {"R2": {h: 100 * float(np.mean(allp == i)) for i, h in enumerate(parts)},
                           "R1": {h: 100 * float(np.mean(allp1 == i)) for i, h in enumerate(parts)}},
        "pick_differs_before_EM_share_pct": 100 * float(np.concatenate(
            [K.picks_margins(P[c])[0] != pick1[c] for c in COND]).mean())})
    rec = {"config": config, "groupings": list(parts), "n_rows": int(X.shape[0]),
           "mu42": mu.tolist(), "sigma42": sd.tolist(), "sigma42_zero_entries": zero,
           "mu42_vs_fsum_max_rel": float(np.max(np.abs(mu - mu_fsum) / np.maximum(np.abs(mu_fsum), 1e-300))),
           "feature_names": names, "code_check": code_check,
           "em": {"pi_train": [1.0 / H] * H, "pi_hat": pi_hat.tolist(), "s_stop": int(s_stop),
                  "n_updates": int(n_upd), "converged_below_tol": bool(conv), "cap_reached": bool(not conv),
                  "last_max_abs_change": last_delta, "tol": TOL, "cap": CAP, "trace": trace,
                  "pi_before_EM_mean_P": Pall.mean(axis=0).tolist()},
           "taus": taus, "n_margin_values": int(2 * len(margins["a"])), "shift_report": shift,
           "R1_halves_C": [float(h["model"].C) for h in pk["halves"]],
           "runtime_s": round(time.time() - t0, 1), "provenance": K.provenance()}
    K.save_json(K.OUT / f"rd2_r2_{config}.json", rec)
    npz = {"mu42": mu, "sigma42": sd, "pi_hat": pi_hat, "taus": np.asarray(taus)}
    for c in COND:
        npz[f"P__{c}"] = Pa[c]
        npz[f"P_noEM__{c}"] = P[c]
        npz[f"P_half0__{c}"], npz[f"P_half1__{c}"] = r2_half[c]
        npz[f"pick__{c}"] = picks[c]
        npz[f"margin__{c}"] = margins[c]
        npz[f"R1__P__{c}"] = P1[c]
    np.savez(K.OUT / f"rd2_r2_{config}.npz", **npz)
    K.log(f"{config}: taus {taus}; picks differ from R1 {shift['pick_differs_from_R1_share_pct']:.3f}%", t0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A0")
