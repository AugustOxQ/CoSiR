"""R3, the realistic-practice learned reader (rule §4.4; A1 version §4.8), our own code.
Usage: rd2_r3.py A0 | A1 [--no-train]

(a) round 1's banks of half 0 and half 1 (D14, SHA-256 asserted); (b) the replacement draws with
    numpy.random.default_rng(21700 / 21800) in the rule's call order; (c) impure banks of purity k = 1, 2, 3 (purity 4 =
    round 1's bank); (d) features from the cross-fitted posteriors, with our own feature code (and checked equal to
    rb_features.both_conditions); purity-4 check against rb_reader_<cfg>.npz half{j}__X; SHA-256 of order, img, cap
    (int64, C order) and of every purity's feature matrix per half; (f) SMD per feature between the 24,576 seed-42 rows and
    the bank rows of each purity, D(k), k* (ties to the larger k), purity-4 SMD check against rb_diag_<cfg>.json;
    (g) if k* < 4: training at k* with round 1's recipe re-implemented here (StandardScaler on the half's whole impure
    bank, multinomial LogisticRegression lbfgs max_iter 2000, C in {0.01, 0.1, 1, 10, 100} by 5-fold CV over episodes
    (KFold(5, shuffle=True, random_state=0)), mean held-out log loss, ties to the smaller C, refit); R3's seed-42
    probabilities, picks, margins and tau_0..3. No T, no cell, no R@1. Writes out/rd2_r3_<cfg>.{json,npz} (+ the
    impure-bank features in out/rd2_r3_<cfg>_bankfeat.npz)."""
import math
import sys
import time
import warnings
from types import SimpleNamespace

import numpy as np

import rd2_core as K
from rd2_core import COND, RC

SEEDS = (21700, 21800)
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
BLOCKS = {"A0": [("affect", "caption"), ("affect", "image"), ("caption", "image")],
          "A1": [("affect", "caption"), ("affect", "csd"), ("affect", "image"), ("caption", "csd"), ("caption", "image"),
                 ("csd", "image")]}
N_BANK = {"A0": 49_152, "A1": 98_304}
HALF_ROWS = (91_949, 91_745)


def replacement_draws(N, A, R, paint, seed):
    """Rule §4.4 item b, in this exact call order on one generator."""
    g = np.random.default_rng(seed)
    U = g.random((N, 2, 4))
    order = np.argsort(U, axis=2, kind="stable")
    img = R[g.integers(0, len(R), size=(N, 2, 4))]
    pa = paint[A][:, None, None]
    rounds_img, redrawn_img = 0, 0
    while True:
        bad = paint[img] == pa
        nb = int(bad.sum())
        if nb == 0:
            break
        img[bad] = R[g.integers(0, len(R), size=nb)]
        rounds_img += 1
        redrawn_img += nb
    cap = R[g.integers(0, len(R), size=(N, 2, 4))]
    rounds_cap, redrawn_cap = 0, 0
    while True:
        bad = (paint[cap] == pa) | (paint[cap] == paint[img])
        nb = int(bad.sum())
        if nb == 0:
            break
        cap[bad] = R[g.integers(0, len(R), size=nb)]
        rounds_cap += 1
        redrawn_cap += nb
    info = {"img_redraw_rounds": rounds_img, "img_redrawn_entries": redrawn_img, "cap_redraw_rounds": rounds_cap,
            "cap_redrawn_entries": redrawn_cap}
    return order, img, cap, info


def impure_pairs(bank, order, img, cap, k):
    """Purity k: on each side the positions order[n, s, 0 .. 3 - k] get the replacement pair at that position."""
    if k == 4:
        return tuple(np.asarray(bank[f]) for f in ("pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt"))
    N = order.shape[0]
    rank = np.empty_like(order)
    ii = np.arange(N)[:, None, None]
    ss = np.arange(2)[None, :, None]
    rank[ii, ss, order] = np.arange(4)[None, None, :]
    rep = rank < (4 - k)
    pai = np.where(rep[:, 0], img[:, 0], bank["pairs_a_img"])
    pat = np.where(rep[:, 0], cap[:, 0], bank["pairs_a_txt"])
    pbi = np.where(rep[:, 1], img[:, 1], bank["pairs_b_img"])
    pbt = np.where(rep[:, 1], cap[:, 1], bank["pairs_b_txt"])
    return pai, pat, pbi, pbt


def smd(x42, xb):
    x42, xb = np.asarray(x42, np.float64), np.asarray(xb, np.float64)
    d = x42.mean(axis=0) - xb.mean(axis=0)
    s = np.sqrt((x42.var(axis=0, ddof=1) + xb.var(axis=0, ddof=1)) / 2.0)
    return d / s


def distribution(x):
    x = np.asarray(x, np.float64)
    return {"n": int(len(x)), "mean": float(x.mean()),
            "deciles": {str(q): float(v) for q, v in zip(range(10, 100, 10), np.percentile(x, range(10, 100, 10)))}}


def bank_labels(block_pairs, block_size, parts):
    ya = np.concatenate([np.full(block_size, parts.index(a), np.int64) for a, _ in block_pairs])
    yb = np.concatenate([np.full(block_size, parts.index(b), np.int64) for _, b in block_pairs])
    return ya, yb


def fit_half(X, y, N, H, t0, tag):
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    epi = np.concatenate([np.arange(N), np.arange(N)])
    fold_ep = np.full(N, -1, np.int64)
    for f, (_, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(N))):
        fold_ep[te] = f
    fold = fold_ep[epi]
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    labels = np.arange(H)
    table, oofs = [], []
    for Cc in C_GRID:
        oof = np.full((len(y), H), np.nan)
        lls, warn = [], 0
        for f in range(5):
            tr, te = fold != f, fold == f
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                m = LogisticRegression(C=Cc, solver="lbfgs", max_iter=2000).fit(Xs[tr], y[tr])
            warn += sum(issubclass(w.category, ConvergenceWarning) for w in caught)
            oof[te] = m.predict_proba(Xs[te])
            lls.append(float(log_loss(y[te], oof[te], labels=labels)))
        table.append({"C": Cc, "mean_log_loss": float(np.mean(lls)), "fold_log_loss": lls, "convergence_warnings": int(warn),
                      "oof_accuracy": 100 * float(np.mean(oof.argmax(1) == y))})
        oofs.append(oof)
        K.log(f"{tag} C {Cc}: mean log loss {table[-1]['mean_log_loss']!r}, warnings {warn}", t0)
    best = 0
    for i in range(1, len(C_GRID)):
        if table[i]["mean_log_loss"] < table[best]["mean_log_loss"]:
            best = i
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(C=C_GRID[best], solver="lbfgs", max_iter=2000).fit(Xs, y)
    refit_warn = int(sum(issubclass(w.category, ConvergenceWarning) for w in caught))
    srt = sorted(r["mean_log_loss"] for r in table)
    rec = {"cv_table": table, "chosen_C": C_GRID[best], "gap_best_to_second": srt[1] - srt[0],
           "refit_convergence_warnings": refit_warn, "oof_accuracy_at_chosen_C": table[best]["oof_accuracy"],
           "n_iter_refit": np.asarray(model.n_iter_).tolist(), "fold_sizes": np.bincount(fold_ep).tolist(),
           "class_counts": np.bincount(y, minlength=H).tolist()}
    return scaler, model, rec, oofs[best]


def main(config, train=True):
    t0 = time.time()
    RC.assert_rule()
    parts = K.CONFIGS[config]
    H = len(parts)
    names = K.feature_names(parts)
    heads = ("affect", "image", "caption") + (("csd",) if config == "A1" else ())
    inputs = (["results/rb_halves.npz", "results/rb_halves.json", f"results/rb_reader_{config}.npz",
               f"results/rb_diag_{config}.json"]
              + [f"results/rb_bank_{config}_half{j}.{e}" for j in (0, 1) for e in ("npz", "json")]
              + [f"results/rb_heads_{h}.{e}" for h in heads for e in ("npz", "json")])
    RC.assert_inputs(inputs)
    cache = K.load_cache()
    F42 = cache["F"][config]
    X42 = np.vstack([F42["a"], F42["b"]])
    hz = np.load(RC.r1_path("results/rb_halves.npz"))
    paint = np.asarray(hz["painting_of_local_row"], np.int64)
    half_of = np.asarray(hz["half_of_local_row"])
    post = {}
    for h in heads:
        z = np.load(RC.r1_path(f"results/rb_heads_{h}.npz"))
        if not np.array_equal(z["filled_by_half"], 1 - half_of):
            raise AssertionError(f"rb_heads_{h}: rows are not scored by the other half's heads")
        post[h] = {"img": z["img"], "txt": z["txt"]}
    rz = np.load(RC.r1_path(f"results/rb_reader_{config}.npz"))
    diag = __import__("json").loads(RC.r1_path(f"results/rb_diag_{config}.json").read_text())
    rec = {"config": config, "groupings": list(parts), "seeds": list(SEEDS), "halves": {}}
    npz = {}
    bankX = {k: {} for k in (1, 2, 3, 4)}
    bankF = {k: {} for k in (1, 2, 3, 4)}
    for j in (0, 1):
        bz = np.load(RC.r1_path(f"results/rb_bank_{config}_half{j}.npz"))
        bank = {f: np.asarray(bz[f]) for f in ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img",
                                               "pairs_b_txt")}
        for f, a in bank.items():
            if a.dtype != np.int64:
                raise AssertionError(f"bank {f} is {a.dtype}")
        blocks = [tuple(s.split("__")) for s in bz["block_pairs"].tolist()]
        if blocks != BLOCKS[config] or int(bz["block_size"]) != 16_384 or int(bz["half"]) != j:
            raise AssertionError(f"bank half {j}: blocks / size / half differ from the rule")
        N = len(bank["anchor"])
        R = np.asarray(hz[f"local_rows_half{j}"], np.int64)
        if N != N_BANK[config] or len(R) != HALF_ROWS[j] or not (np.diff(R) > 0).all():
            raise AssertionError(f"half {j}: N {N} or |R| {len(R)} differ from the rule")
        order, img, cap, info = replacement_draws(N, bank["anchor"], R, paint, SEEDS[j])
        for nm, a in (("order", order), ("img", img), ("cap", cap)):
            if a.dtype != np.int64 or a.shape != (N, 2, 4):
                raise AssertionError(f"{nm} is {a.dtype} {a.shape}")
        # validity of the draws
        pA = paint[bank["anchor"]][:, None, None]
        valid = {"img_rows_in_half": bool(np.isin(img, R).all()), "cap_rows_in_half": bool(np.isin(cap, R).all()),
                 "img_not_anchor_painting": bool((paint[img] != pA).all()),
                 "cap_not_anchor_painting": bool((paint[cap] != pA).all()),
                 "cap_painting_differs_from_img_painting": bool((paint[cap] != paint[img]).all()),
                 "order_rows_are_permutations": bool((np.sort(order, axis=2) == np.arange(4)).all())}
        ep_rows = np.concatenate([bank["candidates"], bank["pairs_a_img"], bank["pairs_a_txt"], bank["pairs_b_img"],
                                  bank["pairs_b_txt"]], axis=1)
        ep_paint = paint[ep_rows]                                            # (N, 29)
        slot_img = paint[img].reshape(N, 8)
        slot_cap = paint[cap].reshape(N, 8)
        reuse = ((slot_img[:, :, None] == ep_paint[:, None, :]).any(2) | (slot_cap[:, :, None] == ep_paint[:, None, :]).any(2))
        hrec = {"N": N, "n_rows_R": int(len(R)), "draw_info": info, "draw_validity": valid,
                "sha256": {"order": K.sha_arr(order, np.int64), "img": K.sha_arr(img, np.int64),
                           "cap": K.sha_arr(cap, np.int64)},
                "diag_slot_reuses_original_episode_painting_pct_all8": 100 * float(reuse.mean()),
                "diag_slot_reuses_original_episode_painting_pct_purity1_slots": None, "purity": {}}
        if not all(valid.values()):
            raise AssertionError(f"half {j}: draws invalid {valid}")
        rank = np.empty_like(order)
        rank[np.arange(N)[:, None, None], np.arange(2)[None, :, None], order] = np.arange(4)[None, None, :]
        rep1 = (rank < 3).reshape(N, 8)
        hrec["diag_slot_reuses_original_episode_painting_pct_purity1_slots"] = 100 * float(reuse[rep1].mean())
        ya, yb = bank_labels(blocks, 16_384, list(parts))
        prev = None
        for k in (4, 3, 2, 1):
            pai, pat, pbi, pbt = impure_pairs(bank, order, img, cap, k)
            F = K.features_both(post, parts, pai, pat, pbi, pbt)
            Fr = RC.rf.both_conditions(post, parts, SimpleNamespace(pairs_a_img=pai, pairs_a_txt=pat, pairs_b_img=pbi,
                                                                    pairs_b_txt=pbt))
            eq_rf = all(np.array_equal(F[c], Fr[c]) for c in COND)
            if not eq_rf:
                raise AssertionError(f"half {j} purity {k}: own features differ from rb_features.both_conditions")
            X = np.vstack([F["a"], F["b"]])
            if not np.isfinite(X).all():
                raise AssertionError("non-finite bank features")
            n_rep = {"side_a_positions_replaced_per_episode": float((pai != bank["pairs_a_img"]).sum(1).mean()),
                     "pairs_changed_vs_round1_a": int(((pai != bank["pairs_a_img"]) | (pat != bank["pairs_a_txt"])).sum()),
                     "pairs_changed_vs_round1_b": int(((pbi != bank["pairs_b_img"]) | (pbt != bank["pairs_b_txt"])).sum())}
            nested_ok = None
            if prev is not None:     # purity k keeps a subset of purity (k+1)'s original pairs; replacements identical
                p_pai, p_pat, p_pbi, p_pbt = prev
                ch = (p_pai != bank["pairs_a_img"]) | (p_pat != bank["pairs_a_txt"])
                nested_ok = bool((pai[ch] == p_pai[ch]).all() and (pat[ch] == p_pat[ch]).all())
            prec = {"sha256_X": K.sha_arr(X, np.float64), "sha256_F_a": K.sha_arr(F["a"], np.float64),
                    "sha256_F_b": K.sha_arr(F["b"], np.float64),
                    "sha256_pairs": {nm: K.sha_arr(a, np.int64) for nm, a in
                                     (("pairs_a_img", pai), ("pairs_a_txt", pat), ("pairs_b_img", pbi), ("pairs_b_txt", pbt))},
                    "own_features_equal_rb_features_both_conditions": bool(eq_rf), "replacement_counts": n_rep,
                    "nested_with_next_purer": nested_ok,
                    "mean_abs_delta": {h: float(np.mean(np.abs(X[:, 6 * i + 2]))) for i, h in enumerate(parts)}}
            if k == 4:
                prec["equals_round1_stored_X"] = bool(np.array_equal(X, rz[f"half{j}__X"]))
                prec["round1_labels_equal_own"] = bool(np.array_equal(rz[f"half{j}__y"], np.concatenate([ya, yb])))
                if not (prec["equals_round1_stored_X"] and prec["round1_labels_equal_own"]):
                    raise SystemExit(f"half {j}: purity-4 check failed")
            hrec["purity"][str(k)] = prec
            bankX[k][j] = X
            bankF[k][j] = F
            prev = (pai, pat, pbi, pbt)
            K.log(f"{config} half {j} purity {k}: features done", t0)
        hrec["labels"] = {"y_a_counts": np.bincount(ya, minlength=H).tolist(), "y_b_counts": np.bincount(yb, minlength=H).tolist()}
        rec["halves"][str(j)] = hrec
        npz[f"half{j}__order"], npz[f"half{j}__img"], npz[f"half{j}__cap"] = order, img, cap
        npz[f"half{j}__ya"], npz[f"half{j}__yb"] = ya, yb
    # SMD, D(k), k*
    table = {}
    for k in (1, 2, 3, 4):
        Xb = np.vstack([bankX[k][0], bankX[k][1]])
        s = smd(X42, Xb)
        if not np.isfinite(s).all():
            raise SystemExit(f"purity {k}: non-finite SMD {dict(zip(names, s.tolist()))}")
        table[k] = {"D": float(np.mean(np.abs(s))), "smd": dict(zip(names, s.tolist())), "n_bank_rows": int(len(Xb)),
                    "D_fsum": math.fsum(np.abs(s)) / len(s)}
    stored = diag["c_shift_report"]["smd"]
    smd4_equal = all(table[4]["smd"][n] == stored[n] for n in names)
    if not smd4_equal:
        raise SystemExit("purity-4 SMDs differ from round 1's shift report")
    kstar = None
    for k in (4, 3, 2, 1):
        if kstar is None or table[k]["D"] < table[kstar]["D"]:
            kstar = k
    rec["D_table"] = {str(k): table[k] for k in (1, 2, 3, 4)}
    rec["purity4_smd_equal_round1_shift_report"] = bool(smd4_equal)
    rec["k_star"] = int(kstar)
    rec["mean_abs_delta_seed42"] = {h: float(np.mean(np.abs(X42[:, 6 * i + 2]))) for i, h in enumerate(parts)}
    rec["n_seed42_rows"] = int(len(X42))
    K.log(f"{config} D(k): {[(k, table[k]['D']) for k in (1, 2, 3, 4)]}; k* = {kstar}", t0)
    np.savez(K.OUT / f"rd2_r3_{config}_bankfeat.npz",
             **{f"k{k}__half{j}__X": bankX[k][j] for k in (1, 2, 3, 4) for j in (0, 1)})
    # training at k*
    if kstar == 4:
        rec["training"] = "k* = 4: R3 is R1 (no retraining; rule §4.4 item h)"
    elif train:
        readers, oofs = [], []
        rec["training"] = {}
        for j in (0, 1):
            ya, yb = npz[f"half{j}__ya"], npz[f"half{j}__yb"]
            y = np.concatenate([ya, yb])
            Xk = bankX[kstar][j]
            N = len(ya)
            sc, m, trec, oof = fit_half(Xk, y, N, H, t0, f"{config} half {j}")
            readers.append((sc, m))
            oofs.append(oof)
            trec["oof_top_probability"] = distribution(oof.max(axis=1))
            rec["training"][str(j)] = trec
            npz[f"half{j}__coef"], npz[f"half{j}__intercept"] = m.coef_, m.intercept_
            npz[f"half{j}__scaler_mean"], npz[f"half{j}__scaler_scale"] = sc.mean_, sc.scale_
            npz[f"half{j}__oof"] = oof
        halves = {c: [K.half_probs_scaled(m, sc.transform(F42[c])) for sc, m in readers] for c in COND}
        P = {c: K.mean_two(*halves[c]) for c in COND}
        pm = {c: K.picks_margins(P[c]) for c in COND}
        taus = K.thresholds(pm["a"][1], pm["b"][1])
        rec["chosen_C"] = {str(j): rec["training"][str(j)]["chosen_C"] for j in (0, 1)}
        rec["taus"] = taus
        rec["top_probability"] = {"bank_oof_pooled_halves": distribution(np.concatenate([o.max(1) for o in oofs])),
                                  "seed42": distribution(np.concatenate([P[c].max(1) for c in COND]))}
        # pick share vs R1 (label-free diagnostic)
        pk = K.load_reader_pickle(config)
        P1 = {c: K.mean_two(*[K.half_probs_scaled(h["model"], h["scaler"].transform(F42[c])) for h in pk["halves"]])
              for c in COND}
        rec["pick_differs_from_R1_share_pct"] = 100 * float(np.concatenate(
            [pm[c][0] != K.picks_margins(P1[c])[0] for c in COND]).mean())
        rec["pick_share_pct"] = {h: 100 * float(np.mean(np.concatenate([pm[c][0] for c in COND]) == i))
                                 for i, h in enumerate(parts)}
        for c in COND:
            npz[f"P__{c}"] = P[c]
            npz[f"P_half0__{c}"], npz[f"P_half1__{c}"] = halves[c]
            npz[f"pick__{c}"], npz[f"margin__{c}"] = pm[c]
        npz["taus"] = np.asarray(taus)
        K.log(f"{config} R3 trained at k* = {kstar}: C {rec['chosen_C']}, taus {taus}", t0)
    else:
        rec["training"] = "skipped (--no-train)"
    rec["runtime_s"] = round(time.time() - t0, 1)
    rec["provenance"] = K.provenance()
    K.save_json(K.OUT / f"rd2_r3_{config}.json", rec)
    np.savez(K.OUT / f"rd2_r3_{config}.npz", **npz)
    K.log(f"{config} done", t0)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A0", train="--no-train" not in sys.argv)
