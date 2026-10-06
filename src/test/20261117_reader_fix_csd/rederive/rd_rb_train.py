"""Phase 2b: spot-check R-b training (rule 4.2 items 1 to 6) against the stored outputs, with our own code.

(1) halves recomputed from the permutation; (2) block layout of every bank (on every episode) and the labels from block
positions; (3) features of 2,000 random bank episodes per half recomputed from the stored cross-fitted posteriors and
the stored bank episodes, and whether bank half j's rows carry posteriors of the heads of half 1 - j; one csd image
head refit per half with the fit_one_head recipe; (4) the CV choice of C from the stored CV table, the fold assignment,
the out-of-fold log losses from the stored OOF probabilities, a refit of each half-reader at its chosen C, and the A1
half-1 near tie (C 10 vs 100) recomputed by a full 5-fold CV; (5) optional full rebuild of one bank.
Writes out/rd_rb_train.json.
"""
import json
import sys
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import rd_core as K
from src.data.artelingo import load_artelingo
from src.eval.aspect_episodes import AspectEpisodes
from src.train.pseudo_partitions import build_episode_bank

BLOCKS = {"A1": [("affect", "caption"), ("affect", "csd"), ("affect", "image"), ("caption", "csd"), ("caption", "image"),
                 ("csd", "image")],
          "AR": [("affect", "caption"), ("affect", "image"), ("affect", "rand"), ("caption", "image"), ("caption", "rand"),
                 ("image", "rand")],
          "A0": [("affect", "caption"), ("affect", "image"), ("caption", "image")]}
THIRD_A0 = {("affect", "caption"): "image", ("affect", "image"): "caption", ("caption", "image"): "affect"}
NPP = 16384
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def sub_episodes(z, idx):
    return AspectEpisodes("x", "y", *(np.asarray(z[k])[idx] for k in
                                      ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")))


def main(rebuild):
    t0 = time.time()
    out = {}
    data = load_artelingo()
    from src.data.artelingo_splits import artelingo_splits
    sp = artelingo_splits(data)
    st = np.asarray(sp.scorer_train)
    groups = np.asarray(sp.groups)
    hz = np.load(K.RES / "rb_halves.npz")

    # ------------------------------------------------ (1) halves
    pid = groups[st]
    paintings = np.unique(pid)
    perm = np.random.default_rng(0).permutation(paintings)
    h0, h1 = perm[:18259], perm[18259:]
    half_of_local = np.where(np.isin(pid, h1), 1, 0).astype(np.int8)
    loc = {0: np.flatnonzero(half_of_local == 0), 1: np.flatnonzero(half_of_local == 1)}
    out["halves"] = {
        "n_paintings": int(len(paintings)), "sizes": [int(len(h0)), int(len(h1))],
        "rows": [int(len(loc[0])), int(len(loc[1]))],
        "scorer_train_equal": bool(np.array_equal(st, hz["scorer_train"])),
        "paintings_sorted_equal": bool(np.array_equal(paintings, hz["paintings_sorted"])),
        "half0_permuted_equal": bool(np.array_equal(h0, hz["paintings_half0"])),
        "half1_permuted_equal": bool(np.array_equal(h1, hz["paintings_half1"])),
        "half_of_local_row_equal": bool(np.array_equal(half_of_local, hz["half_of_local_row"])),
        "local_rows_equal": [bool(np.array_equal(loc[j], hz[f"local_rows_half{j}"])) for j in (0, 1)],
        "global_rows_equal": [bool(np.array_equal(st[loc[j]], hz[f"global_rows_half{j}"])) for j in (0, 1)]}
    print(f"[{time.time() - t0:.0f}s] halves {out['halves']}", flush=True)

    # local labels of the five groupings (scorer-train positions)
    told = np.load(K.TOLD_NPZ)
    e2 = np.load(K.ROOT / "src/test/20261031_pseudo_partitions/results/partitions.npz")
    g1 = np.load(K.ROOT / "src/test/20261116_grouping_step1_style/results/step1_group_style.npz")
    if not np.array_equal(g1["scorer_train"], st):
        raise AssertionError("step1_group_style scorer_train differs")
    labels = {"affect": np.asarray(told["partition_L"], np.int64), "image": e2["image"], "caption": e2["caption"],
              "csd": g1["style_csd"], "rand": g1["style_rand"]}

    # ------------------------------------------------ (2)+(3) banks, labels, features
    heads = {h: np.load(K.RES / f"rb_heads_{h}.npz") for h in K.GROUPINGS}
    post_cf = {h: {"img": heads[h]["img"], "txt": heads[h]["txt"]} for h in K.GROUPINGS}
    filled_ok = {h: bool(np.array_equal(heads[h]["filled_by_half"], 1 - half_of_local)) for h in K.GROUPINGS}
    out["heads_filled_by_other_half"] = filled_ok
    out["banks"] = {}
    for cfg in ("A1", "A0", "AR"):
        g = K.CONFIGS[cfg]
        rz = np.load(K.RES / f"rb_reader_{cfg}.npz")
        for j in (0, 1):
            z = np.load(K.RES / f"rb_bank_{cfg}_half{j}.npz")
            n = len(z["anchor"])
            rec = {"n": int(n), "seed": int(z["seed"]), "seed_ok": int(z["seed"]) == (11700 if j == 0 else 11800),
                   "block_size_ok": int(z["block_size"]) == NPP,
                   "block_pairs_ok": [str(x) for x in z["block_pairs"]] == [f"{a}__{b}" for a, b in BLOCKS[cfg]],
                   "n_ok": n == NPP * len(BLOCKS[cfg])}
            allrows = np.concatenate([np.asarray(z[k]).ravel() for k in
                                      ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")])
            rec["all_rows_in_half"] = bool((half_of_local[allrows] == j).all())
            rec["all_rows_filled_by_other_half"] = all(bool((heads[h]["filled_by_half"][allrows] == 1 - j).all())
                                                       for h in g)
            lay = []
            ya = np.empty(n, np.int64)
            yb = np.empty(n, np.int64)
            for i, (fa, sb) in enumerate(BLOCKS[cfg]):
                s = slice(i * NPP, (i + 1) * NPP)
                Lf, Ls = labels[fa], labels[sb]
                an, ca = z["anchor"][s], z["candidates"][s]
                pai, pat, pbi, pbt = z["pairs_a_img"][s], z["pairs_a_txt"][s], z["pairs_b_img"][s], z["pairs_b_txt"][s]
                ok = ((Lf[pai] == Lf[pat]).all() and (Ls[pai] != Ls[pat]).all() and (Ls[pbi] == Ls[pbt]).all()
                      and (Lf[pbi] != Lf[pbt]).all() and (Lf[ca[:, 0]] == Lf[an]).all() and (Ls[ca[:, 1]] == Ls[an]).all()
                      and (Lf[ca[:, 2:]] != Lf[an][:, None]).all() and (Ls[ca[:, 2:]] != Ls[an][:, None]).all())
                if cfg == "A0":
                    Lt = labels[THIRD_A0[(fa, sb)]]
                    ok = ok and bool((Lt[ca] != Lt[an][:, None]).all())
                lay.append(bool(ok))
                ya[s] = g.index(fa)
                yb[s] = g.index(sb)
            rec["block_layout_ok_per_block"] = lay
            y, epi = rz[f"half{j}__y"], rz[f"half{j}__episode"]
            rec["labels_from_block_positions_equal"] = bool(np.array_equal(y, np.concatenate([ya, yb])))
            rec["episode_index_layout_ok"] = bool(np.array_equal(epi, np.concatenate([np.arange(n), np.arange(n)])))
            # features of 2,000 random episodes from the stored cross-fitted posteriors
            idx = np.sort(np.random.default_rng(1000 + j).choice(n, 2000, replace=False))
            ep = sub_episodes(z, idx)
            X = rz[f"half{j}__X"]
            fa_ = K.rb_features(post_cf, ep, g, "a")
            fb_ = K.rb_features(post_cf, ep, g, "b")
            fa32 = K.rb_features_f32(post_cf, ep, g, "a")
            fb32 = K.rb_features_f32(post_cf, ep, g, "b")
            rec["features_sample_max_abs_f64"] = float(max(np.abs(fa_ - X[idx]).max(), np.abs(fb_ - X[n + idx]).max()))
            rec["features_sample_max_abs_f32agree"] = float(max(np.abs(fa32 - X[idx]).max(),
                                                                np.abs(fb32 - X[n + idx]).max()))
            rec["features_sample_equal_f64"] = bool(np.array_equal(fa_, X[idx]) and np.array_equal(fb_, X[n + idx]))
            rec["features_sample_equal_f32agree"] = bool(np.array_equal(fa32, X[idx]) and np.array_equal(fb32, X[n + idx]))
            # the same features from the WRONG (own-half) heads would need posteriors we do not have; instead check
            # that the stored posteriors at these rows came from the other half (filled_by_half above)
            out["banks"][f"{cfg}_half{j}"] = rec
            print(f"[{time.time() - t0:.0f}s] bank {cfg} half {j}: {rec}", flush=True)

    # ------------------------------------------------ (3b) one cross-fitted csd image head refit per half
    feats = data.img_features
    lab_glob = np.full(len(groups), -1, np.int64)
    lab_glob[st] = labels["csd"]
    out["csd_img_head_refit"] = {}
    for k in (0, 1):
        rows_k = st[loc[k]]
        draw = np.random.default_rng(0).choice(rows_k, 60000, replace=False)
        clf = LogisticRegression(C=1.0, max_iter=300).fit(K.rc.unit(feats[draw]), lab_glob[draw])
        other = loc[1 - k]
        p = clf.predict_proba(K.rc.unit(feats[st[other]]))
        stored = heads["csd"]["img"][other]
        out["csd_img_head_refit"][f"head_of_half{k}_on_half{1 - k}"] = {
            "max_abs": float(np.abs(p - stored).max()), "equal": bool(np.array_equal(p.astype(np.float32), stored)),
            "draw_rows_sha256": K.rg.sha_array(np.sort(draw))}
        print(f"[{time.time() - t0:.0f}s] csd img head half {k}: {out['csd_img_head_refit']}", flush=True)
    hj = json.loads((K.RES / "rb_heads_csd.json").read_text())["heads"]
    for k in (0, 1):
        out["csd_img_head_refit"][f"head_of_half{k}_on_half{1 - k}"]["draw_sha_equal_stored"] = (
            out["csd_img_head_refit"][f"head_of_half{k}_on_half{1 - k}"]["draw_rows_sha256"] == hj[str(k)]["draw_rows_sha256"])

    # ------------------------------------------------ (4) CV choice of C, folds, OOF, refit
    out["readers"] = {}
    for cfg in ("A1", "A0", "AR"):
        rj = json.loads((K.RES / f"rb_reader_{cfg}.json").read_text())
        rz = np.load(K.RES / f"rb_reader_{cfg}.npz")
        rp = K.load_readers(cfg)
        for j in (0, 1):
            hrec = rj["halves"][str(j)]
            tab = hrec["cv_table"]
            means = [float(np.mean(np.asarray(r["fold_log_loss"], np.float64))) for r in tab]
            best = 0
            for i in range(1, len(means)):
                if means[i] < means[best]:
                    best = i
            sorted_means = sorted(means)
            X, y = rz[f"half{j}__X"], rz[f"half{j}__y"]
            n = X.shape[0] // 2
            folds = np.empty(n, np.int64)
            for f, (_, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(n))):
                folds[te] = f
            fold_rows = np.concatenate([folds, folds])
            oof = rz[f"half{j}__oof_proba"]
            oof_ll = [float(log_loss(y[fold_rows == f], oof[fold_rows == f], labels=list(range(len(K.CONFIGS[cfg])))))
                      for f in range(5)]
            sc = StandardScaler().fit(X)
            Cc = C_GRID[best]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                m = LogisticRegression(C=Cc, max_iter=2000).fit(sc.transform(X), y)
            stored_m, stored_s = rp["halves"][j]["model"], rp["halves"][j]["scaler"]
            rec = {"means_recomputed_equal_stored": [means[i] == tab[i]["mean_log_loss"] for i in range(len(tab))],
                   "means_max_abs_vs_stored": float(max(abs(means[i] - tab[i]["mean_log_loss"]) for i in range(len(tab)))),
                   "chosen_C_mine": Cc, "chosen_C_stored": hrec["chosen_C"], "pickle_C": float(stored_m.C),
                   "gap_best_to_second": sorted_means[1] - sorted_means[0],
                   "folds_equal_stored": bool(np.array_equal(folds, rz[f"half{j}__fold"])),
                   "oof_logloss_vs_stored_fold_ll_max_abs": float(max(abs(a - b) for a, b in
                                                                      zip(oof_ll, tab[best]["fold_log_loss"]))),
                   "oof_accuracy_mine": 100 * float(np.mean(oof.argmax(1) == y)),
                   "oof_accuracy_stored": hrec["oof_accuracy_at_chosen_C"],
                   "scaler_mean_equal": bool(np.array_equal(sc.mean_, stored_s.mean_)),
                   "scaler_scale_max_abs": float(np.abs(sc.scale_ - stored_s.scale_).max()),
                   "refit_coef_max_abs": float(np.abs(m.coef_ - stored_m.coef_).max()),
                   "refit_intercept_max_abs": float(np.abs(m.intercept_ - stored_m.intercept_).max()),
                   "refit_coef_equal": bool(np.array_equal(m.coef_, stored_m.coef_)),
                   "refit_warnings": int(sum(issubclass(w.category, ConvergenceWarning) for w in caught)),
                   "class_counts": np.bincount(y).tolist()}
            out["readers"][f"{cfg}_half{j}"] = rec
            print(f"[{time.time() - t0:.0f}s] reader {cfg} half {j}: {rec}", flush=True)

    # A1 half 1: C 10 vs 100 differ by ~1.7e-8 in mean log loss; recompute both by full 5-fold CV
    rz = np.load(K.RES / "rb_reader_A1.npz")
    X, y = rz["half1__X"], rz["half1__y"]
    n = X.shape[0] // 2
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    tie = {}
    for Cc in (1.0, 10.0, 100.0):
        lls = []
        for tr, te in KFold(5, shuffle=True, random_state=0).split(np.arange(n)):
            trr, ter = np.concatenate([tr, tr + n]), np.concatenate([te, te + n])
            m = LogisticRegression(C=Cc, max_iter=2000).fit(Xs[trr], y[trr])
            lls.append(float(log_loss(y[ter], m.predict_proba(Xs[ter]), labels=[0, 1, 2, 3])))
        tie[str(Cc)] = {"fold_ll": lls, "mean": float(np.mean(lls))}
    stored_tab = {str(r["C"]): r for r in json.loads((K.RES / "rb_reader_A1.json").read_text())["halves"]["1"]["cv_table"]}
    tie["vs_stored_max_abs"] = {c: float(max(abs(a - b) for a, b in zip(tie[c]["fold_ll"], stored_tab[c]["fold_log_loss"])))
                                for c in ("1.0", "10.0", "100.0")}
    tie["chosen_by_recomputed"] = min(("1.0", "10.0", "100.0"), key=lambda c: (tie[c]["mean"], float(c)))
    out["A1_half1_near_tie_recomputed"] = tie
    print(f"[{time.time() - t0:.0f}s] near tie {tie}", flush=True)

    # ------------------------------------------------ (5) optional full rebuild of one bank
    if rebuild:
        cfg, j = rebuild.split(":")
        j = int(j)
        g = K.CONFIGS[cfg]
        z = np.load(K.RES / f"rb_bank_{cfg}_half{j}.npz")
        bank = build_episode_bank({h: labels[h] for h in g}, pid, loc[j], n_per_pair=NPP,
                                  seed=11700 if j == 0 else 11800, min_paintings=30)
        out[f"rebuild_{cfg}_half{j}"] = {k: bool(np.array_equal(getattr(bank, k), z[k])) for k in
                                         ("anchor", "candidates", "pairs_a_img", "pairs_a_txt", "pairs_b_img", "pairs_b_txt")}
        print(f"[{time.time() - t0:.0f}s] rebuild {cfg} half {j}: {out[f'rebuild_{cfg}_half{j}']}", flush=True)
    out["runtime_s"] = round(time.time() - t0, 1)
    K.save_json(K.OUT / "rd_rb_train.json", out)
    print("done", out["runtime_s"], flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
