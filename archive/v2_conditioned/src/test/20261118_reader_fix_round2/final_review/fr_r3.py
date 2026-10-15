"""Final review of reader-fix round 2: R3 from scratch with this review's own code (DECISION_RULE.md section 4.4):
replacement draws, impure banks of purity 1 to 4, their features from the cross-fitted posteriors, the purity-4 checks,
SMD, D(k), k*, then the half-readers retrained at k* with this review's own cross-validation loop, and R3's seed-42
probabilities. Compares with results/r3_k_A0.json and results/probs_R3_A0.{json,npz} (read only). Writes
out/fr_r3.json and out/fr_r3_probs.npz.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261118_reader_fix_round2/final_review/fr_r3.py
"""
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
R1RES = ROOT / "src/test/20261117_reader_fix_csd/results"
RES = HERE.parent / "results"
OUT = HERE / "out"
PARTS = ("affect", "image", "caption")
BLOCKS = [("affect", "caption"), ("affect", "image"), ("caption", "image")]
SEEDS = (21700, 21800)
CGRID = (0.01, 0.1, 1.0, 10.0, 100.0)
ROWS = []


def row(name, reported, mine, ok=None):
    ok = (reported == mine) if ok is None else ok
    ROWS.append({"quantity": name, "reported": reported, "rederived": mine, "agree": bool(ok)})
    if not ok:
        print(f"  DISAGREE {name}: {reported!r} vs {mine!r}", flush=True)


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def draws(A, Rw, paint, seed):
    N = len(A)
    g = np.random.default_rng(seed)
    U = g.random((N, 2, 4))
    order = np.argsort(U, axis=2, kind="stable")
    pa = paint[A][:, None, None]
    img = Rw[g.integers(0, len(Rw), size=(N, 2, 4))]
    while True:
        bad = paint[img] == pa
        if not bad.any():
            break
        img[bad] = Rw[g.integers(0, len(Rw), size=bad.sum())]
    cap = Rw[g.integers(0, len(Rw), size=(N, 2, 4))]
    while True:
        bad = (paint[cap] == pa) | (paint[cap] == paint[img])
        if not bad.any():
            break
        cap[bad] = Rw[g.integers(0, len(Rw), size=bad.sum())]
    return order.astype(np.int64), img.astype(np.int64), cap.astype(np.int64)


def impure(bank, order, img, cap, k):
    """Replace, per episode and side, the pairs at positions order[n, s, :4-k]."""
    out = {}
    N = len(order)
    for s, (ki, kt) in enumerate((("pairs_a_img", "pairs_a_txt"), ("pairs_b_img", "pairs_b_txt"))):
        I, Tt = bank[ki].astype(np.int64).copy(), bank[kt].astype(np.int64).copy()
        for q in range(4 - k):
            p = order[:, s, q]
            I[np.arange(N), p] = img[np.arange(N), s, p]
            Tt[np.arange(N), p] = cap[np.arange(N), s, p]
        out[ki], out[kt] = I, Tt
    return out


def feats(post, si, st, ci, ct):
    cols = []
    for h in PARTS:
        PI, PT = post[h]["img"], post[h]["txt"]
        sup = np.einsum("nsc,nsc->ns", PI[si], PT[st])
        con = np.einsum("nsc,nsc->ns", PI[ci], PT[ct])
        S, Cc = sup.mean(axis=1), con.mean(axis=1)
        match = (PI[si].argmax(-1) == PT[st].argmax(-1)).astype(np.float64).mean(axis=1)
        cols += [S.astype(np.float64), Cc.astype(np.float64), (S - Cc).astype(np.float64),
                 sup.astype(np.float64).std(axis=1, ddof=1), con.astype(np.float64).std(axis=1, ddof=1), match]
    return np.stack(cols, axis=1)


def bank_X(post, pr):
    Xa = feats(post, pr["pairs_a_img"], pr["pairs_a_txt"], pr["pairs_b_img"], pr["pairs_b_txt"])
    Xb = feats(post, pr["pairs_b_img"], pr["pairs_b_txt"], pr["pairs_a_img"], pr["pairs_a_txt"])
    return np.vstack([Xa, Xb])


def smd(x, y):
    d = x.mean(0) - y.mean(0)
    s = np.sqrt((x.var(0, ddof=1) + y.var(0, ddof=1)) / 2)
    return d / s


def fit_half(X, y, N):
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    fold_ep = np.full(N, -1)
    for f, (_, te) in enumerate(KFold(5, shuffle=True, random_state=0).split(np.arange(N))):
        fold_ep[te] = f
    fold = np.concatenate([fold_ep, fold_ep])
    losses, oofs, warn = [], [], []
    for C_ in CGRID:
        oof = np.zeros((len(y), 3))
        ls = []
        w = 0
        for f in range(5):
            tr, te = fold != f, fold == f
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                m = LogisticRegression(C=C_, solver="lbfgs", max_iter=2000).fit(Xs[tr], y[tr])
            w += sum(issubclass(c.category, ConvergenceWarning) for c in caught)
            oof[te] = m.predict_proba(Xs[te])
            ls.append(log_loss(y[te], oof[te], labels=[0, 1, 2]))
        losses.append(float(np.mean(ls)))
        oofs.append(oof)
        warn.append(int(w))
    best = 0
    for i in range(1, len(CGRID)):
        if losses[i] < losses[best]:
            best = i
    m = LogisticRegression(C=CGRID[best], solver="lbfgs", max_iter=2000).fit(Xs, y)
    oof = oofs[best]
    return sc, m, {"chosen_C": CGRID[best], "losses": losses, "warnings": warn,
                   "oof_acc": 100 * float(np.mean(oof.argmax(1) == y)), "oof_top": oof.max(1)}


def main():
    t0 = time.time()
    H = np.load(R1RES / "rb_halves.npz")
    paint, half_of = H["painting_of_local_row"], H["half_of_local_row"]
    post = {}
    for h in PARTS:
        z = np.load(R1RES / f"rb_heads_{h}.npz")
        post[h] = {"img": z["img"], "txt": z["txt"]}
    rz = np.load(R1RES / "rb_reader_A0.npz")
    cache = np.load(OUT / "fr_cache.npz")
    x42 = np.vstack([cache["F_A0__a"], cache["F_A0__b"]])
    stored_k = json.loads((RES / "r3_k_A0.json").read_text())
    Xk = {k: [] for k in (1, 2, 3, 4)}
    labels = []
    for j in (0, 1):
        b = np.load(R1RES / f"rb_bank_A0_half{j}.npz")
        bank = {k: b[k] for k in b.files}
        assert [tuple(s.split("__")) for s in bank["block_pairs"].tolist()] == BLOCKS
        N = len(bank["anchor"])
        Rw = H[f"local_rows_half{j}"]
        order, img, cap = draws(bank["anchor"], Rw, paint, SEEDS[j])
        assert (half_of[img] == j).all() and (half_of[cap] == j).all()
        ss = stored_k["sha256"][f"half{j}"]
        row(f"R3 half {j}: SHA-256 of order/img/cap", [ss["order"], ss["img"], ss["cap"]],
            [sha_arr(order), sha_arr(img), sha_arr(cap)])
        # nesting: a pair replaced at purity k is replaced, by the same pair, at every smaller k
        prev = None
        for k in (4, 3, 2, 1):
            pr = impure(bank, order, img, cap, k)
            if prev is not None:
                for key in pr:
                    changed = prev[key] != pr[key]
                    assert (pr[key][prev[key] != bank[key]] == prev[key][prev[key] != bank[key]]).all()
            prev = pr
            X = bank_X(post, pr)
            row(f"R3 half {j}: SHA-256 of purity-{k} features", ss[f"features_purity{k}"], sha_arr(X))
            Xk[k].append(X)
            if k == 4:
                row(f"R3 half {j}: purity-4 features == round 1's half{j}__X", True,
                    bool(np.array_equal(X, rz[f"half{j}__X"])))
            # replaced share per side = (4 - k) / 4 exactly
            for key in ("pairs_a_img", "pairs_b_txt"):
                frac = float(np.mean(pr[key] != bank[key]))
                assert frac <= (4 - k) / 4 + 1e-12
        n_blk = int(bank["block_size"])
        ya = np.concatenate([np.full(n_blk, PARTS.index(a)) for a, _ in BLOCKS])
        yb = np.concatenate([np.full(n_blk, PARTS.index(bb)) for _, bb in BLOCKS])
        labels.append(np.concatenate([ya, yb]))
        row(f"R3 half {j}: labels == round 1's half{j}__y", True, bool(np.array_equal(labels[-1], rz[f"half{j}__y"])))
        print(f"half {j} banks done [{time.time() - t0:.0f}s]", flush=True)
    D, S = {}, {}
    for k in (1, 2, 3, 4):
        S[k] = smd(x42, np.vstack(Xk[k]))
        D[k] = float(np.mean(np.abs(S[k])))
    row("R3: D(1..4)", [stored_k["D"][str(k)] for k in (1, 2, 3, 4)], [D[k] for k in (1, 2, 3, 4)])
    smd_ok = all(list(stored_k["smd"][str(k)].values()) == S[k].tolist() for k in (1, 2, 3, 4))
    row("R3: all 72 SMDs bit-identical", True, bool(smd_ok))
    diag = json.loads((R1RES / "rb_diag_A0.json").read_text())["c_shift_report"]["smd"]
    row("R3: purity-4 SMDs == round 1's shift report", True, list(diag.values()) == S[4].tolist())
    kstar = min((1, 2, 3, 4), key=lambda k: (D[k], -k))
    row("R3: k*", stored_k["k_star"], kstar)
    mad = {"seed42": {h: float(np.mean(np.abs(x42[:, 6 * i + 2]))) for i, h in enumerate(PARTS)}}
    for k in (1, 2, 3, 4):
        Xall = np.vstack(Xk[k])
        mad[k] = {h: float(np.mean(np.abs(Xall[:, 6 * i + 2]))) for i, h in enumerate(PARTS)}
    row("R3: mean |Delta| seed 42 and banks", stored_k["mean_abs_delta"]["seed42"], mad["seed42"])
    # training at k*
    stored_p = json.loads((RES / "probs_R3_A0.json").read_text())
    halves, recs, oof_top = [], {}, []
    for j in (0, 1):
        X, y = Xk[kstar][j], labels[j]
        sc, m, r = fit_half(X, y, len(y) // 2)
        halves.append((sc, m))
        oof_top.append(r.pop("oof_top"))
        recs[j] = r
        sr = stored_p["half_readers"][str(j)]
        row(f"R3 half {j}: chosen C", sr["chosen_C"], r["chosen_C"])
        row(f"R3 half {j}: out-of-fold accuracy at chosen C", sr["oof_accuracy_at_chosen_C"], r["oof_acc"])
        row(f"R3 half {j}: CV mean log losses", [sr["cv_mean_log_loss"][str(c)] for c in CGRID], r["losses"])
        print(f"half {j}: C {r['chosen_C']}, oof {r['oof_acc']:.2f} [{time.time() - t0:.0f}s]", flush=True)
    P = {}
    for c in ("a", "b"):
        F = cache[f"F_A0__{c}"]
        P[c] = (halves[0][1].predict_proba(halves[0][0].transform(F)) + halves[1][1].predict_proba(halves[1][0].transform(F))) / 2
    zp = np.load(RES / "probs_R3_A0.npz")
    for c in ("a", "b"):
        row(f"R3: P__{c} bit-identical to stored", True, bool(np.array_equal(P[c], zp[f"P__{c}"])))
        row(f"R3: P__{c} max abs diff to stored", 0.0, float(np.max(np.abs(P[c] - zp[f"P__{c}"]))),
            ok=float(np.max(np.abs(P[c] - zp[f"P__{c}"]))) < 1e-9)
    agree = {c: 100 * float(np.mean(halves[0][1].predict_proba(halves[0][0].transform(cache[f"F_A0__{c}"])).argmax(1)
                                     == halves[1][1].predict_proba(halves[1][0].transform(cache[f"F_A0__{c}"])).argmax(1)))
             for c in ("a", "b")}
    top42 = float(np.mean(np.concatenate([P["a"].max(1), P["b"].max(1)])))
    topbank = float(np.mean(np.concatenate(oof_top)))
    np.savez(OUT / "fr_r3_probs.npz", P__a=P["a"], P__b=P["b"])
    gaps = {j: sorted(recs[j]["losses"])[1] - sorted(recs[j]["losses"])[0] for j in (0, 1)}
    res = {"D": D, "k_star": kstar, "mean_abs_delta": mad, "half_readers": recs, "loss_gap_best_two": gaps,
           "half_reader_pick_agreement": agree, "top_prob_seed42": top42, "top_prob_bank_oof": topbank,
           "runtime_s": round(time.time() - t0, 1)}
    (OUT / "fr_r3.json").write_text(json.dumps({"results": res, "rows": ROWS}, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k != "half_readers"}, indent=1, default=float))
    print(f"{len(ROWS)} comparisons, {sum(not r['agree'] for r in ROWS)} disagreements", flush=True)


if __name__ == "__main__":
    main()
