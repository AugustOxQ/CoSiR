"""Phase 1, step 1: refit every head (rule §6 item 2) with the allowed fitters, check them bit for bit against the
stored selection posteriors, record coefficient hashes, and cache the refit selection-row posteriors for rd_seed42.py.

  CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
  /root/miniconda3/envs/CoSiR/bin/python src/test/20261125_artelingo_held_test/rederive/rd_heads.py
"""
import hashlib
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rd_common as R  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

sys.path.insert(0, str(R.T / "20261108_new_method_quick_checks"))
sys.path.insert(0, str(R.T / "20261111_community_told_oracle"))
import run_n6 as n6  # noqa: E402
import run_told_oracle as rto  # noqa: E402

HEAD_ROWS = 60_000


class RecLR(LogisticRegression):
    """LogisticRegression that keeps every fitted instance, so the coefficients can be hashed (numerics unchanged)."""
    fitted = []

    def fit(self, X, y, sample_weight=None):
        out = super().fit(X, y, sample_weight)
        RecLR.fitted.append(self)
        return out


def coef_hashes(clf):
    c, b = np.ascontiguousarray(clf.coef_), np.ascontiguousarray(clf.intercept_)
    h2 = hashlib.sha256(c.tobytes())
    h2.update(b.tobytes())
    return {"coef": R.sha_bytes(c), "coef_then_intercept": h2.hexdigest(),
            "coef_intercept_concat_ravel": R.sha_bytes(np.concatenate([c.ravel(), b.ravel()])),
            "coef_intercept_hstack": R.sha_bytes(np.hstack([c, b[:, None]])),
            "coef_f32": R.sha_bytes(c.astype(np.float32)), "shape": list(c.shape), "n_iter": int(clf.n_iter_[0])}


def global_labels(local, scorer_train, n):
    lab = np.full(n, -1, dtype=np.int64)
    lab[scorer_train] = np.asarray(local, dtype=np.int64)
    return lab


def fit(kind, fn):
    start = len(RecLR.fitted)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        res = fn()
    n_warn = int(sum(issubclass(w.category, ConvergenceWarning) for w in caught))
    return res, RecLR.fitted[start:], n_warn


def main():
    n6.LogisticRegression = RecLR
    rto.LogisticRegression = RecLR
    data, sp, labels = R.load_data()
    split_ok = R.check_split(sp)
    ctx = R.SelCtx(data, sp)
    st = np.asarray(sp.scorer_train)
    n = len(ctx.groups)
    sel = ctx.selection
    R.log("data loaded")
    rec = {"split_vs_prepare_npz": split_ok, "head_rows": HEAD_ROWS, "fits": {}, "bitwise": {}, "prov": {},
           "convergence_warnings": {}}
    post = {}

    # affect (D1's Leiden partition L), fit_one_head; identity with told_oracle.json arm L's head
    pl = np.load(R.need(R.T / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz"))["partition_L"]
    (p, prov), clfs, w = fit("affect", lambda: rto.fit_one_head(ctx, global_labels(pl, st, n), st, HEAD_ROWS))
    told = json.loads(R.need(R.T / "20261111_community_told_oracle/results/told_oracle.json").read_text())
    rec["affect_identity"] = json.loads(json.dumps(prov)) == told["arms"]["L"]["head"]
    post["affect"], rec["prov"]["affect"], rec["convergence_warnings"]["affect"] = p, prov, w
    rec["fits"]["affect"] = {m: coef_hashes(c) for m, c in zip(("img", "txt"), clfs)}
    R.log(f"affect heads: identity {rec['affect_identity']}")

    # E2's affect-km, image, caption with run_n6.fit_heads (the same 60,000-row draw)
    z = np.load(R.need(R.T / "20261031_pseudo_partitions/results/partitions.npz"))
    if not np.array_equal(z["local_groups"], np.unique(ctx.groups[st], return_inverse=True)[1]):
        raise AssertionError("E2 partitions not aligned with scorer_train")
    labs = {h: global_labels(z[h], st, n) for h in ("affect", "image", "caption")}
    (pe, prov_e), clfs, w = fit("e2", lambda: n6.fit_heads(ctx, labs, st, HEAD_ROWS))
    names = {"affect": "affect_km", "image": "image", "caption": "caption"}
    for i, h in enumerate(("affect", "image", "caption")):
        post[names[h]] = pe[h]
        rec["fits"][names[h]] = {m: coef_hashes(c) for m, c in zip(("img", "txt"), clfs[2 * i:2 * i + 2])}
    rec["prov"]["e2"], rec["convergence_warnings"]["e2"] = prov_e, w
    stored = np.load(R.need(R.T / "20261108_new_method_quick_checks/results/n6_posteriors.npz"))
    if not np.array_equal(stored["selection"], sel):
        raise AssertionError("n6_posteriors selection differs")
    for h in ("affect", "image", "caption"):
        for m in ("img", "txt"):
            a, b = pe[h][m][sel].astype(np.float32), stored[f"{h}__{m}"]
            rec["bitwise"][f"n6_posteriors/{h}__{m}"] = {
                "equal": bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()),
                "n_diff": int((a != b).sum()), "max_abs_diff": float(np.abs(a.astype(np.float64) - b).max())}
    R.log("E2 heads done")

    # csd: fit_one_head on CLIP B/32 with step 1's style_csd labels
    g = np.load(R.need(R.T / "20261116_grouping_step1_style/results/step1_group_style.npz"))
    if not np.array_equal(g["scorer_train"], st):
        raise AssertionError("step 1 scorer_train differs")
    (pc, prov_c), clfs, w = fit("csd", lambda: rto.fit_one_head(ctx, global_labels(g["style_csd"], st, n), st,
                                                               HEAD_ROWS))
    post["csd"], rec["prov"]["csd"], rec["convergence_warnings"]["csd"] = pc, prov_c, w
    rec["fits"]["csd"] = {m: coef_hashes(c) for m, c in zip(("img", "txt"), clfs)}
    hs = np.load(R.need(R.T / "20261116_grouping_step1_style/results/step1_heads_style.npz"))
    if not np.array_equal(hs["selection"], sel):
        raise AssertionError("step1_heads_style selection differs")
    for m in ("img", "txt"):
        a, b = pc[m][sel].astype(np.float32), hs[f"style_csd__{m}"]
        rec["bitwise"][f"step1_heads_style/style_csd__{m}"] = {
            "equal": bool(a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()),
            "n_diff": int((a != b).sum()), "max_abs_diff": float(np.abs(a.astype(np.float64) - b).max())}
    R.log("csd heads done")

    rec["all_bitwise_equal"] = all(v["equal"] for v in rec["bitwise"].values())
    np.savez(R.OUT / "rd_post.npz", selection=sel,
             **{f"{g_}__{m}": post[g_][m][sel] for g_ in post for m in ("img", "txt")})
    rec["post_npz_sha256"] = R.sha_file(R.OUT / "rd_post.npz")
    rec["time"] = R.now_ams()
    R.write_json(R.RD / "rd_heads.json", rec)
    R.log(f"bitwise equal {rec['all_bitwise_equal']}; affect identity {rec['affect_identity']}")


if __name__ == "__main__":
    main()
