"""Final review of reader-fix round 2: load the seed-42 bundle once (round 1's common.load_bundle, the only shared code
used besides src/eval) and cache, computed with this review's own code, everything the re-derivation needs:
B, B', the step-1 arg-max reader's arrays, clusters, parity, pair index, the grouping scores s_h per direction and the
seed-42 reader features per configuration. Writes only final_review/out/fr_cache.npz.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261118_reader_fix_round2/final_review/fr_prep.py
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
R1DIR = ROOT / "src/test/20261117_reader_fix_csd"
for p in (str(ROOT), str(R1DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
import common as RC1  # noqa: E402  round 1's common.py (bundle loader only)

OUT = HERE / "out"
GROUPS = ("affect", "image", "caption", "csd")
METRICS = ("r1", "gain", "other", "swap", "strict")


def my_features(post, ep, parts):
    """18 (A0) or 24 (A1) features per (episode, condition), following rule section 4.1 literally:
    per grouping S, C, Delta, sd of the 4 support agreements (ddof 1), sd of the 4 contrast agreements, share of the 4
    support pairs whose image and caption arg-max groups coincide. Agreements in float32 (D3), sds in float64."""
    out = {}
    sets = {"a": (ep.pairs_a_img, ep.pairs_a_txt, ep.pairs_b_img, ep.pairs_b_txt),
            "b": (ep.pairs_b_img, ep.pairs_b_txt, ep.pairs_a_img, ep.pairs_a_txt)}
    for c, (si, st, ci, ct) in sets.items():
        cols = []
        for h in parts:
            PI, PT = post[h]["img"], post[h]["txt"]
            sup = np.einsum("nsc,nsc->ns", PI[si], PT[st])
            con = np.einsum("nsc,nsc->ns", PI[ci], PT[ct])
            S, Cc = sup.mean(axis=1), con.mean(axis=1)
            match = (PI[si].argmax(-1) == PT[st].argmax(-1)).astype(np.float64).mean(axis=1)
            cols += [S.astype(np.float64), Cc.astype(np.float64), (S - Cc).astype(np.float64),
                     sup.astype(np.float64).std(axis=1, ddof=1), con.astype(np.float64).std(axis=1, ddof=1), match]
        out[c] = np.stack(cols, axis=1)
    return out


def main():
    t0 = time.time()
    b = RC1.load_bundle(smoke=False)
    ctx, ep, post = b.ctx, b.ctx.pooled, b.post
    z = {"cl": np.asarray(b.cl), "parity": np.asarray(ctx.parity), "pair_index": np.asarray(ctx.pair_index)}
    for c in ("a", "b"):
        for d in ("i2t", "t2i"):
            z[f"B__{c}__{d}"] = np.asarray(b.B[c][d])
    for m in METRICS:
        z[f"pB__{m}"] = np.asarray(b.pB[m])
        for cfg in ("A0", "A1"):
            z[f"pBp_{cfg}__{m}"] = np.asarray(b.pBp[cfg][m])
        for part in ("fused", "cf"):
            z[f"argmaxA0_{part}__{m}"] = np.asarray(b.argmax["A0"][part][m])
    z["argmaxA0_bar_v"] = np.asarray(b.argmax["A0"]["bar_v"])
    for c in ("a", "b"):
        z[f"argmaxA0_pick__{c}"] = np.asarray(b.argmax["A0"]["picks"][c])
    # grouping scores s_h (D4): the query's posterior from its own modality, the candidate's from its own
    for h in GROUPS:
        z[f"s__{h}__i2t"] = np.einsum("nc,nkc->nk", post[h]["img"][ep.anchor], post[h]["txt"][ep.candidates])
        z[f"s__{h}__t2i"] = np.einsum("nc,nkc->nk", post[h]["txt"][ep.anchor], post[h]["img"][ep.candidates])
    for cfg, parts in (("A0", GROUPS[:3]), ("A1", GROUPS)):
        F = my_features(post, ep, parts)
        z[f"F_{cfg}__a"], z[f"F_{cfg}__b"] = F["a"], F["b"]
    # anchor paintings of the episodes (rb_halves maps local rows; here only the cluster ids are needed)
    np.savez(OUT / "fr_cache.npz", **z)
    print(f"cache written: {len(z)} arrays, E = {len(z['cl'])}, clusters = {len(np.unique(z['cl']))} "
          f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
