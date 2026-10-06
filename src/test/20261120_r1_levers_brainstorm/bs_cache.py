"""Brainstorm cache (exploratory, seed 42, decides nothing): build round 1's seed-42 bundle once (read only, CPU) and
store what the brainstorm diagnostics need in cache/bs_cache.npz of this folder (gitignored).

Reads: round 1's common.load_bundle (with its regression check), round 1's rb_eval.seed42_features, the stored R1/R2/R3
probabilities of round 2. Writes only cache/bs_cache.npz here. Never touches other folders, held rows or the GPU.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261120_r1_levers_brainstorm/bs_cache.py
"""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R2DIR = HERE.parent / "20261118_reader_fix_round2"
sys.path.insert(0, str(R2DIR))
import r2_common as R  # noqa: E402

C, rbe = R.C, R.rbe
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, METRICS  # noqa: E402

OUT = HERE / "cache" / "bs_cache.npz"


def main():
    t0 = time.time()
    bundle = C.load_bundle(smoke=False)
    ctx = bundle.ctx
    out = {}
    for d in DIRECTIONS:
        out[f"B__{d}"] = np.asarray(bundle.B["a"][d], np.float32)
        assert np.array_equal(np.asarray(bundle.B["a"][d]), np.asarray(bundle.B["b"][d]))
        out[f"cos__{d}"] = np.asarray(ctx.cos["a"][d], np.float32)
    parts4 = ("affect", "image", "caption", "csd")
    stack = C.grouping_stack(bundle.post, ctx.pooled, parts4)
    for d in DIRECTIONS:
        out[f"stack4__{d}"] = np.asarray(stack[d], np.float32)            # (E, 4, 13)
    F1, chk = rbe.seed42_features(bundle, C.CONFIGS["A1"])                 # 24 features, A0's 18 first
    for c in CONDITIONS:
        out[f"feat_A1__{c}"] = np.asarray(F1[c], np.float64)
    for m in METRICS:
        out[f"pB__{m}"] = np.asarray(bundle.pB[m])
        out[f"pBpA0__{m}"] = np.asarray(bundle.pBp["A0"][m])
        out[f"pBpA1__{m}"] = np.asarray(bundle.pBp["A1"][m])
    out["anchor_group"] = np.asarray(ctx.anchor_group)
    out["pair_index"] = np.asarray(ctx.pair_index)
    out["parity"] = np.asarray(ctx.parity)
    # stored reader probabilities (round 2)
    z1 = np.load(R2DIR / "results/cand_R1_A0.npz")
    z2 = np.load(R2DIR / "results/probs_R2_A0.npz")
    z3 = np.load(R2DIR / "results/probs_R3_A0.npz")
    za1 = np.load(R2DIR / "results/cand_R1_A1.npz")
    for c in CONDITIONS:
        out[f"P_R1__{c}"] = z1[f"probs__{c}"]
        out[f"P_R2__{c}"] = z2[f"P__{c}"]
        out[f"P_R3__{c}"] = z3[f"P__{c}"]
        out[f"P_R1A1__{c}"] = za1[f"probs__{c}"]
        for d in DIRECTIONS:
            out[f"T_R1__{c}__{d}"] = z1[f"T__{c}__{d}"]
    assert np.array_equal(z1["anchor_group"], out["anchor_group"]) and np.array_equal(z1["parity"], out["parity"])
    OUT.parent.mkdir(exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) [{time.time() - t0:.0f}s]; feature checks {chk}")


if __name__ == "__main__":
    main()
