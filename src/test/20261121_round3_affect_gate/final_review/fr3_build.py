"""Final review: build each seed's inputs from the allowed loaders and frozen components, with my own features,
grouping scores and reader arithmetic (fr3_lib). Writes out/fr3_seed{s}.npz.

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
        /root/miniconda3/envs/CoSiR/bin/python fr3_build.py 49 50 51 42
"""
import importlib.util
import json
import sys
import time

import numpy as np

import fr3_lib as L

TEST = L.ROOT / "src/test"
QC = TEST / "20261108_new_method_quick_checks"
TOLD = TEST / "20261111_community_told_oracle"
R1DIR = TEST / "20261117_reader_fix_csd"


def load(name, path):
    m = sys.modules.get(name)
    if m is not None:
        assert str(path) == m.__file__, (name, m.__file__)
        return m
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def modules():
    rc = load("run_checks", str(QC / "run_checks.py"))
    n6 = load("run_n6", str(QC / "run_n6.py"))
    rto = load("run_told_oracle", str(TOLD / "run_told_oracle.py"))
    if str(R1DIR) not in sys.path:
        sys.path.insert(0, str(R1DIR))
    import rb_build  # round 1's, for load_readers("A0", False) only
    return rc, n6, rto, rc.rg, rb_build


def say(msg):
    print(f"[fr3_build {time.strftime('%H:%M:%S')}] {msg}", flush=True)


HEADS = {}


def build(seed):
    from src.data.artelingo_splits import artelingo_splits
    from src.eval.aspect_quick_checks import centered_term, crossfit_condition_free, uniform_probe_scores
    rc, n6, rto, rg, rb = modules()
    t0 = time.time()
    ctx = rg.EvalContext(seed, False)
    ep = ctx.pooled
    scorer_train = np.asarray(artelingo_splits(ctx.data).scorer_train)
    inp = rc.model_inputs(ctx, "A3", scorer_train, False)[0]
    t_n1u = centered_term(inp, ep, uniform=True)
    del inp
    post_e2 = n6.load_posteriors(str(TEST / "20261108_new_method_quick_checks/results/n6_posteriors.npz"), ctx)
    B, Bpicks = crossfit_condition_free(ctx.cos, t_n1u, uniform_probe_scores(post_e2, ep, A0E2), ctx.parity)
    if "aff" not in HEADS:
        with np.load(TEST / "20261111_community_told_oracle/results/per_anchor_told_oracle.npz") as zz:
            part = np.asarray(zz["partition_L"], np.int64)
        lab = np.full(len(ctx.groups), -1, np.int64)
        lab[scorer_train] = part                                   # my own global labels
        HEADS["aff"] = rto.fit_one_head(ctx, lab, scorer_train, 60_000)
        stored = json.loads((TEST / "20261111_community_told_oracle/results/told_oracle.json").read_text())
        HEADS["identity"] = json.loads(json.dumps(HEADS["aff"][1])) == stored["arms"]["L"]["head"]
    post = {"affect": HEADS["aff"][0], "image": post_e2["image"], "caption": post_e2["caption"]}
    Bp, Bppicks = crossfit_condition_free(ctx.cos, t_n1u, uniform_probe_scores(post, ep, L.A0), ctx.parity)
    stack = L.grouping_scores(post, ep)
    F = L.features(post, ep)
    pk = rb.load_readers("A0", False)[0]
    P, T, pick, m = L.read(pk, F, stack)
    arr = {"cl": np.asarray(ctx.anchor_group), "parity": np.asarray(ctx.parity), "pair_index": np.asarray(ctx.pair_index),
           "anchor": np.asarray(ep.anchor)}
    for c in L.CONDS:
        arr[f"F__{c}"] = F[c]
        arr[f"P__{c}"] = P[c]
        arr[f"m__{c}"] = m[c]
        arr[f"pick__{c}"] = pick[c]
        for d in L.DIRS:
            arr[f"T__{c}__{d}"] = T[c][d]
            arr[f"cos__{c}__{d}"] = np.asarray(ctx.cos[c][d])
            arr[f"B__{c}__{d}"] = np.asarray(B[c][d])
            arr[f"Bp__{c}__{d}"] = np.asarray(Bp[c][d])
    for d in L.DIRS:
        arr[f"stack__{d}"] = stack[d]
    meta = {"seed": seed, "B_picks": Bpicks, "Bp_picks": Bppicks, "episodes_sha256": ctx.shas,
            "head_identity_told_oracle_L": bool(HEADS["identity"]), "n": int(ctx.n), "seconds": time.time() - t0}
    arr["meta"] = np.array(json.dumps(meta, default=str))
    np.savez_compressed(L.OUT / f"fr3_seed{seed}.npz", **arr)
    say(f"seed {seed}: n {ctx.n}, head identity {HEADS['identity']}, {time.time() - t0:.0f}s")


A0E2 = ("affect", "image", "caption")     # E2's three k-means-64 groupings as keyed in n6_posteriors (D10)

if __name__ == "__main__":
    L.OUT.mkdir(exist_ok=True)
    for s in sys.argv[1:]:
        build(int(s))
