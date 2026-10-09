"""fr_c: re-derive rule section 10 item 5's descriptive quantities with my own code from the rule's text (and R3 rule
D12, section 7; plan section 10; the module docstring's agent-default definitions for the two-way bootstrap and item
reuse) on a synthetic three-seed input of smoke size, against r6_descriptive.describe. Reviewer C."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

F = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(F))
import r6_common as R  # noqa: E402  (path setup)
import r6_descriptive as DS  # noqa: E402  (the code under review)
import r6_score as S  # noqa: E402  (scorer names and N_TAU only)

import numpy as np  # noqa: E402

rng = np.random.default_rng(2026)
SEEDS = [52, 53, 54]
NPER = 64
N = 3 * NPER
N_ROWS = 3000
groups = np.arange(N_ROWS) // 3                    # three rows per painting
QUART = np.array([0, .25, .5, .75, 1.0])


def pa_rand(cf=False, bias=0.0):
    r1 = rng.choice(QUART, size=N, p=[.3, .2, .2, .15, .15])
    if bias:
        r1 = np.clip(r1 + bias * (rng.random(N) < 0.3), 0, 1)
    other = r1.copy() if cf else rng.choice(QUART, size=N, p=[.5, .2, .15, .1, .05])
    swap = rng.choice(np.array([0, .5, 1.0]), size=N)
    strict = rng.choice(np.array([0, .5, 1.0]), size=N)
    return {"r1": r1, "gain": r1 - other, "other": other, "swap": swap, "strict": strict}


per_seed = []
for s in SEEDS:
    anchor = rng.integers(0, 900, size=N)          # anchors from 300 paintings: clusters repeat
    cand = rng.integers(0, N_ROWS, size=(N, 13))
    members = np.concatenate([anchor[:, None], cand, rng.integers(0, N_ROWS, size=(N, 16))], axis=1)
    pa = {name: pa_rand(cf=name in ("aff_cf", "r1_cf"), bias=0.25 if name == "aff_fused" else 0.0)
          for name in DS.SCORERS}
    gates = {who: [{c: rng.random(N) < 0.3 for c in ("a", "b")} for _ in range(S.N_TAU)] for who in DS.GATE_SETS}
    per_seed.append(SimpleNamespace(
        seed=s, cl=groups[anchor], pair_index=np.repeat(np.arange(3), NPER), pa=pa, gates=gates,
        pick={c: rng.integers(0, 3, size=N).astype(np.int64) for c in ("a", "b")},
        redundancy={h: {"i2t": float(rng.random()), "t2i": float(rng.random())} for h in DS.A0},
        affect_least_redundant=False, anchor_rows=anchor, cand_rows=cand, member_rows=members,
        cand_cl=groups[cand], groups=groups))

rec = DS.describe(per_seed)


# ------------------------------------------------------------------ my own pieces
def boot(values, clusters, n_boot=5000, seed=42, chunk=250):
    """C9 / R3 rule section 2: 5,000 resamples of anchor paintings, default_rng(42), chunks of 250."""
    v = np.asarray(values, float)
    u, idx = np.unique(clusters, return_inverse=True)
    k = len(u)
    sums, cnt = np.zeros(k), np.zeros(k)
    np.add.at(sums, idx, v)
    np.add.at(cnt, idx, 1)
    g = np.random.default_rng(seed)
    out = []
    for st in range(0, n_boot, chunk):
        d = g.integers(0, k, size=(min(chunk, n_boot - st), k))
        out.append(sums[d].sum(1) / cnt[d].sum(1))
    b = np.concatenate(out)
    return {"point": 100 * float(v.mean()),
            "ci95": [100 * float(np.percentile(b, 2.5)), 100 * float(np.percentile(b, 97.5))]}, b


def pc(v, cl):
    return boot(v, cl)[0]


def close(a, b, tol=1e-9):
    return abs(a["point"] - b["point"]) <= tol and all(abs(x - y) <= tol for x, y in zip(a["ci95"], b["ci95"]))


cat = lambda name, m: np.concatenate([d.pa[name][m] for d in per_seed])  # noqa: E731
cl = np.concatenate([d.cl for d in per_seed])
pi = np.concatenate([d.pair_index for d in per_seed])
res = {}

# rows: every scorer, pooled / per seed / per pair, r1 gain swap and AFF minus
bad = []
for name in DS.SCORERS:
    for scope, mask in [("pooled", np.ones(len(cl), bool))] + \
            [(f"seed{s}", np.repeat(np.array(SEEDS) == s, N)) for s in SEEDS] + \
            [(p, pi == i) for i, p in enumerate(R.PAIR_NAMES)]:
        row = (rec["rows"][name]["pooled"] if scope == "pooled" else
               rec["rows"][name]["per_seed"][scope[4:]] if scope.startswith("seed") else
               rec["rows"][name]["per_pair"][scope])
        for m in ("r1", "gain", "swap"):
            if not close(row[m], pc(cat(name, m)[mask], cl[mask])):
                bad.append((name, scope, m))
            if name != "aff_fused" and not close(row["aff_minus"][m],
                                                 pc((cat("aff_fused", m) - cat(name, m))[mask], cl[mask])):
                bad.append((name, scope, "aff_minus", m))
res["rows_bad"] = bad[:10]
res["rows_n_checked"] = len(DS.SCORERS) * 7 * 3

# checks by scope, pooled: P1..P7, S1, S2 from the rule's table (sections 3 and 4)
CHK = {"P1": ("r1", "cosine"), "P2": ("r1", "rca"), "P3": ("r1", "B"), "P4": ("r1", "B0"), "P5": ("r1", "aff_cf"),
       "P6": ("gain", "aff_cf"), "P7": ("gain", "rca"), "S1": ("r1", "B1"), "S2": ("r1", "r1_fused")}
res["checks_pooled"] = all(close(rec["checks_by_scope"]["pooled"][c], pc(cat("aff_fused", m) - cat(o, m), cl))
                           for c, (m, o) in CHK.items())
res["checks_per_pair"] = all(close(rec["checks_by_scope"]["per_pair"][p][c],
                                   pc((cat("aff_fused", m) - cat(o, m))[pi == i], cl[pi == i]))
                             for c, (m, o) in CHK.items() for i, p in enumerate(R.PAIR_NAMES))


# bar margin (R3 rule D12): comparator = argmax mean R@1 of B0, counterpart, B (ties in that order), per scope
def bar(fused, cf, mask):
    means = [(n, cat(k, "r1")[mask].mean()) for n, k in (("B_prime", "B0"), ("counterpart", cf), ("B", "B"))]
    best = means[0]
    for x in means[1:]:
        if x[1] > best[1]:
            best = x
    key = {"B_prime": "B0", "counterpart": cf, "B": "B"}[best[0]]
    v = cat(fused, "r1")[mask] - cat(key, "r1")[mask]
    return best[0], pc(v, cl[mask]), {p: pc(v[pi[mask] == i], cl[mask][pi[mask] == i])
                                      for i, p in enumerate(R.PAIR_NAMES)}


okb = []
for who, (fu, cf) in (("AFF", ("aff_fused", "aff_cf")), ("R1", ("r1_fused", "r1_cf"))):
    name, m, per_pair = bar(fu, cf, np.ones(len(cl), bool))
    got = rec["bar_margin"][who]["pooled"]
    okb.append(got["comparator"] == name and close(got["r1"], m)
               and all(close(got["per_pair_r1"][p], per_pair[p]) for p in R.PAIR_NAMES))
    for s in SEEDS:
        msk = np.repeat(np.array(SEEDS) == s, N)
        name, m, _ = bar(fu, cf, msk)
        got = rec["bar_margin"][who]["per_seed"][str(s)]
        okb.append(got["comparator"] == name and close(got["r1"], m))
res["bar_margin"] = all(okb)
res["bar_comparators"] = {w: rec["bar_margin"][w]["pooled"]["comparator"] for w in ("AFF", "R1")}

# AFF - B1 per pair
res["aff_minus_b1_per_pair"] = all(close(rec["aff_minus_b1"]["per_pair"][p],
                                         pc((cat("aff_fused", "r1") - cat("B1", "r1"))[pi == i], cl[pi == i]))
                                   for i, p in enumerate(R.PAIR_NAMES))

# R1's seven checks, pooled: R1 fused against cosine, RCA, B, B'(A0), its counterpart; gain statistic; gain vs RCA
r1c = rec["r1_checks"]["pooled"]["checks"]
mine7 = [pc(cat("r1_fused", "r1") - cat(o, "r1"), cl) for o in ("cosine", "rca", "B", "B0", "r1_cf")] + \
        [pc(cat("r1_fused", "gain") - cat("r1_cf", "gain"), cl), pc(cat("r1_fused", "gain") - cat("rca", "gain"), cl)]
got7 = sorted((v["point"], v["ci95"][0], v["ci95"][1]) for v in r1c.values())
want7 = sorted((v["point"], v["ci95"][0], v["ci95"][1]) for v in mine7)
res["r1_seven_n"] = len(r1c)
res["r1_seven_match_as_sets"] = len(got7) == 7 and all(abs(a - b) <= 1e-9 for x, y in zip(got7, want7)
                                                         for a, b in zip(x, y))
res["r1_seven_names"] = sorted(r1c)
BYNAME = {"r1_vs_cosine": ("r1", "cosine"), "r1_vs_rca": ("r1", "rca"), "r1_vs_B": ("r1", "B"),
          "r1_vs_Bprime": ("r1", "B0"), "r1_vs_counterpart": ("r1", "r1_cf"), "gain_statistic": ("gain", "r1_cf"),
          "gain_vs_rca": ("gain", "rca")}
res["r1_seven_by_name"] = all(abs(r1c[k]["point"] - 100 * (cat("r1_fused", m) - cat(o, m)).mean()) < 1e-9
                              for k, (m, o) in BYNAME.items())

# two-way bootstrap: my implementation of the module docstring's definition
names = list(CHK)
X = np.stack([cat("aff_fused", m) - cat(o, m) for m, o in CHK.values()], axis=1)
ccl = np.concatenate([d.cand_cl for d in per_seed])
_, ai = np.unique(cl, return_inverse=True)
_, cflat = np.unique(ccl.ravel(), return_inverse=True)
ci_ = cflat.reshape(ccl.shape)
ka, kc = ai.max() + 1, ci_.max() + 1
ga, gc = np.random.default_rng(42), np.random.default_rng([42, 1])
tw, twt = [], []
for st in range(0, 5000, 250):
    da = ga.integers(0, ka, size=(250, ka))
    dc = gc.integers(0, kc, size=(250, kc))
    for r in range(250):
        W = np.bincount(da[r], minlength=ka).astype(float)
        V = np.bincount(dc[r], minlength=kc).astype(float)
        w = W[ai] * V[ci_].mean(1)
        wt = W[ai] * V[ci_[:, :2]].mean(1)
        tw.append(w @ X / w.sum())
        twt.append(wt @ X / wt.sum())
tw, twt = np.array(tw), np.array(twt)
q = rec["two_way_bootstrap"]["quantities"]
dif = []
for j, c in enumerate(names):
    mine = [100 * np.percentile(tw[:, j], 2.5), 100 * np.percentile(tw[:, j], 97.5)]
    mt = [100 * np.percentile(twt[:, j], 2.5), 100 * np.percentile(twt[:, j], 97.5)]
    anc = pc(X[:, j], cl)
    dif.append(max(abs(mine[0] - q[c]["ci95_two_way"][0]), abs(mine[1] - q[c]["ci95_two_way"][1]),
                   abs(mt[0] - q[c]["ci95_two_way_targets"][0]), abs(mt[1] - q[c]["ci95_two_way_targets"][1]),
                   abs(anc["ci95"][0] - q[c]["ci95_anchor"][0]), abs(anc["ci95"][1] - q[c]["ci95_anchor"][1])))
res["two_way_max_abs_diff"] = float(max(dif))
res["two_way_ratio_example"] = q["P1"]["half_width_ratio"]


# item reuse
def reuse(x):
    x = np.asarray(x).ravel()
    return 100 * (1 - len(np.unique(x)) / x.size), 100 * (1 - len(np.unique(groups[x])) / x.size)


ir = rec["item_reuse"]["pooled"]
cand_all = np.concatenate([d.cand_rows for d in per_seed])
mem_all = np.concatenate([d.member_rows for d in per_seed])
anc_all = np.concatenate([d.anchor_rows for d in per_seed])
res["item_reuse"] = all(abs(ir[k]["row_reuse_pct"] - reuse(x)[0]) < 1e-12
                        and abs(ir[k]["painting_reuse_pct"] - reuse(x)[1]) < 1e-12
                        for k, x in (("candidates", cand_all), ("members", mem_all), ("anchors", anc_all)))
res["swap_is_per_anchor_swap"] = close(rec["rows"]["aff_fused"]["pooled"]["swap"], pc(cat("aff_fused", "swap"), cl))
print(json.dumps(res))
(F / "final_review" / "fr_c_desc.json").write_text(json.dumps(res, indent=1))
