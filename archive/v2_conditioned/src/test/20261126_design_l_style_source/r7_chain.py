"""Fresh-seed statistics: DECISION_RULE.md §7.1, §7.5–§7.7, §11 and L16.

Pure functions; fractions in, R@1 points out. No scoring, data loading or writes.
R6's draw loop and R3's sensitivity decomposition are copied, not imported.
"""
import r7_common as R

from statistics import NormalDist

import numpy as np

CHECKS = ("B", "B0", "B1", "BX", "CF")
N_BOOT, BOOT_SEED, CHUNK = 5000, 42, 250
QUARTER_TOL = 1e-9
MULTIPLIERS = {0.0125: {"F1": 3.649, "F2": 3.083, "F3": 3.083},
               0.025: {"F1": 3.417, "F2": 2.802, "F3": 2.802}}
for _alpha, _stages in MULTIPLIERS.items():
    for _stage, _mult in _stages.items():
        _a = _alpha / 5 if _stage == "F1" else _alpha
        assert _mult == round(NormalDist().inv_cdf(1 - _a) + 0.8416, 3), "§7.1 multiplier differs"


def coefficient(alpha):
    """The integer reciprocal of the chain's one-sided level (§7.5)."""
    assert alpha in (0.0125, 0.025), "chain alpha must be 0.0125 or 0.025"
    return 80 if alpha == 0.0125 else 40


def holm(counts, alpha):
    """Five ordered decisions; own_count_passes distinguishes a stop from a failure."""
    c = coefficient(alpha)
    assert set(counts) == set(CHECKS), "Holm needs exactly B, B0, B1, BX, CF"
    for name, count in counts.items():
        assert (not isinstance(count, (bool, np.bool_)) and isinstance(count, (int, np.integer))
                and 0 <= count <= N_BOOT), f"{name}: count must be an integer in 0..5000"
    order = sorted(CHECKS, key=lambda name: (int(counts[name]), CHECKS.index(name)))
    out, earlier_pass = [], True
    for k, name in enumerate(order, 1):
        count = int(counts[name])
        own = c * (count + 1) * (6 - k) <= 5001
        passes = earlier_pass and own
        out.append({"name": name, "k": k, "n_j": count, "passes": bool(passes),
                    "own_count_passes": bool(own), "level_two_sided": 1 - 2 * alpha / (6 - k)})
        earlier_pass = passes
    return out


def _values(values, clusters):
    v, cl = np.asarray(values, dtype=np.float64), np.asarray(clusters)
    assert v.ndim in (1, 2) and cl.shape == (v.shape[0],), "values/clusters shape mismatch"
    assert v.size > 0, "empty differences"
    assert np.isfinite(v).all(), "differences must be finite"
    assert np.abs(v).max() <= 2, "differences must be in fraction units"
    q = np.rint(4 * v)
    assert np.abs(4 * v - q).max() <= QUARTER_TOL, "differences must be quarters (L16)"
    return v, cl, q.astype(np.int64)


def bootstrap_draws(values, clusters, n_boot=N_BOOT, seed=BOOT_SEED, chunk=CHUNK):
    """R6 §3's loop, extended to (episodes, checks) with ONE shared draw set.

    Returns (means, integer_le0), each (n_boot,) for a vector or (n_boot, checks)
    for a matrix. IDs stay painting IDs across seeds; no seed-specific relabeling.
    Every column is checked exactly against the reused cluster_bootstrap at runtime.
    """
    v, cl, q = _values(values, clusters)
    assert n_boot > 0 and chunk > 0, "positive bootstrap size and chunk required"
    vector = v.ndim == 1
    if vector:
        v, q = v[:, None], q[:, None]
    _, idx = np.unique(cl, return_inverse=True)
    idx = idx.reshape(-1)
    k = int(idx.max()) + 1
    assert k >= 2, "bootstrap needs at least two clusters"
    sums = np.column_stack([np.bincount(idx, weights=col, minlength=k) for col in v.T])
    counts = np.bincount(idx, minlength=k).astype(np.float64)
    isums = np.rint(4 * sums).astype(np.int64)
    exact = np.zeros_like(isums)
    np.add.at(exact, idx, q)
    assert np.array_equal(isums, exact), "cluster quarters differ from summed episode quarters"
    rng = np.random.default_rng(seed)
    b = np.empty((n_boot, v.shape[1]), dtype=np.float64)
    le0 = np.empty(b.shape, dtype=bool)
    for start in range(0, n_boot, chunk):
        draws = rng.integers(0, k, size=(min(chunk, n_boot - start), k))
        denom = counts[draws].sum(axis=1)
        stop = start + len(draws)
        for j in range(v.shape[1]):
            b[start:stop, j] = sums[:, j][draws].sum(axis=1) / denom
            le0[start:stop, j] = isums[:, j][draws].sum(axis=1) <= 0
    reference = R.load("src.eval.aspect_metrics").cluster_bootstrap
    for j in range(v.shape[1]):
        ref = reference(v[:, j], cl, n_boot=n_boot, seed=seed, chunk=chunk)
        ci95 = [float(np.percentile(b[:, j], p)) for p in (2.5, 97.5)]
        assert (float(v[:, j].mean()) == ref["point"] and ci95 == ref["ci95"]
                and ref["n_clusters"] == k and b[:, j].shape == (n_boot,)), (
                    f"bootstrap copy column {j} differs from cluster_bootstrap")
    return (b[:, 0], le0[:, 0]) if vector else (b, le0)


def sensitivity(diff, cl, alpha, stage, *, smoke=False):
    """Copied R3 §6.1 ANOVA, with §7.1's check scope and rounded multiplier.

    diff is ONE development seed's per-episode fractions. The projection uses
    three seeds of that scope, not a decomposition of fresh-seed observations.
    smoke permits small synthetic arrays. All widths and x are in R@1 points.
    """
    coefficient(alpha)
    assert stage in ("F1", "F2", "F3"), "unknown sensitivity stage"
    v, cl, _ = _values(diff, cl)
    assert v.ndim == 1, "sensitivity needs one check"
    want = 4096 if stage == "F3" else 12288
    assert smoke or len(v) == want, f"{stage} sensitivity needs {want} episodes"
    d = 100.0 * v
    _, idx = np.unique(cl, return_inverse=True)
    n, P = len(d), int(idx.max()) + 1
    assert P >= 2 and n > P, "need two paintings and more episodes than paintings"
    m = np.bincount(idx, minlength=P).astype(np.float64)
    mean_p = np.bincount(idx, weights=d, minlength=P) / m
    grand = d.mean()
    ssb = float(np.sum(m * (mean_p - grand) ** 2))
    ssw = float(np.sum((d - mean_p[idx]) ** 2))
    msb, msw = ssb / (P - 1), ssw / (n - P)
    sum_m2 = float(np.sum(m ** 2))
    n0 = (n - sum_m2 / n) / (P - 1)
    sig_e2 = msw
    sig_a2 = max(0.0, (msb - sig_e2) / n0)
    se2 = (sig_a2 * (9 * sum_m2 - 6 * n) + sig_e2 * 3 * n) / (3 * n) ** 2
    se = float(np.sqrt(se2))
    b, _ = bootstrap_draws(v, cl)
    ci = [100 * float(np.percentile(b, p)) for p in (2.5, 97.5)]
    mult = MULTIPLIERS[alpha][stage]
    return {"SE": se, "half_width": 1.96 * se, "x": mult * se, "mult": mult,
            "seed42_half_width": 0.5 * (ci[1] - ci[0]), "sigma_a2": float(sig_a2), "sigma_e2": float(sig_e2),
            "n0": float(n0), "n": int(n), "P": P}


def single_pass(count, alpha):
    """F2/F3's integer decision (§7.5); only the caller controls reachability."""
    assert (not isinstance(count, (bool, np.bool_)) and isinstance(count, (int, np.integer))
            and 0 <= count <= N_BOOT), "count must be an integer in 0..5000"
    return bool(coefficient(alpha) * (int(count) + 1) <= 5001)


def _stats(values, draws, le0, a):
    sum4d = int(np.rint(4 * np.asarray(values)).astype(np.int64).sum())
    return {"n_j": int(le0.sum()), "sum4d": sum4d, "point": 25 * sum4d / len(values),
            "ci95": [100 * float(np.percentile(draws, p)) for p in (2.5, 97.5)],
            "ci_level": [100 * float(np.percentile(draws, p)) for p in (100 * a, 100 * (1 - a))],
            "level_two_sided": 1 - 2 * a}


def _reading(record, pick, comparator, sens):
    x = float(sens["x"])
    assert np.isfinite(x) and x >= 0, "detectable margin must be finite and nonnegative"
    record["half_width"] = 0.5 * (record["ci95"][1] - record["ci95"][0])
    record["detectable_margin"] = x
    record["reading"] = (f"inconclusive at a detectable margin of {x:g}" if record["sum4d"] > 0
                         else f"{pick} did not beat {comparator} on fresh episodes")


def chain_record(differences, clusters, pair_index, pick, alpha, sens):
    """§7.5–§7.7 on already pooled differences (CHECKS plus AFF), fractions.

    sens = {F1: {comparator: {x: ...}}, F2: {x: ...}, F3: {x: ...}}.
    Unreached F2/F3 hold status "not tested", point and ci95 in the stage slot
    itself, as §7.6 requires.
    """
    coefficient(alpha)
    assert set(differences) == set(CHECKS) | {"AFF"}, "six paired differences required"
    names = CHECKS + ("AFF",)
    cl, pi = np.asarray(clusters), np.asarray(pair_index)
    assert cl.ndim == 1 and pi.shape == cl.shape, "cluster/pair shape mismatch"
    assert np.issubdtype(pi.dtype, np.integer) and np.isin(pi, (0, 1, 2)).all(), "bad aspect pair IDs"
    assert all(np.asarray(differences[name]).shape == cl.shape for name in names), "difference shape mismatch"
    mask = pi == 2
    assert mask.any(), "missing style x genre episodes"
    values = np.column_stack([differences[name] for name in names])
    draws, le0 = bootstrap_draws(values, cl)
    f3_draws, f3_le0 = bootstrap_draws(values[mask, 2], cl[mask])
    ranks = holm({name: int(le0[:, j].sum()) for j, name in enumerate(CHECKS)}, alpha)
    checks, stopped_at = {}, None
    for rank in ranks:
        name, j = rank["name"], CHECKS.index(rank["name"])
        r = _stats(values[:, j], draws[:, j], le0[:, j], alpha / (6 - rank["k"]))
        r["status"] = (f"not reached: Holm stopped at {stopped_at}" if stopped_at
                       else "passed" if rank["passes"] else "failed")
        if not rank["passes"]:
            _reading(r, pick, name, sens["F1"][name])
            if stopped_at is None:
                stopped_at = name
        checks[name] = r
    f1_pass = all(r["passes"] for r in ranks)
    f1 = {"checks": {name: checks[name] for name in CHECKS},
          "holm_order": [r["name"] for r in ranks],
          "passes": {r["name"]: r["passes"] for r in ranks}, "pass": f1_pass}
    f2 = _stats(values[:, 5], draws[:, 5], le0[:, 5], alpha)
    f3 = _stats(values[mask, 2], f3_draws, f3_le0, alpha)
    stages, last = {}, "F1" if f1_pass else "none"
    reached = f1_pass
    for stage, r, comparator in (("F2", f2, "AFF"), ("F3", f3, "B1")):
        if reached:
            r["pass"] = single_pass(r["n_j"], alpha)
            r["status"] = "passed" if r["pass"] else "failed"
            if r["pass"]:
                last = stage
            else:
                _reading(r, pick, comparator, sens[stage])
            stages[stage] = r
            reached = r["pass"]
        else:
            stages[stage] = {"status": "not tested", **{key: r[key] for key in ("point", "ci95")}}
    return {"pick": pick, "alpha": alpha, "F1": f1, **stages,
            "last_stage_passed": last, "replaces_AFF": last in ("F2", "F3")}


def friday_verdict(chains, *, ceiling_zone=None):
    """§7.7: GO needs any F1; K1 STOP and no picks both imply NO-GO."""
    assert ceiling_zone in (None, "STOP", "WEAK", "CLEAR"), "unknown ceiling zone"
    if ceiling_zone == "STOP":
        return "NO-GO"
    return "GO" if any(chain["F1"]["pass"] for chain in chains.values()) else "NO-GO"


def _picks(picks):
    assert set(picks) <= {"csd", "generic"}, "unknown pick family"
    out = {family: picks[family] for family in ("csd", "generic") if picks.get(family) is not None}
    for family, pick in out.items():
        assert isinstance(pick, str), "pick must be a scorer name"
        kind, sep, x = pick.partition("_")
        allowed = ("V1", "V2", "V3", "V4") if family == "csd" else ("V5", "V6")
        assert sep and kind in ("R1", "XAFF") and x in allowed, f"invalid {family} pick: {pick}"
    return out, 0.0125 if len(out) == 2 else 0.025


def _differences(arrays, pick, *, smoke):
    """Reading-side assertions for the shared arrays.npz keys, without I/O."""
    x = pick.partition("_")[2]
    comparators = {name: name for name in ("B", "B0", "B1", "AFF")}
    comparators.update(BX=f"BX_{x}", CF=f"{pick}_cf")
    required = {"cl", "pair_index", f"{pick}__r1"} | {f"{v}__r1" for v in comparators.values()}
    assert required <= set(arrays), f"missing array keys: {sorted(required - set(arrays))}"
    cl, pi = np.asarray(arrays["cl"]), np.asarray(arrays["pair_index"])
    assert cl.ndim == 1 and np.issubdtype(cl.dtype, np.integer), "painting IDs must be an integer vector"
    n = len(cl)
    assert n > 0 and n % 3 == 0 and (smoke or n == 12288), "expected 3 x 4096 episodes (or smoke)"
    assert pi.shape == (n,) and np.issubdtype(pi.dtype, np.integer), "pair_index shape/type mismatch"
    assert np.array_equal(pi, np.repeat([0, 1, 2], n // 3)), "aspect pairs not in equal ordered blocks"
    scores = {}
    for name in {pick} | set(comparators.values()):
        score = np.asarray(arrays[f"{name}__r1"], dtype=np.float64)
        assert score.shape == cl.shape, f"{name}: R@1 shape differs from painting IDs"
        _values(score, cl)
        assert (score >= 0).all() and (score <= 1).all(), f"{name}: R@1 must be fractions in [0, 1]"
        scores[name] = score
    return {name: scores[pick] - scores[comparators[name]] for name in CHECKS + ("AFF",)}, cl, pi


def sensitivity_record(arrays, picks, *, smoke=False):
    """§7.1 JSON quantities from ONE development seed's shared arrays.npz.

    `picks` is gate_verdict['picks']; null picks are omitted. The runner must
    persist this result before fresh-seed builds. Pure: no seed-42 file is opened.
    """
    chosen, alpha = _picks(picks)
    chains = {}
    for family, pick in chosen.items():
        diffs, cl, pi = _differences(arrays, pick, smoke=smoke)
        mask = pi == 2
        chains[family] = {"pick": pick, "alpha": alpha,
                          "F1": {name: sensitivity(diffs[name], cl, alpha, "F1", smoke=smoke) for name in CHECKS},
                          "F2": sensitivity(diffs["AFF"], cl, alpha, "F2", smoke=smoke),
                          "F3": sensitivity(diffs["B1"][mask], cl[mask], alpha, "F3", smoke=smoke)}
    return {"chains": chains}


def test_record(arrays_by_seed, picks, detectable, *, smoke=False):
    """Return the shared test-verdict core (`chains`, `FV`), without I/O.

    arrays_by_seed: {55: arrays.npz mapping, 56: ..., 57: ...}; sorted internally.
    detectable: sensitivity_record's entire result. Smoke accepts rule seeds
    (9001,9002,9003) or chain-test seeds (9002,9003), as the shared index states.
    R1_Vk/XAFF_Vk picks select their own BX_Vk and <pick>_cf arrays.
    """
    chosen, alpha = _picks(picks)
    if not chosen:
        return {"chains": {}, "FV": "NO-GO"}
    seeds = tuple(sorted(arrays_by_seed))
    rule = R.roles("rule")
    want = ((rule.dev,) + rule.tests, R.roles("chain").tests) if smoke else (R.TEST_SEEDS,)
    assert seeds in want, f"fresh-seed roles: expected {want}, got {seeds}"
    assert "chains" in detectable and set(detectable["chains"]) == set(chosen), "sensitivity families differ"
    chains = {}
    for family, pick in chosen.items():
        sens = detectable["chains"][family]
        assert {"pick", "alpha", "F1", "F2", "F3"} <= set(sens), "missing sensitivity keys"
        assert sens["pick"] == pick and sens["alpha"] == alpha, "sensitivity pick/alpha differs"
        assert set(sens["F1"]) == set(CHECKS), "sensitivity F1 checks differ"
        for stage, checks in (("F1", sens["F1"].values()), ("F2", [sens["F2"]]), ("F3", [sens["F3"]])):
            for check in checks:
                assert {"x", "n"} <= set(check), "missing sensitivity check keys"
                assert np.isfinite(check["x"]) and check["x"] >= 0, "invalid sensitivity x"
                assert smoke or check["n"] == (4096 if stage == "F3" else 12288), "sensitivity scope differs"
        per_seed = [_differences(arrays_by_seed[seed], pick, smoke=smoke) for seed in seeds]
        assert len({len(cl) for _, cl, _ in per_seed}) == 1, "episode counts differ between seeds"
        diffs = {name: np.concatenate([d[name] for d, _, _ in per_seed]) for name in CHECKS + ("AFF",)}
        cl = np.concatenate([cl for _, cl, _ in per_seed])
        pi = np.concatenate([pi for _, _, pi in per_seed])
        chains[family] = chain_record(diffs, cl, pi, pick, alpha, sens)
    return {"chains": chains, "FV": friday_verdict(chains)}
