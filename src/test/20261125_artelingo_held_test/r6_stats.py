"""Round 6 statistics (DECISION_RULE.md §3, §4, §8.2, §8.5, §12; contracts §6). Pure functions on arrays; no data
loading.

- `bootstrap_draws`: a copy of `src.eval.aspect_metrics.cluster_bootstrap`'s loop that returns the 5,000 resample means
  and, per resample, whether the integer form 4·Σ cluster sums is ≤ 0. Every call asserts that the copy gives
  `cluster_bootstrap`'s own `ci95` exactly (rule §3).
- `holm`, `boundary`: Holm's step-down in the rule's integer form, 40·(n + 1)·(m + 1 − k) ≤ 5,001.
- `pass_record`: the content of `held_pass.json` (and of the regression's seed-42 counts file), contracts §6.
- `sigma_split`, `detectable`: the sensitivity inputs and the detectable margins of rule §6.5 and §8.2.

Units: per-episode differences go in as FRACTIONS (as `per_anchor` returns them); points and intervals come out in R@1
points, scaled ×100 after the percentile, as round 1's `common.point_ci` does (hazard 8).
"""
import hashlib
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).resolve().parent


def _r6c():
    """The pieces of contracts §1 this module uses, under r6_common's own names. Local until ticket 01's r6_common is
    merged; then this body becomes `import r6_common; return r6_common`."""
    out = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=HERE,
                         capture_output=True, text=True, check=True).stdout.strip()
    main = Path(out).resolve().parent
    if not (main / "src/test").is_dir():
        raise AssertionError(f"MAIN {main} has no src/test")

    def amsterdam_now() -> str:
        return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d %H:%M:%S")

    def sha256_file(path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    return SimpleNamespace(
        MAIN=main, HERE=HERE, amsterdam_now=amsterdam_now, sha256_file=sha256_file,
        RULE_SHA256="7444a5e338838d837673b82b047c82b033eb1e2d1e0c3ed4e388dd8673070724",
        PAIR_NAMES=("emotion__style", "emotion__genre", "style__genre"),
        DEV_SEED=42, HELD_SEEDS=(52, 53, 54), SMOKE_SEEDS=(9001, 9002, 9003), N_PER_PAIR=4096, N_SMOKE=64)


R6C = _r6c()
MAIN = R6C.MAIN
R3D = MAIN / "src/test/20261121_round3_affect_gate"
R1D = MAIN / "src/test/20261117_reader_fix_csd"
for _p in (str(R3D), str(MAIN)):     # to the front even if present: the env's editable install lists MAIN last,
    while _p in sys.path:             # behind the cwd, so a worktree's own src/ would win otherwise
        sys.path.remove(_p)
    sys.path.insert(0, _p)

import src.eval.aspect_metrics as _AM  # noqa: E402
import r3_stats as RS  # noqa: E402  round 3's statistics (σ split); brings round 1's common as RS.C

C = RS.C
for _mod, _want in ((_AM, MAIN / "src/eval/aspect_metrics.py"), (RS, R3D / "r3_stats.py"), (C, R1D / "common.py")):
    if Path(_mod.__file__).resolve() != _want:
        raise ImportError(f"'{_mod.__name__}' resolved to {_mod.__file__}, not {_want}")
_cluster_bootstrap = _AM.cluster_bootstrap

# ---------------------------------------------------------------------------------------------------------------- bar
N_BOOT = 5000          # C9: 5,000 resamples of anchor paintings, seed 42, chunks of 250 (rule §3)
BOOT_SEED = 42
CHUNK = 250
QUARTER_TOL = 1e-9     # |4·v − rint(4·v)| allowed per episode (float noise); every difference is a multiple of 0.25
MAX_ABS_DIFF = 2.0     # a paired per-episode difference of R@1 or gain fractions lies in [−2, 2] (catches ×100 inputs)

CHECKS = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")      # rule §3; Holm ties in this order
SECONDARY = ("S1", "S2")                                   # rule §4; Holm ties S1 first, tested only after a GO
M_P, M_S = len(CHECKS), len(SECONDARY)
AFF = "aff_fused"
CF = "aff_cf"
# check -> (metric, comparator key of score_seed's dict, quantity text); every one is AFF minus the comparator
QUANTITIES = {
    "P1": ("r1", "cosine", "R@1, AFF - COS"),
    "P2": ("r1", "rca", "R@1, AFF - RCA"),
    "P3": ("r1", "B", "R@1, AFF - B"),
    "P4": ("r1", "B0", "R@1, AFF - B0"),
    "P5": ("r1", CF, "R@1, AFF - CF"),
    "P6": ("gain", CF, "gain statistic: condition gain, AFF - CF (CF gain 0)"),
    "P7": ("gain", "rca", "condition gain, AFF - RCA"),
    "S1": ("r1", "B1", "R@1, AFF - B1"),
    "S2": ("r1", "r1_fused", "R@1, AFF - R1 fused"),
}
assert tuple(QUANTITIES) == CHECKS + SECONDARY

Z_P = 3.532    # rule §8.2: z at one-sided 0.025/7 plus z at 0.8
Z_S = 3.083    # rule §8.2: z at one-sided 0.025/2 (2.2414) plus z at 0.8 (0.8416)
Z_95 = 2.80    # rule §8.2: x95

MODES = ("held", "regression", "smoke")
_HEX64 = re.compile(r"[0-9a-f]{64}")


# ----------------------------------------------------------------------------------------------------------- draws
def bootstrap_draws(values, clusters, n_boot: int = N_BOOT, seed: int = BOOT_SEED, chunk: int = CHUNK):
    """Rule §3's draws. Returns (b, int_le0): b (n_boot,) float64 resample means Σ sums / Σ counts over the drawn
    clusters, exactly as `cluster_bootstrap` forms them; int_le0[r] = (Σ over drawn clusters of rint(4·cluster sum),
    as int64) ≤ 0, which decides "b_r ≤ 0". Raises AssertionError unless every value is a multiple of 0.25 (up to
    QUARTER_TOL) in fraction units, and unless the copy gives `cluster_bootstrap`'s own ci95 exactly."""
    v = np.asarray(values, dtype=np.float64)
    cl = np.asarray(clusters)
    if v.ndim != 1 or cl.shape != v.shape:
        raise ValueError(f"values {v.shape} and clusters {cl.shape} must be equal-length 1-d arrays")
    if not np.isfinite(v).all():
        raise AssertionError("a per-episode difference is not finite")
    if np.abs(v).max() > MAX_ABS_DIFF:
        raise AssertionError("a per-episode difference exceeds 2 in absolute value: not in fraction units")
    q = np.rint(4.0 * v)
    if np.abs(4.0 * v - q).max() > QUARTER_TOL:
        raise AssertionError("a per-episode difference is not a multiple of 0.25 (R3 rule D8)")
    _, idx = np.unique(cl, return_inverse=True)
    idx = idx.reshape(-1)
    k = int(idx.max()) + 1
    if k < 2:
        raise ValueError("bootstrap_draws needs at least two clusters")
    sums = np.bincount(idx, weights=v, minlength=k)
    counts = np.bincount(idx, minlength=k).astype(np.float64)
    isums = np.rint(4.0 * sums).astype(np.int64)
    exact = np.zeros(k, dtype=np.int64)
    np.add.at(exact, idx, q.astype(np.int64))
    if not np.array_equal(isums, exact):
        raise AssertionError("rint(4·cluster sum) differs from the integer sum of the episodes' quarters")
    rng = np.random.default_rng(seed)
    boots, le0 = [], []
    for start in range(0, n_boot, chunk):
        draws = rng.integers(0, k, size=(min(chunk, n_boot - start), k))
        boots.append(sums[draws].sum(axis=1) / counts[draws].sum(axis=1))
        le0.append(isums[draws].sum(axis=1) <= 0)
    b = np.concatenate(boots)
    int_le0 = np.concatenate(le0)
    ref = _cluster_bootstrap(values, clusters, n_boot=n_boot, seed=seed, chunk=chunk)
    got = [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    if got != ref["ci95"] or ref["n_clusters"] != k or b.shape != (n_boot,) or float(v.mean()) != ref["point"]:
        raise AssertionError(f"the bootstrap copy's ci95 {got} differs from cluster_bootstrap's {ref['ci95']}")
    return b, int_le0


def point_ci_from_draws(values, b) -> dict:
    """{"point", "ci95"} in R@1 points, scaled after the percentile (as round 1's common.point_ci)."""
    return {"point": 100 * float(np.asarray(values, dtype=np.float64).mean()),
            "ci95": [100 * float(np.percentile(b, 2.5)), 100 * float(np.percentile(b, 97.5))]}


def holm_interval(b, k: int, m: int) -> list:
    """The two-sided 1 − 0.05/(m + 1 − k) interval of the draws at Holm rank k, in R@1 points (rule §3)."""
    _rank(k, m)
    return [100 * float(np.percentile(b, 100 * 0.025 / (m + 1 - k))),
            100 * float(np.percentile(b, 100 * (1 - 0.025 / (m + 1 - k))))]


# ------------------------------------------------------------------------------------------------------------ Holm
def _rank(k, m):
    if not (isinstance(k, (int, np.integer)) and isinstance(m, (int, np.integer))) or not 1 <= k <= m:
        raise ValueError(f"Holm rank k={k} must be an integer in 1..m={m}")


def boundary(k: int, m: int) -> int:
    """n*_k, the largest count that passes at Holm rank k of m: ⌊5,001 / (40·(m + 1 − k))⌋ − 1 (rule §8.5)."""
    _rank(k, m)
    return 5001 // (40 * (m + 1 - k)) - 1


BOUNDARIES = {"P": (16, 19, 24, 30, 40, 61, 124), "S": (61, 124)}     # rule §8.5
assert tuple(boundary(k, M_P) for k in range(1, M_P + 1)) == BOUNDARIES["P"]
assert tuple(boundary(k, M_S) for k in range(1, M_S + 1)) == BOUNDARIES["S"]


def near_boundary(n: int, k: int, m: int) -> bool:
    """n within one count of n*_k (rule §8.5): reported to the user; the integer rule still decides."""
    return abs(int(n) - boundary(k, m)) <= 1


def holm(counts: dict, names, m: int) -> list:
    """Holm across `names` (m of them) from the integer counts n_j (rule §3; §4 for S1, S2). Ordered by n ascending,
    ties in the order of `names`. The k-th passes iff it and every earlier one pass and 40·(n + 1)·(m + 1 − k) ≤ 5,001,
    the integer form of p = (n + 1)/5,001 ≤ 0.025/(m + 1 − k). `own_count_passes` is that inequality alone (a check
    that fails only because an earlier one failed is "not reached", rule §4)."""
    names = tuple(names)
    if len(names) != m or len(set(names)) != m or set(counts) != set(names):
        raise ValueError(f"Holm needs exactly the {m} names {names} with one count each, got {sorted(counts)}")
    for nm in names:
        n = counts[nm]
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not 0 <= n <= N_BOOT:
            raise ValueError(f"count of {nm} must be an integer in 0..{N_BOOT}, got {n!r}")
    order = sorted(names, key=lambda nm: (int(counts[nm]), names.index(nm)))
    out, earlier_pass = [], True
    for k, nm in enumerate(order, start=1):
        n = int(counts[nm])
        own = 40 * (n + 1) * (m + 1 - k) <= 5001
        passes = earlier_pass and own
        earlier_pass = passes
        out.append({"name": nm, "k": k, "n": n, "passes": bool(passes), "own_count_passes": bool(own),
                    "level_two_sided": 1 - 0.05 / (m + 1 - k)})
    return out


# ----------------------------------------------------------------------------------------------------- pass record
def check_diffs(per_seed) -> dict:
    """{check: (diff (E,) float64 fractions, cl (E,))} for P1..P7, S1, S2, the seeds concatenated in the given order.
    per_seed: score_seed dicts (contracts §5) with "cl" and the per-anchor dicts of aff_fused, aff_cf, cosine, rca, B,
    B0, B1, r1_fused. Raises AssertionError if CF's condition gain is not exactly 0 on every episode."""
    if len(per_seed) == 0:
        raise ValueError("no seeds")
    for s in per_seed:
        n = len(np.asarray(s["cl"]))
        for key in {AFF} | {QUANTITIES[c][1] for c in QUANTITIES}:
            for metric in ("r1", "gain"):
                if np.asarray(s[key][metric]).shape != (n,):
                    raise ValueError(f"{key}[{metric}] does not have the shape of cl ({n},)")
        if not np.all(np.asarray(s[CF]["gain"], dtype=np.float64) == 0):
            raise AssertionError("CF's condition gain is not exactly 0 on every episode (rule §3, P6)")
    cl = np.concatenate([np.asarray(s["cl"]) for s in per_seed])
    out = {}
    for c, (metric, key, _) in QUANTITIES.items():
        d = np.concatenate([np.asarray(s[AFF][metric], dtype=np.float64) - np.asarray(s[key][metric], dtype=np.float64)
                            for s in per_seed])
        out[c] = (d, cl)
    return out


def _seed_ok(mode, seeds, per_seed):
    want = {"held": tuple(R6C.HELD_SEEDS), "regression": (R6C.DEV_SEED,), "smoke": tuple(R6C.SMOKE_SEEDS)}[mode]
    if tuple(seeds) != want:
        raise AssertionError(f"mode {mode!r} reads seeds {want}, got {tuple(seeds)}")
    if len(per_seed) != len(seeds):
        raise ValueError("one score dict per seed")
    per_pair = R6C.N_SMOKE if mode == "smoke" else R6C.N_PER_PAIR
    for s, seed in zip(per_seed, seeds):
        if len(np.asarray(s["cl"])) != 3 * per_pair:
            raise AssertionError(f"seed {seed}: {len(np.asarray(s['cl']))} episodes, not 3 x {per_pair}")


def _extra_ok(extra, seeds) -> dict:
    if set(extra) != {"episodes_sha256", "runner_sha256", "module_sha256"}:
        raise ValueError(f"extra must hold exactly episodes_sha256, runner_sha256, module_sha256; got {sorted(extra)}")
    eps = {str(k): dict(v) for k, v in extra["episodes_sha256"].items()}
    if set(eps) != {str(s) for s in seeds}:
        raise ValueError(f"episodes_sha256 seeds {sorted(eps)} differ from {list(seeds)}")
    for seed, by_pair in eps.items():
        if tuple(sorted(by_pair)) != tuple(sorted(R6C.PAIR_NAMES)) or not all(
                isinstance(h, str) and _HEX64.fullmatch(h) for h in by_pair.values()):
            raise ValueError(f"episodes_sha256[{seed}] must map the three pair names to 64-hex SHA-256s")
    if not (isinstance(extra["runner_sha256"], str) and _HEX64.fullmatch(extra["runner_sha256"])):
        raise ValueError("runner_sha256 must be a 64-hex SHA-256")
    mods = dict(extra["module_sha256"])
    if not mods or not all(isinstance(h, str) and _HEX64.fullmatch(h) for h in mods.values()):
        raise ValueError("module_sha256 must map file names to 64-hex SHA-256s")
    return {"episodes_sha256": {s: {p: eps[s][p] for p in R6C.PAIR_NAMES} for s in (str(x) for x in seeds)},
            "runner_sha256": extra["runner_sha256"], "module_sha256": mods}


def rule_sha256() -> str:
    """SHA-256 of this folder's DECISION_RULE.md, asserted equal to the committed rule's."""
    got = R6C.sha256_file(HERE / "DECISION_RULE.md")
    if got != R6C.RULE_SHA256:
        raise AssertionError(f"DECISION_RULE.md SHA-256 {got} differs from the committed rule's {R6C.RULE_SHA256}")
    return got


def pass_record(per_seed, mode: str, seeds, extra: dict) -> dict:
    """The content of held_pass.json (mode "held"), of the regression's seed-42 counts file ("regression") or of the
    smoke's pass file ("smoke"); contracts §6. per_seed: score_seed dicts in seed order (see check_diffs); extra:
    {"episodes_sha256": {seed: {pair: sha}}, "runner_sha256": sha, "module_sha256": {file: sha}}. No verdict."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    seeds = [int(s) for s in seeds]
    _seed_ok(mode, seeds, per_seed)
    ext = _extra_ok(extra, seeds)
    rule = rule_sha256()
    diffs = check_diffs(per_seed)
    cl = diffs["P1"][1]
    stats = {}
    for c, (d, cl_c) in diffs.items():
        b, int_le0 = bootstrap_draws(d, cl_c)
        pc = point_ci_from_draws(d, b)
        if pc != C.point_ci(d, cl_c):
            raise AssertionError(f"{c}: point and ci95 differ from round 1's common.point_ci")
        stats[c] = {"b": b, "n": int(int_le0.sum()), **pc}

    def fill(names, m, with_pass):
        ranks = holm({c: stats[c]["n"] for c in names}, names, m)
        out = {}
        for r in ranks:
            c, k = r["name"], r["k"]
            rec = {"quantity": QUANTITIES[c][2], "n": r["n"], "point": stats[c]["point"], "ci95": stats[c]["ci95"],
                   "holm_k": k, "ci_holm": holm_interval(stats[c]["b"], k, m),
                   "level_two_sided": r["level_two_sided"]}
            if with_pass:
                rec["passes"] = r["passes"]
                rec["own_count_passes"] = r["own_count_passes"]
            rec["near_boundary"] = near_boundary(r["n"], k, m)
            out[c] = rec
        return {c: out[c] for c in names}, [r["name"] for r in ranks]

    checks, order = fill(CHECKS, M_P, True)
    secondary, _ = fill(SECONDARY, M_S, False)     # no pass field: tested only after a GO, by the apply step
    return {"rule_sha256": rule, "mode": mode, "seeds": seeds, "n_episodes": int(len(cl)),
            "n_clusters": int(len(np.unique(cl))), "checks": checks, "holm_order": order, "secondary": secondary,
            **ext, "time": R6C.amsterdam_now()}


# ----------------------------------------------------------------------------------------------------- sensitivity
def sigma_split(diff, cl) -> dict:
    """Rule §6.5: σ_a² and σ_ε² of a seed's per-episode difference (FRACTIONS in, as round 3), in R@1 points squared,
    taken from round 3's `r3_stats.sensitivity` (R3 rule §6.1's one-way split)."""
    d = np.asarray(diff, dtype=np.float64)
    if not np.isfinite(d).all() or np.abs(d).max() > MAX_ABS_DIFF:
        raise AssertionError("sigma_split takes finite per-episode differences in fraction units")
    r = RS.sensitivity(d, cl)
    return {"sigma_a2": float(r["sigma_a2"]), "sigma_eps2": float(r["sigma_e2"])}


def detectable(sig: dict, M_p, N: int, family: str = "P") -> dict:
    """Rule §8.2: SE² = (σ_a²·Σ_p M_p² + σ_ε²·N) / N², M_p the pooled episode count of anchor painting p, N = Σ M_p.
    family "P" -> {"SE", "x" (3.532·SE), "x95" (2.80·SE)}; "S" -> {"SE", "x2" (3.083·SE), "x95"}. R@1 points."""
    if family not in ("P", "S"):
        raise ValueError(f"family must be 'P' or 'S', got {family!r}")
    M = np.asarray(M_p)
    if M.ndim != 1 or not np.issubdtype(M.dtype, np.integer) or (M <= 0).any():
        raise ValueError("M_p must be a 1-d array of positive integer episode counts")
    if int(M.astype(np.int64).sum()) != int(N):
        raise AssertionError(f"Σ M_p = {int(M.sum())} differs from N = {N}")
    sum_m2 = float(np.sum(M.astype(np.int64) ** 2))
    se = float(np.sqrt((sig["sigma_a2"] * sum_m2 + sig["sigma_eps2"] * N) / float(N) ** 2))
    if family == "P":
        return {"SE": se, "x": Z_P * se, "x95": Z_95 * se}
    return {"SE": se, "x2": Z_S * se, "x95": Z_95 * se}
