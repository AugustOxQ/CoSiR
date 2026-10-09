"""Round 6 descriptive pass, core (DECISION_RULE.md of this folder: section 10 item 5, section 5 item 5, section 8 item
3; R3 rule D7, D12, D14 and section 7 items 2 to 6; plan section 10; ticket 13). Computed only after held_verdict.json
records the phase-2 agreement (run_r6_descriptive.py checks that); decides nothing.

Layers (pure on arrays and per-anchor dicts, except seed_inputs, which scores one bundle):

  load_core(path, seeds, n_per_pair)       held_arrays.npz (contracts section 7) split per seed: the per-anchor arrays
                                           of the eight scorers the held pass scored (CORE)
  arrays_from_scored(scored, seeds)        the same layout written from score_seed dicts (the seed-42 stand-in and the
                                           synthetic tests; ticket 08 writes the real file)
  check_pass(core, pass_rec)               the arrays reproduce the pass file the verdict rests on: every n_j, point and
                                           95% interval of P1 to P7, S1, S2 (r6_stats' bootstrap copy), exactly
  seed_inputs(bundle, picks, lambdas, readers, core, episodes, groups)
                                           one seed after the verdict: score_seed(..., include_pm=True) on the seed's
                                           bundle (the nine PM scorers and R1's counterpart are scored only here);
                                           the bundle's episodes equal the episode file's; its eight core scorers equal
                                           held_arrays.npz's bit for bit; D7's redundancy on the frozen B
  describe(per_seed, external=None)        the record body below
  scorer_row(pa_by_seed, aff_by_seed, cl_by_seed, pi_by_seed)
                                           one scorer's rows (pooled, per seed, per pair): ticket 14's hook. Ticket 14
                                           passes describe(..., external={name: {"pa": {seed: per_anchor}, "label": str,
                                           "info": {...}}}) for DTS, DTS-CF, DTS-N, FT-LP, FT-LB, FT-LoRA and MLLM (seed
                                           52 only: its pooled scope is then seed 52); each gets the same rows as the
                                           core scorers, and its "info" (parsing failures, wordings, a "missing" mark)
                                           goes to record["external"][name] unchanged.
  two_way_bootstrap(values, anchor_cl, cand_cl)
                                           plan section 10's sensitivity check (agent default, defined below)
  item_reuse(anchor_rows, cand_rows, member_rows, groups)
                                           plan section 10's item-reuse rate (agent default, defined below)

Definitions (agent defaults where the rule and the plan leave them open):
  - Intervals: round 1's common.point_ci (cluster_bootstrap over anchor paintings, 5,000 resamples, seed 42, scaled
    to R@1 points after the percentile, hazard 8). Scopes: pooled over the seeds in the given order; each seed; each
    aspect pair pooled over the seeds (R3 rule section 7 item 1). Clusters are anchor paintings throughout.
  - Bar margin (R3 rule D12): common.bar_info, its comparator (B'(A0) = B0, the counterpart, B; ties in that order)
    chosen once per scope over that scope's episodes: pooled over the seeds, and each seed alone; its per-pair
    breakdown uses the comparator of the scope it breaks down. For AFF and for R1.
  - R1's own seven checks: round 3's r3_stats.go_checks with R1 fused in AFF's place and R1's counterpart (cells 58,
    123) in CF's: R@1 against cosine, RCA, B, B0, the counterpart; the gain statistic; gain against RCA. Points and
    95% intervals; "lower_above_0" is descriptive only (rule section 10 item 5; R1 never gives a verdict).
  - Swap success (plan section 5.1): per_anchor's "swap" (p_A above p_B under condition a and p_B above p_A under
    condition b, mean over the two directions), shown in every scorer row next to R@1 and gain.
  - Gate-open shares: round 3's r3_fusion.open_shares (integer counts first) at each tau, overall, per condition and
    per pair, AFF and R1, per seed and pooled. Pick accuracy: common.pick_statistics under D14's told mapping (emotion
    -> affect, style -> image, genre -> image), pooled and per seed. Redundancy (D7): round 3's r3_bundle.redundancy
    on z(s_h) and z(B) of the seed's FROZEN B (rule section 5 item 4), per seed, with D7's "affect least redundant in
    both directions" flag.
  - Two-way (anchor x candidate) bootstrap: per resample r, anchor-painting multiplicities W_r (cluster_bootstrap's
    own draws: default_rng(42), chunks of 250 rows of integers(0, k_a, size=k_a)) and candidate-painting
    multiplicities V_r (an independent default_rng([42, 1]), the same chunking over the k_c candidate paintings). An
    episode e with anchor painting a(e) and candidate paintings c(e, j) weighs w = W_r[a(e)] x mean_j V_r[c(e, j)]
    (Owen's pigeonhole bootstrap of crossed designs, with the episode's several candidates averaged); the resample
    mean is sum(w x) / sum(w). Two candidate sides are reported: all 13 candidates ("two_way"), and the two targets
    p_a, p_b (candidate columns 0 and 1, "two_way_targets"; R@1 turns mostly on them, so this one is the wider).
    Percentile 95% intervals, x100 after the percentile. Quantities: P1 to P7, S1, S2, pooled. Beside each: the
    anchor-only interval from the same anchor draws, asserted equal to common.point_ci's exactly (the anchor stream
    is cluster_bootstrap's), and each two-way half-width over the anchor-only one. The pigeonhole bootstrap is
    mildly conservative (it counts the episode-level noise on both sides).
  - Item reuse: over the pooled episodes (and per seed), slots = episodes x 13 candidate slots (and x 30 member slots:
    anchor, 13 candidates, 4 + 4 example pairs in both modalities); reuse = 100 x (1 - distinct / slots), for rows
    (items) and for paintings, with the mean uses per distinct row and painting; anchors likewise (12,288 slots per
    seed).

descriptive.json (smoke: results/smoke/descriptive.json; --reserve: descriptive_reserve.json), proposed for contracts:
  {"what", "mode": "held"|"smoke", "seeds", "n_per_pair", "n_episodes", "n_clusters",
   "verdict": {"file", "sha256", "verdict", "kind", "pass_file", "held_pass_sha256", "agreement_file",
               "agreement_sha256"},
   "consistency": {"arrays_reproduce_pass": {check: bool}, "bundle_reproduces_arrays": {seed: true},
                   "episodes_match_files": {seed: true}},
   "scorers": [names], "labels": {name: text},
   "rows": {scorer: {"pooled": ROW, "per_seed": {seed: ROW}, "per_pair": {pair: ROW}}},
       ROW = {"n_episodes": int, "r1": PC, "gain": PC, "swap": PC, "aff_minus": {"r1": PC, "gain": PC, "swap": PC}}
       (no "aff_minus" in AFF's own row); PC = {"point": float, "ci95": [lo, hi]} in R@1 points
   "checks_by_scope": {"pooled": {P1..P7, S1, S2: PC}, "per_seed": {seed: {...}}, "per_pair": {pair: {...}}},
   "bar_margin": {"AFF": {"pooled": bar_info, "per_seed": {seed: bar_info}}, "R1": {...},
                  "AFF_vs_counterpart": {"pooled": diff3, "per_seed": {seed: diff3}},
                  "AFF_minus_R1_bar_margin_pooled": PC},
   "aff_minus_b1": {"pooled": PC, "per_seed": {seed: PC}, "per_pair": {pair: PC}},
   "r1_checks": {"pooled": {"checks": {name: {"point", "ci95", "lower_above_0"}}, "all_seven_lower_above_0", "bar"},
                 "per_seed": {seed: {...}}, "note"},
   "two_way_bootstrap": {"definition", "n_boot", "anchor_seed", "candidate_seed", "n_anchor_clusters",
                         "n_candidate_clusters", "n_target_clusters", "n_episodes", "quantities": {P1..S2:
                         {"quantity", "point", "ci95_anchor", "ci95_two_way", "half_width_ratio",
                         "ci95_two_way_targets", "half_width_ratio_targets"}}},
   "item_reuse": {"definition", "pooled": IR, "per_seed": {seed: IR}},
   "gate_open_shares": {"AFF"|"R1": {"pooled": open_shares, "per_seed": {seed: open_shares}}},
   "pick_accuracy": {"told_mapping", "pooled": {"pick_accuracy", "pick_share"}, "per_seed": {seed: {...}}},
   "redundancy_D7": {seed: {"redundancy": {h: {d: float}}, "affect_least_redundant_both_directions": bool}},
   "frozen": {"cells", "picks", "lambdas"},
   "external": {name: info} (ticket 14),
   "rule_sha256", "module_sha256", "runner_sha256", "input_sha256", "time", "runtime_s"}
  seeds are string keys ("52"); pairs are r6_common.PAIR_NAMES.

Guards carry a `# guard:<name>` marker; test_r6_descriptive.py deletes each on a copy and shows that its scenario then
goes through.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_episodes as E  # noqa: E402
import r6_score as S  # noqa: E402
import r6_stats as ST  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, METRICS, cluster_bootstrap  # noqa: E402

C, RF, RS, R3 = R.C, R.RF, R.RS, R.R3
R3B = R._R3B                       # round 3's r3_bundle: redundancy and affect_least_redundant only (never build_bundle)

CORE = S.CORE_SCORERS              # held_arrays.npz's scorers (rule section 8 item 3)
SCORERS = S.ALL_SCORERS            # every scorer of score_seed(..., include_pm=True)
AFF = "aff_fused"
ROW_METRICS = ("r1", "gain", "swap")
GATE_SETS = S.GATE_SETS            # ("aff", "r1")
A0 = tuple(R3.A0)                  # ("affect", "image", "caption")
TOLD_A0 = {"emotion": "affect", "style": "image", "genre": "image"}       # R3 rule D14
ARRAY_EXTRA = ("cl", "pair_index", "seed_index")
N_BOOT, BOOT_SEED, CHUNK = ST.N_BOOT, ST.BOOT_SEED, ST.CHUNK
CAND_SEED = (42, 1)                # the candidate side's generator, independent of the anchor side's
TARGET_COLS = (0, 1)               # p_a, p_b: the candidate columns of the two targets (AspectEpisodes)
LABELS = {"cosine": "COS (cosine)", "rca": "RCA", "diag": "PM diag", "diag_relu": "PM diag_relu",
          "bilinear": "PM bilinear", "kissme": "PM kissme", "xing": "PM xing", "wang": "PM wang", "probe": "PM probe",
          "tip": "PM tip", "value_prototype": "PM value_prototype", "B": "B", "B0": "B'(A0)", "B1": "B'(A1)",
          "aff_fused": "AFF", "aff_cf": "CF (AFF's matched counterpart)", "r1_fused": "R1 fused",
          "r1_cf": "R1's matched counterpart"}

if not (set(LABELS) == set(SCORERS) and set(CORE) < set(SCORERS) and AFF in CORE
        and dict(C.TOLD["A0"]) == TOLD_A0 and tuple(C.POOLED_ORDER) == R.PAIR_NAMES
        and tuple(ST.QUANTITIES) == ST.CHECKS + ST.SECONDARY and A0 == ("affect", "image", "caption")):
    raise ImportError("r6_descriptive: scorer names, D14's told mapping or the pair order differ from the rule's")


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def _pc(values, cl) -> dict:
    """{"point", "ci95"} in R@1 points (round 1's common.point_ci)."""
    r = C.point_ci(_f64(values), np.asarray(cl))
    return {"point": float(r["point"]), "ci95": [float(r["ci95"][0]), float(r["ci95"][1])]}


# ---------------------------------------------------------------- held_arrays.npz (contracts section 7)

def array_keys() -> tuple:
    return tuple(f"{s}__{m}" for s in CORE for m in METRICS) + ARRAY_EXTRA


def arrays_from_scored(scored, seeds) -> dict:
    """Contracts section 7's held_arrays.npz layout from score_seed dicts in seed order: <scorer>__<metric> (float64,
    the CORE scorers), cl and pair_index (int64), seed_index (int64, 0 .. len(seeds) - 1, the seed's position)."""
    _require(len(scored) == len(seeds) and len(seeds) > 0, "one score dict per seed")
    out = {}
    for s in CORE:
        for m in METRICS:
            out[f"{s}__{m}"] = np.concatenate([_f64(x[s][m]) for x in scored])
    out["cl"] = np.concatenate([np.asarray(x["cl"], dtype=np.int64) for x in scored])
    out["pair_index"] = np.concatenate([np.asarray(x["pair_index"], dtype=np.int64) for x in scored])
    out["seed_index"] = np.concatenate([np.full(len(np.asarray(x["cl"])), i, dtype=np.int64)
                                        for i, x in enumerate(scored)])
    return out


def load_core(path, seeds, n_per_pair) -> dict:
    """held_arrays.npz split per seed: {seed: {"cl", "pair_index", "pa": {scorer: {metric: (n,) float64}}}} for the
    CORE scorers. seed_index may hold each seed's position (0, 1, 2) or the seed itself; either way the seeds are
    consecutive blocks of 3 x n_per_pair episodes in the given order, and each block's pair_index is the pairs'
    blocks of n_per_pair. Keys beyond the layout are ignored (listed in "extra_keys"). A position-coded block is
    bound to its seed later: check_pass (the pass file's pooled numbers, seeds in its order) and seed_inputs (cl =
    groups[anchor] of that seed's episode file)."""
    seeds = [int(s) for s in seeds]
    n_seed = len(R.PAIRS) * int(n_per_pair)
    want = array_keys()
    with np.load(path, allow_pickle=False) as z:
        missing = sorted(set(want) - set(z.files))
        _require(not missing, f"{Path(path).name} lacks {missing} (contracts section 7)")  # guard:arrays_keys
        d = {k: z[k] for k in want}
        extra = sorted(set(z.files) - set(want))
    n = len(seeds) * n_seed
    for k in want:
        _require(d[k].shape == (n,), f"{k}: shape {d[k].shape}, not ({n},) = {len(seeds)} seeds x {n_seed}")
    si = d["seed_index"]
    _require(np.issubdtype(si.dtype, np.integer) and np.issubdtype(d["cl"].dtype, np.integer)
             and np.issubdtype(d["pair_index"].dtype, np.integer), "cl, pair_index, seed_index must be integers")
    by_pos = np.repeat(np.arange(len(seeds), dtype=np.int64), n_seed)
    by_seed = np.repeat(np.asarray(seeds, dtype=np.int64), n_seed)
    _require(np.array_equal(si, by_pos) or np.array_equal(si, by_seed),
             f"seed_index is not {len(seeds)} consecutive blocks of {n_seed} in the order {seeds}")
    pi_seed = np.repeat(np.arange(len(R.PAIRS), dtype=np.int64), int(n_per_pair))
    out = {}
    for i, seed in enumerate(seeds):
        sl = slice(i * n_seed, (i + 1) * n_seed)
        _require(np.array_equal(d["pair_index"][sl], pi_seed), f"seed {seed}: pair_index is not the pairs' blocks")
        pa = {}
        for s in CORE:
            pa[s] = {}
            for m in METRICS:
                x = d[f"{s}__{m}"][sl]
                _require(x.dtype == np.float64 and bool(np.isfinite(x).all()), f"{s}__{m}: not finite float64")
                pa[s][m] = np.ascontiguousarray(x)
        out[seed] = {"cl": np.ascontiguousarray(d["cl"][sl]), "pair_index": np.ascontiguousarray(d["pair_index"][sl]),
                     "pa": pa}
    out["extra_keys"] = extra
    return out


def check_pass(core_by_seed, pass_rec) -> dict:
    """The held arrays reproduce the pass file the verdict rests on (contracts section 6): pooled over the seeds in
    the pass file's order, every check's n_j, point and 95% interval equal its record exactly (r6_stats'
    bootstrap_draws, asserted against cluster_bootstrap). -> {check: True}; raises AssertionError on any difference."""
    seeds = [int(s) for s in pass_rec["seeds"]]
    per_seed = [{"cl": core_by_seed[s]["cl"], **core_by_seed[s]["pa"]} for s in seeds]
    diffs = ST.check_diffs(per_seed)
    cl = diffs["P1"][1]
    _require(pass_rec.get("n_episodes") == len(cl) and pass_rec.get("n_clusters") == len(np.unique(cl)),
             "the arrays' episode or cluster count differs from the pass file's")  # guard:pass_counts
    out = {}
    for c in ST.CHECKS + ST.SECONDARY:
        rec = (pass_rec["checks"] if c in ST.CHECKS else pass_rec["secondary"])[c]
        d, cl_c = diffs[c]
        b, int_le0 = ST.bootstrap_draws(d, cl_c)
        pc = ST.point_ci_from_draws(d, b)
        same = (int(int_le0.sum()) == rec["n"] and pc["point"] == rec["point"] and pc["ci95"] == list(rec["ci95"])
                and ST.QUANTITIES[c][2] == rec["quantity"])
        _require(same, f"{c}: held_arrays.npz does not reproduce the pass file's count, point or "
                       f"interval")  # guard:arrays_reproduce_pass
        out[c] = True
    return out


# ---------------------------------------------------------------- one seed, after the verdict

def seed_inputs(bundle, picks, lambdas, readers, core, episodes, groups) -> SimpleNamespace:
    """One seed's descriptive inputs. bundle: the seed's bundle (run_r6_held's held bundle; a selection bundle in the
    seed-42 stand-in and the smoke); core: load_core's entry of this seed (the held pass's arrays); episodes:
    r6_episodes.load_episodes of the seed's episode file; groups: the split's leakage groups (all rows).
    Scores the bundle with score_seed(..., include_pm=True) (on a held bundle it refuses unless held_verdict.json
    exists) and asserts that it reproduces the core arrays bit for bit, so every core number below is the held pass's.
    -> SimpleNamespace(seed, cl, pair_index, pa {scorer: per_anchor}, gates {who: [{c: (n,)} x 4]}, pick {c: (n,)},
    redundancy, affect_least_redundant, anchor_rows, cand_rows, member_rows, cand_cl, checks)."""
    seed = int(bundle.seed)
    groups = np.asarray(groups)
    ep = episodes.pooled
    n = int(bundle.n)
    _require(int(episodes.seed) == seed and len(ep.anchor) == n, f"seed {seed}: the episode file is seed "
                                                                  f"{episodes.seed} with {len(ep.anchor)} episodes")
    same_eps = (dict(bundle.episodes_sha256) == dict(episodes.sha)
                and all(np.array_equal(getattr(bundle.pooled, f), getattr(ep, f)) for f in E.FIELDS)
                and np.array_equal(np.asarray(bundle.anchor), ep.anchor))
    _require(same_eps, f"seed {seed}: the bundle's episodes differ from the episode file's")  # guard:bundle_episodes
    cl = groups[ep.anchor]
    _require(np.array_equal(cl, np.asarray(bundle.cl)) and np.array_equal(cl, core["cl"])
             and np.array_equal(np.asarray(bundle.pair_index), core["pair_index"])
             and np.array_equal(episodes.pair_index, core["pair_index"]),
             f"seed {seed}: cl or pair_index differ between groups[anchor], the bundle and "
             f"held_arrays.npz")  # guard:cl_pair_index
    scored = S.score_seed(bundle, picks, lambdas, readers, include_pm=True)
    diff = [f"{s}__{m}" for s in CORE for m in METRICS
            if not (np.asarray(scored[s][m]).dtype == np.float64 and np.array_equal(scored[s][m], core["pa"][s][m]))]
    _require(not diff, f"seed {seed}: the rebuilt bundle does not reproduce held_arrays.npz for "
                       f"{diff[:6]}")  # guard:core_reproduced
    pa = {s: (core["pa"][s] if s in CORE else {m: _f64(scored[s][m]) for m in METRICS}) for s in SCORERS}
    gates = {who: [{c: np.asarray(scored["gates"][who][c][t]) for c in CONDITIONS} for t in range(S.N_TAU)]
             for who in GATE_SETS}
    frozen_b = S.frozen_nested(bundle.cos, bundle.t_n1u, bundle.t6u_B, picks["B"], bundle.parity)
    red = R3B.redundancy(SimpleNamespace(B=frozen_b, stack=bundle.stack))
    members = np.stack([ep.anchor, *ep.candidates.T, *ep.pairs_a_img.T, *ep.pairs_a_txt.T, *ep.pairs_b_img.T,
                        *ep.pairs_b_txt.T], axis=1)
    return SimpleNamespace(
        seed=seed, cl=cl, pair_index=np.asarray(core["pair_index"]), pa=pa, gates=gates,
        pick={c: np.asarray(scored["reader"]["pick"][c], dtype=np.int64) for c in CONDITIONS},
        redundancy={h: {d: float(v) for d, v in red[h].items()} for h in A0},
        affect_least_redundant=bool(R3B.affect_least_redundant(red)),
        anchor_rows=np.asarray(ep.anchor), cand_rows=np.asarray(ep.candidates), member_rows=members,
        cand_cl=groups[np.asarray(ep.candidates)], groups=groups,
        checks={"episodes_match_file": True, "bundle_reproduces_arrays": True})


# ---------------------------------------------------------------- rows (ticket 14's hook)

def _cat(dicts, keys=None):
    keys = keys or list(dicts[0])
    return {m: np.concatenate([_f64(d[m]) for d in dicts]) for m in keys}


def _row(x, aff, cl, with_aff_minus) -> dict:
    r = {"n_episodes": int(len(cl))}
    for m in ROW_METRICS:
        r[m] = _pc(x[m], cl)
    if with_aff_minus:
        r["aff_minus"] = {m: _pc(_f64(aff[m]) - _f64(x[m]), cl) for m in ROW_METRICS}
    return r


def scorer_row(pa_by_seed, aff_by_seed, cl_by_seed, pi_by_seed, with_aff_minus=True) -> dict:
    """One scorer's rows: {"pooled": ROW, "per_seed": {seed: ROW}, "per_pair": {pair: ROW}} over the seeds of
    ``pa_by_seed`` (in its order; per pair pooled over those seeds). Each argument is {seed: ...}: per-anchor dicts
    (r1, gain, swap arrays in fractions) of the scorer and of AFF, anchor paintings, pair indices. ROW: R@1, gain and
    swap success with 95% intervals, and AFF minus the scorer in each (paired per anchor), in R@1 points."""
    seeds = list(pa_by_seed)
    _require(len(seeds) > 0 and all(s in aff_by_seed and s in cl_by_seed and s in pi_by_seed for s in seeds),
             "scorer_row: AFF, cl and pair_index are needed for every seed of the scorer")
    for s in seeds:
        n = len(np.asarray(cl_by_seed[s]))
        _require(all(np.asarray(pa_by_seed[s][m]).shape == (n,) and np.asarray(aff_by_seed[s][m]).shape == (n,)
                     for m in ROW_METRICS), f"seed {s}: per-anchor arrays do not have cl's length {n}")
    x = _cat([pa_by_seed[s] for s in seeds], ROW_METRICS)
    a = _cat([aff_by_seed[s] for s in seeds], ROW_METRICS)
    cl = np.concatenate([np.asarray(cl_by_seed[s]) for s in seeds])
    pi = np.concatenate([np.asarray(pi_by_seed[s]) for s in seeds])
    out = {"pooled": _row(x, a, cl, with_aff_minus), "per_seed": {}, "per_pair": {}}
    for s in seeds:
        out["per_seed"][str(s)] = _row(pa_by_seed[s], aff_by_seed[s], np.asarray(cl_by_seed[s]), with_aff_minus)
    for i, p in enumerate(R.PAIR_NAMES):
        msk = pi == i
        out["per_pair"][p] = _row({m: x[m][msk] for m in ROW_METRICS}, {m: a[m][msk] for m in ROW_METRICS},
                                  cl[msk], with_aff_minus)
    return out


def checks_by_scope(rows) -> dict:
    """P1 to P7, S1, S2 per scope, read from the rows (no second bootstrap): AFF minus the check's comparator in the
    check's metric (r6_stats.QUANTITIES)."""
    def one(get):
        return {c: get(rows[key])["aff_minus"][metric] for c, (metric, key, _) in ST.QUANTITIES.items()}
    out = {"pooled": one(lambda r: r["pooled"])}
    out["per_seed"] = {s: one(lambda r, s=s: r["per_seed"][s]) for s in rows[AFF]["per_seed"]}
    out["per_pair"] = {p: one(lambda r, p=p: r["per_pair"][p]) for p in R.PAIR_NAMES}
    return out


# ---------------------------------------------------------------- bar margin, R1's checks

def _by(per_seed, key):
    return {d.seed: d.pa[key] for d in per_seed}


def bar_margins(per_seed) -> dict:
    """R3 rule D12 per scope (pooled over the seeds; each seed), for AFF and R1, with the per-pair breakdowns of
    common.bar_info; AFF against its counterpart (diff3); AFF minus R1 in bar margin, pooled (each with its own pooled
    comparator)."""
    cl = np.concatenate([d.cl for d in per_seed])
    pi = np.concatenate([d.pair_index for d in per_seed])
    cat = {k: _cat([d.pa[k] for d in per_seed]) for k in ("aff_fused", "aff_cf", "r1_fused", "r1_cf", "B0", "B")}
    out, vec = {}, {}
    for who, (fused, cf) in (("AFF", ("aff_fused", "aff_cf")), ("R1", ("r1_fused", "r1_cf"))):
        vec[who], pooled = C.bar_info(cat[fused], cat[cf], cat["B0"], cat["B"], cl, pi)
        per = {str(d.seed): C.bar_info(d.pa[fused], d.pa[cf], d.pa["B0"], d.pa["B"], d.cl, d.pair_index)[1]
               for d in per_seed}
        out[who] = {"pooled": pooled, "per_seed": per}
    out["AFF_vs_counterpart"] = {"pooled": C.diff3(cat["aff_fused"], cat["aff_cf"], cl),
                                 "per_seed": {str(d.seed): C.diff3(d.pa["aff_fused"], d.pa["aff_cf"], d.cl)
                                              for d in per_seed}}
    out["AFF_minus_R1_bar_margin_pooled"] = _pc(vec["AFF"] - vec["R1"], cl)
    return out


def _r1_input(d) -> dict:
    pa = d.pa
    return {"cl": d.cl, "pair_index": d.pair_index, "aff": pa["r1_fused"], "cf": pa["r1_cf"], "r1": pa["r1_fused"],
            "cosine": pa["cosine"], "rca": pa["rca"], "B": pa["B"], "Bp": pa["B0"]}


def _seven(inputs) -> dict:
    r = RS.go_checks(inputs)
    checks = {k: {"point": float(v["point"]), "ci95": [float(v["ci95"][0]), float(v["ci95"][1])],
                  "lower_above_0": bool(v["pass"])} for k, v in r["checks"].items()}
    return {"checks": checks, "all_seven_lower_above_0": bool(r["go"])}


def r1_checks(per_seed, bars) -> dict:
    """R1's own seven checks (round 3's go_checks with R1 fused and its counterpart, B'(A0) = B0), pooled and per
    seed, with R1's bar margin of the same scope. Descriptive (rule section 10 item 5)."""
    pooled = {**_seven([_r1_input(d) for d in per_seed]), "bar": bars["R1"]["pooled"]}
    per = {str(d.seed): {**_seven([_r1_input(d)]), "bar": bars["R1"]["per_seed"][str(d.seed)]} for d in per_seed}
    return {"pooled": pooled, "per_seed": per,
            "note": "descriptive: R1 never gives a verdict (R3 rule section 7 item 3; rule section 10 item 5)"}


def aff_minus_b1(rows) -> dict:
    """AFF minus B'(A1) in R@1, pooled, per seed and per pair (read from B1's rows)."""
    r = rows["B1"]
    return {"pooled": r["pooled"]["aff_minus"]["r1"],
            "per_seed": {s: v["aff_minus"]["r1"] for s, v in r["per_seed"].items()},
            "per_pair": {p: v["aff_minus"]["r1"] for p, v in r["per_pair"].items()}}


# ---------------------------------------------------------------- the two-way bootstrap and item reuse

def _chunks(n_boot, chunk):
    for start in range(0, n_boot, chunk):
        yield start, min(chunk, n_boot - start)


def resample_means(values, a_idx, c_idx, W, V) -> np.ndarray:
    """Resample means of the two-way bootstrap for given multiplicities. values (E, q); a_idx (E,) anchor-cluster
    index; c_idx (E, K) candidate-cluster indices; W (R, k_a), V (R, k_c) multiplicities. Episode weight w =
    W[a] x mean_j V[c_j]; mean = sum(w values) / sum(w). -> (R, q). A resample of total weight 0 raises."""
    X = _f64(values)
    X = X[:, None] if X.ndim == 1 else X
    a_idx, c_idx = np.asarray(a_idx), np.asarray(c_idx)
    W, V = _f64(np.atleast_2d(W)), _f64(np.atleast_2d(V))
    _require(X.shape[0] == len(a_idx) == len(c_idx) and len(W) == len(V), "two-way: shapes disagree")
    out = np.empty((len(W), X.shape[1]))
    for r in range(len(W)):
        w = W[r][a_idx] * V[r][c_idx].mean(axis=1)
        tot = float(w.sum())
        _require(tot > 0, f"two-way resample {r}: total weight 0")  # guard:two_way_weight
        out[r] = (w @ X) / tot
    return out


def two_way_bootstrap(values, anchor_cl, cand_cl, names=None, n_boot=N_BOOT, seed=BOOT_SEED, cand_seed=CAND_SEED,
                      chunk=CHUNK, target_cols=TARGET_COLS, return_draws=False) -> dict:
    """The two-way (anchor x candidate) bootstrap (module docstring). values (E,) or (E, q) per-episode fractions;
    anchor_cl (E,) anchor paintings; cand_cl (E, K) candidate paintings (columns target_cols: the two targets).
    -> {"n_anchor_clusters", "n_candidate_clusters", "n_target_clusters", "n_episodes", "quantities": {name:
    {"point", "ci95_anchor", "ci95_two_way", "half_width_ratio", "ci95_two_way_targets",
    "half_width_ratio_targets"}}} in R@1 points (+ "draws": {"anchor", "two_way", "two_way_targets": (n_boot, q)
    resample means} with return_draws). The anchor-only interval of the same draws is asserted equal to
    cluster_bootstrap's (scaled as common.point_ci) exactly. One candidate generator serves both candidate sides:
    a target's multiplicity is its painting's among all candidate paintings."""
    X = _f64(values)
    X = X[:, None] if X.ndim == 1 else X
    E_, q = X.shape
    names = list(names) if names is not None else [str(j) for j in range(q)]
    cand_cl = np.asarray(cand_cl)
    _require(len(names) == q and np.asarray(anchor_cl).shape == (E_,) and cand_cl.ndim == 2
             and cand_cl.shape[0] == E_ and bool(np.isfinite(X).all()), "two-way: shapes or values are not valid")
    _, a_idx = np.unique(np.asarray(anchor_cl), return_inverse=True)
    a_idx = a_idx.reshape(-1)
    _, c_flat = np.unique(cand_cl.reshape(-1), return_inverse=True)
    c_idx = c_flat.reshape(cand_cl.shape)
    t_idx = c_idx[:, list(target_cols)]
    k_a, k_c = int(a_idx.max()) + 1, int(c_idx.max()) + 1
    _require(k_a >= 2 and k_c >= 2, "two-way: at least two anchor and two candidate clusters")
    sums = [np.bincount(a_idx, weights=X[:, j], minlength=k_a) for j in range(q)]
    counts = np.bincount(a_idx, minlength=k_a).astype(np.float64)
    rng_a, rng_c = np.random.default_rng(seed), np.random.default_rng(list(cand_seed))
    draws = {k: np.empty((n_boot, q)) for k in ("anchor", "two_way", "two_way_targets")}
    for start, m in _chunks(n_boot, chunk):
        da = rng_a.integers(0, k_a, size=(m, k_a))           # cluster_bootstrap's own draws
        dc = rng_c.integers(0, k_c, size=(m, k_c))
        den = counts[da].sum(axis=1)
        for j in range(q):
            draws["anchor"][start:start + m, j] = sums[j][da].sum(axis=1) / den
        W = np.stack([np.bincount(row, minlength=k_a) for row in da])
        V = np.stack([np.bincount(row, minlength=k_c) for row in dc])
        draws["two_way"][start:start + m] = resample_means(X, a_idx, c_idx, W, V)
        draws["two_way_targets"][start:start + m] = resample_means(X, a_idx, t_idx, W, V)

    def ci(b):
        return [100 * float(np.percentile(b, 2.5)), 100 * float(np.percentile(b, 97.5))]

    def ratio(c, base):
        hw = (base[1] - base[0]) / 2
        return float((c[1] - c[0]) / 2 / hw) if hw > 0 else None

    out = {}
    for j, name in enumerate(names):
        ref = cluster_bootstrap(X[:, j], np.asarray(anchor_cl), n_boot=n_boot, seed=seed, chunk=chunk)
        ci_a = ci(draws["anchor"][:, j])
        _require(ci_a == [100 * float(v) for v in ref["ci95"]],
                 f"{name}: the anchor-only interval of the two-way draws differs from cluster_bootstrap's (scaled "
                 f"as common.point_ci)")  # guard:two_way_anchor_stream
        ci_t, ci_g = ci(draws["two_way"][:, j]), ci(draws["two_way_targets"][:, j])
        out[name] = {"point": 100 * float(ref["point"]), "ci95_anchor": ci_a, "ci95_two_way": ci_t,
                     "half_width_ratio": ratio(ci_t, ci_a), "ci95_two_way_targets": ci_g,
                     "half_width_ratio_targets": ratio(ci_g, ci_a)}
    rec = {"n_anchor_clusters": k_a, "n_candidate_clusters": k_c, "n_target_clusters": int(len(np.unique(t_idx))),
           "n_episodes": int(E_), "quantities": out}
    if return_draws:
        rec["draws"] = draws
    return rec


def _reuse(rows, groups) -> dict:
    rows = np.asarray(rows).reshape(-1)
    slots = int(rows.size)
    d_rows = int(len(np.unique(rows)))
    d_paint = int(len(np.unique(np.asarray(groups)[rows])))
    return {"slots": slots, "distinct_rows": d_rows, "distinct_paintings": d_paint,
            "row_reuse_pct": 100.0 * (1.0 - d_rows / slots), "painting_reuse_pct": 100.0 * (1.0 - d_paint / slots),
            "mean_uses_per_row": slots / d_rows, "mean_uses_per_painting": slots / d_paint}


def item_reuse(anchor_rows, cand_rows, member_rows, groups) -> dict:
    """Plan section 10's item-reuse rate (module docstring) for the anchors, the 13 candidates and all 30 members."""
    return {"anchors": _reuse(anchor_rows, groups), "candidates": _reuse(cand_rows, groups),
            "members": _reuse(member_rows, groups)}


# ---------------------------------------------------------------- the record

def _concat_gates(gs):
    return [{c: np.concatenate([g[t][c] for g in gs]) for c in CONDITIONS} for t in range(len(gs[0]))]


def describe(per_seed, external=None) -> dict:
    """The descriptive record body (module docstring) from seed_inputs' namespaces in seed order. external: ticket
    14's {name: {"pa": {seed: per_anchor}, "label": str, "info": dict}} (DTS, FT, MLLM rows)."""
    _require(len(per_seed) > 0 and len({d.seed for d in per_seed}) == len(per_seed), "one entry per distinct seed")
    seeds = [d.seed for d in per_seed]
    cl_by = {d.seed: d.cl for d in per_seed}
    pi_by = {d.seed: d.pair_index for d in per_seed}
    aff_by = _by(per_seed, AFF)
    cl = np.concatenate([d.cl for d in per_seed])
    pi = np.concatenate([d.pair_index for d in per_seed])

    rows = {name: scorer_row(_by(per_seed, name), aff_by, cl_by, pi_by, with_aff_minus=name != AFF)
            for name in SCORERS}
    labels = dict(LABELS)
    ext_info = {}
    for name, x in (external or {}).items():
        _require(name not in rows, f"external row {name} collides with a core scorer")
        pa = x.get("pa") or {}
        if pa:
            rows[name] = scorer_row(pa, {s: aff_by[s] for s in pa}, {s: cl_by[s] for s in pa},
                                    {s: pi_by[s] for s in pa})
        labels[name] = x.get("label", name)
        ext_info[name] = {"seeds": [int(s) for s in pa], **dict(x.get("info") or {})}

    bars = bar_margins(per_seed)
    diffs = ST.check_diffs([{"cl": d.cl, **d.pa} for d in per_seed])
    names = list(ST.CHECKS + ST.SECONDARY)
    cand_cl = np.concatenate([d.cand_cl for d in per_seed])
    tw = two_way_bootstrap(np.stack([diffs[c][0] for c in names], axis=1), cl, cand_cl, names)
    for c in names:
        tw["quantities"][c] = {"quantity": ST.QUANTITIES[c][2], **tw["quantities"][c]}

    def reuse(ds):
        return item_reuse(np.concatenate([d.anchor_rows for d in ds]), np.concatenate([d.cand_rows for d in ds]),
                          np.concatenate([d.member_rows for d in ds]), ds[0].groups)

    picks = {c: np.concatenate([d.pick[c] for d in per_seed]) for c in CONDITIONS}

    def pick_stats(pk, p_i, c_l):
        acc, share = C.pick_statistics(pk, A0, TOLD_A0, p_i, c_l)
        return {"pick_accuracy": acc, "pick_share": share}

    rec = {
        "what": "rule section 10 item 5: the descriptive pass after the verdict; decides nothing",
        "seeds": seeds, "n_episodes": int(len(cl)), "n_clusters": int(len(np.unique(cl))),
        "scorers": list(rows), "labels": labels, "rows": rows, "checks_by_scope": checks_by_scope(rows),
        "bar_margin": bars, "aff_minus_b1": aff_minus_b1(rows), "r1_checks": r1_checks(per_seed, bars),
        "two_way_bootstrap": {
            "definition": "anchor-painting multiplicities W (cluster_bootstrap's draws, default_rng(42)) and "
                          "candidate-painting multiplicities V (default_rng([42, 1])); episode weight W[anchor] x "
                          "mean of V over its 13 candidates (two_way) or over its two targets p_a, p_b "
                          "(two_way_targets); resample mean sum(w x) / sum(w); percentile 95%",
            "n_boot": N_BOOT, "anchor_seed": BOOT_SEED, "candidate_seed": list(CAND_SEED), **tw},
        "item_reuse": {"definition": "reuse_pct = 100 x (1 - distinct / slots) over the episodes' anchor, candidate "
                                     "(13 per episode) and member (30 per episode) slots, for rows and paintings",
                       "pooled": reuse(per_seed), "per_seed": {str(d.seed): reuse([d]) for d in per_seed}},
        "gate_open_shares": {
            who.upper(): {"pooled": RF.open_shares(_concat_gates([d.gates[who] for d in per_seed]), pi),
                          "per_seed": {str(d.seed): RF.open_shares(d.gates[who], d.pair_index) for d in per_seed}}
            for who in GATE_SETS},
        "pick_accuracy": {"told_mapping": TOLD_A0, "pooled": pick_stats(picks, pi, cl),
                          "per_seed": {str(d.seed): pick_stats(d.pick, d.pair_index, d.cl) for d in per_seed},
                          "note": "diagnostic only (R3 rule D14)"},
        "redundancy_D7": {str(d.seed): {"redundancy": d.redundancy,
                                        "affect_least_redundant_both_directions": d.affect_least_redundant}
                          for d in per_seed},
        "external": ext_info,
    }
    return C.jsonable(rec)
