"""Round-4 independent re-derivation: shared core (decides nothing by itself).

Written from ../DECISION_RULE.md (round 4, SHA-256 asserted) and round 3's rule (part of it by reference, SHA-256
asserted) alone. It never opens or imports the round-4 implementation (r4_*.py, run_r4_*.py, test_r4_*.py, results/)
or round 3's implementation path (r3_fusion, r3_stats, r3_bundle, r3_common), round 2's r2_fusion or round 1's rc_core.

Reused by import (round 4's brief and rule §8 allow it): round 3's own re-derivation code `rd3_core` and `rd3_family`
(src/test/20261121_round3_affect_gate/rederive/), which is independent of the implementation path and reproduced round
3's targets bit for bit. From it we use the generic pieces: features, grouping scores, z-scores, the float32 combine,
integer rank counts, per-anchor metrics, the painting bootstrap wrapper, the one-way sensitivity projection, the
224-cell family (gates in, chosen cells and per-anchor arrays out). Round-4 specifics are written here: the A1 reader,
v and v75, the gate factors and candidates' gates (D5, D6), the four-way bar comparator (D8), Delta_k (D9), the D10
clauses, the carry (§5 item 8) and the inputs of D11.
"""
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.dont_write_bytecode = True        # no __pycache__ in the read-only folders we import from

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
R4DIR = HERE.parent
ROOT = R4DIR.parents[2]
T = ROOT / "src/test"
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
R3DIR = T / "20261121_round3_affect_gate"
R3RD = R3DIR / "rederive"
IMPL_RESULTS = R4DIR / "results"     # opened only by rd4_compare_phase1.py, after rd4_phase1.json is written and hashed

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(R3RD) not in sys.path:
    sys.path.insert(0, str(R3RD))

import rd3_core as K3  # noqa: E402
import rd3_family as F3  # noqa: E402

for _m, _p in (("rd3_core", R3RD / "rd3_core.py"), ("rd3_family", R3RD / "rd3_family.py")):
    if Path(sys.modules[_m].__file__).resolve() != _p.resolve():
        raise AssertionError(f"{_m} loaded from {sys.modules[_m].__file__}, not {_p}")

# ---------------------------------------------------------------- the rules
RULE4 = R4DIR / "DECISION_RULE.md"
RULE4_SHA = "cf11a8739e963995d6674648845ac9b0957632583a364cb099c354810f8e911b"
RULE3 = R3DIR / "DECISION_RULE.md"
RULE3_SHA = "2d311dbeac1a561fc2a719b730075c30819f4f7f3a592041cb64b65821dc5925"

# Inputs we read (paths under src/test/): round 3's D15 rows we use, round 4's D11 rows we use, round 1's input table
# rows for the step-1 files, and round 2's re-derivation cache (reference only; its SHA-256 as rd2_prep.json records it).
INPUT_SHA = {
    # round 3 D15 / §2
    "20261030_aspect_baselines/results/episodes_seed42.npz":
        "12af979432ff1a20c88b614ab9e672f203eeeed9c2465bff305e72d01e6c0986",
    "20261111_community_told_oracle/results/per_anchor_told_oracle.npz":
        "27e8101161607da2d6936aa5a8c84f8c296629fe08e37a68c8418c9356af1366",
    "20261031_pseudo_partitions/results/partitions.npz":
        "cfd57dbb21216cdc9d1c2e42d1b408648bc18147f62354b52202ade4ab8a9caa",
    "20261108_new_method_quick_checks/results/n6_posteriors.npz":
        "2ad75026e5869c461fed156ffcdcf207c66f04a59dae514358522fbbaa9df2e0",
    "20261101_aspect_factor_gonogo/checkpoints/A3_seed42.pt":
        "dadfef1bed95bbd647c39fa0ea483edea78627d440a23b5a032083e76dd469e2",
    "20261111_community_told_oracle/results/told_oracle.json":
        "76d9ec896b941a518e7db6684805fe0bc329f6afe4a48e1993992fbb221d70d2",
    "20261117_reader_fix_csd/DECISION_RULE.md": "613d8c9d13fb7d4787ec0fb91dc65cfd4a192d07c88ab8f961da3adc1c35d23c",
    "20261117_reader_fix_csd/common.py": "99496dcf859ff0b0f746266125c975c5e2f9632e0c2c91d74df17102c65c10a7",
    "20261117_reader_fix_csd/rb_build.py": "63c7310c890675cc63e49838feb364cdd1c5c68aafdfb0f0f7972128476cba7b",
    "20261117_reader_fix_csd/results/rb_reader_A0.pkl":
        "6387b469662446734e650571c88a152651da3d66bcade28fb6cdd7828f58d34c",
    "20261117_reader_fix_csd/results/rb_reader_A0.json":
        "cd90da922967cf1b62e437d3927c97cd9cfdae2c139ee392744538d3e3e5c6a8",
    "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.npz":
        "628b21aeaf6d306f82bd9abb68e6c257348535c3aff0fad0a2d065fc48302981",
    "20261117_reader_fix_csd/results/cand_Rc_Rb_expected_A0.json":
        "c6e2f83b73c47b9c054e6381a5d020a05618728cb820861b89f0b80de5a8a16e",
    "20261117_reader_fix_csd/results/rc_tau.json": "e10cf52b2e81ea5b5f7243363d7ddf226440c3a95512cbc7e0b9ed18c5e702bf",
    "20261120_r1_levers_brainstorm/results/bs_04_readers.json":
        "42bf6f204598c011bfada46e5eabf42dc7759fc6f6750721527d16239141691e",
    "20261120_r1_levers_brainstorm/results/bs_05_aff.json":
        "a96719ba505b72883bfa2eeaca66ab872da128314654ba8f74580998b2a72b10",
    "20261030_aspect_baselines/results/per_anchor_seed42.npz":
        "a4818ba0fa5f7249355afe2d2483404dcd34d22cae26984b76be787bb6e9e59d",
    # round 4 D11
    "20261121_round3_affect_gate/DECISION_RULE.md": RULE3_SHA,
    "20261117_reader_fix_csd/results/rb_reader_A1.pkl":
        "4e1e4e23c20333b839aa1f92942891bdef51955317640d1014db5d097b7060f6",
    "20261117_reader_fix_csd/results/rb_reader_A1.json":
        "41ec3bd1ef99523430ce6aacb2576c7928446a93b65fb9ded2b3ae6a6089d649",
    "20261117_reader_fix_csd/results/rb_reader_A1.npz":
        "f0297b92276cb0eda69f3717545c74a766203517950622f2966978b209c488cd",
    "20261116_grouping_step1_style/run_step1.py": "f4bea509a6ca4fbb48e60823b4add1ba89570b626416127aaf57db66d63018c0",
    "20261116_grouping_step1_style/results/step1_heads_style.npz":
        "898a37017d82d3e130b20155e90f69e51f82f30892e04316dcb56d7aaaf8df8b",
    "20261118_reader_fix_round2/results/cand_R1_A1.npz":
        "7494f8ef3f17db4ec1ef4cc5e5948b5fbd0c793beaf700a3a8464b296d01e075",
    "20261118_reader_fix_round2/results/cand_R1_A1.json":
        "4d13edaf7f3ea9f25782685c2f7daa9ebafeec6a5595d8de233cda376a4b9eb2",
    # round 1's input table (common.INPUT_SHA), the step-1 files
    "20261116_grouping_step1_style/results/step1_eval_style.npz":
        "8d10a0fbbd34212c73849239faed9d0a07372e68732dc452bf2fd57f7409c68c",
    "20261116_grouping_step1_style/results/step1_group_style.npz":
        "b04d96b4798acdfdbc9ca87436012755a681c931ac4f7612fa5683350b7a20e2",
    # round 2's independent re-derivation cache (reference only; SHA-256 recorded in its rd2_prep.json)
    "20261118_reader_fix_round2/rederive/out/rd2_seed42_cache.npz":
        "a173a7c4a50622a669b5baf5d2df9c645da82beec2daf46bb9f3f638e41a88ce",
}

COND, DIRS, METRICS = K3.COND, K3.DIRS, K3.METRICS
A0 = ("affect", "image", "caption")
A1 = ("affect", "image", "caption", "csd")
PAIR_NAMES = K3.PAIR_NAMES
CANDIDATES = ("V4", "V2", "V24")          # also the carry order (§5 item 8)
READS_CSD = {"V4": False, "V2": True, "V24": True}

# constants stated in the rules (targets and frozen values)
RULE_TAUS = K3.RULE_TAUS                  # round 3 D6
V75 = 0.021043562795966864                # round 4 D5
V75_COUNT = 9216                          # round 4 D5: 1[v < v75] = 1 on 9,216 of 12,288 episodes
BP_A1_R1 = 18.804931640625                # round 4 D4
BP_A0_R1 = 18.436686197916664             # round 3 D11
B_R1 = 18.341064453125                    # round 3 D10
TIE_BAND = 24                             # round 4 §5 item 8
N_EP_SEED42 = 12288

ALLOWED_SEEDS = (42,)                     # phase 1: seed 42 only (never 49 to 54, never smoke seeds)


# ---------------------------------------------------------------- utilities

def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def assert_rules():
    got4, got3 = sha_file(RULE4), sha_file(RULE3)
    if got4 != RULE4_SHA:
        raise SystemExit(f"round-4 DECISION_RULE.md SHA-256 {got4} is not the committed {RULE4_SHA}")
    if got3 != RULE3_SHA:
        raise SystemExit(f"round-3 DECISION_RULE.md SHA-256 {got3} is not {RULE3_SHA}")
    return {"round4_rule": got4, "round3_rule": got3}


def assert_inputs(names):
    out = {}
    for n in names:
        got = sha_file(T / n)
        if got != INPUT_SHA[n]:
            raise SystemExit(f"{n}: SHA-256 {got} differs from the expected {INPUT_SHA[n]}")
        out[n] = got
    return out


def guard_seed(seed):
    if int(seed) not in ALLOWED_SEEDS:
        raise SystemExit(f"seed {seed} may not be touched by the phase-1 re-derivation (seed 42 only)")


def now_ams():
    return datetime.now(ZoneInfo("Europe/Amsterdam")).strftime("%Y-%m-%d %H:%M")


jsonable = K3.jsonable
log = K3.log
point_ci = K3.point_ci
import_path = K3.import_path


def save_json(path, obj):
    obj = jsonable(obj)
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=True))
    return obj


# ---------------------------------------------------------------- readers (round 3 D5; round 4 D3)

def reader_probs(pk, F, parts):
    """P^c = mean over the two half-readers of predict_proba(scaler.transform(x)) (float64), for configuration parts."""
    if tuple(pk["groupings"]) != tuple(parts) or list(pk["feature_names"]) != K3.feature_names(parts):
        raise AssertionError(f"reader groupings or feature layout differ from {parts}'s {6 * len(parts)} features")
    P = {}
    for c in COND:
        hs = []
        for h in pk["halves"]:
            p = h["model"].predict_proba(h["scaler"].transform(F[c]))
            if not np.array_equal(h["model"].classes_, np.arange(p.shape[1])) or p.shape[1] != len(parts):
                raise AssertionError("half-reader classes are not 0..H-1")
            hs.append(np.asarray(p, np.float64))
        P[c] = (hs[0] + hs[1]) / 2.0
        if not np.allclose(P[c].sum(1), 1.0, rtol=0, atol=1e-12):
            raise AssertionError("reader probabilities do not sum to 1")
    return P


def exact_argmax_ties(P):
    """Number of rows whose maximum probability is attained by more than one grouping (exact equality)."""
    P = np.asarray(P)
    return int(((P == P.max(axis=1, keepdims=True)).sum(axis=1) > 1).sum())


# ---------------------------------------------------------------- v, v75 (D5) and the gates (round 3 D6, round 4 D6)

def abstention_signal(F_a, F_b):
    """v = min(S_image^a, C_image^a) (A0 feature columns 6 and 7 of condition a, float64); v^b asserted equal."""
    if F_a.shape[1] < 12:
        raise AssertionError("A0 features expected")
    v = np.minimum(np.asarray(F_a[:, 6], np.float64), np.asarray(F_a[:, 7], np.float64))
    vb = np.minimum(np.asarray(F_b[:, 6], np.float64), np.asarray(F_b[:, 7], np.float64))
    return v, vb


def gates_r1(margins, taus):
    """R1: g_t^c = 1[m^c >= tau_t], float64 compare, float32 0/1 (round 3 D6)."""
    return [{c: (np.asarray(margins[c], np.float64) >= np.float64(t)).astype(np.float32) for c in COND}
            for t in taus]


def gates_aff(margins, picks, taus):
    """AFF: g_t^c = 1[m^c >= tau_t] * 1[pi^c = affect (index 0)], float32 0/1 (round 3 D6)."""
    return [{c: ((np.asarray(margins[c], np.float64) >= np.float64(t)) & (np.asarray(picks[c]) == 0))
             .astype(np.float32) for c in COND} for t in taus]


def factor_abstention(v, v75):
    """a_v = 1[v < v75], the same for both conditions (D5), float32 0/1."""
    f = (np.asarray(v, np.float64) < np.float64(v75)).astype(np.float32)
    return {c: f for c in COND}


def factor_a1_pick(pick_a1):
    """1[pi_A1^c = affect (index 0)], per condition, float32 0/1 (D6)."""
    return {c: (np.asarray(pick_a1[c]) == 0).astype(np.float32) for c in COND}


def factor_ones(E):
    return {c: np.ones(E, np.float32) for c in COND}


def apply_factors(base, *factors):
    """D6: the candidate's gate = base gate times each factor, per tau index and condition (float32 0/1). Asserts: the
    result is float32 with values in {0, 1}, and is 0 wherever the base gate is 0."""
    out = []
    for g in base:
        gt = {}
        for c in COND:
            x = np.asarray(g[c], np.float32).copy()
            for f in factors:
                x = (x * np.asarray(f[c], np.float32)).astype(np.float32)
            if x.dtype != np.float32 or not np.isin(x, (0.0, 1.0)).all():
                raise AssertionError("gate is not a float32 array of 0s and 1s")
            if np.any(x[np.asarray(g[c]) == 0] != 0):
                raise AssertionError("candidate gate open where the base gate is closed")
            gt[c] = x
        out.append(gt)
    return out


def gates_equal(g1, g2):
    return all(np.array_equal(g1[t][c], g2[t][c]) and g1[t][c].dtype == g2[t][c].dtype
               for t in range(len(g1)) for c in COND)


def open_counts(gates):
    return {f"tau_{t}": {c: int(np.asarray(gates[t][c]).sum()) for c in COND} for t in range(len(gates))}


# ---------------------------------------------------------------- integer R@1 (D9) and comparators (D8)

def as_int4(r1):
    """4 x per-anchor R@1 as int64, asserting every value is a multiple of 0.25 (round 4 D9)."""
    x = np.asarray(r1, np.float64) * 4.0
    xi = np.rint(x)
    if not np.array_equal(x, xi):
        raise AssertionError("a per-anchor R@1 is not a multiple of 0.25")
    return xi.astype(np.int64)


def bar_comparator(order):
    """D8: order = [(name, per-anchor dict), ...] in tie order. The largest mean R@1 over the episodes considered wins,
    ties to the earliest. Compared through integer sums (same episode count for all, so the same order as the full-
    precision means); the means are returned for the record."""
    sums = [int(as_int4(p["r1"]).sum()) for _, p in order]
    best = 0
    for i in range(1, len(order)):
        if sums[i] > sums[best]:
            best = i
    means = {n: 100 * float(np.mean(np.asarray(p["r1"], np.float64))) for n, p in order}
    return order[best][0], order[best][1], means, {n: s for (n, _), s in zip(order, sums)}


def evaluate(pn, pc, order, cl, pair_index, p_ref=None):
    """Round 3 D12 / round 4 D8 for one fused reader: bar comparator from `order`, bar margin, margin against the
    counterpart, gain statistic (the counterpart's gain asserted 0 on every episode), either change against the
    counterpart, D10 clauses, per-pair breakdowns with the seed-scope comparator. With p_ref (AFF's fused arrays):
    Delta_k (D9) as an integer with its point and interval."""
    comp_name, comp, means, sums = bar_comparator(order)
    if not np.all(np.asarray(pc["gain"]) == 0):
        raise AssertionError("counterpart gain is not 0 on every episode")
    bar_v = np.asarray(pn["r1"], np.float64) - np.asarray(comp["r1"], np.float64)
    gain_v = np.asarray(pn["gain"], np.float64) - np.asarray(pc["gain"], np.float64)
    marg_v = np.asarray(pn["r1"], np.float64) - np.asarray(pc["r1"], np.float64)
    eith_v = K3.either(pn) - K3.either(pc)
    bar, gain, marg, eith = point_ci(bar_v, cl), point_ci(gain_v, cl), point_ci(marg_v, cl), point_ci(eith_v, cl)
    clauses = {"c1_bar_point_ge_0.5": bool(bar["point"] >= 0.5), "c2_bar_lower_gt_0": bool(bar["ci95"][0] > 0),
               "c3_gain_lower_gt_0": bool(gain["ci95"][0] > 0)}
    clauses["clears_bar"] = all(clauses.values())
    boundary = {"c1_within_1e-12": abs(bar["point"] - 0.5) <= 1e-12, "c2_within_1e-12": abs(bar["ci95"][0]) <= 1e-12,
                "c3_within_1e-12": abs(gain["ci95"][0]) <= 1e-12}
    per_pair = {}
    for i, p in enumerate(PAIR_NAMES):
        m = np.asarray(pair_index) == i
        per_pair[p] = {"bar_margin": point_ci(bar_v[m], cl[m]), "margin": point_ci(marg_v[m], cl[m]),
                       "gain_statistic": point_ci(gain_v[m], cl[m]), "n_episodes": int(m.sum())}
    out = {"r1_means": {"fused": K3.mean_pp(pn), **{f"{n}": v for n, v in means.items()}},
           "r1_int4_sums": {"fused": int(as_int4(pn["r1"]).sum()), **sums},
           "bar_comparator": comp_name, "bar_margin": bar, "margin_vs_counterpart": marg, "gain_statistic": gain,
           "either_change_vs_counterpart": eith, "D10_clauses": clauses, "D10_boundary_flags": boundary,
           "per_pair": per_pair}
    arrays = {"bar_v": bar_v, "gain_v": gain_v, "marg_v": marg_v}
    if p_ref is not None:
        d4 = as_int4(pn["r1"]) - as_int4(p_ref["r1"])
        dk = int(d4.sum())
        d = np.asarray(pn["r1"], np.float64) - np.asarray(p_ref["r1"], np.float64)
        ci = point_ci(d, cl)
        out["Delta_k"] = {"int": dk, "point_pp": 100.0 * dk / (4.0 * len(d4)), "bootstrap_point_pp": ci["point"],
                          "ci95": ci["ci95"], "beats_AFF": bool(dk > 0), "at_zero": bool(dk == 0),
                          "episodes_up": int((d4 > 0).sum()), "episodes_down": int((d4 < 0).sum())}
        arrays["d_vs_AFF"] = d
    return out, arrays


def carry(dev):
    """§5 item 8. dev: {k: {"D10_clauses": {...}, "Delta_k": {"int": ...}}} for k in CANDIDATES."""
    E = [k for k in CANDIDATES if dev[k]["D10_clauses"]["clears_bar"] and dev[k]["Delta_k"]["int"] > 0]
    rec = {"order": list(CANDIDATES), "tie_band": TIE_BAND,
           "Delta_k": {k: dev[k]["Delta_k"]["int"] for k in CANDIDATES},
           "D10_clauses": {k: dev[k]["D10_clauses"] for k in CANDIDATES}, "E": E}
    if not E:
        rec.update({"M": None, "tied": [], "carried": None, "kill": True, "gaps": {}, "gap_exactly_24": []})
        return rec
    M = max(dev[k]["Delta_k"]["int"] for k in E)
    gaps = {k: M - dev[k]["Delta_k"]["int"] for k in E}
    tied = [k for k in E if gaps[k] <= TIE_BAND]
    rec.update({"M": int(M), "gaps": gaps, "tied": tied, "carried": tied[0], "kill": False,
                "gap_exactly_24": [k for k in E if gaps[k] == TIE_BAND]})
    return rec


sensitivity = K3.sensitivity
