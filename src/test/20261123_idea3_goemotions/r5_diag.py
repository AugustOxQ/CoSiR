"""Round 5 measured diagnostics (DECISION_RULE.md section 5, "Measured diagnostics" (a) to (d)), seed 42, descriptive.

Every function takes the placement its inputs were computed from and calls r5_guard.require(placement) first. A 'ge'
placement is further refused until results/carry.json exists (r5_guard.require_carry(carry_path)); a 'clip'
placement may leave carry_path None, which is the item 4 regression path (AFF's AUC and the CLIP pair lift before
carry.json exists). Pure functions on arrays; nothing is read from a file except carry.json's existence and SHA.
"""
import numpy as np
from sklearn.metrics import roc_auc_score

import r5_common as R5
import r5_guard as G

C, RF3, RTO = R5.C, R5.RF3, R5.RTO
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, first_place, per_anchor  # noqa: E402

ASPECTS = (("emotion", "style"), ("emotion", "genre"), ("style", "genre"))     # bs_lib.ASPECTS, C.POOLED_ORDER order
METRICS3 = ("r1", "gain", "either")


def _gate(placement, carry_path, what):
    G.require(placement, what)
    if carry_path is None:
        if placement.kind != "clip":
            raise G.GuardError(f"{what}: a GE placement needs results/carry.json (rule D11)")
        if G.is_released():
            raise G.GuardError(f"{what}: after release the diagnostics need results/carry.json (rule D11)")
        return
    G.require_carry(carry_path)


# ---------------------------------------------------------------- (a), (b) detection AUC

def auc_sets(pair_index):
    """bs_07_detector.py's sets: positive = condition a of emotion x style (pair 0) and emotion x genre (pair 1);
    negatives the other four (pair, condition) values. Labels in the order condition a's E values then condition b's."""
    pi = np.asarray(pair_index)
    emo = {c: np.array([ASPECTS[i][j] == "emotion" for i in pi]) for j, c in enumerate(CONDITIONS)}
    return np.concatenate([emo["a"], emo["b"]])


def _auc(a, b, pair_index):
    y = auc_sets(pair_index)
    allv = np.concatenate([np.asarray(a, np.float64), np.asarray(b, np.float64)])
    if len(allv) != len(y):
        raise ValueError("scores and pair_index differ in length")
    return float(roc_auc_score(y, allv))


def auc_emotion(P_affect, pair_index, placement, carry_path=None):
    """(a) AUC of P^c(affect) as detector of the emotion condition. P_affect: {"a", "b"}, each (E,) or (E, 3) with
    affect in column 0 (A0 order)."""
    _gate(placement, carry_path, "auc_emotion")
    col = lambda x: np.asarray(x)[:, 0] if np.ndim(x) == 2 else np.asarray(x)
    return _auc(col(P_affect["a"]), col(P_affect["b"]), pair_index)


def auc_delta(Ffeat, pair_index, placement, carry_path=None, column=2):
    """(b) the same AUC on feature column 2 (Delta_affect) of each condition. Ffeat: {c: (E, 18)}."""
    _gate(placement, carry_path, "auc_delta")
    return _auc(np.asarray(Ffeat["a"])[:, column], np.asarray(Ffeat["b"])[:, column], pair_index)


# ---------------------------------------------------------------- (c) pair lift

def pair_lift(Pi, Pt, labS, gS, placement, selection, carry_path=None):
    """(c) run_told_oracle.pair_stats_heads on float64 posteriors of the selection rows. If `selection` (mandatory)
    Pt must equal placement.Q[selection] by value, so the posterior is bound to the placement that was guarded. -> the full dict plus the three ratios and the group lift."""
    _gate(placement, carry_path, "pair_lift")
    Pi, Pt = np.asarray(Pi, np.float64), np.asarray(Pt, np.float64)
    if not np.array_equal(Pt, placement.Q[np.asarray(selection)].astype(np.float64), equal_nan=True):
        raise ValueError("Pt is not the placement's rows at the selection")
    full = RTO.pair_stats_heads(Pi, Pt, labS, gS)
    return {"ratio_same_over_diff": full["by_aspect"]["ratio_same_over_diff"],
            "emotionxstyle": full["contrast"]["emotionxstyle"]["ratio"],
            "emotionxgenre": full["contrast"]["emotionxgenre"]["ratio"],
            "group_lift": R5.GROUP_LIFT, "full": full}


# ---------------------------------------------------------------- (d) the sharper term

def reassemble(bundle, T, gates, fam, placement, carry_path=None):
    """Fused and counterpart scores re-assembled with the cells fam's cross-fits chose (run_family's assembly). The
    per_anchor of the result must equal fam's arrays exactly. -> {"fused": scores, "cf": scores}."""
    _gate(placement, carry_path, "reassemble")
    return _reassemble(bundle, T, gates, fam)


def _reassemble(bundle, T, gates, fam):
    parity = np.asarray(bundle.parity)
    zB, gated, Gc = RF3._terms(bundle, T, gates)
    info = RF3.F.rank_info(bundle.B)
    out = {"fused": RF3.F.assemble(zB, info, gated, fam["fpick"], parity),
           "cf": RF3.F.assemble(zB, info, Gc, fam["cpick"], parity)}
    for who in ("fused", "cf"):
        pa = per_anchor(out[who])
        if set(fam[who]) != set(pa):
            raise AssertionError(f"the family's {who} arrays and the re-assembled ones have different metrics")
        for m, v in fam[who].items():
            if not np.array_equal(np.asarray(pa[m]), np.asarray(v)):
                raise AssertionError(f"re-assembled {who} scores differ from the family's {m} array")
    return out


def episode_terms(S):
    """Per-episode terms of one score set S = {c: {d: (E, 13)}}. 'cond': {c: {r1, other}} = mean over the two
    directions of 1[the target of c ranks strictly first] and of the same for the other aspect's candidate;
    'dir': {d: {r1, other}} = mean over the two conditions (bs_09_direction.py's per_dir)."""
    tgt = {"a": (0, 1), "b": (1, 0)}                         # (target column, other aspect's column)
    cell = {c: {d: (first_place(S[c][d], tgt[c][0]), first_place(S[c][d], tgt[c][1])) for d in DIRECTIONS}
            for c in CONDITIONS}
    cond = {c: {"r1": np.mean([cell[c][d][0] for d in DIRECTIONS], axis=0),
                "other": np.mean([cell[c][d][1] for d in DIRECTIONS], axis=0)} for c in CONDITIONS}
    dr = {d: {"r1": np.mean([cell[c][d][0] for c in CONDITIONS], axis=0),
              "other": np.mean([cell[c][d][1] for c in CONDITIONS], axis=0)} for d in DIRECTIONS}
    return {"cond": cond, "dir": dr}


def _metric(t, m):
    return {"r1": t["r1"], "gain": t["r1"] - t["other"], "either": t["r1"] + t["other"]}[m]


def term_diff(terms_fused, terms_cf):
    """Fused minus counterpart, per episode: {'cond': {c: {metric: arr}}, 'dir': {d: {metric: arr}}}."""
    return {k: {x: {m: _metric(terms_fused[k][x], m) - _metric(terms_cf[k][x], m) for m in METRICS3}
                for x in terms_fused[k]} for k in ("cond", "dir")}


def summarize(diff, cl, pair_index):
    """Intervals (point_ci, pp): per aspect pair and condition, and per direction pooled over pairs and conditions."""
    cl, pi = np.asarray(cl), np.asarray(pair_index)
    pc = {p: {c: {m: C.point_ci(diff["cond"][c][m][pi == i], cl[pi == i]) for m in METRICS3}
              for c in CONDITIONS} for i, p in enumerate(C.POOLED_ORDER)}
    pdir = {d: {m: C.point_ci(diff["dir"][d][m], cl) for m in METRICS3} for d in DIRECTIONS}
    return {"per_pair_condition": pc, "per_direction": pdir}


def minus(diff_cand, diff_aff):
    """Candidate's (fused minus counterpart) minus AFF's, per episode (paired)."""
    return {k: {x: {m: diff_cand[k][x][m] - diff_aff[k][x][m] for m in METRICS3} for x in diff_cand[k]}
            for k in ("cond", "dir")}


def chosen_cells(fam, taus):
    """The chosen cell of each cross-fit on each tune half: tau index, tau, lambda_u, lambda_a. taus is required: tau'
    for G-TF, R3.TAUS for AFF and G-T."""
    return {k: {int(h): RF3.describe(int(c), taus) for h, c in fam[p].items()}
            for k, p in (("fused", "fpick"), ("cf", "cpick"))}


def either_cost_per_gain(rec):
    """-(either change against the counterpart) / gain statistic, beside AFF's 0.5238718116415958."""
    return -rec["either_change"] / rec["gain_statistic"]["point"]


def sharper_term(bundle, T, gates, fam, cl, pair_index, taus, placement, carry_path=None, aff=None):
    """(d) for one family: re-assembly (checked against fam), per pair x condition and per direction fused minus
    counterpart in R@1, gain and either with intervals, the chosen cells. `aff`: the return value of this function
    for AFF; if given, 'minus_aff' holds the paired candidate-minus-AFF intervals. 'diff' is kept (episode arrays)
    for that use and is not for the JSON."""
    _gate(placement, carry_path, "sharper_term")
    S = _reassemble(bundle, T, gates, fam)
    diff = term_diff(episode_terms(S["fused"]), episode_terms(S["cf"]))
    out = {**summarize(diff, cl, pair_index), "cells": chosen_cells(fam, taus), "diff": diff}
    if aff is not None:
        out["minus_aff"] = summarize(minus(diff, aff["diff"]), cl, pair_index)
    return out
