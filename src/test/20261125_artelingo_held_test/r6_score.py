"""Round 6 frozen-pick scoring (DECISION_RULE.md of this folder: section 5 item 4, section 8 item 3, section 11;
contracts section 5; ticket 06): every scorer of one seed's bundle (r6_bundle) with the picks frozen from seed 42. No
pick, sigma*, tau or lambda is computed here.

Convention everywhere (hazard 7): the pick of tune half h scores the episodes of parity 1 - h.

  frozen_nested(cos, t_u, t_c, pick, parity)   B, B0, B1: nested_scores(cos, T_N1u, T_6u, lu, la) of each half's pick
                                               (row-wise: every row scored, then taken by parity); float32, as
                                               crossfit_condition_free assembles
  frozen_fused(cos, term, lam, parity)         RCA and PM: fused_scores(cos, term, lam) of each half's lambda
  frozen_cells(B, parity, T, gates, f, c)      AFF, CF, R1 fused, R1 counterpart: a copy of round 3's
                                               r3_fusion.score_frozen (round 2's assemble, k_top 13) returning the
                                               scores, on z(B) of the FROZEN B (hazard 1), with the CF term asserted
                                               condition-free per cell (hazard 10)
  score_seed(bundle, picks, lambdas, readers, *, include_pm)
                                               contracts section 5's dict of per-anchor results, gates, reader, cl,
                                               pair_index

include_pm (keyword, no default): rule section 8 item 3 allows, on held episodes before held_verdict.json exists,
only AFF fused, CF, COS, RCA, B, B0, B1 and R1 fused (CORE_SCORERS). score_seed(..., include_pm=False) scores those
alone: it reads no PM term or lambda, never reads R1's counterpart cells or builds its G_cf, and returns no PM key and
no r1_cf. include_pm=True adds the descriptive extras, the nine PM scorers and R1's counterpart r1_cf with its gain-0
check (contracts section 5, amendment 14:50); on a held bundle it is refused unless held_verdict.json exists (the
descriptive pass, ticket 13). The regression (ticket 07) and the descriptive pass call it with True.

Guards carry a `# guard:<name>` marker; test_r6_score.py deletes each on a copy and shows that its scenario then goes
through.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_bundle as B  # noqa: E402
import r6_context as X  # noqa: E402
import r6_picks as P  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor  # noqa: E402
from src.eval.aspect_nested import _zdict, nested_scores  # noqa: E402
from src.eval.aspect_quick_checks import _require_condition_free  # noqa: E402
from src.eval.aspect_scorers import fused_scores  # noqa: E402

C, RC, F2, RF, R3 = R.C, R.RC, R.F2, R.RF, R.R3

# rule section 5 item 4: the frozen cells, (tune half 0, tune half 1); cell number (t*7 + u)*8 + a, k_top 13
CELLS = {"aff": {"fused": (39, 119), "cf": (149, 10)}, "r1": {"fused": (116, 119), "cf": (58, 123)}}
# cell -> (tau index, lambda_u, lambda_a), the rule's table
CELL_TEXT = {39: (0, 4.0, 16.0), 119: (2, 0.0, 16.0), 149: (2, 4.0, 4.0), 10: (0, 0.5, 0.5), 116: (2, 0.0, 2.0),
             58: (1, 0.0, 0.5), 123: (2, 0.5, 1.0)}
PM_SCORERS = B.PM_NAMES
CORE_SCORERS = ("cosine", "rca", "B", "B0", "B1", "aff_fused", "aff_cf", "r1_fused")    # rule section 8 item 3
DESCRIPTIVE_EXTRAS = PM_SCORERS + ("r1_cf",)                                         # include_pm=True only
ALL_SCORERS = ("cosine", "rca") + PM_SCORERS + ("B", "B0", "B1", "aff_fused", "aff_cf", "r1_fused", "r1_cf")
EXTRA_KEYS = ("gates", "reader", "cl", "pair_index")
CONDITION_FREE = ("cosine", "B", "B0", "B1", "aff_cf", "r1_cf")    # gain 0 on every episode (when scored)
GATE_SETS = ("aff", "r1")
HALVES = (0, 1)
N_TAU = len(R3.TAUS)
VERDICT = R.RESULTS / "held_verdict.json"                          # rule section 8 item 3

if not (CELLS["aff"]["fused"] == tuple(R3.AFF_CELLS["fused"]) and CELLS["aff"]["cf"] == tuple(R3.AFF_CELLS["cf"])
        and CELLS["r1"]["fused"] == tuple(R3.RC_CELLS["fused"]) and CELLS["r1"]["cf"] == tuple(R3.RC_CELLS["cf"])
        and all(F2.cell_values(c) == (13, *CELL_TEXT[c]) for c in CELL_TEXT)
        and sorted(CELL_TEXT) == sorted({c for w in CELLS.values() for p in w.values() for c in p})
        and set(ALL_SCORERS) == set(CORE_SCORERS) | set(DESCRIPTIVE_EXTRAS)
        and len(ALL_SCORERS) == len(CORE_SCORERS) + len(DESCRIPTIVE_EXTRAS)):
    raise ImportError("r6_score: the frozen cells differ from the rule's table or round 3's constants")


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


# ---------------------------------------------------------------- assembly by parity

def _apply_masks(parity) -> dict:
    """{h: parity == 1 - h}: the episodes scored by the pick of tune half h."""
    parity = np.asarray(parity)
    _require(parity.ndim == 1 and bool(np.isin(parity, HALVES).all()) and bool((parity == 0).any())
             and bool((parity == 1).any()), "parity must be a 1-d array of 0/1 with both halves non-empty")
    return {h: parity == 1 - h for h in HALVES}


def _assemble(per_half, parity, dtype) -> dict:
    """{c: {d: (n, 13) dtype}}: the rows of parity 1 - h from the scores of tune half h's pick."""
    masks = _apply_masks(parity)
    out = {c: {d: np.empty(np.asarray(per_half[0][c][d]).shape, dtype) for d in DIRECTIONS} for c in CONDITIONS}
    for h in HALVES:
        for c in CONDITIONS:
            for d in DIRECTIONS:
                out[c][d][masks[h]] = np.asarray(per_half[h][c][d])[masks[h]]
    return out


def frozen_nested(cos, t_u, t_c, pick, parity) -> dict:
    """B, B0, B1 with frozen picks: nested_scores(cos, t_u, t_c, *pick[h]) on the episodes of parity 1 - h (float32,
    as crossfit_condition_free). t_u = T_N1u, t_c = the comparator's T_6u, both condition-free (asserted)."""
    _require_condition_free(t_u, "T_N1u")
    _require_condition_free(t_c, "T_6u")
    pick = P.as_nested_pick(pick, "nested pick")
    per_half = {h: nested_scores(cos, t_u, t_c, *pick[h]) for h in HALVES}
    return _assemble(per_half, parity, np.float32)


def frozen_fused(cos, term, lam, parity) -> dict:
    """RCA and PM with frozen lambdas: fused_scores(cos, term, lam[h]) on the episodes of parity 1 - h (cosine's
    dtype, as crossfit_lambda)."""
    lam = P.as_lambda_pick(lam, "lambda pick")
    per_half = {h: fused_scores(cos, term, lam[h]) for h in HALVES}
    return _assemble(per_half, parity, np.asarray(cos["a"]["i2t"]).dtype)


def cf_terms(gated) -> dict:
    """{t: G_cf} from the gated terms (round 1's g_cf: the f64 mean of the two conditions, condition-free)."""
    return {t: RC.g_cf(gated[t]) for t in gated}


def _cell_picks(cells) -> dict:
    picks = ({int(h): int(c) for h, c in cells.items()} if isinstance(cells, dict)
             else {h: int(c) for h, c in enumerate(cells)})
    _require(sorted(picks) == list(HALVES) and all(0 <= c < F2.CELLS_PER_KAPPA for c in picks.values()),
             f"cells must be one per tune half, each in 0..{F2.CELLS_PER_KAPPA - 1}: {picks}")
    return picks


def frozen_cells(base, parity, T, gates, fused_cells, cf_cells=None):
    """Round 3's frozen-cell line (r3_fusion.score_frozen) returning the scores: z(B) + lu*z(B) + la*term of the cell
    of tune half h on the episodes of parity 1 - h. Fused term: gate x z(T) (z first); CF term: G_cf of the same
    gate set, asserted condition-free for each CF cell used. base: the frozen B. -> (fused, cf), {c: {d: (n, 13)
    float64}}, finite (asserted). cf_cells=None: the fused scores alone, no G_cf built, cf is None."""
    _require(len(gates) == N_TAU, f"{len(gates)} gate sets, expected {N_TAU}")
    parity = np.asarray(parity)
    _apply_masks(parity)
    fp = _cell_picks(fused_cells)
    zB, zT = _zdict(base), _zdict(T)
    gated = {t: RC.gated_terms(zT, gates[t]) for t in range(len(gates))}
    info = F2.rank_info(base)                                  # asserts B condition-free
    fused = F2.assemble(zB, info, gated, fp, parity)
    C._assert_finite(fused, "frozen fused")
    if cf_cells is None:
        return fused, None
    cp = _cell_picks(cf_cells)
    G = cf_terms(gated)
    for h in HALVES:
        t = F2.decode_cell(cp[h])[1]
        _require_condition_free({c: {d: np.asarray(G[t][c][d]) for d in DIRECTIONS} for c in CONDITIONS},
                                f"the CF term of cell {cp[h]} (tune half {h})")  # guard:cf_cell
    cf = F2.assemble(zB, info, G, cp, parity)
    C._assert_finite(cf, "frozen counterpart")
    return fused, cf


# ---------------------------------------------------------------- one seed

def score_seed(bundle, picks, lambdas, readers, *, include_pm) -> dict:
    """Contracts section 5. picks: {"B", "B0", "B1": {half: [lu, la]}} (load_picks or picks_seed42; "0"/"1" keys
    accepted); lambdas: {scorer: {half: lam}} (frozen_lambdas; rca always read, the nine PM only with include_pm);
    readers: round 1's A0 half-readers (r6_bundle.load_readers).
    -> {scorer: per_anchor dict} for CORE_SCORERS (ALL_SCORERS with include_pm: the nine PM scorers and r1_cf
    added), in that order, then "gates" {"aff",
    "r1": {c: (4, n) float32}}, "reader" {"P", "m", "pick": {c: ...}}, "cl", "pair_index"."""
    msg = f"include_pm must be True or False, not {include_pm!r} (rule section 8 item 3)"
    _require(isinstance(include_pm, bool), msg)  # guard:include_pm_bool
    n = int(bundle.n)
    _require(len(R.PAIRS) * X.ADMITTED.get((bundle.mode, int(bundle.seed)), -1) == n,
             f"{bundle.mode} seed {bundle.seed} with {n} episodes is not an admitted round-6 seed "
             f"({sorted(X.ADMITTED)})")  # guard:admitted
    parity = np.asarray(bundle.parity)
    _require(parity.shape == (n,) and np.array_equal(parity, np.arange(n) % 2),
             "parity must be the episode-index parity (rule section 5 item 3)")  # guard:parity
    if include_pm and bundle.mode == "held":
        _require(Path(VERDICT).is_file(), f"no PM metric on held episodes before {Path(VERDICT).name} exists (rule "
                                          f"section 8 item 3)")  # guard:pm_before_verdict
    B.check_readers(readers)
    nested = {name: P.as_nested_pick(picks[name], name) for name in P.NESTED}
    lam_names = ("rca",) + (PM_SCORERS if include_pm else ())
    lams = {name: P.as_lambda_pick(lambdas[name], name) for name in lam_names}

    cos = bundle.cos
    out = {"cosine": per_anchor(cos)}
    for name in lam_names:
        out[name] = per_anchor(frozen_fused(cos, bundle.pm_terms[name], lams[name], parity))
    frozen = {}
    for name in P.NESTED:
        frozen[name] = frozen_nested(cos, bundle.t_n1u, getattr(bundle, P.TERM_OF[name]), nested[name], parity)
        out[name] = per_anchor(frozen[name])

    rd = RF.reader(bundle, readers)                         # D5: P, T, m, pick from bundle.F and bundle.stack
    taus = R3.assert_taus()                                 # D6
    gates = {"aff": RF.gates_aff(rd["m"], rd["pick"], taus), "r1": RF.gates_r1(rd["m"], taus)}
    for who in GATE_SETS:
        with_cf = who == "aff" or include_pm               # R1's counterpart: descriptive only (rule section 8.3)
        fused, cf = frozen_cells(frozen["B"], parity, rd["T"], gates[who], CELLS[who]["fused"],
                                 CELLS[who]["cf"] if with_cf else None)
        out[f"{who}_fused"] = per_anchor(fused)
        if with_cf:
            out[f"{who}_cf"] = per_anchor(cf)
    moved = [k for k in CONDITION_FREE if k in out and not bool((np.asarray(out[k]["gain"]) == 0).all())]
    _require(not moved, f"condition gain must be 0 on every episode for {moved} (CF and the R1 counterpart "
                        f"condition-free, hazard 10)")  # guard:cf_gain

    out["gates"] = {who: {c: np.stack([gates[who][t][c] for t in range(N_TAU)]) for c in CONDITIONS}
                    for who in GATE_SETS}
    out["reader"] = {k: rd[k] for k in ("P", "m", "pick")}
    out["cl"] = np.asarray(bundle.cl)
    out["pair_index"] = np.asarray(bundle.pair_index)
    order = (ALL_SCORERS if include_pm else CORE_SCORERS) + EXTRA_KEYS
    return {k: out[k] for k in order}
