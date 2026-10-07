"""Round 5: the candidates G-T and G-TF and their families (DECISION_RULE.md of this folder: D6, D7, D11; §4 items 5 to
7; §5 items 4 and 5; §6.3 and §6.4; §10 list A item 5).

  candidate(name, bundle, ext, tau_prime=None)   D6: the reader outputs (P, T, m, pick), the thresholds and the gates
  tau_prime(m)                                   D6: tau' = rc_core.thresholds of G-TF's seed-42 margins (24,576)
  check_taus(taus)                               a tau' read from a record: four finite non-decreasing floats
  expected_d6(name, bundle, ext, tau_prime=None) D6 recomputed independently (D7's gate check)
  run_candidate(name, bundle, ext, cand, tau_prime=None)
                                                 D7: round 3's run_family from the candidate's own term and gates,
                                                 after D7's check that cand equals expected_d6; the gates are stored

D6's table, one reader (round 3's r3_fusion.reader with the bundle's frozen A0 half-readers) and one gate function
(r3_fusion.gates_aff, AFF's gate):
  G-T   reader on F (the CLIP placement's features; P, m, pi are AFF's), term on stack_G, AFF's tau_0..tau_3
  G-TF  reader on F_G, term on stack_G, tau'_0..tau'_3
tau' is computed only on seed 42 (non-smoke) from G-TF's own margins, and on every other seed it is passed in from
results/dev_seed42.json and never recomputed (a missing tau' is refused there, a passed one refused on seed 42).

Every function that takes an extension calls r5_bundle.require_ext (r5_guard.require on its placement) first: a GE
extension is refused before the guard is released (D11). Nothing here prints.
"""
from types import SimpleNamespace

import numpy as np

import r5_common as R5
import r5_bundle as R5B

R3, RF3, C, rc_core = R5.R3, R5.RF3, R5.C, R5.rc_core
rbe, rf = R3.rbe, R3.rf
from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS  # noqa: E402

TAUS = tuple(R3.TAUS)                    # AFF's tau_0..tau_3 (D1)
N_EPISODES_42 = 12_288
N_MARGINS_42 = 2 * N_EPISODES_42         # D6: tau' needs the 24,576 seed-42 margins of G-TF
N_TAU = len(TAUS)
N_GROUPINGS = len(R3.A0)
KEYS = ("P", "T", "m", "pick", "gates", "taus")
# D6: (source of the reader's features, source of the term's stack): "bundle" = the bundle's F / stack (the CLIP
# placement), "ext" = the extension's F_Q / stack_Q
READER_INPUTS = {"G-T": ("bundle", "ext"), "G-TF": ("ext", "ext")}


def is_seed42(bundle) -> bool:
    """The development seed: episode seed 42, not a smoke run."""
    return int(bundle.seed) == 42 and not bool(bundle.smoke)


# ---------------------------------------------------------------- tau'

def tau_prime(m) -> tuple:
    """D6: tau'_0..tau'_3 = round 1's rc_core.thresholds of G-TF's margins (numpy.percentile, linear, at 0, 25, 50 and
    75, condition a's margins first, float64); its count must be seed 42's 24,576. -> tuple of four floats."""
    if not isinstance(m, dict) or set(m) != set(CONDITIONS):
        raise ValueError("tau' takes the margins of both conditions {'a': ..., 'b': ...}")
    for c in CONDITIONS:
        x = np.asarray(m[c])
        if x.ndim != 1 or x.dtype != np.float64 or not np.isfinite(x).all():
            raise ValueError(f"margins of condition {c} must be a finite float64 vector")
    taus, n = rc_core.thresholds({"a": m["a"], "b": m["b"]})
    if n != N_MARGINS_42:
        raise ValueError(f"tau' is computed from seed 42's 24,576 margins, got {n}")
    return tuple(float(t) for t in taus)


def check_taus(taus) -> tuple:
    """A tau' read from results/dev_seed42.json: four finite, non-decreasing floats. -> tuple of floats (unchanged)."""
    if not isinstance(taus, (tuple, list)) or len(taus) != N_TAU:
        raise ValueError(f"tau' must be {N_TAU} thresholds")
    out = []
    for t in taus:
        if isinstance(t, bool) or not isinstance(t, (float, int, np.floating, np.integer)):
            raise ValueError("tau' must hold numbers")
        out.append(float(t))
    if not all(np.isfinite(out)) or any(out[i] > out[i + 1] for i in range(N_TAU - 1)):
        raise ValueError("tau' must be finite and non-decreasing")
    return tuple(out)


def _thresholds(name, bundle, m, given):
    """D6's thresholds: G-T AFF's tau; G-TF tau' computed on seed 42 only, given from the record elsewhere."""
    if name == "G-T":
        if given is not None:
            raise ValueError("G-T uses AFF's tau_0..tau_3 (D1, D6); no tau' is passed for G-T")
        return TAUS
    if is_seed42(bundle):
        if given is not None:
            raise ValueError("on seed 42 tau' is computed from G-TF's margins (D6), never passed in")
        return tau_prime(m)
    if given is None:
        raise ValueError("on every seed but 42, tau' is read from results/dev_seed42.json and passed in; it is never "
                         "recomputed (D6)")
    return check_taus(given)


# ---------------------------------------------------------------- D6

def _check_name(name):
    if name not in R5.CANDIDATES:
        raise ValueError(f"unknown candidate {name!r}; the candidates are {R5.CANDIDATES}")


def _readers(bundle):
    readers = getattr(bundle, "readers", None)
    if readers is None:
        raise ValueError("the bundle carries no readers (round 1's A0 half-readers); they are never loaded here")
    return readers


def check_gates(gates, n) -> bool:
    """D6: four gate sets, each condition a float32 (n,) array of 0s and 1s. Raises AssertionError."""
    if len(gates) != N_TAU:
        raise AssertionError(f"{len(gates)} gate sets, expected {N_TAU}")
    for t, g in enumerate(gates):
        if set(g) != set(CONDITIONS):
            raise AssertionError(f"gate set {t}: keys must be the conditions")
        for c in CONDITIONS:
            x = np.asarray(g[c])
            if x.dtype != np.float32 or x.shape != (n,) or not np.all((x == 0) | (x == 1)):
                raise AssertionError(f"gate tau_{t}/{c}: must be a float32 ({n},) array of 0s and 1s (D6)")
    return True


def candidate(name, bundle, ext, tau_prime=None) -> dict:
    """D6 for candidate ``name`` ('G-T' or 'G-TF') on a bundle and its placement extension (r5_bundle.extend or
    load_ext, paired with the bundle). ``tau_prime``: G-TF's tau' from results/dev_seed42.json on every seed but 42
    (required there; refused on seed 42, where it is computed, and for G-T).
    -> {"P": {c: (n, 3) float64}, "T": {c: {d: (n, 13) float32}}, "m": {c: (n,) float64}, "pick": {c: (n,) int64},
        "gates": [ {c: (n,) float32 0/1} ] x 4, "taus": (4 floats)}."""
    R5B.require_ext(ext, f"r5_fusion.candidate {name}")
    _check_name(name)
    R5B.check_pair(bundle, ext)
    readers = _readers(bundle)
    src = {"bundle": bundle, "ext": ext}
    f_src, s_src = READER_INPUTS[name]
    r = RF3.reader(SimpleNamespace(F=src[f_src].F, stack=src[s_src].stack), readers=readers)
    taus = _thresholds(name, bundle, r["m"], tau_prime)
    gates = RF3.gates_aff(r["m"], r["pick"], taus)
    check_gates(gates, int(bundle.n))
    return {"P": r["P"], "T": r["T"], "m": r["m"], "pick": r["pick"], "gates": gates, "taus": taus}


def expected_d6(name, bundle, ext, tau_prime=None) -> dict:
    """D6 recomputed for D7's gate check, without candidate()'s wiring: G-T = AFF's reader outputs and gates (the
    readers on the bundle's F with the bundle's stack, AFF's tau), with T = common.expected_term(stack_Q, P); G-TF = the
    half-readers' averaged probabilities on F_Q, round 1's picks and margins, tau' (computed on seed 42, given
    elsewhere), T = common.expected_term(stack_Q, P'). Same keys as candidate()."""
    R5B.require_ext(ext, f"r5_fusion.expected_d6 {name}")
    _check_name(name)
    readers = _readers(bundle)
    if name == "G-T":
        aff = RF3.reader(SimpleNamespace(F=bundle.F, stack=bundle.stack), readers=readers)      # AFF exactly (D1)
        P, m, pick = aff["P"], aff["m"], aff["pick"]
    else:
        if list(readers["feature_names"]) != rf.feature_names(R3.A0):
            raise AssertionError("the readers were trained on another feature layout")
        P = {c: rf.average_probs(rbe.half_reader_probs(readers, ext.F[c], N_GROUPINGS)) for c in CONDITIONS}
        pm = {c: rf.picks_and_margins(P[c]) for c in CONDITIONS}
        m, pick = {c: pm[c][1] for c in CONDITIONS}, {c: pm[c][0] for c in CONDITIONS}
    taus = _thresholds(name, bundle, m, tau_prime)
    return {"P": P, "T": C.expected_term(ext.stack, P), "m": m, "pick": pick,
            "gates": RF3.gates_aff(m, pick, taus), "taus": taus}


def _same(x, y) -> bool:
    x, y = np.asarray(x), np.asarray(y)
    return bool(x.shape == y.shape and x.dtype == y.dtype and np.array_equal(x, y))


def gate_check(cand, want) -> dict:
    """D7's check: cand equals the independent recomputation exactly (P, T, m, pick per condition and direction,
    taus as floats, the gates at every tau index and condition in value, shape and dtype). -> {part: True,
    "tau_t": {c: True}}; raises AssertionError naming every part that differs (no value)."""
    out = {"P": all(_same(cand["P"][c], want["P"][c]) for c in CONDITIONS),
           "T": all(_same(cand["T"][c][d], want["T"][c][d]) for c in CONDITIONS for d in DIRECTIONS),
           "m": all(_same(cand["m"][c], want["m"][c]) for c in CONDITIONS),
           "pick": all(_same(cand["pick"][c], want["pick"][c]) for c in CONDITIONS),
           "taus": tuple(cand["taus"]) == tuple(want["taus"]) and len(cand["gates"]) == N_TAU}
    for t in range(N_TAU):
        out[f"tau_{t}"] = {c: t < len(cand["gates"]) and _same(cand["gates"][t][c], want["gates"][t][c])
                           for c in CONDITIONS}
    bad = [k for k, v in out.items() if (v is not True if not isinstance(v, dict) else not all(v.values()))]
    if bad:
        raise AssertionError(f"rule D7: the candidate's {bad} differ from D6 recomputed (the gates a family runs from "
                             f"must equal D6's at every tau index and condition)")
    return out


# ---------------------------------------------------------------- D7

def run_candidate(name, bundle, ext, cand, tau_prime=None) -> dict:
    """D7: round 3's run_family(bundle, cand["T"], cand["gates"]) (224 cells, sigma*, the fused reader's min-margin
    cross-fit and the matched counterpart from the candidate's OWN term and gates, its max-R@1 cross-fit), after D7's
    gate check: cand must equal expected_d6 exactly. ``tau_prime``: G-TF's tau' from the record on every seed but 42
    (required there, refused on seed 42 and for G-T). -> run_family's dict ("fpick", "cpick", "sigma", "fused", "cf",
    "details", "ctrl") plus "candidate", "gates" (the very list the family ran from), "taus" and "gate_check"."""
    R5B.require_ext(ext, f"r5_fusion.run_candidate {name}")
    _check_name(name)
    if not isinstance(cand, dict) or set(cand) != set(KEYS):
        raise ValueError(f"cand must be candidate()'s dict with keys {KEYS}")
    R5B.check_pair(bundle, ext)
    checked = gate_check(cand, expected_d6(name, bundle, ext, tau_prime))   # value, shape and dtype of every gate
    fam = RF3.run_family(bundle, cand["T"], cand["gates"])
    fam["candidate"] = name
    fam["gates"] = cand["gates"]
    fam["taus"] = tuple(cand["taus"])
    fam["gate_check"] = checked
    return fam
