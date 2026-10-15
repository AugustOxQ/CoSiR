"""Round 6 picks frozen from seed 42 (DECISION_RULE.md of this folder: section 5 item 4, section 6 item 3, section 11;
contracts section 5; ticket 06). No pick is ever made on held episodes: this module makes or reads seed 42's.

  crossfit_seed42(bundle)        B, B0, B1 cross-fitted on the seed-42 selection bundle with crossfit_condition_free
                                 (R3 rule D10, D11, round 4's D4): -> (picks, scores); picks = {"B": {0: [lu, la], 1:
                                 [...]}, "B0", "B1", "mean_r1": {name: 100 * mean per-anchor R@1, exact float}}, keyed
                                 by tune half (int)
  picks_seed42(bundle)           the picks alone (contracts section 5)
  frozen_equals_crossfit(...)    the frozen assembly (r6_score.frozen_nested) with these picks equals the cross-fit
                                 element for element, per comparator
  picks_record(picks, checks)    picks_seed42.json's content: string tune-half keys "0", "1", mean_r1, targets, checks,
                                 passed, module_sha256, time
  load_picks(path)               the B, B0, B1 picks of a passed picks_seed42.json whose mean R@1 equal the targets
  frozen_lambdas()               RCA's and each PM scorer's lambda_picks of baselines_seed42.json ("inf" parsed), keyed
                                 by tune half (int)
  lambda_convention(bundle, l)   crossfit_lambda rerun on the seed-42 bundle per scorer, compared key for key
  assert_lambda_convention(...)  the same, raising on any disagreement

Convention everywhere (hazard 7): the pick of tune half h was tuned on the episodes of parity h and scores the episodes
of parity 1 - h. Stored lambda keys "0" and "1" are tune halves; a stored lambda may be the string "inf".

Guards carry a `# guard:<name>` marker; test_r6_score.py deletes each on a copy and shows that its scenario then goes
through.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r6_common as R  # noqa: E402  (first, before anything that imports src)
import r6_bundle as B  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, DIRECTIONS, per_anchor  # noqa: E402
from src.eval.aspect_nested import nested_cells  # noqa: E402
from src.eval.aspect_quick_checks import crossfit_condition_free  # noqa: E402
from src.eval.aspect_scorers import EDGE_EXTENSION, LAMBDA_GRID, crossfit_lambda  # noqa: E402

NESTED = ("B", "B0", "B1")
TERM_OF = {"B": "t6u_B", "B0": "t6u_B0", "B1": "t6u_B1"}       # the bundle's T_6u of each comparator
# rule section 6 item 3: the mean R@1 (x100) of the cross-fitted B, B'(A0), B'(A1) on seed 42 (each an integer number
# of quarter-hits over 49,152 rankings, so equality is exact)
MEAN_R1_TARGETS = {"B": 18.341064453125, "B0": 18.436686197916664, "B1": 18.804931640625}
LAMBDA_SCORERS = ("rca",) + B.PM_NAMES
LAMBDA_VALUES = tuple(LAMBDA_GRID) + tuple(EDGE_EXTENSION)      # crossfit_lambda's possible picks (inf included)
HALVES = (0, 1)
PICKS_NAME = "picks_seed42.json"
N_SEED42 = len(R.PAIRS) * R.N_PER_PAIR
CELLS = tuple(nested_cells())

if not (tuple(MEAN_R1_TARGETS) == NESTED == tuple(TERM_OF) and len(LAMBDA_SCORERS) == 10
        and float("inf") in LAMBDA_VALUES):
    raise ImportError("r6_picks: names or grids differ from the ones this module was written for")


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


# ---------------------------------------------------------------- pick formats

def _halves(p, what) -> dict:
    """{0: x, 1: y} from a dict keyed by tune half, as ints or as the strings "0", "1" (exactly two halves)."""
    _require(isinstance(p, dict) and len(p) == 2, f"{what}: one pick per tune half is expected, got {p!r}")
    out = {}
    for k, v in p.items():
        h = {"0": 0, "1": 1}.get(k) if type(k) is str else (k if type(k) is int and k in HALVES else None)
        _require(h is not None and h not in out, f"{what}: tune-half keys must be 0 and 1 (or \"0\" and \"1\"), "
                                                  f"got {list(p)}")
        out[h] = v
    return {h: out[h] for h in HALVES}


def as_nested_pick(p, what) -> dict:
    """{0: [lu, la], 1: [lu, la]} (floats) from crossfit_condition_free's picks or a picks_seed42.json entry; each
    pick must be one of the 56 nested cells."""
    out = {}
    for h, v in _halves(p, what).items():
        _require(isinstance(v, (list, tuple)) and len(v) == 2
                 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v),
                 f"{what}: the pick of tune half {h} must be [lambda_u, lambda_a], got {v!r}")
        cell = (float(v[0]), float(v[1]))
        _require(cell in CELLS, f"{what}: the pick {cell} of tune half {h} is not a nested cell")  # guard:nested_cell
        out[h] = [cell[0], cell[1]]
    return out


def as_lambda_pick(p, what) -> dict:
    """{0: lam, 1: lam} (floats, inf allowed) from crossfit_lambda's picks or parsed lambda_picks."""
    out = {}
    for h, v in _halves(p, what).items():
        lam = float("inf") if v == "inf" else v
        _require(isinstance(lam, (int, float)) and not isinstance(lam, bool),
                 f"{what}: the lambda of tune half {h} must be a number or \"inf\", got {v!r}")
        out[h] = float(lam)
    return out


# ---------------------------------------------------------------- B, B0, B1 on seed 42 (rule section 6 item 3)

def check_seed42(bundle):
    """The bundle is the seed-42 development bundle: selection rows, 4,096 episodes per pair, parity by index."""
    n = int(bundle.n)
    _require(bundle.mode == "selection" and int(bundle.seed) == R.DEV_SEED and not bool(bundle.smoke)
             and n == N_SEED42,
             f"the seed-42 selection bundle ({N_SEED42} episodes) is expected, not {bundle.mode} seed {bundle.seed} "
             f"with {n} episodes")  # guard:seed42
    _require(np.array_equal(np.asarray(bundle.parity), np.arange(n) % 2), "parity must be the episode-index parity")


def crossfit_seed42(bundle):
    """-> (picks, scores): crossfit_condition_free(cos, T_N1u, T_6u, parity) per comparator, its picks per tune half
    (second return value) and its mean R@1 x 100; scores = {name: the cross-fitted scores}."""
    check_seed42(bundle)
    picks, mean, scores = {}, {}, {}
    for name in NESTED:
        s, p = crossfit_condition_free(bundle.cos, bundle.t_n1u, getattr(bundle, TERM_OF[name]), bundle.parity)
        picks[name] = as_nested_pick(p, name)
        mean[name] = 100 * float(np.mean(per_anchor(s)["r1"]))
        scores[name] = s
    picks["mean_r1"] = mean
    return picks, scores


def picks_seed42(bundle) -> dict:
    """Contracts section 5: {"B": {0: [lu, la], 1: [...]}, "B0": ..., "B1": ..., "mean_r1": {...}}."""
    return crossfit_seed42(bundle)[0]


def frozen_equals_crossfit(bundle, picks, scores) -> dict:
    """{name: bool}: r6_score.frozen_nested with the picks equals the cross-fitted scores bit for bit (all four
    condition x direction arrays, dtype and shape included)."""
    import r6_score as S                           # r6_score imports this module
    out = {}
    for name in NESTED:
        f = S.frozen_nested(bundle.cos, bundle.t_n1u, getattr(bundle, TERM_OF[name]), picks[name], bundle.parity)
        out[name] = all(np.asarray(f[c][d]).dtype == np.asarray(scores[name][c][d]).dtype
                        and np.asarray(f[c][d]).shape == np.asarray(scores[name][c][d]).shape
                        and np.asarray(f[c][d]).tobytes() == np.asarray(scores[name][c][d]).tobytes()
                        for c in CONDITIONS for d in DIRECTIONS)
    return out


def mean_equal_target(mean_r1) -> dict:
    """{name: bool}: exact float equality with the rule's target."""
    return {name: (type(mean_r1.get(name)) is float and mean_r1[name] == MEAN_R1_TARGETS[name]) for name in NESTED}


def picks_record(picks, checks) -> dict:
    """picks_seed42.json's content (contracts section 5): the picks with string tune-half keys, mean_r1, the targets,
    the checks ({check: {name: bool}}, mean_r1_equal_target first), passed, module_sha256, time."""
    checks = {"mean_r1_equal_target": mean_equal_target(picks["mean_r1"]), **checks}
    passed = all(bool(v) for chk in checks.values() for v in chk.values())
    rec = {name: {str(h): list(picks[name][h]) for h in HALVES} for name in NESTED}
    rec.update({"convention": "the pick of tune half h scores the episodes of parity 1 - h",
                "mean_r1": dict(picks["mean_r1"]), "targets": dict(MEAN_R1_TARGETS), "checks": checks,
                "passed": passed, "rule_sha256": R.RULE_SHA256, "module_sha256": R.r6_module_shas(),
                "time": R.amsterdam_now()})
    return rec


def load_picks(path=None) -> dict:
    """{"B", "B0", "B1": {0: [lu, la], 1: [...]}} from a picks_seed42.json (default RESULTS/picks_seed42.json) that
    passed and whose mean R@1 equal the rule's targets exactly."""
    path = Path(path or R.RESULTS / PICKS_NAME)
    rec = json.loads(path.read_text())
    ok = mean_equal_target(rec.get("mean_r1", {}))
    _require(rec.get("passed") is True and all(ok.values()),
             f"{path}: not a passed record, or its mean R@1 differs from the rule's target ({ok}); "
             f"rule section 6 item 3")  # guard:picks_target
    return {name: as_nested_pick(rec[name], f"{path.name} {name}") for name in NESTED}


# ---------------------------------------------------------------- RCA and PM lambdas (rule section 5 item 4)

def parse_lambda_picks(rec) -> dict:
    """{scorer: {0: lam, 1: lam}} for RCA and the nine PM scorers from a baselines json: lambda_picks keys exactly
    "0" and "1" (tune halves), values numbers of crossfit_lambda's grid or the string "inf"."""
    out = {}
    for name in LAMBDA_SCORERS:
        raw = rec["scorers"][name]["lambda_picks"]
        _require(isinstance(raw, dict) and sorted(raw) == ["0", "1"],
                 f"{name}: lambda_picks keys must be \"0\" and \"1\", got {raw!r}")
        for v in raw.values():
            _require(v == "inf" or (isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v)),
                     f"{name}: a lambda pick must be a finite number or \"inf\", got {v!r}")
        lam = as_lambda_pick(raw, name)
        _require(all(v in LAMBDA_VALUES for v in lam.values()),
                 f"{name}: lambda picks {lam} are not on crossfit_lambda's grid {LAMBDA_VALUES}")  # guard:lambda_grid
        out[name] = lam
    return out


def frozen_lambdas() -> dict:
    """RCA's and each PM scorer's frozen lambdas from baselines_seed42.json (SHA-256 asserted), keyed by tune half."""
    R.assert_input(B.BASELINES42_REL)
    return parse_lambda_picks(json.loads(Path(R.INPUT_PATHS[B.BASELINES42_REL]).read_text()))


def lambda_convention(bundle, lambdas) -> dict:
    """{scorer: bool}: crossfit_lambda(cos, term, parity) rerun on the seed-42 bundle picks, half for half, the given
    lambdas (the stored "0" is the pick tuned on half 0)."""
    check_seed42(bundle)
    out = {}
    for name, lam in lambdas.items():
        rerun = as_lambda_pick(crossfit_lambda(bundle.cos, bundle.pm_terms[name], bundle.parity)[1], name)
        out[name] = rerun == as_lambda_pick(lam, name)
    return out


def assert_lambda_convention(bundle, lambdas) -> dict:
    """lambda_convention, raising when any scorer disagrees; -> {scorer: True}."""
    got = lambda_convention(bundle, lambdas)
    _require(all(got.values()), f"crossfit_lambda rerun on seed 42 disagrees with the stored lambda picks for "
                                f"{[k for k, v in got.items() if not v]}")  # guard:lambda_rerun
    return got
