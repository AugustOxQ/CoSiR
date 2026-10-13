"""The stage 0b runner of round 9 (DECISION_RULE.md of this folder: section 3.2, section 10 row 2, section 12 row C12;
ticket 12): compares the CLIP recipe replay with CLIP's pinned parts.

The replay is made first, by the recipe scripts' own command lines, in chain order, into `results/replay/clip_b32/`
(`run_r9_stage0b_replay.sh`, about 55 minutes, uses the GPU for the content graph and for A3). This script does the
comparison (about 10 minutes, CPU only) and writes `results/stage0b.json` (it never overwrites). The last step of
the queue is `run_r9_stage0b.py --make-cells`: the cells need the replay's reader, tau, A3 and picks, and ticket 11's
script has no `--upstream`, so its own functions (`setup`, `load_taus`, `cells_bundle`, `compute_cells`,
`write_cells`; the body of its `main`) run here with the replay routed in (see `routed`).

What `results/stage0b.json` holds (rule 3.2):
  item 1  every CPU-made part, element by element per key (not file hashes), CLIP's pinned file against the
          replay's: image and caption k-means (`partitions`), the content graph (GPU-made, so bit-equal on this
          machine only), bank AIC, the heads (`n6_posteriors`, `step1_heads_style`), the reader (`rb_heads_*`,
          `rb_bank_A0_half0/1`, `rb_reader_A0.pkl`, and `rb_reader_A0.json` by fields under the role
          `rb_reader_A0_json`: numbers equal, timings, provenance and file hashes excluded), tau, the RCA and PM
          lambda picks (`baselines_seed42`, the ten `lambda_picks`). All exact.
  item 2  A3's case from L1 (`a3_L1/`) and L2 (`--l2-replay`, or `a3_L2/` beside L1): the identity gates of
          `identity_gates.json`, `state_equal` of L1 against the pinned checkpoint (08's function), then
          case 1: the 131 items of stage 0a, scored through the replay's parts, all exact (and the pass counts);
          case 2 (L1 != L2): L1 and L2 each pass item 3 (L2 needs its own chain, `--l2-replay`);
          case 3 (L1 = L2 != pinned) and "pending L2" (L1 != pinned, no L2): stop, not a pass.
  item 3  the tolerance: the picks exact (AFF cells, CF cells, R1 cells, sigma*, the B, B0, B1 picks, tau, the RCA and PM
          lambdas); seed-42 R@1 of B, B0, B1, AFF and CF and the points of P1 to P7 within 0.05 R@1 points of the
          stored values; the count of episodes whose per-anchor R@1 changed (for the scorers that have a stored
          per-anchor array: AFF, AFF's counterpart, R1 and its counterpart, cosine, RCA, the PM scorers).
  item 4  the SHA-256 of every recipe script, read from each part's `.record.json` (`script_sha256`) and checked equal
          to the current script bytes; `passed` and the case.
Also written: `resolved_inputs`, the path and SHA-256 that `r9_common.input_pins` gave each routed input (build 5
note: without the override stage 0a would pass on CLIP's pinned files and test nothing).

How the replay is routed (`routed`): `input_pins("clip_b32")` returns the replay's A3, `partitions.npz`, reader (pickle
and JSON) and `rc_tau.json` with the replay file's own SHA-256; `r9_backbone.CLIP_PARTS` returns the replay's `picked`,
`picks_seed42`; `r3_common.assert_taus` returns the replay's tau and `r9_score.CELLS` the replay's cells. The lambdas are
parsed from the replay's `baselines_seed42.json`. Not routed: the frozen roles, and the pinned `baselines_seed42.json`
as the PM fit rows (a frozen role that shares its file), and the pinned posteriors the refit heads are checked against.
The stage 0a functions (`setup`, `seed_bundle`, `regression_items`, `compare_items`, `compare_counts`) are
`run_r9_stage0a.py`'s own.

    cd <integration checkout> && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/run_r9_stage0b.py [--replay DIR] [--l2-replay DIR] \
        [--no-score] > src/test/20261202_r9_scope/results/stage0b_run.log 2>&1

Exit 0 pass, 3 not a pass (the record is written either way), 2 a missing file.
"""
import argparse
import contextlib
import copy
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first, before anything that imports src)
import r9_backbone as BB  # noqa: E402
import r9_bundle as B  # noqa: E402
import r9_picks as P  # noqa: E402
import r9_score as S  # noqa: E402

import numpy as np  # noqa: E402

BACKBONE = "clip_b32"
EXIT_FAIL, EXIT_MISSING = 3, 2
TOL_R1_POINTS = 0.05                                    # rule 3.2 item 3 (C12 exception)
OUT_NAME = "stage0b.json"
PARTIAL_NAME = "stage0b_partial.json"
DEFAULT_REPLAY = R.RESULTS / "replay" / BACKBONE
QUEUE_SCRIPT = HERE / "run_r9_stage0b_replay.sh"
MISSING = "<missing>"

PINNED = dict(BB.CLIP_PARTS)                            # CLIP's pinned files, before any routing
NPZ_ROLES = ("partitions", "graph", "bank_AIC", "n6_posteriors", "step1_heads_style",
             *(f"rb_heads_{g}" for g in ("affect", "image", "caption", "csd", "rand")),
             "rb_bank_A0_half0", "rb_bank_A0_half1")
ITEM1_ROLES = NPZ_ROLES + ("rb_reader_A0", "rb_reader_A0_json", "rc_tau", "baselines_seed42")
A3_FED_ROLES = ("picks_seed42", "cells_seed42")        # item 3 (and item 2 case 1) read these: they follow A3
A3_ROLES = ("A3_seed42", "picked")                       # item 2
ROUTED_ROLES = ("A3_seed42", "picked", "partitions", "rb_reader_A0", "rb_reader_A0_json", "rc_tau", "picks_seed42")
ROUTED_RELS = {rel: role for role in ROUTED_ROLES for rel in R.INPUT_SHA256 if R.TEST / rel == PINNED[role]}
RECIPE_SCRIPTS = ("r9_recipe_groups.py", "r9_recipe_a3.py", "r9_recipe_reader.py", "r9_recipe_baselines.py",
                  "r9_recipe_cells.py")
A3_FED_SCORERS = ("aff_fused", "aff_cf", "r1_fused", "r1_cf")
READER_JSON_EXCLUDED = ("provenance", "runtime_s", "fit_s", "halves_npz_sha256", "heads_npz_sha256",
                        "banks_npz_sha256", "heads_files_sha256", "banks_files_sha256", "model_spec", "exploratory")
PICKS_FIELDS = ("B", "B0", "B1", "mean_r1", "lambda_picks", "coef_sha256", "episodes_sha256", "fit_rows_sha256",
                "passed")
RC_TAU_FIELDS = ("taus", "percentiles", "n_values", "config")
MAX_DIFFS = 20


def log(msg):
    print(f"[run_r9_stage0b] {msg}", flush=True)


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def plain(x):
    """The JSON form (int keys to strings, tuples to lists, numpy scalars to Python)."""
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, Path):
            return str(v)
        raise TypeError(type(v))
    return json.loads(json.dumps(x, default=conv))


def read_json(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------- the replay's files

def replay_files(replay, a3_run="L1") -> dict:
    """{role: path} of the replay folder: every recipe role under CLIP's file name in the folder; A3 and picked in
    `a3_<run>/` (08's default folder); the cells under `cells_seed42.json` (11's name); the picks file is the failed
    one when only that exists (10 exits 3 with `picks_seed42_failed.json`; rule 3.2 item 3 reads its picks)."""
    replay = Path(replay)
    files = {role: replay / Path(PINNED[role]).name for role in BB.RECIPE_ROLES}
    for role in A3_ROLES:
        files[role] = replay / f"a3_{a3_run}" / Path(PINNED[role]).name
    files["cells_seed42"] = replay / "cells_seed42.json"
    failed = replay / "picks_seed42_failed.json"
    if not files["picks_seed42"].exists() and failed.exists():
        files["picks_seed42"] = failed
    return files


def missing_files(files, roles) -> list:
    return [f"{role}: {files[role]}" for role in roles if not Path(files[role]).is_file()]


# ---------------------------------------------------------------- item 1: element by element

def array_diff(got, want, name) -> list:
    """Differences between two arrays: dtype, shape, any element (NaN equal to NaN in the same place)."""
    got, want = np.asarray(got), np.asarray(want)
    if got.dtype != want.dtype:
        return [f"{name}: dtype {got.dtype} != {want.dtype}"]
    if got.shape != want.shape:
        return [f"{name}: shape {got.shape} != {want.shape}"]
    equal = np.array_equal(got, want, equal_nan=True) if got.dtype.kind == "f" else np.array_equal(got, want)
    if equal:
        return []
    n = int((got != want).sum()) if got.dtype.kind != "f" else int(((got != want) & ~(np.isnan(got) & np.isnan(want))).sum())
    return [f"{name}: {n} of {got.size} elements differ"]


def compare_npz(replay, pinned) -> list:
    with np.load(replay, allow_pickle=False) as a, np.load(pinned, allow_pickle=False) as b:
        out = [f"key only in the replay: {k}" for k in a.files if k not in b.files]
        out += [f"key only in the pinned file: {k}" for k in b.files if k not in a.files]
        for k in b.files:
            if k in a.files:
                out += array_diff(a[k], b[k], k)
    return out


def deep_diff(a, b, path="") -> list:
    """Differences between two Python objects (pickles): types, array elements, dict keys, sequence lengths, floats
    with ==, objects by their attributes (a fitted scikit-learn model is its attributes)."""
    if type(a) is not type(b):
        return [f"{path or '.'}: type {type(a).__name__} != {type(b).__name__}"]
    if isinstance(a, np.ndarray):
        return array_diff(a, b, path or ".")
    if isinstance(a, dict):
        out = [f"{path}/{k}: only in the replay" for k in a if k not in b]
        out += [f"{path}/{k}: only in the pinned file" for k in b if k not in a]
        for k in a:
            if k in b:
                out += deep_diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in deep_diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float):
        return [] if (a == b or (a != a and b != b)) else [f"{path}: {a!r} != {b!r}"]
    if hasattr(a, "__dict__") and not isinstance(a, type):
        return deep_diff(vars(a), vars(b), path)
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def _strip(x, excluded):
    if isinstance(x, dict):
        return {k: _strip(v, excluded) for k, v in x.items() if k not in excluded}
    if isinstance(x, list):
        return [_strip(v, excluded) for v in x]
    return x


def json_view(role, rec) -> dict:
    """The part of a JSON part that must be equal: tau's values; the reader's fields without timings, provenance and
    file hashes (numbers equal); the ten lambda picks and the identity fields of the baselines; the picks and their
    identity fields; the whole cells record."""
    if role == "rc_tau":
        return {k: rec.get(k, MISSING) for k in RC_TAU_FIELDS}
    if role == "rb_reader_A0_json":
        return _strip(rec, READER_JSON_EXCLUDED)
    if role == "baselines_seed42":
        scorers = rec.get("scorers", {})
        return {"fit_rows": rec.get("fit_rows", MISSING), "fit_rows_sha256": rec.get("fit_rows_sha256", MISSING),
                "episodes_sha256": rec.get("episodes_sha256", MISSING),
                "lambda_picks": {name: scorers.get(name, {}).get("lambda_picks", MISSING)
                                 for name in P.LAMBDA_SCORERS}}
    if role == "picks_seed42":
        return {k: rec.get(k, MISSING) for k in PICKS_FIELDS}
    if role == "cells_seed42":
        return rec
    raise KeyError(role)


def compare_role(role, replay, pinned) -> list:
    """Differences between the replay's file and CLIP's pinned file of one role (empty list: equal)."""
    if role in NPZ_ROLES:
        return compare_npz(replay, pinned)
    if role == "rb_reader_A0":
        with open(replay, "rb") as f, open(pinned, "rb") as g:      # our own replay output and CLIP's pinned pickle
            return deep_diff(pickle.load(f), pickle.load(g))
    return deep_diff(plain(json_view(role, read_json(replay))), plain(json_view(role, read_json(pinned))))


def compare_item1(files, pinned=None, roles=ITEM1_ROLES) -> dict:
    pinned = PINNED if pinned is None else pinned
    out = {}
    for role in roles:
        path = Path(files[role])
        if not path.is_file():
            out[role] = {"equal": False, "n_differences": 1, "differences": [f"replay file missing: {path}"]}
            continue
        diff = compare_role(role, path, pinned[role])
        out[role] = {"equal": not diff, "n_differences": len(diff), "differences": diff[:MAX_DIFFS],
                     "replay_path": str(path), "pinned_path": str(pinned[role])}
        log(f"item 1 {role}: {'equal' if not diff else f'{len(diff)} differences'}")
    return {"passed": all(r["equal"] for r in out.values()), "roles": out}


# ---------------------------------------------------------------- item 2: the case

def select_case(l1_equals_pinned, l2_equals_l1=None) -> str:
    """Rule 3.2 item 2. l2_equals_l1: None when there is no L2."""
    if l1_equals_pinned:
        return "case1"
    if l2_equals_l1 is None:
        return "pending_l2"
    return "case3" if l2_equals_l1 else "case2"


def check_identity_gates(gates, run) -> list:
    """Problems with 08's identity_gates.json (rule 3.2 item 2: config, inputs, step-1 loss; every case)."""
    problems = []
    if gates.get("backbone") != BACKBONE:
        problems.append(f"backbone {gates.get('backbone')!r}")
    if gates.get("run") != run:
        problems.append(f"run {gates.get('run')!r}, expected {run!r}")
    if (gates.get("config") or {}).get("passed") is not True:
        problems.append("config gate not passed")
    inputs = gates.get("inputs") or {}
    if inputs.get("passed") is not True or inputs.get("pinned") is not True:
        problems.append("inputs gate not passed against the pinned inputs")
    if (gates.get("step1_loss") or {}).get("passed") is not True:
        problems.append("step-1 loss gate not passed")
    return problems


def decide(case, *, item1, gates, regression, item3, item4) -> bool:
    """`passed`: case 1 needs the 131 items exact; case 2 needs item 3 for L1 and L2; case 3 and a pending L2 stop."""
    if case not in ("case1", "case2"):
        return False
    return bool(item1 and gates and item4 and (regression if case == "case1" else item3))


def a3_case(replay, l2_replay=None) -> dict:
    """A3's case. L1: `replay/a3_L1`. L2: `l2_replay/a3_L2`, else `replay/a3_L2` when it exists (08 run alone)."""
    import r9_recipe_a3 as A3
    l1 = Path(replay) / "a3_L1"
    ckpt = l1 / "A3_seed42.pt"
    gates = read_json(l1 / "identity_gates.json")
    problems = check_identity_gates(gates, "L1")
    diff_pinned = A3.state_diff(ckpt, PINNED["A3_seed42"])
    l2_dir = Path(l2_replay) / "a3_L2" if l2_replay else Path(replay) / "a3_L2"
    l2_ckpt = l2_dir / "A3_seed42.pt"
    equal_l2 = None
    diff_l2 = []
    if l2_ckpt.is_file():
        diff_l2 = A3.state_diff(ckpt, l2_ckpt)
        equal_l2 = not diff_l2
        problems += [f"L2: {p}" for p in check_identity_gates(read_json(l2_dir / "identity_gates.json"), "L2")]
    case = select_case(not diff_pinned, equal_l2)
    return {"case": case, "l1_state_equal_pinned": not diff_pinned, "l1_vs_pinned_differences": diff_pinned[:MAX_DIFFS],
            "l2_present": equal_l2 is not None, "l1_state_equal_l2": equal_l2,
            "l1_vs_l2_differences": diff_l2[:MAX_DIFFS], "identity_gate_problems": problems,
            "identity_gates_passed": not problems, "l1_checkpoint": str(ckpt),
            "l1_checkpoint_sha256": R.sha256_file(ckpt), "l2_checkpoint": str(l2_ckpt) if equal_l2 is not None else None,
            "gates": gates}


# ---------------------------------------------------------------- item 3: picks exact, 0.05 R@1 points

def within_tolerance(got, want, tol=TOL_R1_POINTS) -> bool:
    return abs(float(got) - float(want)) <= tol


def item3_check(got, want) -> dict:
    """got, want: {"picks": {name: value}, "r1": {name: R@1 points}, "points": {P1..P7: points}}. Picks must be equal
    (exact), the R@1 numbers and the P points within 0.05 R@1 points. A missing number fails."""
    failures = []
    picks_exact = {}
    for k, w in want["picks"].items():
        g = got["picks"].get(k, MISSING)
        picks_exact[k] = bool(plain(g) == plain(w)) if g != MISSING else False
        if not picks_exact[k]:
            failures.append(f"pick {k}: {g!r} != {w!r}")
    out = {"picks_exact": picks_exact}
    for group in ("r1", "points"):
        out[group] = {}
        for k, w in want[group].items():
            g = got[group].get(k, MISSING)
            if g == MISSING:
                out[group][k] = {"got": None, "want": w, "diff": None, "within": False}
                failures.append(f"{group} {k}: missing")
                continue
            ok = within_tolerance(g, w)
            out[group][k] = {"got": g, "want": w, "diff": g - w, "within": ok}
            if not ok:
                failures.append(f"{group} {k}: {g!r} is {g - w:+.4f} from {w!r} (limit {TOL_R1_POINTS})")
    return {"passed": not failures, "failures": failures, **out}


def count_changed(got, want) -> dict:
    """Episodes whose per-anchor R@1 changed: {"per_scorer": {name: n}, "any_a3_fed": n over AFF, AFF's counterpart, R1
    and its counterpart (the union of their changed episodes), "n_episodes"}."""
    per, union, n = {}, None, None
    for name, w in want.items():
        g = np.asarray(got[name], np.float64)
        w = np.asarray(w, np.float64)
        _require(g.shape == w.shape and g.ndim == 1, f"{name}: per-anchor R@1 of shape {g.shape}, stored {w.shape}")
        n = len(w)
        changed = g != w
        per[name] = int(changed.sum())
        if name in A3_FED_SCORERS:
            union = changed if union is None else (union | changed)
    return {"per_scorer": per, "any_a3_fed": int(union.sum()) if union is not None else 0, "n_episodes": n}


def pick_numbers(paths) -> dict:
    """The exact-equality numbers of item 3, read from files of the roles cells_seed42, picks_seed42, rc_tau and
    baselines_seed42 (the replay's, or CLIP's pinned ones)."""
    cells, picks = read_json(paths["cells_seed42"]), read_json(paths["picks_seed42"])
    base, tau = read_json(paths["baselines_seed42"]), read_json(paths["rc_tau"])["taus"]
    out = {"aff_cells": cells["aff"], "cf_cells": cells["cf"], "r1_fused_cells": cells["r1_fused"],
           "r1_counterpart_cells": cells["r1_counterpart"], "sigma_star": cells["sigma_star"],
           "B": picks["B"], "B0": picks["B0"], "B1": picks["B1"], "tau": tau}
    for name in P.LAMBDA_SCORERS:
        out[f"lambda_{name}"] = base["scorers"][name]["lambda_picks"]
    return plain(out)


def pinned_numbers(ref_counts) -> dict:
    """The stored values of item 3: CLIP's pinned picks, tau and lambdas; stage 0a's R@1 targets; the P1 to P7 points
    of seed42_pass_counts.json."""
    import run_r9_stage0a as RS
    r1 = {name: RS.EXTRA_TARGETS[f"{name}_mean_r1"] for name in P.NESTED}
    r1.update(AFF=RS.TARGETS["aff_fused_r1"], CF=RS.TARGETS["aff_cf_r1"])
    return {"picks": pick_numbers(PINNED), "r1": r1,
            "points": {k: v["point"] for k, v in ref_counts["checks"].items()}}


def chain_numbers(files, scored, counts) -> dict:
    """The same numbers from one replay chain: its files and the scoring through them."""
    r1 = {name: 100 * float(np.mean(scored[name]["r1"])) for name in P.NESTED}
    r1.update(AFF=100 * float(np.mean(scored["aff_fused"]["r1"])), CF=100 * float(np.mean(scored["aff_cf"]["r1"])))
    return {"picks": pick_numbers(files), "r1": r1,
            "points": {k: v["point"] for k, v in counts["checks"].items()}}


def stored_per_anchor_r1() -> dict:
    """The stored per-anchor R@1 arrays (CLIP, seed 42, 12,288 episodes): AFF and R1 fused and counterpart from round 3's
    seed42_arrays.npz, cosine, RCA and the nine PM scorers from AB's per_anchor_seed42.npz. B, B0 and B1 have only their
    means stored."""
    import run_r9_stage0a as RS
    out = {}
    R.assert_input(RS.R3_ARRAYS_REL)
    with np.load(R.INPUT_PATHS[RS.R3_ARRAYS_REL]) as z:
        for who in RS.WHO:
            for part in RS.PARTS:
                out[f"{who}_{part}"] = z[f"{who}_{part}__r1"]
    R.assert_input(RS.PER_ANCHOR42_REL)
    with np.load(R.INPUT_PATHS[RS.PER_ANCHOR42_REL]) as z:
        for s in RS.AB_SCORERS:
            out[s] = z[f"{s}__r1"]
    return out


# ---------------------------------------------------------------- item 4: the scripts

def record_files(replay) -> list:
    replay = Path(replay)
    return sorted(replay.glob("*.record.json")) + sorted(replay.glob("a3_*/*.record.json"))


def check_scripts(replay, scripts_dir=HERE) -> dict:
    """Every `.record.json` of the replay names the script that made the file and its SHA-256; each must equal the
    current bytes of that script in `scripts_dir`, and each of the five recipe scripts must appear."""
    scripts_dir = Path(scripts_dir)
    seen, problems = {}, []
    for rec_path in record_files(replay):
        rec = read_json(rec_path)
        name = Path(rec.get("script", "")).name
        got = rec.get("script_sha256")
        if not name or not got:
            problems.append(f"{rec_path.name}: no script or script_sha256")
            continue
        current = scripts_dir / name
        if not current.is_file():
            problems.append(f"{rec_path.name}: script {name} is not in {scripts_dir}")
            continue
        now = R.sha256_file(current)
        entry = seen.setdefault(name, {"sha256": now, "records": 0, "records_equal": 0})
        entry["records"] += 1
        if got == now:
            entry["records_equal"] += 1
        else:
            problems.append(f"{rec_path.name}: {name} SHA-256 {got} differs from the current script's {now}")
    for name in RECIPE_SCRIPTS:
        if name not in seen:
            problems.append(f"no record names {name}")
    return {"passed": not problems, "problems": problems, "scripts": seen}


# ---------------------------------------------------------------- routing the replay into the stage 0a code

def cells_for_score(rec) -> dict:
    """r9_score.CELLS' shape from a cells record."""
    return {"aff": {"fused": tuple(rec["aff"]), "cf": tuple(rec["cf"])},
            "r1": {"fused": tuple(rec["r1_fused"]), "cf": tuple(rec["r1_counterpart"])}}


@contextlib.contextmanager
def routed(files, cells=None):
    """Within the block, CLIP's reads of recipe parts go to the replay (module docstring). `cells`: r9_score.CELLS' shape
    (cells_for_score), or None to leave the cells alone (the cells step itself). Everything is restored on exit."""
    for role in ROUTED_ROLES:
        _require(role in files, f"routed: no replay file for {role}")
        _require(Path(files[role]).resolve() != Path(PINNED[role]).resolve(),
                 f"routed: the replay's {role} is CLIP's pinned file {PINNED[role]}")      # guard:routed_not_pinned
    saved_pins, saved_taus = R.input_pins, R.R3.assert_taus
    saved_parts = {role: BB.CLIP_PARTS[role] for role in ROUTED_ROLES}
    saved_cells = copy.deepcopy(S.CELLS)

    def pins(backbone="clip_b32"):
        out = dict(saved_pins(backbone))
        if backbone == "clip_b32":
            for rel, role in ROUTED_RELS.items():
                path = Path(files[role])
                out[rel] = (path, R._sha_cached(path))                  # FileNotFoundError, never CLIP's pinned file
        return out

    def taus():
        return tuple(read_json(files["rc_tau"])["taus"])

    R.input_pins, R.R3.assert_taus = pins, taus
    for role in ROUTED_ROLES:
        BB.CLIP_PARTS[role] = Path(files[role])
    if cells is not None:
        for who in cells:
            S.CELLS[who].update(cells[who])
    try:
        yield
    finally:
        R.input_pins, R.R3.assert_taus = saved_pins, saved_taus
        BB.CLIP_PARTS.update(saved_parts)
        S.CELLS.clear()
        S.CELLS.update(saved_cells)


def resolved_inputs(files) -> dict:
    """What `input_pins` resolves each routed input to: the path read and the SHA-256 it is pinned to (the replay
    file's own), beside CLIP's pinned path and SHA-256."""
    out = {}
    for rel, role in ROUTED_RELS.items():
        path = Path(files[role])
        out[rel] = {"role": role, "path": str(path), "sha256": R._sha_cached(path), "pinned_path": str(R.TEST / rel),
                    "pinned_sha256": R.INPUT_SHA256[rel]}
    return out


# ---------------------------------------------------------------- the cells step (ticket 11's main, routed)

def make_cells(replay, a3_run="L1"):
    """The replay's last step: `cells_seed42.json` through ticket 11's own functions, on the replay's parts."""
    import r9_context as X
    import r9_recipe_cells as RC
    files = replay_files(replay, a3_run)
    absent = missing_files(files, ROUTED_ROLES)
    _require(not absent, f"the replay is incomplete: {absent}", SystemExit)
    RC.preflight(replay)
    t0 = time.time()
    with routed(files):
        taus = RC.load_taus(BACKBONE)
        env = RC.setup(BACKBONE)
        ctx = X.RowContext("selection", R.DEV_SEED, env.data, env.split, env.labels, env.heads, env.value_sets,
                           R.N_PER_PAIR)
        RC.check_episode_sha(ctx.episode_sha)
        bundle = RC.cells_bundle(ctx, BACKBONE)
        log(f"cells bundle [{time.time() - t0:.0f}s]")
        rec, details = RC.compute_cells(bundle, env.readers, taus, BACKBONE)
        _require(details["b_mean_r1"] == P.MEAN_R1_TARGETS["B"],
                 f"B's seed-42 mean R@1 {details['b_mean_r1']!r} is not the rule's {P.MEAN_R1_TARGETS['B']!r}")
        from r9_recipe_groups import read_shas
        path = RC.write_cells(replay, rec, backbone=BACKBONE, input_sha256=env.inputs,
                              upstream_sha256={role: R.sha256_file(BB.parts(BACKBONE)[role])
                                               for role in RC.UPSTREAM_ROLES},
                              extra={"feature_sha256": read_shas(BACKBONE), "taus": taus, "details": details,
                                     "episodes_sha256": dict(ctx.episode_sha), "time": R.amsterdam_now(),
                                     "runtime_s": round(time.time() - t0, 1)})
    log(f"written {path} (SHA-256 {R.sha256_file(path)})")
    return 0


def cells_upstream_problems(files) -> list:
    """The cells record names the SHA-256 of the replay files it was made from; they must be the files now."""
    import r9_recipe_cells as RC
    rec_path = Path(str(files["cells_seed42"]) + ".record.json")
    if not rec_path.is_file():
        return [f"{rec_path} is missing"]
    up = read_json(rec_path).get("upstream_sha256", {})
    problems = []
    for role in RC.UPSTREAM_ROLES:
        now = R.sha256_file(files[role])
        if up.get(role) != now:
            problems.append(f"cells record: {role} was {up.get(role)}, the replay file is now {now}")
    return problems


# ---------------------------------------------------------------- scoring one chain (stage 0a through the replay)

def score_chain(files, refs) -> dict:
    """Stage 0a's setup, bundle, scoring and 131 items on the replay's parts; item 3's numbers; the changed-episode
    counts. Heavy (about 8 minutes)."""
    import run_r9_stage0a as RS
    import r9_stats as ST
    t0 = time.time()
    cells_rec = read_json(files["cells_seed42"])
    with routed(files, cells=cells_for_score(cells_rec)):
        picks_rec = read_json(files["picks_seed42"])
        picks = (P.load_picks(files["picks_seed42"], BACKBONE) if picks_rec.get("passed") is True
                 else {name: P.as_nested_pick(picks_rec[name], name) for name in P.NESTED})
        lambdas = P.parse_lambda_picks(read_json(files["baselines_seed42"]))
        env = RS.setup()
        bundle = RS.seed_bundle(env, "selection", R.DEV_SEED, R.N_PER_PAIR)
        P.check_seed42(bundle)
        scored = S.score_seed(bundle, picks, lambdas, env.readers, include_pm=True)
        log(f"scored [{time.time() - t0:.0f}s]")
        orders = {"refit": refs[RS.REFIT_NAME], "picks": picks_rec}
        items, _ = RS.regression_items(env, bundle, scored, orders)
        extra = {"episodes_sha256": {R.DEV_SEED: dict(bundle.episodes_sha256)}, "runner_sha256": RS.runner_sha256(),
                 "module_sha256": R.r9_module_shas()}
        counts = ST.pass_record([scored], "regression", [R.DEV_SEED], extra)
        counts.update({"input_sha256": dict(env.inputs), "coef_sha256": env.coef_sha256})
    diff = {"items": RS.compare_items(items, refs[RS.REGRESSION_NAME]["items"]),
            "pass_counts": RS.compare_counts(counts, refs[RS.COUNTS_NAME])}
    stored = stored_per_anchor_r1()
    changed = count_changed({k: scored[k]["r1"] for k in stored}, stored)
    item3 = item3_check(chain_numbers(files, scored, counts), pinned_numbers(refs[RS.COUNTS_NAME]))
    log(f"chain done [{time.time() - t0:.0f}s]: {len(diff['items'])} item and {len(diff['pass_counts'])} count "
        f"differences, item 3 {'passed' if item3['passed'] else 'FAILED'}")
    return {"regression": {"passed": not diff["items"] and not diff["pass_counts"], "n_items": len(items),
                           "n_failed": sum(not it["equal"] for it in items.values()), "differences": diff,
                           "items_equal": {k: it["equal"] for k, it in items.items()},
                           "items_not_equal": {k: it for k, it in items.items() if not it["equal"]}},
            "item3": item3, "changed_episodes": changed, "runtime_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------- the run

def check_out(path):
    if Path(path).exists():
        raise FileExistsError(f"refusing to overwrite {path}; a finished stage 0b record is never replaced")


def write_new(path, rec):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x") as f:
        f.write(json.dumps(plain(rec), indent=1) + "\n")
    return path


def run(replay=None, l2_replay=None, score=True, out=None) -> int:
    t0 = time.time()
    import run_r9_stage0a as RS
    replay = Path(replay or DEFAULT_REPLAY)
    out = Path(out or R.RESULTS / (OUT_NAME if score else PARTIAL_NAME))
    check_out(out)
    files = replay_files(replay)
    need = list(ITEM1_ROLES) + list(A3_ROLES) + list(A3_FED_ROLES)
    absent = missing_files(files, need)
    if absent:
        print("the replay is incomplete:\n  " + "\n  ".join(absent))
        return EXIT_MISSING
    refs = RS.load_references()
    log("references loaded; item 1")
    item1 = compare_item1(files)
    stale = cells_upstream_problems(files)
    item1["cells_upstream_current"] = {"equal": not stale, "differences": stale}
    item1["passed"] = item1["passed"] and not stale
    log("item 2: A3's case")
    a3 = a3_case(replay, l2_replay)
    case = a3["case"]
    scripts = check_scripts(replay)
    if l2_replay:
        scripts_l2 = check_scripts(l2_replay)
        scripts["problems"] += [f"L2 chain: {p}" for p in scripts_l2["problems"]]
        scripts["passed"] = scripts["passed"] and scripts_l2["passed"]
        scripts["scripts_l2_chain"] = scripts_l2["scripts"]
    chains, errors = {}, {}
    todo = {"L1": (replay, files)}
    if case == "case2" and l2_replay:
        todo["L2"] = (Path(l2_replay), replay_files(l2_replay, "L2"))
    if score and case in ("case1", "case2"):
        for name, (folder, f) in todo.items():
            try:
                chains[name] = score_chain(f, refs)
            except Exception:                                   # a crash is a failed chain, recorded
                errors[name] = traceback.format_exc()
                log(f"chain {name} crashed:\n{errors[name]}")
    regression_ok = bool(chains.get("L1", {}).get("regression", {}).get("passed"))
    item3_ok = bool(chains) and all(c["item3"]["passed"] for c in chains.values())
    if case == "case2":
        item3_ok = item3_ok and set(chains) == {"L1", "L2"}
    passed = score and decide(case, item1=item1["passed"], gates=a3["identity_gates_passed"],
                              regression=regression_ok, item3=item3_ok, item4=scripts["passed"])
    rec = {"passed": bool(passed), "case": case, "scored": score, "time": R.amsterdam_now(),
           "runtime_s": round(time.time() - t0, 1), "replay": str(replay), "l2_replay": str(l2_replay) if l2_replay else None,
           "rule_sha256": R.RULE_SHA256, "runner_sha256": R.sha256_file(Path(__file__).resolve()),
           "queue_script_sha256": R.sha256_file(QUEUE_SCRIPT) if QUEUE_SCRIPT.is_file() else None,
           "module_sha256": R.r9_module_shas(),
           "resolved_inputs": resolved_inputs(files),
           "item1": item1,
           "item2": {**{k: v for k, v in a3.items() if k != "gates"}, "identity_gates": a3["gates"],
                     "regression_L1": chains.get("L1", {}).get("regression"),
                     "regression_exact": regression_ok},
           "item3": {name: {"result": c["item3"], "changed_episodes": c["changed_episodes"]}
                     for name, c in chains.items()},
           "item3_passed": item3_ok,
           "item4": {"recipe_script_sha256": {n: scripts["scripts"].get(n, {}).get("sha256") for n in RECIPE_SCRIPTS},
                     **scripts},
           "errors": errors}
    if not score:
        rec["note"] = "partial run (--no-score): items 1, 2 (gates and case) and 4 only; never a pass"
    write_new(out, rec)
    for role, r in item1["roles"].items():
        if not r["equal"]:
            print(f"item 1 {role}: {r['differences'][:3]}")
    print(f"stage 0b: {'PASSED' if passed else 'NOT A PASS'} (case {case}; item 1 {item1['passed']}, gates "
          f"{a3['identity_gates_passed']}, 131 items exact {regression_ok}, item 3 {item3_ok}, item 4 "
          f"{scripts['passed']}); record {out}")
    return 0 if passed else EXIT_FAIL


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay", type=Path, default=None, help=f"the replay folder (default {DEFAULT_REPLAY})")
    ap.add_argument("--l2-replay", type=Path, default=None,
                    help="a second chain made from A3 run L2 (its a3_L2/); needed only if L1 is not bit-equal")
    ap.add_argument("--no-score", action="store_true",
                    help="items 1, 2 (gates, case) and 4 only; writes stage0b_partial.json; never a pass")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--a3-run", choices=("L1", "L2"), default="L1", help="with --make-cells: whose a3_<run>/ the chain uses")
    ap.add_argument("--make-cells", action="store_true",
                    help="the replay's last step: cells_seed42.json through ticket 11's functions, then stop")
    a = ap.parse_args(argv)
    if a.make_cells:
        return make_cells(a.replay or DEFAULT_REPLAY, a.a3_run)
    return run(a.replay, a.l2_replay, score=not a.no_score, out=a.out)


if __name__ == "__main__":
    sys.exit(main())
