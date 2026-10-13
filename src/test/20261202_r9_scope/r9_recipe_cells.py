"""Recipe V (ticket 11; rule section 3.2 item 3, section 3.4; R3 rule D4 to D9): the 224-cell picks, sigma*, the
switched-off check and the seed-42 gate share, for one backbone.

    cd <checkout> && CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
    /root/miniconda3/envs/CoSiR/bin/python src/test/20261202_r9_scope/r9_recipe_cells.py --backbone B [--out DIR]

Writes `cells_seed42.json` and `cells_seed42.json.record.json` to DIR (default `results/recipe/<B>/`; it refuses to
overwrite). The file is `{"aff", "cf", "r1_fused", "r1_counterpart", "sigma_star", "switched_off", "gate_share",
"gate_open", "gate_total"}` (r9_picks.read_cells is its schema), index = tune half 0, 1.

Where the numbers come from (CLIP, seed 42):
  AFF  39, 119 (fused) and 149, 10 (counterpart): picked on CLIP seed 42 by `20261120_r1_levers_brainstorm/bs_05_aff.py`
       (`L.Family(data, MDs, tied=True).crossfit()` with the AFF gate sets; results `bs_05_aff.json` A0.AFF
       fused_cells / cf_cells), then recorded as constants in `r3_common.AFF_CELLS` (R3 rule section 5 item 3) and
       re-derived by round 3's `run_r3_seed42.py` item 3 (`r3_fusion.run_family` on AFF's gates).
  R1   116, 119 (fused) and 58, 123 (counterpart), sigma* 0, 0: round 1's R-c (`20261117_reader_fix_csd/run_rc.py`
       on the parent Rb_expected_A0, `cand_Rc_Rb_expected_A0.json`), recorded as `r3_common.RC_CELLS` (R3 rule section 5
       item 2) and re-derived by `run_r3_seed42.py` item 2 (`run_family` on R1's gates).
  sigma* 0, 0: `F2.control_choice` (the smallest of the 30 control sums with the largest integer rho of (1+sigma)
       z(B)); it depends on B and the parity only, so AFF's and R1's are equal (asserted).
The rule (R3 rule D8 item 5, D9; round 2's integer criteria, `r2_fusion.select_fused` / `select_cf`, which
`bs_05_aff.py` calls too): per tune half h, over the 224 cells (cell number (t*7 + u)*8 + a, t = tau index from
`parts(backbone)["rc_tau"]`), AFF's fused cell is the one with the largest min(rho(S) - rho_ctrl, gamma(S)) (integers
over the tune half's episodes), the counterpart cell the one with the largest rho of the matched counterpart; ties go
to the lowest cell number. `pick_cells` is that step on integer tables (the tests run it on synthetic tables);
`family_picks` is `r3_fusion.run_family` itself (no copy), whose picks the tests compare with `bs_lib.Family`'s.
The brainstorm scored in float64, this module (as round 3, D8) in float32 on z-scored terms: the CLIP run asserts the
cells equal the brainstorm's recorded ones, in (tau index, lambda_u, lambda_a) form. For every backbone the record's
`details.float64_tied_family` says whether `bs_lib.Family(..., tied=True)` (float64) gives the same integer tables and
picks, and names the cells that differ; it decides nothing.
Checks that hold for every backbone (compute_cells): the thresholds are those of the bundle's own R1 margins
(`rc_core.thresholds`, the function that made rc_tau.json); the cross-fitted B and its mean R@1 equal the backbone's
picks_seed42.json (the B that the held read rebuilds). The CLI also asserts the seed-42 episode hashes equal the frozen ones.

Switched off (D4): both halves' AFF cells have lambda_a = 0 (`F2.cell_values(cell)[3] == 0`); recorded here, the stage
2 driver (ticket 13) acts on it. The gate share is r9_pins.gate_share at AFF's chosen cells (the gates of AFF's tau on
the other parity, conditions a and b, both tune halves).

The bundle: the CLI builds the seed-42 selection bundle on the refit heads, as `run_r9_stage0a.py` does, but only the
fields the cells need (`cells_bundle`: T_N1u, B's T_6u, the A0 posteriors, the grouping stack and the 18 features;
`r9_bundle.build_bundle_r6` steps 2, 3, 4, 5, 7 and 8, then validate's stack/F/Delta checks) and no PM fit or PM term
(about 150 s of the full bundle). B is cross-fitted (D10). `compute_cells(bundle, readers, taus)` takes any bundle with these fields (ticket 12 passes its own).
For CLIP the CLI also asserts B's mean R@1 (18.341064453125) and that the record equals CLIP's `cells_seed42_clip.json`
and `bs_05_aff.json`'s cells.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import r9_common as R  # noqa: E402  (first, before anything that imports src)
import r9_bundle as RB  # noqa: E402
import r9_context as X  # noqa: E402
import r9_episodes as E  # noqa: E402
import r9_heads as H  # noqa: E402
import r9_picks as P  # noqa: E402
import r9_pins as PINS  # noqa: E402
from r9_backbone import BACKBONES, parts  # noqa: E402

import numpy as np  # noqa: E402

from src.eval.aspect_metrics import CONDITIONS, per_anchor  # noqa: E402
from src.eval.aspect_nested import _zdict  # noqa: E402
from src.eval.aspect_quick_checks import _require_condition_free, centered_term, crossfit_condition_free  # noqa: E402
from src.eval.aspect_quick_checks import uniform_probe_scores  # noqa: E402

C, RC, F2, RF, R3, rbe = R.C, R.RC, R.F2, R.RF, R.R3, R.rbe

N_CELLS = F2.CELLS_PER_KAPPA                      # 224
HALVES = (0, 1)
N_TAU = F2.N_TAU
OUT_NAME = "cells_seed42.json"
RECORD_KEYS = P.CELLS_KEYS
BS05_REL = "20261120_r1_levers_brainstorm/results/bs_05_aff.json"
UPSTREAM_ROLES = ("rc_tau", "rb_reader_A0", "rb_reader_A0_json", "A3_seed42", "picked")


def _require(cond, msg, exc=AssertionError):
    if not cond:
        raise exc(msg)


def log(msg):
    print(f"[r9_recipe_cells] {msg}", flush=True)


# ---------------------------------------------------------------- the picks

def pick_cells(fri, fgi, cri, ctrl, parity) -> dict:
    """R3 rule D8 item 5 and D9 on integer tables. fri, fgi: (224, n) per-episode 4*R@1 and 4*gain of the fused cells;
    cri: (224, n) 4*R@1 of the matched counterpart; ctrl: {half: (sigma*, rho_ctrl)}; parity: (n,) of 0/1, both halves
    present. Per tune half h (episodes with parity h): fused = the cell with the largest min(rho - rho_ctrl, gamma),
    counterpart = the largest rho, integers, ties to the lowest cell number. -> {"fpick", "cpick": {half: cell}}."""
    parity = np.asarray(parity)
    _require(parity.ndim == 1 and bool(np.isin(parity, HALVES).all()) and all(bool((parity == h).any()) for h in HALVES),
             "pick_cells: parity must be a 1-d array of 0/1 with both halves present")
    for name, t in (("fri", fri), ("fgi", fgi), ("cri", cri)):
        _require(np.asarray(t).shape == (N_CELLS, len(parity)) and np.issubdtype(np.asarray(t).dtype, np.integer),
                 f"pick_cells: {name} must be an integer table of shape ({N_CELLS}, {len(parity)})")
    _require(sorted(ctrl) == list(HALVES), "pick_cells: ctrl needs one entry per tune half")
    return {"fpick": F2.select_fused(fri, fgi, ctrl, parity), "cpick": F2.select_cf(cri, parity)}


def family_picks(base, T, gates, parity) -> dict:
    """The 224 cells of one gate set on B's scores ``base`` ({c: {d: (n, 13)}}, the cross-fitted B) and the reader's T:
    r3_fusion.run_family itself (z-scores before the gate, the gated terms, G_cf, the integer statistics, sigma* by tune
    half, the picks of pick_cells' rule; D8, D9). -> its dict: "fpick", "cpick", "sigma", "ctrl", "details", ..."""
    _require(len(gates) == N_TAU, f"{len(gates)} gate sets, expected {N_TAU}")
    return RF.run_family(SimpleNamespace(B=base, parity=np.asarray(parity)), T, gates)


def sigma_star(aff, r1) -> list:
    """sigma* by tune half. It depends on B and the parity only (D8 item 5), so AFF's and R1's families agree."""
    _require(aff["sigma"] == r1["sigma"], f"sigma* of AFF {aff['sigma']} and R1 {r1['sigma']} differ")
    return [float(aff["sigma"][h]) for h in HALVES]


def switched_off(cells) -> bool:
    """D4: true when AFF's cell has lambda_a = 0 in both tune halves (r9_pins.switched_off)."""
    return PINS.switched_off(cells)


def cells_record(aff, cf, r1_fused, r1_counterpart, sigma, gates_aff, parity) -> dict:
    """The cells_seed42.json content. aff, cf, r1_fused, r1_counterpart: (cell of half 0, cell of half 1); sigma: [.., ..];
    gates_aff: {c: (4, n)} AFF's gates by tau index; parity: (n,)."""
    cells = {k: [int(v) for v in x] for k, x in (("aff", aff), ("cf", cf), ("r1_fused", r1_fused),
                                                   ("r1_counterpart", r1_counterpart))}
    _require(all(len(x) == 2 and all(0 <= v < N_CELLS for v in x) for x in cells.values()), f"cells: {cells}")
    rec = {**cells, "sigma_star": [float(x) for x in sigma], "switched_off": switched_off(cells["aff"]),
           **PINS.gate_share(gates_aff, parity, cells["aff"])}
    _require(tuple(rec) == RECORD_KEYS, f"the record keys {tuple(rec)} are not {RECORD_KEYS}")
    return rec


def check_taus(margins, taus, n):
    """M1: the thresholds are those of the bundle's own R1 margins (rc_core.thresholds made rc_tau.json from them)."""
    got = RC.thresholds(margins)
    _require(got == (list(taus), 2 * n),
             f"tau: the thresholds {list(taus)} are not those of the bundle's R1 margins {got[0]} (over {got[1]} "
             f"margins, expected {2 * n})")


def check_b_picks(bpick, b_mean_r1, backbone):
    """M2: the cross-fitted B equals the B of the backbone's picks_seed42.json (what the held read rebuilds), and its mean
    R@1 equals the file's exactly."""
    path = Path(parts(backbone)["picks_seed42"])
    want = P.load_picks(path, backbone)["B"]
    got = P.as_nested_pick(bpick, "B")
    _require(got == want, f"B: the cross-fitted picks {got} are not those of {path}: {want}")
    mean = json.loads(path.read_text())["mean_r1"]["B"]
    _require(b_mean_r1 == mean, f"B: the mean R@1 {b_mean_r1!r} is not {path}'s {mean!r}; the mean R@1 must be equal")


def differing_cells(x, y) -> list:
    """The cell numbers (rows) where two (224, n) integer tables differ."""
    x, y = np.asarray(x), np.asarray(y)
    _require(x.shape == y.shape, f"tables of shape {x.shape} and {y.shape}")
    return [int(i) for i in np.flatnonzero((x != y).any(axis=1))]


def float64_tied_check(base, T, gates, parity, run) -> dict:
    """L4, decides nothing: does bs_lib.Family(..., tied=True) (the brainstorm's float64 scoring, which picked CLIP's
    cells) give the same integer tables and picks as this module's float32 ones? run: the family's run_family result.
    -> {"tables_equal", "fused_cells_differ", "cf_cells_differ", "fpick_equal", "cpick_equal"}."""
    sys.path.insert(0, str(R.TEST / "20261120_r1_levers_brainstorm"))
    import bs_lib as L
    parity = np.asarray(parity)
    zB32, zT32 = _zdict(base), _zdict(T)
    gated = {t: RC.gated_terms(zT32, gates[t]) for t in range(len(gates))}
    fri, fgi, cri = F2.cell_statistics(zB32, F2.rank_info(base), gated, {t: RC.g_cf(gated[t]) for t in gated}, 1)
    ctrl = F2.control_choice(zB32, parity)
    again = pick_cells(fri, fgi, cri, ctrl, parity)                    # this module's float32 picks, from its own tables
    _require(again["fpick"] == run["fpick"] and again["cpick"] == run["cpick"],
             "the float32 picks of the recomputed tables differ from run_family's")
    zB = {c: {d: L.zrows(base[c][d]) for d in L.DIRECTIONS} for c in L.CONDITIONS}
    data = SimpleNamespace(zB=zB, E=len(parity), parity=parity, ctrl=F2.control_choice(zB, parity))
    fam = L.Family(data, [L.MD(L.zterm(T), g) for g in gates], tied=True).stats()
    fp, cp = fam.crossfit()
    fused_differ = sorted(set(differing_cells(fri, fam.fri)) | set(differing_cells(fgi, fam.fgi)))
    cf_differ = differing_cells(cri, fam.cri)
    return {"tables_equal": not fused_differ and not cf_differ, "fused_cells_differ": fused_differ,
            "cf_cells_differ": cf_differ, "fpick_equal": {h: int(fp[h]) for h in HALVES} == run["fpick"],
            "cpick_equal": {h: int(cp[h]) for h in HALVES} == run["cpick"]}


def compute_cells(bundle, readers, taus, backbone):
    """The cells record of one seed-42 bundle. bundle: cos, t_n1u, t6u_B, parity, stack, F (r9_bundle's fields); readers:
    round 1's A0 half-readers; taus: the backbone's four thresholds (asserted to be the bundle's own R1 margins');
    backbone: whose picks_seed42.json B must equal. -> (record, details); details holds each family's integer criteria at
    the chosen cells, the B scores' mean R@1, the thresholds and the float64 tied-family comparison (L4)."""
    taus = [float(t) for t in taus]
    _require(len(taus) == N_TAU, f"{len(taus)} thresholds, expected {N_TAU}")
    parity = np.asarray(bundle.parity)
    base, bpick = crossfit_condition_free(bundle.cos, bundle.t_n1u, bundle.t6u_B, parity)    # D10
    b_mean_r1 = 100 * float(np.mean(per_anchor(base)["r1"]))
    check_b_picks(bpick, b_mean_r1, backbone)                                                  # M2
    rd = RF.reader(bundle, readers)                                                            # D5
    check_taus(rd["m"], taus, len(parity))                                                     # M1
    g_aff, g_r1 = RF.gates_aff(rd["m"], rd["pick"], taus), RF.gates_r1(rd["m"], taus)          # D6
    t0 = time.time()
    aff = family_picks(base, rd["T"], g_aff, parity)
    log(f"AFF family [{time.time() - t0:.0f}s]: fused {aff['fpick']} counterpart {aff['cpick']}")
    r1 = family_picks(base, rd["T"], g_r1, parity)
    log(f"R1 family [{time.time() - t0:.0f}s]: fused {r1['fpick']} counterpart {r1['cpick']}")
    gates = {c: np.stack([g_aff[t][c] for t in range(N_TAU)]) for c in CONDITIONS}
    rec = cells_record([aff["fpick"][h] for h in HALVES], [aff["cpick"][h] for h in HALVES],
                       [r1["fpick"][h] for h in HALVES], [r1["cpick"][h] for h in HALVES],
                       sigma_star(aff, r1), gates, parity)
    f64 = {"aff": float64_tied_check(base, rd["T"], g_aff, parity, aff),
           "r1": float64_tied_check(base, rd["T"], g_r1, parity, r1)}
    f64["equal"] = all(f64[k]["tables_equal"] and f64[k]["fpick_equal"] and f64[k]["cpick_equal"] for k in ("aff", "r1"))
    log(f"float64 tied family (bs_lib) gives the same tables and picks: {f64['equal']} [{time.time() - t0:.0f}s]")
    details = {"aff": aff["details"], "r1": r1["details"], "ctrl": aff["ctrl"], "b_mean_r1": b_mean_r1, "taus": taus,
               "float64_tied_family": {"equal": f64["equal"], "aff": f64["aff"], "r1": f64["r1"]}}
    return rec, details


def check_episode_sha(sha):
    """L6: the bundle's seed-42 episode hashes equal the frozen ones (rule section 11; the hashes of AB's baselines file)."""
    want = E.identity_targets()[R.DEV_SEED]["episodes_sha256"]
    for pair in R.PAIR_NAMES:
        _require(sha.get(pair) == want[pair], f"episodes: the hash of pair {pair} is {sha.get(pair)!r}, the frozen one "
                                              f"is {want[pair]!r}")


# ---------------------------------------------------------------- the CLIP checks

def check_clip(rec):
    """CLIP: the record equals CLIP's cells part (cells_seed42_clip.json: R3's constants, sigma*, the gate share)."""
    want = P.read_cells(parts("clip_b32")["cells_seed42"])
    for key in RECORD_KEYS:
        _require(rec[key] == want[key], f"CLIP: {key} is {rec[key]!r}, CLIP's pinned value is {want[key]!r}")
    _require(rec == want, "CLIP: the cells record differs from the pinned cells part")
    return want


def check_brainstorm(rec):
    """CLIP: AFF's and R1's cells equal bs_05_aff.json's (the brainstorm that picked them), as (tau index, lambda_u,
    lambda_a) for the fused cells and (tau index, lambda_u, lambda_a) for the counterpart (its cf_cells)."""
    R3.assert_inputs([BS05_REL])
    bs = json.loads((R.TEST / BS05_REL).read_text())["A0"]
    for who, key_f, key_c in (("AFF", "aff", "cf"), ("R1", "r1_fused", "r1_counterpart")):
        for h in HALVES:
            f, c = bs[who]["fused_cells"][h], bs[who]["cf_cells"][h]
            _require(f[2] == f[3], f"bs_05 {who} fused cell {f} is not the tied family")
            _require(tuple(F2.cell_values(rec[key_f][h])[1:]) == (f[0], f[1], f[2]),
                     f"{key_f}[{h}] = {rec[key_f][h]} is not bs_05_aff.json's {who} fused cell {f}")
            _require(tuple(F2.cell_values(rec[key_c][h])[1:]) == tuple(c),
                     f"{key_c}[{h}] = {rec[key_c][h]} is not bs_05_aff.json's {who} counterpart cell {c}")


# ---------------------------------------------------------------- the bundle and the files

def load_taus(backbone) -> list:
    """The four thresholds of parts(backbone)["rc_tau"]; CLIP's must equal the rule's (R3 rule D6)."""
    taus = json.loads(Path(parts(backbone)["rc_tau"]).read_text())["taus"]
    _require(len(taus) == N_TAU and all(type(t) is float for t in taus), f"rc_tau.json taus: {taus!r}")
    if backbone == "clip_b32":
        _require(tuple(taus) == tuple(R3.TAUS), "rc_tau.json taus differ from the rule's (R3 D6)")
    return taus


def setup(backbone) -> SimpleNamespace:
    """run_r9_stage0a.setup without the PM fit: inputs asserted, rows, heads refit and checked, readers."""
    H.require_threads()
    t0 = time.time()
    inputs = R.assert_inputs(backbone)
    rows = R.load_rows(backbone)
    log(f"data and split [{time.time() - t0:.0f}s]")
    heads = H.fit_heads_r9(rows.data, rows.split.groups, rows.split.scorer_train, backbone=backbone)
    check = H.check_selection(heads, rows.data, rows.split.selection, backbone)
    log(f"heads [{time.time() - t0:.0f}s]")
    return SimpleNamespace(inputs=inputs, **vars(rows), heads=heads, head_check=check,
                           readers=RB.load_readers(backbone))


def cells_bundle(ctx, backbone) -> SimpleNamespace:
    """The fields of r9_bundle.build_bundle_r6 that the cells use, by the same calls (its steps 2, 3, 4, 5, 7, 8; its
    docstring has the rule references); no PM term."""
    cl = RB.check_context(ctx)
    ep, n = ctx.pooled, int(ctx.n)
    inp = RB.model_inputs(ctx, backbone)                                                   # step 2: T_N1u
    t_n1u = centered_term(inp, ep, uniform=True)
    del inp
    post_e2 = {part: ctx.post[RB.E2_FROM[part]] for part in RB.E2_PARTS}                  # step 3: B's T_6u
    t6u_B = uniform_probe_scores(post_e2, ep, RB.E2_PARTS)
    post = {h: ctx.post[h] for h in RB.A0}                                                 # steps 4, 5
    for h in RB.A0:
        for side in ("img", "txt"):
            _require(bool(np.isfinite(post[h][side][ctx.rows]).all()), f"{h}/{side}: non-finite posteriors on rows")
    stack = C.grouping_stack(post, ep, RB.A0)                                              # step 7
    F, checks = rbe.seed42_features(SimpleNamespace(ctx=ctx, post=post), RB.A0)            # step 8
    b = SimpleNamespace(n=n, cl=cl, parity=np.asarray(ctx.parity), pair_index=np.asarray(ctx.pair_index), cos=ctx.cos,
                        t_n1u=t_n1u, t6u_B=t6u_B, stack=stack, F=F, mode=ctx.mode, seed=int(ctx.seed),
                        smoke=bool(ctx.smoke), checks=dict(checks))
    for k in ("cos", "t_n1u", "t6u_B"):
        RB.check_scores(getattr(b, k), n, k, np.float32)
        _require_condition_free(getattr(b, k), k)
    RB.check_stack_f(b)                                                                    # validate's stack, F, Delta
    P.check_seed42(b)                                                                      # selection rows, seed 42
    return b


def output_paths(out):
    return [Path(out) / OUT_NAME, Path(out) / (OUT_NAME + ".record.json")]


def preflight(out):
    for path in output_paths(out):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}; remove {' and '.join(map(str, output_paths(out)))} "
                                  f"to rerun")


def write_cells(out, rec, *, backbone, input_sha256, upstream_sha256, extra) -> Path:
    """cells_seed42.json plus its record, both created exclusively."""
    path, rec_path = output_paths(out)
    preflight(out)
    _require(tuple(rec) == RECORD_KEYS, f"the record keys {tuple(rec)} are not {RECORD_KEYS}")
    Path(out).mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(json.dumps(rec, indent=1) + "\n")
    _require(P.read_cells(path) == rec, "the cells file written differs from the record")
    meta = {"backbone": backbone, "input_sha256": dict(input_sha256), "upstream_sha256": dict(upstream_sha256),
            "script": str(Path(__file__)), "script_sha256": R.sha256_file(__file__),
            "module_sha256": R.r9_module_shas(), "rule_sha256": R.RULE_SHA256,
            "keys_made": list(rec), "keys_copied": [], **extra}
    with rec_path.open("x") as f:
        f.write(json.dumps(meta, indent=1) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backbone", required=True, choices=BACKBONES)
    ap.add_argument("--out", type=Path, default=None, help="default results/recipe/<backbone>/")
    a = ap.parse_args(argv)
    out = Path(a.out or R.RESULTS / "recipe" / a.backbone)
    preflight(out)                                                                         # before any work
    t0 = time.time()
    taus = load_taus(a.backbone)
    env = setup(a.backbone)
    ctx = X.RowContext("selection", R.DEV_SEED, env.data, env.split, env.labels, env.heads, env.value_sets,
                       R.N_PER_PAIR)
    check_episode_sha(ctx.episode_sha)                                                     # L6
    bundle = cells_bundle(ctx, a.backbone)
    log(f"bundle [{time.time() - t0:.0f}s]")
    rec, details = compute_cells(bundle, env.readers, taus, a.backbone)
    if a.backbone == "clip_b32":
        _require(details["b_mean_r1"] == P.MEAN_R1_TARGETS["B"],
                 f"B's seed-42 mean R@1 {details['b_mean_r1']!r} is not the rule's {P.MEAN_R1_TARGETS['B']!r}")
        check_clip(rec)
        check_brainstorm(rec)
        log("CLIP: the record equals the pinned cells part and bs_05_aff.json's cells")
    from r9_recipe_groups import read_shas
    path = write_cells(out, rec, backbone=a.backbone, input_sha256=env.inputs,
                       upstream_sha256={role: R.sha256_file(parts(a.backbone)[role]) for role in UPSTREAM_ROLES},
                       extra={"feature_sha256": read_shas(a.backbone), "taus": taus, "details": details,
                              "episodes_sha256": dict(ctx.episode_sha), "time": R.amsterdam_now(),
                              "runtime_s": round(time.time() - t0, 1)})
    print(f"written {path} (SHA-256 {R.sha256_file(path)})")
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main())
